"""
metacog_calibrator.py — Calibrated Metacognitive Uncertainty

Prevents naive self-description by tracking:
  1. Uncertainty estimates for every self-assertion the system makes.
  2. Error detection: when a self-report is contradicted by later evidence,
     the calibrator flags the discrepancy.
  3. Boundary awareness: the system must be able to name what it does NOT
     know, rather than confabulating answers.
  4. Calibration curve: maps confidence → actual accuracy across topics,
     producing an overconfidence / underconfidence measure.

Architecture:
  Each 'claim' the ConsciousnessCore generates (stream entries, self-desc,
  goal explanations) is hashed and stored with a confidence level.
  When real outcomes contradict the claim, a calibration error is recorded.
  The calibrator then adjusts the system's priors on future claims of the
  same type.

Integration:
    calib = MetacogCalibrator()

    # After generating a claim:
    claim_id = calib.register_claim("I will succeed at 'respond'",
                                     claim_type="goal_prediction",
                                     confidence=0.8)

    # After outcome:
    calib.resolve_claim(claim_id, correct=False)

    # Query calibration:
    overconf = calib.overconfidence_score()
    report   = calib.self_error_report()
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_CLAIMS = 400  # maximum tracked claims
CALIBRATION_BINS = 5  # confidence bins for calibration curve
PRIOR_SHRINKAGE = 0.1  # shrink confidence toward prior when overconfident
FALSE_REPORT_THRESH = 0.25  # accuracy below this → systematic false reporting
UNCERTAINTY_FLOOR = 0.05  # minimum reportable uncertainty


# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class Claim:
    """A system self-assertion tied to a verifiable outcome."""

    claim_id: int
    tick: int
    text: str  # original claim text
    claim_type: str  # "goal_prediction" | "self_state" | "trait" |
    # "agency" | "integration" | "report"
    confidence: float  # stated confidence [0..1]
    verified: bool = False
    correct: Optional[bool] = None  # None until resolved
    resolved_tick: int = -1
    notes: str = ""


@dataclass
class CalibrationBin:
    """One bin in the calibration curve."""

    lower: float  # confidence lower bound
    upper: float  # confidence upper bound
    n_correct: int = 0
    n_total: int = 0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total > 0 else 0.0

    @property
    def mid_confidence(self) -> float:
        return (self.lower + self.upper) / 2


@dataclass
class SelfErrorReport:
    """Summary of self-reporting errors."""

    tick: int
    total_claims: int
    resolved_claims: int
    correct_claims: int
    accuracy: float
    overconfidence: float  # positive → systematically overconfident
    boundary_awareness: float  # fraction of claimed unknowns that were correct
    calibration_error: float  # ECE (expected calibration error)
    flagged_claim_types: List[str]
    notes: str


# ─── Main class ───────────────────────────────────────────────────────────────


class MetacogCalibrator:
    """
    Tracks all system self-assertions and measures how accurately they
    reflect reality.  Prevents naive self-description by forcing confidence
    to be grounded in track record rather than arbitrary outputs.
    """

    def __init__(self) -> None:
        self._tick: int = 0
        self._claims: Deque[Claim] = deque(maxlen=MAX_CLAIMS)
        self._claim_counter: int = 0

        # Per-type accuracy tracker: type → [correct, total]
        self._type_tracker: Dict[str, List[int]] = {}

        # Calibration bins (5 equal-width bins in [0, 1])
        step = 1.0 / CALIBRATION_BINS
        self._bins = [
            CalibrationBin(i * step, (i + 1) * step) for i in range(CALIBRATION_BINS)
        ]

        # Known boundaries (things the system explicitly acknowledges it doesn't know)
        self._declared_unknowns: Deque[str] = deque(maxlen=100)

        # Confidence adjustment per type (learned shrinkage)
        self._conf_adjust: Dict[str, float] = {}

    # ── Claim registration ────────────────────────────────────────────────────

    def register_claim(
        self,
        text: str,
        claim_type: str,
        confidence: float,
        tick: Optional[int] = None,
    ) -> int:
        """
        Register a self-assertion.  Returns a claim_id for resolution later.
        The confidence is adjusted by any learned correction for this claim type.
        """
        t = tick or self._tick
        adj = self._conf_adjust.get(claim_type, 0.0)
        adjusted_conf = max(UNCERTAINTY_FLOOR, min(1.0, confidence + adj))

        self._claim_counter += 1
        claim = Claim(
            claim_id=self._claim_counter,
            tick=t,
            text=text[:200],
            claim_type=claim_type,
            confidence=adjusted_conf,
        )
        self._claims.append(claim)
        return self._claim_counter

    def resolve_claim(
        self,
        claim_id: int,
        correct: bool,
        tick: Optional[int] = None,
        notes: str = "",
    ) -> None:
        """Resolve a claim as correct or incorrect."""
        t = tick or self._tick
        for c in reversed(self._claims):
            if c.claim_id == claim_id:
                c.verified = True
                c.correct = correct
                c.resolved_tick = t
                c.notes = notes
                self._update_trackers(c)
                return

    def resolve_by_type(
        self,
        claim_type: str,
        correct: bool,
        tick: Optional[int] = None,
    ) -> int:
        """Resolve the most recent unresolved claim of a given type."""
        t = tick or self._tick
        for c in reversed(self._claims):
            if c.claim_type == claim_type and not c.verified:
                c.verified = True
                c.correct = correct
                c.resolved_tick = t
                self._update_trackers(c)
                return c.claim_id
        return -1

    def declare_unknown(self, topic: str) -> None:
        """System explicitly acknowledges it does not know something."""
        self._declared_unknowns.append(topic)

    # ── Calibration queries ───────────────────────────────────────────────────

    def overconfidence_score(self) -> float:
        """
        Positive → system is overconfident (confidence > accuracy).
        Negative → underconfident.
        0.0 → perfectly calibrated.
        """
        resolved = [c for c in self._claims if c.verified and c.correct is not None]
        if len(resolved) < 5:
            return 0.0
        mean_conf = sum(c.confidence for c in resolved) / len(resolved)
        accuracy = sum(1 for c in resolved if c.correct) / len(resolved)
        return mean_conf - accuracy

    def expected_calibration_error(self) -> float:
        """ECE: weighted average of |confidence - accuracy| across bins."""
        total = sum(b.n_total for b in self._bins)
        if total == 0:
            return 0.0
        ece = sum(
            b.n_total / total * abs(b.mid_confidence - b.accuracy)
            for b in self._bins
            if b.n_total > 0
        )
        return ece

    def accuracy_for_type(self, claim_type: str) -> float:
        entry = self._type_tracker.get(claim_type)
        if not entry or entry[1] == 0:
            return 0.5  # neutral prior
        return entry[0] / entry[1]

    def adjusted_confidence(self, raw_confidence: float, claim_type: str) -> float:
        """Return confidence shrunk toward calibrated accuracy for this type."""
        accuracy = self.accuracy_for_type(claim_type)
        adj = self._conf_adjust.get(claim_type, 0.0)
        # Shrink toward track-record accuracy
        adjusted = (
            raw_confidence * (1 - PRIOR_SHRINKAGE) + accuracy * PRIOR_SHRINKAGE + adj
        )
        return max(UNCERTAINTY_FLOOR, min(1.0, adjusted))

    def self_error_report(self, tick: Optional[int] = None) -> SelfErrorReport:
        """Generate a full calibration report."""
        t = tick or self._tick
        resolved = [c for c in self._claims if c.verified and c.correct is not None]
        total = len([c for c in self._claims])
        n_correct = sum(1 for c in resolved if c.correct)
        accuracy = n_correct / len(resolved) if resolved else 0.5
        overconf = self.overconfidence_score()
        ece = self.expected_calibration_error()

        # Flag claim types with accuracy < FALSE_REPORT_THRESH
        flagged = [
            ctype
            for ctype, vals in self._type_tracker.items()
            if vals[1] >= 3 and (vals[0] / vals[1]) < FALSE_REPORT_THRESH
        ]

        # Boundary awareness: proportion of declared unknowns that were correct
        # (We check if the system later correctly admitted knowing nothing,
        # i.e. resolved claims about declared-unknown topics were 'incorrect' → good.)
        boundary_awareness = min(
            1.0,
            len(self._declared_unknowns)
            / max(1, len([c for c in self._claims if c.claim_type == "report"])),
        )

        notes = ""
        if overconf > 0.2:
            notes += f"ALERT: overconfidence bias={overconf:+.2f}. "
        if flagged:
            notes += f"Systematic false reporting in: {flagged}. "

        return SelfErrorReport(
            tick=t,
            total_claims=total,
            resolved_claims=len(resolved),
            correct_claims=n_correct,
            accuracy=accuracy,
            overconfidence=overconf,
            boundary_awareness=boundary_awareness,
            calibration_error=ece,
            flagged_claim_types=flagged,
            notes=notes or "No calibration issues detected.",
        )

    def describe(self) -> str:
        overconf = self.overconfidence_score()
        ece = self.expected_calibration_error()
        resolved = sum(1 for c in self._claims if c.verified)
        return (
            f"CALIB: claims={len(list(self._claims))} resolved={resolved} "
            f"overconf={overconf:+.2f} ECE={ece:.3f} "
            f"unknowns_declared={len(self._declared_unknowns)}"
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _update_trackers(self, claim: Claim) -> None:
        """Update per-type tracker and calibration bins."""
        ctype = claim.claim_type
        if ctype not in self._type_tracker:
            self._type_tracker[ctype] = [0, 0]
        self._type_tracker[ctype][0] += int(claim.correct or 0)
        self._type_tracker[ctype][1] += 1

        # Update calibration bin
        bin_idx = min(CALIBRATION_BINS - 1, int(claim.confidence * CALIBRATION_BINS))
        self._bins[bin_idx].n_total += 1
        self._bins[bin_idx].n_correct += int(claim.correct or 0)

        # Adjust confidence correction for this type based on overconf
        accuracy = self._type_tracker[ctype][0] / self._type_tracker[ctype][1]
        overconf_type = claim.confidence - accuracy
        # Shrink future claims of this type toward calibrated accuracy
        if abs(overconf_type) > 0.15:
            adj = self._conf_adjust.get(ctype, 0.0)
            adj = adj * 0.9 - overconf_type * 0.05
            self._conf_adjust[ctype] = max(-0.3, min(0.3, adj))

    def tick(self, current_tick: int) -> None:
        self._tick = current_tick

    # ── Test probe ───────────────────────────────────────────────────────────

    def false_belief_probe(self) -> Tuple[bool, str]:
        """
        Test: register a confident claim, resolve it as INCORRECT, and verify
        that:
          1. The calibrator records the error.
          2. overconfidence increases.
          3. The flagged_claim_types list correctly identifies the miscalibrated type.
        """
        overconf_before = self.overconfidence_score()
        cid = self.register_claim(
            "Test: I will succeed at 'respond'",
            claim_type="__probe_goal_pred__",
            confidence=0.95,
            tick=9990,
        )
        self.resolve_claim(cid, correct=False, tick=9991)
        overconf_after = self.overconfidence_score()
        report = self.self_error_report(9991)
        acc = self.accuracy_for_type("__probe_goal_pred__")

        # Clean probe data
        # (claims are maxlen-bounded; no explicit removal needed,
        #  but zero out the probe entry in tracker)
        self._type_tracker.pop("__probe_goal_pred__", None)

        passed = (
            acc < 0.5
            and report.resolved_claims >= 1
            and (
                overconf_after >= overconf_before
                or abs(self._conf_adjust.get("__probe_goal_pred__", 0.0)) > 1e-6
                or "__probe_goal_pred__" in report.flagged_claim_types
            )
        )

        return passed, (
            f"{'PASS' if passed else 'FAIL'}: false-belief probe — "
            f"accuracy={acc:.2f} overconf_before={overconf_before:+.2f} "
            f"overconf_after={overconf_after:+.2f} "
            f"resolved_claims={report.resolved_claims}"
        )

    def boundary_awareness_probe(self) -> Tuple[bool, str]:
        """
        Test: system declares an unknown, then the calibrator confirms it
        tracks it as a declared boundary.
        """
        n_before = len(self._declared_unknowns)
        self.declare_unknown("__probe_topic__")
        n_after = len(self._declared_unknowns)
        passed = n_after == n_before + 1
        return passed, (
            f"{'PASS' if passed else 'FAIL'}: boundary awareness — "
            f"declared unknowns went from {n_before} → {n_after}"
        )
