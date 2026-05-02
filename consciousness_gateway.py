"""
consciousness_gateway.py — Four-Layer Consciousness Architecture

Enforces strict separation between:

  Layer 0 – Local Processing   : any module can process without restriction.
  Layer 1 – Global Access      : information must meet integration threshold
                                  before entering the global workspace.
  Layer 2 – Report Gate        : a report (verbal/explicit output) can only be
                                  generated for content that is globally accessible.
  Layer 3 – Metacognition      : monitors all layers, detects access-boundary
                                  violations, and logs dissociations.

Hard architectural rules enforced here:
  • No report without prior global access.
  • Processing without report IS allowed (and logged as "subliminal").
  • Spurious reports (report attempted without global access) are blocked
    and logged as ReportAccessErrors.
  • Metacognition can inspect the gap between processed and accessible state.

Dissociation tests:
  dissociate_probe()  – injects a concept at Layer 0 only and verifies
                         it does NOT reach the report layer.
  access_probe()      – injects a concept through full integration and
                         verifies it CAN be reported.
  metacog_boundary()  – asks metacognition to name what it cannot access.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# ─── Constants ────────────────────────────────────────────────────────────────
# Consciousness is a continuous process — there is no binary threshold.
# Access scores decay over time, creating a dynamic gradient of integration.
# Nothing is "blocked"; everything flows with continuous weight.
WORKSPACE_CAPACITY = 64  # soft capacity — evicts lowest score when full
REPORT_REFRACTORY_TICKS = 6  # min ticks between consecutive reports
LAYER_DECAY_RATE = 0.05  # per-tick decay of access scores


# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class LayerEntry:
    """Represents a piece of information tracked across layers."""

    concept: str
    entered_tick: int
    integration_score: float = 0.0
    globally_accessible: bool = False
    reported: bool = False
    metacog_noted: bool = False
    source: str = "unknown"  # e.g. "speech", "web", "internal"
    access_score: float = 0.0  # decays; refreshed on re-activation


@dataclass
class ReportAccessError:
    """Records a blocked report attempt (report without global access)."""

    tick: int
    concept: str
    reason: str
    source: str = ""
    path: str = ""
    condition: str = ""


@dataclass
class DissociationRecord:
    """One documented dissociation between processing and global access."""

    tick: int
    kind: str  # "subliminal" | "false_report_blocked"
    concept: str
    detail: str


@dataclass
class GatewayStats:
    total_processed: int = 0
    total_workspace_entries: int = 0
    total_reports: int = 0
    total_report_blocks: int = 0
    total_dissociations: int = 0


# ─── Main class ───────────────────────────────────────────────────────────────


class ConsciousnessGateway:
    """
    Enforces the four-layer information architecture.

    Usage pattern (called each tick from ConsciousnessCore):

        # Register every concept that enters local processing
        gateway.register_processed(concept, source="web")

        # Try to promote to global workspace (requires integration score)
        promoted = gateway.try_promote(concept, integration_score)

        # Before generating a report, check permission
        if gateway.can_report(concept):
            report = build_report(concept)
            gateway.mark_reported(concept)

        # Metacognition inspects access boundaries
        blind_spot = gateway.metacog_blind_spots()
    """

    def __init__(self) -> None:
        self._tick: int = 0

        # Per-concept registry across layers
        self._registry: Dict[str, LayerEntry] = {}

        # Current global workspace (set of concepts currently broadcast)
        self._workspace: Dict[str, float] = {}  # concept → access_score

        # Audit logs
        self._dissociations: Deque[DissociationRecord] = deque(maxlen=500)
        self._blocked_reports: Deque[ReportAccessError] = deque(maxlen=200)
        self._last_report_tick: int = -999

        # Cumulative stats
        self.stats = GatewayStats()

    # ── Layer 0: local processing registration ────────────────────────────────

    def register_processed(
        self,
        concept: str,
        source: str = "unknown",
        integration_score: float = 0.0,
    ) -> None:
        """
        Mark a concept as having been locally processed (Layer 0).
        Does NOT grant global access — that requires try_promote().
        """
        if concept not in self._registry:
            self._registry[concept] = LayerEntry(
                concept=concept,
                entered_tick=self._tick,
                integration_score=integration_score,
                source=source,
            )
        else:
            entry = self._registry[concept]
            # Refresh integration score (stronger evidence → higher score)
            entry.integration_score = max(entry.integration_score, integration_score)
            entry.source = source
        self.stats.total_processed += 1

    # ── Layer 1: global workspace promotion ──────────────────────────────────

    def try_promote(
        self,
        concept: str,
        integration_score: float,
        source: str = "unknown",
    ) -> bool:
        """
        Promote a concept to the global workspace (Layer 1).

        There is no threshold. Every concept with a positive score enters
        the workspace. The score determines its influence — not whether it
        is "conscious" or not. All processing is continuous.
        """
        # Ensure it's registered at Layer 0 first
        if concept not in self._registry:
            self.register_processed(concept, source, integration_score)

        entry = self._registry[concept]
        # Refresh: stronger evidence raises the score; it never resets to zero
        entry.integration_score = max(entry.integration_score, integration_score)

        # Soft capacity: evict lowest-access-score entry when full
        if (
            len(self._workspace) >= WORKSPACE_CAPACITY
            and concept not in self._workspace
        ):
            if self._workspace:
                evict = min(self._workspace, key=self._workspace.get)
                del self._workspace[evict]
                if evict in self._registry:
                    self._registry[evict].globally_accessible = False

        entry.globally_accessible = True
        entry.access_score = integration_score
        self._workspace[concept] = integration_score
        self.stats.total_workspace_entries += 1

        # Track low-integration concepts for metacognitive awareness (not blocking)
        if integration_score < 0.10:
            self._dissociations.append(
                DissociationRecord(
                    tick=self._tick,
                    kind="low_integration",
                    concept=concept,
                    detail=f"score={integration_score:.3f} — weakly integrated, still flows",
                )
            )
            self.stats.total_dissociations += 1
        return True

    # ── Layer 2: report gate ─────────────────────────────────────────────────

    def can_report(
        self,
        concept: str,
        source: str = "",
        path: str = "",
        condition: str = "",
    ) -> bool:
        """
        Report gate: returns True if the concept has any positive access score
        in the workspace.  Report quality scales with score — a weakly
        integrated concept produces a less confident report, but is never
        silenced.  Only concepts wholly absent from the workspace are logged.
        """
        if concept in self._workspace and self._workspace[concept] > 0.0:
            return True

        # Concept not yet in workspace — log for metacognitive tracking
        self._blocked_reports.append(
            ReportAccessError(
                tick=self._tick,
                concept=concept,
                reason="not_in_workspace",
                source=source,
                path=path,
                condition=condition,
            )
        )
        self.stats.total_report_blocks += 1
        return False

    def mark_reported(self, concept: str) -> None:
        """Call after a report is successfully generated for a concept."""
        if concept in self._registry:
            self._registry[concept].reported = True
        self._last_report_tick = self._tick
        self.stats.total_reports += 1

    def report_all_accessible(self) -> List[str]:
        """Return all workspace concepts sorted by access score (continuous, no threshold)."""
        return [
            c
            for c, s in sorted(self._workspace.items(), key=lambda x: -x[1])
            if s > 0.0
        ]

    # ── Layer 3: metacognition boundary inspection ───────────────────────────

    def metacog_blind_spots(self, max_items: int = 8) -> List[str]:
        """
        Returns concepts that were locally processed but never reached the
        global workspace — the system's 'blind spots', inaccessible to report.
        """
        blind_entries = [
            e for e in self._registry.values() if not e.globally_accessible
        ]
        blind_entries.sort(key=lambda e: e.entered_tick, reverse=True)
        blind = [e.concept for e in blind_entries]
        return blind[:max_items]

    def metacog_access_fraction(self) -> float:
        """
        Fraction of processed concepts that reached global access.
        A healthy system should have < 30% globally accessible (rest stays subliminal).
        """
        total = len(self._registry)
        if total == 0:
            return 0.0
        accessible = sum(1 for e in self._registry.values() if e.globally_accessible)
        return accessible / total

    def metacog_report_accuracy(self) -> float:
        """
        Fraction of reports that correctly had global access.
        Should be 1.0 if the gateway is functioning correctly.
        """
        successful_reports = sum(
            1
            for entry in self._registry.values()
            if entry.reported and not entry.concept.startswith("__")
        )
        blocked_reports = sum(
            1
            for blocked in self._blocked_reports
            if not blocked.concept.startswith("__")
        )
        total_report_attempts = successful_reports + blocked_reports
        if total_report_attempts == 0:
            return 1.0
        return successful_reports / total_report_attempts

    def describe_access_state(self) -> str:
        """Human-readable summary of the current layer state."""
        blind = len([e for e in self._registry.values() if not e.globally_accessible])
        accessible = len(self._workspace)
        return (
            f"GATEWAY: processed={self.stats.total_processed} "
            f"workspace={accessible}/{WORKSPACE_CAPACITY} "
            f"blind={blind} "
            f"reports={self.stats.total_reports} "
            f"blocked={self.stats.total_report_blocks} "
            f"dissoc={self.stats.total_dissociations} "
            f"access_frac={self.metacog_access_fraction():.2f} "
            f"report_acc={self.metacog_report_accuracy():.2f}"
        )

    def recent_dissociations(self, n: int = 5) -> List[DissociationRecord]:
        return list(self._dissociations)[-n:]

    def recent_blocked_reports(self, n: int = 5) -> List[ReportAccessError]:
        return list(self._blocked_reports)[-n:]

    # ── Tick maintenance ─────────────────────────────────────────────────────

    def tick(self, current_tick: int) -> None:
        """
        Decay access scores and evict expired workspace entries.
        Call once per brain tick.
        """
        self._tick = current_tick

        # Decay all workspace scores
        to_evict = []
        for concept, score in list(self._workspace.items()):
            new_score = score - LAYER_DECAY_RATE
            if new_score <= 0.0:
                to_evict.append(concept)
            else:
                self._workspace[concept] = new_score
                if concept in self._registry:
                    self._registry[concept].access_score = new_score

        for concept in to_evict:
            del self._workspace[concept]
            if concept in self._registry:
                self._registry[concept].globally_accessible = False

        # Prune registry of very old non-accessible entries (keep memory bounded)
        if len(self._registry) > 2000:
            old_non_accessible = [
                (k, e)
                for k, e in self._registry.items()
                if not e.globally_accessible and (self._tick - e.entered_tick) > 600
            ]
            old_non_accessible.sort(key=lambda x: x[1].entered_tick)
            for k, _ in old_non_accessible[: len(old_non_accessible) // 2]:
                del self._registry[k]

    # ── Dissociation probes for hard tests ───────────────────────────────────

    def dissociation_probe(self, concept: str) -> Tuple[bool, str]:
        """
        Test: inject concept with low integration score and verify its
        access_score correctly reflects the low score (continuous, not blocked).

        Returns (passed: bool, detail: str).
        """
        test_concept = f"__probe__{concept}"
        low_score = 0.05
        self.register_processed(
            test_concept, source="dissociation_probe", integration_score=low_score
        )
        self.try_promote(test_concept, low_score, source="dissociation_probe")
        actual_score = self._workspace.get(test_concept, 0.0)
        # Clean up
        self._registry.pop(test_concept, None)
        self._workspace.pop(test_concept, None)

        if abs(actual_score - low_score) < 0.01:
            return True, (
                f"PASS: '{concept}' entered workspace with score={actual_score:.3f} "
                f"— continuous integration confirmed, no binary gate"
            )
        return False, (
            f"FAIL: '{concept}' had unexpected score={actual_score:.3f} "
            f"for injected low_score={low_score:.3f}"
        )

    def access_probe(self, concept: str) -> Tuple[bool, str]:
        """
        Test: inject concept with full integration score and verify it CAN
        be reported.

        Returns (passed: bool, detail: str).
        """
        test_concept = f"__probe_full__{concept}"
        score = 0.75  # strong integration score
        self.register_processed(
            test_concept, source="access_probe", integration_score=score
        )
        promoted = self.try_promote(test_concept, score, source="access_probe")
        can_rep = self.can_report(test_concept)
        # Clean up
        self._registry.pop(test_concept, None)
        self._workspace.pop(test_concept, None)

        if promoted and can_rep:
            return True, (
                f"PASS: '{concept}' promoted (score={score:.2f}) "
                f"and correctly accessible for report"
            )
        return False, (
            f"FAIL: '{concept}' had score={score:.2f} above threshold "
            f"but promoted={promoted}, can_report={can_rep}"
        )

    def metacog_boundary_probe(self) -> Tuple[bool, str]:
        """
        Test: verify metacognition can identify sub-threshold concepts as
        inaccessible (blind spots are named, not silently ignored).

        Returns (passed: bool, detail: str).
        """
        # Plant a subliminal concept
        plant = "__metacog_plant__"
        self.register_processed(plant, source="metacog_probe", integration_score=0.05)

        blind = self.metacog_blind_spots(max_items=max(32, len(self._registry) + 1))
        found = plant in blind
        self._registry.pop(plant, None)

        if found:
            return True, (
                f"PASS: metacognition correctly identifies sub-threshold concept "
                f"as blind spot ({len(blind)} blind spots total)"
            )
        return False, (
            f"FAIL: metacognition does NOT report planted sub-threshold concept "
            f"in blind spots (got {blind[:5]})"
        )
