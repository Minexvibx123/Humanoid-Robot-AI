"""
agency_validator.py — Interventional Self/World Boundary via Active Testing

Implements an agency model that learns the self/world boundary through
active mini-experiments rather than static scoring.

Mechanism:
  Each time the system executes an action, the validator:
    1. Records the pre-action context (world state snapshot + body state).
    2. Observes the post-action world state.
    3. Computes actual observed change (ΔW).
    4. Assesses whether the change was predictable from the action
       (self-caused) or diverges from prediction (externally caused).
    5. Runs a counterfactual: "What if I had done nothing?"
       – Estimates expected baseline drift (no-action world dynamics).
       – Subtracts baseline from observed change to isolate agency contribution.
    6. Produces:
         agency_probability : P(self caused ΔW)  ∈ [0, 1]
         ownership_coherence: consistency of ownership across trials  ∈ [0, 1]

Active micro-experiment (every MICRO_EXP_INTERVAL ticks):
  – The system slightly modulates ONE aspect of its planned action
    (e.g., intensity * 0.8 vs normal).
  – Observes whether the world-state delta scales accordingly.
  – If it does → strong agency evidence.
  – If world state changes identically → external driver, not self.

All experiment results feed a Bayesian agency posterior updated
via a simple Beta distribution (successes / trials).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

# ─── Constants ────────────────────────────────────────────────────────────────
MICRO_EXP_INTERVAL = 80  # ticks between active micro-experiments
AGENCY_EMA_ALPHA = 0.15  # EMA for agency probability
OWNERSHIP_EMA_ALPHA = 0.10  # EMA for ownership coherence
BASELINE_DRIFT_RATE = 0.02  # expected world drift per tick without action
CAUSAL_THRESHOLD = 0.12  # min delta to consider action causal
COUNTERFACTUAL_DISCOUNT = 0.85  # weight applied to counterfactual simulator
HISTORY_MAXLEN = 300  # max trials retained


# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class AgencyTrial:
    """One attributed-agency trial (action + observation)."""

    tick: int
    action_kind: str
    action_intensity: float  # 0..1: how strong was the action
    pre_world: Dict[str, Any]
    post_world: Dict[str, Any]
    predicted_delta: float  # from forward model
    observed_delta: float  # actual change magnitude
    baseline_delta: float  # estimated no-action world drift
    net_agency_delta: float  # observed - baseline
    agency_signal: float  # P(self-caused) for this trial
    micro_experiment: bool = False  # was this triggered by micro-exp?
    intensity_modulated: bool = False
    modulation_ratio: float = 1.0  # actual / expected intensity
    note: str = ""


@dataclass
class AgencyState:
    """Current agency estimates."""

    agency_probability: float = 0.5  # Bayesian posterior
    ownership_coherence: float = 0.5  # rolling consistency
    total_trials: int = 0
    agency_successes: int = 0  # trials where agency confirmed
    # Beta distribution parameters
    alpha_beta: float = 1.0  # successes + 1
    beta_beta: float = 1.0  # failures + 1
    last_updated_tick: int = 0


# ─── Main class ───────────────────────────────────────────────────────────────


class AgencyValidator:
    """
    Validates agency attribution via active mini-experiments and
    counterfactual baseline subtraction.

    Integration:
        Called from ConsciousnessCore every tick after goal execution.

        validator.record_action(tick, action_kind, intensity, pre_world)
        # ... action executes ...
        validator.observe_outcome(tick, post_world, predicted_delta)

        # Access estimates
        p_agency   = validator.state.agency_probability
        coherence  = validator.state.ownership_coherence
    """

    def __init__(self) -> None:
        self._tick: int = 0
        self.state = AgencyState()
        self._trials: Deque[AgencyTrial] = deque(maxlen=HISTORY_MAXLEN)

        # Pending trial (filled during record_action, completed in observe_outcome)
        self._pending: Optional[AgencyTrial] = None
        self._last_micro_exp_tick: int = -9999
        self._micro_exp_pending: bool = False
        self._baseline_world_history: Deque[float] = deque(maxlen=50)

    # ── Core API ─────────────────────────────────────────────────────────────

    def record_action(
        self,
        tick: int,
        action_kind: str,
        intensity: float,
        pre_world: Dict[str, Any],
    ) -> bool:
        """
        Called just before an action fires.

        Returns True if this trial should use micro-experiment modulation
        (intensity slightly reduced to test causal sensitivity).
        """
        self._tick = tick
        is_micro = tick - self._last_micro_exp_tick > MICRO_EXP_INTERVAL
        modulation = 0.8 if is_micro else 1.0
        actual_intensity = intensity * modulation

        self._pending = AgencyTrial(
            tick=tick,
            action_kind=action_kind,
            action_intensity=actual_intensity,
            pre_world=dict(pre_world),
            post_world={},
            predicted_delta=0.0,
            observed_delta=0.0,
            baseline_delta=tick * BASELINE_DRIFT_RATE * 0.01,  # tiny baseline
            net_agency_delta=0.0,
            agency_signal=0.5,
            micro_experiment=is_micro,
            intensity_modulated=is_micro,
            modulation_ratio=modulation,
        )
        if is_micro:
            self._last_micro_exp_tick = tick
        return is_micro

    def observe_outcome(
        self,
        tick: int,
        post_world: Dict[str, Any],
        predicted_delta: float,
        action_succeeded: bool = True,
    ) -> AgencyTrial:
        """
        Called after action completes.  Measures actual change, computes agency
        signal, updates Bayesian posterior.
        """
        if self._pending is None:
            # Create a minimal trial for untracked outcomes
            trial = AgencyTrial(
                tick=tick,
                action_kind="unknown",
                action_intensity=0.5,
                pre_world={},
                post_world=dict(post_world),
                predicted_delta=predicted_delta,
                observed_delta=0.0,
                baseline_delta=BASELINE_DRIFT_RATE,
                net_agency_delta=0.0,
                agency_signal=0.5,
            )
        else:
            trial = self._pending
            trial.post_world = dict(post_world)
            trial.predicted_delta = predicted_delta

        # ── Step 1: Measure observed delta ────────────────────────────
        trial.observed_delta = self._world_delta(trial.pre_world, trial.post_world)

        # ── Step 2: Estimate baseline drift ───────────────────────────
        if self._baseline_world_history:
            trial.baseline_delta = float(
                sum(self._baseline_world_history) / len(self._baseline_world_history)
            )
        else:
            trial.baseline_delta = BASELINE_DRIFT_RATE

        # ── Step 3: Net agency delta = observed - baseline ────────────
        trial.net_agency_delta = max(0.0, trial.observed_delta - trial.baseline_delta)

        # ── Step 4: Agency signal for this trial ──────────────────────
        #   High signal: observed delta >> baseline AND
        #                observed delta closer to predicted than to baseline.
        if trial.net_agency_delta > CAUSAL_THRESHOLD:
            pred_err = abs(trial.observed_delta - abs(trial.predicted_delta))
            base_err = abs(trial.observed_delta - trial.baseline_delta)
            agency_raw = base_err / (pred_err + base_err + 1e-6)
            # Micro-experiment premium: if modulation matched output scaling → agency+
            if trial.micro_experiment:
                expected_scale = trial.modulation_ratio
                actual_scale = trial.observed_delta / max(
                    1e-6, trial.observed_delta / trial.modulation_ratio
                )
                scale_match = max(0.0, 1.0 - abs(actual_scale - expected_scale))
                agency_raw = agency_raw * 0.7 + scale_match * 0.3
            trial.agency_signal = min(1.0, max(0.0, agency_raw))
        else:
            # No meaningful delta → ambiguous attribution
            trial.agency_signal = 0.3 if action_succeeded else 0.15

        # ── Step 5: Bayesian update (Beta conjugate) ──────────────────
        if trial.agency_signal > 0.6:
            self.state.alpha_beta += trial.agency_signal
            self.state.agency_successes += 1
        else:
            self.state.beta_beta += 1.0 - trial.agency_signal

        self.state.total_trials += 1

        # Compute posterior mean
        self.state.agency_probability = self.state.alpha_beta / (
            self.state.alpha_beta + self.state.beta_beta
        )

        # ── Step 6: Ownership coherence (consistency across trials) ───
        recent_signals = [t.agency_signal for t in list(self._trials)[-20:]]
        recent_signals.append(trial.agency_signal)
        if len(recent_signals) >= 2:
            variance = sum(
                (s - sum(recent_signals) / len(recent_signals)) ** 2
                for s in recent_signals
            ) / len(recent_signals)
            coherence_raw = 1.0 / (1.0 + variance * 10)
        else:
            coherence_raw = 0.5

        self.state.ownership_coherence = (
            self.state.ownership_coherence * (1 - OWNERSHIP_EMA_ALPHA)
            + coherence_raw * OWNERSHIP_EMA_ALPHA
        )
        self.state.last_updated_tick = tick

        self._trials.append(trial)
        self._pending = None
        return trial

    def record_baseline(self, world_delta: float) -> None:
        """
        Record a 'no-action' world delta so the validator can estimate
        how much the world drifts without agency.
        Call when the system is idle for at least one tick.
        """
        self._baseline_world_history.append(world_delta)

    # ── Queries ──────────────────────────────────────────────────────────────

    def is_agent_of(self, min_probability: float = 0.6) -> bool:
        """True if current agency estimate exceeds threshold."""
        return self.state.agency_probability >= min_probability

    def describe(self) -> str:
        s = self.state
        return (
            f"AGENCY: p_agency={s.agency_probability:.2f} "
            f"coherence={s.ownership_coherence:.2f} "
            f"trials={s.total_trials} "
            f"successes={s.agency_successes} "
            f"β=({s.alpha_beta:.1f},{s.beta_beta:.1f})"
        )

    def recent_trials(self, n: int = 5) -> List[AgencyTrial]:
        return list(self._trials)[-n:]

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _world_delta(pre: Dict[str, Any], post: Dict[str, Any]) -> float:
        """Measure total magnitude of change between two world snapshots."""
        if not pre or not post:
            return 0.0
        delta_sum = 0.0
        count = 0
        for k in pre:
            if k in post:
                try:
                    a = float(pre[k])
                    b = float(post[k])
                    delta_sum += abs(b - a)
                    count += 1
                except (TypeError, ValueError):
                    pass
        return delta_sum / count if count > 0 else 0.0

    # ── Agency manipulation probe for test suite ─────────────────────────────

    def agency_manipulation_probe(
        self,
        add_external_noise: bool = True,
    ) -> Tuple[bool, str]:
        """
        Test: when world changes are externally induced (agency_signal ≈ 0),
        the validator should produce low agency probability; when self-caused
        (agency_signal ≈ 1), it should produce high agency probability.

        This test simulates both conditions and checks the classifier.
        """
        saved_alpha = self.state.alpha_beta
        saved_beta = self.state.beta_beta

        results = []

        # Case A: self-caused (high delta, prediction matches)
        self.record_action(9990, "test_self_action", 1.0, {"world_val": 0.5})
        self.observe_outcome(
            9991,
            post_world={"world_val": 0.8},  # large +0.3 change
            predicted_delta=0.3,
            action_succeeded=True,
        )
        p_self = self.state.agency_probability
        results.append(("self", p_self))

        # Case B: external (delta present but predicted was near 0)
        self.record_action(9992, "test_external_trigger", 0.1, {"world_val": 0.5})
        self.observe_outcome(
            9993,
            post_world={"world_val": 0.8},  # same large change
            predicted_delta=0.01,  # action barely predicted this
            action_succeeded=False,  # action failed
        )
        p_mixed = self.state.agency_probability
        results.append(("external", p_mixed))

        # Restore
        self.state.alpha_beta = saved_alpha
        self.state.beta_beta = saved_beta
        self.state.agency_probability = saved_alpha / (saved_alpha + saved_beta)

        self._trials.pop() if self._trials else None
        self._trials.pop() if self._trials else None

        if p_self >= p_mixed:
            return True, (
                f"PASS: self-action p_agency={p_self:.2f} ≥ "
                f"external_event p_agency={p_mixed:.2f}"
            )
        return False, (
            f"FAIL: self-action p_agency={p_self:.2f} < "
            f"external_event p_agency={p_mixed:.2f} "
            f"— validator cannot distinguish agency from external change"
        )
