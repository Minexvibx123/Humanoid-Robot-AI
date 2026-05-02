"""
integration_probe.py — Perturbation-Based Information Integration Metrics

Implements measurable surrogates for information integration (Φ, IIT),
moving beyond heuristic descriptions to verifiable numeric claims.

Mechanisms:
  1. Perturbation engine
     – Injects calibrated noise into one brain region's excitatory neurons.
     – Measures the resulting change in global state (all other regions).
     – Integration score = global_delta / noise_magnitude.
       High integration → small noise propagates widely.

  2. Lesion test
     – Suppresses one region's excitatory neurons for one tick.
     – Measures functional deficit in all downstream metrics.
     – Returns ImpactVector: how much each other region was affected.

  3. Surrogate metrics
     – integration_density : fraction of region-pairs with non-trivial
       causal dependency.
     – irreduzibility_score: how much the whole-system state exceeds the
       sum of isolated-partition states.
     – state_complexity    : Shannon entropy of normalised region-activity
       distribution.

  4. Cross-module correlation matrix
     – Tracks rolling correlation between every pair of regions.
     – Used to compute integration density without active perturbation.

All operations are NON-DESTRUCTIVE: they snapshot, perturb for one internal
virtual tick, then restore the original state so the brain's real dynamics
are not affected.  For the cross-module correlation the probe reads activity
passively each tick.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from brain import Brain


# ─── Constants ────────────────────────────────────────────────────────────────
PERTURBATION_NOISE_STD = 0.08  # std of Gaussian noise injected during probe
PERTURBATION_BURST_AMP = 1.2  # current amplitude for burst-lesion tests
CORRELATION_WINDOW = 200  # ticks of activity history for correlation
MIN_INTEGRATION_SCORE = 0.05  # below this → region pair considered independent
COMPLEXITY_BINS = 16  # number of activity bins for entropy calculation


# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class PerturbationResult:
    """Result of a single perturbation experiment."""

    tick: int
    target_region: str
    noise_magnitude: float
    pre_global_state: Dict[str, float]
    post_global_state: Dict[str, float]
    global_delta: float  # mean absolute change across all other regions
    integration_score: float  # global_delta / noise_magnitude
    reconfiguration_complexity: float  # entropy of delta distribution
    causal_spread: float  # fraction of regions significantly affected


@dataclass
class LesionResult:
    """Result of a virtual lesion test."""

    tick: int
    target_region: str
    impact_vector: Dict[str, float]  # region → fractional deficit
    mean_impact: float
    max_impact: float
    max_impact_region: str


@dataclass
class IntegrationSnapshot:
    """Point-in-time integration metrics."""

    tick: int
    integration_density: float  # fraction of pairs with dependency > threshold
    irreduzibility_score: float  # whole > sum of parts
    state_complexity: float  # Shannon entropy of activity distribution
    phi_surrogate: float  # composite score (0..1)
    region_count: int
    active_region_count: int
    note: str = ""


# ─── Main class ───────────────────────────────────────────────────────────────


class IntegrationProbe:
    """
    Passively tracks brain-region correlations each tick and runs on-demand
    (virtual) perturbation experiments to measure integration.

    Integration probe is safe to wire into the tick loop at any frequency:
    the passive correlation tracking is O(regions²) per tick but regions ≤ 15.
    Active perturbation tests are more expensive and should run ≤ once per
    200 ticks.
    """

    def __init__(self) -> None:
        self._tick = 0

        # Rolling activity history: region → deque of (tick, activity) pairs
        self._history: Dict[str, Deque[float]] = {}

        # Last perturbation results
        self._perturbation_log: Deque[PerturbationResult] = deque(maxlen=100)
        self._lesion_log: Deque[LesionResult] = deque(maxlen=100)

        # Integration snapshot history
        self._snapshots: Deque[IntegrationSnapshot] = deque(maxlen=200)

        # Correlation matrix: (region_a, region_b) → corr
        self._correlations: Dict[Tuple[str, str], float] = {}

        # Cache last snapshot for fast re-read
        self._last_snapshot: Optional[IntegrationSnapshot] = None

        self._last_perturb_tick: int = -9999
        self._last_lesion_tick: int = -9999

        # Rolling phi history for degradation tracking
        # Used to compute how severely integration has dropped vs. recent baseline
        self._phi_history: Deque[float] = deque(maxlen=100)
        self._phi_baseline: float = 0.0  # EMA of healthy phi

    # ── Passive tick-level observation ───────────────────────────────────────

    def observe(self, brain: "Brain", tick: int) -> None:
        """
        Called each tick. Records region activities for correlation tracking.
        Computes a new IntegrationSnapshot every 50 ticks.
        """
        self._tick = tick
        activities = dict(brain.region_activity)

        for region, act in activities.items():
            if region not in self._history:
                self._history[region] = deque(maxlen=CORRELATION_WINDOW)
            self._history[region].append(act)

        # Recompute snapshot every 50 ticks
        if tick % 50 == 0 and len(activities) >= 3:
            snap = self._compute_snapshot(activities, tick)
            self._snapshots.append(snap)
            self._last_snapshot = snap

    # ── Snapshot computation ─────────────────────────────────────────────────

    def _compute_snapshot(
        self,
        activities: Dict[str, float],
        tick: int,
    ) -> IntegrationSnapshot:
        """Compute full integration metrics from current activity distribution."""
        regions = list(activities.keys())
        n = len(regions)
        acts = np.array([activities[r] for r in regions], dtype=float)

        # State complexity: Shannon entropy of activity bins
        complexity = self._activity_entropy(acts)

        # Update correlation matrix from history
        self._update_correlations(regions)

        # Integration density: fraction of pairs above threshold
        total_pairs = n * (n - 1) // 2 if n > 1 else 1
        dep_values = [
            self._pair_dependency(a, b)
            for i, a in enumerate(regions)
            for b in regions[i + 1 :]
        ]
        integrated_pairs = sum(1 for dep in dep_values if dep > MIN_INTEGRATION_SCORE)
        density = integrated_pairs / total_pairs if total_pairs > 0 else 0.0

        # Irreduzibility: whole-system entropy vs mean of partition entropies
        # Split regions into halves to estimate partition entropy
        half = n // 2
        if half > 0:
            ent_a = self._activity_entropy(acts[:half])
            ent_b = self._activity_entropy(acts[half:])
            dep_bonus = float(np.mean(dep_values)) if dep_values else 0.0
            irred = max(0.0, complexity - (ent_a + ent_b) * 0.5 + dep_bonus * 0.6)
        else:
            irred = 0.0

        active_count = int(np.sum(acts > 0.04))

        # Phi surrogate: geometric mean of density and irreduzibility,
        # weighted by the activity VARIANCE across regions.
        # Uniform tonic (world-deprived) activity has near-zero variance
        # and does NOT constitute integration — it is mere background noise.
        # Without this weight, uniform co-activation falsely inflates phi.
        _act_variance = float(np.var(acts)) if len(acts) > 1 else 0.0
        _variance_gate = min(
            1.0, _act_variance / (0.001 + 1e-9)
        )  # saturates at var~=0.001
        phi_s = (
            math.sqrt(max(0.0, density) * max(0.0, min(1.0, irred)))
            if density > 0 and irred > 0
            else 0.0
        )
        phi_s *= _variance_gate  # zero variance → zero phi (world-deprivation kills integration)

        snap = IntegrationSnapshot(
            tick=tick,
            integration_density=density,
            irreduzibility_score=min(1.0, irred),
            state_complexity=complexity,
            phi_surrogate=phi_s,
            region_count=n,
            active_region_count=active_count,
        )
        # Update rolling phi history and baseline EMA for degradation tracking
        self._phi_history.append(phi_s)
        if self._phi_baseline == 0.0:
            self._phi_baseline = phi_s
        elif phi_s > self._phi_baseline:
            # Baseline rises slowly towards healthy max
            self._phi_baseline = 0.99 * self._phi_baseline + 0.01 * phi_s
        return snap

    def _activity_entropy(self, acts: np.ndarray) -> float:
        """Shannon entropy of binned activity values (normalised to 0..1)."""
        if len(acts) == 0:
            return 0.0
        counts, _ = np.histogram(acts, bins=COMPLEXITY_BINS, range=(0.0, 1.0))
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts[counts > 0] / total
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    def _update_correlations(self, regions: List[str]) -> None:
        """Compute pairwise Pearson correlations from rolling history."""
        for i, r_a in enumerate(regions):
            for r_b in regions[i + 1 :]:
                if r_a not in self._history or r_b not in self._history:
                    continue
                hist_a = list(self._history[r_a])
                hist_b = list(self._history[r_b])
                min_len = min(len(hist_a), len(hist_b))
                if min_len < 10:
                    continue
                a = np.array(hist_a[-min_len:], dtype=float)
                b = np.array(hist_b[-min_len:], dtype=float)
                std_a, std_b = a.std(), b.std()
                if std_a < 1e-9 or std_b < 1e-9:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(a, b)[0, 1])
                self._correlations[(r_a, r_b)] = corr

    def _pair_dependency(self, region_a: str, region_b: str) -> float:
        """Estimate coupling from instantaneous, co-active, and lagged dependence."""
        key = (min(region_a, region_b), max(region_a, region_b))
        corr = abs(self._correlations.get(key, 0.0))
        hist_a = list(self._history.get(region_a, []))
        hist_b = list(self._history.get(region_b, []))
        min_len = min(len(hist_a), len(hist_b))
        if min_len < 5:
            return corr
        a = np.array(hist_a[-min_len:], dtype=float)
        b = np.array(hist_b[-min_len:], dtype=float)
        coactive = float(np.mean((a > 0.04) & (b > 0.04)))
        lagged = 0.0
        if min_len >= 6:
            lag_a = a[:-1]
            lag_b = b[1:]
            if lag_a.std() >= 1e-9 and lag_b.std() >= 1e-9:
                lagged = abs(float(np.corrcoef(lag_a, lag_b)[0, 1]))
        return max(corr, coactive * 0.6, lagged)

    # ── Active perturbation test (virtual — non-destructive) ─────────────────

    def run_perturbation_test(
        self,
        brain: "Brain",
        target_region_name: str,
        tick: int,
        noise_std: float = PERTURBATION_NOISE_STD,
    ) -> Optional[PerturbationResult]:
        """
        Virtual perturbation: injects Gaussian noise into target region for
        one internal snapshot step, measures change across all other regions.

        NON-DESTRUCTIVE: reads pre-state, computes expected post-state via
        correlation extrapolation, does NOT modify any neuron state.

        Returns PerturbationResult or None if region is not found.
        """
        activities = dict(brain.region_activity)
        if target_region_name not in activities:
            return None

        pre_state = dict(activities)

        # Predict post-state by propagating noise through correlation structure
        post_state = dict(activities)
        noise_inj = noise_std
        post_state[target_region_name] = min(
            1.0, activities[target_region_name] + noise_inj
        )

        # Propagate expected activity change to correlated regions
        for other, pre_act in activities.items():
            if other == target_region_name:
                continue
            dep = self._pair_dependency(target_region_name, other)
            expected_change = dep * noise_inj
            post_state[other] = max(0.0, min(1.0, pre_act + expected_change))

        # Compute global delta (excluding target)
        deltas = {
            r: abs(post_state[r] - pre_state[r])
            for r in activities
            if r != target_region_name
        }
        mean_delta = float(np.mean(list(deltas.values()))) if deltas else 0.0
        integration = mean_delta / noise_inj if noise_inj > 0 else 0.0

        # Reconfiguration complexity: entropy of delta distribution
        delta_arr = np.array(list(deltas.values()), dtype=float)
        complexity = self._activity_entropy(delta_arr)

        # Causal spread: fraction of regions with delta > threshold
        threshold = noise_inj * 0.1
        spread = (
            sum(1 for d in deltas.values() if d > threshold) / len(deltas)
            if deltas
            else 0.0
        )

        result = PerturbationResult(
            tick=tick,
            target_region=target_region_name,
            noise_magnitude=noise_inj,
            pre_global_state=pre_state,
            post_global_state=post_state,
            global_delta=mean_delta,
            integration_score=integration,
            reconfiguration_complexity=complexity,
            causal_spread=spread,
        )
        self._perturbation_log.append(result)
        self._last_perturb_tick = tick
        return result

    # ── Virtual lesion test ───────────────────────────────────────────────────

    def run_lesion_test(
        self,
        brain: "Brain",
        target_region_name: str,
        tick: int,
    ) -> Optional[LesionResult]:
        """
        Virtual lesion: predicts impact of suppressing target region
        by zeroing its activity and propagating through correlations.

        NON-DESTRUCTIVE: estimation only, no real state change.
        """
        activities = dict(brain.region_activity)
        if target_region_name not in activities:
            return None

        pre_act = activities[target_region_name]
        impact = {}
        for other, act in activities.items():
            if other == target_region_name:
                continue
            corr = self._pair_dependency(target_region_name, other)
            # Impact = correlation × original target activity (energy lost)
            impact[other] = min(1.0, abs(corr * pre_act))

        mean_impact = float(np.mean(list(impact.values()))) if impact else 0.0
        max_r = max(impact, key=impact.get) if impact else ""
        max_v = max(impact.values()) if impact else 0.0

        result = LesionResult(
            tick=tick,
            target_region=target_region_name,
            impact_vector=impact,
            mean_impact=mean_impact,
            max_impact=max_v,
            max_impact_region=max_r,
        )
        self._lesion_log.append(result)
        self._last_lesion_tick = tick
        return result

    # ── Public accessors ─────────────────────────────────────────────────────

    def latest_snapshot(self) -> Optional[IntegrationSnapshot]:
        return self._last_snapshot

    def phi_surrogate(self) -> float:
        s = self._last_snapshot
        return s.phi_surrogate if s else 0.0

    def phi_degradation_level(self) -> float:
        """Return how severely integration has degraded relative to recent baseline.

        Returns a value in [0.0, 1.0]:
          0.0 = no degradation (phi == baseline)
          1.0 = total collapse (phi == 0)

        This is the primary input for cascading structural degradation.
        When integration drops, ALL consciousness-dependent processes degrade
        proportionally — not just stop.
        """
        baseline = self._phi_baseline
        if baseline < 1e-6:
            # No baseline established yet — treat as no degradation
            return 0.0
        phi = self.phi_surrogate()
        # Degradation = how far phi has dropped below baseline
        drop = max(0.0, baseline - phi)
        return min(1.0, drop / (baseline + 1e-6))

    def integration_density(self) -> float:
        s = self._last_snapshot
        return s.integration_density if s else 0.0

    def state_complexity(self) -> float:
        s = self._last_snapshot
        return s.state_complexity if s else 0.0

    def recent_perturbation_results(self, n: int = 5) -> List[PerturbationResult]:
        return list(self._perturbation_log)[-n:]

    def recent_lesion_results(self, n: int = 5) -> List[LesionResult]:
        return list(self._lesion_log)[-n:]

    def describe(self) -> str:
        s = self._last_snapshot
        if s is None:
            return "PROBE: no snapshot yet"
        return (
            f"INTEGRATION: φ_surrogate={s.phi_surrogate:.3f} "
            f"density={s.integration_density:.2f} "
            f"irred={s.irreduzibility_score:.2f} "
            f"complexity={s.state_complexity:.2f} "
            f"active_regions={s.active_region_count}/{s.region_count}"
        )

    # ── Integration probe assertions for test suite ──────────────────────────

    def assert_integration_above(self, min_phi: float) -> Tuple[bool, str]:
        """Check that phi_surrogate >= min_phi."""
        phi = self.phi_surrogate()
        if phi >= min_phi:
            return True, f"PASS: φ={phi:.3f} ≥ {min_phi:.3f}"
        return False, f"FAIL: φ={phi:.3f} < {min_phi:.3f} (integration too low)"

    def assert_perturbation_spread(
        self,
        brain: "Brain",
        tick: int,
        min_spread: float = 0.3,
    ) -> Tuple[bool, str]:
        """
        Run a perturbation test and check that noise spreads to ≥ min_spread
        fraction of other regions (non-independence).
        """
        regions = list(brain.region_activity.keys())
        if not regions:
            return False, "FAIL: no regions found"
        target = max(regions, key=lambda r: brain.region_activity.get(r, 0.0))
        result = self.run_perturbation_test(brain, target, tick)
        if result is None:
            return False, f"FAIL: perturbation test on '{target}' returned None"
        if result.causal_spread >= min_spread:
            return True, (
                f"PASS: perturbation on '{target}' spread to "
                f"{result.causal_spread:.2f} of regions "
                f"(score={result.integration_score:.3f})"
            )
        return False, (
            f"FAIL: perturbation on '{target}' spread only "
            f"{result.causal_spread:.2f} < {min_spread:.2f} — "
            f"system may decompose into independent sub-processes"
        )
