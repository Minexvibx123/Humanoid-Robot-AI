"""
brain.py — Central Brain Orchestrator

Wires all regions together, runs the simulation clock,
manages structural plasticity, and produces actions.

Also hosts the UtterancePlan post-build block where Modules 19–20
of HumanInteractionSuite (MicrobehaviorController + PresenceSynchronizer)
adjust deliberation_delay_ms, speed_factor, head_nod, gaze_at_person,
and confidence before speech output is dispatched.

Architecture (signal flow):
                                         ┌─────────────┐
  Camera ──► VisualEncoder ──► SensoryV ─►             │
                                          │  Thalamus   │
  Mic    ──► AudioEncoder  ──► SensoryA ─►             │
                                         └──────┬──────┘
                          ┌─────────────────────┼────────────┐
                          ▼                     ▼            ▼
                     PrimaryVisual        PrimaryAuditory    │
                          │                     │            │
                          └─────────┬───────────┘            │
                                    ▼                        │
                             AssociationCortex               │
                                    │                        │
                          ┌─────────┴──────────┐            │
                          ▼                    ▼             │
                     Hippocampus          Amygdala ──────────┘
                          │                    │
                          └─────────┬──────────┘
                                    ▼
                            PrefrontalCortex
                                    │
                               MotorCortex → action string
"""

from __future__ import annotations

import collections
import json
import os
import queue
import random
import threading
import time
from typing import Dict, List, Optional

import numpy as np

import neuron as _neuron_mod  # module-level ref for per-tick A_PLUS modulation
from actions import ActionToolbelt
from body_schema import BodySchema
from consciousness import (
    ConsciousnessCore,
    ConsciousnessState,
    EpistemicStatus,
    GoalSystemFailure,
)
from consciousness_testbed import ConsciousnessTestbed
from dialogue_manager import DialogueManager
from emotion import EmotionalState, EmotionEngine
from neuron import Neuron
from regions import (
    Amygdala,
    AssociationCortex,
    Hippocampus,
    MotorCortex,
    PrefrontalCortex,
    PrimaryAuditoryCortex,
    PrimaryVisualCortex,
    Region,
    SensoryInputRegion,
    Thalamus,
    _connect_random,
    _spatial_delay,
    _spatial_weight,
)
from robot_controller import RobotController
from robot_serial import ArduinoSerialLink
from safety_supervisor import SafetySupervisor
from sensors import AudioEncoder, SpeechListener, VisionAnalyzer, VisualEncoder
from sim_bridge import BackendMode, BodyInterface
from skill_library import SkillLibrary
from social_manager import SocialManager
from speech_output import SpeechOutput
from synapse import Synapse
from task_executive import TaskExecutive
from telemetry_bus import TelemetryBus
from web_sensor import N_NEURONS as WEB_N
from web_sensor import WebSensor
from world_state import WorldState

# Use all physical cores for numba's OpenMP parallel LIF (set before first import of regions)
try:
    import os as _os

    import numba as _numba

    _n_threads = max(4, _os.cpu_count() or 4)
    _numba.set_num_threads(_n_threads)
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SIM_DT = 1.0  # ms per simulation tick
# Optional wall-clock throttle for the main loop.
# Default is unthrottled because a hard 5 s sleep dominates tick time and
# completely masks the actual compute cost of the neural stack.
REAL_TICK_MS = max(0, int(os.environ.get("REAL_TICK_MS", "0")))
PLASTICITY_INTERVAL = 30  # STDP sweep every 30 ticks (was 3 — caused 356ms/tick)
AUTO_SAVE_INTERVAL = (
    10_000  # autosave every 10 000 ticks (was 2 000 — saves less often = less overhead)
)

# Tonic baseline current applied to the sensory_web region every tick.
# Models persistent excitatory drive from thalamus/brainstem.
# Must be > (V_THRESHOLD - V_REST) = 15 nA to cause spontaneous firing.
TONIC_CURRENT = 14.0  # nA — V_inf = V_REST+14 = -56mV (1mV below threshold)

# Synapse growth parameters
PRUNE_THRESHOLD = 0.008  # weight below this → pruned
SPROUT_THRESHOLD = 2.0  # weight above this → candidate to sprout (was 2.5)
SPROUT_PROB = 0.55  # probability saturated syn sprouts (was 0.35)
MAX_SYNAPSES = 4_000_000  # hard cap — 4 M inter-regional synapses (was 1.5 M)
COACTIVATION_PER_TICK = 800  # max new Hebbian synapses per tick (was 200)
BASE_A_PLUS = 0.015  # baseline STDP LTP rate (was 0.01 — +50% learning speed)
SPATIAL_LOCAL_RADIUS = 8.0  # nominal spatial neighborhood radius for new sprouts

# ── Sleep / Wake rhythm ──────────────────────────────────────────────────────
# Every SLEEP_CYCLE_TICKS the brain alternates between WAKE (normal activity)
# and SLEEP (reduced tonic, intense replay, memory consolidation).
SLEEP_CYCLE_TICKS = 8_000  # ticks per full wake→sleep→wake cycle (~80s at 100Hz)
SLEEP_FRACTION = 0.25  # fraction of cycle spent in sleep (= 2000 ticks)
SLEEP_TONIC = 3.0  # nA tonic during sleep (V_inf ≈ -67mV — subthreshold)
WAKE_TONIC = TONIC_CURRENT  # restored on wake

# ─────────────────────────────────────────────────────────────
# Session Metrics — lightweight per-session JSONL telemetry
# ─────────────────────────────────────────────────────────────
SESSION_SAMPLE_INTERVAL = 500  # sample every N ticks (was 100)


class SessionMetrics:
    """Collects per-tick snapshots and exports JSONL on session end."""

    def __init__(self) -> None:
        self._samples: List[Dict] = []
        self._start_tick: int = 0
        self._start_wall: float = time.time()

    def sample(self, brain: "Brain") -> None:
        """Capture a metric snapshot — called every SESSION_SAMPLE_INTERVAL ticks."""
        em = brain.emotion_state
        cc = brain._consciousness
        sm = cc.self_model
        sample = {
            "tick": brain.tick_count,
            "wall_s": round(time.time() - self._start_wall, 2),
            "sleeping": brain.sleeping,
            # Emotion (8 dims)
            "emo_joy": round(em.joy, 3),
            "emo_sadness": round(em.sadness, 3),
            "emo_anger": round(em.anger, 3),
            "emo_fear": round(em.fear, 3),
            "emo_surprise": round(em.surprise, 3),
            "emo_calm": round(em.calm, 3),
            "emo_stress": round(em.stress, 3),
            "emo_fatigue": round(em.fatigue, 3),
            # Self-model
            "sm_energy": round(sm.energy, 3),
            "sm_uncertainty": round(sm.uncertainty, 3),
            "sm_identity": round(sm.identity_stability, 3),
            "sm_agency": round(sm.agency_score, 3),
            "sm_body_load": round(sm.body_load, 3),
            "sm_body_pain": round(sm.body_pain, 3),
            # Cognition
            "ignitions": cc._ignition_count,
            "concepts": sm.concepts_learned,
            "episodic_count": len(cc.episodic._events),
            "stream_len": len(cc.stream),
            # Drives
            "dr_info": round(cc.drives.information_hunger, 3),
            "dr_coherence": round(cc.drives.coherence_need, 3),
            "dr_expression": round(cc.drives.expression_pressure, 3),
            "dr_rest": round(cc.drives.rest_need, 3),
            # Infrastructure
            "synapses_formed": brain._synapses_formed,
            "region_activity": {
                k: round(v, 3) for k, v in brain.region_activity.items()
            },
            # Sensor health
            "sensor_health": sm.sensor_health.describe(),
        }
        try:
            sample["continuity"] = round(cc.continuity.overall_continuity(), 3)
        except Exception:
            pass
        try:
            sample["pred_error"] = round(brain._consciousness_pred_error, 3)
        except Exception:
            pass
        self._samples.append(sample)

    def export_jsonl(self, path: str = "session_metrics.jsonl") -> str:
        """Write all samples as JSONL. Returns the path written."""
        with open(path, "w", encoding="utf-8") as f:
            for s in self._samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return path

    @property
    def n_samples(self) -> int:
        return len(self._samples)


# ─────────────────────────────────────────────────────────────────────────────
# IrreversibleDegradationTracker — Permanent Substrate Damage (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────


class IrreversibleDegradationTracker:
    """Tracks three independent channels of permanent, non-reversible degradation.

    Channel 1 — Region fatigue:
      High sustained activity in any region → slow accumulation of fatigue [0,1].
      Fatigued regions receive an inhibitory injection each tick, proportional
      to their fatigue score.  Fatigue only very slowly recovers (RECOVERY ≪
      ACCUMULATION), so regions overworked in the past are permanently weaker.
      At fatigue > 0.55, the region begins to self-suppress.

    Channel 2 — Thermal scarring:
      When thermal_load exceeds SCAR_THRESHOLD, a permanent tonic-ceiling reduction
      accumulates in `thermal_scar` [0, 8 nA].  This ceiling is NEVER repaired —
      it is applied every tick to clamp Neuron.global_tonic_current.  The system
      can never again sustain the tonic drive it had before overheating.

    Channel 3 — Structural poverty prune:
      When energy_reserve < POVERTY_THRESHOLD for POVERTY_TICKS consecutive ticks,
      the POVERTY_PRUNE weakest inter-regional synapses are permanently zeroed
      (weight → 0.0).  Structural plasticity may eventually prune them, but even
      at weight=0 they remain as dead connections that block sprouting at those
      locations — structural impoverishment that outlasts the energy crisis.
    """

    _SCAR_THRESHOLD = 0.85
    _SCAR_RATE = 0.00028  # thermal_scar gain per tick above threshold
    _POVERTY_THRESHOLD = 0.12
    _POVERTY_TICKS = 48  # consecutive ticks in poverty before prune
    _POVERTY_PRUNE = 6  # synapses zeroed per poverty event
    _FATIGUE_RATE = 0.00020  # fatigue gain per unit activity above 0.3
    _FATIGUE_RECOVERY = 0.000042  # fatigue loss per tick (partial, never full)
    _TONIC_SCAR_CAP = 7.0  # maximum permanent tonic ceiling reduction

    def __init__(self) -> None:
        self.region_fatigue: Dict[str, float] = {}
        self.thermal_scar: float = 0.0
        self.poverty_ticks: int = 0
        self.poverty_events: int = 0
        self.total_synapses_pruned: int = 0

    def update(self, regions: list, body: object, synapses: list) -> List[str]:
        """Advance degradation one tick.  Returns list of stream messages."""
        msgs: List[str] = []

        # ── Channel 1: Region fatigue ─────────────────────────────────────
        for r in regions:
            _act = r.activity()
            _fat = self.region_fatigue.get(r.name, 0.0)
            if _act > 0.30:
                _fat = min(1.0, _fat + _act * self._FATIGUE_RATE)
            else:
                _fat = max(0.0, _fat - self._FATIGUE_RECOVERY)
            self.region_fatigue[r.name] = _fat
            # Fatigue-driven self-suppression
            if _fat > 0.55:
                _n = len(getattr(r, "_exc_cache", []))
                if _n > 0:
                    _k = max(1, int(_n * 0.14))
                    _amp = -(_fat - 0.50) * 3.5
                    r.inject([_amp] * _k + [0.0] * (_n - _k))

        # ── Channel 2: Thermal scarring ───────────────────────────────────
        if body.thermal_load > self._SCAR_THRESHOLD:
            _delta = (body.thermal_load - self._SCAR_THRESHOLD) * self._SCAR_RATE * 100
            _prev = self.thermal_scar
            self.thermal_scar = min(self._TONIC_SCAR_CAP, self.thermal_scar + _delta)
            if int(self.thermal_scar * 10) != int(_prev * 10):
                msgs.append(
                    f"[THERMAL-SCAR] scar={self.thermal_scar:.2f}nA "
                    f"thermal={body.thermal_load:.2f} "
                    f"→ tonic_ceil permanently reduced"
                )

        # ── Channel 3: Structural poverty prune ───────────────────────────
        if body.energy_reserve < self._POVERTY_THRESHOLD:
            self.poverty_ticks += 1
        else:
            self.poverty_ticks = max(0, self.poverty_ticks - 1)

        if self.poverty_ticks >= self._POVERTY_TICKS and synapses:
            self.poverty_ticks = 0
            self.poverty_events += 1
            import random as _rnd_pov

            _weak = sorted(synapses, key=lambda s: s.weight)
            _pool = _weak[: min(self._POVERTY_PRUNE * 5, max(1, len(_weak) // 5))]
            _to_zero = _rnd_pov.sample(_pool, min(self._POVERTY_PRUNE, len(_pool)))
            for s in _to_zero:
                s.weight = 0.0
            self.total_synapses_pruned += len(_to_zero)
            msgs.append(
                f"[POVERTY-PRUNE ev={self.poverty_events}] "
                f"zeroed={len(_to_zero)} total_pruned={self.total_synapses_pruned} "
                f"energy={body.energy_reserve:.2f}"
            )

        return msgs

    def effective_tonic_ceiling(self, base: float = 18.0) -> float:
        """Permanent ceiling = base − accumulated thermal scar.  Never repairs."""
        return max(5.0, base - self.thermal_scar)

    def region_capacity(self, name: str) -> float:
        """Residual capacity [0.30, 1.0] for a region.  Fatigue reduces capacity."""
        return max(0.30, 1.0 - self.region_fatigue.get(name, 0.0) * 0.70)


# ─────────────────────────────────────────────────────────────────────────────
# CascadeFailureMonitor — Multi-System Coupling Breakdown (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────


class CascadeFailureMonitor:
    """Detects simultaneous multi-system failure and triggers cascade events.

    Monitors five independent signals.  When ≥ THRESHOLD_COUNT are simultaneously
    in failure range, a CASCADE is triggered:

      Effects of cascade:
        1. Amygdala fear burst: injects AMY_BURST nA into 60% of excitatory neurons
        2. PFC strong inhibition: clamps all PFC excitatory neurons at -PFC_SUPPRESS
        3. AssocCortex partial inhibition: 50% of neurons suppressed
        4. Energy drain: immediate panic-metabolic cost drains energy_reserve
        5. Integrity damage: stress-induced structural damage to integrity

    Cascades are NOT compensable by short-term regulation:
      The energy + integrity drain from a cascade brings the system closer to the
      next cascade threshold — creating a positive-feedback spiral that can
      only be broken by sustained recovery (which requires ticks with no new
      high-activity/high-PE input).

    COOLDOWN ensures one cascade per COOLDOWN ticks maximum, but that cooldown
    is short enough that cascades can re-trigger during sustained failure states.
    """

    _THRESHOLD_COUNT = 3
    _COOLDOWN = 75  # ticks between cascade events
    _AMY_BURST = 20.0
    _PFC_SUPPRESS = -9.0
    _ASSOC_SUPPRESS = -4.5
    _ENERGY_DRAIN = 0.028
    _INTEGRITY_DRAIN = 0.018

    def __init__(self) -> None:
        self.cascade_count: int = 0
        self._cooldown_left: int = 0
        self.cascading: bool = False
        self.failure_log: List[str] = []

    def _count_failures(
        self,
        energy: float,
        integrity: float,
        thermal: float,
        mean_pe: float,
        regions: list,
    ) -> tuple:
        reasons: List[str] = []
        if energy < 0.20:
            reasons.append(f"energy={energy:.2f}")
        if integrity < 0.30:
            reasons.append(f"integrity={integrity:.2f}")
        if thermal > 0.80:
            reasons.append(f"thermal={thermal:.2f}")
        if mean_pe > 0.40:
            reasons.append(f"mean_pe={mean_pe:.3f}")
        # Activity coefficient of variation: chaos detection
        _acts = [r.activity() for r in regions if r.activity() > 0.01]
        if len(_acts) > 2:
            import statistics as _stat

            _cv = _stat.stdev(_acts) / (_stat.mean(_acts) + 1e-9)
            if _cv > 1.75:
                reasons.append(f"act_cv={_cv:.2f}")
        return len(reasons), reasons

    def update(
        self,
        energy: float,
        integrity: float,
        thermal: float,
        mean_pe: float,
        body: object,
        regions: list,
        amygdala: object,
        prefrontal: object,
        assoc: object,
    ) -> Optional[str]:
        """Check for cascade.  Returns log message or None.  Directly injects."""
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self.cascading = False
            return None

        n_fail, reasons = self._count_failures(
            energy, integrity, thermal, mean_pe, regions
        )

        if n_fail < self._THRESHOLD_COUNT:
            self.cascading = False
            return None

        # ── CASCADE TRIGGERED ─────────────────────────────────────────────
        self.cascade_count += 1
        self.cascading = True
        self._cooldown_left = self._COOLDOWN

        # 1. Amygdala fear burst
        _amy_exc = getattr(amygdala, "_exc_cache", [])
        _an = len(_amy_exc)
        if _an > 0:
            _k = min(max(1, int(_an * 0.60)), _an)
            amygdala.inject([self._AMY_BURST] * _k + [0.0] * (_an - _k))

        # 2. PFC strong inhibition
        _pfc_exc = getattr(prefrontal, "_exc_cache", [])
        _pn = len(_pfc_exc)
        if _pn > 0:
            prefrontal.inject([self._PFC_SUPPRESS] * _pn)

        # 3. AssocCortex partial inhibition
        _asc_exc = getattr(assoc, "_exc_cache", [])
        _asn = len(_asc_exc)
        if _asn > 0:
            _k2 = _asn // 2
            assoc.inject([self._ASSOC_SUPPRESS] * _k2 + [0.0] * (_asn - _k2))

        # 4. Body damage: energy drain + integrity loss
        body.energy_reserve = max(0.05, body.energy_reserve - self._ENERGY_DRAIN)
        body.integrity = max(0.05, body.integrity - self._INTEGRITY_DRAIN)

        msg = (
            f"[CASCADE n={self.cascade_count}] {'; '.join(reasons)} "
            f"→ amygdala_burst PFC_clamp assoc_suppress "
            f"energy→{body.energy_reserve:.2f} integrity→{body.integrity:.2f}"
        )
        self.failure_log.append(msg)
        if len(self.failure_log) > 25:
            self.failure_log.pop(0)
        return msg


# ─────────────────────────────────────────────────────────────────────────────
# PassivityDriftEngine — Autonomous Deterioration Without Action (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────


class PassivityDriftEngine:
    """Enforces existential pressure: the world and the system deteriorate
    autonomously, and only specific actions can slow or reverse the drift.

    Three independent drift channels operate every tick:

    Channel A — Metabolic Depletion:
      Energy drains at BASE_ENERGY_DRIFT per tick regardless of activity.
      The more idle ticks pass (no meaningful motor output), the faster the
      drain accelerates.  Only 'explore' / 'respond' actions reset the idle
      counter; 'rest' slows it but does not reverse metabolic debt.
      Missed intervention windows permanently increase the drift multiplier.

    Channel B — Integrity Dissolution:
      Structural integrity degrades at BASE_INTEGRITY_DRIFT per tick when no
      'consolidate' or 'rest' action occurs.  If integrity_debt exceeds
      DEBT_THRESHOLD, the dissolution rate DOUBLES — a tipping point beyond
      which passive waiting accelerates damage.  Debt itself compounds beyond
      the threshold, creating a positive-feedback spiral.

    Channel C — World Entropy:
      The latent world state drifts toward maximum entropy (all dimensions
      toward 0.5) when the system is passive.  Only 'explore' actions push
      the latent state away from 0.5.  Without exploration, obs_surprise
      systematically rises over hundreds of ticks, eventually triggering
      cascade failure.

    All three drift multipliers are strictly monotone increasing — they
    encode the permanent behavioral cost of passivity.  There is no
    mechanism to reduce them below their current value.
    """

    _BASE_ENERGY_DRIFT = 0.00018
    _BASE_INTEGRITY_DRIFT = 0.000080
    _IDLE_ACCEL_RATE = 0.000035
    _IDLE_MAX_DRAIN = 0.00065
    _DEBT_THRESHOLD = 0.28
    _WINDOW_TICKS = 120
    _MULT_INCREASE = 0.042
    _MULT_CAP = 2.8

    def __init__(self) -> None:
        self.idle_ticks: int = 0
        self.integrity_debt: float = 0.0
        self.world_entropy: float = 0.0
        self.energy_drift_mult: float = 1.0
        self.integrity_drift_mult: float = 1.0
        self.world_entropy_mult: float = 1.0
        self._energy_window: int = 0
        self._integrity_window: int = 0
        self._world_window: int = 0
        self.missed_windows: int = 0

    def update(
        self, action: str, body: object, latent_world: "LatentWorldDynamics"
    ) -> List[str]:
        """Advance drift one tick.  Modifies body + latent_world in place."""
        msgs: List[str] = []
        _act = (action or "").lower()

        # ── Channel A: Metabolic depletion ─────────────────────────────────
        _energetic = any(
            k in _act
            for k in (
                "explore",
                "respond",
                "speak",
                "reach",
                "approach",
                "interact",
                "execute",
            )
        )
        if _energetic:
            self.idle_ticks = max(0, self.idle_ticks - 8)
            self._energy_window = 0
        else:
            self.idle_ticks += 1
            self._energy_window += 1
            if self._energy_window >= self._WINDOW_TICKS:
                self.energy_drift_mult = min(
                    self._MULT_CAP, self.energy_drift_mult + self._MULT_INCREASE
                )
                self._energy_window = 0
                self.missed_windows += 1
                msgs.append(
                    f"[DRIFT-ENERGY] mult={self.energy_drift_mult:.2f} "
                    f"idle={self.idle_ticks}"
                )

        _accel = min(self._IDLE_MAX_DRAIN, self.idle_ticks * self._IDLE_ACCEL_RATE)
        _e_drain = (self._BASE_ENERGY_DRIFT + _accel) * self.energy_drift_mult
        body.energy_reserve = max(0.04, body.energy_reserve - _e_drain)

        # ── Channel B: Integrity dissolution ───────────────────────────────
        _consolidating = any(k in _act for k in ("consolidate", "rest", "sleep"))
        if _consolidating:
            self.integrity_debt = max(0.0, self.integrity_debt - 0.040)
            self._integrity_window = 0
        else:
            self.integrity_debt += 0.0012
            self._integrity_window += 1
            if self._integrity_window >= self._WINDOW_TICKS:
                self.integrity_drift_mult = min(
                    self._MULT_CAP, self.integrity_drift_mult + self._MULT_INCREASE
                )
                self._integrity_window = 0
                self.missed_windows += 1
                msgs.append(
                    f"[DRIFT-INTEG] mult={self.integrity_drift_mult:.2f} "
                    f"debt={self.integrity_debt:.3f}"
                )

        _i_rate = self._BASE_INTEGRITY_DRIFT * self.integrity_drift_mult
        if self.integrity_debt > self._DEBT_THRESHOLD:
            _i_rate *= 2.0  # tipping point: dissolution doubles
            self.integrity_debt = min(1.0, self.integrity_debt + 0.0006)
        body.integrity = max(0.05, body.integrity - _i_rate)

        # ── Channel C: World entropy drift ─────────────────────────────────
        _exploring = any(
            k in _act for k in ("explore", "search", "observe", "scan", "investigate")
        )
        if _exploring:
            self.world_entropy = max(0.0, self.world_entropy - 0.035)
            self._world_window = 0
            import numpy as _np_pde

            latent_world._state += _np_pde.random.normal(0, 0.012, latent_world._NDIM)
            latent_world._state = latent_world._state.clip(0.0, 1.0)
        else:
            self.world_entropy = min(
                1.0, self.world_entropy + 0.00060 * self.world_entropy_mult
            )
            self._world_window += 1
            if self._world_window >= self._WINDOW_TICKS:
                self.world_entropy_mult = min(
                    self._MULT_CAP, self.world_entropy_mult + self._MULT_INCREASE
                )
                self._world_window = 0
                self.missed_windows += 1
                msgs.append(
                    f"[DRIFT-WORLD] mult={self.world_entropy_mult:.2f} "
                    f"entropy={self.world_entropy:.3f}"
                )

        if self.world_entropy > 0.20:
            import numpy as _np_pde2

            _pull = (0.5 - latent_world._state) * 0.0008 * self.world_entropy
            latent_world._state = (latent_world._state + _pull).clip(0.0, 1.0)

        return msgs

    @property
    def total_pressure(self) -> float:
        """Composite passivity pressure [0, 1]."""
        return min(
            1.0,
            self.idle_ticks * self._IDLE_ACCEL_RATE * 5
            + self.integrity_debt * 0.5
            + self.world_entropy * 0.3,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PathDependencyMatrix — Irreversible Dynamic Parameter Shifts (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────


class PathDependencyMatrix:
    """Records behavioral history and permanently alters system-wide
    dynamic parameters at epoch boundaries.

    Every EPOCH_TICKS ticks, four behavioral dimensions are evaluated:
      D1 — Overexertion frequency (mean_activity > OVER_THRESH)
      D2 — Energy poverty frequency
      D3 — Cascade incidence
      D4 — Attractor lock incidence

    Each dimension, when exceeded, permanently shifts live class attributes:
      D1 → degradation._FATIGUE_RATE × 1.06   (neural overwork scar)
      D2 → PassivityDriftEngine energy_drift_mult × 1.04
      D3 → CascadeFailureMonitor._COOLDOWN − 3  (re-triggers faster)
      D4 → AttractorDominanceField._WTA_THRESHOLD − 0.008  (locks easier)

    These shifts are applied to live objects in-place.  There is no undo.
    The system's behavioral past literally changes its future physics.
    """

    _EPOCH_TICKS = 200
    _OVER_THRESH = 0.45
    _POVERTY_THRESH = 0.20
    _LOCK_FRAC = 0.30

    def __init__(self) -> None:
        self._tick_in_epoch: int = 0
        self._epoch_count: int = 0
        self._over_count: int = 0
        self._poverty_count: int = 0
        self._cascade_count: int = 0
        self._lock_count: int = 0
        self.fatigue_rate_mult: float = 1.0
        self.energy_drift_mult: float = 1.0
        self.cascade_cooldown_red: int = 0
        self.wta_threshold_red: float = 0.0
        self.epoch_log: List[str] = []

    def record_tick(
        self, mean_activity: float, energy: float, cascade_hit: bool, wta_locked: bool
    ) -> None:
        self._tick_in_epoch += 1
        if mean_activity > self._OVER_THRESH:
            self._over_count += 1
        if energy < self._POVERTY_THRESH:
            self._poverty_count += 1
        if cascade_hit:
            self._cascade_count += 1
        if wta_locked:
            self._lock_count += 1

    def maybe_commit_epoch(
        self,
        degradation: "IrreversibleDegradationTracker",
        cascade_monitor: "CascadeFailureMonitor",
        attractor_dom: object,
        passivity_drift: "PassivityDriftEngine",
    ) -> Optional[str]:
        """At epoch boundary, apply permanent parameter shifts."""
        if self._tick_in_epoch < self._EPOCH_TICKS:
            return None

        self._epoch_count += 1
        _msgs: List[str] = []

        if self._over_count > self._EPOCH_TICKS * 0.30:
            degradation._FATIGUE_RATE = min(0.002, degradation._FATIGUE_RATE * 1.06)
            self.fatigue_rate_mult *= 1.06
            _msgs.append(f"D1:fatigue_rate×{self.fatigue_rate_mult:.2f}")

        if self._poverty_count > self._EPOCH_TICKS * 0.20:
            passivity_drift.energy_drift_mult = min(
                passivity_drift._MULT_CAP, passivity_drift.energy_drift_mult * 1.04
            )
            self.energy_drift_mult *= 1.04
            _msgs.append(f"D2:drift×{self.energy_drift_mult:.2f}")

        if self._cascade_count > 0:
            cascade_monitor._COOLDOWN = max(20, cascade_monitor._COOLDOWN - 3)
            self.cascade_cooldown_red += 3
            _msgs.append(f"D3:cascade_cd={cascade_monitor._COOLDOWN}")

        if self._lock_count > self._EPOCH_TICKS * self._LOCK_FRAC:
            _delta = 0.008
            attractor_dom._WTA_THRESHOLD = max(
                0.35, attractor_dom._WTA_THRESHOLD - _delta
            )
            self.wta_threshold_red += _delta
            _msgs.append(
                f"D4:wta={attractor_dom._WTA_THRESHOLD:.3f}"
                f"(-{self.wta_threshold_red:.3f})"
            )

        self._tick_in_epoch = self._over_count = self._poverty_count = 0
        self._cascade_count = self._lock_count = 0

        if _msgs:
            _log = f"[PATH-DEP epoch={self._epoch_count}] {'; '.join(_msgs)}"
            self.epoch_log.append(_log)
            if len(self.epoch_log) > 40:
                self.epoch_log.pop(0)
            return _log
        return None


# ─────────────────────────────────────────────────────────────────────────────
class LatentWorldDynamics:
    """8-dimensional hidden state model of the world, updated each tick.

    The system's sensory stream is treated as a noisy linear projection of an
    8-dimensional latent state.  The latent state evolves with nonlinear
    (Lorenz-inspired) mixing, process noise, and action-induced perturbations.

    This creates an irreversible causal structure: the system's past actions
    genuinely change the future trajectory of the latent state.  Surprising
    observations (high prediction error) are detected as the Mahalanobis
    distance between predicted and observed projections.

    Output:
      obs_surprise  — scalar [0, ∞) passed to brain as extra PE
      _state        — 8D float vector (internal, for self-model access)
    """

    _NDIM = 8
    _PROC_NOISE = 0.008  # per-tick process noise sigma
    _OBS_NOISE = 0.03  # observation noise sigma
    _ACTION_DELAY = 2  # ticks before action perturbation reaches latent state
    _MIX_ALPHA = 0.15  # nonlinear mixing strength

    def __init__(self) -> None:
        import numpy as _np

        self._state: "_np.ndarray" = _np.random.normal(0.5, 0.1, self._NDIM).clip(
            0.0, 1.0
        )
        self._C: "_np.ndarray" = (
            _np.random.randn(4, self._NDIM) * 0.25
        )  # obs projection
        self._action_queue: list = []  # [[remaining_delay, perturbation_vector], ...]
        self._risky_queue: list = []  # [ticks_until_integrity_damage, ...]
        self.obs_surprise: float = 0.0
        # Pending delayed integrity damage from risky actions
        self.pending_integrity_dmg: float = 0.0

    def _nonlinear_step(self, s: object) -> object:
        """Lorenz-like mixing: s[i] ← s[i] + α × (s[(i+1)%N] - s[i]) × s[(i+2)%N]"""
        import numpy as _np

        ds = _np.zeros(self._NDIM)
        for i in range(self._NDIM):
            ds[i] = (
                self._MIX_ALPHA
                * (s[(i + 1) % self._NDIM] - s[i])
                * s[(i + 2) % self._NDIM]
            )
        return (s + ds).clip(0.0, 1.0)

    # Actions whose latent consequences carry elevated risk of hidden-state damage
    _RISKY_ACTIONS = frozenset(
        {
            "explore",
            "execute",
            "interact",
            "reach",
            "grab",
            "speak",
            "approach",
            "search",
            "activate",
        }
    )

    def push_action(self, action_str: str) -> None:
        """Enqueue an action-induced perturbation with a fixed causal delay.

        Risky actions (matching _RISKY_ACTIONS keywords) produce 1.8× larger
        latent perturbations — their consequences are more unpredictable and
        more damaging when the hidden state diverges from observation.
        The damage appears as obs_surprise spikes 2 ticks later.
        """
        import hashlib as _hl

        import numpy as _np

        seed = int(_hl.md5(action_str.encode()).hexdigest()[:8], 16) % (2**31)
        rng = _np.random.default_rng(seed)
        _risky = any(k in action_str.lower() for k in self._RISKY_ACTIONS)
        _scale = 1.85 if _risky else 1.0
        pert = rng.normal(0, 0.07 * _scale, self._NDIM).clip(
            -0.15 * _scale, 0.15 * _scale
        )
        self._action_queue.append([self._ACTION_DELAY, pert])
        # Tag risky actions for delayed integrity damage tracking
        if _risky:
            self._risky_queue.append(self._ACTION_DELAY + 1)

    def step(self, sensory_activity: float) -> float:
        """Advance latent state one tick.  Returns obs_surprise in [0, 1]."""
        import numpy as _np

        # Apply pending action perturbations
        remaining = []
        for entry in self._action_queue:
            entry[0] -= 1
            if entry[0] <= 0:
                self._state = (self._state + entry[1]).clip(0.0, 1.0)
            else:
                remaining.append(entry)
        self._action_queue = remaining

        # Evolve: nonlinear mixing + process noise
        self._state = self._nonlinear_step(self._state)
        self._state += _np.random.normal(0, self._PROC_NOISE, self._NDIM)
        self._state = self._state.clip(0.0, 1.0)

        # Predicted scalar observation vs actual
        _obs_pred = float(self._C.dot(self._state).mean()) * 0.5 + 0.5
        _obs_pred = max(0.0, min(1.0, _obs_pred))
        _obs_actual = max(0.0, min(1.0, float(sensory_activity)))

        _err = abs(_obs_actual - _obs_pred)
        self.obs_surprise = min(1.0, _err / (self._OBS_NOISE * 6.0 + 1e-9))

        # Tick down risky-action damage timers
        _remaining_rq = []
        for _rt in self._risky_queue:
            if _rt <= 1:
                # Timer expired: accumulate delayed integrity damage
                # Damage is proportional to how surprising the observation was
                self.pending_integrity_dmg += 0.008 * (1.0 + self.obs_surprise * 2.0)
            else:
                _remaining_rq.append(_rt - 1)
        self._risky_queue = _remaining_rq

        return self.obs_surprise


# ─────────────────────────────────────────────────────────────────────────────
# EdgeOfChaosController — Critical-State Detection + Amplification (Phase 5)
# ─────────────────────────────────────────────────────────────────────────────


class EdgeOfChaosController:
    """Monitors proximity to phase transition and amplifies perturbations there.

    Computes a Lyapunov proxy from the coefficient of variation (CV) of each
    region's activity over a rolling window:
      lyapunov_proxy = mean(σ_i / μ_i)   for all tracked regions

    Regime classification:
      lyapunov_proxy < EDGE_LOW   → "stable"  (sub-critical, too ordered)
      EDGE_LOW ≤ proxy ≤ EDGE_HIGH → "edge"   (critical zone — max complexity)
      lyapunov_proxy > EDGE_HIGH  → "chaotic" (super-critical, cascade-prone)

    In the edge zone: small perturbations are amplified by injecting mixed
    excitatory/inhibitory currents proportional to edge_proximity.  This
    makes the system MORE sensitive near criticality, not less — small
    fluctuations can flip it between order and collapse.

    Permanent parameter shifts (irreversible):
      After SUPER_CRIT_HOLD ticks in chaotic regime:
        → CohesionEngine._BIND_CURR × 0.96  (order becomes harder to build)
      After SUB_CRIT_HOLD ticks in stable regime:
        → DisintegrationField._BASE_NOISE_RATE × 1.02
        (prolonged order accelerates latent decay — entropy builds under calm)

    Outputs:
      lyapunov_proxy  float [0,∞)
      edge_proximity  float [0,1] — 1 = deepest in edge zone
      regime          str         — "stable" | "edge" | "chaotic"
      amplify_current float       — nA of perturbation injected this tick
    """

    _WINDOW = 25
    _EDGE_LOW = 0.28
    _EDGE_HIGH = 0.68
    _AMPLIFY_SCALE = 3.5
    _SUPER_CRIT_HOLD = 40
    _SUB_CRIT_HOLD = 80

    def __init__(self) -> None:
        self._act_history: Dict[str, List[float]] = {}
        self.lyapunov_proxy: float = 0.0
        self.edge_proximity: float = 0.0
        self.regime: str = "stable"
        self._super_crit_ticks: int = 0
        self._sub_crit_ticks: int = 0
        self.amplify_current: float = 0.0
        self._total_chaotic_ep: int = 0

    def update(
        self, regions: List, cohesion_eng: object, disint_field: object
    ) -> List[str]:
        msgs: List[str] = []
        import random as _r_ec

        # Record activity per region
        for r in regions:
            _nm = type(r).__name__
            _act = r.activity() if hasattr(r, "activity") else 0.0
            if _nm not in self._act_history:
                self._act_history[_nm] = []
            self._act_history[_nm].append(_act)
            if len(self._act_history[_nm]) > self._WINDOW:
                self._act_history[_nm].pop(0)

        # Lyapunov proxy: mean CV across regions
        _cvs: List[float] = []
        for _nm, hist in self._act_history.items():
            if len(hist) < 5:
                continue
            _mn = sum(hist) / len(hist) + 1e-9
            _sd = (sum((h - _mn) ** 2 for h in hist) / len(hist)) ** 0.5
            _cvs.append(min(2.0, _sd / _mn))
        if _cvs:
            self.lyapunov_proxy = sum(_cvs) / len(_cvs)

        # Classify regime
        if self.lyapunov_proxy < self._EDGE_LOW:
            self.regime = "stable"
            self.edge_proximity = self.lyapunov_proxy / max(1e-9, self._EDGE_LOW)
            self._super_crit_ticks = 0
            self._sub_crit_ticks += 1
        elif self.lyapunov_proxy <= self._EDGE_HIGH:
            self.regime = "edge"
            _mid = (self._EDGE_LOW + self._EDGE_HIGH) / 2.0
            _half = (self._EDGE_HIGH - self._EDGE_LOW) / 2.0
            self.edge_proximity = max(
                0.0, 1.0 - abs(self.lyapunov_proxy - _mid) / _half
            )
            self._super_crit_ticks = 0
            self._sub_crit_ticks = 0
        else:
            self.regime = "chaotic"
            self.edge_proximity = 0.0
            self._super_crit_ticks += 1
            self._sub_crit_ticks = 0

        # Amplify perturbations in edge zone
        if self.regime == "edge" and self.edge_proximity > 0.25:
            self.amplify_current = self._AMPLIFY_SCALE * self.edge_proximity
            for r in regions:
                _ne = getattr(r, "_n_exc", 20)
                _inj = [
                    self.amplify_current * _r_ec.gauss(0, 0.8)
                    for _ in range(min(_ne, 20))
                ]
                r.inject(_inj)
        else:
            self.amplify_current = 0.0

        # Permanent parameter drift after sustained extreme regimes
        if self._super_crit_ticks >= self._SUPER_CRIT_HOLD:
            self._super_crit_ticks = 0
            self._total_chaotic_ep += 1
            if hasattr(cohesion_eng, "_BIND_CURR"):
                cohesion_eng._BIND_CURR = max(0.8, cohesion_eng._BIND_CURR * 0.96)
            msgs.append(
                f"[EDGE-CHAOS chaotic_ep={self._total_chaotic_ep}] "
                f"bind_curr→{getattr(cohesion_eng,'_BIND_CURR',0):.2f} "
                f"(order harder to establish)"
            )

        if self._sub_crit_ticks >= self._SUB_CRIT_HOLD:
            self._sub_crit_ticks = 0
            if hasattr(disint_field, "_BASE_NOISE_RATE"):
                disint_field._BASE_NOISE_RATE = min(
                    0.00025, disint_field._BASE_NOISE_RATE * 1.02
                )
            msgs.append(
                f"[EDGE-CHAOS sub-crit] noise_rate→"
                f"{getattr(disint_field,'_BASE_NOISE_RATE',0):.6f} "
                f"(stability breeds latent decay)"
            )

        return msgs


class Brain:
    """
    Main simulation loop.
    Call brain.start() to run in a background thread.
    brain.last_action gives the most recent decoded action.
    brain.speech_in gives the last recognised speech string.
    """

    def __init__(
        self,
        camera_index: int = 0,
        use_camera: bool = True,
        use_microphone: bool = True,
        use_web: bool = True,
        web_fetch_interval: float = 15.0,
    ) -> None:
        # Reset class-level ID counters so neuron/synapse nids are identical
        # between sessions — this makes persistence work deterministically.
        Neuron._id_counter = 0
        Synapse._id_counter = 0
        # Tonic background drive: brings all neurons near (but below) threshold
        # so that synaptic input can reliably trigger propagation.
        Neuron.global_tonic_current = TONIC_CURRENT

        # ── Sensors ──────────────────────────────────────────
        self._vis_enc = VisualEncoder(n_neurons=64, camera_index=camera_index)
        self._vis_analyzer = VisionAnalyzer()  # object/person/gesture detection
        self._latest_raw_frame = None  # cached for GUI live-view
        self._aud_enc = AudioEncoder(n_neurons=32)
        self._speech = SpeechListener(language="de-DE")
        self._web_enc = WebSensor(fetch_interval_s=web_fetch_interval, n_neurons=WEB_N)
        self._use_camera = use_camera
        self._use_mic = use_microphone
        self._use_web = use_web

        # ── Regions ──────────────────────────────────────────
        self.sensory_v = SensoryInputRegion("sensory_visual", n_inputs=64)
        self.sensory_a = SensoryInputRegion("sensory_auditory", n_inputs=32)
        self.sensory_w = SensoryInputRegion("sensory_web", n_inputs=WEB_N)
        self.thalamus = Thalamus()
        self.visual = PrimaryVisualCortex()
        self.auditory = PrimaryAuditoryCortex()
        self.assoc = AssociationCortex()
        self.hippocampus = Hippocampus()
        self.amygdala = Amygdala()
        self.prefrontal = PrefrontalCortex()
        self.motor = MotorCortex()

        self._all_regions: List[Region] = [
            self.sensory_v,
            self.sensory_a,
            self.sensory_w,
            self.thalamus,
            self.visual,
            self.auditory,
            self.assoc,
            self.hippocampus,
            self.amygdala,
            self.prefrontal,
            self.motor,
        ]

        # ── Inter-regional connections ────────────────────────
        self._inter_synapses: List[Synapse] = []
        self._wire_regions()

        # ── State ────────────────────────────────────────────
        self.t: float = 0.0  # simulation time ms
        self.tick_count: int = 0
        self.last_action: str = "idle"  # semantic motor decode (intention)
        self.last_executed_action: str = "idle"  # actually executed by executive/robot
        self.last_robot_command: str = "idle"
        self.speech_in: str = ""
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        # Diagnostics
        self.region_activity: Dict[str, float] = {}
        self._synapses_formed: int = 0  # new synapses formed this session
        self._synapses_restored: int = 0  # synapses loaded from persistence
        self._last_save_n: int = 0  # synapse count at last save

        # ── Emotion and Consciousness ────────────────────────
        self._emotion_engine = EmotionEngine()
        self._consciousness = ConsciousnessCore()
        self.emotion_state = EmotionalState()
        self.consciousness_state = ConsciousnessState()

        # ── Latent world dynamics — causal hidden state model ────────────
        self._latent_world: LatentWorldDynamics = LatentWorldDynamics()
        # Mean prediction error from region-level predictive coding
        self._mean_pred_error: float = 0.0

        # ── Local autonomous processes — bypass consciousness ────────────
        # Process 1 (metabolic):  every 3 ticks — direct energy/tonic adjust
        # Process 2 (defensive):  any tick when mean_pred_error > 0.35
        self._lap_metabolic_tick: int = 0  # tick counter for metabolic process
        self._lap_defensive_on: bool = False

        # ── Irreversible degradation tracker — permanent substrate damage ─
        self._degradation: IrreversibleDegradationTracker = (
            IrreversibleDegradationTracker()
        )

        # ── Cascade failure monitor — multi-system coupling breakdown ─────
        self._cascade: CascadeFailureMonitor = CascadeFailureMonitor()

        # ── Passivity drift engine — action imperative (Phase 4) ──────────
        self._passivity_drift: PassivityDriftEngine = PassivityDriftEngine()

        # ── Path dependency matrix — irreversible history (Phase 4) ───────
        self._path_dep: PathDependencyMatrix = PathDependencyMatrix()

        # ── Edge-of-chaos controller — critical-state amplification (Phase 5) ─
        self._edge_chaos: EdgeOfChaosController = EdgeOfChaosController()

        # ── Sleep / Wake state ───────────────────────────────
        self.sleeping: bool = False  # True during sleep phase
        self._sleep_entered: int = 0  # tick when sleep began
        # Scratch attribute read by emotion.py's predictive-error hook
        self._consciousness_pred_error: float = 0.0
        # Phase 5: fragile identity segments from ContinuityMonitor
        self._continuity_fragile: List[str] = []
        # ── Outbound communication (AI self-initiated messages) ───────
        self.outbound_messages: List[str] = []

        # -- Direct-reply queue (thread-safe) ---------------------------------
        # GUI thread posts user_text via request_reply(); tick thread drains
        # it inside _loop() by calling respond_to() -- no tick-thread race.
        self._reply_requests: queue.Queue = queue.Queue()
        self._reply_results: collections.deque = collections.deque(maxlen=10_000)

        # -- Async autosave state -------------------------------------------
        # save_brain() can take seconds with millions of synapses; run it in
        # a daemon thread so the tick loop is never blocked.
        self._save_in_progress: bool = False
        self._save_lock = threading.Lock()

        # ── Action toolbelt (AI-driven PC control) ───────────
        self._actions = ActionToolbelt()
        self._actions._brain = self  # perception-action feedback loop
        self._robot_controller = RobotController()
        self._gaze_dynamics = None  # initialised lazily after imports settle

        # ── Session telemetry ────────────────────────────────
        self._session_metrics = SessionMetrics()
        self._robot_serial = ArduinoSerialLink()
        self.action_log: List[dict] = []  # completed action records

        env_port = os.environ.get("AI_ROBOT_COM_PORT", "").strip()
        env_baud = int(os.environ.get("AI_ROBOT_BAUD", "115200") or "115200")
        if env_port:
            self._robot_serial.connect(env_port, env_baud)

        # ── Humanoid Stack (Schicht A–F) ──────────────────────
        self._body_schema = BodySchema()
        self._telemetry_bus = TelemetryBus()
        self._safety = SafetySupervisor()
        self._world_state = WorldState()
        self._skill_library = SkillLibrary()
        self._task_executive = TaskExecutive(self._skill_library)
        # Wire capability model from consciousness into executive planner
        self._task_executive.set_capability_model(
            self._consciousness.self_model.capabilities.confidence
        )
        # Wire causal graph for goal outcome learning
        self._task_executive._causal_graph = self._consciousness.causal_graph
        self._social_manager = SocialManager()
        self._dialogue_manager = DialogueManager()
        self._speech_output = SpeechOutput()
        self._consciousness_testbed = ConsciousnessTestbed()
        self._last_consciousness_test_summary: str = ""

        # Wire jaw-sync callbacks: TTS start/stop → social_manager + robot
        def _on_speech_start():
            self._social_manager.robot_started_speaking(self.tick_count)
            self._robot_controller.jaw_open()

        def _on_speech_end():
            self._social_manager.robot_stopped_speaking(self.tick_count)
            self._robot_controller.jaw_close()

        self._speech_output.on_start = _on_speech_start
        self._speech_output.on_end = _on_speech_end

        # Wire motor cue callback: UtterancePlan motor cues → RobotController
        def _on_motor_cue(cue_type: str, params: dict) -> None:
            if cue_type == "head_nod":
                self._robot_controller.nod_head()
            elif cue_type == "gaze_at_person":
                self._robot_controller.gaze_at_person()

        self._speech_output._on_motor_cue = _on_motor_cue
        _sim_mode = BackendMode.SIMULATED if not env_port else BackendMode.REAL
        self._body_interface = BodyInterface(
            mode=_sim_mode,
            body_schema=self._body_schema,
            serial_link=self._robot_serial,
            safety=self._safety,
        )
        # Subscribe world_state to telemetry bus events
        from telemetry_bus import (
            EVENT_FACE_DETECTED,
            EVENT_GESTURE_DETECTED,
            EVENT_OBJECT_DETECTED,
            EVENT_PERSON_LOST,
            EVENT_PERSON_SEEN,
            EVENT_SPEAKER_ACTIVE,
        )

        for evt_kind in (
            EVENT_PERSON_SEEN,
            EVENT_PERSON_LOST,
            EVENT_OBJECT_DETECTED,
            EVENT_GESTURE_DETECTED,
            EVENT_FACE_DETECTED,
            EVENT_SPEAKER_ACTIVE,
        ):
            self._telemetry_bus.subscribe(evt_kind, self._world_state.process_event)

    def _distance_scaled_synapse(
        self,
        src_n: Neuron,
        tgt_n: Neuron,
        *,
        base_weight: float,
        delay_range: tuple[float, float] = (0.5, 3.0),
        distance_scale: float = SPATIAL_LOCAL_RADIUS,
    ) -> Synapse:
        dist = src_n.distance_to(tgt_n)
        weight = _spatial_weight(dist, base_weight, distance_scale)
        delay = _spatial_delay(dist, delay_range, distance_scale)
        return Synapse(src_n, tgt_n, weight=weight, delay=delay)

    def _local_spatial_candidates(
        self,
        src_n: Neuron,
        pool: List[Neuron],
        *,
        already_ids: set[int],
        max_distance: float,
        limit: int = 32,
    ) -> List[Neuron]:
        n = len(pool)
        if n == 0:
            return []

        # ── NumPy vectorised distance (replaces O(N) Python loop + sorted) ──
        # Use pre-cached position arrays when pool is a region's _exc_cache.
        # For small/dynamic pools, build the array inline (still much faster
        # than per-element Python function calls + sorted()).
        _cache_key = id(pool)
        _pos_arr: np.ndarray
        _nid_arr: np.ndarray
        _entry = getattr(self, "_lsc_cache", {}).get(_cache_key)
        if _entry is not None and _entry[0] == n:  # valid if pool size unchanged
            _, _pos_arr, _nid_arr = _entry
        else:
            # Check if pool is a region's _exc_cache — use its pre-built arrays
            _reg_hit = False
            for _r in self._all_regions:
                if pool is _r._exc_cache:
                    _pos_arr = _r._exc_pos_arr
                    _nid_arr = _r._exc_nid_arr
                    _reg_hit = True
                    break
            if not _reg_hit:
                _pos_arr = np.empty((n, 3), dtype=np.float32)
                for _i, _p in enumerate(pool):
                    _pos_arr[_i, 0] = _p.x
                    _pos_arr[_i, 1] = _p.y
                    _pos_arr[_i, 2] = _p.z
                _nid_arr = np.array([_p.nid for _p in pool], dtype=np.int64)
            _cache = getattr(self, "_lsc_cache", {})
            _cache[_cache_key] = (n, _pos_arr, _nid_arr)
            self._lsc_cache = _cache

        _src = np.array([src_n.x, src_n.y, src_n.z], dtype=np.float32)
        _diffs = _pos_arr - _src
        _dists = np.sqrt((_diffs * _diffs).sum(axis=1))

        # Mask self and already-connected targets
        _valid = np.ones(n, dtype=bool)
        _valid[_nid_arr == src_n.nid] = False
        if already_ids:
            _valid &= ~np.isin(
                _nid_arr,
                np.fromiter(already_ids, dtype=np.int64, count=len(already_ids)),
            )
        _dists[~_valid] = np.inf

        # argpartition: O(N) vs sorted O(N log N) — huge win for large N
        _k = min(limit, n)
        _finite = int(np.sum(_valid))
        if _finite == 0:
            return []
        _k = min(_k, _finite)
        _part = np.argpartition(_dists, _k - 1)[:_k]
        _part = _part[np.argsort(_dists[_part])]

        local = [pool[_i] for _i in _part if _dists[_i] <= max_distance]
        if local:
            return local
        return [pool[_i] for _i in _part if _dists[_i] != np.inf]

    def spatial_layout_snapshot(self) -> Dict[str, object]:
        regions: List[Dict[str, object]] = []
        for region in self._all_regions:
            centroid = region.centroid()
            exc = region._exc_cache[:40]
            sample_points = [
                {"x": round(n.x, 3), "y": round(n.y, 3), "z": round(n.z, 3)}
                for n in exc
            ]
            regions.append(
                {
                    "name": region.name,
                    "centroid": tuple(round(v, 3) for v in centroid),
                    "origin": tuple(round(v, 3) for v in region.origin),
                    "extent": tuple(round(v, 3) for v in region.extent),
                    "sample_points": sample_points,
                    "n_neurons": len(region.neurons),
                }
            )
        return {"regions": regions}

    # ─────────────────────────────────────────────────────────
    # Wiring
    # ─────────────────────────────────────────────────────────

    def _wire_regions(self) -> None:
        c = self._inter_synapses

        # Sensory → Thalamus
        c += self.sensory_v.connect_to(self.thalamus, p=0.20, w_mean=2.0)
        c += self.sensory_a.connect_to(self.thalamus, p=0.20, w_mean=2.0)

        # Web/semantic pathway: routes through thalamus like all sensory streams.
        # All information must pass through the same perceptual bottleneck —
        # no modality has privileged direct access to higher association areas.
        c += self.sensory_w.connect_to(self.thalamus, p=0.20, w_mean=2.0)

        # Thalamus → Primary cortices  (thalamus 6x larger → p/6)
        c += self.thalamus.connect_to(self.visual, p=0.025, w_mean=2.0)
        c += self.thalamus.connect_to(self.auditory, p=0.025, w_mean=2.0)

        # Primary cortices → Association  (visual 6.7x, auditory 6.25x larger → p/~6)
        c += self.visual.connect_to(self.assoc, p=0.015, w_mean=1.8)
        c += self.auditory.connect_to(self.assoc, p=0.016, w_mean=1.8)

        # Association → Hippocampus + Amygdala  (assoc 6.7x larger → p/6.7)
        c += self.assoc.connect_to(self.hippocampus, p=0.012, w_mean=1.5)
        c += self.assoc.connect_to(self.amygdala, p=0.012, w_mean=1.5)

        # Hippocampus + Amygdala → Prefrontal  (hippo 7.5x, amygdala 6.7x larger)
        c += self.hippocampus.connect_to(self.prefrontal, p=0.011, w_mean=1.5)
        c += self.amygdala.connect_to(self.prefrontal, p=0.015, w_mean=1.8)

        # Prefrontal → Motor  (pfc 7.1x larger → p/7.1)
        c += self.prefrontal.connect_to(self.motor, p=0.017, w_mean=1.8)

        # Top-down feedback: Prefrontal → Association (attention / context)
        c += _connect_random(
            self.prefrontal.excitatory,
            self.assoc.excitatory,
            p=0.007,
            w_mean=1.0,
            delay_range=(5.0, 15.0),
        )

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Load persistence, start sensors, and launch the background tick thread."""
        import threading

        from persistence import load_brain

        self._synapses_restored = load_brain(self)
        if self._use_camera:
            self._vis_enc.start()
            self._vis_analyzer.start()
        if self._use_mic:
            self._aud_enc.start()
            self._speech.start()
        if self._use_web:
            self._web_enc.start()
        _tts_ok = self._speech_output.start()
        try:
            if _tts_ok:
                self._consciousness.stream.append(
                    f"[TTS] {self._speech_output.startup_status()}"
                )
            else:
                self._consciousness.stream.append(
                    f"[TTS] {self._speech_output.startup_status()}"
                )
        except Exception:
            pass
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def start_headless(self) -> None:
        """Start sensors and load persistence WITHOUT launching the tick thread.

        Use this when you want to call brain._tick() manually from a single
        thread (e.g. a chat REPL).  Avoids race conditions between the tick
        thread and external callers.
        """
        from persistence import load_brain

        self._synapses_restored = load_brain(self)
        if self._use_camera:
            self._vis_enc.start()
            self._vis_analyzer.start()
        if self._use_mic:
            self._aud_enc.start()
            self._speech.start()
        if self._use_web:
            self._web_enc.start()
        _tts_ok = self._speech_output.start()
        try:
            if _tts_ok:
                self._consciousness.stream.append(
                    f"[TTS] {self._speech_output.startup_status()}"
                )
            else:
                self._consciousness.stream.append(
                    f"[TTS] {self._speech_output.startup_status()}"
                )
        except Exception:
            pass
        self._running = True  # needed for autosave + sensor checks

    def stop_headless(self) -> None:
        """Stop sensors and save state (headless-mode counterpart of start_headless)."""
        self._running = False
        self._speech_output.stop()
        from persistence import save_brain

        save_brain(self)
        # Export session telemetry
        if self._session_metrics.n_samples > 0:
            try:
                self._session_metrics.export_jsonl()
            except Exception:
                pass
        if self._use_camera:
            self._vis_enc.stop()
            self._vis_analyzer.stop()
        if self._use_mic:
            self._aud_enc.stop()
            self._speech.stop()
        if self._use_web:
            self._web_enc.stop()

    def stop(self) -> None:
        self._running = False
        self._speech_output.stop()
        if hasattr(self, "_thread") and self._thread is not None:
            self._thread.join(timeout=5.0)
        from persistence import save_brain

        save_brain(self)
        # Export session telemetry
        if self._session_metrics.n_samples > 0:
            try:
                self._session_metrics.export_jsonl()
            except Exception:
                pass
        if self._use_camera:
            self._vis_enc.stop()
            self._vis_analyzer.stop()
        if self._use_mic:
            self._aud_enc.stop()
            self._speech.stop()
        if self._use_web:
            self._web_enc.stop()

    # ─────────────────────────────────────────────────────────
    # Main simulation loop
    # ─────────────────────────────────────────────────────────

    def request_reply(self, user_text: str) -> None:
        """Thread-safe: queue a user message so respond_to() runs in the tick thread.

        Immediately primes drives (notify_direct_address) so that ticks between
        now and when respond_to() starts are already oriented toward responding.
        The actual respond_to() call happens inside _loop() on the tick thread,
        and the result is appended to self._reply_results for the GUI to collect.
        """
        self._consciousness.notify_direct_address(user_text, self)
        self._reply_requests.put(user_text)

    def _loop(self) -> None:
        while self._running:
            # Serve at most ONE pending reply request per tick so that
            # deliberation doesn't monopolise the loop. Backpressure:
            # if the consciousness deliberation queue is full, skip
            # accepting new requests until space frees up.
            try:
                _delib_q = getattr(self._consciousness, "_deliberation_queue", None)
                _can_accept = _delib_q is None or len(_delib_q) < 5
                if _can_accept:
                    user_text = self._reply_requests.get_nowait()
                    # Register incoming turn in dialogue manager
                    _speaker = self._social_manager.primary_interlocutor() or "user"
                    self._dialogue_manager.process_incoming(
                        raw_text=user_text,
                        speaker=_speaker,
                        tick=self.tick_count,
                        asr_confidence=0.9,
                    )
                    # ── Feed utterance outcomes back into ToM learning ──
                    _utt_outcomes = self._dialogue_manager.pop_outcomes()
                    if _utt_outcomes and hasattr(self._consciousness, "theory_of_mind"):
                        for _pid_oc, _oc_type in _utt_outcomes.items():
                            self._consciousness.theory_of_mind.record_comm_outcome(
                                self.tick_count, _pid_oc, _oc_type
                            )
                            # Also update PersonModel trust from comm outcomes
                            try:
                                _pm_oc = self._social_manager.person_model(int(_pid_oc))
                                if _pm_oc is not None:
                                    if _oc_type == "understood":
                                        _pm_oc.trust = min(1.0, _pm_oc.trust + 0.005)
                                    elif _oc_type == "repair_requested":
                                        _pm_oc.trust = max(0.0, _pm_oc.trust - 0.01)
                                    elif _oc_type == "topic_shifted":
                                        _pm_oc.note_preference("topic_change", 0.05)
                                    elif _oc_type == "disengaged":
                                        _pm_oc.trust = max(0.0, _pm_oc.trust - 0.015)
                                        _pm_oc.note_preference("brevity", 0.1)
                                    elif _oc_type == "minimal_response":
                                        _pm_oc.note_preference("brevity", 0.08)
                            except (ValueError, TypeError):
                                pass
                    # ── Speech Act Planning BEFORE content generation ──
                    # Query ToM FIRST so plan_response can use the social model
                    _body_urg = getattr(self._consciousness.body, "error_risk", 0.0)
                    _comm_drv = self._consciousness.comm_drive_level
                    _tom_strat_utt = {}
                    if _speaker and hasattr(self._consciousness, "theory_of_mind"):
                        _tom_strat_utt = (
                            self._consciousness.theory_of_mind.recommend_strategy(
                                str(_speaker)
                            )
                        )
                    _planned_act = self._dialogue_manager.plan_response(
                        tick=self.tick_count,
                        last_incoming=self._dialogue_manager.last_incoming_turn,
                        comm_drive=_comm_drv,
                        body_urgency=_body_urg,
                        tom_strategy=_tom_strat_utt,
                        known_concepts=set(self._consciousness._concepts),
                    )
                    # Pass planned act into consciousness so it can
                    # steer content assembly (REPAIR → ask clarification,
                    # SILENCE → suppress, ASK → formulate question, etc.)
                    response, _ = self._consciousness.respond_to(
                        user_text, self, planned_speech_act=_planned_act
                    )
                    if response:
                        self._consciousness.human_interaction.observe_reply(
                            response, self._consciousness, self
                        )
                        # Route through dialogue pipeline → TTS (ToM already queried above)
                        _uplan = self._dialogue_manager.build_utterance(
                            response,
                            _speaker,
                            self._consciousness,
                            speech_act=_planned_act,
                            tom_strategy=_tom_strat_utt,
                        )
                        # ── Deliberation delay: natural thinking gap ──────
                        # Scales with query complexity and internal uncertainty.
                        # Implemented as leading silence in the TTS worker so
                        # the tick loop is never blocked.
                        try:
                            import random as _rnd
                            _unc = getattr(self._consciousness, "uncertainty_overall", 0.5)
                            _q_words = len(user_text.split())
                            # base: 200ms + up to 600ms for complex queries
                            # + up to 500ms when uncertain
                            _delay_ms = int(
                                200
                                + min(_q_words * 18, 600)
                                + _unc * 500
                                + _rnd.randint(-80, 120)
                            )
                            _delay_ms = max(150, min(1800, _delay_ms))
                            # Repairs / backchannels feel unnatural with long pauses
                            if _planned_act and _planned_act.value in ("backchannel", "repair"):
                                _delay_ms = min(_delay_ms, 350)
                            _uplan.deliberation_delay_ms = _delay_ms
                        except Exception:
                            pass

                        # ── Modules 19-20: Microbehavior + PresenceSynchronizer ──
                        # Use HumanInteractionSuite state to adjust UtterancePlan
                        # motor cues, timing and prosody.  All writes go through
                        # fields that UtterancePlan already exposes so no new
                        # infrastructure is needed.
                        try:
                            _hi_bp = self._consciousness.human_interaction
                            _mb = _hi_bp.microbehavior
                            _pres = _hi_bp.presence

                            # MicrobehaviorController → motor cues on UtterancePlan
                            # head_tilt_bias: if non-zero → trigger a nod when speaking
                            if _mb.head_tilt_bias > 0.0:
                                _uplan.head_nod = True
                            # gaze_micro_variance: high variance → less fixed gaze
                            _uplan.gaze_at_person = _mb.gaze_micro_variance < 0.40

                            # PresenceSynchronizer → timing + prosody
                            if _pres.timing_mode == "slow":
                                # Fatigue: widen deliberation gap, slow TTS
                                _uplan.deliberation_delay_ms = min(
                                    2400,
                                    _uplan.deliberation_delay_ms + 350,
                                )
                                _uplan.speed_factor = max(0.80, _uplan.speed_factor - 0.12)
                            elif _pres.timing_mode == "eager":
                                # High energy: trim gap, slightly faster
                                _uplan.deliberation_delay_ms = max(
                                    100,
                                    _uplan.deliberation_delay_ms - 120,
                                )
                                _uplan.speed_factor = min(1.15, _uplan.speed_factor + 0.08)

                            # sync_score → overall confidence on the plan
                            _uplan.confidence = max(0.05, min(1.0, _pres.sync_score))
                        except Exception:
                            pass  # presence cues are non-critical

                        self._speech_output.speak_utterance(_uplan)
                        self._dialogue_manager.mark_output_delivered(self.tick_count)
                        self._reply_results.appendleft(response)
            except queue.Empty:
                pass
            try:
                self._tick()
            except Exception as _exc:
                # Log the error to the thought stream instead of killing the loop
                try:
                    self._consciousness.stream.append(
                        f"[TICK-ERROR t={self.tick_count}] {_exc}"
                    )
                    self.tick_count += 1  # advance so we don't retry the same tick
                except Exception:
                    pass
            # REAL_TICK_MS = 0 → no throttle; run at full CPU speed
            if REAL_TICK_MS > 0:
                time.sleep(REAL_TICK_MS / 1000.0)

    def _tick(self) -> None:
        t = self.t
        _mean_act = 0.0

        # 1) Read sensors and inject.
        if self._use_camera:
            vis_currents = self._vis_enc.encode()  # non-blocking: returns latest cached
            with self._vis_enc._lock:
                raw_frame = self._vis_enc._latest_frame  # protected read
            self._latest_raw_frame = raw_frame

            # Submit frame to async worker every 3rd tick (non-blocking)
            if self.tick_count % 3 == 0 and raw_frame is not None:
                self._vis_analyzer.submit(raw_frame)

            detections = self._vis_analyzer.detections  # last completed result
            if detections:
                sal = self._vis_analyzer.detection_strength
                boosted = [min(c * (1.0 + sal), 30.0) for c in vis_currents]
                self.sensory_v.inject(boosted)
                self.inject_text_input(" ".join(detections))
                if self._use_web:
                    for det in detections[:2]:
                        base = det.split(":")[0]
                        if base not in ("hand", "face"):
                            self._web_enc.add_interest_topic(base.capitalize())
            else:
                self.sensory_v.inject(vis_currents)
        if self._use_mic:
            self.sensory_a.inject(self._aud_enc.encode())
            # ── Domain H: push prosodic affect into primary TrackedPerson ──
            try:
                _aff_pid = self._social_manager.primary_interlocutor()
                if _aff_pid is not None:
                    _aff_person = self._world_state.persons.get(_aff_pid)
                    if _aff_person is None:
                        try:
                            _aff_person = self._world_state.persons.get(int(_aff_pid))
                        except (ValueError, TypeError):
                            pass
                    if _aff_person is not None:
                        _aff_person.speech_energy = self._aud_enc.speech_energy_ema
                        _aff_person.speech_tempo_var = self._aud_enc.speech_tempo_var
                        if self._aud_enc.inferred_affect != "unknown":
                            _aff_person.speech_affect = self._aud_enc.inferred_affect
            except Exception:
                pass
        if self._use_web:
            raw_web = self._web_enc.encode()
            # Web signal enters via thalamus pathway — same as all sensory streams.
            # Amplify for spiking threshold; no privileged direct cortical injection.
            self.sensory_w.inject([min(c * 2.5, 30.0) for c in raw_web])
        # sensory_w needs no extra injection without web — tonic alone drives it

        # 1a) Sensor health — update per-channel availability tracking
        _sh = self._consciousness.self_model.sensor_health
        _tc = self.tick_count
        if self._use_camera:
            if self._vis_enc._running and self._vis_enc._latest_frame is not None:
                _sh.camera.mark_ok(_tc)
            else:
                _sh.camera.mark_fail(_tc)
        if self._use_mic:
            if self._aud_enc._running:
                _sh.mic.mark_ok(_tc)
            else:
                _sh.mic.mark_fail(_tc)
        if self._use_web:
            if self._web_enc._running:
                _sh.web.mark_ok(_tc)
            else:
                _sh.web.mark_fail(_tc)
        if self._speech._running:
            _sh.speech.mark_ok(_tc)
        else:
            _sh.speech.mark_fail(_tc)

        # 1b) Humanoid stack: push perception events to telemetry bus + world state
        if self._use_camera:
            _det_list = self._vis_analyzer.detections or []
            self._telemetry_bus.push_detection_events(
                _det_list, self.latest_detection_targets, self.tick_count
            )
        if self._speech.latest_text:
            self._telemetry_bus.push_speech_event(
                self._speech.latest_text, self.tick_count
            )
        _tframe = self._body_interface.build_telemetry_frame()
        if _tframe:
            self._telemetry_bus.push_frame(_tframe)
            self._body_schema.update_from_telemetry(_tframe)
        self._telemetry_bus.flush()
        self._world_state.tick(self.tick_count)

        # 2) Speech → dialogue manager → semantic pathway injection
        if self._speech.latest_text:
            self.speech_in = self._speech.latest_text
            self._speech.latest_text = ""
            # Register in dialogue manager for structured turn tracking
            _spk = self._social_manager.primary_interlocutor() or "unknown"
            self._dialogue_manager.process_incoming(
                raw_text=self.speech_in,
                speaker=_spk,
                tick=self.tick_count,
                asr_confidence=getattr(self._speech, "last_confidence", 0.8),
            )
            # Feed utterance outcomes back into ToM learning
            _spk_outcomes = self._dialogue_manager.pop_outcomes()
            if _spk_outcomes and hasattr(self._consciousness, "theory_of_mind"):
                for _pid_o, _oc_str in _spk_outcomes.items():
                    self._consciousness.theory_of_mind.record_comm_outcome(
                        self.tick_count, _pid_o, _oc_str
                    )
            self.inject_text_input(self.speech_in)
            # Redirect web curiosity toward spoken keywords (self-directed)
            if self._use_web:
                words = [
                    w for w in self.speech_in.split() if w.isalpha() and len(w) > 3
                ]
                if words:
                    topic = "_".join(w.capitalize() for w in words[:2])
                    self._web_enc.add_interest_topic(topic)

        # 3) Tick all regions sequentially.
        # Spike delivery is event-driven: each neuron pushes arrives spikes directly
        # into post-synaptic queues on fire, so no separate synapse.tick() pass needed.
        for r in self._all_regions:
            r.tick(t)

        # ── 3b. Predictive-coding feedback + metabolic accounting ────────────────
        # Call update_prediction() on every region to:
        #   a) Compute signed prediction error (actual − expected activity)
        #   b) Inject corrective feedback current (negative-feedback loop)
        #   c) Accumulate per-region _pe_ema for downstream use
        # Store mean |error| on self for InferentialSelfEstimator in consciousness.
        _mean_act = sum(r.activity() for r in self._all_regions) / max(
            1, len(self._all_regions)
        )
        _pe_sum = 0.0
        for _r in self._all_regions:
            _pe_sum += abs(_r.update_prediction())
        self._mean_pred_error = _pe_sum / max(1, len(self._all_regions))

        # ── 3b-2. Latent world dynamics ───────────────────────────────────────
        # Step the hidden-state world model; blend its observation surprise into
        # the mean_pred_error so consciousness gets a richer surprise signal.
        _sensory_act = (
            self.sensory_v.activity() * 0.5
            + self.sensory_a.activity() * 0.3
            + self.sensory_w.activity() * 0.2
        )
        _lwd_surprise = self._latent_world.step(_sensory_act)
        # Blend: 70% region-PE, 30% latent-world surprise
        self._mean_pred_error = 0.70 * self._mean_pred_error + 0.30 * _lwd_surprise
        # Push last action into latent world so its causal effects propagate
        self._latent_world.push_action(self.last_action)

        # ── 3b-3. Local autonomous processes (bypass consciousness) ───────────
        # Process 1 — Metabolic (every 3 ticks):
        #   Direct energy drain + tonic scaling WITHOUT going through tick().
        #   This ensures metabolic effects are always applied, even during
        #   high-cognitive-load ticks when consciousness.tick() is slow.
        self._lap_metabolic_tick += 1
        if self._lap_metabolic_tick >= 3:
            self._lap_metabolic_tick = 0
            _lap_body = self._consciousness.body
            _act_load = _mean_act  # computed earlier in 3b; defaulted at tick start
            # Drain slightly more than the tick-level 3c drain (complementary)
            _lap_drain = _act_load * 0.0008
            _lap_body.energy_reserve = max(
                0.05, min(1.0, _lap_body.energy_reserve - _lap_drain)
            )
            # Tonic scaled directly by energy — no consciousness involvement
            from neuron import Neuron as _Nlap

            _Nlap.global_tonic_current = max(
                SLEEP_TONIC,
                _Nlap.global_tonic_current * (0.97 + 0.03 * _lap_body.energy_reserve),
            )

        # Process 2 — Defensive reflex (any tick when mean_pred_error > 0.35):
        #   Inject fast inhibitory current into PFC to suppress runaway activity.
        #   This is a brainstem-level reflex — not a goal, not a thought.
        if self._mean_pred_error > 0.35:
            if not self._lap_defensive_on:
                self._lap_defensive_on = True
                _pfc_n = len(self.prefrontal.excitatory)
                if _pfc_n > 0:
                    # Strong inhibitory current: clamps 30% of PFC neurons
                    _n_inh = max(1, _pfc_n // 3)
                    self.prefrontal.inject([-6.0] * _n_inh + [0.0] * (_pfc_n - _n_inh))
        else:
            self._lap_defensive_on = False

        # ── 3c. Energy drain + thermal load from actual neural activity ────────
        # Mean activity across all regions → proportional energy drain each tick.
        # Thermal load accumulates and slowly dissipates (like processor heat).
        # This makes neural activity a genuine metabolic cost, not just a metric.
        # (_mean_act already computed in 3b above)
        _body_hw = self._consciousness.body
        _drain = _mean_act * 0.0025  # energy cost per unit activity
        _regen = 0.00015  # slow baseline regeneration
        _body_hw.energy_reserve = max(
            0.05, min(1.0, _body_hw.energy_reserve - _drain + _regen)
        )
        _body_hw.thermal_load = min(
            1.0, _body_hw.thermal_load * 0.992 + _mean_act * 0.015
        )
        # High thermal → force immediate tonic reduction this tick
        if _body_hw.thermal_load > 0.78:
            _thermal_penalty = (_body_hw.thermal_load - 0.78) * 2.0  # up to −0.44
            from neuron import Neuron as _Nhw

            _Nhw.global_tonic_current = max(
                SLEEP_TONIC, _Nhw.global_tonic_current * (1.0 - _thermal_penalty)
            )

        # ── 3d. Irreversible degradation — permanent substrate damage ─────────
        # Three channels: region fatigue (inject suppression), thermal scarring
        # (permanent tonic ceiling reduction), structural poverty (prune synapses).
        # None of these recover fully — past overload is permanently encoded.
        _degrad_msgs = self._degradation.update(
            self._all_regions, _body_hw, self._inter_synapses
        )
        for _dm in _degrad_msgs:
            self._consciousness.stream.append(_dm)
        # Apply permanent thermal scar: clamp global tonic at scar ceiling
        _tonic_ceil = self._degradation.effective_tonic_ceiling(WAKE_TONIC + 2.0)
        if Neuron.global_tonic_current > _tonic_ceil:
            Neuron.global_tonic_current = _tonic_ceil

        # ── 3e. Cascade failure detection — multi-system breakdown ────────────
        # If ≥3 signals simultaneously in failure range, cascade triggers:
        # amygdala burst + PFC/assoc suppression + direct energy + integrity drain.
        # Cascade itself moves system closer to next cascade → positive feedback.
        _casc_msg = self._cascade.update(
            energy=_body_hw.energy_reserve,
            integrity=_body_hw.integrity,
            thermal=_body_hw.thermal_load,
            mean_pe=self._mean_pred_error,
            body=_body_hw,
            regions=self._all_regions,
            amygdala=self.amygdala,
            prefrontal=self.prefrontal,
            assoc=self.assoc,
        )
        if _casc_msg:
            self._consciousness.stream.append(_casc_msg)

        # ── 3f. Latent world damage — delayed consequences of past actions ────
        # Surprising observations (obs_surprise > threshold) damage body state.
        # Risky actions accumulate pending integrity damage (applied here).
        if self._latent_world.obs_surprise > 0.45:
            _int_dmg = (self._latent_world.obs_surprise - 0.45) * 0.014
            _body_hw.integrity = max(0.05, _body_hw.integrity - _int_dmg)
        if self._latent_world.obs_surprise > 0.70:
            # Panic energy spike from severely unexpected observations
            _body_hw.energy_reserve = max(0.05, _body_hw.energy_reserve - 0.022)
        # Apply delayed integrity damage from risky actions
        if self._latent_world.pending_integrity_dmg > 0.001:
            _body_hw.integrity = max(
                0.05, _body_hw.integrity - self._latent_world.pending_integrity_dmg
            )
            self._latent_world.pending_integrity_dmg = 0.0

        # ── 3g. Passivity drift + path dependency (Phase 4) ─────────────────
        # Three autonomous drift channels penalise passivity regardless of
        # sensor state.  Only specific actions slow each channel.  Missed
        # intervention windows permanently increase drift multipliers —
        # behavioral history shapes the underlying physics.
        _pde_msgs = self._passivity_drift.update(
            action=self.last_action,
            body=_body_hw,
            latent_world=self._latent_world,
        )
        for _pm in _pde_msgs:
            self._consciousness.stream.append(_pm)

        # Path dependency epoch commit: every 200-tick boundary, evaluate
        # behavioral history and permanently shift system-wide dynamic parameters.
        self._path_dep.record_tick(
            mean_activity=_mean_act,
            energy=_body_hw.energy_reserve,
            cascade_hit=self._cascade.cascading,
            wta_locked=self._consciousness.attractor_dom.locked,
        )
        _pdep_msg = self._path_dep.maybe_commit_epoch(
            degradation=self._degradation,
            cascade_monitor=self._cascade,
            attractor_dom=self._consciousness.attractor_dom,
            passivity_drift=self._passivity_drift,
        )
        if _pdep_msg:
            self._consciousness.stream.append(_pdep_msg)

        # DeceptiveAttractorField: detect still/overdrive traps and apply
        # hidden costs — energy drain or thermal scar that appear only after
        # the system has been lured into the trap zone.
        _trap_msgs = self._consciousness.deceptive_attractors.update(
            mean_activity=_mean_act,
            body=_body_hw,
            degradation=self._degradation,
        )
        for _tm in _trap_msgs:
            self._consciousness.stream.append(_tm)

        # ── 3h. Edge-of-chaos controller (Phase 5) ───────────────────────────
        # Detects critical-state proximity from region activity divergence.
        # In the edge zone, perturbations are amplified → small inputs have
        # outsized effects, making the system operate near bifurcation.
        # Permanently weakens binding or raises noise after sustained extremes.
        _ec_msgs = self._edge_chaos.update(
            regions=self._all_regions,
            cohesion_eng=self._consciousness.cohesion,
            disint_field=self._consciousness.disintegration,
        )
        for _ecm in _ec_msgs:
            self._consciousness.stream.append(_ecm)

        # 5) Amygdala valence update
        # Use semantic appraisal when workspace has concepts; fallback to activity rate
        _ws_concepts = self._consciousness.workspace_concepts()

        # 5b) Hippocampal closed loop — translate fired neuron indices back to
        #     recalled concepts and re-inject into workspace.
        #
        #     HOW SYNAPSES BECOME USEFUL:
        #     STDP trains synapse weights to strengthen connections between neurons
        #     that co-fire. Each neuron at index k in hippocampus._exc_cache was
        #     "tuned" to concept C by semantic_encode() via hash(C) % n injection.
        #     After STDP, neurons for co-occurring concepts get strongly connected.
        #     When a cue concept activates neuron k, downstream concept neurons
        #     fire too — that firing IS memory recall. We detect those fires here
        #     and translate the index back to a concept (using _concept_at_index),
        #     then expand the workspace with the recalled content.
        #
        #     This closes the loop:
        #     Text → LIF spikes → STDP → strong hipp synapses → concept neurons fire
        #     → _concept_at_index lookup → recalled concepts enter workspace
        #     → enriched reasoning, grounded responses, emotion from real content.
        _cs = self._consciousness
        _hipp_exc = self.hippocampus._exc_cache
        _hipp_cai = self.hippocampus._concept_at_index
        if _hipp_cai and self.hippocampus.activity() > 0.04:
            _ws_set = set(_ws_concepts)
            # 5b-i: Fired neurons → recalled concept labels (use numpy fired mask)
            _hipp_fired = self.hippocampus._fired_this_tick
            _hipp_exc_pos = self.hippocampus._exc_positions
            _neural_recalled = [
                _hipp_cai[k]
                for k, pos in enumerate(_hipp_exc_pos)
                if _hipp_fired[pos] and k in _hipp_cai and _hipp_cai[k] not in _ws_set
            ][:5]
            # 5b-ii: Semantic memory recall using workspace as cue
            _mem_recalled = (
                [
                    c
                    for c in self.hippocampus.semantic_recall(_ws_concepts, top_n=6)
                    if c not in _ws_set
                ][:5]
                if _ws_concepts
                else []
            )
            # Merge: neural recall first (LIF-driven), then semantic recall
            _all_recalled = list(dict.fromkeys(_neural_recalled + _mem_recalled))
            if _all_recalled:
                # Re-inject recalled concepts into workspace via _extract_concepts
                _cs._extract_concepts(
                    " ".join(_all_recalled[:7]),
                    _cs._hipp_act_ema,
                    _cs._pfc_act_ema,
                )
                # Refresh workspace after recall injection
                _ws_concepts = _cs.workspace_concepts()

        # 5b-iii: Trace-weighted salience — substrate-anchored semantic signal.
        # Read the hippocampal spike-trace for concept-labeled neurons.
        # A neuron with a high trace fired recently and strongly; its concept
        # gets a direct salience bump proportional to firing strength.
        # This makes neural activity the *primary* driver of concept salience,
        # bypassing the text pipeline entirely for previously-learned concepts.
        _trace_readout = self.hippocampus.active_concept_readout(top_k=10)
        for _tc, _t_strength in _trace_readout:
            _cs._bump_salience(_tc, _t_strength * 0.20)

        if _ws_concepts:
            _pfc_goal = (
                self.prefrontal.active_goal or self.consciousness_state.goal or ""
            )
            self.amygdala.semantic_appraise(
                _ws_concepts, [_pfc_goal] if _pfc_goal else []
            )
        else:
            self.amygdala.update_valence(self.assoc.activity())

        # 6) Motor decode — integration SHAPES action, never gates it.
        # phi_surrogate is stored for continuous influence on downstream scoring.
        # When workspace concepts exist, semantic_decode uses the goal+emotion+phi
        # signal.  When concepts are absent (sensors inactive), no grounded action
        # can be selected — last_action degrades to empty, which IS the degraded state.
        _phi = self._consciousness.integration_probe.phi_surrogate()
        self._last_phi = _phi

        # 6a) World-dependency enforcement: track sensor-free ticks.
        # Without world coupling the system MUST degrade — phi falls naturally
        # as region activity decays without input.  Make this explicit.
        _has_world = self._use_camera or self._use_mic or self._use_web
        if _has_world:
            self._sensor_free_ticks = 0
            # Restore tonic drive when world input returns.
            # Apply energy scaling: low energy reduces wake tonic proportionally.
            # At energy=1.0 → full WAKE_TONIC; at energy=0.30 → 30 % of WAKE_TONIC.
            _eff_tonic_ws = WAKE_TONIC * max(
                0.30, self._consciousness.body.energy_reserve
            )
            Neuron.global_tonic_current = _eff_tonic_ws
        else:
            self._sensor_free_ticks = getattr(self, "_sensor_free_ticks", 0) + 1
        if self._sensor_free_ticks > 0 and self._sensor_free_ticks % 200 == 1:
            self._consciousness.stream.append(
                f"[WORLD-DEPRIVATION t={self.tick_count}] "
                f"all sensors disabled for {self._sensor_free_ticks} ticks — "
                f"phi={_phi:.4f}, workspace={len(_ws_concepts)} concepts. "
                f"Integration without world coupling is structurally degraded."
            )

        # ── Point 5: World deprivation → hallucination/reconstruction mode ──
        # Sensor absence increases internal noise and reconstruction activity
        # rather than collapsing the system.  The self-model gets noisier
        # (not smaller), tonic drifts only mildly, and sensory_w receives
        # small random afferent currents to model endogenous generation.
        if self._sensor_free_ticks > 50:
            _deprivation_pressure = min(1.0, self._sensor_free_ticks / 500.0)
            _sm = getattr(self._consciousness, "self_model", None)
            if _sm is not None:
                import random as _rnd_dep

                # Add gaussian noise: self-model stays present but uncertain
                _sm.agency_confidence = max(
                    0.05,
                    min(
                        0.95,
                        _sm.agency_confidence
                        + _rnd_dep.gauss(0, _deprivation_pressure * 0.015),
                    ),
                )
                _sm.continuity_estimate = max(
                    0.05,
                    min(
                        0.95,
                        _sm.continuity_estimate
                        + _rnd_dep.gauss(0, _deprivation_pressure * 0.012),
                    ),
                )
                _sm.uncertainty = min(
                    1.0, _sm.uncertainty + _deprivation_pressure * 0.003
                )
            # Mild tonic drift (sleep-like, not collapse): 0.25× not 0.6×
            _deprived_tonic = max(
                SLEEP_TONIC, WAKE_TONIC * (1.0 - _deprivation_pressure * 0.25)
            )
            Neuron.global_tonic_current = _deprived_tonic
            # Hallucination: inject small random afferent noise into sensory_w
            # to model endogenous/reconstructive generation, NOT zero-silence.
            import random as _rnd_hal

            _sw_n = len(self.sensory_w._exc_cache)
            _noise_amp = _deprivation_pressure * 3.0
            self.sensory_w.inject([_rnd_hal.gauss(0, _noise_amp) for _ in range(_sw_n)])
            # Boost coherence_need: system searches/reconstructs more actively
            _c_dep = self._consciousness
            if hasattr(_c_dep, "drives"):
                _c_dep.drives.coherence_need = min(
                    1.0, _c_dep.drives.coherence_need + _deprivation_pressure * 0.02
                )

        if _ws_concepts:
            self.last_action = self.motor.semantic_decode(
                goal=self.prefrontal.active_goal or self.consciousness_state.goal,
                em_stress=self.emotion_state.stress,
                em_curiosity=self.emotion_state.curiosity,
                em_fatigue=self.emotion_state.fatigue,
                em_arousal=self.emotion_state.arousal(),
                concepts=_ws_concepts,
            )
        else:
            # No grounded workspace content → motor cannot select a meaningful action.
            # This is correct degradation: world absence causes action collapse.
            self.last_action = ""

        # 7) Record activity
        for region in self._all_regions:
            self.region_activity[region.name] = region.activity()

        # 8) Emotion engine — derives emotional state from regional activity
        self.emotion_state = self._emotion_engine.tick(self)

        # 9) Update web sensor urgency from curiosity (more fetch when curious)
        if self._use_web:
            self._web_enc.set_urgency(self.emotion_state.fetch_urgency())

        # 10) Modulate STDP LTP rate with emotion (dopamine-like neuromodulation)
        # RPE (reward prediction error): positive surprise → stronger LTP,
        # negative surprise → weaker learning (matches neuromodulatory gating).
        _rpe = self.emotion_state.reward_prediction_error
        _rpe_boost = max(
            0.2, 1.0 + max(0.0, _rpe) * 6.0
        )  # only positive RPE boosts LTP
        _neuron_mod.A_PLUS = (
            BASE_A_PLUS * self.emotion_state.ltp_modulation() * _rpe_boost
        )

        # 11) Rapid Hebbian co-activation synapse formation (rate modulated by emotion)
        _growth_cap = max(
            10, int(COACTIVATION_PER_TICK * self.emotion_state.growth_rate())
        )
        self._coactivation_sprout(cap=_growth_cap)

        # 12) STDP-driven structural plasticity (periodic pruning + weight-based sprouting)
        if self.tick_count % PLASTICITY_INTERVAL == 0:
            self._structural_plasticity()

        # 13) Consciousness / reasoning / memory replay
        # GoalSystemFailure is now a last-resort fallback only — both main
        # raise paths in _evaluate_goal have been replaced with degraded-but-
        # running continuations.  If it still fires (e.g. via a future code
        # path), degrade quality but keep the motor active.
        try:
            self.consciousness_state = self._consciousness.tick(self)
        except GoalSystemFailure as _gsf:
            _degraded = ConsciousnessState()
            _degraded.goal = "explore"  # safe default — motor keeps running
            _degraded.prediction_error = 0.8
            _degraded.thought = f"[DEGRADED phi={self._last_phi:.4f}]"
            self.consciousness_state = _degraded
            # Noise injection: jitter self-model around current value.
            # Do NOT floor at 0 — system stays noisy but alive.
            _sm_fail = getattr(self._consciousness, "self_model", None)
            if _sm_fail is not None:
                import random as _rnd_gsf

                _sm_fail.agency_confidence = max(
                    0.05,
                    min(0.95, _sm_fail.agency_confidence + _rnd_gsf.gauss(0, 0.03)),
                )
                _sm_fail.continuity_estimate = max(
                    0.05,
                    min(0.95, _sm_fail.continuity_estimate + _rnd_gsf.gauss(0, 0.03)),
                )
                _sm_fail.uncertainty = min(0.95, _sm_fail.uncertainty + 0.01)
            self._consciousness.stream.append(
                f"[DEGRADED t={self.tick_count}] GoalSystemFailure fallback: {_gsf} "
                f"phi={self._last_phi:.4f} — motor kept active, self-model noisy"
            )
            # last_action NOT cleared — motor continues

        # 13a) Write prediction error for emotion module
        self._consciousness_pred_error = self.consciousness_state.prediction_error

        # 13b-attention) Top-down attention gating ───────────────────────────
        # The focused region receives a small excitatory boost from PFC
        # (corticothalamic top-down attention signal).  Non-focused *sensory*
        # regions get mild tonic suppression so attended content wins out.
        # Blend with AttentionController top-down priorities.
        _attn_focus = self.consciousness_state.focus_region
        _attn_ctrl = self._consciousness.attention_ctrl
        _attn_map: Dict[str, object] = {
            "sensory_visual": self.sensory_v,
            "sensory_auditory": self.sensory_a,
            "sensory_web": self.sensory_w,
            "thalamus": self.thalamus,
            "visual": self.visual,
            "auditory": self.auditory,
            "association": self.assoc,
            "hippocampus": self.hippocampus,
            "amygdala": self.amygdala,
            "prefrontal": self.prefrontal,
            "motor": self.motor,
        }
        if _attn_focus and _attn_focus in _attn_map:
            _fr = _attn_map[_attn_focus]
            _exc = getattr(_fr, "excitatory", [])
            _n = len(_exc)
            # Top-down bias from attention controller modulates boost strength
            _td_bias = _attn_ctrl.get_bias(_attn_focus)
            _boost = 1.8 + _td_bias * 1.0  # [1.8, 2.8] with bias
            if _n > 0:
                # Boost top 10% (up to 20) neurons in focused region
                _k = max(1, min(20, _n // 10))
                _fr.inject([_boost] * _k + [0.0] * (_n - _k))
            # Mild suppression on sensory regions not currently focused
            for _sname in ("sensory_visual", "sensory_auditory", "sensory_web"):
                if _sname != _attn_focus:
                    _sr = _attn_map[_sname]
                    _sne = getattr(_sr, "excitatory", [])
                    _sn = len(_sne)
                    if _sn > 0:
                        _sr.inject([-0.4] * min(8, _sn) + [0.0] * max(0, _sn - 8))

        # 13b) Drain outbound messages from communication drive → dialogue + TTS
        while self._consciousness.comm_drive.outbox:
            _msg = self._consciousness.comm_drive.outbox.popleft()
            _anchor = self._consciousness._current_report_anchor()
            try:
                _approved_msg = self._consciousness.generate_report(
                    _msg, concept=_anchor, source="comm_drive"
                )
            except Exception:
                continue
            self.outbound_messages.append(_approved_msg)
            # Route self-initiated messages through dialogue pipeline for TTS
            _addr = self._social_manager.primary_interlocutor() or "world"
            # ── Speech Act Planning for self-initiated speech ──
            # Just like user-input, plan the speech act before building
            # utterance so the system can ASK, BACKCHANNEL, REPAIR, etc.
            _body_urg = getattr(self._consciousness.body, "error_risk", 0.0)
            _comm_drv = self._consciousness.comm_drive_level
            # Query ToM FIRST so plan_response can use the social model
            _tom_strat_comm = {}
            if _addr != "world" and hasattr(self._consciousness, "theory_of_mind"):
                _tom_strat_comm = self._consciousness.theory_of_mind.recommend_strategy(
                    str(_addr)
                )
            _planned_act_comm = self._dialogue_manager.plan_response(
                tick=self.tick_count,
                last_incoming=None,
                comm_drive=_comm_drv,
                body_urgency=_body_urg,
                tom_strategy=_tom_strat_comm,
            )
            # Query ToM for prosody adaptation on self-initiated speech
            _tom_strat_comm = {}
            if _addr != "world" and hasattr(self._consciousness, "theory_of_mind"):
                _tom_strat_comm = self._consciousness.theory_of_mind.recommend_strategy(
                    str(_addr)
                )
            _uplan = self._dialogue_manager.build_utterance(
                _approved_msg,
                _addr,
                self._consciousness,
                speech_act=_planned_act_comm,
                tom_strategy=_tom_strat_comm,
            )
            self._speech_output.speak_utterance(_uplan)
            self._dialogue_manager.mark_output_delivered(self.tick_count)

        # 13d) Action toolbelt — AI decides & executes an action if warranted
        em = self.emotion_state
        action_result = self._actions.tick(
            self.tick_count, em, self.consciousness_state, self._consciousness
        )
        if action_result:
            last_action = self._actions.history[-1] if self._actions.history else None
            action_kind = last_action.kind if last_action else "?"
            if action_kind in ("look_at", "set_pose", "mirror_gesture", "track_person"):
                action_result = self._robot_controller.apply_action(
                    action_kind,
                    last_action.args if last_action else {},
                    self.tick_count,
                )
                frame = self._robot_controller.export_arduino_head_serial_frame(
                    self.tick_count
                )
                self._robot_serial.send_frame(frame)
                self.last_robot_command = frame.strip()[:160]
            self.action_log.append(
                {
                    "tick": self.tick_count,
                    "kind": action_kind,
                    "reason": last_action.reason if last_action else "",
                    "result": action_result,
                }
            )
            if len(self.action_log) > 200:
                self.action_log = self.action_log[-200:]
            # ── Tool output classification: filter before injection ──
            _tool_class = self._classify_tool_result(action_result, action_kind)
            if _tool_class == "valid_content":
                # Only valid content enters the semantic pipeline
                self.inject_text_input(action_result[:300])
            elif _tool_class == "operational_error":
                # Errors become episodic meta-events, not knowledge input
                self._consciousness.episodic.record(
                    self.tick_count,
                    "tool_error",
                    f"[{action_kind}] {action_result[:120]}",
                    self.emotion_state.describe(),
                )
                self._consciousness.stream.append(
                    f"[TOOL] {action_kind} error: {action_result[:60]}"
                )
            elif _tool_class == "empty_result":
                self._consciousness.stream.append(f"[TOOL] {action_kind}: no result")
            # low_confidence_result: inject but tag it
            elif _tool_class == "low_confidence":
                self.inject_text_input(f"(uncertain) {action_result[:280]}")

        # 13e) Robot controller feedback loop — compare current body command to world state
        self._robot_controller.control_step(self.tick_count)
        self._robot_controller.observe_world(
            self.latest_detections,
            self.latest_detection_targets,
            self.consciousness_state.focus_region,
            getattr(self._consciousness.task_frame, "plan_phase", "observe"),
            self.tick_count,
        )

        # 13f) Humanoid stack tick: body sync, safety, skills, executive, social
        self._body_schema.sync_from_controller(self._robot_controller)
        self._body_schema.step()
        _social_dist = self._world_state.zone.nearest_person_distance_cm
        self._safety.tick(self._body_schema, self._telemetry_bus, _social_dist)

        # 3.11 — E-stop → consciousness: inject halt thought when estop fires/clears
        _estop_now = self._safety.state.estop_active
        if _estop_now and not getattr(self, "_estop_was_active", False):
            self._consciousness.stream.append(
                f"[SAFETY] Emergency stop activated: {self._safety.state.reason}"
            )
            self._consciousness.episodic.record(
                self.tick_count,
                "safety_estop",
                f"ESTOP: {self._safety.state.reason}",
                "",
            )
            # Switch goal to halt so planner doesn't keep issuing actions
            if self._consciousness.state.goal not in ("halt", "rest"):
                self._consciousness.state.goal = "halt"
        elif not _estop_now and getattr(self, "_estop_was_active", False):
            self._consciousness.stream.append("[SAFETY] E-stop cleared — resuming")
        self._estop_was_active = _estop_now

        # Update body predicates for state-space planner
        self._world_state.update_body_predicates(self._body_schema)
        _skill_result = self._task_executive.tick(
            self.tick_count, self._body_schema, self._world_state, self._safety
        )
        self._social_manager.tick(
            self.tick_count, self._world_state, self._task_executive
        )
        # Periodically synchronize PersonModel ↔ MentalModel
        if self.tick_count % 100 == 0 and hasattr(
            self._consciousness, "theory_of_mind"
        ):
            self._social_manager.sync_with_tom(self._consciousness.theory_of_mind)
        self._dialogue_manager.tick(self.tick_count)
        # ── Backchannel injection during active listening ──────────────────
        # When the primary interlocutor is speaking and we're not generating
        # a reply, emit a short natural listening signal ("Hmm.", "Ja.", etc.)
        # to signal active listening.  Rate-limited by should_emit_backchannel().
        _bc_pid = self._social_manager.primary_interlocutor()
        if _bc_pid is not None:
            _bc_person = self._world_state.persons.get(_bc_pid)
            if _bc_person is None:
                # Try string→int matching
                try:
                    _bc_person = self._world_state.persons.get(int(_bc_pid))
                except (ValueError, TypeError):
                    pass
            _person_is_speaking = (
                _bc_person is not None and getattr(_bc_person, "speaking", False)
            )
            if _person_is_speaking and self._reply_requests.empty():
                _sp_ticks = getattr(self, "_person_speaking_ticks", 0) + 1
                self._person_speaking_ticks = _sp_ticks
                if self._dialogue_manager.should_emit_backchannel(
                    _sp_ticks, self.tick_count
                ):
                    _bc_lang = getattr(
                        getattr(self._consciousness, "lang", None), "_lang", "de"
                    )
                    _bc_emo = self.emotion_state.dominant()
                    _bc_text = self._dialogue_manager.generate_backchannel(
                        lang=_bc_lang, emotion=_bc_emo
                    )
                    if _bc_text:
                        _bc_plan = self._dialogue_manager.build_utterance(
                            _bc_text,
                            str(_bc_pid),
                            self._consciousness,
                            speech_act=None,
                            tick=self.tick_count,
                            confidence=0.9,
                        )
                        _bc_plan.speed_factor = 0.85  # gentle, non-intrusive
                        _bc_plan.head_nod = True       # nod accompanies backchannel
                        self._speech_output.speak_utterance(_bc_plan)
                        self._dialogue_manager.record_backchannel_sent(self.tick_count)
            else:
                self._person_speaking_ticks = 0
        self._body_interface.tick()

        # ── Gaze dynamics: natural gaze + blink cycle ─────────────────────
        try:
            if self._gaze_dynamics is None:
                from robot_controller import GazeDynamics
                self._gaze_dynamics = GazeDynamics()
            _is_speaking = self._speech_output.is_speaking
            self._gaze_dynamics.tick(self.tick_count, _is_speaking, self._robot_controller)
        except Exception:
            pass

        # 5.4 — TurnState → consciousness self_model: expose active turn state
        _sm_primary = self._social_manager.primary_interlocutor()
        if _sm_primary is not None:
            _ts = self._social_manager.turn_state_for(_sm_primary)
            self._consciousness.self_model.turn_state = _ts.value

        # Update last_executed_action from real skill name, not status strings.
        # Priority: active skill name > last completed skill > action toolbelt > semantic decode
        _exec_skill = self._task_executive.current_skill_name()
        if _exec_skill:
            self.last_executed_action = _exec_skill
        elif _skill_result and _skill_result.startswith("goal_done:"):
            # Goal just finished — keep the last skill that actually ran
            _ag = self._task_executive.active_goal
            if _ag is None and self._task_executive._history:
                _last_g = self._task_executive._history[-1]
                if _last_g.steps:
                    self.last_executed_action = _last_g.steps[-1].skill_name
        elif action_result:
            _last_act = self._actions.history[-1] if self._actions.history else None
            self.last_executed_action = (
                _last_act.kind if _last_act else self.last_action
            )
        else:
            self.last_executed_action = self.last_action

        # Feed proprioceptive snapshot into consciousness
        self._consciousness._proprioceptive = (
            self._body_schema.proprioceptive_snapshot()
        )

        # 13d) Sleep / Wake rhythm ─────────────────────────────────────────
        _phase_pos = self.tick_count % SLEEP_CYCLE_TICKS
        _sleep_onset = int(SLEEP_CYCLE_TICKS * (1.0 - SLEEP_FRACTION))
        _was_sleeping = self.sleeping
        self.sleeping = _phase_pos >= _sleep_onset

        if self.sleeping != _was_sleeping:
            if self.sleeping:
                # Enter sleep — reduce tonic drive so neurons can consolidate
                from neuron import Neuron as _N

                _N.global_tonic_current = SLEEP_TONIC
                self._consciousness.episodic.record(
                    self.tick_count,
                    "sleep",
                    "Entering sleep phase — consolidation replay",
                )
            else:
                # Wake up — restore activity and record summary
                from neuron import Neuron as _N

                # Energy-scaled tonic: low energy → quieter wake state
                _eff_tonic_wake = WAKE_TONIC * max(
                    0.30, self._consciousness.body.energy_reserve
                )
                _N.global_tonic_current = _eff_tonic_wake
                self._consciousness.episodic.record(
                    self.tick_count, "sleep", "Waking — resuming active processing"
                )

        # During sleep: body recovery + episodic emotional consolidation ─────
        if self.sleeping:
            _body = self._consciousness.body
            # Homeostatic recovery: repair energy, integrity, error_risk during sleep
            _body.energy_reserve = min(1.0, _body.energy_reserve + 0.0003)
            _body.integrity = min(1.0, _body.integrity + 0.0002)
            _body.error_risk = max(0.0, _body.error_risk - 0.0002)
            _body.thermal_load = max(0.0, _body.thermal_load - 0.0002)
            _body.regen_need = max(0.0, _body.regen_need - 0.0001)

            # Episodic consolidation: re-inject top emotionally significant memories
            # every 50 sleep ticks so STDP can strengthen those engrams
            if self.tick_count % 50 == 0:
                _ep = self._consciousness.episodic
                _evts = list(_ep._events)
                # Score events by emotional keyword richness (proxy for salience)
                _sal_kws = (
                    "joy",
                    "stress",
                    "surprise",
                    "ignition",
                    "IGNITION",
                    "insight",
                    "biography",
                    "strategy",
                )
                _scored = [
                    (sum(kw in (e.emotion_snapshot + e.content) for kw in _sal_kws), e)
                    for e in _evts
                ]
                _scored.sort(key=lambda x: x[0], reverse=True)
                for _score, _evt in _scored[:3]:
                    if _score > 0 and getattr(self, "_sensor_free_ticks", 0) == 0:
                        # Replay-inject only when world-coupled; deprived state must
                        # not receive endogenous reactivation that mimics real input.
                        self.inject_text_input(_evt.content[:120])

            # ── Counterfactual simulation (every 200 sleep ticks) ─────
            # Take a past negative episode and ask "what if I had done X instead?"
            # Generates hypothetical variations that strengthen alternative pathways.
            if self.tick_count % 200 == 0:
                self._offline_counterfactual()

            # ── Belief consolidation (every 300 sleep ticks) ──────────
            if self.tick_count % 300 == 0:
                self._consciousness.belief_store.decay()

            # ── Hypothesis testing (every 400 sleep ticks) ────────────
            # Re-evaluate contradicted beliefs during offline processing
            if self.tick_count % 400 == 0:
                self._sleep_hypothesis_test()

            # ── Episodic→Belief grounding (every 350 sleep ticks) ─────
            # Cross-reference recent episodic experience against beliefs.
            # Episodes with prediction/observed_outcome data provide hard
            # evidence; repeatedly falsified beliefs are quarantined.
            if self.tick_count % 350 == 0:
                self._sleep_episodic_belief_grounding()

            # ── Causal→Belief extraction (every 450 sleep ticks) ──────
            # Promote high-confidence causal graph patterns into explicit
            # beliefs so the system can reason about "if X then Y" rules
            # consciously, not just implicitly in goal scoring.
            if self.tick_count % 450 == 0:
                self._sleep_causal_to_belief()

            # ── Phase 5: Targeted identity consolidation (every 500 sleep ticks) ──
            if self.tick_count % 500 == 0:
                self._sleep_identity_consolidation()

            # ── Phase 6: Narrative → Identity target sync (every 600 sleep ticks) ──
            if self.tick_count % 600 == 0:
                self._sleep_narrative_identity_sync()

        # ── Waking: lightweight online belief cross-check (every 30 ticks) ──
        else:
            if self.tick_count % 30 == 0:
                self._waking_belief_revision()

        # 14) Autosave
        if self.tick_count > 0 and self.tick_count % AUTO_SAVE_INTERVAL == 0:
            self._autosave()

        self.t += SIM_DT
        self.tick_count += 1

        # 15) Session metrics sampling
        if self.tick_count % SESSION_SAMPLE_INTERVAL == 0:
            try:
                self._session_metrics.sample(self)
            except Exception:
                pass

        if self.tick_count > 0 and self.tick_count % 300 == 0:
            try:
                _results = self._consciousness_testbed.run_all(
                    self._consciousness, self
                )
                _failed = [r.test_name for r in _results if not r.passed]
                self._last_consciousness_test_summary = (
                    f"passed={sum(1 for r in _results if r.passed)}/{len(_results)} "
                    f"failed={_failed[:4]}"
                )
                self._consciousness.stream.append(
                    f"[CTEST] {self._last_consciousness_test_summary}"
                )
                self._consciousness.episodic.record(
                    self.tick_count,
                    "consciousness_tests",
                    self._last_consciousness_test_summary[:140],
                    self.emotion_state.describe(),
                )
            except Exception as _exc:
                self._consciousness.stream.append(f"[CTEST] runtime_error:{_exc}")

    # ─────────────────────────────────────────────────────────────
    # Offline sleep: counterfactual simulation + hypothesis testing
    # ─────────────────────────────────────────────────────────────

    def _offline_counterfactual(self) -> None:
        """Counterfactual episode simulation during sleep."""
        import random as _rnd

        _ep = self._consciousness.episodic
        _evts = list(_ep._events)
        _neg = [
            e
            for e in _evts
            if e.kind in ("postmortem", "goal", "world_surprise")
            and any(
                kw in e.content.lower()
                for kw in ("fail", "error", "unexpected", "violation")
            )
        ]
        if not _neg:
            return

        target = _rnd.choice(_neg[-5:])
        alternatives = [
            "approached more cautiously",
            "checked preconditions first",
            "waited before acting",
            "used a simpler strategy",
            "paid more attention to the person",
        ]
        alt = _rnd.choice(alternatives)
        dream_content = (
            f"[DREAM] Revisiting: {target.content[:80]}. "
            f"Counterfactual: what if I had {alt}? "
            f"Simulating alternative outcome..."
        )
        # Counterfactual replay injects text into the neural substrate.
        # When sensors are absent this creates spurious activity variance that
        # can briefly rescue phi — producing ghost coherence from pure memory.
        # Block the injection; the episodic record is preserved either way.
        if getattr(self, "_sensor_free_ticks", 0) == 0:
            self.inject_text_input(dream_content[:150])
        self._consciousness.episodic.record(
            self.tick_count, "dream", dream_content[:120], "dreaming"
        )
        self._consciousness.world_model.record_action(
            f"counterfactual:{target.content[:30]}", alt, 0.1
        )

    def _sleep_hypothesis_test(self) -> None:
        """Review contradicted beliefs during sleep."""
        bs = self._consciousness.belief_store
        contradictions = bs.contradictions(min_count=2)
        for subj, rel, obj, entry in contradictions[:3]:
            if entry.contradiction_count > entry.evidence_count:
                entry.confidence *= 0.7
                if entry.confidence < 0.1:
                    bs._beliefs.get(subj, {}).get(rel, {}).pop(obj, None)
                    self._consciousness.episodic.record(
                        self.tick_count,
                        "belief_revision",
                        f"Dropped belief: {subj} {rel} {obj} "
                        f"(contradictions={entry.contradiction_count})",
                        "sleep_consolidation",
                    )
            elif entry.evidence_count > entry.contradiction_count * 2:
                entry.confidence = min(0.95, entry.confidence * 1.05)

    def _sleep_episodic_belief_grounding(self) -> None:
        """Cross-reference recent episodic experience against active beliefs.

        Episodes carry prediction/observed_outcome fields. When an observed
        outcome contradicts a belief, that belief's contradiction_count rises
        and confidence drops. When an outcome confirms a belief, evidence_count
        rises and confidence is boosted. Repeatedly falsified beliefs are
        quarantined automatically.
        """
        bs = self._consciousness.belief_store
        ep = self._consciousness.episodic
        # Gather recent episodes with prediction+outcome data
        recent = [
            e for e in list(ep._events)[-200:] if e.prediction and e.observed_outcome
        ]
        if not recent:
            return

        _revised = 0
        for evt in recent[-15:]:  # process at most 15 per sleep cycle
            pred = evt.prediction.lower()
            outcome = evt.observed_outcome.lower()
            # Determine if prediction matched outcome
            _match = (
                outcome.startswith("success")
                or outcome.startswith("confirm")
                or "as expected" in outcome
                or "matched" in outcome
            )
            _contra = (
                outcome.startswith("fail")
                or outcome.startswith("unexpected")
                or "violated" in outcome
                or "wrong" in outcome
                or "contradict" in outcome
            )
            if not _match and not _contra:
                continue

            # Extract likely belief subject from the event content/prediction
            _tokens = [t for t in pred.split() if len(t) > 3][:3]
            if not _tokens:
                continue
            _subj = _tokens[0]

            # Search beliefs that mention the subject
            _subj_beliefs = bs._beliefs.get(_subj, {})
            for rel, objs in _subj_beliefs.items():
                for obj, entry in list(objs.items()):
                    if _match:
                        entry.evidence_count += 1
                        entry.confidence = min(0.95, entry.confidence * 1.03)
                    elif _contra:
                        entry.contradiction_count += 1
                        entry.confidence *= 0.85
                        _revised += 1
                        # Auto-quarantine if heavily falsified
                        if entry.contradiction_count > entry.evidence_count + 3:
                            if entry.confidence < 0.25:
                                bs._quarantined.append((_subj, rel, obj, entry))
                                objs.pop(obj, None)
                                self._consciousness.episodic.record(
                                    self.tick_count,
                                    "belief_revision",
                                    f"Quarantined belief from experience: "
                                    f"{_subj} {rel} {obj} "
                                    f"(falsified {entry.contradiction_count}x)",
                                    "sleep_episodic_grounding",
                                )

        if _revised > 0:
            self._consciousness.stream.append(
                f"[SLEEP] Episodic belief grounding: {_revised} beliefs revised "
                f"from {len(recent)} experience episodes"
            )

    def _sleep_causal_to_belief(self) -> None:
        """Promote high-confidence causal graph edges into explicit beliefs.

        Edges with high reliability + enough support are converted to
        'causes'/'enables' beliefs in the BeliefStore.  These become
        consciously accessible rules like "greeting a person often succeeds"
        that can surface in introspection and response grounding.
        """
        cg = self._consciousness.causal_graph
        bs = self._consciousness.belief_store
        _promoted = 0
        _MIN_RELIABILITY = 0.65
        _MIN_SUPPORT = 8
        _MAX_PER_CYCLE = 5
        for _key, edge in cg._edges.items():
            if _promoted >= _MAX_PER_CYCLE:
                break
            if edge.reliability < _MIN_RELIABILITY:
                continue
            if edge.support_count < _MIN_SUPPORT:
                continue
            # Parse the cause into subject + action
            _parts = edge.cause.split("|")
            _subj = _parts[0] if _parts else edge.cause
            _action = _parts[1] if len(_parts) > 1 else ""
            if not _subj or len(_subj) < 3:
                continue
            # Determine relation from success/reward characteristics
            _rel = "causes" if edge.avg_reward > 0.1 else "enables"
            _obj = edge.effect if edge.effect else _action
            if not _obj or len(_obj) < 3 or _obj == _subj:
                continue
            # Store as experience-based belief with moderate confidence
            _conf = min(0.85, 0.4 + edge.reliability * 0.4)
            bs._store(
                _subj,
                _rel,
                _obj,
                _conf,
                source="causal_graph",
                tick=self.tick_count,
                epistemic_status=EpistemicStatus.INFERENCE,
            )
            _promoted += 1
        if _promoted > 0:
            self._consciousness.stream.append(
                f"[SLEEP] Causal→Belief: {_promoted} experience rules promoted"
            )

    def _sleep_narrative_identity_sync(self) -> None:
        """Phase 6: Map recurring narrative patterns back to IdentityArc target values.

        Repeated chapter types across recent history provide evidence about
        which dimensions the system exercises and should aspire to strengthen.
        Growth/learning → analytical_depth/reflectiveness targets up;
        Conflict → impulse_control target up;
        Social + success → social_dominance/empathy targets adjusted.
        """
        cs = self._consciousness
        nt = cs.narrative_thread
        ia = cs.identity_arc

        recent = nt.recent_chapters(5)
        if not recent:
            return

        # Count chapter types weighted by recency (most recent = highest weight)
        type_weights: Dict[str, float] = {}
        for i, ch in enumerate(recent):
            w = (i + 1) / len(recent)
            type_weights[ch.chapter_type] = type_weights.get(ch.chapter_type, 0.0) + w

        _success_kws = ("breakthrough", "steady progress", "effective")
        _fail_kws = ("unresolved", "strategy shift", "no clear resolution")
        _success_n = sum(
            1 for ch in recent if any(kw in ch.resolution for kw in _success_kws)
        )
        _fail_n = sum(
            1 for ch in recent if any(kw in ch.resolution for kw in _fail_kws)
        )
        _net = (_success_n - _fail_n) / max(len(recent), 1)

        _NUDGE = 0.01  # max per-dimension nudge per sync
        _log: list = []

        def _nudge(dim_name: str, delta: float) -> None:
            d = ia.dimensions.get(dim_name)
            if d is None:
                return
            d.target = max(0.1, min(0.98, d.target + delta))
            _log.append(
                f"{dim_name}.target{'+'if delta>0 else ''}{delta:+.3f}={d.target:.3f}"
            )

        # Growth / learning → raise analytical depth + reflectiveness
        if (
            type_weights.get("growth", 0) >= 1.5
            or type_weights.get("learning", 0) >= 1.5
        ):
            if _net >= 0:
                _nudge("analytical_depth", _NUDGE)
            if type_weights.get("learning", 0) >= 1.0:
                _nudge("reflectiveness", _NUDGE * 0.5)

        # Social chapters → adjust social_dominance by success direction
        if type_weights.get("social", 0) >= 1.5:
            _delta = _NUDGE if _net > 0 else -_NUDGE * 0.5
            _nudge("social_dominance", _delta)
            if _net > 0.3:
                _dim_emp = ia.dimensions.get("empathy_selective")
                if _dim_emp and _dim_emp.target < 0.65:
                    _nudge("empathy_selective", _NUDGE * 0.5)

        # Conflict → impulse_control target up (need more self-regulation)
        if type_weights.get("conflict", 0) >= 1.5:
            _nudge("impulse_control", _NUDGE)
            if _fail_n > _success_n:
                _dim_r = ia.dimensions.get("ruthlessness")
                if _dim_r and _dim_r.target > 0.4:
                    _nudge("ruthlessness", -_NUDGE * 0.5)

        # Repeated turning points → autonomy nudge
        _turn_count = sum(1 for ch in recent if ch.turning_point)
        if _turn_count >= 2:
            _nudge("autonomy", _NUDGE * 0.5)

        # ── Lesson / conflict / resolution text → dimension-specific evidence ──
        _DIM_KWS: Dict[str, str] = {
            "loyal": "loyalty_intensity",
            "treu": "loyalty_intensity",
            "honest": "honesty_orientation",
            "ehrlich": "honesty_orientation",
            "patient": "impulse_control",
            "geduldig": "impulse_control",
            "empath": "empathy_selective",  # prefix: empathetic / empathisch
            "mitfühlend": "empathy_selective",
            "confident": "social_dominance",
            "selbstbewusst": "social_dominance",
            "reflect": "reflectiveness",
            "nachdenklich": "reflectiveness",
            "analytical": "analytical_depth",
        }
        _HALF = _NUDGE * 0.5
        _res_success_kws = (
            "resolve",
            "success",
            "breakthrough",
            "effective",
            "gelöst",
            "erfolgreich",
        )
        _res_fail_kws = (
            "unresolved",
            "failed",
            "no progress",
            "ungeklärt",
            "gescheitert",
        )
        for ch in recent:
            _combined = " ".join(ch.lessons).lower() + " " + ch.conflict.lower()
            for kw, dim in _DIM_KWS.items():
                if kw in _combined:
                    _nudge(dim, _HALF)

            # Resolution polarity: success boosts reflectiveness, failure boosts impulse_control
            _res_low = ch.resolution.lower()
            if any(s in _res_low for s in _res_success_kws):
                _nudge("reflectiveness", _HALF)
            elif any(f in _res_low for f in _res_fail_kws):
                _nudge("impulse_control", _HALF)

        # ── RelationshipArc trust trends → empathy / dominance targets ──
        _seen_pids: set = set()
        for ch in recent:
            for pid in ch.persons_involved:
                if pid in _seen_pids:
                    continue
                _seen_pids.add(pid)
                _arc = nt._relationship_arcs.get(pid)
                if _arc is None:
                    continue
                _trend = _arc.trust_trend()
                if _trend == "high_trust":
                    _nudge("empathy_selective", _HALF)
                elif _trend == "declining":
                    _nudge("social_dominance", -_HALF * 0.5)
                    _nudge("impulse_control", _HALF * 0.5)
                elif _trend == "low_trust":
                    _nudge("social_dominance", -_HALF)

        if _log:
            cs.stream.append(f"[NARR→ID] Narrative→Identity sync: {'; '.join(_log)}")
            cs.episodic.record(
                self.tick_count,
                "identity_narrative",
                f"Narr→Identity: {', '.join(_log[:3])}",
                "sleep",
            )

    def _waking_belief_revision(self) -> None:
        """Online lightweight cross-check of recent experience vs active beliefs.

        Complements the deep sleep path (_sleep_episodic_belief_grounding) with
        a fast waking pass: scan the last few episodes with prediction+outcome
        data and immediately bump evidence or contradiction counts on matching
        beliefs, so the system doesn't drift for hours on stale beliefs.
        """
        bs = self._consciousness.belief_store
        ep = self._consciousness.episodic
        recent = [
            e for e in list(ep._events)[-30:] if e.prediction and e.observed_outcome
        ]
        if not recent:
            return

        _revised = 0
        for evt in recent[-5:]:  # at most 5 per waking interval
            pred = evt.prediction.lower()
            outcome = evt.observed_outcome.lower()
            _match = (
                outcome.startswith("success")
                or "as expected" in outcome
                or "matched" in outcome
            )
            _contra = (
                outcome.startswith("fail")
                or "unexpected" in outcome
                or "violated" in outcome
                or "wrong" in outcome
                or "contradict" in outcome
            )
            if not _match and not _contra:
                continue
            _tokens = [t for t in pred.split() if len(t) > 3][:5]
            if not _tokens:
                continue
            # Find matching belief subjects: try all tokens, prefer explicit field
            _matching_subjects = [t for t in _tokens if t in bs._beliefs]
            # Use explicit belief_subject field if present (structured episode)
            _explicit_subj = getattr(evt, "belief_subject", None)
            if _explicit_subj and _explicit_subj in bs._beliefs:
                _matching_subjects = [_explicit_subj] + [
                    s for s in _matching_subjects if s != _explicit_subj
                ]
            if not _matching_subjects:
                # Fallback: first token even if not currently in belief store
                _matching_subjects = [_tokens[0]]
            for _subj in _matching_subjects[:2]:  # at most 2 subjects per event
                for rel, objs in bs._beliefs.get(_subj, {}).items():
                    for obj, entry in list(objs.items()):
                        if _match:
                            entry.evidence_count = min(entry.evidence_count + 1, 50)
                            entry.confidence = min(0.95, entry.confidence * 1.02)
                        elif _contra:
                            entry.contradiction_count += 1
                            entry.confidence *= 0.93  # gentler than sleep path
                            _revised += 1
        if _revised > 0:
            self._consciousness.stream.append(
                f"[WAKING] Online belief revision: {_revised} belief(s) updated"
            )

    def _sleep_identity_consolidation(self) -> None:
        """
        Phase 5: Targeted identity consolidation during sleep.
        When ContinuityMonitor signals fragile segments, sleep focuses
        on reinforcing those specific identity aspects.
        """
        cs = self._consciousness
        fragile = getattr(self, "_continuity_fragile", [])
        if not fragile:
            return

        for segment in fragile:
            if segment == "memory":
                # Replay identity-relevant episodes (biography, guideline)
                bio_events = [
                    e
                    for e in cs.episodic._events
                    if e.kind in ("biography", "guideline", "self")
                ]
                for evt in bio_events[-3:]:
                    # Neural injection only when world-coupled; memory replay
                    # must not bootstrap phi in a sensor-deprived system.
                    if getattr(self, "_sensor_free_ticks", 0) == 0:
                        self.inject_text_input(evt.content[:120])
                cs.episodic.record(
                    self.tick_count,
                    "sleep_repair",
                    f"Consolidating memory identity ({len(bio_events)} bio events)",
                    "sleep",
                )

            elif segment == "agency":
                # Replay successful action episodes — read-only reconstruction.
                # No direct agency_score mutation: replay reconstructs past states,
                # it does NOT generate new agency.  Agency must be earned through
                # grounded world-action-outcome loops, not internal replay.
                causal_events = [
                    e
                    for e in cs.episodic._events
                    if e.kind == "causal" and "actual=+" in e.content
                ]
                for evt in causal_events[-3:]:
                    if getattr(self, "_sensor_free_ticks", 0) == 0:
                        self.inject_text_input(evt.content[:120])
                cs.episodic.record(
                    self.tick_count,
                    "sleep_repair",
                    f"Replayed {len(causal_events[-3:])} agency episodes "
                    f"(read-only; no state mutation without world coupling)",
                    "sleep",
                )

            elif segment == "values":
                # Reinforce personal guidelines
                for gl in cs.autobiography.guidelines:
                    gl.strength = min(1.0, gl.strength + 0.1)
                    if getattr(self, "_sensor_free_ticks", 0) == 0:
                        self.inject_text_input(gl.text[:120])
                cs.episodic.record(
                    self.tick_count,
                    "sleep_repair",
                    f"Reinforcing {len(cs.autobiography.guidelines)} guidelines",
                    "sleep",
                )

    # ─────────────────────────────────────────────────────────
    # Structural plasticity — prune weak, sprout near-saturated
    # ─────────────────────────────────────────────────────────

    def _structural_plasticity(self) -> None:
        import random

        # Build a name→region map once for fast lookup
        _region_map: Dict[str, object] = {r.name: r for r in self._all_regions}

        survivors: List[Synapse] = []
        sprouts: List[Synapse] = []
        MAX_SPROUTS_PER_CALL = 200  # cap new synapses from sprouting per call

        for syn in self._inter_synapses:
            if syn.is_depressed(PRUNE_THRESHOLD):
                try:
                    syn.pre.efferents.remove(syn)
                    syn.post.afferents.remove(syn)
                except ValueError:
                    pass
            else:
                survivors.append(syn)
                if (
                    len(sprouts) < MAX_SPROUTS_PER_CALL
                    and syn.is_potentiated(SPROUT_THRESHOLD)
                    and random.random() < SPROUT_PROB
                ):
                    post_region_name = syn.post.region
                    region = _region_map.get(post_region_name)
                    tgt_pool = getattr(region, "_exc_cache", []) if region else []
                    if tgt_pool:
                        already_ids = {s.post.nid for s in syn.pre.efferents}
                        local_radius = (
                            max(
                                getattr(region, "extent", (SPATIAL_LOCAL_RADIUS,))[0],
                                SPATIAL_LOCAL_RADIUS,
                            )
                            * 0.65
                        )
                        cands = self._local_spatial_candidates(
                            syn.pre,
                            tgt_pool,
                            already_ids=already_ids,
                            max_distance=local_radius,
                            limit=24,
                        )
                        if cands:
                            tgt = random.choice(cands)
                            sprouts.append(
                                self._distance_scaled_synapse(
                                    syn.pre,
                                    tgt,
                                    base_weight=syn.weight * 0.34,
                                    delay_range=(
                                        max(0.5, syn.delay * 0.55),
                                        max(1.0, syn.delay * 1.10),
                                    ),
                                    distance_scale=local_radius,
                                )
                            )

        self._inter_synapses = survivors + sprouts

    # ─────────────────────────────────────────────────────────
    # Rapid Hebbian co-activation synapse formation
    # ─────────────────────────────────────────────────────────

    def _coactivation_sprout(self, cap: int = COACTIVATION_PER_TICK) -> None:
        """
        Form new synapses guided by semantic concept co-occurrence.

        Old behaviour: any neuron with STDP trace > 0.05 got connected —
        producing random connectivity unrelated to what is being learned.

        New behaviour:
        1. Use the concept graph to find semantically related concept pairs.
        2. Map each concept to a hash-indexed assembly in its home region.
        3. Form synapses between recently-active neurons in those assemblies.

        Synaptic connections now encode LEARNED ASSOCIATIONS between semantic
        units — not between random co-firing neurons.

        Fallback (when concepts are sparse): legacy trace-based sprouting.
        """
        n_current = len(self._inter_synapses)
        if n_current >= MAX_SYNAPSES:
            # Random eviction — avoid O(N log N) sort on millions of synapses
            import random as _r

            evict_idx = set(_r.sample(range(n_current), min(cap, n_current)))
            to_remove = [self._inter_synapses[i] for i in evict_idx]
            for syn in to_remove:
                try:
                    syn.pre.efferents.remove(syn)
                    syn.post.afferents.remove(syn)
                except ValueError:
                    pass
            self._inter_synapses = [
                s for i, s in enumerate(self._inter_synapses) if i not in evict_idx
            ]

        import random

        from regions import REWARD_CONCEPTS, THREAT_CONCEPTS

        new_count = 0

        # ── Semantic-guided sprouting ─────────────────────────────────
        cg = self._consciousness.concept_graph
        workspace = self._consciousness.workspace_concepts()

        if workspace and cg._edges:
            for concept in workspace[:6]:
                if new_count >= cap:
                    break
                neighbors = cg.neighbors(concept, top_n=4)
                for neighbor_concept, edge_weight in neighbors:
                    if new_count >= cap:
                        break
                    if edge_weight < 0.15:
                        continue  # too weak — not a real association yet

                    # Source region: where this concept is being processed
                    src_region = self.assoc
                    # Target region: based on semantic category of neighbor
                    if neighbor_concept.lower() in THREAT_CONCEPTS:
                        tgt_region = self.amygdala
                    elif neighbor_concept.lower() in REWARD_CONCEPTS:
                        tgt_region = self.hippocampus
                    else:
                        tgt_region = self.prefrontal

                    TRACE_MIN = 0.05
                    src_active = src_region.active_excitatory(TRACE_MIN)
                    tgt_pool = tgt_region._exc_cache
                    if not src_active or not tgt_pool:
                        continue

                    # Concept hash selects the assembly index within the region
                    c_idx = abs(hash(concept)) % len(src_region._exc_cache)
                    n_idx = abs(hash(neighbor_concept)) % len(tgt_region._exc_cache)

                    # Prefer neurons near each concept's assembly index
                    # Use direct slice on _exc_cache — avoids O(N) dict.get()/abs() per neuron
                    _src_lo = max(0, c_idx - 40)
                    _src_hi = min(len(src_region._exc_cache), c_idx + 41)
                    src_near = [
                        n
                        for n in src_region._exc_cache[_src_lo:_src_hi]
                        if n.trace > TRACE_MIN
                    ] or src_active[:4]

                    for src_n in src_near[:2]:
                        if new_count >= cap:
                            break
                        already_ids = {s.post.nid for s in src_n.efferents}
                        # Slice tgt_pool around assembly index — O(1) vs O(N) dict loop
                        _tgt_lo = max(0, n_idx - 50)
                        _tgt_hi = min(len(tgt_pool), n_idx + 51)
                        indexed_near = [
                            tn
                            for tn in tgt_pool[_tgt_lo:_tgt_hi]
                            if tn.nid not in already_ids
                        ]
                        local_radius = (
                            max(
                                max(src_region.extent),
                                max(tgt_region.extent),
                                SPATIAL_LOCAL_RADIUS,
                            )
                            * 0.55
                        )
                        tgt_near = self._local_spatial_candidates(
                            src_n,
                            indexed_near or tgt_pool,
                            already_ids=already_ids,
                            max_distance=local_radius,
                            limit=32,
                        )
                        if not tgt_near:
                            continue
                        tgt_n = tgt_near[0]
                        base_w = max(0.08, min(0.8, edge_weight * 0.18 + 0.04))
                        syn = self._distance_scaled_synapse(
                            src_n,
                            tgt_n,
                            base_weight=base_w,
                            delay_range=(0.5, 3.0),
                            distance_scale=local_radius,
                        )
                        self._inter_synapses.append(syn)
                        new_count += 1
                        self._synapses_formed += 1

        # ── Fallback: legacy trace-based sprouting ────────────────────
        if new_count < cap // 2:
            TRACE_MIN = 0.05
            pairs = [
                (self.sensory_w, self.assoc),
                (self.sensory_w, self.hippocampus),
                (self.thalamus, self.visual),
                (self.thalamus, self.auditory),
                (self.visual, self.assoc),
                (self.auditory, self.assoc),
                (self.assoc, self.hippocampus),
                (self.assoc, self.prefrontal),
                (self.hippocampus, self.prefrontal),
                (self.amygdala, self.prefrontal),
                (self.prefrontal, self.motor),
            ]
            for src_r, tgt_r in pairs:
                if new_count >= cap:
                    break
                src_active = src_r.active_excitatory(0.05)
                if not src_active:
                    continue
                tgt_pool = tgt_r._exc_cache
                if not tgt_pool:
                    continue
                for src_n in random.sample(src_active, min(3, len(src_active))):
                    if new_count >= cap:
                        break
                    already_ids = {s.post.nid for s in src_n.efferents}
                    local_radius = (
                        max(max(src_r.extent), max(tgt_r.extent), SPATIAL_LOCAL_RADIUS)
                        * 0.75
                    )
                    cands = self._local_spatial_candidates(
                        src_n,
                        tgt_pool,
                        already_ids=already_ids,
                        max_distance=local_radius,
                        limit=36,
                    )
                    if not cands:
                        continue
                    tgt_n = random.choice(cands)
                    base_w = max(0.04, random.gauss(0.10, 0.03))
                    syn = self._distance_scaled_synapse(
                        src_n,
                        tgt_n,
                        base_weight=base_w,
                        delay_range=(0.5, 3.0),
                        distance_scale=local_radius,
                    )
                    self._inter_synapses.append(syn)
                    new_count += 1
                    self._synapses_formed += 1

        # ── STDP → semantic memory bridge ─────────────────────────────
        # When STDP has potentiated synapses between hippocampus neurons that
        # represent specific concepts, those strong weights SHOULD strengthen
        # the semantic memory edge between those concepts. This is the reverse
        # direction:  LIF synapse weight → semantic_memory update.
        # Only runs once every 3 calls (every ~3 ticks) to keep cost low.
        if self.tick_count % 3 == 0:
            _hipp_cai = self.hippocampus._concept_at_index
            _hipp_nid_idx = self.hippocampus._exc_nid_idx
            _pfc_cai = self.prefrontal._concept_at_index
            _pfc_nid_idx = self.prefrontal._exc_nid_idx
            _sm = self.hippocampus._semantic_memory
            _cg = self._consciousness.concept_graph
            for syn in self._inter_synapses:
                if syn.weight < 3.5:
                    continue  # only strongly potentiated synapses qualify
                # Find post-concept (post neuron in hippocampus?)
                post_idx = _hipp_nid_idx.get(syn.post.nid)
                if post_idx is None:
                    continue
                post_c = _hipp_cai.get(post_idx)
                if not post_c:
                    continue
                # Find pre-concept (pre neuron in hippocampus or PFC)
                pre_idx_h = _hipp_nid_idx.get(syn.pre.nid)
                if pre_idx_h is not None:
                    pre_c = _hipp_cai.get(pre_idx_h)
                else:
                    pre_idx_p = _pfc_nid_idx.get(syn.pre.nid)
                    pre_c = _pfc_cai.get(pre_idx_p) if pre_idx_p is not None else None
                if not pre_c or pre_c == post_c:
                    continue
                # Strengthen semantic memory edge proportional to excess weight
                boost = (syn.weight - 3.0) * 0.04
                _sm.setdefault(pre_c, {})[post_c] = min(
                    5.0, _sm.get(pre_c, {}).get(post_c, 0.0) + boost
                )
                _sm.setdefault(post_c, {})[pre_c] = min(
                    5.0, _sm.get(post_c, {}).get(pre_c, 0.0) + boost
                )
                # Also strengthen concept graph edge
                _cg.observe_pair(pre_c, post_c, boost * 2.0)

    # ─────────────────────────────────────────────────────────────
    # Speech / text  → neural injection
    # ─────────────────────────────────────────────────────────────

    # ── Tool output classification ─────────────────────────────
    _TOOL_ERROR_PREFIXES = (
        "Search error:",
        "Fetch error:",
        "Read error:",
        "Write error:",
        "Open error:",
        "Run error:",
        "Timeout.",
        "No query.",
        "No URL.",
        "No command.",
        "No app specified.",
        "File not found:",
    )

    def _classify_tool_result(self, result: str, action_kind: str) -> str:
        """Classify a tool result for routing: valid_content | operational_error | empty_result | low_confidence."""
        if not result or result.isspace():
            return "empty_result"
        stripped = result.strip()
        if any(stripped.startswith(p) for p in self._TOOL_ERROR_PREFIXES):
            return "operational_error"
        if stripped == "(no output)":
            return "empty_result"
        if stripped in ("No direct answer found.", "Nothing to write."):
            return "low_confidence"
        if len(stripped) < 8:
            return "low_confidence"
        return "valid_content"

    def inject_text_input(self, text: str) -> None:
        """
        Encode text as spike currents AND process it through the semantic pipeline.

        Neural path (LIF substrate):
          text → encode_text() → currents → sensory_w, prefrontal, hippocampus, assoc

        Semantic path (consciousness layer):
          text → concept extraction → hippocampus.semantic_encode()
               → amygdala.semantic_appraise() → PFC working memory update

        Both paths run together. The neural path drives STDP plasticity;
        the semantic path makes the system actually understand what was said.
        """
        from web_sensor import encode_text

        currents = encode_text(text, n_neurons=WEB_N)
        boosted = [min(c * 1.8, 22.0) for c in currents]
        self.sensory_w.inject(boosted)
        pfc_exc = self.prefrontal._exc_cache
        pfc_boost = [min(c * 0.5, 15.0) for c in boosted[: len(pfc_exc)]]
        self.prefrontal.inject(pfc_boost)
        _cn = len(currents)
        hipp_n = len(self.hippocampus._exc_cache)
        self.hippocampus.inject(
            [min(currents[i % _cn] * 1.0, 14.0) for i in range(hipp_n)]
        )
        assoc_n = len(self.assoc._exc_cache)
        self.assoc.inject([min(currents[i % _cn] * 0.8, 12.0) for i in range(assoc_n)])
        # ── Semantic path: extract concepts and store in hippocampus ─────
        cs = self._consciousness
        cs._extract_concepts(text, cs._hipp_act_ema, cs._pfc_act_ema)
        workspace = cs.workspace_concepts()
        if workspace:
            em_label = (
                self.emotion_state.dominant()
                if hasattr(self, "emotion_state")
                else "neutral"
            )
            # Full semantic pipeline: hippocampus, concept_graph, beliefs, meta
            cs.record_semantic_input(workspace, self.tick_count, em_label, self)
            # Let the amygdala appraise what was injected
            pfc_goal = self.prefrontal.active_goal or ""
            self.amygdala.semantic_appraise(workspace, [pfc_goal] if pfc_goal else [])
        # Learn propositions from the original grammatical text
        cs.belief_store.learn_from_text(
            text, base_confidence=0.50, source="inject", tick=self.tick_count
        )

    # ─────────────────────────────────────────────────────────────
    # Autosave
    # ─────────────────────────────────────────────────────────────

    def _autosave(self) -> None:
        """Kick off a non-blocking background save so the tick thread never blocks.

        Takes a cheap snapshot of the synapse list reference (not a deep copy)
        and launches a daemon thread.  A lock prevents multiple concurrent
        saves from running simultaneously.
        """
        with self._save_lock:
            if self._save_in_progress:
                return  # previous save still running — skip this cycle
            self._save_in_progress = True

        def _do_save() -> None:
            try:
                from persistence import save_brain

                n = save_brain(self)
                self._last_save_n = n
            except Exception as _exc:
                import logging

                logging.getLogger("brain").warning("Autosave failed: %s", _exc)
            finally:
                with self._save_lock:
                    self._save_in_progress = False

        threading.Thread(target=_do_save, daemon=True, name="AutoSave").start()

    # ─────────────────────────────────────────────────────────
    # Communication interface
    # ─────────────────────────────────────────────────────────

    @property
    def latest_detections(self) -> List[str]:
        """Most recent object/person/gesture labels from the camera."""
        return self._vis_analyzer.detections

    @property
    def latest_detection_targets(self) -> List[Dict[str, object]]:
        """Most recent normalized detection boxes and centers from the camera."""
        return self._vis_analyzer.targets

    @property
    def latest_frame(self):
        """Latest raw BGR frame from the camera (numpy array or None)."""
        return self._latest_raw_frame

    @property
    def robot_controller_summary(self) -> str:
        return self._robot_controller.summary()

    @property
    def robot_controller_state(self) -> Dict[str, object]:
        state = self._robot_controller.snapshot()
        state["serial"] = self._robot_serial.snapshot()
        return state

    def connect_robot_serial(self, port: str, baudrate: int = 115200) -> bool:
        ok = self._robot_serial.connect(port.strip(), baudrate)
        if ok:
            self._robot_serial.send_frame(
                self._robot_controller.export_head_config_serial_frame()
            )
        return ok

    def disconnect_robot_serial(self) -> None:
        self._robot_serial.disconnect()

    def update_head_servo_config(self, code: str, **kwargs) -> bool:
        ok = self._robot_controller.update_head_servo_config(code, **kwargs)
        if ok and self._robot_serial.snapshot().get("connected"):
            self._robot_serial.send_frame(
                self._robot_controller.export_head_config_serial_frame()
            )
        return ok

    def send_head_servo_config(self) -> bool:
        frame = self._robot_controller.export_head_config_serial_frame()
        sent = self._robot_serial.send_frame(frame)
        if sent:
            self.last_robot_command = frame.strip()[:160]
        return sent

    def set_manual_head_pose(
        self,
        *,
        yaw_deg: float | None = None,
        pitch_deg: float | None = None,
        roll_deg: float | None = None,
        jaw_deg: float | None = None,
    ) -> str:
        result = self._robot_controller.set_manual_head_pose(
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            jaw_deg=jaw_deg,
        )
        frame = self._robot_controller.export_arduino_head_serial_frame(self.tick_count)
        self._robot_serial.send_frame(frame)
        self.last_robot_command = frame.strip()[:160]
        return result

    def set_head_targets(self, joint_values: Dict[str, float | None]) -> str:
        result = self._robot_controller.set_head_targets(joint_values)
        frame = self._robot_controller.export_arduino_head_serial_frame(self.tick_count)
        self._robot_serial.send_frame(frame)
        self.last_robot_command = frame.strip()[:160]
        return result

    def apply_head_preset(self, preset_name: str) -> str:
        result = self._robot_controller.apply_head_preset(preset_name)
        frame = self._robot_controller.export_arduino_head_serial_frame(self.tick_count)
        self._robot_serial.send_frame(frame)
        self.last_robot_command = frame.strip()[:160]
        return result

    @property
    def wants_to_communicate(self) -> bool:
        """True if the AI has generated an unsent outbound message."""
        return bool(self.outbound_messages)

    def get_outbound_messages(self) -> List[str]:
        """Consume and return all pending outbound messages the AI wants to send."""
        msgs = list(self.outbound_messages)
        self.outbound_messages.clear()
        return msgs

    # ─────────────────────────────────────────────────────────

    def status(self) -> str:
        lines = [
            f"[t={self.t:.0f}ms | tick={self.tick_count} | action={self.last_action} "
            f"| goal={self.consciousness_state.goal} | robot={self.last_robot_command[:48]}]"
        ]
        for name, act in self.region_activity.items():
            bar = "█" * int(act * 20)
            lines.append(f"  {name:<18} {act:.3f}  {bar}")
        if self.speech_in:
            lines.append(f"  speech: '{self.speech_in}'")
        lines.append(f"  amygdala.valence: {self.amygdala.valence:.3f}")
        # Emotion display
        lines.append("\n── EMOTION ─────────────────────────────────")
        lines.append(self.emotion_state.full_display())
        # Synapse stats
        syn_total = len(self._inter_synapses)
        lines.append(
            f"\n  synapses: {syn_total:,} active | "
            f"+{self._synapses_formed:,} formed | "
            f"{self._synapses_restored:,} restored"
        )
        # Web sensor stats
        if self._use_web and self._web_enc.last_items:
            last = self._web_enc.last_items[-1]
            lines.append(
                f"  web [{last['type']}]: {last['title'][:55]} "
                f"(q={len(self._web_enc._queue)})"
            )
            # Show topic counts
            hits = self._web_enc._topic_hits
            if hits:
                tc = ", ".join(f"{k}:{v}" for k, v in hits.most_common(4))
                lines.append(f"  web sources: {tc}")
        # Consciousness stream
        lines.append("\n── CONSCIOUSNESS ───────────────────────────")
        cs = self._consciousness
        if cs.state.thought:
            lines.append(f"  thought: {cs.state.thought}")
        if cs.state.ignition:
            lines.append(f"  GLOBAL IGNITION #{cs.ignition_count}")
        if cs.state.replaying:
            lines.append("  [MEMORY REPLAY active]")
        recent = cs.recent_conclusions
        if recent:
            lines.append(f"  last insight: {recent[-1]}")
        # Show last few stream entries
        stream = list(cs.stream)[-3:]
        for s in stream:
            lines.append(f"  > {s}")
        known = cs.known_concepts
        if known:
            lines.append(f"  known concepts ({len(known)}): {', '.join(known[-8:])}")
        return "\n".join(lines)
