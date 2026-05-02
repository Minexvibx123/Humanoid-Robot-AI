"""
neuron.py — Biological Leaky Integrate-and-Fire (LIF) Neuron

Each neuron is a discrete unit that:
  - accumulates membrane potential from incoming spikes
  - fires an action potential when threshold is crossed
  - enters a refractory period after firing (no input accepted)
  - leaks charge over time (RC membrane model)

Spike delivery is event-driven: pre-synaptic neurons push
(weight_signed, arrival_time) tuples into this neuron's _spike_inputs queue
when they fire.  neuron.tick() sums active conductances in O(queue_size)
instead of O(N_afferents), giving ~20x speedup at 13k neuron scale.

STDP is applied inline at fire-time using neuron.trace (per-neuron eligibility
trace) — no per-synapse trace state needed.

NO probability distributions. Firing is deterministic based on physics.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from synapse import Synapse


# ─────────────────────────────────────────────────────────────
# Membrane constants (biologically inspired, scaled for simulation)
# ─────────────────────────────────────────────────────────────
V_REST = -70.0  # mV  resting membrane potential
V_THRESHOLD = -55.0  # mV  action potential threshold
V_SPIKE = 40.0  # mV  peak of action potential
V_RESET = -75.0  # mV  after-spike hyperpolarisation
TAU_M = 20.0  # ms  membrane time constant (leak rate)
T_REFRAC = 2.0  # ms  absolute refractory period

# ─────────────────────────────────────────────────────────────
# Event-driven conductance decay table
# exp(-n / TAU_SYN) for n = 0..25 ticks, TAU_SYN = 5 ms
# After 25 ticks the conductance is ~0.7% of peak — negligible.
# ─────────────────────────────────────────────────────────────
_TAU_SYN = 5.0
_MAX_SYN_AGE = 25
_SYN_DECAY_TABLE: List[float] = [
    math.exp(-n / _TAU_SYN) for n in range(_MAX_SYN_AGE + 1)
]

# ─────────────────────────────────────────────────────────────
# STDP parameters (applied inline in tick for performance)
# ─────────────────────────────────────────────────────────────
A_PLUS = 0.01  # LTP magnitude per unit pre-trace (externally modulated by emotion)
_A_MINUS = 0.012  # LTD magnitude per unit post-trace
_W_MIN = 0.0
_W_MAX = 50.0
_TRACE_MIN = 0.30  # trace threshold: ~24ms STDP window (exp(-24/20)=0.30)


class Neuron:
    """
    Leaky Integrate-and-Fire neuron.

    dt  = simulation timestep in ms (default 1 ms)
    """

    _id_counter = 0
    # Global tonic bias current (nA) added to every neuron every tick.
    # Models persistent background excitatory drive from brainstem / thalamus.
    # Set via Neuron.global_tonic_current = X in Brain.__init__.
    global_tonic_current: float = 0.0

    def __init__(
        self,
        region: str = "cortex",
        neuron_type: str = "excitatory",  # "excitatory" | "inhibitory"
        dt: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> None:
        Neuron._id_counter += 1
        self.nid: int = Neuron._id_counter
        self.region: str = region
        self.neuron_type: str = neuron_type
        self.dt: float = dt  # ms per tick
        self.x: float = float(x)
        self.y: float = float(y)
        self.z: float = float(z)

        # Membrane state
        self.v: float = V_REST  # current membrane potential (mV)
        self.fired: bool = False  # did we spike this tick?
        self._refrac_remaining: float = 0.0  # ms of refractory period left

        # Connectivity (filled by Synapse objects)
        self.afferents: List["Synapse"] = []  # incoming synapses
        self.efferents: List["Synapse"] = []  # outgoing synapses

        # Event-driven spike queue: list of (weight_signed, arrival_time) tuples.
        # Pre-synaptic neurons push here when they fire; neuron.tick() consumes.
        self._spike_inputs: list = []

        # Spike history (capped at 100 to prevent unbounded growth)
        self.spike_times: List[float] = []

        # Trace for STDP (eligibility) — decays each tick, reset to 1.0 on spike.
        # When _np_region_trace is set by the Region, .trace is backed by the
        # numpy array so numba-computed traces are visible without an O(N) sync.
        self._trace: float = 0.0
        self._np_region_trace: Optional[np.ndarray] = None  # set by Region.__init__
        self._np_region_idx: int = 0  # local index in region array
        self.trace_tau: float = 20.0  # ms decay constant for STDP trace

    # ── trace property — backed by numpy array when region uses numba ─────────
    @property
    def trace(self) -> float:
        if self._np_region_trace is not None:
            return float(self._np_region_trace[self._np_region_idx])
        return self._trace

    @trace.setter
    def trace(self, val: float) -> None:
        self._trace = val
        if self._np_region_trace is not None:
            self._np_region_trace[self._np_region_idx] = val

    # ─────────────────────────────────────────────────────────
    # Core tick — call once per simulation step
    # ─────────────────────────────────────────────────────────

    def tick(self, t: float, external_current: float = 0.0) -> bool:
        """
        Advance neuron by one timestep dt.

        t                : current simulation time in ms
        external_current : injected current in nA (from sensors etc.)

        Returns True if the neuron fired this tick.
        """
        self.fired = False

        # ── Refractory period ──────────────────────────────
        if self._refrac_remaining > 0.0:
            self._refrac_remaining -= self.dt
            self.v = V_RESET
            self.trace *= 0.9512  # inline trace decay (exp(-1/20))
            # Expire stale spike inputs to keep the queue lean
            if self._spike_inputs:
                tmin = t - _MAX_SYN_AGE
                self._spike_inputs = [
                    (w, ta) for w, ta in self._spike_inputs if ta > tmin
                ]
            return False

        # ── Sum event-driven conductances ─────────────────
        # Each entry: (weight_signed, arrival_time)
        # Conductance = weight_signed * exp(-(t - ta) / TAU_SYN)
        I_syn = 0.0
        if self._spike_inputs:
            _decay = _SYN_DECAY_TABLE
            _max = _MAX_SYN_AGE
            keep = []
            for w, ta in self._spike_inputs:
                age = t - ta
                if age < 0.0:
                    keep.append((w, ta))  # not arrived yet
                elif age <= _max:
                    I_syn += w * _decay[int(age)]
                    keep.append((w, ta))  # still active
                # else: expired — drop
            self._spike_inputs = keep

        I_total = I_syn + external_current + Neuron.global_tonic_current

        # ── Leaky integration (Euler) ──────────────────────
        # dV/dt = (-(V - V_rest) + R·I) / tau_m
        # R assumed = 1 MOhm (unit simplification)
        dv = (-(self.v - V_REST) + I_total) * (self.dt / TAU_M)
        self.v += dv

        # ── Threshold crossing → spike ─────────────────────
        if self.v >= V_THRESHOLD:
            self.v = V_SPIKE
            self.fired = True
            self.spike_times.append(t)
            if len(self.spike_times) > 100:
                del self.spike_times[:-100]
            self._refrac_remaining = T_REFRAC
            self.trace = 1.0

            # Push spike event into each post-neuron's queue
            # + STDP LTD (pre fires now, post fired recently → weaken)
            for syn in self.efferents:
                syn.post._spike_inputs.append((syn.weight * syn._sign, t + syn.delay))
                _pt = syn.post.trace
                if _pt > _TRACE_MIN:
                    _w = syn.weight - _A_MINUS * _pt
                    syn.weight = _w if _w > _W_MIN else _W_MIN

            # STDP LTP (post fires now = self, pre fired recently → strengthen)
            for syn in self.afferents:
                _prt = syn.pre.trace
                if _prt > _TRACE_MIN:
                    _w = syn.weight + A_PLUS * _prt
                    syn.weight = _w if _w < _W_MAX else _W_MAX
        else:
            self.trace *= 0.9512  # inline trace decay (exp(-1/20))

        # Clamp to biological range
        self.v = max(V_RESET - 5.0, min(self.v, V_SPIKE))
        return self.fired

    # ─────────────────────────────────────────────────────────
    # STDP trace
    # ─────────────────────────────────────────────────────────

    def _decay_trace(self) -> None:
        # Use same 20ms tau as STDP; import constant lazily to avoid circular dep
        self.trace *= 0.9512  # exp(-1/20) pre-computed for dt=1ms

    # ─────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Reset membrane to resting state (keep learned weights)."""
        self.v = V_REST
        self.fired = False
        self._refrac_remaining = 0.0
        self.trace = 0.0

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def set_position(self, x: float, y: float, z: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance_to(self, other: "Neuron") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def firing_rate(self, window_ms: float = 1000.0, t_now: float = 0.0) -> float:
        """Mean firing rate (Hz) over the last window_ms milliseconds."""
        cutoff = t_now - window_ms
        recent = [s for s in self.spike_times if s >= cutoff]
        return len(recent) / (window_ms / 1000.0)

    def __repr__(self) -> str:
        return (
            f"Neuron(id={self.nid}, region={self.region}, "
            f"type={self.neuron_type}, V={self.v:.1f}mV, fired={self.fired})"
        )
