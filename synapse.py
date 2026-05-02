"""
synapse.py — Biological Synapse (event-driven)

Each synapse stores weight, delay, and sign. Spike delivery is event-driven:
when a pre-synaptic neuron fires it pushes (conductance, arrival_time) tuples
directly into the post-synaptic neuron's _spike_inputs queue (see neuron.py).

No per-tick synapse.tick() loop needed — eliminates the O(N_synapses) bottleneck
and replaces it with O(N_spikes_this_tick) work.

STDP (LTP / LTD) is applied inline in neuron.tick() using the neuron-level
eligibility trace (neuron.trace), which is equivalent to the per-synapse trace
for single-synapse pairs and avoids per-synapse state allocation.

Learning rule:
  post fires AFTER  pre  → LTP (strengthen)
  pre  fires AFTER  post → LTD (weaken)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuron import Neuron


W_MIN = 0.0  # minimum synaptic weight
W_MAX = 50.0  # maximum synaptic weight (mV-equivalent conductance)


class Synapse:
    """
    A directed connection: pre → post.

    Stores weight, delay, and sign.  Spike delivery is event-driven via
    post._spike_inputs — no tick() needed.
    """

    _id_counter = 0

    def __init__(
        self,
        pre: "Neuron",
        post: "Neuron",
        weight: float = 0.5,
        delay: float = 1.0,
    ) -> None:
        Synapse._id_counter += 1
        self.sid: int = Synapse._id_counter
        self.pre = pre
        self.post = post
        self.weight: float = max(W_MIN, min(W_MAX, weight))
        self.delay: float = delay  # ms axonal delay
        self.distance: float = pre.distance_to(post)

        # Pre-computed sign: +1 excitatory, -1 inhibitory
        self._sign: float = 1.0 if pre.neuron_type == "excitatory" else -1.0

        pre.efferents.append(self)
        post.afferents.append(self)

    # ──────────────────────────────────────────────────────────
    # Structural plasticity queries (used by brain._structural_plasticity)
    # ──────────────────────────────────────────────────────────

    def is_potentiated(self, threshold: float = 4.0) -> bool:
        """True if weight is near maximum — candidate to sprout a sibling."""
        return self.weight >= threshold

    def is_depressed(self, threshold: float = 0.05) -> bool:
        """True if weight is near zero — candidate for pruning."""
        return self.weight <= threshold

    def __repr__(self) -> str:
        return (
            f"Synapse(id={self.sid}, "
            f"pre={self.pre.nid}→post={self.post.nid}, "
            f"w={self.weight:.3f}, d={self.delay:.1f}ms, dist={self.distance:.2f})"
        )
