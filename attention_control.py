"""
attention_control.py — Top-Down Attention Controller

Provides strategic, goal-driven control over the neural attention system:
  • AttentionPriority: weighted priority for a brain region or topic
  • AttentionController: manages top-down bias signals

Replaces pure bottom-up saliency with a blend of:
  - Bottom-up (sensory surprise)
  - Top-down (goal relevance, value expectation)
  - Social (ToM-driven: attend to person who needs response)
  - Homeostatic (attend to out-of-range body signals)

Integration:
  - brain.py step 13b: injects top-down bias into attention weights
  - consciousness.py: queries attention for global workspace access
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AttentionPriority:
    """A single attention target with its priority and source."""

    target: str  # region name, topic, or person_id
    weight: float  # [0, 1]
    source: str  # "goal", "value", "social", "homeostatic", "sensory"
    reason: str = ""
    decay_rate: float = 0.02  # per-tick decay
    created_tick: int = 0

    def decay(self) -> None:
        self.weight = max(0.0, self.weight - self.decay_rate)


@dataclass
class AttentionUtility:
    """Tracks the learned utility of directing attention to a target/source."""

    surprise_before: float = 0.0  # surprise level when attention was set
    surprise_after: float = 0.0  # surprise after attention window
    goal_success: float = 0.0  # goal outcome during attention
    n_samples: int = 0
    ema_utility: float = 0.0  # exponential moving average of utility


class AttentionController:
    """
    Top-down attention controller.

    Maintains a priority map that biases neural attention each tick.
    Higher-level systems (goals, values, ToM) register priorities;
    the controller blends them with bottom-up saliency.
    """

    MAX_PRIORITIES = 30
    BLEND_TOP_DOWN = 0.6  # weight of top-down vs bottom-up
    MIN_WEIGHT = 0.05  # below this, priority is removed

    def __init__(self) -> None:
        self._priorities: Dict[str, AttentionPriority] = {}
        self._total_ticks: int = 0
        # ── Learned utility per (target, source) ──
        self._utility: Dict[str, AttentionUtility] = {}  # key = "target:source"
        self._pending_eval: List[Tuple[str, str, float, int]] = (
            []
        )  # (target, source, surprise_before, tick)

    def set_priority(
        self,
        target: str,
        weight: float,
        source: str,
        reason: str = "",
        decay_rate: float = 0.02,
        tick: int = 0,
        surprise_now: float = 0.0,
    ) -> None:
        """Set or overwrite a top-down attention priority.
        If surprise_now is provided, start tracking utility for this focus."""
        if (
            len(self._priorities) >= self.MAX_PRIORITIES
            and target not in self._priorities
        ):
            # Evict weakest
            weakest = min(self._priorities, key=lambda k: self._priorities[k].weight)
            if self._priorities[weakest].weight < weight:
                del self._priorities[weakest]
            else:
                return  # don't overwrite stronger priorities

        # Apply learned utility bonus: if we know this focus helps, boost it
        _ukey = f"{target}:{source}"
        _util = self._utility.get(_ukey)
        if _util is not None and _util.n_samples >= 3:
            weight = min(1.0, weight + _util.ema_utility * 0.2)

        self._priorities[target] = AttentionPriority(
            target=target,
            weight=min(1.0, max(0.0, weight)),
            source=source,
            reason=reason,
            decay_rate=decay_rate,
            created_tick=tick,
        )

        # Register for utility evaluation
        if surprise_now > 0.01:
            self._pending_eval.append((target, source, surprise_now, tick))

    def boost(self, target: str, amount: float) -> None:
        """Boost an existing priority's weight."""
        if target in self._priorities:
            p = self._priorities[target]
            p.weight = min(1.0, p.weight + amount)

    def suppress(self, target: str, amount: float) -> None:
        """Suppress an existing priority's weight."""
        if target in self._priorities:
            p = self._priorities[target]
            p.weight = max(0.0, p.weight - amount)

    def tick(self) -> None:
        """Decay all priorities by one step."""
        self._total_ticks += 1
        expired = []
        for key, prio in self._priorities.items():
            prio.decay()
            if prio.weight < self.MIN_WEIGHT:
                expired.append(key)
        for key in expired:
            del self._priorities[key]

    def evaluate_utility(
        self,
        current_surprise: float,
        goal_success: float,
        current_tick: int,
        eval_window: int = 40,
    ) -> None:
        """Evaluate pending attention focuses: did they reduce surprise
        or contribute to goal success? Updates learned utility accordingly."""
        remaining = []
        for target, source, surprise_before, set_tick in self._pending_eval:
            if (current_tick - set_tick) < eval_window:
                remaining.append((target, source, surprise_before, set_tick))
                continue
            # Enough time has passed — evaluate
            _ukey = f"{target}:{source}"
            if _ukey not in self._utility:
                self._utility[_ukey] = AttentionUtility()
            util = self._utility[_ukey]
            util.surprise_before = surprise_before
            util.surprise_after = current_surprise
            util.goal_success = goal_success
            util.n_samples += 1
            # Utility = surprise_reduction + goal_success_bonus
            _surprise_reduction = max(0.0, surprise_before - current_surprise)
            _raw_utility = _surprise_reduction * 0.6 + goal_success * 0.4
            _alpha = 0.2 if util.n_samples > 5 else 0.4
            util.ema_utility = util.ema_utility * (1 - _alpha) + _raw_utility * _alpha
        self._pending_eval = remaining
        # Evict stale utilities (>100 entries)
        if len(self._utility) > 100:
            _sorted = sorted(self._utility.items(), key=lambda kv: kv[1].n_samples)
            for k, _ in _sorted[:20]:
                del self._utility[k]

    def get_bias(self, region_name: str) -> float:
        """
        Get top-down bias for a specific brain region.
        Returns a value in [0, 1] that should be blended with bottom-up saliency.
        """
        prio = self._priorities.get(region_name)
        return prio.weight if prio else 0.0

    def blend_attention(self, bottom_up: Dict[str, float]) -> Dict[str, float]:
        """
        Blend bottom-up saliency with top-down priorities.
        Returns merged attention map.
        """
        merged: Dict[str, float] = {}
        all_targets = set(bottom_up.keys()) | set(self._priorities.keys())
        for target in all_targets:
            bu = bottom_up.get(target, 0.0)
            td = self._priorities[target].weight if target in self._priorities else 0.0
            merged[target] = (1.0 - self.BLEND_TOP_DOWN) * bu + self.BLEND_TOP_DOWN * td
        return merged

    def top_priorities(self, n: int = 5) -> List[AttentionPriority]:
        """Return the N highest-weight priorities."""
        return sorted(self._priorities.values(), key=lambda p: -p.weight)[:n]

    def summary(self) -> str:
        top = self.top_priorities(3)
        parts = [f"{p.target}({p.weight:.2f}/{p.source})" for p in top]
        return "Attn: " + ", ".join(parts) if parts else "Attn: unfocused"

    def to_dict(self) -> Dict:
        return {
            "priorities": {
                k: {
                    "target": v.target,
                    "weight": v.weight,
                    "source": v.source,
                    "reason": v.reason[:60],
                    "decay_rate": v.decay_rate,
                    "created_tick": v.created_tick,
                }
                for k, v in self._priorities.items()
            },
            "total_ticks": self._total_ticks,
            "utility": {
                k: {"ema_utility": v.ema_utility, "n_samples": v.n_samples}
                for k, v in self._utility.items()
                if v.n_samples >= 2  # only persist meaningful data
            },
        }

    def from_dict(self, data: Dict) -> None:
        self._priorities.clear()
        for k, v in data.get("priorities", {}).items():
            self._priorities[k] = AttentionPriority(
                target=v.get("target", k),
                weight=v.get("weight", 0.0),
                source=v.get("source", "unknown"),
                reason=v.get("reason", ""),
                decay_rate=v.get("decay_rate", 0.02),
                created_tick=v.get("created_tick", 0),
            )
        self._total_ticks = data.get("total_ticks", 0)
        self._utility.clear()
        for k, v in data.get("utility", {}).items():
            self._utility[k] = AttentionUtility(
                ema_utility=v.get("ema_utility", 0.0), n_samples=v.get("n_samples", 0)
            )
