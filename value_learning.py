"""
value_learning.py — Temporal-Difference Value Learning

Learns long-term state value estimates through TD(0) updates:
  • ValueEstimate: per-state-cluster value with confidence
  • ValueModel: maps state signatures to expected future reward
  • TD update after every tick or completed skill

Unlike momentary reward prediction error (emotion.py), this module
builds stable VALUE FUNCTIONS over state space — enabling the system
to invest effort now for future benefit.

Integration:
  - consciousness.py: goal evaluation uses value predictions
  - task_executive.py: skill selection weighted by state values
  - persistence.py: serialises value model to SQLite
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# ValueEstimate — learned value for a state cluster
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ValueEstimate:
    """Value estimate for a state cluster."""

    value: float = 0.0  # estimated long-term reward
    confidence: float = 0.0  # how many updates back this estimate
    update_count: int = 0  # total TD updates received
    last_tick: int = 0  # tick of most recent update
    trend: float = 0.0  # EMA of value change direction

    def td_update(
        self,
        reward: float,
        next_value: float,
        gamma: float = 0.95,
        alpha: float = 0.1,
        tick: int = 0,
    ) -> float:
        """
        TD(0) update: V(s) ← V(s) + α[r + γV(s') - V(s)]
        Returns the TD error (surprise signal).
        """
        td_error = reward + gamma * next_value - self.value
        self.value += alpha * td_error
        self.update_count += 1
        self.confidence = min(1.0, self.update_count / 50.0)
        self.last_tick = tick
        # Track trend direction
        self.trend = self.trend * 0.9 + td_error * 0.1
        return td_error

    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "update_count": self.update_count,
            "last_tick": self.last_tick,
            "trend": self.trend,
        }

    @staticmethod
    def from_dict(d: Dict) -> "ValueEstimate":
        return ValueEstimate(
            value=d.get("value", 0.0),
            confidence=d.get("confidence", 0.0),
            update_count=d.get("update_count", 0),
            last_tick=d.get("last_tick", 0),
            trend=d.get("trend", 0.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ValueModel — the full value function approximation
# ─────────────────────────────────────────────────────────────────────────────


class ValueModel:
    """
    Maps state signatures to long-term value estimates via TD learning.

    State signatures are compact string descriptors (e.g. "goal:explore|emo:curious|social:0").
    The model learns which states lead to high cumulative reward over time.

    Used by:
      - Goal evaluation: prefer goals leading to high-value states
      - Action selection: weight actions by expected state value improvement
      - Self-model: track whether the system is generally improving
    """

    MAX_STATES = 10_000
    GAMMA = 0.95  # discount factor
    ALPHA = 0.08  # learning rate
    DECAY = 0.9999  # slow decay of unused state values

    def __init__(self) -> None:
        self._values: Dict[str, ValueEstimate] = {}
        self._recent_td_errors: Deque[float] = deque(maxlen=200)
        self._global_value_trend: float = 0.0
        self._last_state: Optional[str] = None
        self._last_reward: float = 0.0

    def state_value(self, state_sig: str) -> Tuple[float, float]:
        """Return (value, confidence) for a state. Unknown states → (0, 0)."""
        est = self._values.get(state_sig)
        if est is not None:
            return est.value, est.confidence
        return 0.0, 0.0

    def update(
        self, state_sig: str, reward: float, next_state_sig: str, tick: int = 0
    ) -> float:
        """
        TD(0) update for transitioning from state_sig to next_state_sig.
        Returns the TD error.
        """
        next_val, _ = self.state_value(next_state_sig)

        if state_sig not in self._values:
            if len(self._values) >= self.MAX_STATES:
                # Evict least-updated state
                worst = min(self._values, key=lambda k: self._values[k].update_count)
                del self._values[worst]
            self._values[state_sig] = ValueEstimate()

        td_error = self._values[state_sig].td_update(
            reward, next_val, self.GAMMA, self.ALPHA, tick
        )

        self._recent_td_errors.append(td_error)
        self._global_value_trend = self._global_value_trend * 0.95 + td_error * 0.05
        self._last_state = state_sig
        self._last_reward = reward
        return td_error

    def step(self, state_sig: str, reward: float, tick: int = 0) -> float:
        """
        Simplified step: automatically uses last state as predecessor.
        Call once per tick with current state signature and instantaneous reward.
        """
        if self._last_state is not None:
            td = self.update(self._last_state, self._last_reward, state_sig, tick)
        else:
            td = 0.0
        self._last_state = state_sig
        self._last_reward = reward
        return td

    def best_next_state(self, candidates: List[str]) -> Tuple[str, float]:
        """Choose the highest-value state from candidates."""
        if not candidates:
            return "", 0.0
        best_sig = candidates[0]
        best_val = -999.0
        for sig in candidates:
            val, _ = self.state_value(sig)
            if val > best_val:
                best_val = val
                best_sig = sig
        return best_sig, best_val

    def advantage(self, state_sig: str, action_state_sig: str) -> float:
        """Value advantage of taking an action: V(next) - V(current)."""
        current_val, _ = self.state_value(state_sig)
        next_val, _ = self.state_value(action_state_sig)
        return next_val - current_val

    @property
    def mean_td_error(self) -> float:
        if not self._recent_td_errors:
            return 0.0
        return sum(self._recent_td_errors) / len(self._recent_td_errors)

    @property
    def value_trend(self) -> float:
        """Positive = improving over time, negative = declining."""
        return self._global_value_trend

    def improving(self) -> bool:
        """Is the system generally moving toward better states?"""
        return self._global_value_trend > 0.01

    def top_states(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return the N highest-valued states."""
        ranked = sorted(self._values.items(), key=lambda kv: kv[1].value, reverse=True)
        return [(k, v.value) for k, v in ranked[:n]]

    def bottom_states(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return the N lowest-valued states (to avoid)."""
        ranked = sorted(self._values.items(), key=lambda kv: kv[1].value)
        return [(k, v.value) for k, v in ranked[:n]]

    def decay(self) -> None:
        """Slowly decay unused state values."""
        to_remove = []
        for key, est in self._values.items():
            est.value *= self.DECAY
            if abs(est.value) < 0.001 and est.update_count < 3:
                to_remove.append(key)
        for key in to_remove:
            del self._values[key]

    def summarise(self) -> str:
        n = len(self._values)
        if n == 0:
            return "value_model: no data"
        vals = [v.value for v in self._values.values()]
        avg = sum(vals) / n
        trend = "improving" if self.improving() else "stable/declining"
        return (
            f"value_model: {n} states, avg={avg:.3f}, "
            f"trend={trend}, td_err={self.mean_td_error:.3f}"
        )

    def to_dict(self) -> Dict:
        return {
            "values": {k: v.to_dict() for k, v in self._values.items()},
            "global_trend": self._global_value_trend,
        }

    def from_dict(self, data: Dict) -> None:
        for key, vd in data.get("values", {}).items():
            self._values[key] = ValueEstimate.from_dict(vd)
        self._global_value_trend = data.get("global_trend", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# State signature builder — creates compact state descriptors
# ─────────────────────────────────────────────────────────────────────────────


def build_state_signature(
    goal: str,
    dominant_emotion: str,
    social_presence: float,
    energy: float,
    n_concepts: int,
    focus_region: str = "",
) -> str:
    """
    Build a compact state signature for value learning.
    Discretises continuous values to keep state space manageable.
    """
    social_bin = (
        "hi" if social_presence > 0.5 else ("lo" if social_presence < 0.15 else "mid")
    )
    energy_bin = "hi" if energy > 0.6 else ("lo" if energy < 0.3 else "mid")
    concept_bin = "many" if n_concepts > 10 else ("few" if n_concepts < 3 else "some")
    focus_short = focus_region[:8] if focus_region else "none"

    return f"g:{goal}|e:{dominant_emotion}|s:{social_bin}|nrg:{energy_bin}|c:{concept_bin}|f:{focus_short}"
