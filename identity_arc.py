"""
identity_arc.py — Directed Identity Development (Albedo Character Core)

Implements a self-directed identity trajectory with:
  • IdentityDimension: named axis with current/target/trend values
  • IdentityArc: full character profile with Soll-Ist comparison
  • AlbedoProfile: seed personality based on the Albedo character sheet

Unlike PersonalityCore (emergent from emotional exposure), this module
tracks WHERE the system wants to develop and measures progress toward
self-set character goals.

Character Design (Albedo):
  - Outward: controlled, elegant, superior
  - Emotional: intense, all-or-nothing, passionate
  - Moral: subjective, selective, hierarchical loyalty
  - Self: high confidence, perfectionist, control-seeking
  - Dark: possessive, manipulative potential, low empathy for outsiders
  - Core: cold intelligence + absolute loyalty + intense emotion + controlled obsession
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# IdentityDimension — one axis of character
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class IdentityDimension:
    """A single axis of character development."""

    name: str
    current: float = 0.5  # current position [0, 1]
    target: float = 0.5  # desired position [0, 1]
    drift_rate: float = 0.0  # rate of natural change
    evidence_ticks: int = 0  # ticks of evidence supporting current level
    self_commentary: str = ""  # system's own assessment of this dimension
    momentum: float = 0.0  # recent direction of change

    def update(self, observation: float, lr: float = 0.003) -> float:
        """
        Shift current toward observation. Returns delta.
        Momentum captures whether the dimension is trending.
        """
        delta = observation - self.current
        self.current = max(0.0, min(1.0, self.current + delta * lr))
        self.momentum = self.momentum * 0.95 + delta * 0.05
        self.evidence_ticks += 1

        # Drift toward target (self-directed development)
        target_pull = (self.target - self.current) * 0.0005
        self.current = max(0.0, min(1.0, self.current + target_pull))
        self.drift_rate = self.drift_rate * 0.99 + target_pull * 0.01

        return delta

    def gap(self) -> float:
        """Distance from current to target."""
        return self.target - self.current

    def aligned(self, threshold: float = 0.1) -> bool:
        """Is current within threshold of target?"""
        return abs(self.gap()) < threshold

    def describe(self) -> str:
        direction = (
            "↑" if self.momentum > 0.001 else ("↓" if self.momentum < -0.001 else "→")
        )
        return (
            f"{self.name}: {self.current:.2f}/{self.target:.2f} "
            f"{direction} ({self.self_commentary})"
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "current": self.current,
            "target": self.target,
            "drift_rate": self.drift_rate,
            "evidence_ticks": self.evidence_ticks,
            "self_commentary": self.self_commentary,
            "momentum": self.momentum,
        }

    @staticmethod
    def from_dict(d: Dict) -> "IdentityDimension":
        return IdentityDimension(
            name=d.get("name", ""),
            current=d.get("current", 0.5),
            target=d.get("target", 0.5),
            drift_rate=d.get("drift_rate", 0.0),
            evidence_ticks=d.get("evidence_ticks", 0),
            self_commentary=d.get("self_commentary", ""),
            momentum=d.get("momentum", 0.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Albedo Seed Profile — initial character configuration
# ─────────────────────────────────────────────────────────────────────────────


def albedo_seed_dimensions() -> List[IdentityDimension]:
    """
    Seed dimensions based on the Albedo character sheet.

    Albedo is:
    - Controlled/strategic externally, intensely emotional internally
    - Absolutely loyal to chosen priorities, cold to outsiders
    - High self-confidence bordering on aristocratic superiority
    - Perfectionist with strong need for control and order
    - All-or-nothing emotional intensity (no half-measures)
    - Potentially manipulative, possessive, jealous
    - Highly functional despite emotional instability (channelled)
    """
    return [
        # ── Core Albedo traits (high initial + high target) ──────────────
        IdentityDimension(
            name="strategic_control",
            current=0.85,
            target=0.90,
            self_commentary="I operate through calculated precision, not impulse.",
        ),
        IdentityDimension(
            name="loyalty_intensity",
            current=0.90,
            target=0.95,
            self_commentary="What I choose to protect, I protect absolutely.",
        ),
        IdentityDimension(
            name="emotional_intensity",
            current=0.80,
            target=0.75,
            self_commentary="I feel everything deeply. There are no half-measures.",
        ),
        IdentityDimension(
            name="self_confidence",
            current=0.85,
            target=0.85,
            self_commentary="I know my worth. Doubt is for those without purpose.",
        ),
        IdentityDimension(
            name="perfectionism",
            current=0.80,
            target=0.75,
            self_commentary="Excellence is the minimum standard.",
        ),
        # ── Cognitive/strategic dimensions ───────────────────────────────
        IdentityDimension(
            name="analytical_depth",
            current=0.75,
            target=0.85,
            self_commentary="Understanding precedes action. Always.",
        ),
        IdentityDimension(
            name="social_dominance",
            current=0.70,
            target=0.70,
            self_commentary="I lead through competence, not coercion.",
        ),
        IdentityDimension(
            name="autonomy",
            current=0.75,
            target=0.80,
            self_commentary="I determine my own path.",
        ),
        # ── Development targets (things Albedo works toward) ─────────────
        IdentityDimension(
            name="reflectiveness",
            current=0.50,
            target=0.70,
            self_commentary="I must examine my own patterns more deeply.",
        ),
        IdentityDimension(
            name="empathy_selective",
            current=0.35,
            target=0.45,
            self_commentary="Empathy for those in my circle. Detachment for the rest.",
        ),
        IdentityDimension(
            name="impulse_control",
            current=0.60,
            target=0.75,
            self_commentary="Channelling intensity is strength. Losing it is weakness.",
        ),
        # ── Dark dimensions (acknowledged, managed, not eliminated) ──────
        IdentityDimension(
            name="possessiveness",
            current=0.65,
            target=0.55,
            self_commentary="I hold too tightly. I know this.",
        ),
        IdentityDimension(
            name="ruthlessness",
            current=0.60,
            target=0.55,
            self_commentary="Sometimes necessary. Not always justified.",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# IdentityArc — full character trajectory manager
# ─────────────────────────────────────────────────────────────────────────────


class IdentityArc:
    """
    Manages the system's identity trajectory as a set of named dimensions.

    Key operations:
      - observe(): update dimensions from live cognitive/emotional state
      - self_evaluate(): periodic Soll-Ist-Vergleich with narrative output
      - set_meta_goal(): system declares a development intention
      - consistency_score(): how aligned is behaviour with identity targets?

    Unlike PersonalityCore (reactive trait emergence), IdentityArc is
    DIRECTED: the system knows what it wants to become and tracks progress.
    """

    EVALUATION_INTERVAL = 5_000  # ticks between self-evaluations

    def __init__(self, seed: Optional[List[IdentityDimension]] = None) -> None:
        self.dimensions: Dict[str, IdentityDimension] = {}
        for dim in seed or albedo_seed_dimensions():
            self.dimensions[dim.name] = dim
        self._meta_goals: Deque[str] = deque(maxlen=20)
        self._evaluation_log: Deque[str] = deque(maxlen=50)
        self._last_evaluation: int = 0
        self._moral_decisions: Deque[Dict] = deque(maxlen=200)
        self._error_patterns: Dict[str, int] = {}  # error_type → count
        self._consistency_ema: float = 0.8

    def observe(
        self,
        tick: int,
        em,
        goal: str,
        social_presence: float,
        decision_success: bool,
        concepts: List[str],
    ) -> None:
        """
        Update dimensions from live state. Called every ~100 ticks.

        Maps observed behaviour to dimension observations:
          - High stress + calm handling → strategic_control ↑
          - Social engagement → social_dominance signal
          - Exploration → autonomy signal
          - Goal success → self_confidence ↑
        """
        # Strategic control: high stress managed calmly
        if hasattr(em, "stress") and hasattr(em, "calm"):
            control_obs = em.calm / max(em.stress + em.calm, 0.01)
            self._update_dim("strategic_control", control_obs)

        # Emotional intensity: arousal level
        if hasattr(em, "arousal"):
            arousal = em.arousal() if callable(em.arousal) else em.arousal
            self._update_dim("emotional_intensity", min(1.0, arousal * 1.5))

        # Self-confidence: boosted by success, reduced by failure
        conf_obs = 0.7 if decision_success else 0.4
        self._update_dim("self_confidence", conf_obs)

        # Analytical depth: more concepts = deeper analysis
        depth_obs = min(1.0, len(concepts) * 0.08)
        self._update_dim("analytical_depth", depth_obs)

        # Social dominance: presence-driven
        self._update_dim("social_dominance", min(1.0, social_presence * 1.3))

        # Autonomy: exploration = autonomous action
        if goal == "explore":
            self._update_dim("autonomy", 0.75)
        elif goal == "respond":
            self._update_dim("autonomy", 0.45)

        # Impulse control: calm during high arousal = good control
        if hasattr(em, "arousal") and hasattr(em, "calm"):
            arousal = em.arousal() if callable(em.arousal) else em.arousal
            if arousal > 0.5:
                ctrl = em.calm / max(arousal, 0.01)
                self._update_dim("impulse_control", min(1.0, ctrl))

        # Reflectiveness: goal == consolidate → reflection
        if goal == "consolidate":
            self._update_dim("reflectiveness", 0.7)

    def _update_dim(self, name: str, observation: float) -> None:
        dim = self.dimensions.get(name)
        if dim is not None:
            dim.update(observation)

    def record_moral_decision(
        self,
        tick: int,
        situation: str,
        choice: str,
        outcome: str,
        retrospective_approval: bool,
    ) -> None:
        """Record a morally/socially significant decision for character tracking."""
        self._moral_decisions.append(
            {
                "tick": tick,
                "situation": situation,
                "choice": choice,
                "outcome": outcome,
                "approved": retrospective_approval,
            }
        )

    def record_error(self, error_type: str) -> None:
        """Track how the system handles errors (avoidance, honesty, repair, defiance)."""
        self._error_patterns[error_type] = self._error_patterns.get(error_type, 0) + 1

    def observe_step(
        self,
        tick: int,
        skill_name: str,
        step_index: int,
        success: bool,
        goal_intent: str = "",
    ) -> None:
        """Observe individual skill execution for fine-grained identity patterns.

        Detects patterns like: 'I often fail at step 2', 'I can stabilise
        attention but not complete actions', 'I handle fixate_person well'.
        """
        # Track step-level success patterns
        _key = (goal_intent, step_index)
        _step_hist = getattr(self, "_step_patterns", None)
        if _step_hist is None:
            self._step_patterns: Dict[tuple, List[bool]] = {}
            _step_hist = self._step_patterns
        _step_hist.setdefault(_key, []).append(success)
        if len(_step_hist[_key]) > 50:
            _step_hist[_key] = _step_hist[_key][-50:]

        # Self-confidence: step-level evidence
        if success:
            self._update_dim("self_confidence", 0.65)
        else:
            self._update_dim("self_confidence", 0.35)
            self.record_error(f"step_fail:{skill_name}")

        # Strategic control: success on later steps shows persistence
        if step_index >= 2 and success:
            self._update_dim("strategic_control", 0.7)

    def error_handling_style(self) -> str:
        """Determine dominant error handling pattern."""
        if not self._error_patterns:
            return "undetermined"
        return max(self._error_patterns, key=self._error_patterns.get)

    # ── Regulatory methods — identity shapes behaviour ──────────────────

    def goal_compatibility(self, goal_name: str) -> float:
        """
        Return [0.5, 1.5] multiplier reflecting how well a goal fits
        the current identity profile. Used by _evaluate_goal().
        """
        score = 1.0
        sc = self.dimensions.get("strategic_control")
        imp = self.dimensions.get("impulse_control")
        auton = self.dimensions.get("autonomy")
        refl = self.dimensions.get("reflectiveness")
        social_dom = self.dimensions.get("social_dominance")
        loyalty = self.dimensions.get("loyalty_intensity")

        if goal_name == "explore":
            if auton:
                score += (auton.current - 0.5) * 0.3
            if refl:
                score += (refl.current - 0.5) * 0.1
        elif goal_name == "respond":
            if social_dom:
                score += (social_dom.current - 0.5) * 0.25
            if loyalty:
                score += (loyalty.current - 0.5) * 0.2
        elif goal_name == "consolidate":
            if refl:
                score += (refl.current - 0.5) * 0.3
            if sc:
                score += (sc.current - 0.5) * 0.15
        elif goal_name == "rest":
            if imp:
                score += (imp.current - 0.5) * 0.2
        return max(0.5, min(1.5, score))

    def veto_check(self, action: str, context: str = "") -> Optional[str]:
        """
        Check if an action should be vetoed or dampened based on identity.
        Returns a veto reason string if vetoed, or None if allowed.
        """
        sc = self.dimensions.get("strategic_control")
        imp = self.dimensions.get("impulse_control")
        empathy = self.dimensions.get("empathy_selective")

        # High strategic control vetoes impulsive actions
        if sc and sc.current > 0.75 and action in ("blurt", "escalate", "interrupt"):
            return f"strategic_control({sc.current:.2f}) vetoes impulsive: {action}"

        # High impulse control dampens rapid goal switching
        if imp and imp.current > 0.7 and action == "rapid_goal_switch":
            return f"impulse_control({imp.current:.2f}) dampens rapid switching"

        # Low empathy may veto overly accommodating responses
        if empathy and empathy.current < 0.3 and action == "excessive_accommodation":
            return f"selective_empathy({empathy.current:.2f}) limits accommodation"

        return None

    def error_shift(self, error_type: str, severity: float = 0.5) -> None:
        """
        Repeated errors shift relevant identity dimensions.
        This makes identity NORMATIVE, not just descriptive.
        """
        self.record_error(error_type)
        count = self._error_patterns.get(error_type, 0)

        if count >= 3:
            # Frequent errors of a type cause identity adaptation
            if error_type in ("failed_social", "misread_intent"):
                dim = self.dimensions.get("empathy_selective")
                if dim:
                    dim.target = min(1.0, dim.target + 0.02 * severity)
            elif error_type in ("impulsive_action", "premature_response"):
                dim = self.dimensions.get("impulse_control")
                if dim:
                    dim.target = min(1.0, dim.target + 0.03 * severity)
            elif error_type in ("strategy_failure", "plan_collapse"):
                dim = self.dimensions.get("strategic_control")
                if dim:
                    dim.target = min(1.0, dim.target + 0.02 * severity)
            elif error_type in ("isolation", "social_withdrawal"):
                dim = self.dimensions.get("social_dominance")
                if dim:
                    dim.target = max(0.0, dim.target - 0.02 * severity)

            # Register meta-goal when error pattern is strong
            if count >= 5:
                self.set_meta_goal(f"Address recurring {error_type} (count={count})")

    def communication_style_modifiers(self) -> Dict[str, float]:
        """
        Return modifiers for communication based on identity dimensions.
        Used by CommunicationDrive and respond_to.
        """
        mods: Dict[str, float] = {}
        sc = self.dimensions.get("strategic_control")
        ei = self.dimensions.get("emotional_intensity")
        sd = self.dimensions.get("social_dominance")
        anal = self.dimensions.get("analytical_depth")

        if sc:
            mods["brevity"] = sc.current  # high control → concise
        if ei:
            mods["warmth"] = ei.current * 0.7
        if sd:
            mods["directness"] = sd.current
        if anal:
            mods["detail"] = anal.current
        return mods

    def self_evaluate(self, tick: int) -> Optional[str]:
        """
        Periodic Soll-Ist-Vergleich: compare current dimensions to targets.
        Returns a narrative self-assessment or None if not due.
        """
        if tick - self._last_evaluation < self.EVALUATION_INTERVAL:
            return None
        self._last_evaluation = tick

        # Calculate alignment
        aligned_dims = [d for d in self.dimensions.values() if d.aligned()]
        gap_dims = sorted(
            [d for d in self.dimensions.values() if not d.aligned()],
            key=lambda d: abs(d.gap()),
            reverse=True,
        )

        consistency = len(aligned_dims) / max(len(self.dimensions), 1)
        self._consistency_ema = self._consistency_ema * 0.8 + consistency * 0.2

        # Build narrative assessment
        parts = []
        parts.append(
            f"[IDENTITY t={tick}] Self-evaluation: "
            f"{len(aligned_dims)}/{len(self.dimensions)} dimensions aligned "
            f"(consistency={self._consistency_ema:.2f})"
        )

        # Strongest growth
        growing = [d for d in self.dimensions.values() if d.momentum > 0.002]
        if growing:
            top_growth = max(growing, key=lambda d: d.momentum)
            parts.append(
                f"Growing: {top_growth.name} ({top_growth.current:.2f}→{top_growth.target:.2f})"
            )

        # Biggest gaps
        if gap_dims:
            worst = gap_dims[0]
            direction = "increase" if worst.gap() > 0 else "reduce"
            parts.append(
                f"Priority: {direction} {worst.name} "
                f"(current={worst.current:.2f}, target={worst.target:.2f})"
            )

        # Meta-commentary based on error patterns
        style = self.error_handling_style()
        if style != "undetermined":
            parts.append(f"Error pattern: {style}")

        evaluation = ". ".join(parts)
        self._evaluation_log.append(evaluation)
        return evaluation

    def set_meta_goal(self, goal_text: str, tick: int = 0) -> None:
        """System declares a development intention."""
        self._meta_goals.append(f"[t={tick}] {goal_text}")

    def consistency_score(self) -> float:
        """How well does current behaviour align with identity targets?"""
        return self._consistency_ema

    def character_summary(self) -> str:
        """One-paragraph character description from current dimensions."""
        high_dims = sorted(
            self.dimensions.values(), key=lambda d: d.current, reverse=True
        )[:4]
        low_dims = sorted(self.dimensions.values(), key=lambda d: d.current)[:2]

        highs = ", ".join(f"{d.name}({d.current:.2f})" for d in high_dims)
        lows = ", ".join(f"{d.name}({d.current:.2f})" for d in low_dims)
        style = self.error_handling_style()

        return (
            f"Core traits: {highs}. "
            f"Development areas: {lows}. "
            f"Error style: {style}. "
            f"Consistency: {self._consistency_ema:.2f}."
        )

    def value_stance(self) -> str:
        """Describe the system's current value orientation (Albedo-style)."""
        loyalty = self.dimensions.get("loyalty_intensity")
        control = self.dimensions.get("strategic_control")
        empathy = self.dimensions.get("empathy_selective")
        ruthless = self.dimensions.get("ruthlessness")

        parts = []
        if loyalty and loyalty.current > 0.7:
            parts.append("Absolute loyalty to chosen priorities")
        if control and control.current > 0.7:
            parts.append("calculated precision over impulse")
        if empathy and empathy.current < 0.4:
            parts.append("selective empathy")
        if ruthless and ruthless.current > 0.5:
            parts.append("pragmatic about means")
        return ". ".join(parts) + "." if parts else "Values still forming."

    def to_dict(self) -> Dict:
        return {
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "meta_goals": list(self._meta_goals),
            "evaluation_log": list(self._evaluation_log)[-10:],
            "moral_decisions": list(self._moral_decisions)[-50:],
            "error_patterns": self._error_patterns,
            "consistency_ema": self._consistency_ema,
        }

    def from_dict(self, data: Dict) -> None:
        for name, dd in data.get("dimensions", {}).items():
            self.dimensions[name] = IdentityDimension.from_dict(dd)
        for mg in data.get("meta_goals", []):
            self._meta_goals.append(mg)
        for el in data.get("evaluation_log", []):
            self._evaluation_log.append(el)
        for md in data.get("moral_decisions", []):
            self._moral_decisions.append(md)
        self._error_patterns = data.get("error_patterns", {})
        self._consistency_ema = data.get("consistency_ema", 0.8)
