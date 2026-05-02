"""
emotion.py — Appraisal-Based Emotion Engine

Emotions are derived from WHAT is being processed (semantic appraisal),
not from HOW MANY neurons fired. Based on Scherer's (2001) Component
Process Model and the OCC model (Ortony, Clore, Collins 1988):

  Appraisal dimension    → Emotion triggered
  ─────────────────────────────────────────────────────────────────────
  Reward + goal-relevant  → joy
  Novelty + curiosity     → curiosity
  Low threat + low stress → calm
  Threat + low agency     → stress / fear
  Negative + irreversible → sadness (lingering)
  Threat + high agency    → anger (mobilising)
  High novelty            → surprise
  Sustained high load     → fatigue

Personality modulates all appraisals: a curious personality amplifies
curiosity from novelty; an alert personality amplifies stress from threat.

The neural substrate (LIF firing rates) is used only as a FALLBACK
when no semantic concepts are available (cold-start / silent ticks).

8-dimensional state:  joy, stress, curiosity, calm, sadness, anger,
                      surprise, fatigue  + reward_prediction_error
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from brain import Brain


@dataclass
class EmotionalState:
    joy: float = 0.0
    stress: float = 0.0
    curiosity: float = 0.0
    calm: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    surprise: float = 0.0
    fatigue: float = 0.0
    reward_prediction_error: float = 0.0

    # ── Derived properties ────────────────────────────────────

    def dominant(self) -> str:
        d = {
            "joy": self.joy,
            "stress": self.stress,
            "curiosity": self.curiosity,
            "calm": self.calm,
            "sadness": self.sadness,
            "anger": self.anger,
            "surprise": self.surprise,
            "fatigue": self.fatigue,
        }
        return max(d, key=d.get)

    def valence(self) -> float:
        """Overall hedonic valence in [-1, +1]."""
        pos = self.joy + self.curiosity * 0.7 + self.calm * 0.5
        neg = self.stress + self.sadness + self.anger
        return math.tanh(pos - neg)

    def arousal(self) -> float:
        """Overall activation level in [0, 1]."""
        return min(
            1.0,
            self.joy * 0.5
            + self.stress
            + self.curiosity * 0.6
            + self.surprise
            + self.anger * 0.8,
        )

    def describe(self) -> str:
        dom = self.dominant()
        a = self.arousal()
        lvl = (
            "slight"
            if a < 0.2
            else "moderate" if a < 0.5 else "strong" if a < 0.8 else "intense"
        )
        phrases = {
            "joy": f"{lvl} joy — positive, engaged",
            "stress": f"{lvl} stress — vigilant, alert",
            "curiosity": f"{lvl} curiosity — exploring, seeking",
            "calm": f"{lvl} calm — clear, composed",
            "sadness": f"{lvl} sadness — reflective, processing",
            "anger": f"{lvl} frustration — activated, reactive",
            "surprise": f"{lvl} surprise — novelty detected",
            "fatigue": f"{lvl} fatigue — resting, consolidating",
        }
        return phrases.get(dom, "neutral")

    # ── Neuromodulatory multipliers ───────────────────────────

    def ltp_modulation(self) -> float:
        """Dopamine-like LTP rate multiplier [0.2, 3.5]."""
        base = (
            1.0
            + self.joy * 1.6
            + self.curiosity * 1.0
            + self.stress * 0.25
            - self.fatigue * 0.7
            - self.sadness * 0.3
        )
        return max(0.2, min(3.5, base))

    def growth_rate(self) -> float:
        """Synapse growth rate multiplier [0.2, 2.8]."""
        base = (
            1.0
            + self.curiosity * 1.3
            + self.joy * 0.5
            - self.fatigue * 0.8
            - self.sadness * 0.2
        )
        return max(0.2, min(2.8, base))

    def fetch_urgency(self) -> float:
        """Web-sensor fetch urgency multiplier [0.4, 3.5]."""
        return max(0.4, min(3.5, 0.5 + self.curiosity * 2.5 + self.surprise * 1.0))

    # ── Display ───────────────────────────────────────────────

    @staticmethod
    def _bar(val: float, width: int = 10) -> str:
        filled = int(round(max(0.0, min(1.0, val)) * width))
        return "█" * filled + "░" * (width - filled)

    def full_display(self) -> str:
        rows = [
            ("joy", self.joy),
            ("stress", self.stress),
            ("curiosity", self.curiosity),
            ("calm", self.calm),
            ("sadness", self.sadness),
            ("anger", self.anger),
            ("surprise", self.surprise),
            ("fatigue", self.fatigue),
        ]
        lines = []
        for name, val in rows:
            lines.append(f"  {name:<10} {self._bar(val)} {val:.2f}")
        lines.append(f"  → {self.describe()}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Emotion Engine — appraisal-based, not activity-rate-based
# ─────────────────────────────────────────────────────────────


class ExperienceAppraisal:
    """
    Learns concept->emotion associations from experience.
    Instead of relying solely on hardcoded lexicons, this module builds
    associations between concepts and emotional outcomes over time.
    When a concept co-occurs with a strong emotional state, the association
    is strengthened. Enables learning new emotionally relevant categories.
    """

    MAX_CONCEPTS = 5000
    LEARN_RATE = 0.08
    DECAY_RATE = 0.9995
    MIN_OBS_FOR_CONFIDENCE = 5  # observations before concept is trusted

    def __init__(self) -> None:
        self._associations: Dict[str, Dict[str, float]] = {}
        self._observation_count: Dict[str, int] = {}
        self._variance: Dict[str, float] = {}  # running variance per concept

    def observe(self, concepts: List[str], emotion: "EmotionalState") -> None:
        """Learn from co-occurrence of concepts with current emotional state."""
        if not concepts:
            return
        arousal = emotion.arousal()
        if arousal < 0.15:
            return
        em_vec = {
            "threat": emotion.stress * 0.6 + emotion.anger * 0.4,
            "reward": emotion.joy * 0.7 + emotion.curiosity * 0.3,
            "novelty": emotion.surprise * 0.8 + emotion.curiosity * 0.2,
            "social": max(emotion.joy, emotion.sadness) * 0.5,
        }
        lr = self.LEARN_RATE * min(1.0, arousal * 2.0)
        for concept in concepts[:10]:
            c = concept.lower()
            if c not in self._associations:
                if len(self._associations) >= self.MAX_CONCEPTS:
                    least = min(
                        self._observation_count, key=self._observation_count.get
                    )
                    del self._associations[least]
                    del self._observation_count[least]
                self._associations[c] = {
                    "threat": 0.0,
                    "reward": 0.0,
                    "novelty": 0.0,
                    "social": 0.0,
                }
                self._observation_count[c] = 0
            assoc = self._associations[c]
            self._observation_count[c] = self._observation_count.get(c, 0) + 1
            # Track variance: mean squared deviation from current association
            delta_sq = sum((em_vec[d] - assoc[d]) ** 2 for d in em_vec) / len(em_vec)
            prev_var = self._variance.get(c, 0.5)
            self._variance[c] = prev_var * 0.9 + delta_sq * 0.1
            for dim, val in em_vec.items():
                assoc[dim] = assoc[dim] * (1 - lr) + val * lr

    def concept_confidence(self, concept: str) -> float:
        """Per-concept confidence [0,1] based on observations and stability."""
        c = concept.lower()
        n = self._observation_count.get(c, 0)
        if n < self.MIN_OBS_FOR_CONFIDENCE:
            return 0.0
        # Sigmoid ramp: confident after ~20 observations
        obs_factor = min(1.0, (n - self.MIN_OBS_FOR_CONFIDENCE) / 15.0)
        # Variance penalty: high variance = low confidence
        var = self._variance.get(c, 0.5)
        stability = max(0.0, 1.0 - var * 4.0)
        return obs_factor * stability

    def appraise(self, concepts: List[str]) -> Dict[str, float]:
        """Return learned emotional appraisal for concepts."""
        result = {"threat": 0.0, "reward": 0.0, "novelty": 0.0, "social": 0.0}
        if not concepts:
            return result
        n = 0
        for concept in concepts:
            c = concept.lower()
            assoc = self._associations.get(c)
            if assoc:
                for dim in result:
                    result[dim] += assoc[dim]
                n += 1
        if n > 0:
            for dim in result:
                result[dim] = min(1.0, result[dim] / n)
        return result

    def decay(self) -> None:
        """Slow decay of unused associations."""
        for assoc in self._associations.values():
            for dim in assoc:
                assoc[dim] *= self.DECAY_RATE

    def known_concepts(self) -> int:
        return len(self._associations)

    def strongest_associations(self, n: int = 10) -> List[tuple]:
        """Return top N concepts by total association strength."""
        scored = []
        for c, assoc in self._associations.items():
            strength = sum(abs(v) for v in assoc.values())
            scored.append((c, strength, dict(assoc)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]


class EmotionEngine:
    """
    Computes the 8D EmotionalState each tick via semantic appraisal.

    PRIMARY PATH (when workspace concepts are available):
      1. Get active concepts from the consciousness workspace
      2. Amygdala.semantic_appraise() → threat/reward/novelty/social/agency
      3. Personality weights modulate each appraisal dimension
      4. Body state adds fatigue/stress/calibration biases
      5. EMA smoothing produces gradual emotional transitions

    FALLBACK PATH (no concepts available, e.g. cold-start):
      Use region firing rates — kept from the legacy implementation.
      This prevents dead emotion during the first few hundred ticks
      before the semantic pipeline has extracted any concepts.
    """

    _EMA = 0.06  # emotion time constant (~17-tick rise/fall)
    _ACT_EMA = 0.20  # activity EMA for fallback path

    def __init__(self) -> None:
        self.state = EmotionalState()
        self._prev_assoc: float = 0.0
        self._active_ticks: int = 600
        self._tick_count: int = 0
        self._act_ema: dict = {}
        self._valence_ema: float = 0.0
        self.experience = ExperienceAppraisal()

    # ─────────────────────────────────────────────────────────

    def tick(self, brain: "Brain") -> EmotionalState:
        """Derive emotional state. Always uses semantic appraisal path.

        When no workspace concepts exist the appraisal returns near-zero
        values — the correct degraded state, not a legacy heuristic surrogate.
        """
        self._tick_count += 1

        # Get workspace concepts from consciousness
        concepts: List[str] = []
        cs = getattr(brain, "_consciousness", None)
        if cs is not None:
            concepts = cs.workspace_concepts()

        return self._appraisal_tick(brain, concepts, cs)

    # ─────────────────────────────────────────────────────────
    # PRIMARY: Appraisal-based emotion computation
    # ─────────────────────────────────────────────────────────

    def _appraisal_tick(
        self,
        brain: "Brain",
        concepts: List[str],
        cs: object,
    ) -> EmotionalState:
        """
        Derive emotion from what is being processed, not from firing rates.
        OCC/CPM appraisal: threat × agency × relevance × novelty → emotion.
        """
        # ── Step 1: get amygdala appraisal ────────────────────────────
        goal_str = getattr(brain.consciousness_state, "goal", "") or ""
        pfc_goal = getattr(brain.prefrontal, "active_goal", "") or ""
        goals = [g for g in (goal_str, pfc_goal) if g]

        appraisal = brain.amygdala.semantic_appraise(concepts, goals)

        # Blend hardcoded lexicon appraisal with learned experience
        # Per-concept confidence: only well-learned concepts influence blend
        exp = self.experience.appraise(concepts)
        total_conf = 0.0
        n_conf = 0
        for c in concepts:
            conf = self.experience.concept_confidence(c)
            if conf > 0.0:
                total_conf += conf
                n_conf += 1
        avg_conf = (total_conf / n_conf) if n_conf > 0 else 0.0
        learn_w = min(0.4, avg_conf * 0.4)  # max 40% from learned, scaled by evidence
        base_w = 1.0 - learn_w

        threat = appraisal.get("threat", 0.0) * base_w + exp["threat"] * learn_w
        reward = appraisal.get("reward", 0.0) * base_w + exp["reward"] * learn_w
        novelty = appraisal.get("novelty", 0.0) * base_w + exp["novelty"] * learn_w
        social = appraisal.get("social", 0.0) * base_w + exp["social"] * learn_w
        agency = appraisal.get("agency", 0.0)
        relevance = appraisal.get("relevance", 0.1)
        valence = appraisal.get("valence", 0.0)

        # ── Step 2: personality modulation ────────────────────────────
        # Personality grows from accumulated emotional exposure (not preset)
        def pt(name: str) -> float:
            return min(
                1.0,
                getattr(cs, "personality", None)
                and cs.personality._exposure.get(name, 0.0) * 2.0
                or 0.0,
            )

        # ── Step 3: appraisal dimensions → emotion targets ───────────
        # Joy: reward + goal-relevance, amplified by joy-trait
        t_joy = min(1.0, reward * 0.55 + relevance * 0.20 + social * 0.15) * (
            0.6 + pt("joy") * 0.8
        )

        # Curiosity: novelty + information hunger + curiosity trait
        drives = getattr(cs, "drives", None)
        info_hunger = drives.information_hunger if drives else 0.3
        t_curiosity = min(1.0, novelty * 0.45 + reward * 0.15 + info_hunger * 0.25) * (
            0.6 + pt("curiosity") * 0.8
        )

        # Stress: threat × (1 - coping_capacity)
        body = getattr(cs, "body", None)
        energy = body.energy_reserve if body else 0.7
        coping = min(1.0, agency * 0.6 + energy * 0.4)
        t_stress = min(1.0, threat * 0.75 * max(0.1, 1.0 - coping)) * (
            0.7 + pt("stress") * 0.6
        )

        # Calm: absence of threat/stress + coherence
        t_calm = min(
            1.0, (1.0 - threat) * (1.0 - t_stress * 0.5) * (0.4 + energy * 0.4)
        ) * (0.5 + pt("calm") * 0.8)

        # Sadness: threat without agency + lingering negative valence
        t_sadness = min(
            1.0, max(0.0, threat - agency) * 0.45 + max(0.0, -valence) * 0.25
        ) * (0.5 + pt("sadness") * 0.7)

        # Anger: threat + agency (mobilised response)
        t_anger = min(1.0, threat * agency * 0.55) * (0.5 + pt("anger") * 0.7)

        # Surprise: novelty spike
        t_surprise = min(1.0, novelty * 0.65) * (0.6 + pt("surprise") * 0.6)

        # Fatigue: body and cognitive load
        error_risk = body.error_risk if body else 0.2
        t_fatigue = min(1.0, (1.0 - energy) * 0.5 + error_risk * 0.25)

        # ── Step 4: body homeostatic biases ──────────────────────────
        if body is not None:
            biases = body.emotion_bias()
            t_fatigue = min(1.0, t_fatigue + biases.get("fatigue", 0.0))
            t_stress = min(1.0, t_stress + biases.get("stress", 0.0))
            t_calm = max(0.0, t_calm + biases.get("calm", 0.0))
            t_joy = max(0.0, t_joy + biases.get("joy", 0.0))

        # ── Step 5: EMA towards targets ──────────────────────────────
        e = self._EMA
        s = self.state
        new_state = EmotionalState(
            joy=s.joy * (1 - e) + t_joy * e,
            stress=s.stress * (1 - e) + t_stress * e,
            curiosity=s.curiosity * (1 - e) + t_curiosity * e,
            calm=s.calm * (1 - e) + t_calm * e,
            sadness=s.sadness * (1 - e) + t_sadness * e,
            anger=s.anger * (1 - e) + t_anger * e,
            surprise=s.surprise * (1 - e) + t_surprise * e,
            fatigue=s.fatigue * (1 - e) + t_fatigue * e,
        )

        # RPE: signed valence change (dopamine-like signal)
        new_state.reward_prediction_error = new_state.valence() - s.valence()
        self.state = new_state

        # Learn from this tick: associate active concepts with resulting emotion
        if concepts:
            self.experience.observe(concepts, self.state)

            # Periodic memory decay (every 100 ticks)
            self._exp_tick = getattr(self, "_exp_tick", 0) + 1
            if self._exp_tick % 100 == 0:
                self.experience.decay()

        return self.state


# (Legacy activity-rate fallback removed — emotion is always grounded in
#  semantic content.  No concepts ⇒ neutral/degraded state, not heuristics.)
