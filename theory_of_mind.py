"""
theory_of_mind.py — Mental Model of Other Agents

Models the inferred internal states of other agents:
  • MentalModel: per-person belief/goal/preference/knowledge estimates
  • TheoryOfMind: aggregated social inference engine

Enables:
  - Strategic communication (adapt message style to inferred preferences)
  - Empathic response (detect likely emotional state of interlocutor)
  - Trust repair (track violations and repair attempts)
  - Perspective taking (what does person X know/want?)

Integration:
  - social_manager.py: feeds interaction data into mental models
  - consciousness.py: uses ToM for communication decisions
  - persistence.py: serialises mental models
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# MentalModel — inferred mind of one other agent
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MentalModel:
    """Model of another agent's inferred mental state."""

    person_id: str
    # ── Inferred internal states ────────────────────────────
    inferred_goals: List[str] = field(default_factory=list)
    inferred_beliefs: Dict[str, float] = field(
        default_factory=dict
    )  # topic → confidence
    likely_preferences: Dict[str, float] = field(
        default_factory=dict
    )  # behaviour → preference [-1, 1]
    inferred_emotion: str = "neutral"
    inferred_attention: str = "unknown"  # what they seem focused on
    knowledge_estimate: Dict[str, float] = field(
        default_factory=dict
    )  # topic → estimated knowledge [0,1]
    # ── Interaction model ───────────────────────────────────
    communication_style: str = "balanced"  # "brief", "explanatory", "warm", "formal"
    response_pattern: str = (
        "neutral"  # "responsive", "avoidant", "dominant", "cooperative"
    )
    trust_repair_history: List[Dict] = field(default_factory=list)
    # ── Confidence in this model ────────────────────────────
    observation_count: int = 0
    model_confidence: float = 0.1  # [0, 1] — how much data backs this model
    last_update_tick: int = 0
    # ── Second-order model: what does this person think the AI believes? ──
    # Keys are topic strings; values are attributed belief strengths [0, 1].
    # Represents "I think you think I know/believe X about Y."
    beliefs_about_ai: Dict[str, float] = field(default_factory=dict)
    second_order_confidence: float = 0.0  # confidence in our model of their model of us

    def update_from_interaction(
        self,
        tick: int,
        spoke: bool,
        words_heard: int,
        engagement: float,
        valence: float,
        topics: List[str],
    ) -> None:
        """Update model from a single interaction observation."""
        self.observation_count += 1
        self.last_update_tick = tick
        self.model_confidence = min(1.0, self.observation_count / 30.0)

        # Infer attention
        if engagement > 0.7:
            self.inferred_attention = "focused"
        elif engagement > 0.3:
            self.inferred_attention = "partial"
        else:
            self.inferred_attention = "distracted"

        # Infer emotion from valence
        if valence > 0.3:
            self.inferred_emotion = "positive"
        elif valence < -0.3:
            self.inferred_emotion = "negative"
        else:
            self.inferred_emotion = "neutral"

        # Update knowledge estimates based on topics discussed
        for topic in topics:
            if spoke and words_heard > 0:
                # If they spoke about it, they probably know something
                current = self.knowledge_estimate.get(topic, 0.3)
                self.knowledge_estimate[topic] = min(1.0, current * 0.9 + 0.5 * 0.1)
            else:
                # Topic was raised but no response — may not know
                current = self.knowledge_estimate.get(topic, 0.3)
                self.knowledge_estimate[topic] = max(0.0, current * 0.95)

        # Infer communication preference from response patterns
        if words_heard > 50:
            self.communication_style = "explanatory"
        elif words_heard < 5 and spoke:
            self.communication_style = "brief"
        elif engagement > 0.6:
            self.communication_style = "warm"

        # Infer response pattern from engagement + speaking
        if spoke and engagement > 0.5:
            self._shift_pattern("cooperative")
        elif spoke and engagement < 0.3:
            self._shift_pattern("dominant")
        elif not spoke and engagement < 0.3:
            self._shift_pattern("avoidant")

    def _shift_pattern(self, observed: str, lr: float = 0.15) -> None:
        """EMA shift of response pattern toward observed."""
        self.response_pattern = observed  # simplified — could do EMA over categories

    def update_ai_belief_attribution(self, topic: str, strength: float) -> None:
        """Record that this person has heard the AI claim *topic* with given strength.
        Updates the second-order model: what this person now attributes to the AI."""
        current = self.beliefs_about_ai.get(topic, 0.0)
        self.beliefs_about_ai[topic] = current * 0.7 + strength * 0.3
        self.second_order_confidence = min(1.0, self.second_order_confidence + 0.015)
        if len(self.beliefs_about_ai) > 60:
            # Retain only the strongest attributions
            self.beliefs_about_ai = dict(
                sorted(self.beliefs_about_ai.items(), key=lambda x: -x[1])[:60]
            )

    def second_order_summary(self) -> str:
        """Top attributed AI-beliefs as a readable string (for stream/logs)."""
        if not self.beliefs_about_ai or self.second_order_confidence < 0.05:
            return ""
        top = sorted(self.beliefs_about_ai.items(), key=lambda x: -x[1])[:3]
        topics = ", ".join(f"{t}({v:.2f})" for t, v in top)
        return f"{self.person_id}_thinks_AI_knows: {topics}"

    def update_preference(self, behaviour: str, reaction: float) -> None:
        """Record that person reacted to a behaviour with positive/negative valence."""
        current = self.likely_preferences.get(behaviour, 0.0)
        self.likely_preferences[behaviour] = current * 0.8 + reaction * 0.2

    def record_trust_event(self, tick: int, event_type: str, description: str) -> None:
        """Record a trust-relevant event (violation, repair, breakthrough)."""
        self.trust_repair_history.append(
            {
                "tick": tick,
                "type": event_type,
                "description": description[:80],
            }
        )
        if len(self.trust_repair_history) > 20:
            self.trust_repair_history = self.trust_repair_history[-20:]

    def expects_what(self) -> str:
        """What does this person likely expect from us right now?"""
        if self.inferred_emotion == "negative":
            return "empathy_or_space"
        if self.response_pattern == "cooperative":
            return "engagement"
        if self.response_pattern == "avoidant":
            return "minimal_interaction"
        if self.inferred_attention == "focused":
            return "substantive_response"
        return "neutral_acknowledgment"

    def knows_about(self, topic: str) -> float:
        """Estimate how much this person knows about a topic [0,1]."""
        return self.knowledge_estimate.get(topic, 0.3)

    def describe(self) -> str:
        goals = ", ".join(self.inferred_goals[:3]) if self.inferred_goals else "unknown"
        prefs = ", ".join(
            f"{k}:{v:+.1f}"
            for k, v in sorted(
                self.likely_preferences.items(), key=lambda x: -abs(x[1])
            )[:3]
        )
        return (
            f"model[{self.person_id}] conf={self.model_confidence:.2f} "
            f"emo={self.inferred_emotion} attn={self.inferred_attention} "
            f"style={self.communication_style} pattern={self.response_pattern} "
            f"goals=[{goals}] prefs=[{prefs}]"
        )

    def to_dict(self) -> Dict:
        return {
            "person_id": self.person_id,
            "inferred_goals": self.inferred_goals[:10],
            "inferred_beliefs": dict(list(self.inferred_beliefs.items())[:50]),
            "likely_preferences": dict(list(self.likely_preferences.items())[:30]),
            "inferred_emotion": self.inferred_emotion,
            "inferred_attention": self.inferred_attention,
            "knowledge_estimate": dict(list(self.knowledge_estimate.items())[:50]),
            "communication_style": self.communication_style,
            "response_pattern": self.response_pattern,
            "trust_repair_history": self.trust_repair_history[-10:],
            "observation_count": self.observation_count,
            "model_confidence": self.model_confidence,
            "beliefs_about_ai": dict(list(self.beliefs_about_ai.items())[:30]),
            "second_order_confidence": self.second_order_confidence,
            "last_update_tick": self.last_update_tick,
        }

    @staticmethod
    def from_dict(d: Dict) -> "MentalModel":
        mm = MentalModel(person_id=d.get("person_id", ""))
        mm.inferred_goals = d.get("inferred_goals", [])
        mm.inferred_beliefs = d.get("inferred_beliefs", {})
        mm.likely_preferences = d.get("likely_preferences", {})
        mm.inferred_emotion = d.get("inferred_emotion", "neutral")
        mm.inferred_attention = d.get("inferred_attention", "unknown")
        mm.knowledge_estimate = d.get("knowledge_estimate", {})
        mm.communication_style = d.get("communication_style", "balanced")
        mm.response_pattern = d.get("response_pattern", "neutral")
        mm.trust_repair_history = d.get("trust_repair_history", [])
        mm.observation_count = d.get("observation_count", 0)
        mm.model_confidence = d.get("model_confidence", 0.1)
        mm.last_update_tick = d.get("last_update_tick", 0)
        mm.beliefs_about_ai = d.get("beliefs_about_ai", {})
        mm.second_order_confidence = d.get("second_order_confidence", 0.0)
        return mm


# ─────────────────────────────────────────────────────────────────────────────
# TheoryOfMind — aggregated social inference engine
# ─────────────────────────────────────────────────────────────────────────────


class TheoryOfMind:
    """
    Aggregated engine for modelling other agents' minds.

    Maintains per-person MentalModels and provides:
      - Social strategy recommendations (how to interact)
      - Perspective-taking queries (what does X think about Y?)
      - Communication style adaptation (brief vs explanatory)
      - Trust repair tracking
    """

    MAX_MODELS = 100

    def __init__(self) -> None:
        self._models: Dict[str, MentalModel] = {}

    def get_model(self, person_id: str) -> MentalModel:
        """Get or create a mental model for a person."""
        if person_id not in self._models:
            if len(self._models) >= self.MAX_MODELS:
                # Evict least-observed
                worst = min(
                    self._models, key=lambda k: self._models[k].observation_count
                )
                del self._models[worst]
            self._models[person_id] = MentalModel(person_id=person_id)
        return self._models[person_id]

    def observe_interaction(
        self,
        tick: int,
        person_id: str,
        spoke: bool = False,
        words_heard: int = 0,
        engagement: float = 0.5,
        valence: float = 0.0,
        topics: List[str] = None,
        action: str = "",
        success: bool = True,
    ) -> None:
        """Record an interaction observation.

        Extended with action/success for skill-event integration:
        tracks what the robot DID toward a person and whether it worked.
        """
        model = self.get_model(person_id)
        model.update_from_interaction(
            tick, spoke, words_heard, engagement, valence, topics or []
        )
        # Track robot's action outcomes toward this person
        if action:
            _action_hist = getattr(model, "_action_outcomes", None)
            if _action_hist is None:
                model._action_outcomes = []
                _action_hist = model._action_outcomes
            _action_hist.append((tick, action, success))
            if len(_action_hist) > 100:
                model._action_outcomes = _action_hist[-100:]

    def recommend_strategy(self, person_id: str) -> Dict[str, str]:
        """
        Recommend communication strategy for this person.
        Returns dict with style, tone, action recommendations.
        Uses trust history, learned preferences, action outcomes,
        and distance preferences for richer adaptation.
        """
        model = self._models.get(person_id)
        if model is None or model.model_confidence < 0.1:
            return {
                "style": "balanced",
                "tone": "neutral",
                "action": "observe_first",
                "trust": 0.5,
                "success_rate": 0.5,
            }

        style = model.communication_style
        expects = model.expects_what()

        # ── Trust score derived from repair history + action outcomes ──
        _trust = 0.5
        for ev in model.trust_repair_history[-10:]:
            if ev.get("type") == "violation":
                _trust -= 0.1
            elif ev.get("type") in ("repair", "breakthrough"):
                _trust += 0.08
        _trust = max(0.0, min(1.0, _trust))

        # ── Action outcome success rate ──
        _action_hist = getattr(model, "_action_outcomes", [])
        _recent_actions = _action_hist[-15:] if _action_hist else []
        _success_rate = 1.0
        if _recent_actions:
            _successes = sum(1 for _, _, s in _recent_actions if s)
            _success_rate = _successes / len(_recent_actions)

        # ── Learned preferences: what does person respond well to? ──
        _best_pref = ""
        _best_pref_val = 0.0
        for beh, val in model.likely_preferences.items():
            if val > _best_pref_val:
                _best_pref = beh
                _best_pref_val = val
        _worst_pref = ""
        _worst_pref_val = 0.0
        for beh, val in model.likely_preferences.items():
            if val < -_worst_pref_val:
                _worst_pref = beh
                _worst_pref_val = abs(val)

        # ── Tone: combine emotion, trust, and response pattern ──
        tone = "empathetic" if model.inferred_emotion == "negative" else "warm"
        if model.response_pattern == "avoidant":
            tone = "gentle"
        elif model.response_pattern == "dominant":
            tone = "direct"
        if _trust < 0.3:
            tone = "careful"  # low trust → cautious communication

        # ── Action: richer rules based on full model data ──
        action = "engage"
        if expects == "empathy_or_space":
            # Check if person prefers space vs support based on outcomes
            if _success_rate < 0.4 or _trust < 0.3:
                action = "give_space"
            else:
                action = "offer_support"
        elif expects == "minimal_interaction":
            action = "acknowledge_briefly"
        elif expects == "substantive_response":
            action = "provide_detail"
        elif expects == "engagement" and _success_rate > 0.7:
            action = "engage"
        elif _success_rate < 0.3 and len(_recent_actions) >= 3:
            action = "give_space"  # repeated failures → back off

        # Override: if preferred behaviour is known, adapt action
        if _best_pref and _best_pref_val > 0.3:
            style = _best_pref  # use what they respond well to

        result = {"style": style, "tone": tone, "action": action}

        # ── Extras: pass trust + preference for downstream consumers ──
        result["trust"] = round(_trust, 4)
        result["success_rate"] = round(_success_rate, 4)
        result["model_confidence"] = round(model.model_confidence, 4)
        if _best_pref:
            result["preferred_behaviour"] = _best_pref
        if _worst_pref:
            result["avoided_behaviour"] = _worst_pref

        # ── Second-order corrections: topics where person's model of AI may
        # be misaligned with what the AI actually claims to believe.
        # Used downstream to flag potential clarifications in dialogue.
        _so_corrections: List[str] = []
        if model.second_order_confidence > 0.05:
            for _so_topic, _so_attr in model.beliefs_about_ai.items():
                _ai_belief = model.inferred_beliefs.get(_so_topic, -1.0)
                if _ai_belief >= 0.0 and abs(_so_attr - _ai_belief) > 0.40:
                    _so_corrections.append(_so_topic)
        if _so_corrections:
            result["second_order_corrections"] = _so_corrections[:3]

        return result

    def observe_ai_statement(
        self, tick: int, person_id: str, topics: List[str], strength: float = 0.6
    ) -> None:
        """Update second-order model after AI made a statement to this person.

        Call this after every assembled reply so the system tracks what topics
        it has claimed to the person — enabling "I think you think I know X".
        strength: how assertive/confident the claim was [0, 1].
        """
        model = self.get_model(person_id)
        for topic in topics[:6]:
            model.update_ai_belief_attribution(topic, strength)

    def get_second_order_view(self, person_id: str, topic: str) -> float:
        """What does person X think the AI believes about *topic*? Returns [0, 1]."""
        model = self._models.get(person_id)
        if model is None:
            return 0.0
        return model.beliefs_about_ai.get(topic, 0.0)

    def record_comm_outcome(self, tick: int, person_id: str, outcome_type: str) -> None:
        """Record how a person responded to our last utterance — closes the social learning loop.

        outcome_type: 'understood' | 'repair_requested' | 'topic_shifted' |
                      'delayed_response' | 'acknowledged'
        """
        model = self.get_model(person_id)
        # Append to per-model communication outcome history
        _hist = getattr(model, "_comm_outcomes", None)
        if _hist is None:
            model._comm_outcomes = []
            _hist = model._comm_outcomes
        _hist.append((tick, outcome_type))
        if len(_hist) > 50:
            model._comm_outcomes = _hist[-50:]

        # Update communication style preferences based on outcome
        if outcome_type == "understood":
            # Current style worked → positive signal; reset disengagement streak
            model.update_preference(model.communication_style, +0.3)
            model._disengagement_streak = 0  # type: ignore[attr-defined]
        elif outcome_type == "repair_requested":
            # Current style failed → negative + trust dent
            model.update_preference(model.communication_style, -0.3)
            model.record_trust_event(
                tick, "repair", f"Communication failed: repair requested (t={tick})"
            )
        elif outcome_type == "topic_shifted":
            # Person disengaged → mild negative
            model.update_preference(model.communication_style, -0.1)
        elif outcome_type == "delayed_response":
            # Long pause → mild negative
            model.update_preference(model.communication_style, -0.15)
        elif outcome_type == "disengaged":
            # Active social withdrawal — stronger signal than delay.
            # Penalise current style and track consecutive disengagements.
            model.update_preference(model.communication_style, -0.25)
            _streak = getattr(model, "_disengagement_streak", 0) + 1
            model._disengagement_streak = _streak  # type: ignore[attr-defined]
            model.record_trust_event(
                tick,
                "social_withdrawal",
                f"Disengagement #{_streak} detected (t={tick})",
            )
            # After repeated disengagement: mark person as avoidant
            if _streak >= 3:
                model._shift_pattern("avoidant")
                # Explicitly flag "give_space" as the preferred action
                model.update_preference("give_space", +0.4)
        elif outcome_type == "minimal_response":
            # Person replied with very few words — signals preference for brevity.
            # Shift communication style toward "brief" over several such responses.
            if model.communication_style != "brief":
                model.update_preference(model.communication_style, -0.1)
            model.update_preference("brief", +0.2)
            # After repeated minimal responses, actively set style to brief
            _min_hist = [
                r
                for _, r in getattr(model, "_comm_outcomes", [])
                if r == "minimal_response"
            ]
            if len(_min_hist) >= 3:
                model.communication_style = "brief"
        # "acknowledged" → no strong signal; leave as-is

    def social_summary(self) -> str:
        """Brief overview of all known social models."""
        if not self._models:
            return "No social models yet."
        parts = []
        for pid, model in sorted(
            self._models.items(), key=lambda x: -x[1].model_confidence
        ):
            parts.append(
                f"{pid}({model.inferred_emotion}/{model.model_confidence:.1f})"
            )
        return "ToM: " + ", ".join(parts[:5])

    def to_dict(self) -> Dict:
        return {
            "models": {k: v.to_dict() for k, v in self._models.items()},
        }

    def from_dict(self, data: Dict) -> None:
        for pid, md in data.get("models", {}).items():
            self._models[pid] = MentalModel.from_dict(md)
