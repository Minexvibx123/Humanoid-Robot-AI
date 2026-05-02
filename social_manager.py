"""
social_manager.py — Social Interaction Layer

Manages the social dimension of human-robot interaction:
  • Turn-taking protocol     (who speaks when)
  • Gaze management          (eye contact, shared attention)
  • Speaker tracking         (who is talking, engagement level)
  • Conversation context     (per-person state)
  • Social gesture selection (head nods, acknowledgments)
  • Proxemic awareness       (respect personal space)

Reads from:  world_state, telemetry_bus (speech events)
Writes to:   task_executive (social goals), skill_library (gaze skills)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from body_schema import BodySchema
    from task_executive import TaskExecutive
    from theory_of_mind import TheoryOfMind
    from world_state import TrackedPerson, WorldState


# ─────────────────────────────────────────────────────────────
# Conversation state per person
# ─────────────────────────────────────────────────────────────


class TurnState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    THINKING = "thinking"
    WAITING = "waiting"  # waiting for person to respond


@dataclass
class ConversationState:
    """Per-person conversation memory."""

    person_id: int
    turn_state: TurnState = TurnState.IDLE
    turns_count: int = 0
    words_heard: int = 0
    words_spoken: int = 0
    last_speech_tick: int = 0
    silence_ticks: int = 0
    engagement: float = 0.5  # [0,1]
    rapport: float = 0.5  # [0,1] built over time
    greeted: bool = False
    farewell_sent: bool = False


# ─────────────────────────────────────────────────────────────
# Persistent person model — long-term social memory
# ─────────────────────────────────────────────────────────────


@dataclass
class PersonModel:
    """Long-term model of a specific person, persisted across sessions.

    Accumulates behavioural observations to support consistent,
    individualised interaction over repeated encounters.
    """

    person_id: int
    trust: float = 0.5  # [0,1] builds slowly
    familiarity: float = 0.0  # [0,1] grows with encounters
    total_encounters: int = 0
    total_words_heard: int = 0
    total_words_spoken: int = 0
    # Observed behavioural tendencies (label → strength)
    preferences: Dict[str, float] = field(default_factory=dict)
    # Inferred goals / interests
    inferred_interests: List[str] = field(default_factory=list)
    # Compact interaction summaries (last N encounters)
    interaction_log: List[str] = field(default_factory=list)
    # Emotional valence history: running average of emotion during encounters
    avg_valence: float = 0.0
    last_encounter_tick: int = 0
    relationship_type: str = "emergent_contact"
    relationship_confidence: float = 0.35
    relationship_scores: Dict[str, float] = field(default_factory=dict)
    relationship_history: List[str] = field(default_factory=list)
    # Cumulative count of conflict/negative encounters — survives across sessions
    conflict_encounter_count: int = 0
    # Tick of last forgetting pass (for controlled decay scheduling)
    _last_forgetting_tick: int = 0

    # ── Learning fields: repair, outcome, topic success ──────────────────
    # How many times we were asked to repeat/clarify with this person
    repair_count: int = 0
    # Last tick a repair event was observed (for recency-sensitive caution)
    last_repair_tick: int = 0
    # EMA of recent interaction outcome valence (positive = good, negative = bad)
    # Updated externally after each response via record_communication_outcome()
    recent_outcome_ema: float = 0.5
    # Weighted success score per topic/interest — higher = use this topic more
    # Dict[topic: str, score: float 0-1]
    successful_topics: Dict[str, float] = field(default_factory=dict)

    MAX_LOG = 50  # keep last 50 interaction summaries
    MAX_INTERESTS = 30  # cap inferred interests
    MAX_REL_HISTORY = 12
    MAX_TOPICS = 40  # cap successful_topics dict

    def record_encounter(
        self,
        tick: int,
        duration_ticks: int,
        words_heard: int,
        words_spoken: int,
        rapport: float,
        dominant_emotion: str,
    ) -> None:
        self.total_encounters += 1
        self.total_words_heard += words_heard
        self.total_words_spoken += words_spoken
        self.last_encounter_tick = tick
        # Familiarity grows logarithmically
        self.familiarity = min(1.0, self.familiarity + 0.05)
        # Trust: positive interactions slowly build trust
        if rapport > 0.6:
            self.trust = min(1.0, self.trust + 0.01)
        elif rapport < 0.3:
            self.trust = max(0.0, self.trust - 0.005)
        # Log
        summary = (
            f"t={tick} dur={duration_ticks} w_in={words_heard} "
            f"w_out={words_spoken} rap={rapport:.2f} emo={dominant_emotion}"
        )
        self.interaction_log.append(summary)
        if len(self.interaction_log) > self.MAX_LOG:
            self.interaction_log = self.interaction_log[-self.MAX_LOG :]
        # Periodic forgetting pass (self-gated; only runs every ~5000 ticks)
        self.apply_forgetting(tick)

    def note_interest(self, topic: str) -> None:
        if topic not in self.inferred_interests:
            self.inferred_interests.append(topic)
            if len(self.inferred_interests) > self.MAX_INTERESTS:
                self.inferred_interests.pop(0)

    def note_preference(self, label: str, strength: float = 0.1) -> None:
        cur = self.preferences.get(label, 0.0)
        self.preferences[label] = min(1.0, cur + strength)

    def update_valence(self, valence: float, alpha: float = 0.1) -> None:
        self.avg_valence = self.avg_valence * (1 - alpha) + valence * alpha

    def record_conflict_encounter(
        self, tick: int, emotion: str = "conflict", topic: str = ""
    ) -> None:
        """Increment the conflict counter and slightly erode trust.

        This is separate from record_encounter() so that conflict signals
        are reliably tracked even within a single encounter.  Trust erosion
        is smaller than the encounter-level penalty to avoid cascades.
        """
        self.conflict_encounter_count += 1
        self.trust = max(0.0, self.trust - 0.008)
        note = f"t={tick} conflict emo={emotion}"
        if topic:
            note += f" topic={topic[:30]}"
        self.interaction_log.append(note)
        if len(self.interaction_log) > self.MAX_LOG:
            self.interaction_log = self.interaction_log[-self.MAX_LOG :]

    def apply_forgetting(self, current_tick: int, interval: int = 5000) -> None:
        """Controlled forgetting pass — run approximately every `interval` ticks.

        Rules:
        - Weak preferences (strength < 0.15) decay towards 0 and are pruned.
        - Strong preferences (strength >= 0.5) and long-term patterns are stable.
        - Inferred interests older than the tracked window are trimmed if the
          list exceeds MAX_INTERESTS // 2.  The most-recently observed ones
          (tail) are retained — they are the most relevant.
        - relationship_history is trimmed to MAX_REL_HISTORY.
        - Social long-term patterns (trust, familiarity, conflict_count) do NOT
          decay here; they are only updated by direct observation methods.
        """
        if current_tick - self._last_forgetting_tick < interval:
            return
        self._last_forgetting_tick = current_tick

        # Decay weak preferences
        _DECAY_RATE = 0.05  # per forgetting pass
        _PRUNE_THRESHOLD = 0.03
        updated: Dict[str, float] = {}
        for label, strength in self.preferences.items():
            if strength >= 0.5:
                # Long-term reinforced preference — stable
                updated[label] = strength
            elif strength >= 0.15:
                # Moderate — slow decay
                updated[label] = max(0.0, strength - _DECAY_RATE)
            # else: weak preference (< 0.15) — forgotten (not added to updated)
        self.preferences = {k: v for k, v in updated.items() if v >= _PRUNE_THRESHOLD}

        # Trim excess inferred interests (keep most recent half)
        _cap = self.MAX_INTERESTS // 2
        if len(self.inferred_interests) > _cap:
            self.inferred_interests = self.inferred_interests[-_cap:]

        # Trim relationship history
        if len(self.relationship_history) > self.MAX_REL_HISTORY:
            self.relationship_history = self.relationship_history[-self.MAX_REL_HISTORY:]

        # Decay successful_topics scores slowly — relevance fades over time
        _TOPIC_DECAY = 0.03
        self.successful_topics = {
            t: max(0.0, s - _TOPIC_DECAY)
            for t, s in self.successful_topics.items()
            if s - _TOPIC_DECAY > 0.02
        }

    def record_repair_event(self, tick: int) -> None:
        """Called when a repair speech act was triggered for this person.

        Tracks how frequently the person needs clarification so that
        interaction_style() can return clarity="high" when appropriate.
        """
        self.repair_count += 1
        self.last_repair_tick = tick

    def record_communication_outcome(
        self, valence: float, tick: int, alpha: float = 0.15
    ) -> None:
        """Update the EMA of recent interaction outcome valence.

        Called after each response is produced.  Positive valence (high
        engagement, positive emotion) reinforces current interaction style;
        negative valence (confusion, flat response, conflict) signals that
        the current approach should be adjusted.

        Args:
            valence: outcome quality in [-1, 1].  Use em.valence() or
                     engagement-derived signal.
            tick:    current tick (for logging)
            alpha:   EMA learning rate
        """
        self.recent_outcome_ema = (
            (1.0 - alpha) * self.recent_outcome_ema + alpha * (valence * 0.5 + 0.5)
        )
        # Clip to [0, 1]
        self.recent_outcome_ema = max(0.0, min(1.0, self.recent_outcome_ema))

    def record_topic_outcome(
        self, topic: str, success: bool, alpha: float = 0.12
    ) -> None:
        """Update the per-topic success score.

        A successful topic (positive social outcome after it was discussed)
        gets its score raised; a failed topic gets it lowered.  Topics that
        consistently lead to positive outcomes become preferred in response
        assembly.

        Args:
            topic:   the topic/interest key (lower-case word)
            success: True if the topic led to a positive outcome
            alpha:   EMA learning rate
        """
        current = self.successful_topics.get(topic, 0.5)
        target = 1.0 if success else 0.0
        self.successful_topics[topic] = (1.0 - alpha) * current + alpha * target
        # Prune if over cap: keep highest-scoring topics
        if len(self.successful_topics) > self.MAX_TOPICS:
            _sorted = sorted(self.successful_topics.items(), key=lambda x: -x[1])
            self.successful_topics = dict(_sorted[: self.MAX_TOPICS])

    def top_successful_topics(self, n: int = 5) -> List[str]:
        """Return top-n topics by success score (for use in response assembly)."""
        _sorted = sorted(self.successful_topics.items(), key=lambda x: -x[1])
        return [t for t, _ in _sorted[:n] if _ > 0.55]

    def observe_relationship_signals(self, cues: Dict[str, float], tick: int) -> str:
        """Adapt active relationship type from rolling evidence instead of one-shot rules."""
        labels = [
            "mistrustful_questioner",
            "familiar_cooperative_speaker",
            "curious_questioner",
            "guarded_interlocutor",
            "trusted_cooperative_partner",
            "emergent_contact",
        ]
        for label in labels:
            prev = self.relationship_scores.get(label, 0.0)
            obs = float(cues.get(label, 0.0))
            self.relationship_scores[label] = prev * 0.82 + obs * 0.18

        dominant = max(labels, key=lambda lab: self.relationship_scores.get(lab, 0.0))
        current = self.relationship_type or "emergent_contact"
        dominant_score = self.relationship_scores.get(dominant, 0.0)
        current_score = self.relationship_scores.get(current, 0.0)

        self.relationship_history.append(dominant)
        if len(self.relationship_history) > self.MAX_REL_HISTORY:
            self.relationship_history = self.relationship_history[
                -self.MAX_REL_HISTORY :
            ]

        recent = self.relationship_history[-5:]
        dominant_votes = sum(1 for item in recent if item == dominant)
        changed = False
        if (
            dominant != current
            and dominant_votes >= 3
            and dominant_score >= current_score + 0.08
        ):
            self.relationship_type = dominant
            changed = True
        elif current not in labels:
            self.relationship_type = dominant
            changed = True

        self.relationship_confidence = max(0.2, min(1.0, dominant_score))
        if changed:
            self.interaction_log.append(
                f"t={tick} relationship_shift={current}->{self.relationship_type} conf={self.relationship_confidence:.2f}"
            )
            if len(self.interaction_log) > self.MAX_LOG:
                self.interaction_log = self.interaction_log[-self.MAX_LOG :]
        return self.relationship_type

    def interaction_style(self) -> Dict:
        """Synthesize clear behavioral guidance from accumulated trajectory data.

        Computes actionable dimensions directly from stored fields — no new
        scores, no new storage.  Used by the response pipeline to modulate:
        length, warmth, formality, caution, and conversational initiative.

        New dimensions from learning fields:
          clarity:   "high" | "normal"     — from repair_count / total_encounters
          outcome:   "positive" | "neutral" | "negative"  — from recent_outcome_ema
        """
        # ── Length target ─────────────────────────────────────────────────
        concise = self.preferences.get("concise_speech", 0.0)
        verbose = self.preferences.get("verbose_speech", 0.0)
        if concise > verbose + 0.15:
            length_target = "short"
        elif verbose > concise + 0.15:
            length_target = "long"
        else:
            length_target = "medium"

        # ── Warmth [0,1] — trust + familiarity + positive history ─────────
        warmth = min(
            1.0,
            self.trust * 0.45
            + min(self.familiarity, 1.0) * 0.35
            + max(0.0, self.avg_valence) * 0.20,
        )

        # ── Formality — new person is formal; trusted/familiar is casual ──
        if self.familiarity < 0.15 or self.total_encounters == 0:
            formality = "formal"
        elif self.familiarity > 0.55 and self.trust > 0.55:
            formality = "casual"
        else:
            formality = "neutral"

        # ── Caution — trust erosion + conflict rate ────────────────────────
        conflict_rate = self.conflict_encounter_count / max(self.total_encounters, 1)
        caution_score = (1.0 - self.trust) * 0.55 + conflict_rate * 0.45
        if caution_score > 0.55:
            caution = "high"
        elif caution_score > 0.25:
            caution = "medium"
        else:
            caution = "low"

        # ── Initiative — driven by trust/familiarity AND recent outcome ───
        # Poor recent outcome lowers initiative regardless of prior trust.
        _outcome_ok = self.recent_outcome_ema >= 0.45
        if self.trust > 0.65 and self.familiarity > 0.5 and _outcome_ok:
            initiative = "proactive"
        elif self.trust < 0.3 or caution == "high" or self.recent_outcome_ema < 0.3:
            initiative = "reactive"
        else:
            initiative = "neutral"

        # ── Clarity — raised when repairs are frequent ────────────────────
        repair_rate = self.repair_count / max(self.total_encounters, 1)
        clarity = "high" if repair_rate > 0.25 or self.repair_count >= 3 else "normal"

        # ── Recent outcome quality ─────────────────────────────────────────
        if self.recent_outcome_ema >= 0.65:
            outcome = "positive"
        elif self.recent_outcome_ema <= 0.35:
            outcome = "negative"
        else:
            outcome = "neutral"

        return {
            "length_target": length_target,
            "warmth": round(warmth, 3),
            "formality": formality,
            "caution": caution,
            "initiative": initiative,
            "is_known": self.total_encounters > 0,
            "is_familiar": self.familiarity > 0.4 and self.total_encounters > 2,
            "clarity": clarity,
            "outcome": outcome,
        }


# ─────────────────────────────────────────────────────────────
# Gaze target types
# ─────────────────────────────────────────────────────────────


class GazeMode(Enum):
    DIRECT = "direct"  # eye contact with speaker
    AVERT = "avert"  # natural aversion (avoid staring)
    SHARED = "shared"  # look at what person is looking at
    SCANNING = "scanning"  # no specific target, look around
    DOWN = "down"  # thinking / submissive


@dataclass
class GazeTarget:
    mode: GazeMode = GazeMode.SCANNING
    person_id: Optional[int] = None
    yaw: float = 90.0
    pitch: float = 90.0


# ─────────────────────────────────────────────────────────────
# Social Manager
# ─────────────────────────────────────────────────────────────


class SocialManager:
    """
    Manages social interaction protocols.
    Called once per tick by the brain.
    """

    # Timing
    AVERT_AFTER_TICKS = 60  # break eye contact after ~6s
    GREET_DISTANCE_CM = 150  # auto-greet within this range
    FAREWELL_ABSENT = 90  # ticks before farewell on disappearance
    NOD_INTERVAL = 30  # head nod while listening
    TURN_SILENCE_LIMIT = 50  # ticks of silence before switching turn

    def __init__(self) -> None:
        self._conversations: Dict[int, ConversationState] = {}
        self._person_models: Dict[int, PersonModel] = {}
        self._gaze = GazeTarget()
        self._robot_speaking = False
        self._robot_speech_tick = 0
        self._gaze_hold_ticks = 0
        self._last_nod_tick = 0
        self._social_events: List[str] = []

    # ── External signals ──────────────────────────────────────

    def robot_started_speaking(self, tick: int) -> None:
        self._robot_speaking = True
        self._robot_speech_tick = tick

    def robot_stopped_speaking(self, tick: int) -> None:
        self._robot_speaking = False

    def person_spoke(
        self, person_id: int, tick: int, word_count: int = 0, speech_text: str = ""
    ) -> None:
        """Notify that a person is speaking or spoke."""
        conv = self._get_or_create(person_id)
        conv.last_speech_tick = tick
        conv.silence_ticks = 0
        conv.words_heard += word_count
        if conv.turn_state != TurnState.LISTENING:
            conv.turn_state = TurnState.LISTENING
            conv.turns_count += 1
        # Learn interests and preferences from speech content
        if speech_text:
            self._learn_from_speech(person_id, speech_text)

    def _learn_from_speech(self, person_id: int, text: str) -> None:
        """Extract topics and behavioral cues from speech to learn preferences."""
        pm = self._get_or_create_model(person_id)
        words = text.lower().split()
        # Extract candidate interest words (5+ chars, skip common)
        _SKIP = {
            "about",
            "would",
            "could",
            "should",
            "there",
            "where",
            "which",
            "their",
            "these",
            "those",
            "haben",
            "nicht",
            "einen",
            "keine",
            "dieser",
            "diese",
            "wegen",
            "damit",
        }
        for w in words:
            if len(w) >= 5 and w not in _SKIP:
                pm.note_interest(w)
        # Detect behavioral preferences from speech patterns
        if len(words) < 4:
            pm.note_preference("concise_speech", 0.02)
        elif len(words) > 20:
            pm.note_preference("verbose_speech", 0.02)
        # Detect question patterns → curious person
        if any(w in text.lower() for w in ("?", "warum", "wieso", "why", "how")):
            pm.note_preference("inquisitive", 0.03)

    def learn_behavior_preference(
        self, person_id: int, label: str, strength: float = 0.05
    ) -> None:
        """External call to note a behavioral preference for a person."""
        pm = self._get_or_create_model(person_id)
        pm.note_preference(label, strength)

    # ── Main tick ─────────────────────────────────────────────

    def tick(
        self,
        tick: int,
        world: "WorldState",
        executive: Optional["TaskExecutive"] = None,
    ) -> None:
        """Advance social state. Called once per brain tick."""
        self._social_events.clear()
        visible_ids = set()

        # Update conversation states for all visible persons
        for pid, person in world.persons.items():
            visible_ids.add(pid)
            conv = self._get_or_create(pid)

            # Auto-greet
            if not conv.greeted and person.distance_cm < self.GREET_DISTANCE_CM:
                conv.greeted = True
                self._social_events.append(f"greet:{pid}")
                if executive:
                    executive.submit_goal("greet_person", f"person {pid}", tick=tick)

            # Track silence
            if (tick - conv.last_speech_tick) > 0:
                conv.silence_ticks += 1
            else:
                conv.silence_ticks = 0

            # Turn management
            self._update_turn(conv, tick)

            # Engagement decay/growth
            if person.speaking:
                conv.engagement = min(1.0, conv.engagement + 0.01)
            else:
                conv.engagement = max(0.0, conv.engagement - 0.002)

            # Rapport slowly builds with interaction
            if conv.turns_count > 0:
                conv.rapport = min(1.0, conv.rapport + 0.0005)

            # Learn proxemic preferences from observed distance
            pm = self._get_or_create_model(pid)
            if person.distance_cm < 60:
                pm.note_preference("close_proximity", 0.005)
            elif person.distance_cm > 200:
                pm.note_preference("far_proximity", 0.005)
            # Update rolling valence from engagement (proxy for interaction quality)
            pm.update_valence(conv.engagement - 0.5)

        # Detect disappeared persons → farewell
        gone_ids = [pid for pid in self._conversations if pid not in visible_ids]
        for pid in gone_ids:
            conv = self._conversations[pid]
            conv.silence_ticks += 1
            if conv.silence_ticks > self.FAREWELL_ABSENT and not conv.farewell_sent:
                conv.farewell_sent = True
                self._social_events.append(f"farewell:{pid}")
                # Finalize encounter into persistent person model
                pm = self._get_or_create_model(pid)
                pm.record_encounter(
                    tick,
                    conv.turns_count * 10,
                    conv.words_heard,
                    conv.words_spoken,
                    conv.rapport,
                    "neutral",  # caller may override with actual emotion
                )
            if conv.silence_ticks > self.FAREWELL_ABSENT * 3:
                del self._conversations[pid]

        # Gaze management
        self._update_gaze(tick, world)

    # ── Turn management ───────────────────────────────────────

    def _update_turn(self, conv: ConversationState, tick: int) -> None:
        if self._robot_speaking:
            conv.turn_state = TurnState.SPEAKING
        elif conv.silence_ticks > self.TURN_SILENCE_LIMIT:
            if conv.turn_state == TurnState.LISTENING:
                conv.turn_state = TurnState.THINKING
        elif conv.turn_state == TurnState.SPEAKING:
            conv.turn_state = TurnState.WAITING

    # ── Gaze management ──────────────────────────────────────

    def _update_gaze(self, tick: int, world: "WorldState") -> None:
        self._gaze_hold_ticks += 1
        speaker = world.speaking_person()

        if speaker:
            # Look at the speaker
            if self._gaze_hold_ticks < self.AVERT_AFTER_TICKS:
                self._gaze = GazeTarget(
                    GazeMode.DIRECT,
                    speaker.person_id,
                    90.0 + (speaker.center_x - 0.5) * 70.0,
                    90.0 + (0.5 - speaker.center_y) * 44.0,
                )
            else:
                # Natural aversion
                self._gaze = GazeTarget(GazeMode.AVERT, speaker.person_id, 85.0, 92.0)
                if self._gaze_hold_ticks > self.AVERT_AFTER_TICKS + 15:
                    self._gaze_hold_ticks = 0
        elif world.zone.n_persons_visible > 0:
            engaged = world.most_engaged_person()
            if engaged:
                self._gaze = GazeTarget(
                    GazeMode.DIRECT,
                    engaged.person_id,
                    90.0 + (engaged.center_x - 0.5) * 70.0,
                    90.0 + (0.5 - engaged.center_y) * 44.0,
                )
        else:
            self._gaze = GazeTarget(GazeMode.SCANNING)
            self._gaze_hold_ticks = 0

    # ── Queries ───────────────────────────────────────────────

    @property
    def gaze(self) -> GazeTarget:
        return self._gaze

    @property
    def social_events(self) -> List[str]:
        return list(self._social_events)

    def primary_interlocutor(self) -> Optional[int]:
        """Return person_id with highest engagement."""
        best_pid, best_eng = None, -1.0
        for pid, conv in self._conversations.items():
            if conv.engagement > best_eng:
                best_eng = conv.engagement
                best_pid = pid
        return best_pid

    def turn_state_for(self, person_id: int) -> TurnState:
        conv = self._conversations.get(person_id)
        return conv.turn_state if conv else TurnState.IDLE

    def should_nod(self, tick: int) -> bool:
        """Return True if robot should do a listening nod."""
        # Nod while listening to active speaker, every NOD_INTERVAL ticks
        for conv in self._conversations.values():
            if conv.turn_state == TurnState.LISTENING:
                if (tick - self._last_nod_tick) > self.NOD_INTERVAL:
                    self._last_nod_tick = tick
                    return True
        return False

    def conversation_count(self) -> int:
        return len(self._conversations)

    def describe(self) -> str:
        active = sum(
            1 for c in self._conversations.values() if c.turn_state != TurnState.IDLE
        )
        gaze = self._gaze.mode.value
        primary = self.primary_interlocutor()
        return (
            f"social persons={len(self._conversations)} active={active} "
            f"gaze={gaze} primary={primary}"
        )

    def social_summary(self) -> Dict:
        """Compact summary for consciousness."""
        primary = self.primary_interlocutor()
        conv = self._conversations.get(primary) if primary else None
        pm = self._person_models.get(primary) if primary else None
        rel = self.relationship_profile(primary) if primary is not None else {}
        return {
            "n_conversations": len(self._conversations),
            "gaze_mode": self._gaze.mode.value,
            "primary_person": primary,
            "turn_state": conv.turn_state.value if conv else "idle",
            "engagement": conv.engagement if conv else 0.0,
            "rapport": conv.rapport if conv else 0.0,
            "robot_speaking": self._robot_speaking,
            "trust": pm.trust if pm else 0.5,
            "familiarity": pm.familiarity if pm else 0.0,
            "total_encounters": pm.total_encounters if pm else 0,
            "relationship_type": rel.get("relationship_type", ""),
            "relationship_label": rel.get("relationship_label", ""),
        }

    def relationship_profile(
        self, person_id: Optional[int], theory_of_mind: Optional["TheoryOfMind"] = None
    ) -> Dict:
        """Cluster a person into a reusable relationship type for social semantics."""
        if person_id is None:
            return {}
        _conv_key = self._resolve_person_key(self._conversations, person_id)
        _pm_key = self._resolve_person_key(self._person_models, person_id)
        conv = self._conversations.get(_conv_key)
        pm = self._person_models.get(_pm_key)
        mm = None
        if theory_of_mind is not None:
            try:
                mm = theory_of_mind.get_model(str(person_id))
            except Exception:
                mm = None

        trust = float(getattr(pm, "trust", 0.5) if pm is not None else 0.5)
        familiarity = float(getattr(pm, "familiarity", 0.0) if pm is not None else 0.0)
        engagement = float(
            getattr(conv, "engagement", 0.0) if conv is not None else 0.0
        )
        preferences = (
            dict(getattr(pm, "preferences", {}) or {}) if pm is not None else {}
        )
        inquisitive = float(preferences.get("inquisitive", 0.0))
        concise = float(preferences.get("concise_speech", 0.0))
        verbose = float(preferences.get("verbose_speech", 0.0))
        response_pattern = str(
            getattr(mm, "response_pattern", "neutral") if mm is not None else "neutral"
        )
        comm_style = str(
            getattr(mm, "communication_style", "balanced")
            if mm is not None
            else "balanced"
        )

        cues = {
            "mistrustful_questioner": (
                max(0.0, 0.45 - trust) * 1.5
                + inquisitive * 2.2
                + (0.20 if response_pattern == "avoidant" else 0.0)
            ),
            "familiar_cooperative_speaker": (
                familiarity * 0.9
                + trust * 0.9
                + (0.35 if response_pattern == "cooperative" else 0.0)
            ),
            "curious_questioner": (
                inquisitive * 2.4
                + (0.25 if comm_style == "explanatory" else 0.0)
                + (0.10 if verbose > concise else 0.0)
            ),
            "guarded_interlocutor": (
                max(0.0, 0.40 - trust) * 1.4
                + (0.30 if response_pattern == "avoidant" else 0.0)
                + (0.18 if concise > verbose else 0.0)
            ),
            "trusted_cooperative_partner": (
                trust * 0.8
                + familiarity * 0.7
                + engagement * 0.4
                + (0.25 if response_pattern == "cooperative" else 0.0)
            ),
            "emergent_contact": 0.25 + max(0.0, 0.35 - familiarity),
        }
        relationship_type = (
            pm.observe_relationship_signals(cues, getattr(pm, "last_encounter_tick", 0))
            if pm is not None
            else "emergent_contact"
        )
        relationship_confidence = (
            getattr(pm, "relationship_confidence", 0.35) if pm is not None else 0.35
        )

        labels = {
            "mistrustful_questioner": "misstrauischer Fragesteller",
            "familiar_cooperative_speaker": "vertrauter kooperativer Sprecher",
            "curious_questioner": "neugieriger Fragesteller",
            "guarded_interlocutor": "vorsichtiger Gesprächspartner",
            "trusted_cooperative_partner": "vertrauter kooperativer Partner",
            "emergent_contact": "offener neuer Kontakt",
        }
        style = (
            "brief"
            if concise > verbose and concise > 0.02
            else "expansive" if verbose > concise else comm_style
        )
        return {
            "person_id": person_id,
            "relationship_type": relationship_type,
            "relationship_label": labels.get(
                relationship_type, relationship_type.replace("_", " ")
            ),
            "relationship_confidence": relationship_confidence,
            "trust": trust,
            "familiarity": familiarity,
            "engagement": engagement,
            "style": style,
            "response_pattern": response_pattern,
            "preferences": preferences,
        }

    def style_for_person(
        self,
        person_id: int,
        theory_of_mind: Optional["TheoryOfMind"] = None,
    ) -> Dict:
        """Return a unified behavioral style dict for this person.

        Merges PersonModel trajectory (trust, familiarity, preferences,
        conflict) with ToM communication_style observations so the
        response pipeline has a single authoritative guidance object.

        Keys:
          length_target: "short" | "medium" | "long"
          warmth:        float 0-1
          formality:     "formal" | "neutral" | "casual"
          caution:       "high" | "medium" | "low"
          initiative:    "proactive" | "neutral" | "reactive"
          is_known:      bool  (has at least one prior encounter)
          is_familiar:   bool  (familiarity > 0.4, encounters > 2)
        """
        pm = self.person_model(person_id)
        if pm is None:
            # Completely unknown person — apply safe defaults
            return {
                "length_target": "medium",
                "warmth": 0.4,
                "formality": "formal",
                "caution": "low",
                "initiative": "neutral",
                "is_known": False,
                "is_familiar": False,
                "clarity": "normal",
                "outcome": "neutral",
            }
        style = pm.interaction_style()

        # Refine with ToM communication_style when available
        if theory_of_mind is not None:
            try:
                mm = theory_of_mind.get_model(str(person_id))
                tom_style = getattr(mm, "communication_style", "balanced")
                # ToM "brief" tightens length if PersonModel hasn't already set short
                if tom_style == "brief" and style["length_target"] == "medium":
                    style["length_target"] = "short"
                elif tom_style == "explanatory" and style["length_target"] == "medium":
                    style["length_target"] = "long"
                # ToM warmth/formal as a tiebreaker
                if tom_style == "warm" and style["formality"] == "neutral":
                    style["formality"] = "casual"
                elif tom_style == "formal" and style["formality"] == "neutral":
                    style["formality"] = "formal"
            except Exception:
                pass  # ToM not available — PersonModel style is still valid
        return style

    # ── Internals ─────────────────────────────────────────────

    def _get_or_create(self, person_id: int) -> ConversationState:
        if person_id not in self._conversations:
            self._conversations[person_id] = ConversationState(person_id=person_id)
        return self._conversations[person_id]

    def _get_or_create_model(self, person_id: int) -> PersonModel:
        if person_id not in self._person_models:
            self._person_models[person_id] = PersonModel(person_id=person_id)
        return self._person_models[person_id]

    def _resolve_person_key(self, mapping: Dict, person_id: Any) -> Any:
        if person_id in mapping:
            return person_id
        text = str(person_id)
        if text in mapping:
            return text
        if text.startswith("person_"):
            suffix = text.split("person_", 1)[1]
            if suffix in mapping:
                return suffix
            if suffix.isdigit():
                int_suffix = int(suffix)
                if int_suffix in mapping:
                    return int_suffix
        if text.isdigit():
            int_text = int(text)
            if int_text in mapping:
                return int_text
        return person_id

    def person_model(self, person_id: int) -> Optional[PersonModel]:
        return self._person_models.get(
            self._resolve_person_key(self._person_models, person_id)
        )

    @property
    def person_models(self) -> Dict[int, PersonModel]:
        return self._person_models

    def sync_with_tom(self, theory_of_mind: "TheoryOfMind") -> None:
        """Synchronize PersonModel ↔ TheoryOfMind.MentalModel.

        PersonModel holds behavioural observations (trust, familiarity,
        encounter counts, interests).  MentalModel holds inferred mental
        states (goals, beliefs, knowledge, communication style).

        This method ensures both models share a consistent view:
        - PersonModel.trust → MentalModel trust_repair_history sentinel
        - PersonModel.inferred_interests → MentalModel knowledge_estimate
        - MentalModel.communication_style → PersonModel.preferences['style']
        - MentalModel.inferred_emotion → PersonModel.avg_valence direction
        """
        for pid, pm in self._person_models.items():
            mm = theory_of_mind.get_model(str(pid))
            if mm is None:
                continue
            # PM → MM: interests become knowledge topics
            for interest in pm.inferred_interests:
                if interest not in mm.knowledge_estimate:
                    mm.knowledge_estimate[interest] = 0.3
            # PM → MM: trust level as synthetic trust event
            if pm.trust > 0.7 and mm.model_confidence > 0.2:
                _has_positive = any(
                    e.get("type") == "positive_trust"
                    for e in mm.trust_repair_history[-5:]
                )
                if not _has_positive:
                    mm.record_trust_event(
                        0, "positive_trust", f"high_trust={pm.trust:.2f}"
                    )
            # MM → PM: communication style as preference
            if mm.communication_style:
                pm.preferences["comm_style"] = {
                    "brief": 0.2,
                    "explanatory": 0.6,
                    "warm": 0.8,
                    "formal": 0.4,
                    "balanced": 0.5,
                }.get(mm.communication_style, 0.5)
            # MM → PM: inferred emotion → valence direction
            _emo_val = {"positive": 0.6, "neutral": 0.0, "negative": -0.4}
            _v = _emo_val.get(mm.inferred_emotion, 0.0)
            if abs(_v) > 0.1:
                pm.update_valence(_v, alpha=0.05)
