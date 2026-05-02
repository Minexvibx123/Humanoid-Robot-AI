"""
dialogue_manager.py — Structured Dialogue Layer

Bridges the gap between raw ASR text and the consciousness system by
introducing structured conversational objects:

  • DialogueTurn      — one utterance with social/semantic metadata
  • CommonGround      — per-person shared context (topic, referents, open Qs)
  • DialogueState     — per-person conversation state machine
  • SpeechActPlanner  — decides WHAT to do (answer, ask, confirm, repair, …)
  • UtterancePlan     — bundles text + prosody + body cues for output

Data flow (one tick):
  1. ASR text → DialogueTurn (enriched with world/social context)
  2. DialogueTurn updates CommonGround + DialogueState
  3. ConsciousnessCore processes + produces reply text
  4. SpeechActPlanner selects speech-act type (assert, ask, repair …)
  5. UtterancePlan bundles text + motor cues → SpeechOutput + RobotController

Reads from:  world_state, social_manager, theory_of_mind, consciousness
Writes to:   speech_output (TTS), robot_controller (jaw/gaze), social_manager (turn events)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from consciousness import ConsciousnessCore
    from social_manager import SocialManager
    from theory_of_mind import TheoryOfMind
    from world_state import TrackedPerson, WorldState


# ─────────────────────────────────────────────────────────────
# Speech Act taxonomy
# ─────────────────────────────────────────────────────────────


class SpeechAct(Enum):
    """What the system intends to DO with its next utterance."""

    ASSERT = "assert"  # state a fact / belief
    ASK = "ask"  # request information
    CONFIRM = "confirm"  # acknowledge / agree
    DENY = "deny"  # disagree / correct
    REPAIR = "repair"  # clarification request
    GREET = "greet"  # social opening
    FAREWELL = "farewell"  # social closing
    BACKCHANNEL = "backchannel"  # "hmm", "ja"
    HESITATE = "hesitate"  # signal uncertainty
    OBSERVE = "observe"  # narrate perception
    SILENCE = "silence"  # choose not to speak


# ─────────────────────────────────────────────────────────────
# Resolved Referent — a pronoun / demonstrative bound to an entity
# ─────────────────────────────────────────────────────────────


@dataclass
class ResolvedReferent:
    """A pronoun or demonstrative ('er', 'das', 'this') resolved to a world entity.

    Created by Tier -1.1 in _assemble_grounded_reply() and stored in
    CommonGround.active_referents so all downstream pipeline stages (LLM,
    speech act planner, ToM) know what the user was referring to.
    """

    referent_id: str         # "person:p0" | "object:cup" — key into WorldState
    referent_type: str       # "person" | "object" | "unknown"
    source_phrase: str       # the original surface form: "er", "das", "it", …
    confidence: float        # salience score at resolution time [0, 1]
    resolution_source: str   # "world_model" | "common_ground" | "label_match"
    tick: int = 0
    discourse_status: str = "active"  # "active" | "backgrounded" | "resolved"
    salience: float = 0.0

    def to_dict(self) -> dict:
        return {
            "referent_id": self.referent_id,
            "referent_type": self.referent_type,
            "source_phrase": self.source_phrase,
            "confidence": self.confidence,
            "resolution_source": self.resolution_source,
            "tick": self.tick,
            "discourse_status": self.discourse_status,
            "salience": self.salience,
        }


# ─────────────────────────────────────────────────────────────
# Dialogue Turn — one utterance with metadata
# ─────────────────────────────────────────────────────────────


@dataclass
class DialogueTurn:
    """Structured representation of a single utterance."""

    tick: int = 0
    wall_time: float = 0.0  # time.time()
    speaker: str = "unknown"  # "self" | person_id | "unknown"
    addressee: str = "self"  # who is being spoken to
    raw_text: str = ""
    # ASR metadata
    asr_confidence: float = 1.0  # [0,1] — 1.0 for keyboard input
    is_question: bool = False
    is_command: bool = False
    # Social context snapshot (filled by DialogueManager)
    speaker_distance_cm: float = 999.0
    speaker_engagement: float = 0.0
    speaker_gesture: str = "none"
    speaker_speaking: bool = False
    # Emotional tone (inferred from prosody or text)
    emotional_tone: str = "neutral"  # valence label
    # World references (objects / persons mentioned)
    referents: List[str] = field(default_factory=list)
    # Structurally resolved referents (pronoun → entity, built by Tier -1.1)
    resolved_referents: List["ResolvedReferent"] = field(default_factory=list)
    # Dialogue-internal
    speech_act: SpeechAct = SpeechAct.ASSERT  # inferred act of speaker
    topic: str = ""  # extracted topic keyword
    open_question: str = ""  # if is_question: the question text

    def describe(self) -> str:
        qa = "Q" if self.is_question else "A"
        return (
            f"[{self.speaker}→{self.addressee}] {qa} "
            f"conf={self.asr_confidence:.2f} "
            f"act={self.speech_act.value} "
            f'"{self.raw_text[:60]}"'
        )


# ─────────────────────────────────────────────────────────────
# Common Ground — shared context per interlocutor
# ─────────────────────────────────────────────────────────────


@dataclass
class CommonGround:
    """Mutual knowledge state between self and one person."""

    person_id: str
    # Current conversation topic (most recent substantive word/phrase)
    current_topic: str = ""
    # Last N topics discussed (for topic continuity)
    topic_history: List[str] = field(default_factory=list)
    # Facts mutually confirmed in this session
    confirmed_facts: List[str] = field(default_factory=list)
    # Open questions not yet answered
    open_questions: List[str] = field(default_factory=list)
    # Visible objects both parties can refer to
    shared_referents: List[str] = field(default_factory=list)
    # Last misunderstood term (for repair tracking)
    last_misunderstanding: str = ""
    # Preferred communication style (learned over time)
    preferred_style: str = "balanced"  # brief | balanced | explanatory
    # Structurally resolved referents, keyed by surface phrase ("das", "er", …)
    active_referents: Dict[str, "ResolvedReferent"] = field(default_factory=dict)
    # Binding: open question text → referent_id it concerns
    _question_referent_bindings: Dict[str, str] = field(default_factory=dict)
    # How many turns in current exchange
    turn_count: int = 0
    # Timestamp of last exchange
    last_exchange_tick: int = 0

    MAX_TOPICS = 20
    MAX_FACTS = 50
    MAX_QUESTIONS = 10

    def advance_topic(self, topic: str, tick: int) -> None:
        if topic and topic != self.current_topic:
            self.topic_history.append(self.current_topic)
            if len(self.topic_history) > self.MAX_TOPICS:
                self.topic_history = self.topic_history[-self.MAX_TOPICS :]
            self.current_topic = topic
        self.last_exchange_tick = tick
        self.turn_count += 1

    def confirm_fact(self, fact: str) -> None:
        if fact not in self.confirmed_facts:
            self.confirmed_facts.append(fact)
            if len(self.confirmed_facts) > self.MAX_FACTS:
                self.confirmed_facts = self.confirmed_facts[-self.MAX_FACTS :]

    def ask_question(self, q: str) -> None:
        if q not in self.open_questions:
            self.open_questions.append(q)
            if len(self.open_questions) > self.MAX_QUESTIONS:
                self.open_questions.pop(0)

    def ask_question_about(self, q: str, referent: "ResolvedReferent") -> None:
        """Ask an open question bound to a specific referent.

        Stores the question normally and records the binding so that when the
        question is answered the referent context is available.
        """
        self.ask_question(q)
        self._question_referent_bindings[q] = referent.referent_id

    def referent_for_question(self, q: str) -> Optional[str]:
        """Return the referent_id bound to an open question, if any."""
        return self._question_referent_bindings.get(q)

    def resolve_question(self, q: str) -> None:
        self.open_questions = [oq for oq in self.open_questions if oq != q]
        self._question_referent_bindings.pop(q, None)

    def update_shared_referents(self, world: "WorldState") -> None:
        """Refresh shared referents from currently visible entities."""
        refs: List[str] = []
        for pid, person in world.persons.items():
            refs.append(f"person:{pid}")
        for oid, obj in world.objects.items():
            refs.append(f"object:{obj.label}")
        self.shared_referents = refs[:20]

    def update_referent(self, ref: "ResolvedReferent") -> None:
        """Store or overwrite a resolved referent by its surface phrase."""
        self.active_referents[ref.source_phrase] = ref

    def get_most_salient_referent(self) -> "Optional[ResolvedReferent]":
        """Return the active referent with the highest salience score."""
        if not self.active_referents:
            return None
        return max(self.active_referents.values(), key=lambda r: r.salience)

    def background_old_referents(self, current_tick: int, ttl: int = 120) -> None:
        """Mark referents not updated recently as backgrounded."""
        for ref in self.active_referents.values():
            if (current_tick - ref.tick) > ttl and ref.discourse_status == "active":
                ref.discourse_status = "backgrounded"

    def to_dict(self) -> Dict:
        return {
            "person_id": self.person_id,
            "current_topic": self.current_topic,
            "topic_history": self.topic_history[-10:],
            "confirmed_facts": self.confirmed_facts[-20:],
            "open_questions": self.open_questions,
            "shared_referents": self.shared_referents[:10],
            "last_misunderstanding": self.last_misunderstanding,
            "preferred_style": self.preferred_style,
            "active_referents": {k: v.to_dict() for k, v in self.active_referents.items()},
            "question_referent_bindings": dict(self._question_referent_bindings),
            "turn_count": self.turn_count,
            "last_exchange_tick": self.last_exchange_tick,
        }

    @staticmethod
    def from_dict(d: Dict) -> "CommonGround":
        cg = CommonGround(person_id=d.get("person_id", ""))
        cg.current_topic = d.get("current_topic", "")
        cg.topic_history = d.get("topic_history", [])
        cg.confirmed_facts = d.get("confirmed_facts", [])
        cg.open_questions = d.get("open_questions", [])
        cg.shared_referents = d.get("shared_referents", [])
        cg.last_misunderstanding = d.get("last_misunderstanding", "")
        cg.preferred_style = d.get("preferred_style", "balanced")
        raw_refs = d.get("active_referents", {})
        for phrase, rd in raw_refs.items():
            if isinstance(rd, dict):
                from dataclasses import fields as _dfields
                cg.active_referents[phrase] = ResolvedReferent(
                    referent_id=rd.get("referent_id", ""),
                    referent_type=rd.get("referent_type", "unknown"),
                    source_phrase=rd.get("source_phrase", phrase),
                    confidence=float(rd.get("confidence", 0.5)),
                    resolution_source=rd.get("resolution_source", "unknown"),
                    tick=int(rd.get("tick", 0)),
                    discourse_status=rd.get("discourse_status", "active"),
                    salience=float(rd.get("salience", 0.0)),
                )
        cg._question_referent_bindings = dict(d.get("question_referent_bindings", {}))
        cg.turn_count = d.get("turn_count", 0)
        cg.last_exchange_tick = d.get("last_exchange_tick", 0)
        return cg


# ─────────────────────────────────────────────────────────────
# Dialogue State — per-person exchange state
# ─────────────────────────────────────────────────────────────


class DialoguePhase(Enum):
    IDLE = "idle"
    OPENING = "opening"
    ACTIVE = "active"
    CLOSING = "closing"
    SUSPENDED = "suspended"  # person walked away but may return


@dataclass
class DialogueState:
    """Tracks the phase and history of one conversation."""

    person_id: str
    phase: DialoguePhase = DialoguePhase.IDLE
    turns: Deque[DialogueTurn] = field(default_factory=lambda: deque(maxlen=50))
    started_tick: int = 0
    last_turn_tick: int = 0
    # Was our last output understood? (inferred from next human turn)
    last_understood: bool = True
    # Consecutive repair attempts (>2 = abandon attempt)
    repair_count: int = 0

    def add_turn(self, turn: DialogueTurn) -> None:
        self.turns.append(turn)
        self.last_turn_tick = turn.tick
        if self.phase == DialoguePhase.IDLE:
            self.phase = DialoguePhase.OPENING
            self.started_tick = turn.tick
        elif self.phase == DialoguePhase.OPENING:
            self.phase = DialoguePhase.ACTIVE

    def most_recent_referent(self) -> Optional["ResolvedReferent"]:
        """Scan recent turns backwards and return the most recently resolved
        referent (highest salience among active ones), or None.

        This enables multi-turn referent continuity: a follow-up turn that
        contains no pronoun can still inherit the referent from the previous
        turn.
        """
        for turn in reversed(list(self.turns)):
            if turn.resolved_referents:
                active = [
                    r
                    for r in turn.resolved_referents
                    if r.discourse_status in ("active", "backgrounded")
                ]
                if active:
                    return max(active, key=lambda r: r.salience)
        return None

    def referent_history(self, n: int = 5) -> List["ResolvedReferent"]:
        """Return up to `n` most recent resolved referents across turns.

        Ordered newest-first, de-duplicated by referent_id.
        """
        seen_ids: set = set()
        result: List["ResolvedReferent"] = []
        for turn in reversed(list(self.turns)):
            for rr in reversed(turn.resolved_referents):
                if rr.referent_id not in seen_ids:
                    seen_ids.add(rr.referent_id)
                    result.append(rr)
                    if len(result) >= n:
                        return result
        return result

    def check_timeout(self, current_tick: int, timeout: int = 200) -> None:
        """If no turn for `timeout` ticks, suspend the dialogue."""
        if self.phase in (DialoguePhase.ACTIVE, DialoguePhase.OPENING):
            if (current_tick - self.last_turn_tick) > timeout:
                self.phase = DialoguePhase.SUSPENDED


# ─────────────────────────────────────────────────────────────
# Utterance Plan — what to say and how to say it
# ─────────────────────────────────────────────────────────────


@dataclass
class UtterancePlan:
    """Bundles the AI's planned output with motor/prosodic cues."""

    text: str = ""
    speech_act: SpeechAct = SpeechAct.ASSERT
    addressee: str = ""
    # Prosodic hints for TTS
    pitch_shift: float = 0.0  # [-1, 1] — lower/higher
    speed_factor: float = 1.0  # 0.8=slow, 1.2=fast
    emphasis_words: List[str] = field(default_factory=list)
    # Motor cues for RobotController
    gaze_at_person: bool = True
    jaw_sync: bool = True  # sync jaw to TTS duration
    head_nod: bool = False  # nod while speaking
    # Deliberation gap — silence before TTS starts (simulates human "thinking time")
    deliberation_delay_ms: int = 0  # 0 = no gap; 200-1500 for natural pauses
    # Beat position: tick offset from utterance start where a gesture beat fits
    beat_tick: int = 0  # 0 = no beat computed yet
    # Metadata
    confidence: float = 1.0  # [0,1] how confident the system is
    tick: int = 0

    def describe(self) -> str:
        return (
            f"[utter] act={self.speech_act.value} "
            f'conf={self.confidence:.2f} "{self.text[:60]}"'
        )


# ─────────────────────────────────────────────────────────────
# Speech Act Planner — decides what communicative action to take
# ─────────────────────────────────────────────────────────────


class SpeechActPlanner:
    """
    Given the current dialogue state, common ground, consciousness state,
    and incoming turn, decide what speech act the AI should perform next.

    This is NOT about content generation — it's about communicative intent:
    should I answer, ask, confirm, repair, stay silent, or hedge?
    """

    # Max repair attempts before giving up on a clarification
    MAX_REPAIR = 3

    def plan(
        self,
        incoming: Optional[DialogueTurn],
        dialogue: DialogueState,
        ground: CommonGround,
        cs_state: Optional[object] = None,
        asr_confidence: float = 1.0,
        body_urgency: float = 0.0,
        comm_drive: float = 0.0,
        tom_strategy: Optional[Dict] = None,
        known_concepts: Optional[set] = None,
    ) -> SpeechAct:
        """
        Select the most appropriate speech act for the next turn.

        Priority chain:
          1. Body urgency  → OBSERVE (report internal state)
          2. Low ASR conf  → REPAIR  (ask for repetition)
          3. Greeting       → GREET
          4. Question       → ASSERT  (answer) or HESITATE (uncertain)
          5. Statement      → CONFIRM / DENY / ASK follow-up
          6. Low comm_drive → SILENCE (nothing to say)
          7. Open questions → ASK

        ToM biasing: low trust/success → safer acts (BACKCHANNEL/HESITATE);
        high trust + success history → more assertive acts (ASSERT/ASK).
        """
        # ── ToM-guided speech act biasing ────────────────────────────────
        # Theory of mind shapes WHICH communicative act to use, not just delivery.
        _tom_trust = float(tom_strategy.get("trust", 0.5)) if tom_strategy else 0.5
        _tom_success = (
            float(tom_strategy.get("success_rate", 0.5)) if tom_strategy else 0.5
        )
        _tom_model_conf = (
            float(tom_strategy.get("model_confidence", 1.0)) if tom_strategy else 1.0
        )
        # Low trust/success → prefer safe, minimal acts (BACKCHANNEL, HESITATE)
        _low_social_conf = (
            _tom_trust < 0.35 or _tom_success < 0.3 or _tom_model_conf < 0.2
        )
        # High trust + good success history → be more assertive/inquisitive
        _high_social_conf = _tom_trust > 0.65 and _tom_success > 0.6

        if incoming is None:
            # No incoming turn — check if we want to initiate
            if body_urgency > 0.5:
                return SpeechAct.OBSERVE
            if comm_drive > 0.6 and ground.open_questions:
                return SpeechAct.ASK
            # High social confidence lowers threshold; low social conf raises it
            _init_threshold = (
                0.65 if _high_social_conf else (0.85 if _low_social_conf else 0.75)
            )
            if comm_drive > _init_threshold:
                return SpeechAct.BACKCHANNEL if _low_social_conf else SpeechAct.ASSERT
            return SpeechAct.SILENCE

        # 1. Body urgency overrides social protocol
        if body_urgency > 0.7:
            return SpeechAct.OBSERVE

        # 2. Low ASR confidence → repair
        if asr_confidence < 0.4 and dialogue.repair_count < self.MAX_REPAIR:
            dialogue.repair_count += 1
            return SpeechAct.REPAIR

        # Reset repair count on successful recognition
        if asr_confidence >= 0.5:
            dialogue.repair_count = 0

        # 2c. Ambiguous referent → repair (prevent silent misidentification)
        # If the incoming turn contains a pronoun that was resolved with low
        # confidence AND no fallback from history was available, trigger a
        # targeted clarification question instead of proceeding on a wrong
        # binding.
        if incoming is not None and dialogue.repair_count < self.MAX_REPAIR:
            _low_conf_refs = [
                rr
                for rr in incoming.resolved_referents
                if rr.confidence < 0.35 and rr.discourse_status == "active"
            ]
            if _low_conf_refs:
                dialogue.repair_count += 1
                # Record the ambiguous phrase in common ground for repair text
                ground.last_misunderstanding = _low_conf_refs[0].source_phrase
                return SpeechAct.REPAIR

        # 2b. Semantic confusion detection
        # If the incoming text contains mostly unknown tokens (not in our known
        # concept space and not in common ground), initiate repair once per
        # REPAIR budget — this catches domain-confusion not caught by ASR score.
        if (
            incoming is not None
            and dialogue.repair_count < self.MAX_REPAIR
            and known_concepts is not None
            and len(known_concepts) > 0
        ):
            _in_toks = {
                w.lower() for w in incoming.raw_text.split() if len(w) > 4
            }
            _cg_toks = {
                tok
                for fact in ground.confirmed_facts
                for tok in fact.lower().split()
                if len(tok) > 4
            }
            _known = known_concepts | _cg_toks
            _unknown_ratio = (
                len(_in_toks - _known) / max(len(_in_toks), 1)
            )
            if _unknown_ratio > 0.75 and len(_in_toks) >= 4:
                ground.last_misunderstanding = incoming.raw_text[:60]
                dialogue.repair_count += 1
                return SpeechAct.REPAIR

        # 3. Opening / greeting
        if dialogue.phase == DialoguePhase.OPENING:
            return SpeechAct.GREET

        # 4. Incoming question → answer or hedge
        if incoming.is_question:
            if ground.open_questions:
                ground.resolve_question(incoming.raw_text)
            # Low social confidence: hedge rather than asserting directly
            if _low_social_conf:
                return SpeechAct.HESITATE
            return SpeechAct.ASSERT

        # 5. Incoming assertion → confirm, deny, or ask follow-up
        if incoming.speech_act == SpeechAct.ASSERT:
            # If the topic is new, ask about it
            if incoming.topic and incoming.topic != ground.current_topic:
                # Low social confidence: acknowledge the shift but don't interrogate yet
                return SpeechAct.BACKCHANNEL if _low_social_conf else SpeechAct.ASK
            return SpeechAct.CONFIRM

        # 6. Backchannel if nothing else
        _backchannel_threshold = 0.2 if _high_social_conf else 0.3
        if comm_drive < _backchannel_threshold:
            return SpeechAct.BACKCHANNEL

        # Low social confidence: prefer acknowledging over asserting until trust builds
        if _low_social_conf:
            return SpeechAct.BACKCHANNEL

        return SpeechAct.ASSERT


# ─────────────────────────────────────────────────────────────
# Dialogue Manager — orchestrates the full dialogue pipeline
# ─────────────────────────────────────────────────────────────


class DialogueManager:
    """
    Central dialogue orchestrator.

    Per tick:
      1. Processes incoming ASR text into DialogueTurns
      2. Maintains DialogueState + CommonGround per interlocutor
      3. Plans speech acts via SpeechActPlanner
      4. Produces UtterancePlans for SpeechOutput

    Does NOT generate response text — that remains in ConsciousnessCore.
    Instead provides the structured context (speech act, common ground,
    referents) that consciousness uses to formulate responses.
    """

    DIALOGUE_TIMEOUT_TICKS = 300  # suspend after 300 ticks of silence
    DEFAULT_WPM = 170  # assumed words per minute for beat timing

    def __init__(self) -> None:
        self._dialogues: Dict[str, DialogueState] = {}
        self._grounds: Dict[str, CommonGround] = {}
        self._planner = SpeechActPlanner()
        self._turn_log: Deque[DialogueTurn] = deque(maxlen=200)
        self._pending_utterance: Optional[UtterancePlan] = None
        # Current active person (primary interlocutor)
        self._active_person: Optional[str] = None
        # Outcome tracking: last self-turn per person for post-hoc derivation
        self._last_self_per_person: Dict[str, DialogueTurn] = {}
        self._pending_outcomes: Dict[str, str] = {}
        # Question patterns for is_question detection
        self._q_de = {
            "wer",
            "was",
            "wo",
            "wann",
            "warum",
            "wie",
            "wieso",
            "weshalb",
            "woher",
            "wohin",
            "welch",
            "wieviel",
            "ob",
        }
        self._q_en = {
            "who",
            "what",
            "where",
            "when",
            "why",
            "how",
            "which",
            "whose",
            "whom",
            "does",
            "did",
            "can",
            "could",
            "would",
            "should",
            "is",
            "are",
            "will",
            "have",
            "do",
        }

    # ── Public: process incoming speech ─────────────────────

    def process_incoming(
        self,
        raw_text: str,
        tick: int,
        speaker: str = "user",
        speaker_id: str = "",
        asr_confidence: float = 1.0,
        world: Optional["WorldState"] = None,
        social: Optional["SocialManager"] = None,
    ) -> DialogueTurn:
        """
        Convert raw ASR text into a structured DialogueTurn and update
        dialogue state + common ground.
        """
        # Accept both 'speaker' and 'speaker_id' for flexibility
        _spk = speaker_id or speaker
        turn = DialogueTurn(
            tick=tick,
            wall_time=time.time(),
            speaker=_spk,
            addressee="self",
            raw_text=raw_text,
            asr_confidence=asr_confidence,
        )

        # Enrich with social context
        if world is not None:
            person = world.persons.get(speaker_id)
            if person is None:
                # Try numeric matching
                for pid, p in world.persons.items():
                    if str(pid) == str(speaker_id) or p.face_visible:
                        person = p
                        break
            if person is not None:
                turn.speaker_distance_cm = person.distance_cm
                turn.speaker_engagement = person.engagement_score
                turn.speaker_gesture = person.gesture
                turn.speaker_speaking = person.speaking

        # Detect question
        turn.is_question = self._detect_question(raw_text)
        if turn.is_question:
            turn.speech_act = SpeechAct.ASK
            turn.open_question = raw_text

        # Extract topic (longest substantive word ≥ 4 chars)
        turn.topic = self._extract_topic(raw_text)

        # Extract referents from visible world
        if world is not None:
            turn.referents = self._extract_referents(raw_text, world)

        # ── Pronoun resolution ──────────────────────────────────────────
        # Resolve any pronoun / demonstrative in the incoming text to a
        # concrete world entity and store the result as a ResolvedReferent
        # on the DialogueTurn.  Ambiguous resolutions (conf < 0.35) trigger
        # a repair question so that a silent misidentification never occurs.
        _resolved_rrs: List[ResolvedReferent] = []
        if world is not None:
            _PRN_TRIGGERS = {
                "das", "dies", "dieses", "jenes", "er", "sie", "ihn", "ihm",
                "it", "this", "that", "he", "him", "she", "her", "they",
                "dort", "da", "there", "here",
            }
            _text_lc = raw_text.lower()
            _tokens = [t.strip(".,!?;:\"'") for t in _text_lc.split()]
            for _phrase in _tokens:
                if _phrase not in _PRN_TRIGGERS:
                    continue
                _ent_key = world.resolve_reference(
                    _phrase,
                    current_tick=tick,
                    topic_tokens=[
                        w.strip(".,!?;:\"'")
                        for w in raw_text.lower().split()
                        if len(w) >= 4
                    ],
                )
                if _ent_key is None:
                    # Try to inherit from dialogue history (multi-turn continuity)
                    _ds_probe = self._dialogues.get(speaker_id)
                    if _ds_probe is not None:
                        _inherited = _ds_probe.most_recent_referent()
                        if _inherited is not None:
                            _ent_key = _inherited.referent_id
                if _ent_key is not None:
                    _sal = world.compute_salience(tick, topic_tokens=[
                        w.strip(".,!?;:\"'")
                        for w in raw_text.lower().split()
                        if len(w) >= 4
                    ])
                    _conf = float(
                        _sal.get(_ent_key, 0.6)
                        if isinstance(_sal, dict)
                        else 0.6
                    )
                    _rtype = (
                        _ent_key.split(":")[0]
                        if ":" in _ent_key
                        else "unknown"
                    )
                    _rr = ResolvedReferent(
                        referent_id=_ent_key,
                        referent_type=_rtype,
                        source_phrase=_phrase,
                        confidence=_conf,
                        resolution_source="world_model",
                        tick=tick,
                        discourse_status="active",
                        salience=_conf,
                    )
                    _resolved_rrs.append(_rr)
                    world.note_mentioned(_ent_key)
                    break  # one pronoun resolved per turn is sufficient
        turn.resolved_referents = _resolved_rrs

        # Derive outcome from prior self-turn for this person (Finding #2)
        _prior_self = self._last_self_per_person.get(_spk)
        if _prior_self is not None:
            _outcome = self._derive_outcome(_prior_self, turn)
            if _outcome:
                self._pending_outcomes[_spk] = _outcome

        # Update dialogue state
        ds = self._get_or_create_dialogue(speaker_id, tick)
        ds.add_turn(turn)

        # Update common ground
        cg = self._get_or_create_ground(speaker_id)
        if turn.topic:
            cg.advance_topic(turn.topic, tick)
        if turn.is_question:
            # If the question involves a resolved referent, bind it
            if _resolved_rrs:
                cg.ask_question_about(raw_text, _resolved_rrs[0])
            else:
                cg.ask_question(raw_text)
        # Store resolved referents in CommonGround for multi-turn access
        for _rr in _resolved_rrs:
            cg.update_referent(_rr)
        if world is not None:
            cg.update_shared_referents(world)

        self._active_person = speaker_id
        self._turn_log.append(turn)

        # Notify social manager of speech event
        if social is not None:
            try:
                social.person_spoke(speaker_id, tick, len(raw_text.split()), raw_text)
            except Exception:
                pass

        return turn

    # ── Public: plan output speech act ──────────────────────

    def plan_response(
        self,
        tick: int,
        last_incoming: Optional[DialogueTurn] = None,
        comm_drive: float = 0.0,
        body_urgency: float = 0.0,
        tom_strategy: Optional[Dict] = None,
        known_concepts: Optional[set] = None,
    ) -> SpeechAct:
        """Decide what speech act to perform next."""
        pid = self._active_person
        if pid is None:
            return SpeechAct.SILENCE

        ds = self._dialogues.get(pid)
        cg = self._grounds.get(pid)
        if ds is None or cg is None:
            return SpeechAct.SILENCE

        asr_conf = last_incoming.asr_confidence if last_incoming else 1.0
        return self._planner.plan(
            last_incoming,
            ds,
            cg,
            asr_confidence=asr_conf,
            body_urgency=body_urgency,
            comm_drive=comm_drive,
            tom_strategy=tom_strategy,
            known_concepts=known_concepts,
        )

    # ── Public: build utterance plan ────────────────────────

    def build_utterance(
        self,
        text: str,
        addressee: str = "",
        consciousness: Optional["ConsciousnessCore"] = None,
        speech_act: Optional[SpeechAct] = None,
        tick: int = 0,
        confidence: float = 1.0,
        tom_strategy: Optional[Dict] = None,
    ) -> UtterancePlan:
        """Package response text with the planned speech act + motor cues.

        If speech_act is None, auto-detect from text and consciousness state.
        """
        if speech_act is None:
            # Auto-detect speech act from text content
            _lower = text.lower().strip()
            if self._detect_question(text):
                speech_act = SpeechAct.ASK
            elif any(
                _lower.startswith(g) for g in ("hallo", "hi ", "hey", "guten", "hello")
            ):
                speech_act = SpeechAct.GREET
            elif any(
                _lower.startswith(g) for g in ("tschüss", "bye", "auf wieder", "bis ")
            ):
                speech_act = SpeechAct.FAREWELL
            elif any(
                _lower.startswith(g)
                for g in ("ja", "genau", "stimmt", "richtig", "yes", "right", "okay")
            ):
                speech_act = SpeechAct.CONFIRM
            elif any(
                _lower.startswith(g) for g in ("nein", "no ", "falsch", "stimmt nicht")
            ):
                speech_act = SpeechAct.DENY
            else:
                speech_act = SpeechAct.ASSERT
        plan = UtterancePlan(
            text=text,
            speech_act=speech_act,
            addressee=addressee or self._active_person or "",
            tick=tick,
            confidence=confidence,
        )

        # Adjust prosody based on speech act
        if speech_act == SpeechAct.ASK or speech_act == SpeechAct.REPAIR:
            plan.pitch_shift = 0.15  # rising intonation
            plan.speed_factor = 0.9  # slightly slower for questions
        elif speech_act == SpeechAct.HESITATE:
            plan.speed_factor = 0.8
            plan.confidence = min(confidence, 0.5)
        elif speech_act == SpeechAct.GREET:
            plan.head_nod = True
        elif speech_act == SpeechAct.BACKCHANNEL:
            plan.head_nod = True
            plan.gaze_at_person = True
        elif speech_act == SpeechAct.ASSERT:
            plan.head_nod = False
            plan.gaze_at_person = True

        # Low confidence → hesitation cue
        if confidence < 0.4:
            plan.speech_act = SpeechAct.HESITATE

        # ── Emotion-driven prosody ────────────────────────────────────────
        # Apply prosodic modulation based on emotional state passed via
        # consciousness object or directly.  Operates on top of speech-act
        # hints above — only adjusts within safe limits.
        _em_state = None
        if consciousness is not None:
            _em_state = getattr(consciousness, "_last_emotion_snapshot", None)
            if _em_state is None:
                # Fallback: read from the brain ref attached to consciousness
                _brain_ref_dm = getattr(consciousness, "_brain_ref", None)
                if _brain_ref_dm is not None:
                    _em_state = getattr(_brain_ref_dm, "emotion_state", None)
        if _em_state is not None:
            _dom = _em_state.dominant() if hasattr(_em_state, "dominant") else "calm"
            _stress = float(getattr(_em_state, "stress", 0.0))
            _fatigue = float(getattr(_em_state, "fatigue", 0.0))
            _joy = float(getattr(_em_state, "joy", 0.0))
            _curiosity = float(getattr(_em_state, "curiosity", 0.0))
            if _dom == "stress" or _stress > 0.6:
                plan.speed_factor = min(1.3, plan.speed_factor * 1.12)  # faster
                plan.pitch_shift = max(-0.4, plan.pitch_shift - 0.05)
            elif _dom == "fatigue" or _fatigue > 0.65:
                plan.speed_factor = max(0.75, plan.speed_factor * 0.88)  # slower
                plan.pitch_shift = max(-0.4, plan.pitch_shift - 0.1)    # lower pitch
            elif _dom == "joy" or _joy > 0.6:
                plan.pitch_shift = min(0.4, plan.pitch_shift + 0.08)   # higher
                plan.speed_factor = min(1.2, plan.speed_factor * 1.05)
            elif _dom == "curiosity" or _curiosity > 0.5:
                plan.pitch_shift = min(0.4, plan.pitch_shift + 0.05)
            elif _dom == "sadness":
                plan.speed_factor = max(0.75, plan.speed_factor * 0.9)
                plan.pitch_shift = max(-0.4, plan.pitch_shift - 0.08)

        # ── ToM-driven prosody: adapt delivery to addressee ──
        if tom_strategy:
            _tom_tone = tom_strategy.get("tone", "")
            _tom_style = tom_strategy.get("style", "")
            _tom_trust = tom_strategy.get("trust", 0.5)
            # Tone adjustments
            if _tom_tone == "warm" or _tom_tone == "friendly":
                plan.pitch_shift = max(-0.3, plan.pitch_shift - 0.05)
                plan.speed_factor = min(1.2, plan.speed_factor * 1.05)
                plan.head_nod = True
            elif _tom_tone == "cautious" or _tom_tone == "formal":
                plan.speed_factor = min(1.0, plan.speed_factor * 0.92)
            elif _tom_tone == "gentle":
                plan.pitch_shift = max(-0.3, plan.pitch_shift - 0.1)
                plan.speed_factor = min(1.0, plan.speed_factor * 0.9)
            # Style adjustments
            if _tom_style == "brief":
                plan.speed_factor = min(1.3, plan.speed_factor * 1.1)
            elif _tom_style == "explanatory":
                plan.speed_factor = max(0.8, plan.speed_factor * 0.93)
            # Low trust → more cautious delivery
            if isinstance(_tom_trust, (int, float)) and _tom_trust < 0.3:
                plan.speed_factor = max(0.8, plan.speed_factor * 0.9)
                plan.gaze_at_person = True  # maintain gaze to build trust

            # Update common ground preferred_style from ToM
            if _tom_style and addressee:
                _cg = self._grounds.get(addressee)
                if _cg is not None:
                    _cg.preferred_style = _tom_style

        # ── Beat position for gesture-speech synchronisation (Domain D) ──
        # Estimate TTS duration (WPM → seconds) and place the gesture beat
        # at ~35% through the utterance — on the first stressed phrase.
        # Stored as a tick offset from the utterance start so task_executive
        # can schedule MirrorGesture / ExpressEmotion at the right moment.
        try:
            _wpm = max(80, int(self.DEFAULT_WPM * plan.speed_factor)) if hasattr(self, "DEFAULT_WPM") else 170
            _words = max(1, len(plan.text.split()))
            _dur_s = (_words / _wpm) * 60.0
            plan.beat_tick = int(_dur_s * 0.35 * 40)   # 40 ticks/s × 35% offset
        except Exception:
            plan.beat_tick = 0

        self._pending_utterance = plan
        return plan

    # ── Public: mark output delivered ──────────────────────

    def mark_output_delivered(self, tick_or_plan=None, tick: int = 0) -> None:
        """Record that the AI spoke — updates state as a self-turn.

        Can be called as:
          mark_output_delivered(tick)           — uses self._pending_utterance
          mark_output_delivered(plan, tick)     — explicit plan
        """
        if isinstance(tick_or_plan, int):
            tick = tick_or_plan
            plan = self._pending_utterance
        else:
            plan = tick_or_plan
        if plan is None:
            return
        self_turn = DialogueTurn(
            tick=tick,
            wall_time=time.time(),
            speaker="self",
            addressee=plan.addressee,
            raw_text=plan.text,
            speech_act=plan.speech_act,
            topic=self._extract_topic(plan.text),
        )

        pid = plan.addressee or self._active_person
        if pid:
            ds = self._dialogues.get(pid)
            if ds:
                ds.add_turn(self_turn)
            cg = self._grounds.get(pid)
            if cg:
                if self_turn.topic:
                    cg.advance_topic(self_turn.topic, tick)
                # Carry active referents into the self-turn for history continuity
                # so that multi-turn referent_history() spans both sides.
                active_rrs = [
                    rr
                    for rr in cg.active_referents.values()
                    if rr.discourse_status == "active"
                ]
                if active_rrs:
                    self_turn.resolved_referents = list(active_rrs)
                # Semantically resolve open questions: only when the utterance
                # shares topic tokens with the question (not a blanket resolve).
                if plan.speech_act in (SpeechAct.ASSERT, SpeechAct.CONFIRM):
                    _utt_toks = {t.lower() for t in plan.text.split() if len(t) > 3}
                    for _oq in list(cg.open_questions):
                        _oq_toks = {t.lower() for t in _oq.split() if len(t) > 3}
                        if not _oq_toks or _utt_toks & _oq_toks:
                            cg.resolve_question(_oq)
                            break  # one resolution per turn
            # Track last self-turn per person for outcome derivation
            self._last_self_per_person[pid] = self_turn

        self._turn_log.append(self_turn)
        self._pending_utterance = None

    # ── Public: generate repair text ────────────────────────

    def _derive_outcome(
        self, self_turn: "DialogueTurn", incoming: "DialogueTurn"
    ) -> str:
        """Derive communicative outcome: how did the person respond to our last utterance?

        Returns one of: 'repair_requested', 'understood', 'topic_shifted',
        'delayed_response', 'disengaged', 'minimal_response', 'acknowledged'.
        """
        _raw = incoming.raw_text.lower()
        # Person explicitly asked for clarification
        _repair_kws = {
            "nochmal",
            "wiederholen",
            "was?",
            "huh?",
            "repeat",
            "pardon",
            "nicht verstanden",
            "verstehe nicht",
            "unclear",
            "was hast",
        }
        if incoming.speech_act == SpeechAct.REPAIR or any(
            kw in _raw for kw in _repair_kws
        ):
            return "repair_requested"
        # Person acknowledged (strong understanding signal)
        _ack_kws = {
            "ja",
            "okay",
            "ok",
            "stimmt",
            "genau",
            "verstehe",
            "yes",
            "right",
            "correct",
            "got it",
            "alright",
        }
        if any(kw in _raw.split() for kw in _ack_kws):
            return "understood"
        # Topic shifted significantly (disengagement signal)
        _self_topic = self_turn.topic or ""
        _in_topic = incoming.topic or ""
        if _self_topic and _in_topic and _self_topic != _in_topic:
            _st = set(_self_topic.lower().split())
            _it = set(_in_topic.lower().split())
            if not (_st & _it):
                return "topic_shifted"
        # Very long pause before response → possible ignore
        if incoming.tick - self_turn.tick > self.DIALOGUE_TIMEOUT_TICKS:
            return "delayed_response"
        # Nonverbal disengagement: low engagement score from sensory system
        if incoming.speaker_engagement < 0.2 and incoming.speaker_engagement > 0.0:
            return "disengaged"
        # Minimal response: very short reply without clear acknowledgment
        _response_words = [w for w in _raw.split() if len(w) > 1]
        if len(_response_words) <= 3:
            _ack_kws_short = {"ja", "ok", "okay", "yes", "nein", "no", "mm", "hmm"}
            if not any(kw in _raw.split() for kw in _ack_kws_short):
                return "minimal_response"
        # Default: turn received, conversation continues
        return "acknowledged"

    def pop_outcomes(self) -> Dict[str, str]:
        """Return and clear accumulated dialogue outcomes (person_id → outcome_type)."""
        outcomes = dict(self._pending_outcomes)
        self._pending_outcomes.clear()
        return outcomes

    def generate_repair_text(self, lang: str = "de") -> str:
        """Create a clarification request.

        If the misunderstanding is a resolved pronoun (low confidence), name
        the candidate entity so the user can confirm or correct.
        """
        pid = self._active_person
        cg = self._grounds.get(pid) if pid else None
        ds = self._dialogues.get(pid) if pid else None

        # Check for ambiguous referent in the last incoming turn
        if cg is not None and cg.last_misunderstanding:
            _phrase = cg.last_misunderstanding
            # Does CommonGround have a candidate entity for this phrase?
            _cand_ref = cg.active_referents.get(_phrase)
            if _cand_ref is not None:
                _cand_label = _cand_ref.referent_id.split(":", 1)[-1]
                if lang == "de":
                    return (
                        f"Meinst du mit '{_phrase}' {_cand_label}? "
                        f"Oder etwas anderes?"
                    )
                else:
                    return (
                        f"When you say '{_phrase}', do you mean {_cand_label}? "
                        f"Or something else?"
                    )
            # No candidate — generic phrasing
            if lang == "de":
                return (
                    f"Ich bin mir nicht sicher, worauf '{_phrase}' sich bezieht. "
                    f"Kannst du das genauer sagen?"
                )
            else:
                return (
                    f"I'm not sure what '{_phrase}' refers to. "
                    f"Could you be more specific?"
                )

        # Generic repair (low ASR / semantic confusion)
        if lang == "de":
            return "Entschuldige, das habe ich nicht richtig verstanden. Kannst du das wiederholen?"
        else:
            return "Sorry, I didn't catch that. Could you repeat?"

    # ── Public: generate backchannel ────────────────────────

    def generate_backchannel(
        self,
        lang: str = "de",
        emotion: str = "calm",
        topic: str = "",
    ) -> str:
        """Context- and emotion-aware short acknowledgment signal.

        Called during active listening — produces a natural reactive token
        ("hmm", "ja", "ach so") suited to the current emotional tone.
        """
        import random

        _de = {
            "curiosity": ["Interessant.", "Ach so?", "Wirklich?", "Oh."],
            "joy": ["Ja!", "Schön.", "Gut.", "Prima."],
            "stress": ["Hmm.", "Okay.", "Ich hör zu."],
            "sadness": ["Ich verstehe.", "Mhm.", "Okay."],
            "surprise": ["Oh.", "Wirklich?", "Tatsächlich."],
            "calm": ["Ja.", "Mhm.", "Verstehe.", "Okay.", "Hmm."],
            "fatigue": ["Mhm.", "Okay.", "Hmm."],
        }
        _en = {
            "curiosity": ["Interesting.", "Oh?", "Really?", "Hmm."],
            "joy": ["Yes!", "Nice.", "Good.", "Great."],
            "stress": ["Hmm.", "Okay.", "I see."],
            "sadness": ["I understand.", "Mhm.", "Okay."],
            "surprise": ["Oh.", "Really?", "Indeed."],
            "calm": ["Yes.", "Mhm.", "I see.", "Okay.", "Hmm."],
            "fatigue": ["Mhm.", "Okay.", "Hmm."],
        }
        pool = (_de if lang == "de" else _en)
        options = pool.get(emotion, pool["calm"])
        return random.choice(options)

    def should_emit_backchannel(
        self,
        person_speaking_ticks: int,
        current_tick: int,
        min_interval: int = 40,
    ) -> bool:
        """Return True when a backchannel is due during active listening.

        Fires once per min_interval ticks while person has been speaking
        continuously.  Prevents flooding: only emits after at least
        min_interval ticks since the last backchannel.
        """
        if person_speaking_ticks < 15:
            return False
        last_bc = getattr(self, "_last_backchannel_tick", 0)
        return (current_tick - last_bc) >= min_interval

    def record_backchannel_sent(self, tick: int) -> None:
        """Mark that a backchannel was emitted at this tick."""
        self._last_backchannel_tick = tick

    # ── Public: tick — maintenance ──────────────────────────

    def tick(self, current_tick: int) -> None:
        """Periodic maintenance: check timeouts, decay referent salience."""
        for ds in self._dialogues.values():
            ds.check_timeout(current_tick, self.DIALOGUE_TIMEOUT_TICKS)
        # Decay referents that haven't been mentioned recently
        for cg in self._grounds.values():
            cg.background_old_referents(current_tick, ttl=120)

    # ── Public: queries ──────────────────────────────────────

    @property
    def active_ground(self) -> Optional[CommonGround]:
        if self._active_person:
            return self._grounds.get(self._active_person)
        return None

    @property
    def active_dialogue(self) -> Optional[DialogueState]:
        if self._active_person:
            return self._dialogues.get(self._active_person)
        return None

    @property
    def last_incoming_turn(self) -> Optional[DialogueTurn]:
        for turn in reversed(self._turn_log):
            if turn.speaker != "self":
                return turn
        return None

    def describe(self) -> str:
        n = len(self._dialogues)
        active = self._active_person or "none"
        phase = "idle"
        ds = self.active_dialogue
        if ds:
            phase = ds.phase.value
        cg = self.active_ground
        topic = cg.current_topic if cg else ""
        oq = len(cg.open_questions) if cg else 0
        return (
            f"dialogue persons={n} active={active} "
            f"phase={phase} topic='{topic}' open_q={oq}"
        )

    def dialogue_summary(self) -> Dict:
        """Compact summary for consciousness tick integration."""
        cg = self.active_ground
        ds = self.active_dialogue
        return {
            "active_person": self._active_person,
            "phase": ds.phase.value if ds else "idle",
            "turn_count": cg.turn_count if cg else 0,
            "current_topic": cg.current_topic if cg else "",
            "open_questions": len(cg.open_questions) if cg else 0,
            "last_misunderstanding": cg.last_misunderstanding if cg else "",
            "preferred_style": cg.preferred_style if cg else "balanced",
            "pending_utterance": self._pending_utterance is not None,
        }

    # ── Serialization ────────────────────────────────────────

    def to_dict(self) -> Dict:
        # Persist dialogue states alongside common ground so conversation
        # phase, repair_count, and recent turns survive restarts.
        dialogues_data = {}
        for pid, ds in self._dialogues.items():
            recent_turns = []
            for t in list(ds.turns)[-3:]:
                recent_turns.append(
                    {
                        "tick": t.tick,
                        "speaker": t.speaker,
                        "addressee": t.addressee,
                        "raw_text": t.raw_text[:200],
                        "speech_act": t.speech_act.value,
                        "is_question": t.is_question,
                    }
                )
            dialogues_data[pid] = {
                "phase": ds.phase.value,
                "started_tick": ds.started_tick,
                "last_turn_tick": ds.last_turn_tick,
                "repair_count": ds.repair_count,
                "last_understood": ds.last_understood,
                "recent_turns": recent_turns,
            }
        return {
            "grounds": {pid: cg.to_dict() for pid, cg in self._grounds.items()},
            "active_person": self._active_person,
            "dialogues": dialogues_data,
        }

    def load_from_dict(self, d: Dict) -> None:
        for pid, cg_data in d.get("grounds", {}).items():
            self._grounds[pid] = CommonGround.from_dict(cg_data)
        self._active_person = d.get("active_person")
        # Restore dialogue states
        for pid, ds_data in d.get("dialogues", {}).items():
            ds = DialogueState(
                person_id=pid,
                started_tick=ds_data.get("started_tick", 0),
                last_turn_tick=ds_data.get("last_turn_tick", 0),
                repair_count=ds_data.get("repair_count", 0),
                last_understood=ds_data.get("last_understood", True),
            )
            # Restore phase
            _phase_str = ds_data.get("phase", "idle")
            for p in DialoguePhase:
                if p.value == _phase_str:
                    ds.phase = p
                    break
            # Restore minimal recent turns
            for td in ds_data.get("recent_turns", []):
                _act = SpeechAct.ASSERT
                for sa in SpeechAct:
                    if sa.value == td.get("speech_act", "assert"):
                        _act = sa
                        break
                turn = DialogueTurn(
                    tick=td.get("tick", 0),
                    speaker=td.get("speaker", "unknown"),
                    addressee=td.get("addressee", "self"),
                    raw_text=td.get("raw_text", ""),
                    speech_act=_act,
                    is_question=td.get("is_question", False),
                )
                ds.turns.append(turn)
            self._dialogues[pid] = ds

    # ── Internals ─────────────────────────────────────────────

    def _get_or_create_dialogue(self, person_id: str, tick: int) -> DialogueState:
        if person_id not in self._dialogues:
            self._dialogues[person_id] = DialogueState(
                person_id=person_id, started_tick=tick
            )
        return self._dialogues[person_id]

    def _get_or_create_ground(self, person_id: str) -> CommonGround:
        if person_id not in self._grounds:
            self._grounds[person_id] = CommonGround(person_id=person_id)
        return self._grounds[person_id]

    def _detect_question(self, text: str) -> bool:
        """Heuristic: question mark or question word at start."""
        if "?" in text:
            return True
        words = text.lower().split()
        if not words:
            return False
        first = words[0].rstrip(".,!?")
        return first in self._q_de or first in self._q_en

    def _extract_topic(self, text: str) -> str:
        """Extract the most salient content word as topic."""
        _SKIP = {
            "nicht",
            "haben",
            "werden",
            "können",
            "sagen",
            "machen",
            "gehen",
            "kommen",
            "lassen",
            "wollen",
            "sollen",
            "müssen",
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
            "please",
            "think",
            "really",
            "actually",
            "maybe",
            "perhaps",
            "probably",
        }
        words = text.split()
        best = ""
        for w in words:
            clean = w.strip(".,!?;:\"'()[]{}").lower()
            if len(clean) >= 4 and clean not in _SKIP and clean.isalpha():
                if len(clean) > len(best):
                    best = clean
        return best

    def _extract_referents(self, text: str, world: "WorldState") -> List[str]:
        """Match words in text to visible world entities."""
        refs: List[str] = []
        text_lower = text.lower()
        # Check for person references
        for pid in world.persons:
            if str(pid) in text_lower or "person" in text_lower:
                refs.append(f"person:{pid}")
        # Check for object label matches
        for oid, obj in world.objects.items():
            if obj.label.lower() in text_lower:
                refs.append(f"object:{obj.label}")
        return refs[:5]
