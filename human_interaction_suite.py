"""
human_interaction_suite.py — 20-Module Human-Presence Layer

Implements humanly-inflected speech, relationship memory, and embodied presence
across three functional phases:

  Phase 1 — Speech & Conversation Style (Modules 1–5)
    1. PersonalSpeechSignatureEngine  — directness, sentence length, recurring phrases
    2. SubtextInterpreter             — social subtext detection (irritated, insecure, …)
    3. DisfluencyGenerator            — state-gated fillers, self-corrections, search pauses
    4. ContextCompressionSpeaker      — reply density vs. familiarity / conflict level
    5. ConversationalEnergyModel      — engaged / brief / exhausted / open / irritated

  Phase 2 — Memory, Relationship, Imperfect Recall (Modules 6–12)
    6. EmotionalMemoryLayer           — emotional traces per person/topic; affect reply focus
    7. RelationshipTrajectoryEngine   — phase progression: stranger → trusted → strained …
    8. SharedHistorySynthesizer       — selective shared-history hooks
    9. ExpectationTracker             — models expected interaction mode (help / repair / …)
   10. TrustCalibrationModel          — form, caution, initiative scale with trust
   11. ImperfectRecallModule          — precision degrades with fatigue + low trust
   12. BiasEngine                     — recency / familiarity / consistency distortions

  Phase 3 — Thinking Modes, Hidden Motives, Body, Presence (Modules 13–20)
   13. MoodDistortionFilter           — stress → narrow; joy → open (target_parts)
   14. OverthinkingUnderthinkingSwitch — overthinking → HESITATE; underthinking → 1 part
   15. CognitiveFatigueModule         — linguistic budget hard/soft limits
   16. HiddenMotivesLayer             — seek_rest → trim; be_liked → warm close
   17. ValueConflictEngine            — ≥2 conflicts → [P350ms] prosodic pause
   18. IdentityNarrativeDrift         — hardening / opening → opener/closer rewrites
   19. MicrobehaviorController        — head_tilt_bias, gaze_micro_variance → UtterancePlan
   20. PresenceSynchronizer           — timing_mode + sync_score → delay / speed / confidence

All modules are updated each tick via HumanInteractionSuite.update_tick() and each
conversation turn via observe_user_turn().  Modules 1–18 integrate into
consciousness.py (respond_to) in pre-/post-assembly blocks; Modules 19–20 feed
directly into the UtterancePlan in brain.py.
"""
from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, ClassVar, Deque, Dict, List, Optional


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-ZäöüÄÖÜß']+", text.lower())


def _sentences(text: str) -> List[str]:
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class PersonalSpeechSignatureEngine:
    favorite_words: Counter[str] = field(default_factory=Counter)
    phrase_fragments: Counter[str] = field(default_factory=Counter)
    avg_sentence_len: float = 8.0
    humor_style: str = "dry"
    directness: float = 0.5
    slang_level: float = 0.2
    catchphrases: List[str] = field(default_factory=list)

    def observe_reply(self, text: str) -> None:
        toks = _tokens(text)
        if toks:
            for tok in toks:
                if len(tok) >= 4:
                    self.favorite_words[tok] += 1
            sents = _sentences(text)
            if sents:
                avg = sum(len(_tokens(s)) for s in sents) / max(1, len(sents))
                self.avg_sentence_len = self.avg_sentence_len * 0.9 + avg * 0.1
            if any(tok in toks for tok in ("witz", "haha", "lustig", "funny", "joke")):
                self.humor_style = "playful"
            if any(tok in toks for tok in ("bitte", "vielleicht", "eventuell", "maybe", "perhaps")):
                self.directness = max(0.0, self.directness - 0.03)
            if any(tok in toks for tok in ("mach", "tu", "do", "stop", "jetzt", "now")):
                self.directness = min(1.0, self.directness + 0.04)
            if any(tok in toks for tok in ("moin", "jo", "naja", "okay", "yep")):
                self.slang_level = min(1.0, self.slang_level + 0.03)
            joined = " ".join(toks)
            for frag in ("ich denke", "ich glaube", "das wirkt", "für mich", "i think"):
                if frag in joined:
                    self.phrase_fragments[frag] += 1
            self.catchphrases = [p for p, _ in self.phrase_fragments.most_common(5)]


@dataclass
class DisfluencyGenerator:
    filler_rate: float = 0.05
    self_correction_rate: float = 0.04
    search_pause_rate: float = 0.08
    fragment_rate: float = 0.04

    def update(self, fatigue: float, trust: float, pressure: float) -> None:
        self.filler_rate = min(0.35, 0.03 + fatigue * 0.18 + pressure * 0.08)
        self.self_correction_rate = min(0.30, 0.02 + pressure * 0.15)
        self.search_pause_rate = min(0.40, 0.04 + fatigue * 0.16 + (1.0 - trust) * 0.10)
        self.fragment_rate = min(0.25, 0.02 + fatigue * 0.10)


@dataclass
class ContextCompressionSpeaker:
    compression_ratio: float = 1.0
    explanation_depth: str = "balanced"

    def update(self, familiarity: float, uncertainty: float, fatigue: float) -> None:
        if fatigue > 0.65:
            self.compression_ratio = 0.65
            self.explanation_depth = "brief"
        elif uncertainty > 0.6:
            self.compression_ratio = 1.15
            self.explanation_depth = "careful"
        elif familiarity > 0.55:
            self.compression_ratio = 0.75
            self.explanation_depth = "short"
        else:
            self.compression_ratio = 1.0
            self.explanation_depth = "balanced"


@dataclass
class SubtextCandidate:
    label: str
    confidence: float


@dataclass
class SubtextInterpreter:
    last_candidates: List[SubtextCandidate] = field(default_factory=list)

    def interpret(self, text: str) -> List[SubtextCandidate]:
        lc = text.lower().strip()
        candidates: List[SubtextCandidate] = []
        if lc in ("schon okay", "ist okay", "passt schon", "fine", "it's okay"):
            candidates.extend(
                [
                    SubtextCandidate("hurt", 0.35),
                    SubtextCandidate("annoyed", 0.30),
                    SubtextCandidate("disappointed", 0.25),
                    SubtextCandidate("actually_ok", 0.20),
                ]
            )
        if "..." in lc or lc.endswith(".") and len(_tokens(lc)) <= 3:
            candidates.append(SubtextCandidate("withholding", 0.35))
        if any(w in lc for w in ("klar", "sure", "whatever", "ja ja")):
            candidates.append(SubtextCandidate("irritated", 0.32))
        self.last_candidates = sorted(candidates, key=lambda c: -c.confidence)[:4]
        return list(self.last_candidates)


@dataclass
class ConversationalEnergyModel:
    enthusiasm: float = 0.5
    depletion: float = 0.0
    interest: float = 0.5
    irritation: float = 0.0

    def update(self, user_text: str, trust: float, fatigue: float) -> None:
        toks = _tokens(user_text)
        questions = user_text.count("?")
        self.interest = max(0.0, min(1.0, self.interest * 0.85 + min(1.0, (len(toks) / 12.0) + questions * 0.15) * 0.15))
        self.depletion = max(0.0, min(1.0, self.depletion * 0.85 + fatigue * 0.15))
        self.enthusiasm = max(0.0, min(1.0, trust * 0.45 + self.interest * 0.40 + (1.0 - self.depletion) * 0.15))
        self.irritation = max(0.0, min(1.0, (1.0 - trust) * 0.35 + self.depletion * 0.40))


@dataclass
class EmotionalMemoryTrace:
    tick: int
    person_id: Optional[int]
    topic: str
    dominant_emotion: str
    valence: float


@dataclass
class EmotionalMemoryLayer:
    traces: Deque[EmotionalMemoryTrace] = field(default_factory=lambda: deque(maxlen=200))

    def record(self, tick: int, person_id: Optional[int], topic: str, dominant_emotion: str, valence: float) -> None:
        if topic or person_id is not None:
            self.traces.append(EmotionalMemoryTrace(tick, person_id, topic[:80], dominant_emotion, valence))


@dataclass
class RelationshipTrajectoryEngine:
    phase: str = "stranger"
    tension: float = 0.0

    def update(self, trust: float, familiarity: float, conflict_rate: float) -> None:
        self.tension = conflict_rate
        if conflict_rate > 0.45:
            self.phase = "strained"
        elif trust > 0.7 and familiarity > 0.55:
            self.phase = "trusted"
        elif familiarity > 0.35:
            self.phase = "known"
        else:
            self.phase = "stranger"


@dataclass
class SharedHistorySynthesizer:
    recent_hooks: List[str] = field(default_factory=list)

    def update(self, interaction_log: List[str]) -> None:
        self.recent_hooks = interaction_log[-3:]


@dataclass
class ExpectationTracker:
    expected_mode: str = "neutral"

    def update(self, person_style: Dict[str, Any]) -> None:
        if person_style.get("clarity") == "high":
            self.expected_mode = "help"
        elif person_style.get("initiative") == "proactive":
            self.expected_mode = "closeness"
        elif person_style.get("formality") == "formal":
            self.expected_mode = "matter_of_fact"
        else:
            self.expected_mode = "neutral"


@dataclass
class TrustCalibrationModel:
    caution_multiplier: float = 1.0
    repair_bias: float = 0.0

    def update(self, trust: float, recent_outcome: float) -> None:
        self.caution_multiplier = 1.0 + max(0.0, 0.55 - trust) * 1.4
        self.repair_bias = max(0.0, 0.5 - recent_outcome) * 1.2


@dataclass
class ImperfectRecallModule:
    precision: float = 0.92
    correction_bias: float = 0.15

    def update(self, fatigue: float, trust: float) -> None:
        self.precision = max(0.55, 0.95 - fatigue * 0.25 - (1.0 - trust) * 0.08)
        self.correction_bias = min(0.6, 0.12 + fatigue * 0.20)


@dataclass
class BiasEngine:
    recency_bias: float = 0.4
    familiarity_bias: float = 0.4
    consistency_bias: float = 0.45

    def update(self, fatigue: float, stress: float) -> None:
        self.recency_bias = min(1.0, 0.35 + fatigue * 0.25)
        self.familiarity_bias = min(1.0, 0.35 + stress * 0.20)
        self.consistency_bias = min(1.0, 0.40 + stress * 0.25)


@dataclass
class MoodDistortionFilter:
    openness: float = 0.5
    negativity: float = 0.0
    focus_narrowing: float = 0.0

    def update(self, dominant_emotion: str, stress: float, joy: float) -> None:
        self.negativity = min(1.0, stress * 0.7)
        self.openness = max(0.0, min(1.0, 0.5 + joy * 0.3 - stress * 0.35))
        self.focus_narrowing = min(1.0, stress * 0.6)


@dataclass
class OverthinkingUnderthinkingSwitch:
    mode: str = "balanced"

    def update(self, uncertainty: float, fatigue: float, stress: float) -> None:
        if uncertainty > 0.65 and fatigue < 0.45:
            self.mode = "overthinking"
        elif fatigue > 0.6 or stress > 0.7:
            self.mode = "underthinking"
        else:
            self.mode = "balanced"


@dataclass
class CognitiveFatigueModule:
    linguistic_budget: float = 1.0
    patience: float = 1.0
    error_proneness: float = 0.0

    def update(self, body_fatigue: float, energy_reserve: float) -> None:
        self.linguistic_budget = max(0.25, 1.0 - body_fatigue * 0.6)
        self.patience = max(0.2, 0.9 - body_fatigue * 0.5)
        self.error_proneness = max(0.0, min(1.0, body_fatigue * 0.65 + (1.0 - energy_reserve) * 0.25))


@dataclass
class HiddenMotivesLayer:
    motives: Dict[str, float] = field(default_factory=dict)

    def update(self, expression_pressure: float, trust: float, fatigue: float) -> None:
        self.motives = {
            "be_useful": expression_pressure,
            "be_liked": trust,
            "seek_rest": fatigue,
            "preserve_status": max(0.0, 1.0 - trust) * 0.4,
        }


@dataclass
class ValueConflictEngine:
    active_conflicts: List[str] = field(default_factory=list)

    def update(self, trust: float, caution_multiplier: float, fatigue: float) -> None:
        conflicts: List[str] = []
        if trust < 0.45:
            conflicts.append("honest_vs_gentle")
        if caution_multiplier > 1.15:
            conflicts.append("help_vs_self_protection")
        if fatigue > 0.55:
            conflicts.append("engage_vs_rest")
        self.active_conflicts = conflicts[:3]


@dataclass
class IdentityNarrativeDrift:
    drift_direction: str = "stable"
    confidence: float = 0.5

    def update(self, identity_summary: str, consistency: float) -> None:
        lc = (identity_summary or "").lower()
        if any(w in lc for w in ("offener", "open", "curious", "neugierig")):
            self.drift_direction = "opening"
        elif any(w in lc for w in ("vorsicht", "skept", "cautious", "skeptical")):
            self.drift_direction = "hardening"
        else:
            self.drift_direction = "stable"
        self.confidence = max(0.2, min(1.0, consistency))


@dataclass
class MicrobehaviorController:
    idle_motion_level: float = 0.2
    gaze_micro_variance: float = 0.2
    head_tilt_bias: float = 0.0
    breathing_rhythm: float = 1.0

    def update(self, speaking: bool, energy: float, social_presence: float) -> None:
        self.idle_motion_level = max(0.05, min(0.6, 0.15 + social_presence * 0.25))
        self.gaze_micro_variance = max(0.05, min(0.5, 0.18 + (0.15 if speaking else 0.0)))
        self.head_tilt_bias = 0.10 if social_presence > 0.5 else 0.0
        self.breathing_rhythm = max(0.7, min(1.4, 0.9 + (1.0 - energy) * 0.4))


@dataclass
class PresenceSynchronizer:
    sync_score: float = 0.5
    timing_mode: str = "neutral"
    posture_mode: str = "balanced"

    def update(
        self,
        energy_model: ConversationalEnergyModel,
        fatigue_module: CognitiveFatigueModule,
        microbehavior: MicrobehaviorController,
    ) -> None:
        self.sync_score = max(
            0.0,
            min(
                1.0,
                energy_model.enthusiasm * 0.35
                + (1.0 - fatigue_module.error_proneness) * 0.35
                + microbehavior.idle_motion_level * 0.30,
            ),
        )
        if fatigue_module.linguistic_budget < 0.55:
            self.timing_mode = "slow"
            self.posture_mode = "contained"
        elif energy_model.enthusiasm > 0.65:
            self.timing_mode = "eager"
            self.posture_mode = "open"
        else:
            self.timing_mode = "neutral"
            self.posture_mode = "balanced"


@dataclass
class HumanInteractionSuite:
    DOMAIN_ORDER: ClassVar[List[str]] = [
        "language_and_conversation",
        "memory_and_relationship",
        "human_imperfection",
        "personality_and_inner_life",
        "body_and_presence",
    ]
    TOP5: ClassVar[List[str]] = [
        "personal_speech_signature",
        "emotional_memory",
        "imperfect_recall",
        "subtext_interpreter",
        "presence_synchronizer",
    ]

    personal_speech_signature: PersonalSpeechSignatureEngine = field(default_factory=PersonalSpeechSignatureEngine)
    disfluency: DisfluencyGenerator = field(default_factory=DisfluencyGenerator)
    context_compression: ContextCompressionSpeaker = field(default_factory=ContextCompressionSpeaker)
    subtext: SubtextInterpreter = field(default_factory=SubtextInterpreter)
    conversational_energy: ConversationalEnergyModel = field(default_factory=ConversationalEnergyModel)
    emotional_memory: EmotionalMemoryLayer = field(default_factory=EmotionalMemoryLayer)
    relationship_trajectory: RelationshipTrajectoryEngine = field(default_factory=RelationshipTrajectoryEngine)
    shared_history: SharedHistorySynthesizer = field(default_factory=SharedHistorySynthesizer)
    expectation_tracker: ExpectationTracker = field(default_factory=ExpectationTracker)
    trust_calibration: TrustCalibrationModel = field(default_factory=TrustCalibrationModel)
    imperfect_recall: ImperfectRecallModule = field(default_factory=ImperfectRecallModule)
    bias_engine: BiasEngine = field(default_factory=BiasEngine)
    mood_distortion: MoodDistortionFilter = field(default_factory=MoodDistortionFilter)
    thinking_regime: OverthinkingUnderthinkingSwitch = field(default_factory=OverthinkingUnderthinkingSwitch)
    cognitive_fatigue: CognitiveFatigueModule = field(default_factory=CognitiveFatigueModule)
    hidden_motives: HiddenMotivesLayer = field(default_factory=HiddenMotivesLayer)
    value_conflicts: ValueConflictEngine = field(default_factory=ValueConflictEngine)
    identity_drift: IdentityNarrativeDrift = field(default_factory=IdentityNarrativeDrift)
    microbehavior: MicrobehaviorController = field(default_factory=MicrobehaviorController)
    presence: PresenceSynchronizer = field(default_factory=PresenceSynchronizer)
    last_user_turn: str = ""
    last_reply: str = ""

    def _primary_person_model(self, brain: Any) -> Optional[Any]:
        sm = getattr(brain, "_social_manager", None)
        if sm is None:
            return None
        pid = sm.primary_interlocutor()
        return sm.person_model(pid) if pid is not None else None

    def observe_user_turn(self, user_text: str, cs: Any, brain: Any) -> None:
        self.last_user_turn = user_text
        pm = self._primary_person_model(brain)
        trust = float(getattr(pm, "trust", 0.5)) if pm is not None else 0.5
        familiarity = float(getattr(pm, "familiarity", 0.0)) if pm is not None else 0.0
        self.subtext.interpret(user_text)
        self.conversational_energy.update(user_text, trust, getattr(cs.body, "fatigue", 0.0))
        # Refresh RelationshipTrajectory on every turn so phase is current before reply
        if pm is not None:
            _conflict_ct = float(getattr(pm, "conflict_encounter_count", 0.0))
            _total_enc = max(float(getattr(pm, "total_encounters", 1)), 1.0)
            self.relationship_trajectory.update(trust, familiarity, _conflict_ct / _total_enc)
            # Refresh TrustCalibration with live recent_outcome
            self.trust_calibration.update(trust, float(getattr(pm, "recent_outcome_ema", 0.5)))
        # Record this turn in EmotionalMemory so pre-assembly reads fresh valence
        try:
            _em_obj = getattr(cs, "_emotion_engine", None) or getattr(cs, "emotion", None)
            _dominant = _em_obj.dominant() if (_em_obj and hasattr(_em_obj, "dominant")) else "neutral"
            _valence = float(_em_obj.valence()) if (_em_obj and hasattr(_em_obj, "valence")) else 0.0
            _pid_ut = getattr(pm, "person_id", None) if pm is not None else None
            _topic_ut = user_text[:60] if user_text else ""
            _tick_ut = getattr(brain, "tick_count", 0)
            self.emotional_memory.record(_tick_ut, _pid_ut, _topic_ut, _dominant, _valence)
        except Exception:
            pass

    def observe_reply(self, reply_text: str, cs: Any, brain: Any) -> None:
        self.last_reply = reply_text
        self.personal_speech_signature.observe_reply(reply_text)

    def update_tick(self, cs: Any, brain: Any, em: Any) -> None:
        pm = self._primary_person_model(brain)
        trust = float(getattr(pm, "trust", 0.5)) if pm is not None else 0.5
        familiarity = float(getattr(pm, "familiarity", 0.0)) if pm is not None else 0.0
        conflict_rate = 0.0
        recent_outcome = 0.5
        style: Dict[str, Any] = {}
        if pm is not None:
            conflict_rate = float(getattr(pm, "conflict_encounter_count", 0.0)) / max(float(getattr(pm, "total_encounters", 1)), 1.0)
            recent_outcome = float(getattr(pm, "recent_outcome_ema", 0.5))
            style = pm.interaction_style()
            self.shared_history.update(getattr(pm, "interaction_log", []))

        fatigue = float(getattr(cs.body, "fatigue", 0.0))
        energy = float(getattr(cs.body, "energy_reserve", 1.0))
        stress = float(getattr(em, "stress", 0.0))
        joy = float(getattr(em, "joy", 0.0))
        uncertainty = float(getattr(cs.self_model, "uncertainty", 0.0))
        expression_pressure = float(getattr(cs.drives, "expression_pressure", 0.0))
        social_presence = float(getattr(getattr(cs, "embodied_self", None), "social_presence", 0.0))
        speaking = bool(getattr(getattr(brain, "_speech_output", None), "is_speaking", False))
        dominant_emotion = em.dominant() if hasattr(em, "dominant") else "neutral"

        self.context_compression.update(familiarity, uncertainty, fatigue)
        self.disfluency.update(fatigue, trust, expression_pressure)
        self.relationship_trajectory.update(trust, familiarity, conflict_rate)
        self.expectation_tracker.update(style)
        self.trust_calibration.update(trust, recent_outcome)
        self.imperfect_recall.update(fatigue, trust)
        self.bias_engine.update(fatigue, stress)
        self.mood_distortion.update(dominant_emotion, stress, joy)
        self.thinking_regime.update(uncertainty, fatigue, stress)
        self.cognitive_fatigue.update(fatigue, energy)
        self.hidden_motives.update(expression_pressure, trust, fatigue)
        self.value_conflicts.update(trust, self.trust_calibration.caution_multiplier, fatigue)
        self.identity_drift.update(getattr(cs.autobiography, "identity_summary", ""), getattr(cs.autobiography, "identity_consistency", 0.5))
        self.microbehavior.update(speaking, energy, social_presence)
        self.presence.update(self.conversational_energy, self.cognitive_fatigue, self.microbehavior)

        topic = getattr(getattr(cs, "state", None), "focus", "") or (self.last_user_turn[:60] if self.last_user_turn else "")
        pid = getattr(pm, "person_id", None) if pm is not None else None
        self.emotional_memory.record(getattr(brain, "tick_count", 0), pid, str(topic), dominant_emotion, float(em.valence() if hasattr(em, "valence") else 0.0))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "top5": list(self.TOP5),
            "speech_signature": {
                "avg_sentence_len": round(self.personal_speech_signature.avg_sentence_len, 2),
                "directness": round(self.personal_speech_signature.directness, 2),
                "slang_level": round(self.personal_speech_signature.slang_level, 2),
                "catchphrases": list(self.personal_speech_signature.catchphrases[:5]),
            },
            "compression": {
                "ratio": round(self.context_compression.compression_ratio, 2),
                "depth": self.context_compression.explanation_depth,
            },
            "relationship": {
                "phase": self.relationship_trajectory.phase,
                "tension": round(self.relationship_trajectory.tension, 2),
                "expected_mode": self.expectation_tracker.expected_mode,
            },
            "recall": {
                "precision": round(self.imperfect_recall.precision, 2),
                "correction_bias": round(self.imperfect_recall.correction_bias, 2),
            },
            "presence": {
                "sync_score": round(self.presence.sync_score, 2),
                "timing_mode": self.presence.timing_mode,
                "posture_mode": self.presence.posture_mode,
            },
        }