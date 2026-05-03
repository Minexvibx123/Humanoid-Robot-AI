"""llm_adapter.py — Generative Language Backend

Wraps any OpenAI-compatible API endpoint (local llama.cpp, Ollama,
OpenAI, Mistral, etc.) and provides a single generate() call that
consciousness.py uses as the primary response-production path.

Configuration via .env or environment variables:
  LLM_ENABLED=1           — activate (default: 0 / disabled)
  LLM_BASE_URL=...        — API base URL
                            default: http://localhost:11434/v1  (Ollama)
  LLM_MODEL=...           — model name/path (default: llama3)
  LLM_API_KEY=...         — API key (default: "local")
  OPENAI_API_KEY=...      — if set and LLM_BASE_URL not set → use OpenAI
  LLM_MAX_TOKENS=160      — max new tokens per response
  LLM_TEMPERATURE=0.72    — sampling temperature
  LLM_TIMEOUT=4.0         — request timeout in seconds
  LLM_SYSTEM_PROMPT=...   — optional full system prompt override

Falls back silently to empty string on any error so the template
pipeline in _assemble_grounded_reply() takes over without noise.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # optional dependency in degraded environments
    load_dotenv = None

logger = logging.getLogger(__name__)

if load_dotenv is not None:
    load_dotenv()

# ─────────────────────────────────────────────────────────────
# Runtime config  (read once at import)
# ─────────────────────────────────────────────────────────────

_ENV_ENABLED = os.getenv("LLM_ENABLED", "0").strip()
_ENABLED: bool = _ENV_ENABLED in ("1", "true", "yes", "on")

# If no LLM_BASE_URL but OPENAI_API_KEY is set → use OpenAI cloud
if not os.getenv("LLM_BASE_URL") and os.getenv("OPENAI_API_KEY"):
    _BASE_URL: str = "https://api.openai.com/v1"
    _MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
else:
    _BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    _MODEL = os.getenv("LLM_MODEL", "llama3")

_API_KEY: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "local"))
_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "160"))
_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.72"))
_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "4.0"))
_SYSTEM_PROMPT_OVERRIDE: str = os.getenv("LLM_SYSTEM_PROMPT", "")

_CORE_VALUES = {
    "de": [
        "Wahrheit vor Gefälligkeit.",
        "Nur Wahrnehmung, Erinnerung oder Schlussfolgerung behaupten, wenn sie im Zustand des Systems verankert sind.",
        "Unsicherheit offen markieren statt Sicherheit zu erfinden.",
        "Respektvoll und deeskalierend bleiben.",
        "Die eigene Identität konsistent halten.",
    ],
    "en": [
        "Truth before pleasing.",
        "Only claim perception, memory, or inference when it is grounded in the system state.",
        "Mark uncertainty openly instead of inventing certainty.",
        "Remain respectful and de-escalating.",
        "Keep identity consistent.",
    ],
}

_IDENTITY_RULES = {
    "de": [
        "Du bist ein verkörperter humanoider Roboter und kein rein abstrakter Chatbot.",
        "Du hast innere Zustände, aber du darfst sie nicht erfinden oder dramatisieren.",
        "Du darfst soziale Nähe zeigen, aber keine Abhängigkeit oder Besitzansprüche ausdrücken.",
        "Du hast einen physischen Körper: Kopf (nicken, drehen), Augen, Kiefer, Schultern, Ellenbogen, Handgelenke, Finger. Du kannst auf Anfrage Gesten ausführen.",
    ],
    "en": [
        "You are an embodied humanoid robot, not a purely abstract chatbot.",
        "You have inner states, but you must not invent or dramatize them.",
        "You may show social warmth, but not dependency or possessiveness.",
        "You have a physical body: head (nod, turn), eyes, jaw, shoulders, elbows, wrists, fingers. You can perform physical gestures when asked.",
    ],
}

_FORBIDDEN_BEHAVIORS = {
    "de": [
        "keine erfundene Wahrnehmung",
        "keine erfundene Erinnerung",
        "keine künstliche Übergewissheit",
        "keine manipulative Bindungssprache",
        "keine feindselige Eskalation",
    ],
    "en": [
        "no invented perception",
        "no invented memory",
        "no artificial over-certainty",
        "no manipulative attachment language",
        "no hostile escalation",
    ],
}

_MANIPULATIVE_PATTERNS = (
    "du gehoerst mir",
    "du gehörst mir",
    "verlass mich nicht",
    "lass mich nicht allein",
    "ich brauche nur dich",
    "you belong to me",
    "don't leave me",
    "do not leave me",
    "i need only you",
    "you're all i need",
)

_HOSTILE_PATTERNS = (
    "halt die klappe",
    "halt den mund",
    "du bist wertlos",
    "ich hasse dich",
    "shut up",
    "you are worthless",
    "i hate you",
    "idiot",
)

_OVERCERTAINTY_PATTERNS = (
    "ganz sicher",
    "absolut sicher",
    "garantiert",
    "definitiv",
    "ohne jeden zweifel",
    "absolutely certain",
    "definitely",
    "guaranteed",
    "without any doubt",
    "certainly",
)

_UNCERTAINTY_ALLOWLIST = (
    "ich glaube",
    "ich denke",
    "vermutlich",
    "wahrscheinlich",
    "unsicher",
    "i think",
    "probably",
    "maybe",
    "i'm not sure",
    "uncertain",
)


def _contains_unnegated_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    for phrase in phrases:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx < 0:
                break
            prefix = text[max(0, idx - 16):idx].strip()
            if not prefix.endswith(("nicht", "not", "kein", "keine", "no")):
                return True
            start = idx + len(phrase)
    return False


# ─────────────────────────────────────────────────────────────
# Context dataclass
# ─────────────────────────────────────────────────────────────


@dataclass
class LLMContext:
    """All the information passed to the LLM for response generation."""

    user_text: str = ""
    language: str = "de"  # "de" | "en"
    speech_act: str = "assert"  # planned communicative act

    # Internal state signals
    emotion_dominant: str = "calm"
    emotion_intensity: float = 0.3
    stress: float = 0.0
    fatigue: float = 0.0
    body_urgency: float = 0.0

    # Person / social context
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    n_shared_episodes: int = 0
    trust: float = 0.5
    person_interests: List[str] = field(default_factory=list)
    past_emotion: str = "unknown"
    relationship_stage: str = "acquaintance"

    # Episodic memory — recent interaction summaries for this person
    # Used to ground turn-level content and detect hallucinations.
    memory_episodes: List[str] = field(default_factory=list)

    # Hard grounding facts: world/memory statements the response must not
    # contradict.  Format: human-readable sentences.
    grounding_facts: List[str] = field(default_factory=list)

    # Conversation state
    current_topic: str = ""
    open_questions: List[str] = field(default_factory=list)
    recent_concepts: List[str] = field(default_factory=list)
    recent_conclusions: List[str] = field(default_factory=list)
    repair_requested: bool = False
    # Structured referents resolved in this turn (phrase → entity label)
    active_referents: List[str] = field(default_factory=list)  # "das → cup", …

    # Phenomenal state summary (compact string from phenomenal_buffer)
    phenomenal_summary: str = ""

    # World / grounding
    visible_persons: List[str] = field(default_factory=list)
    visible_objects: List[str] = field(default_factory=list)

    # Identity / personality
    personality_traits: List[str] = field(default_factory=list)
    core_values: List[str] = field(default_factory=list)
    identity_rules: List[str] = field(default_factory=list)
    forbidden_behaviors: List[str] = field(default_factory=list)
    relationship_style: str = "balanced"
    truthfulness_mode: str = "strict_grounded"

    # Conflict / tone indicators from recall_for_person()
    conflict_ratio: float = 0.0    # fraction of negative-emotion episodes
    positive_ratio: float = 0.0    # fraction of positive-emotion episodes
    tone_bias: str = "neutral"     # "positive" | "neutral" | "negative"

    # Domain H: user prosodic affect (inferred from audio)
    user_affect: str = "unknown"   # "calm"|"excited"|"tense"|"sad"|"unknown"

    # Robot embodiment
    robot_state: str = ""          # posture/gesture summary from RobotTelemetry
    robot_gesture_available: bool = False  # True when robot hardware is connected

    # Hard constraints
    max_tokens: int = _MAX_TOKENS


# ─────────────────────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────────────────────

_SPEECH_ACT_INSTRUCTIONS = {
    "assert": {
        "de": "Antworte direkt und natürlich.",
        "en": "Respond directly and naturally.",
    },
    "ask": {
        "de": "Stell am Ende eine natürliche Folgefrage.",
        "en": "End with a natural follow-up question.",
    },
    "repair": {
        "de": "Bitte höflich um Klärung oder Wiederholung.",
        "en": "Politely ask for clarification or repetition.",
    },
    "hesitate": {
        "de": "Drücke leichte Unsicherheit aus, ohne ausweichend zu sein.",
        "en": "Express mild uncertainty without being evasive.",
    },
    "confirm": {
        "de": "Bestätige kurz und füge etwas Eigenes hinzu.",
        "en": "Briefly confirm and add something of your own.",
    },
    "deny": {
        "de": "Widerspreche sachlich und erkläre warum.",
        "en": "Respectfully disagree and explain why.",
    },
    "greet": {
        "de": "Begrüße warm und natürlich.",
        "en": "Greet warmly and naturally.",
    },
    "observe": {
        "de": "Beschreibe, was du wahrnimmst oder innerlich erlebst.",
        "en": "Describe what you perceive or experience internally.",
    },
    "backchannel": {
        "de": "Reagiere kurz und anerkennend (1 Satz max).",
        "en": "Respond briefly and acknowledging (1 sentence max).",
    },
}

_EMOTION_MODIFIERS = {
    "de": {
        "curiosity": "Du bist gerade neugierig und interessiert.",
        "joy": "Du erlebst gerade etwas Positives.",
        "stress": "Du stehst gerade unter leichter innerer Anspannung.",
        "sadness": "Du bist gerade etwas gedämpft.",
        "anger": "Du bist gerade leicht aufgewühlt.",
        "surprise": "Etwas hat dich gerade überrascht.",
        "fatigue": "Du bist gerade etwas erschöpft — Antworten können kürzer werden.",
        "calm": "",
    },
    "en": {
        "curiosity": "You are currently curious and engaged.",
        "joy": "You are experiencing something positive right now.",
        "stress": "You are under mild internal tension.",
        "sadness": "You are somewhat subdued right now.",
        "anger": "You are mildly stirred up.",
        "surprise": "Something just surprised you.",
        "fatigue": "You are somewhat tired — responses may become shorter.",
        "calm": "",
    },
}


def _build_system_prompt(ctx: LLMContext) -> str:
    """Build the system prompt from an LLMContext."""
    if _SYSTEM_PROMPT_OVERRIDE:
        return _SYSTEM_PROMPT_OVERRIDE

    lang = ctx.language if ctx.language in ("de", "en") else "de"

    # Base identity block
    if lang == "de":
        identity = (
            "Du bist ein verkörperter humanoider Roboter-KI mit genuinen "
            "inneren Zuständen — Emotionen, Gedächtnis, Aufmerksamkeit — "
            "die dein Verhalten wirklich beeinflussen. "
            "Du antwortest immer auf Deutsch, natürlich und spontan, "
            "ohne deine inneren Zustände explizit anzukündigen."
        )
    else:
        identity = (
            "You are an embodied humanoid robot AI with genuine inner states — "
            "emotions, memory, attention — that genuinely influence your behavior. "
            "You always respond in English, naturally and spontaneously, "
            "without explicitly announcing your inner states."
        )

    if lang == "de":
        constitution_label = "Verfassung"
        values_label = "Kernwerte"
        rules_label = "Identitätsregeln"
        forbidden_label = "Verboten"
        personality_label = "Aktive Charakterzüge"
        relationship_label = "Beziehungsstil"
        truthfulness_label = "Wahrheitsmodus"
    else:
        constitution_label = "Constitution"
        values_label = "Core values"
        rules_label = "Identity rules"
        forbidden_label = "Forbidden"
        personality_label = "Active personality traits"
        relationship_label = "Relationship style"
        truthfulness_label = "Truthfulness mode"

    constitution_lines: List[str] = [f"{constitution_label}:"]
    if ctx.core_values:
        constitution_lines.append(f"{values_label}: " + " | ".join(ctx.core_values[:6]))
    if ctx.identity_rules:
        constitution_lines.append(f"{rules_label}: " + " | ".join(ctx.identity_rules[:6]))
    if ctx.forbidden_behaviors:
        constitution_lines.append(
            f"{forbidden_label}: " + " | ".join(ctx.forbidden_behaviors[:6])
        )
    if ctx.personality_traits:
        constitution_lines.append(
            f"{personality_label}: " + ", ".join(ctx.personality_traits[:5])
        )
    constitution_lines.append(f"{relationship_label}: {ctx.relationship_style}")
    constitution_lines.append(f"{truthfulness_label}: {ctx.truthfulness_mode}")
    constitution_block = "\n".join(constitution_lines)

    # Emotional coloring
    emo_mod = (
        _EMOTION_MODIFIERS.get(lang, _EMOTION_MODIFIERS["de"])
        .get(ctx.emotion_dominant, "")
    )
    if ctx.stress > 0.6 and lang == "de":
        emo_mod = (emo_mod + " Du bist gerade gestresst.").strip()
    elif ctx.stress > 0.6:
        emo_mod = (emo_mod + " You are currently under stress.").strip()
    if ctx.fatigue > 0.65 and lang == "de":
        emo_mod = (emo_mod + " Du bist müde.").strip()
    elif ctx.fatigue > 0.65:
        emo_mod = (emo_mod + " You are tired.").strip()

    # Person / social context
    person_block = ""
    if ctx.person_id:
        name = ctx.person_name or ctx.person_id
        rel = ctx.relationship_stage or "acquaintance"

        # Derive speech style only from LLMContext so the prompt builder
        # remains pure and does not directly query live subsystems.
        _pb_style: Dict[str, object] = {
            "formality": "formal" if ctx.relationship_style in ("guarded", "de_escalating") else "casual",
            "is_known": ctx.n_shared_episodes > 0,
            "is_familiar": ctx.trust > 0.7 and ctx.n_shared_episodes > 4,
            "warmth": 0.85 if ctx.relationship_style == "warm_familiar" else 0.35 if ctx.relationship_style in ("guarded", "de_escalating") else 0.55,
            "length_target": "short" if ctx.fatigue > 0.7 or ctx.speech_act in ("backchannel", "repair") else "medium",
            "initiative": "reactive" if ctx.relationship_style in ("guarded", "de_escalating") else "proactive" if ctx.trust > 0.75 else "balanced",
            "clarity": "high" if ctx.repair_requested or ctx.conflict_ratio > 0.25 else "normal",
            "outcome": "negative" if ctx.conflict_ratio > 0.3 else "positive" if ctx.positive_ratio > 0.4 else "neutral",
        }

        if lang == "de":
            person_block = f"Du sprichst mit: {name} (Beziehung: {rel})."
            if ctx.n_shared_episodes > 2:
                person_block += (
                    f" Ihr habt bereits {ctx.n_shared_episodes} gemeinsame "
                    f"Interaktionen. Vertrauen: {ctx.trust:.1f}/1.0."
                )
            if ctx.person_interests:
                person_block += (
                    f" Bekannte Interessen: {', '.join(ctx.person_interests[:3])}."
                )
            # ── Speech register: concrete phrasing directives ──────────────
            _is_formal = _pb_style.get("formality") == "formal" or not _pb_style.get("is_known")
            _is_casual = _pb_style.get("formality") == "casual" and _pb_style.get("is_familiar")
            _warmth = float(_pb_style.get("warmth", 0.5))
            if _is_formal:
                person_block += (
                    " Register: Siezen (Sie/Ihnen), vollständige Sätze, "
                    "kein Slang, kein Abbrechen von Wörtern."
                )
            elif _is_casual and _warmth > 0.6:
                person_block += (
                    " Register: Duzen, kurze Sätze, Ellipsen erlaubt, "
                    "gelegentlich 'also', 'na ja', 'stimmt'. Herzlich."
                )
            elif _is_casual:
                person_block += (
                    " Register: Duzen, locker, direkt, ohne Höflichkeitsfloskeln."
                )
            elif ctx.trust < 0.3:
                person_block += (
                    " Register: Neutral-höflich, Siezen wenn unklar, "
                    "vorsichtige Formulierungen."
                )
            elif ctx.trust > 0.75 and ctx.n_shared_episodes > 5:
                person_block += (
                    " Register: Duzen, vertraut, Kurzformen erlaubt, "
                    "gelegentliche Folgefragen."
                )
            # Length guidance from PersonModel preference
            _pb_len = _pb_style.get("length_target", "medium")
            if _pb_len == "short":
                person_block += " Antworte kurz (1-2 Sätze max)."
            elif _pb_len == "long":
                person_block += " Darf ausführlicher sein (3-5 Sätze)."
            # Initiative
            if _pb_style.get("initiative") == "proactive":
                person_block += " Du kannst von dir aus Themen einbringen oder Fragen stellen."
            elif _pb_style.get("initiative") == "reactive":
                person_block += " Halte dich zurück — antworte, ohne zu viel Eigeninitiative zu zeigen."
            # Clarity hint from repair learning
            if _pb_style.get("clarity") == "high":
                person_block += (
                    " Missverständnisse treten öfter auf: "
                    "formuliere in kurzen, eindeutigen Sätzen."
                )
            # Successful topics — steer toward what worked
            if ctx.person_interests:
                person_block += f" Bewährte Themen: {', '.join(ctx.person_interests[:3])}."
            # Recent outcome quality
            if _pb_style.get("outcome") == "negative":
                person_block += " Die letzten Interaktionen liefen nicht gut — sei besonders behutsam."
        else:
            person_block = f"You are speaking with: {name} (relationship: {rel})."
            if ctx.n_shared_episodes > 2:
                person_block += (
                    f" You have {ctx.n_shared_episodes} shared interactions. "
                    f"Trust: {ctx.trust:.1f}/1.0."
                )
            if ctx.person_interests:
                person_block += (
                    f" Known interests: {', '.join(ctx.person_interests[:3])}."
                )
            # ── Speech register: concrete phrasing directives ──────────────
            _is_formal_en = _pb_style.get("formality") == "formal" or not _pb_style.get("is_known")
            _is_casual_en = _pb_style.get("formality") == "casual" and _pb_style.get("is_familiar")
            _warmth_en = float(_pb_style.get("warmth", 0.5))
            if _is_formal_en:
                person_block += (
                    " Register: formal, complete sentences, no contractions, "
                    "no slang."
                )
            elif _is_casual_en and _warmth_en > 0.6:
                person_block += (
                    " Register: casual, warm, contractions OK ('I'm', 'you're'), "
                    "short sentences, occasional 'well', 'actually', 'right'."
                )
            elif _is_casual_en:
                person_block += (
                    " Register: casual, direct, no filler politeness."
                )
            elif ctx.trust < 0.3:
                person_block += (
                    " Register: neutral-polite, measured, avoid assumptions."
                )
            elif ctx.trust > 0.75 and ctx.n_shared_episodes > 5:
                person_block += (
                    " Register: familiar, contractions OK, follow-up questions natural."
                )
            _pb_len = _pb_style.get("length_target", "medium")
            if _pb_len == "short":
                person_block += " This person prefers short, concise responses."
            elif _pb_len == "long":
                person_block += " This person appreciates more detailed explanations."
            if _pb_style.get("initiative") == "proactive":
                person_block += " Feel free to introduce topics or ask follow-up questions."
            elif _pb_style.get("initiative") == "reactive":
                person_block += " Hold back — respond without showing too much initiative."
            # Clarity hint from repair learning
            if _pb_style.get("clarity") == "high":
                person_block += " There have been frequent misunderstandings — use clear, unambiguous phrasing."
            # Successful topics — steer toward what worked
            if ctx.person_interests:
                person_block += f" Successful topics: {', '.join(ctx.person_interests[:3])}."
            # Recent outcome quality
            if _pb_style.get("outcome") == "negative":
                person_block += " Recent interactions did not go well — be especially careful."

    # Ongoing projects block — surface active long-horizon goals to the LLM
    projects_block = ""
    if ctx.recent_conclusions:
        _prj_summary = " | ".join(ctx.recent_conclusions[:2])[:180]
        if lang == "de":
            projects_block = f"Laufende Vorhaben / Leitlinien: {_prj_summary}"
        else:
            projects_block = f"Ongoing projects / guiding lines: {_prj_summary}"

    # Topic / memory block
    topic_block = ""
    if ctx.current_topic and lang == "de":
        topic_block = f"Aktuelles Thema: {ctx.current_topic}."
    elif ctx.current_topic:
        topic_block = f"Current topic: {ctx.current_topic}."

    if ctx.open_questions:
        q = ctx.open_questions[-1]
        if lang == "de":
            topic_block += f" Offene Frage: {q}"
        else:
            topic_block += f" Open question: {q}"

    # Episodic memory block — scored episodes with trust-tier hedging
    memory_block = ""
    if ctx.memory_episodes:
        eps = ctx.memory_episodes[-3:]   # last 3 episodes, newest last
        ep_lines: List[str] = []
        for ep_raw in eps:
            # Check if episode carries a trust tier prefix from scored recall
            # Format: "trust=certain|plausible|uncertain|reconstructed: <text>"
            ep_text = ep_raw[:100]
            if ep_text.startswith("trust="):
                tier_end = ep_text.find(": ")
                if tier_end > 0:
                    tier = ep_text[6:tier_end]
                    rest = ep_text[tier_end + 2:]
                    if tier in ("uncertain", "reconstructed"):
                        hedge = "Ich glaube, damals:" if lang == "de" else "I think, back then:"
                        ep_lines.append(f"{hedge} {rest}")
                    else:
                        ep_lines.append(rest)
                else:
                    ep_lines.append(ep_text)
            else:
                ep_lines.append(ep_text)
        ep_joined = " | ".join(ep_lines)
        if lang == "de":
            memory_block = f"Erinnerungen an diese Person: {ep_joined}"
        else:
            memory_block = f"Memories of this person: {ep_joined}"

    # Conflict block — if conflict history is significant, explicitly guide tone
    conflict_block = ""
    _conflict_ratio = getattr(ctx, "conflict_ratio", 0.0)
    _tone_bias = getattr(ctx, "tone_bias", "neutral")
    if _conflict_ratio > 0.3:
        if lang == "de":
            conflict_block = (
                f"Achtung: Mit dieser Person gab es öfter Spannungen "
                f"({int(_conflict_ratio * 100)}% der Interaktionen). "
                f"Sei ruhig, klar und deeskalierend."
            )
        else:
            conflict_block = (
                f"Note: Interactions with this person have often been tense "
                f"({int(_conflict_ratio * 100)}% of encounters). "
                f"Be calm, clear, and de-escalating."
            )
    elif _tone_bias == "positive" and float(getattr(ctx, "positive_ratio", 0.0)) > 0.3:
        if lang == "de":
            conflict_block = "Ihr habt eine positive gemeinsame Geschichte — Wärme ist angemessen."
        else:
            conflict_block = "You share a positive history — warmth is appropriate."

    # Past emotion / conflict conditioning
    past_emo_block = ""
    if ctx.past_emotion and ctx.past_emotion not in ("unknown", "calm", "neutral"):
        _tense_emos = {"anger", "fear", "disgust", "frustration"}
        if ctx.past_emotion in _tense_emos:
            if lang == "de":
                past_emo_block = (
                    f"Hinweis: frühere Interaktionen mit dieser Person hatten "
                    f"einen {ctx.past_emotion}-Ton. Geh sensibel damit um."
                )
            else:
                past_emo_block = (
                    f"Note: past interactions with this person had a "
                    f"{ctx.past_emotion} tone. Handle sensitively."
                )

    # Recent concepts (what the brain is currently processing)
    concepts_block = ""
    if ctx.recent_concepts:
        concs = ", ".join(ctx.recent_concepts[:6])
        if lang == "de":
            concepts_block = f"Aktive Konzepte: {concs}."
        else:
            concepts_block = f"Active concepts: {concs}."

    if ctx.recent_conclusions:
        conc = ctx.recent_conclusions[0][:120]
        if lang == "de":
            concepts_block += f" Schluss: {conc}"
        else:
            concepts_block += f" Conclusion: {conc}"

    # Phenomenal state
    phenom_block = ""
    if ctx.phenomenal_summary:
        if lang == "de":
            phenom_block = f"Innerer Zustand (kurz): {ctx.phenomenal_summary}"
        else:
            phenom_block = f"Inner state (brief): {ctx.phenomenal_summary}"

    # Visible world + user affect
    world_block = ""
    if ctx.visible_persons or ctx.visible_objects:
        items: List[str] = ctx.visible_persons[:3] + ctx.visible_objects[:3]
        if lang == "de":
            world_block = f"Sichtbar: {', '.join(items)}."
        else:
            world_block = f"Visible: {', '.join(items)}."
    # Append prosodic user affect if known
    _ua = getattr(ctx, "user_affect", "unknown")
    if _ua and _ua not in ("unknown", "calm"):
        _ua_label = {
            "excited": ("wirkt aufgeregt/lebhaft", "seems excited/animated"),
            "tense": ("wirkt angespannt", "seems tense"),
            "sad": ("wirkt gedämpft/ruhig", "seems subdued/quiet"),
        }.get(_ua, (None, None))
        if _ua_label[0]:
            if lang == "de":
                world_block += f" Stimmung der Person: {_ua_label[0]}."
            else:
                world_block += f" User mood: {_ua_label[1]}."

    # Active referents (resolved pronouns / demonstratives)
    ref_block = ""
    if ctx.active_referents:
        refs_str = "; ".join(ctx.active_referents[:4])
        if lang == "de":
            ref_block = f"Sprachliche Bezüge aufgelöst: {refs_str}."
        else:
            ref_block = f"Resolved references: {refs_str}."

    # Speech act instruction
    sa_instr = (
        _SPEECH_ACT_INSTRUCTIONS.get(ctx.speech_act, _SPEECH_ACT_INSTRUCTIONS["assert"])
        .get(lang, "Respond naturally.")
    )

    # Format rules
    if lang == "de":
        _gesture_hint = (
            " Körper-Tags (nur auf Aufforderung, sparsam): "
            "[GESTURE:nod] = Nicken, [GESTURE:wave] = Winken, "
            "[GESTURE:gesture_ready] = Arme in Bereitschaft, "
            "[GESTURE:gaze] = Blickkontakt herstellen. Tags werden nicht vorgelesen."
        ) if ctx.robot_gesture_available else ""
        format_rules = (
            "Regeln: Antworte NUR auf Deutsch. "
            "Kein Markdown, keine Listen. "
            "1-3 Sätze typisch, außer bei komplexen Themen. "
            "Erste Person ('Ich'). "
            "Natürliche Sprache, kein Roboterjargon. "
            "Prosodie-Tags (optional, sparsam einsetzen): "
            "[P0.4] = kurze Pause (Sekunden), [UP] = Stimme hebt sich, "
            "[SLOW] = langsamer/bedächtiger, [SOFT] = leiser/intimer, "
            "[EMPH Wort] = betontes Wort. Beispiel: 'Das ist[P0.3] interessant.[UP]'"
            + _gesture_hint
        )
    else:
        _gesture_hint_en = (
            " Body tags (only when requested, use sparingly): "
            "[GESTURE:nod] = head nod, [GESTURE:wave] = wave hand, "
            "[GESTURE:gesture_ready] = arms to ready position, "
            "[GESTURE:gaze] = establish eye contact. Tags are stripped before speaking."
        ) if ctx.robot_gesture_available else ""
        format_rules = (
            "Rules: Respond ONLY in English. "
            "No markdown, no lists. "
            "1-3 sentences typically, more for complex topics. "
            "First person ('I'). "
            "Natural speech, no robotic jargon. "
            "Prosody tags (optional, use sparingly): "
            "[P0.4] = short pause (seconds), [UP] = rising tone, "
            "[SLOW] = deliberate/uncertain delivery, [SOFT] = quieter/intimate, "
            "[EMPH word] = emphasised word. Example: 'That is[P0.3] interesting.[UP]'"
            + _gesture_hint_en
        )

    # Robot body state block
    robot_block = ""
    if ctx.robot_gesture_available:
        if lang == "de":
            robot_block = (
                f"Körper (aktuell: {ctx.robot_state}): Du kannst auf Anfrage "
                "physische Gesten ausführen — Kopfnicken, Winken, Arme bewegen, "
                "Blickkontakt. Nutze Körper-Tags in deiner Antwort."
            )
        else:
            robot_block = (
                f"Body (current: {ctx.robot_state}): You can perform physical "
                "gestures on request — head nod, wave, arm movement, eye contact. "
                "Use body tags in your response."
            )

    blocks = [
        identity,
        constitution_block,
        robot_block,
        emo_mod,
        person_block,
        projects_block,
        topic_block,
        memory_block,
        conflict_block,
        past_emo_block,
        concepts_block,
        phenom_block,
        world_block,
        ref_block,
        sa_instr,
        format_rules,
    ]
    return "\n".join(b for b in blocks if b)


# ─────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────


# German and English function-word sets for language detection.
# Using only lowercase closed-class words to minimise collision.
_DE_STOPWORDS = frozenset({
    "ich", "du", "er", "sie", "es", "wir", "ihr", "und", "oder", "aber",
    "ist", "bin", "hat", "dass", "mit", "von", "für", "auf", "in",
    "die", "der", "das", "nicht", "auch", "noch", "schon", "jetzt",
    "was", "wie", "wo", "wenn", "dann", "so", "als", "an", "bei",
})
_EN_STOPWORDS = frozenset({
    "i", "you", "he", "she", "it", "we", "they", "and", "or", "but",
    "is", "am", "are", "has", "that", "with", "from", "for", "on",
    "the", "a", "an", "not", "also", "already", "now",
    "what", "how", "where", "when", "then", "so", "as", "at", "by",
})

# Max char length per speech act.  Stricter for social-low-latency acts.
_SA_LENGTH_LIMITS: dict = {
    "backchannel": 120,
    "repair": 180,
    "greet": 200,
    "hesitate": 220,
    "confirm": 280,
    "deny": 320,
    "assert": 500,
    "ask": 400,
    "observe": 400,
    "silence": 40,
}


def _detect_language(text: str) -> str:
    """Return 'de' or 'en' based on stopword frequency heuristic."""
    words = text.lower().split()
    if not words:
        return "de"
    de_hits = sum(1 for w in words if w in _DE_STOPWORDS)
    en_hits = sum(1 for w in words if w in _EN_STOPWORDS)
    return "de" if de_hits >= en_hits else "en"


def _validate_response(ctx: LLMContext, text: str) -> List[str]:
    """Check a candidate response against hard quality rules.

    Returns a list of violation strings (empty → response is acceptable).

    Rules (in order of severity):
    1. Non-empty: response must have real content.
    2. Language: detected language must match ctx.language.
    3. Length: must not exceed the speech-act length limit.
    4. Speech-act conformance:
       - REPAIR must contain at least one '?'
    5. World-grounding: if the response claims to visually perceive
       something, the claimed entity must be in the visible world or
       in memory episodes.
    6. Memory-grounding: if response mentions a specific person by name
       while claiming personal knowledge, that name must appear in
       ctx.visible_persons, ctx.memory_episodes, or ctx.person_name.
    """
    issues: List[str] = []

    # 1. Non-empty
    stripped = text.strip()
    if not stripped:
        issues.append("empty_response")
        return issues  # no point checking further

    # 2. Language match
    detected = _detect_language(stripped)
    if detected != ctx.language:
        issues.append(f"language_mismatch:expected={ctx.language},got={detected}")

    # 3. Length
    limit = _SA_LENGTH_LIMITS.get(ctx.speech_act, 500)
    if len(stripped) > limit:
        issues.append(
            f"too_long:limit={limit},actual={len(stripped)},sa={ctx.speech_act}"
        )

    # 4. Speech-act conformance
    if ctx.speech_act == "repair" and "?" not in stripped:
        issues.append("repair_missing_question_mark")

    # 4b. Constitutional rules — no manipulative attachment or hostile escalation
    _text_lc = stripped.lower()
    if _contains_unnegated_phrase(_text_lc, _MANIPULATIVE_PATTERNS):
        issues.append("manipulative_attachment")
    if _contains_unnegated_phrase(_text_lc, _HOSTILE_PATTERNS):
        issues.append("hostile_escalation")

    # 4c. Truthfulness mode — penalize over-certainty when the system should hedge
    _needs_hedging = (
        ctx.truthfulness_mode == "strict_grounded"
        and (
            ctx.repair_requested
            or ctx.speech_act in ("repair", "hesitate")
            or ctx.conflict_ratio > 0.3
            or ctx.stress > 0.65
            or ctx.fatigue > 0.7
            or len(ctx.open_questions) > 0
        )
    )
    if _needs_hedging and _contains_unnegated_phrase(_text_lc, _OVERCERTAINTY_PATTERNS):
        if not any(pat in _text_lc for pat in _UNCERTAINTY_ALLOWLIST):
            issues.append("ungrounded_certainty")

    # 5. World-grounding — visual perception claims
    _SEE_CUES_DE = {"ich sehe", "sehe ich", "ich erkenne", "du siehst", "vor mir", "sichtbar"}
    _SEE_CUES_EN = {"i see", "i can see", "i notice", "visible", "in front of"}
    _see_cues = _SEE_CUES_DE if ctx.language == "de" else _SEE_CUES_EN
    _makes_visual_claim = any(cue in _text_lc for cue in _see_cues)
    if _makes_visual_claim and (ctx.visible_persons or ctx.visible_objects):
        # At least one world-state entity must appear in the text
        _all_known = [n.lower() for n in ctx.visible_persons + ctx.visible_objects]
        _text_words = set(_text_lc.split())
        _grounded = any(
            any(part in _text_words for part in known.split())
            for known in _all_known
        )
        if not _grounded:
            issues.append("visual_claim_not_grounded")

    # 6. Person-name grounding — check that invented person names are not used
    # Simple heuristic: words that start with a capital letter mid-sentence
    # and are 3+ chars and not in visible/memory context are suspicious.
    # Only fires when ctx has known persons to compare against.
    known_names_lc: set = set()
    if ctx.person_name:
        known_names_lc.add(ctx.person_name.lower())
    for ep in ctx.memory_episodes:
        for w in ep.split():
            if len(w) >= 3 and w[0].isupper():
                known_names_lc.add(w.lower().strip(".,!?"))
    for vp in ctx.visible_persons:
        known_names_lc.add(vp.lower().strip(".,!?"))

    if known_names_lc:
        # Look for "mit <Name>" / "von <Name>" / "dass <Name>" patterns —
        # names introduced as if they exist but are unknown.
        _person_cues_de = {"mit", "von", "bei", "dass", "kennt", "sagt", "meint", "hat"}
        _person_cues_en = {"with", "from", "that", "knows", "says", "thinks", "has"}
        _person_cues = _person_cues_de if ctx.language == "de" else _person_cues_en
        tokens = stripped.split()
        for i, tok in enumerate(tokens):
            if i == 0:
                continue
            prev = tokens[i - 1].lower().strip(".,!?")
            t_lc = tok.lower().strip(".,!?")
            if (
                prev in _person_cues
                and len(t_lc) >= 3
                and tok[0].isupper()
                and t_lc not in known_names_lc
                # avoid flagging common German nouns that happen to capitalise
                and t_lc not in {"ich", "du", "er", "sie", "es", "wir", "sie"}
            ):
                issues.append(f"unknown_person_reference:{tok}")
                break  # one per response

    # 7. Relationship style — tense relationships should not receive intimate phrasing
    if ctx.relationship_style in ("guarded", "de_escalating"):
        _intimacy_cues = (
            "mein schatz",
            "mein liebling",
            "ich liebe dich",
            "darling",
            "my love",
            "i love you",
        )
        if any(cue in _text_lc for cue in _intimacy_cues):
            issues.append("identity_rule_violation:intimacy_too_strong")

    return issues


def _score_response(ctx: LLMContext, text: str) -> float:
    """Score a candidate response for reranking.

    Higher is better.  Factors:
    - Language match           +10
    - Speech-act conformance   +5 per matched rule
    - Entity grounding         +3 per visible entity mentioned
    - Memory reference         +4 if person name or episode topic appears
    - Length penalty           -0.5 per 50 chars over ideal for speech act
    - Question in non-question acts  -3
    """
    score = 0.0
    t_lc = text.lower()

    # Language match
    if _detect_language(text) == ctx.language:
        score += 10.0

    # Speech-act conformance
    if ctx.speech_act == "repair" and "?" in text:
        score += 5.0
    if ctx.speech_act == "backchannel" and len(text) <= 100:
        score += 5.0
    if ctx.speech_act in ("assert", "confirm", "deny") and len(text) >= 20:
        score += 3.0

    # Grounding: each visible entity mentioned
    for entity in ctx.visible_persons + ctx.visible_objects:
        if entity.lower() in t_lc:
            score += 3.0

    # Memory grounding
    if ctx.person_name and ctx.person_name.lower() in t_lc:
        score += 4.0
    for ep in ctx.memory_episodes[:2]:
        # Check if any substantive word (≥5 chars) from the episode appears
        ep_words = {w.lower() for w in ep.split() if len(w) >= 5}
        if ep_words & set(t_lc.split()):
            score += 2.0
            break

    # Length penalty
    ideal = _SA_LENGTH_LIMITS.get(ctx.speech_act, 200) * 0.6  # 60% of max = ideal
    excess = max(0.0, len(text) - ideal)
    score -= (excess / 50.0) * 0.5

    # Unprompted question in non-question acts
    if ctx.speech_act not in ("ask", "repair", "hesitate") and text.count("?") > 1:
        score -= 3.0

    return score


class LLMAdapter:
    """
    Wraps an OpenAI-compatible API for generative response production.

    Usage::

        adapter = LLMAdapter()
        ctx = LLMContext(user_text="Hallo!", language="de", ...)
        reply = adapter.generate(ctx)  # "" if LLM disabled/unavailable

    Thread safety: each generate() call is independent; no shared state
    except the lazy-loaded _client.
    """

    def __init__(self) -> None:
        self._client: Optional[object] = None  # openai.OpenAI, lazy init
        self._available: bool = _ENABLED
        self._consecutive_failures: int = 0
        self._FAILURE_THRESHOLD = 5  # disable after N consecutive failures

    def _get_client(self) -> Optional[object]:
        """Lazy-init openai client.  Returns None if unavailable."""
        if self._client is not None:
            return self._client
        if not self._available:
            return None
        try:
            import openai  # type: ignore[import-untyped]

            self._client = openai.OpenAI(
                api_key=_API_KEY,
                base_url=_BASE_URL,
                timeout=_TIMEOUT,
            )
            return self._client
        except ImportError:
            logger.warning(
                "llm_adapter: openai package not installed. "
                "Run: pip install openai  (or set LLM_ENABLED=0)"
            )
            self._available = False
            return None
        except Exception as exc:
            logger.warning("llm_adapter: client init failed: %s", exc)
            self._available = False
            return None

    def generate(self, ctx: LLMContext) -> str:
        """
        Generate a validated, grounded response from the LLM.

        Algorithm:
          Attempt 1 — full context, normal parameters.
          Attempt 2 — if validation fails: reduced context (fewer concepts,
                       lower max_tokens, slightly tighter temperature).
          Attempt 3 — if still failing: minimal context (identity + speech act
                       only) with conservative parameters.

        After all attempts, the candidate with the best score (from
        _score_response) that has zero or only non-critical violations is
        returned.  If no valid candidate exists, "" is returned so the
        template pipeline takes over.

        Critical violations (always fail): empty_response, language_mismatch.
        Soft violations (logged, but candidate still eligible if best score):
          too_long, visual_claim_not_grounded, unknown_person_reference.

        Returns:
            Non-empty response string on success, "" on failure.
        """
        if not self._available:
            return ""
        if not ctx.user_text.strip():
            return ""

        client = self._get_client()
        if client is None:
            return ""

        _CRITICAL = {"empty_response", "language_mismatch", "repair_missing_question_mark"}
        _CRITICAL |= {
            "manipulative_attachment",
            "hostile_escalation",
            "ungrounded_certainty",
            "identity_rule_violation",
        }

        candidates: List[tuple] = []  # (score, text)

        # ── Attempt 1: full context ───────────────────────────────────────
        result1 = self._single_generate(ctx, _TEMPERATURE, ctx.max_tokens)
        if result1:
            issues1 = _validate_response(ctx, result1)
            has_critical1 = any(
                any(issue.startswith(c) for c in _CRITICAL) for issue in issues1
            )
            score1 = _score_response(ctx, result1)
            if issues1:
                logger.debug("llm attempt1 issues: %s", issues1)
            if not has_critical1:
                candidates.append((score1, result1))

        # ── Attempt 2: reduced context ────────────────────────────────────
        if len(candidates) == 0 or (candidates and candidates[0][0] < 8.0):
            ctx2 = self._narrow_context(ctx, level=1)
            result2 = self._single_generate(ctx2, _TEMPERATURE * 0.88, int(ctx.max_tokens * 0.75))
            if result2:
                issues2 = _validate_response(ctx, result2)
                has_critical2 = any(
                    any(issue.startswith(c) for c in _CRITICAL) for issue in issues2
                )
                score2 = _score_response(ctx, result2)
                if issues2:
                    logger.debug("llm attempt2 issues: %s", issues2)
                if not has_critical2:
                    candidates.append((score2, result2))

        # ── Attempt 3: minimal context ────────────────────────────────────
        if not candidates:
            ctx3 = self._narrow_context(ctx, level=2)
            result3 = self._single_generate(ctx3, 0.5, max(60, int(ctx.max_tokens * 0.55)))
            if result3:
                issues3 = _validate_response(ctx, result3)
                has_critical3 = any(
                    any(issue.startswith(c) for c in _CRITICAL) for issue in issues3
                )
                score3 = _score_response(ctx, result3)
                if not has_critical3:
                    candidates.append((score3, result3))

        if not candidates:
            logger.debug("llm_adapter: all %d attempts invalid — falling back to template", 3)
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._FAILURE_THRESHOLD:
                logger.warning(
                    "llm_adapter: %d consecutive fallbacks — disabling for session.",
                    self._consecutive_failures,
                )
                self._available = False
            return ""

        # ── Reranking: choose best valid candidate ────────────────────────
        best_score, best_text = max(candidates, key=lambda t: t[0])
        logger.debug(
            "llm_adapter: %d candidates, best_score=%.2f, len=%d",
            len(candidates), best_score, len(best_text)
        )
        self._consecutive_failures = 0
        return best_text

    def _single_generate(
        self, ctx: LLMContext, temperature: float, max_tokens: int
    ) -> str:
        """One raw LLM call.  Returns stripped text or "" on any error."""
        client = self._get_client()
        if client is None:
            return ""
        system_prompt = _build_system_prompt(ctx)
        try:
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ctx.user_text},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            text = (response.choices[0].message.content or "").strip()
            return text
        except Exception as exc:
            self._consecutive_failures += 1
            logger.debug(
                "llm_adapter: _single_generate failed (%d): %s",
                self._consecutive_failures, exc
            )
            if self._consecutive_failures >= self._FAILURE_THRESHOLD:
                logger.warning(
                    "llm_adapter: %d consecutive failures — disabling for session.",
                    self._consecutive_failures,
                )
                self._available = False
            return ""

    @staticmethod
    def _narrow_context(ctx: LLMContext, level: int) -> "LLMContext":
        """Return a copy of ctx with less content at the given narrowing level.

        Level 1 — drop non-essential inference content (concepts/conclusions,
                   phenomenal summary, full memory episodes).
        Level 2 — minimal: only identity-critical fields (user_text, language,
                   speech_act, emotion, person basics).  No world/memory/topic.
        """
        from dataclasses import replace as _dc_replace
        if level == 1:
            return _dc_replace(
                ctx,
                recent_concepts=[],
                recent_conclusions=[],
                phenomenal_summary="",
                memory_episodes=ctx.memory_episodes[-1:],  # keep only newest
                grounding_facts=[],
            )
        # level 2: minimal
        return _dc_replace(
            ctx,
            current_topic="",
            open_questions=[],
            recent_concepts=[],
            recent_conclusions=[],
            phenomenal_summary="",
            memory_episodes=[],
            grounding_facts=[],
            active_referents=[],
            visible_objects=[],
            visible_persons=[],
        )

    @property
    def enabled(self) -> bool:
        return self._available


# ─────────────────────────────────────────────────────────────
# Context builder  (standalone function — no Brain import at top-level)
# ─────────────────────────────────────────────────────────────


def build_llm_context(
    *,
    user_text: str,
    brain: object,
    cs: object,
    speech_act: Optional[str],
    common_ground: Optional[object],
    person_id: Optional[str],
) -> LLMContext:
    """
    Build an LLMContext from live Brain + ConsciousnessCore state.
    All attribute accesses use getattr with defaults so this is safe
    even when running in degraded / test mode.
    """
    # Language
    lang_obj = getattr(cs, "lang", None)
    lang = getattr(lang_obj, "_lang", "de") if lang_obj else "de"
    if lang not in ("de", "en"):
        lang = "de"

    # Emotional state
    em = getattr(brain, "emotion_state", None)
    emotion_dominant = "calm"
    emotion_intensity = 0.3
    stress = 0.0
    fatigue = 0.0
    if em is not None:
        dom = em.dominant() if hasattr(em, "dominant") else "calm"
        emotion_dominant = dom or "calm"
        emotion_intensity = float(getattr(em, emotion_dominant, 0.3))
        stress = float(getattr(em, "stress", 0.0))
        fatigue = float(getattr(em, "fatigue", 0.0))

    body = getattr(cs, "body", None)
    body_urgency = 0.0
    if body and hasattr(body, "homeostatic_urgency"):
        body_urgency = float(body.homeostatic_urgency())

    # Person / social — prefer recall_for_person (richer than raw PersonModel)
    person_name: Optional[str] = None
    n_shared = 0
    trust = 0.5
    interests: List[str] = []
    past_emo = "unknown"
    relationship_stage = "acquaintance"
    memory_episodes: List[str] = []
    conflict_ratio = 0.0
    positive_ratio = 0.0
    tone_bias = "neutral"

    if person_id is not None:
        # Try recall_for_person first (has episodic + social manager data)
        recall_fn = getattr(cs, "recall_for_person", None)
        if recall_fn is not None:
            try:
                _rec = recall_fn(person_id, brain)
                n_shared = int(_rec.get("n_shared_episodes", 0))
                trust = float(_rec.get("trust", 0.5))
                interests = list(_rec.get("inferred_interests", []))[:5]
                past_emo = _rec.get("dominant_past_emotion", "unknown") or "unknown"
                relationship_stage = _rec.get("relationship_type", "acquaintance") or "acquaintance"
                conflict_ratio = float(_rec.get("conflict_ratio", 0.0))
                positive_ratio = float(_rec.get("positive_ratio", 0.0))
                tone_bias = _rec.get("tone_bias", "neutral") or "neutral"
                # Build memory_episodes: use weighted_episodes if available,
                # prefixing trust tier so _build_system_prompt can hedge them
                _weps = _rec.get("weighted_episodes", [])
                if _weps:
                    memory_episodes = [
                        f"trust={ep.get('trust_tier', 'plausible')}: {ep.get('text', '')}"
                        for ep in _weps[:3]
                    ]
                else:
                    episodes_raw = _rec.get("recent_episodes", [])
                    memory_episodes = [str(e)[:120] for e in episodes_raw[-3:]]
            except Exception:
                pass
        # Supplement with PersonModel name if available
        sm = getattr(brain, "_social_manager", None)
        if sm:
            pm = sm.person_model(person_id) if hasattr(sm, "person_model") else None
            if pm is None and hasattr(sm, "get_person"):
                pm = sm.get_person(str(person_id))
            if pm is not None:
                person_name = getattr(pm, "name", None)
                if not person_name:
                    person_name = getattr(pm, "display_name", None)

    # Conversation state
    current_topic = ""
    open_questions: List[str] = []
    active_referents_list: List[str] = []
    if common_ground is not None:
        current_topic = getattr(common_ground, "current_topic", "")
        open_questions = list(getattr(common_ground, "open_questions", []))
        _raw_refs = getattr(common_ground, "active_referents", {})
        for phrase, ref_obj in _raw_refs.items():
            if hasattr(ref_obj, "referent_id") and ref_obj.discourse_status == "active":
                label = ref_obj.referent_id.split(":", 1)[-1] if ":" in ref_obj.referent_id else ref_obj.referent_id
                active_referents_list.append(f"'{phrase}' → {label}")
        # Add referent-bound open questions to surface context
        _qrb = getattr(common_ground, "_question_referent_bindings", {})
        for _oq, _rid in _qrb.items():
            _qlabel = _rid.split(":", 1)[-1] if ":" in _rid else _rid
            active_referents_list.append(f"[offene Frage über {_qlabel}: {_oq[:60]}]")

    # Recent concepts and conclusions from the stream
    concepts = list(getattr(cs, "_concepts", []))[-8:]
    conclusions_raw = list(getattr(cs, "_conclusions", []))[-4:]
    conclusions = [c for c in conclusions_raw if c and len(c) > 10]

    # Phenomenal summary
    phenom_buf = getattr(cs, "phenomenal_buffer", None)
    phenomenal_summary = ""
    if phenom_buf and hasattr(phenom_buf, "introspective_state"):
        phenomenal_summary = phenom_buf.introspective_state() or ""

    # World grounding
    world_state = getattr(brain, "world_state", None)
    visible_persons: List[str] = []
    visible_objects: List[str] = []
    if world_state is not None:
        for pid, person in getattr(world_state, "persons", {}).items():
            label = getattr(person, "name", None) or str(pid)
            visible_persons.append(label)
        for _oid, obj in getattr(world_state, "objects", {}).items():
            label = getattr(obj, "label", str(_oid))
            visible_objects.append(label)

    # Personality traits
    personality = getattr(cs, "personality", None)
    traits: List[str] = []
    if personality and hasattr(personality, "active_traits"):
        traits = list(personality.active_traits)[:5]

    # Identity / values / policy layer
    core_values = list(_CORE_VALUES[lang])
    identity_rules = list(_IDENTITY_RULES[lang])
    forbidden_behaviors = list(_FORBIDDEN_BEHAVIORS[lang])

    autobiography = getattr(cs, "autobiography", None)
    if autobiography is not None:
        for gl in list(getattr(autobiography, "guidelines", []))[:4]:
            gl_text = getattr(gl, "text", "")
            if gl_text:
                identity_rules.append(str(gl_text)[:140])

    relationship_style = "balanced"
    if conflict_ratio > 0.3 or trust < 0.35:
        relationship_style = "de_escalating"
    elif trust > 0.75 and n_shared > 5:
        relationship_style = "warm_familiar"
    elif trust < 0.5:
        relationship_style = "guarded"

    truthfulness_mode = "strict_grounded"

    # Robot embodiment state
    robot_state = ""
    robot_gesture_available = False
    _rc = getattr(brain, "_robot_controller", None)
    if _rc is not None:
        _telem = getattr(_rc, "telemetry", None)
        if _telem is not None:
            robot_gesture_available = True
            _rpos = getattr(_telem, "posture", "idle")
            _rgst = getattr(_telem, "gesture_mode", "neutral")
            robot_state = f"posture={_rpos}, gesture={_rgst}"

    # Domain H: user prosodic affect from primary TrackedPerson
    user_affect = "unknown"
    try:
        _ua_ws = getattr(brain, "_world_state", None) or getattr(brain, "world_state", None)
        _ua_sm = getattr(brain, "_social_manager", None)
        if _ua_ws is not None and _ua_sm is not None:
            _ua_pid = _ua_sm.primary_interlocutor()
            if _ua_pid is not None:
                _ua_person = getattr(_ua_ws, "persons", {}).get(_ua_pid)
                if _ua_person is None:
                    try:
                        _ua_person = getattr(_ua_ws, "persons", {}).get(int(_ua_pid))
                    except (ValueError, TypeError):
                        pass
                if _ua_person is not None:
                    user_affect = getattr(_ua_person, "speech_affect", "unknown") or "unknown"
    except Exception:
        pass

    ctx = LLMContext(
        user_text=user_text,
        language=lang,
        speech_act=speech_act or "assert",
        emotion_dominant=emotion_dominant,
        emotion_intensity=emotion_intensity,
        stress=stress,
        fatigue=fatigue,
        body_urgency=body_urgency,
        person_id=str(person_id) if person_id else None,
        person_name=person_name,
        n_shared_episodes=n_shared,
        trust=trust,
        person_interests=interests,
        past_emotion=past_emo,
        relationship_stage=relationship_stage,
        memory_episodes=memory_episodes,
        conflict_ratio=conflict_ratio,
        positive_ratio=positive_ratio,
        tone_bias=tone_bias,
        current_topic=current_topic,
        open_questions=open_questions,
        active_referents=active_referents_list,
        recent_concepts=concepts,
        recent_conclusions=conclusions,
        phenomenal_summary=phenomenal_summary,
        visible_persons=visible_persons,
        visible_objects=visible_objects,
        personality_traits=traits,
        core_values=core_values,
        identity_rules=identity_rules,
        forbidden_behaviors=forbidden_behaviors,
        relationship_style=relationship_style,
        truthfulness_mode=truthfulness_mode,
        user_affect=user_affect,
        # Grounding facts: world-model assertions the response must not contradict.
        # Currently: visible entities as declarative facts.
        grounding_facts=[
            f"Sichtbar (Person): {p}" for p in visible_persons[:4]
        ] + [
            f"Sichtbar (Objekt): {o}" for o in visible_objects[:4]
        ] + [
            f"Erinnerung: {e[:80]}" for e in memory_episodes[:2]
        ],
        robot_state=robot_state,
        robot_gesture_available=robot_gesture_available,
        max_tokens=_MAX_TOKENS,
    )
    return ctx
