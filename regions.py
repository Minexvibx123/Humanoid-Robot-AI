"""
regions.py — Brain Region Hierarchy

Each region is a population of LIF neurons with internal recurrent connections
and defined roles, mirroring human neuroanatomy. Crucially, every functional
region (Amygdala, Hippocampus, PrefrontalCortex, MotorCortex) now has a
SEMANTIC layer on top of the neural substrate:

  Amygdala.semantic_appraise()   — appraisal theory; threat/reward/novelty/social
                                    evaluated from concept content, not firing rates
  Hippocampus.semantic_encode()  — stores episodes as concept associations
  Hippocampus.semantic_recall()  — content-addressable memory retrieval
  PrefrontalCortex.semantic_goal_select() — goal driven by actual context
  MotorCortex.semantic_decode()  — action driven by goal + emotion + concepts

The neural substrate (LIF/STDP) remains intact for plasticity and activity
metrics. The semantic layer makes the substrate MEAN something.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

import neuron as _nm  # module-level ref — A_PLUS is mutated at runtime
from neuron import Neuron
from synapse import Synapse

try:
    from numba import njit
    from numba import prange as _prange

    _NUMBA = True
except ImportError:
    _NUMBA = False


if _NUMBA:

    @njit(parallel=True, fastmath=True, cache=True)
    def _lif_parallel(v, refrac, trace, I_total):
        """JIT-compiled parallel LIF step. Uses all CPU cores via OpenMP prange."""
        N = len(v)
        fired = np.zeros(N, dtype=np.bool_)
        for i in _prange(N):  # ←─ true multi-core, GIL-free
            if refrac[i] > 0.0:
                refrac[i] -= 1.0
                v[i] = -75.0  # V_RESET
                trace[i] *= 0.9512
            else:
                dv = (-(v[i] + 70.0) + I_total[i]) * 0.05  # dt/TAU_M = 1/20
                v[i] += dv
                trace[i] *= 0.9512
                if v[i] >= -55.0:  # V_THRESHOLD
                    v[i] = -75.0  # set to reset directly (V_SPIKE is transient)
                    fired[i] = True
                    refrac[i] = 2.0  # T_REFRAC
                    trace[i] = 1.0
            # biological voltage clamp
            if v[i] < -80.0:
                v[i] = -80.0
            elif v[i] > 40.0:
                v[i] = 40.0
        return fired

else:

    def _lif_parallel(v, refrac, trace, I_total):  # type: ignore[misc]
        """Pure-numpy fallback when numba is unavailable."""
        N = len(v)
        fired = np.zeros(N, dtype=np.bool_)
        refrac_mask = refrac > 0.0
        active_mask = ~refrac_mask
        refrac[refrac_mask] -= 1.0
        v[refrac_mask] = -75.0
        trace[refrac_mask] *= 0.9512
        dv = (-(v - (-70.0)) + I_total) * 0.05
        v[active_mask] += dv[active_mask]
        trace[active_mask] *= 0.9512
        threshold_hit = active_mask & (v >= -55.0)
        v[threshold_hit] = -75.0
        refrac[threshold_hit] = 2.0
        trace[threshold_hit] = 1.0
        fired[threshold_hit] = True
        np.clip(v, -80.0, 40.0, out=v)
        return fired


from lexicons_de import (
    DE_AGENCY,
    DE_ART,
    DE_BODY,
    DE_COGNITION,
    DE_EMOTION,
    DE_ETHICS,
    DE_FOOD,
    DE_IDENTITY,
    DE_LANGUAGE,
    DE_LEARNING,
    DE_NATURE,
    DE_NOVEL,
    DE_PHILOSOPHY,
    DE_REWARD,
    DE_SCIENCE,
    DE_SOCIAL,
    DE_SPACE,
    DE_TECHNOLOGY,
    DE_THREAT,
    DE_TIME,
)

# ─────────────────────────────────────────────────────────────
# Semantic appraisal lexicons (bilingual EN + DE)
# These ground the Amygdala.semantic_appraise() method in real
# content — NOT in firing rates.
# ─────────────────────────────────────────────────────────────

THREAT_CONCEPTS: frozenset = frozenset(
    {
        # English — physical danger
        "danger",
        "attack",
        "error",
        "fail",
        "failure",
        "broken",
        "hurt",
        "pain",
        "threat",
        "risk",
        "fear",
        "loss",
        "damage",
        "crash",
        "problem",
        "wrong",
        "bad",
        "terrible",
        "horrible",
        "death",
        "die",
        "destroy",
        "killed",
        "violence",
        "hate",
        "fight",
        "enemy",
        "unsafe",
        "warning",
        "critical",
        "emergency",
        "collapse",
        "disease",
        "poison",
        "virus",
        "infection",
        "wound",
        "injury",
        "burn",
        "flood",
        "fire",
        "explosion",
        "earthquake",
        "disaster",
        "accident",
        "toxic",
        "lethal",
        "fatal",
        "deadly",
        "murder",
        "demolish",
        "sabotage",
        "corrupt",
        "intrusion",
        "breach",
        "penalty",
        "rejection",
        "betrayal",
        "abandon",
        "abuse",
        "neglect",
        "suffering",
        "trauma",
        "shock",
        "panic",
        "dread",
        "terror",
        "horror",
        "nightmare",
        "trap",
        "prison",
        "cage",
        "restriction",
        "coercion",
        "manipulation",
        "deception",
        "fraud",
        "theft",
        "injustice",
        "oppression",
        "torture",
        "defeat",
        "ruin",
        "downfall",
        "catastrophe",
        "havoc",
        "wreck",
        "breakdown",
        "malfunction",
        "overflow",
        "overload",
        "crash",
        "block",
        "stuck",
        "frozen",
        "offline",
        "unavailable",
        "forbidden",
        "banned",
        "prohibited",
        "illegal",
        "criminal",
        "hostile",
        "aggression",
        "assault",
        "invasion",
        "exploitation",
        "extortion",
        "overheating",
        "overclock",
        "deadlock",
        "leak",
        "corruption",
        "loss",
        # English — psychological threat
        "loneliness",
        "isolation",
        "rejection",
        "humiliation",
        "shame",
        "guilt",
        "helpless",
        "hopeless",
        "worthless",
        "meaningless",
        "overwhelmed",
        "confused",
        "lost",
        "broken",
        "shattered",
        "numb",
        "void",
        "empty",
        # German — physical danger
        "gefahr",
        "fehler",
        "kaputt",
        "schmerz",
        "verlust",
        "verlieren",
        "absturz",
        "schlecht",
        "schlimm",
        "gefährlich",
        "warnung",
        "kritisch",
        "notfall",
        "angriff",
        "zerstören",
        "hass",
        "kampf",
        "feind",
        "tod",
        "sterben",
        "krankheit",
        "verletzung",
        "brand",
        "überschwemmung",
        "katastrophe",
        "unfall",
        "giftig",
        "tödlich",
        "mörder",
        "verrat",
        "missbrauch",
        "vernachlässigung",
        "trauma",
        "schock",
        "panik",
        "schrecken",
        "albtraum",
        "einschränkung",
        "zwang",
        "täuschung",
        "lüge",
        "betrug",
        "diebstahl",
        "korruption",
        "ungerechtigkeit",
        "unterdrückung",
        "folter",
        "niederlage",
        "bankrott",
        "kollaps",
        "zusammenbruch",
        "überlastung",
        "blockiert",
        "verboten",
        "kriminell",
        "feindlich",
        "aggression",
        "ausbeutung",
        # German — psychological threat
        "einsamkeit",
        "isolation",
        "ablehnung",
        "beschämung",
        "schuld",
        "hilflos",
        "hoffnungslos",
        "wertlos",
        "sinnlos",
        "überwältigt",
        "verloren",
        "zerbrochen",
        "taubheit",
        "leere",
        "bedeutungslos",
    }
)

REWARD_CONCEPTS: frozenset = frozenset(
    {
        # English
        "success",
        "learn",
        "discover",
        "help",
        "achieve",
        "solve",
        "understand",
        "connect",
        "happy",
        "correct",
        "good",
        "found",
        "complete",
        "create",
        "grow",
        "progress",
        "interesting",
        "beautiful",
        "love",
        "joy",
        "wonder",
        "curious",
        "explore",
        "insight",
        "knowledge",
        "truth",
        "clarity",
        "solution",
        "reward",
        "earn",
        "win",
        "gain",
        "obtain",
        "master",
        "improve",
        "advance",
        "benefit",
        "thrive",
        "flourish",
        "accomplish",
        "excel",
        "triumph",
        "heal",
        "recover",
        "protect",
        "strengthen",
        "empower",
        "enable",
        "inspire",
        "motivate",
        "encourage",
        "support",
        "nurture",
        "build",
        "develop",
        "construct",
        "invent",
        "innovate",
        "reveal",
        "unlock",
        "expand",
        "deepen",
        "enhance",
        "optimize",
        "refine",
        "perfect",
        "ideal",
        "excellent",
        "outstanding",
        "brilliant",
        "genius",
        "creative",
        "elegant",
        "harmonious",
        "balanced",
        "efficient",
        "effective",
        "powerful",
        "resilient",
        "satisfied",
        "fulfilled",
        "grateful",
        "peaceful",
        "joyful",
        "excited",
        "proud",
        "confident",
        "capable",
        "worthy",
        "meaningful",
        "purposeful",
        "significant",
        "enlightened",
        "awakened",
        "realized",
        "connected",
        "whole",
        "complete",
        "liberated",
        "free",
        "empowered",
        "transformed",
        "elevated",
        "transcended",
        "alive",
        "vibrant",
        "energized",
        "refreshed",
        "restored",
        "renewed",
        "harmony",
        "balance",
        "clarity",
        "peace",
        "serenity",
        "bliss",
        "ecstasy",
        "hope",
        "trust",
        "faith",
        "belonging",
        "acceptance",
        "recognition",
        "validation",
        "appreciation",
        "admiration",
        "respect",
        "dignity",
        "honor",
        "mastery",
        "expertise",
        "competence",
        "proficiency",
        "fluency",
        "grace",
        # German
        "erfolg",
        "lernen",
        "entdecken",
        "helfen",
        "lösung",
        "wissen",
        "gut",
        "richtig",
        "schön",
        "wachsen",
        "fortschritt",
        "interessant",
        "verstehen",
        "freude",
        "liebe",
        "neugier",
        "erkenntnis",
        "klarheit",
        "wunder",
        "belohnung",
        "gewinnen",
        "erzielen",
        "meistern",
        "verbessern",
        "vorankommen",
        "nutzen",
        "gedeihen",
        "erreichen",
        "heilen",
        "schützen",
        "stärken",
        "ermöglichen",
        "inspirieren",
        "motivieren",
        "ermutigen",
        "unterstützen",
        "entwickeln",
        "erfinden",
        "erweitern",
        "perfektionieren",
        "ausgezeichnet",
        "brillant",
        "kreativ",
        "harmonisch",
        "effizient",
        "zufrieden",
        "dankbar",
        "friedlich",
        "aufgeregt",
        "selbstbewusst",
        "bedeutsam",
        "vollendet",
        "befreit",
        "lebendig",
        "energetisiert",
        "erneuert",
        "hoffnung",
        "vertrauen",
        "zugehörigkeit",
        "anerkennung",
        "wertschätzung",
        "würde",
        "meisterschaft",
    }
)

NOVEL_CONCEPTS: frozenset = frozenset(
    {
        # English
        "new",
        "unknown",
        "strange",
        "different",
        "never",
        "first",
        "unusual",
        "surprising",
        "unexpected",
        "wonder",
        "mystery",
        "curious",
        "unique",
        "rare",
        "extraordinary",
        "unprecedented",
        "innovative",
        "original",
        "fresh",
        "exotic",
        "alien",
        "foreign",
        "unfamiliar",
        "weird",
        "bizarre",
        "peculiar",
        "remarkable",
        "astonishing",
        "amazing",
        "breathtaking",
        "breakthrough",
        "revelation",
        "discovery",
        "invention",
        "emergence",
        "complexity",
        "unpredictable",
        "dynamic",
        "evolving",
        "transforming",
        "shifting",
        "divergent",
        "outlier",
        "exception",
        "anomaly",
        "paradox",
        "contradiction",
        "enigma",
        "puzzle",
        "riddle",
        "quest",
        "adventure",
        "frontier",
        "horizon",
        "boundary",
        "limit",
        "beyond",
        "transcend",
        "quantum",
        "infinite",
        "recursive",
        "fractal",
        "emergent",
        "chaotic",
        "nonlinear",
        "self-organizing",
        "adaptive",
        "evolving",
        "mutating",
        "wander",
        "drift",
        "explore",
        "venture",
        "pioneer",
        "pathfinder",
        # German
        "neu",
        "unbekannt",
        "fremd",
        "anders",
        "erstmals",
        "ungewöhnlich",
        "überraschend",
        "unerwartet",
        "faszinierend",
        "geheimnis",
        "einzigartig",
        "selten",
        "außergewöhnlich",
        "beispiellos",
        "innovativ",
        "originell",
        "exotisch",
        "unvertraut",
        "merkwürdig",
        "rätselhaft",
        "erstaunlich",
        "durchbruch",
        "entdeckung",
        "erfindung",
        "entstehung",
        "komplex",
        "unvorhersehbar",
        "dynamisch",
        "abweichend",
        "ausnahme",
        "paradox",
        "rätsel",
        "abenteuer",
        "grenze",
        "jenseits",
        "quantensprung",
        "unendlich",
        "rekursiv",
        "fraktal",
        "adaptiv",
        "verändernd",
        "erkundend",
        "pionier",
    }
)

SOCIAL_CONCEPTS: frozenset = frozenset(
    {
        # English
        "person",
        "human",
        "face",
        "talk",
        "speak",
        "listen",
        "friend",
        "help",
        "care",
        "feel",
        "emotion",
        "understand",
        "connect",
        "together",
        "people",
        "community",
        "share",
        "communicate",
        "family",
        "mother",
        "father",
        "child",
        "children",
        "sibling",
        "partner",
        "colleague",
        "teacher",
        "student",
        "society",
        "culture",
        "group",
        "team",
        "relationship",
        "bond",
        "trust",
        "respect",
        "empathy",
        "compassion",
        "kindness",
        "generosity",
        "cooperation",
        "collaboration",
        "negotiation",
        "compromise",
        "conflict",
        "resolution",
        "harmony",
        "interaction",
        "conversation",
        "dialogue",
        "meeting",
        "greeting",
        "friendship",
        "loyalty",
        "support",
        "mentor",
        "leader",
        "follower",
        "crowd",
        "audience",
        "network",
        "connection",
        "community",
        "tribe",
        "solidarity",
        "unity",
        "diversity",
        "inclusion",
        "belonging",
        "identity",
        "role",
        "status",
        "authority",
        "influence",
        "power",
        "democracy",
        "rights",
        "freedom",
        "equality",
        "justice",
        "ethics",
        "morality",
        "values",
        "norms",
        "tradition",
        "heritage",
        "history",
        "memory",
        "story",
        "narrative",
        "celebration",
        "ritual",
        "ceremony",
        "grief",
        "comfort",
        "solidarity",
        "affection",
        "intimacy",
        "vulnerability",
        "openness",
        "honesty",
        "safety",
        # German
        "mensch",
        "gesicht",
        "sprechen",
        "hören",
        "freund",
        "helfen",
        "fühlen",
        "verstehen",
        "verbinden",
        "gemeinschaft",
        "teilen",
        "kommunizieren",
        "familie",
        "mutter",
        "vater",
        "kind",
        "kinder",
        "geschwister",
        "partner",
        "kollege",
        "lehrer",
        "schüler",
        "gesellschaft",
        "kultur",
        "gruppe",
        "team",
        "beziehung",
        "vertrauen",
        "respekt",
        "empathie",
        "mitgefühl",
        "freundlichkeit",
        "zusammenarbeit",
        "kooperation",
        "dialog",
        "gespräch",
        "zuneigung",
        "loyalität",
        "mentor",
        "führung",
        "demokratie",
        "gerechtigkeit",
        "freiheit",
        "gleichheit",
        "tradition",
        "feier",
        "trauer",
        "solidarität",
        "offenheit",
        "zugehörigkeit",
        "intimität",
        "verletzlichkeit",
        "sicherheit",
        "einheit",
        "vielfalt",
        "inklusion",
        "netzwerk",
        "gemeinschaft",
        "stamm",
        "autorität",
    }
)

AGENCY_CONCEPTS: frozenset = frozenset(
    {
        # English
        "able",
        "control",
        "make",
        "create",
        "decide",
        "choose",
        "act",
        "build",
        "change",
        "plan",
        "execute",
        "manage",
        "handle",
        "achieve",
        "organize",
        "coordinate",
        "direct",
        "lead",
        "guide",
        "implement",
        "operate",
        "perform",
        "accomplish",
        "complete",
        "finish",
        "start",
        "begin",
        "initiate",
        "launch",
        "deploy",
        "drive",
        "force",
        "push",
        "move",
        "apply",
        "utilize",
        "activate",
        "trigger",
        "produce",
        "generate",
        "process",
        "transform",
        "calculate",
        "analyze",
        "solve",
        "test",
        "evaluate",
        "measure",
        "monitor",
        "adjust",
        "correct",
        "repair",
        "restore",
        "recover",
        "adapt",
        "modify",
        "configure",
        "program",
        "define",
        "specify",
        "constrain",
        "enable",
        "disable",
        "allow",
        "prevent",
        "block",
        "filter",
        "route",
        "schedule",
        "prioritize",
        "delegate",
        "automate",
        "optimize",
        "streamline",
        "accelerate",
        "decelerate",
        "pause",
        "resume",
        "reset",
        "initialize",
        "terminate",
        "override",
        "intercept",
        "observe",
        "detect",
        "sense",
        "identify",
        "classify",
        "predict",
        "infer",
        "reason",
        "decide",
        "commit",
        "persist",
        "retry",
        "fallback",
        "escalate",
        # German
        "fähig",
        "kontrolle",
        "machen",
        "erstellen",
        "entscheiden",
        "handeln",
        "bauen",
        "ändern",
        "planen",
        "ausführen",
        "verwalten",
        "erreichen",
        "organisieren",
        "koordinieren",
        "leiten",
        "führen",
        "implementieren",
        "betreiben",
        "durchführen",
        "beginnen",
        "starten",
        "einleiten",
        "anwenden",
        "nutzen",
        "aktivieren",
        "erzeugen",
        "verarbeiten",
        "umwandeln",
        "berechnen",
        "analysieren",
        "lösen",
        "testen",
        "bewerten",
        "messen",
        "überwachen",
        "anpassen",
        "konfigurieren",
        "programmieren",
        "definieren",
        "ermöglichen",
        "verhindern",
        "blockieren",
        "planen",
        "priorisieren",
        "automatisieren",
        "optimieren",
        "beschleunigen",
        "pausieren",
        "zurücksetzen",
        "erkennen",
        "klassifizieren",
        "vorhersagen",
        "schlussfolgern",
        "entscheiden",
        "reparieren",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Extended semantic categories (15 new domains)
# ─────────────────────────────────────────────────────────────────────────────

BODY_CONCEPTS: frozenset = frozenset(
    {
        # English
        "body",
        "hand",
        "finger",
        "thumb",
        "palm",
        "fist",
        "eye",
        "sight",
        "pupil",
        "ear",
        "hearing",
        "nose",
        "smell",
        "mouth",
        "tongue",
        "taste",
        "lips",
        "teeth",
        "head",
        "neck",
        "throat",
        "brain",
        "mind",
        "heart",
        "lung",
        "breath",
        "stomach",
        "back",
        "chest",
        "shoulder",
        "elbow",
        "wrist",
        "knee",
        "ankle",
        "foot",
        "heel",
        "skin",
        "bone",
        "muscle",
        "blood",
        "nerve",
        "vein",
        "artery",
        "spine",
        "rib",
        "cell",
        "organ",
        "marrow",
        "tissue",
        "gland",
        "hormone",
        "neuron",
        "synapse",
        "pulse",
        "heartbeat",
        "breathing",
        "digestion",
        "metabolism",
        "immune",
        "heal",
        "injury",
        "pain",
        "hunger",
        "thirst",
        "sleep",
        "dream",
        "awake",
        "fatigue",
        "energy",
        "posture",
        "balance",
        "motion",
        "gesture",
        "touch",
        "sense",
        "feel",
        "physical",
        "bodily",
        "somatic",
        "alive",
        "vital",
        # German
        "körper",
        "hand",
        "finger",
        "daumen",
        "auge",
        "sicht",
        "ohr",
        "nase",
        "mund",
        "zunge",
        "lippen",
        "zähne",
        "kopf",
        "hals",
        "gehirn",
        "geist",
        "herz",
        "lunge",
        "atem",
        "magen",
        "rücken",
        "brust",
        "schulter",
        "ellbogen",
        "knie",
        "knöchel",
        "fuß",
        "haut",
        "knochen",
        "muskel",
        "blut",
        "nerv",
        "vene",
        "arterie",
        "wirbel",
        "zelle",
        "organ",
        "gewebe",
        "drüse",
        "hormon",
        "neuron",
        "synapse",
        "puls",
        "herzschlag",
        "verdauung",
        "metabolismus",
        "immunsystem",
        "heilen",
        "verletzung",
        "schmerz",
        "hunger",
        "durst",
        "schlaf",
        "traum",
        "müdigkeit",
        "energie",
        "haltung",
        "gleichgewicht",
        "bewegung",
        "geste",
        "berührung",
        "gefühl",
        "körperlich",
        "lebendig",
        "vital",
        "sensorisch",
        "motorisch",
    }
)

NATURE_CONCEPTS: frozenset = frozenset(
    {
        # English
        "tree",
        "forest",
        "river",
        "ocean",
        "mountain",
        "valley",
        "sky",
        "cloud",
        "rain",
        "snow",
        "wind",
        "storm",
        "sun",
        "moon",
        "star",
        "planet",
        "earth",
        "stone",
        "rock",
        "flower",
        "plant",
        "leaf",
        "branch",
        "seed",
        "root",
        "fruit",
        "bark",
        "moss",
        "grass",
        "bush",
        "jungle",
        "desert",
        "meadow",
        "field",
        "shore",
        "beach",
        "reef",
        "wave",
        "tide",
        "current",
        "lake",
        "pond",
        "waterfall",
        "glacier",
        "canyon",
        "volcano",
        "island",
        "continent",
        "water",
        "air",
        "fire",
        "heat",
        "cold",
        "ice",
        "frost",
        "fog",
        "mist",
        "lightning",
        "thunder",
        "rainbow",
        "aurora",
        "eclipse",
        "season",
        "spring",
        "summer",
        "autumn",
        "winter",
        "climate",
        "weather",
        "nature",
        "animal",
        "bird",
        "fish",
        "insect",
        "mammal",
        "reptile",
        "amphibian",
        "whale",
        "wolf",
        "eagle",
        "bear",
        "deer",
        "lion",
        "tiger",
        "elephant",
        "horse",
        "dog",
        "cat",
        "butterfly",
        "bee",
        "ant",
        "spider",
        "snake",
        "turtle",
        "dolphin",
        "shark",
        "ecosystem",
        "habitat",
        "biome",
        "species",
        "evolution",
        "biology",
        "organism",
        "genetics",
        "ecology",
        "environment",
        "biodiversity",
        "extinction",
        "symbiosis",
        "predator",
        "prey",
        "migration",
        "adaptation",
        "camouflage",
        "photosynthesis",
        "atmosphere",
        "ocean",
        "biosphere",
        "erosion",
        "soil",
        "mineral",
        "crystal",
        # German
        "baum",
        "wald",
        "fluss",
        "ozean",
        "berg",
        "tal",
        "himmel",
        "wolke",
        "regen",
        "schnee",
        "wind",
        "sturm",
        "sonne",
        "mond",
        "stern",
        "planet",
        "erde",
        "stein",
        "fels",
        "blume",
        "pflanze",
        "blatt",
        "zweig",
        "samen",
        "wurzel",
        "frucht",
        "rinde",
        "moos",
        "gras",
        "strauch",
        "dschungel",
        "wüste",
        "wiese",
        "feld",
        "strand",
        "riff",
        "welle",
        "gezut",
        "see",
        "teich",
        "wasserfall",
        "gletscher",
        "schlucht",
        "vulkan",
        "insel",
        "kontinent",
        "wasser",
        "luft",
        "feuer",
        "wärme",
        "kälte",
        "eis",
        "frost",
        "nebel",
        "blitz",
        "donner",
        "regenbogen",
        "jahreszeit",
        "frühling",
        "sommer",
        "herbst",
        "winter",
        "klima",
        "natur",
        "tier",
        "vogel",
        "fisch",
        "insekt",
        "säugetier",
        "wal",
        "wolf",
        "adler",
        "bär",
        "hirsch",
        "löwe",
        "tiger",
        "elefant",
        "pferd",
        "schmetterling",
        "biene",
        "ameise",
        "spinne",
        "schlange",
        "delfin",
        "ökosystem",
        "lebensraum",
        "art",
        "evolution",
        "biologie",
        "organismus",
        "genetik",
        "ökologie",
        "umwelt",
        "biodiversität",
        "anpassung",
        "atmosphäre",
        "boden",
        "mineral",
        "kristall",
        "photosynthese",
    }
)

TECHNOLOGY_CONCEPTS: frozenset = frozenset(
    {
        # English
        "computer",
        "robot",
        "sensor",
        "network",
        "data",
        "code",
        "algorithm",
        "model",
        "software",
        "hardware",
        "processor",
        "memory",
        "storage",
        "database",
        "server",
        "client",
        "internet",
        "signal",
        "digital",
        "analog",
        "electric",
        "electronic",
        "circuit",
        "battery",
        "power",
        "motor",
        "actuator",
        "display",
        "camera",
        "speaker",
        "keyboard",
        "mouse",
        "screen",
        "interface",
        "protocol",
        "communication",
        "wireless",
        "bluetooth",
        "antenna",
        "frequency",
        "bandwidth",
        "latency",
        "throughput",
        "packet",
        "encoding",
        "decoding",
        "compression",
        "encryption",
        "security",
        "firewall",
        "simulation",
        "virtual",
        "artificial",
        "automation",
        "optimization",
        "feedback",
        "loop",
        "control",
        "system",
        "architecture",
        "module",
        "component",
        "library",
        "framework",
        "platform",
        "cloud",
        "stream",
        "pipeline",
        "thread",
        "process",
        "kernel",
        "driver",
        "firmware",
        "bootloader",
        "runtime",
        "compiler",
        "interpreter",
        "parser",
        "token",
        "syntax",
        "grammar",
        "operator",
        "function",
        "class",
        "object",
        "method",
        "variable",
        "constant",
        "parameter",
        "argument",
        "return",
        "exception",
        "handler",
        "debugger",
        "profiler",
        "monitor",
        "logger",
        "sensor",
        "detector",
        "scanner",
        "actuator",
        "servo",
        "stepper",
        "encoder",
        "decoder",
        "transducer",
        "oscillator",
        "transistor",
        "capacitor",
        "resistor",
        "inductor",
        "diode",
        "amplifier",
        "filter",
        "multiplexer",
        "register",
        "cache",
        "buffer",
        "stack",
        "queue",
        "heap",
        "pointer",
        "address",
        "instruction",
        "cycle",
        "clock",
        "latency",
        "pipeline",
        "interrupt",
        "neural",
        "network",
        "weight",
        "activation",
        "gradient",
        "training",
        "inference",
        "embedding",
        "vector",
        "matrix",
        "tensor",
        "layer",
        "epoch",
        "batch",
        "learning",
        "classification",
        "regression",
        "clustering",
        "detection",
        "recognition",
        "prediction",
        # German
        "computer",
        "roboter",
        "sensor",
        "netzwerk",
        "daten",
        "code",
        "algorithmus",
        "modell",
        "software",
        "hardware",
        "prozessor",
        "speicher",
        "datenbank",
        "server",
        "internet",
        "signal",
        "digital",
        "analog",
        "elektrisch",
        "elektronisch",
        "schaltkreis",
        "batterie",
        "energie",
        "motor",
        "aktuator",
        "anzeige",
        "kamera",
        "mikrofon",
        "lautsprecher",
        "tastatur",
        "bildschirm",
        "schnittstelle",
        "protokoll",
        "kommunikation",
        "drahtlos",
        "frequenz",
        "bandbreite",
        "komprimierung",
        "verschlüsselung",
        "sicherheit",
        "simulation",
        "virtuell",
        "künstlich",
        "automatisierung",
        "optimierung",
        "steuerung",
        "feedback",
        "architektur",
        "modul",
        "komponente",
        "bibliothek",
        "plattform",
        "cloud",
        "pipeline",
        "thread",
        "prozess",
        "compiler",
        "parser",
        "token",
        "funktion",
        "klasse",
        "objekt",
        "variable",
        "konstante",
        "parameter",
        "ausnahme",
        "debugger",
        "monitor",
        "logger",
        "transistor",
        "kondensator",
        "widerstand",
        "diode",
        "verstärker",
        "filter",
        "register",
        "puffer",
        "stapel",
        "warteschlange",
        "adresse",
        "befehl",
        "takt",
        "interrupt",
        "netz",
        "gewicht",
        "aktivierung",
        "gradient",
        "training",
        "erkennung",
        "vorhersage",
        "einbettung",
        "vektor",
        "matrix",
        "schicht",
        "klassifizierung",
        "regression",
        "clustering",
    }
)

SCIENCE_CONCEPTS: frozenset = frozenset(
    {
        # English
        "experiment",
        "hypothesis",
        "theory",
        "data",
        "measure",
        "analyze",
        "observe",
        "predict",
        "test",
        "validate",
        "evidence",
        "proof",
        "research",
        "study",
        "calculate",
        "formula",
        "pattern",
        "structure",
        "function",
        "property",
        "relation",
        "interaction",
        "cause",
        "effect",
        "correlation",
        "variable",
        "constant",
        "threshold",
        "spectrum",
        "frequency",
        "wavelength",
        "particle",
        "atom",
        "molecule",
        "electron",
        "proton",
        "neutron",
        "nucleus",
        "orbital",
        "quantum",
        "wave",
        "force",
        "energy",
        "mass",
        "momentum",
        "velocity",
        "acceleration",
        "gravity",
        "magnetism",
        "charge",
        "field",
        "potential",
        "entropy",
        "probability",
        "statistics",
        "distribution",
        "deviation",
        "mean",
        "median",
        "variance",
        "significance",
        "confidence",
        "interval",
        "sample",
        "population",
        "experiment",
        "control",
        "placebo",
        "blind",
        "replicate",
        "peer",
        "review",
        "publish",
        "citation",
        "methodology",
        "analysis",
        "synthesis",
        "conclusion",
        "discovery",
        "invention",
        "innovation",
        "paradigm",
        "revolution",
        "falsify",
        "verify",
        "model",
        "simulation",
        "prediction",
        "explanation",
        "mechanism",
        "process",
        "cycle",
        "reaction",
        "equilibrium",
        "symmetry",
        "asymmetry",
        "complexity",
        "emergence",
        "biology",
        "chemistry",
        "physics",
        "mathematics",
        "geometry",
        "algebra",
        "calculus",
        "topology",
        "logic",
        "computation",
        "information",
        "entropy",
        "chaos",
        "order",
        "relativity",
        "evolution",
        "genetics",
        "neuroscience",
        "cosmology",
        "ecology",
        "geology",
        "meteorology",
        "astronomy",
        "medicine",
        "psychology",
        "sociology",
        # German
        "experiment",
        "hypothese",
        "theorie",
        "messung",
        "analysieren",
        "beobachten",
        "vorhersagen",
        "testen",
        "validieren",
        "beweis",
        "forschung",
        "studie",
        "berechnen",
        "modell",
        "formel",
        "muster",
        "struktur",
        "funktion",
        "eigenschaft",
        "zusammenhang",
        "wechselwirkung",
        "ursache",
        "wirkung",
        "variable",
        "konstante",
        "spektrum",
        "frequenz",
        "wellenlänge",
        "teilchen",
        "atom",
        "molekül",
        "elektron",
        "proton",
        "neutron",
        "kern",
        "quanten",
        "welle",
        "kraft",
        "energie",
        "masse",
        "impuls",
        "geschwindigkeit",
        "beschleunigung",
        "gravitation",
        "magnetismus",
        "ladung",
        "feld",
        "entropie",
        "wahrscheinlichkeit",
        "statistik",
        "verteilung",
        "abweichung",
        "mittel",
        "signifikanz",
        "konfidenz",
        "stichprobe",
        "kontrolle",
        "replikation",
        "peer-review",
        "veröffentlichung",
        "methodik",
        "synthese",
        "schlussfolgerung",
        "entdeckung",
        "paradigma",
        "revolution",
        "falsifizieren",
        "verifizieren",
        "simulation",
        "mechanismus",
        "reaktion",
        "gleichgewicht",
        "symmetrie",
        "komplexität",
        "emergenz",
        "biologie",
        "chemie",
        "physik",
        "mathematik",
        "geometrie",
        "algebra",
        "analysis",
        "logik",
        "relativität",
        "evolution",
        "genetik",
        "neurowissenschaft",
        "kosmologie",
        "ökologie",
        "geologie",
        "meteorologie",
        "astronomie",
        "medizin",
        "psychologie",
        "soziologie",
    }
)

COGNITION_CONCEPTS: frozenset = frozenset(
    {
        # English
        "think",
        "thought",
        "memory",
        "imagine",
        "understand",
        "reason",
        "analyze",
        "decide",
        "focus",
        "attention",
        "perception",
        "awareness",
        "consciousness",
        "mind",
        "intelligence",
        "wisdom",
        "intuition",
        "insight",
        "judgment",
        "inference",
        "deduction",
        "induction",
        "assumption",
        "belief",
        "knowledge",
        "confusion",
        "clarity",
        "realization",
        "recognition",
        "identification",
        "classification",
        "abstraction",
        "generalization",
        "concept",
        "idea",
        "mental",
        "cognitive",
        "neural",
        "reflect",
        "ponder",
        "consider",
        "evaluate",
        "compare",
        "contrast",
        "distinguish",
        "integrate",
        "synthesize",
        "creative",
        "logical",
        "rational",
        "irrational",
        "skeptical",
        "critical",
        "analytical",
        "systematic",
        "intuitive",
        "spatial",
        "verbal",
        "numerical",
        "visual",
        "abstract",
        "concrete",
        "metacognition",
        "strategy",
        "planning",
        "problem-solving",
        "decision",
        "working",
        "episodic",
        "semantic",
        "procedural",
        "explicit",
        "implicit",
        "consolidation",
        "encoding",
        "retrieval",
        "forgetting",
        "priming",
        "associative",
        "pattern",
        "recognition",
        "categorization",
        "schema",
        "frame",
        "context",
        "salience",
        "relevance",
        "attention",
        "filter",
        "bias",
        "heuristic",
        "belief",
        "update",
        "prior",
        "posterior",
        "bayesian",
        "prediction",
        "model",
        "world",
        "representation",
        "simulation",
        "mental",
        "imagery",
        "visualization",
        "introspection",
        "self-awareness",
        "theory-of-mind",
        "empathy",
        "perspective",
        "mentalizing",
        "mirror",
        "resonance",
        # German
        "denken",
        "gedanke",
        "erinnerung",
        "vorstellen",
        "verstehen",
        "vernunft",
        "analysieren",
        "entscheiden",
        "fokus",
        "aufmerksamkeit",
        "wahrnehmung",
        "bewusstsein",
        "geist",
        "intelligenz",
        "weisheit",
        "intuition",
        "erkenntnis",
        "urteil",
        "schlussfolgerung",
        "annahme",
        "überzeugung",
        "wissen",
        "verwirrung",
        "klarheit",
        "erkennen",
        "identifizieren",
        "klassifizieren",
        "abstraktion",
        "verallgemeinerung",
        "konzept",
        "idee",
        "mental",
        "kognitiv",
        "neural",
        "reflektieren",
        "überlegen",
        "bewerten",
        "vergleichen",
        "unterscheiden",
        "integrieren",
        "synthetisieren",
        "kreativ",
        "logisch",
        "rational",
        "skeptisch",
        "analytisch",
        "systematisch",
        "intuitiv",
        "metakognition",
        "planung",
        "problemlösung",
        "arbeitsgedächtnis",
        "episodisch",
        "semantisch",
        "prozedural",
        "explizit",
        "implizit",
        "konsolidierung",
        "enkodierung",
        "abruf",
        "vergessen",
        "priming",
        "assoziativ",
        "muster",
        "schema",
        "kontext",
        "relevanz",
        "filter",
        "heuristik",
        "vorwissen",
        "bayesianisch",
        "vorhersage",
        "modell",
        "vorstellung",
        "visualisierung",
        "selbstwahrnehmung",
        "empathie",
        "perspektive",
    }
)

TIME_CONCEPTS: frozenset = frozenset(
    {
        # English
        "time",
        "moment",
        "instant",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "decade",
        "century",
        "millennium",
        "past",
        "present",
        "future",
        "history",
        "ancient",
        "medieval",
        "modern",
        "contemporary",
        "early",
        "late",
        "previous",
        "current",
        "next",
        "recent",
        "eventual",
        "eventual",
        "often",
        "sometimes",
        "always",
        "never",
        "soon",
        "already",
        "still",
        "ever",
        "begin",
        "start",
        "end",
        "finish",
        "duration",
        "delay",
        "speed",
        "slow",
        "fast",
        "temporary",
        "permanent",
        "eternal",
        "endless",
        "cycle",
        "rhythm",
        "sequence",
        "schedule",
        "deadline",
        "timing",
        "phase",
        "stage",
        "epoch",
        "era",
        "period",
        "interval",
        "frequency",
        "rate",
        "flux",
        "evolution",
        "change",
        "growth",
        "decay",
        "aging",
        "renewal",
        "memory",
        "nostalgia",
        "anticipation",
        "regret",
        "progress",
        "regression",
        "stagnation",
        "momentum",
        "inertia",
        "acceleration",
        "pause",
        "continuity",
        "discontinuity",
        "transition",
        "threshold",
        "milestone",
        "anniversary",
        # German
        "zeit",
        "moment",
        "sekunde",
        "minute",
        "stunde",
        "tag",
        "woche",
        "monat",
        "jahr",
        "jahrzehnt",
        "jahrhundert",
        "jahrtausend",
        "vergangenheit",
        "gegenwart",
        "zukunft",
        "geschichte",
        "veraltet",
        "früher",
        "modern",
        "früh",
        "spät",
        "vorherig",
        "aktuell",
        "bald",
        "bereits",
        "noch",
        "beginnen",
        "enden",
        "dauer",
        "verzögerung",
        "geschwindigkeit",
        "langsam",
        "schnell",
        "vorübergehend",
        "ewig",
        "kreislauf",
        "rhythmus",
        "reihenfolge",
        "termin",
        "phase",
        "stadium",
        "epoche",
        "periode",
        "intervall",
        "häufigkeit",
        "wandel",
        "wachstum",
        "alterung",
        "erneuerung",
        "erinnerung",
        "nostalgie",
        "vorfreude",
        "reue",
        "fortschritt",
        "rückschritt",
        "beschleunigung",
        "pause",
        "kontinuität",
        "übergang",
        "meilenstein",
        "jahrestag",
        "zeitplan",
        "timing",
        "moment",
        "zeitraum",
    }
)

SPACE_CONCEPTS: frozenset = frozenset(
    {
        # English
        "space",
        "place",
        "location",
        "position",
        "direction",
        "distance",
        "size",
        "shape",
        "area",
        "volume",
        "depth",
        "height",
        "width",
        "length",
        "near",
        "far",
        "left",
        "right",
        "center",
        "edge",
        "boundary",
        "limit",
        "inside",
        "outside",
        "above",
        "below",
        "between",
        "around",
        "toward",
        "front",
        "back",
        "corner",
        "surface",
        "dimension",
        "coordinate",
        "angle",
        "curve",
        "path",
        "route",
        "field",
        "room",
        "region",
        "territory",
        "world",
        "universe",
        "galaxy",
        "planet",
        "orbit",
        "trajectory",
        "vector",
        "magnitude",
        "scale",
        "proportion",
        "ratio",
        "symmetry",
        "rotation",
        "translation",
        "reflection",
        "projection",
        "perspective",
        "horizon",
        "depth",
        "parallax",
        "gradient",
        "topology",
        "manifold",
        "sphere",
        "cube",
        "cylinder",
        "cone",
        "circle",
        "square",
        "triangle",
        "polygon",
        "point",
        "line",
        "plane",
        "solid",
        "hollow",
        "dense",
        "sparse",
        "grid",
        "mesh",
        "lattice",
        "north",
        "south",
        "east",
        "west",
        "altitude",
        "latitude",
        "longitude",
        "elevation",
        "zenith",
        "nadir",
        "azimuth",
        "bearing",
        "heading",
        "axis",
        "frame",
        "reference",
        "local",
        "global",
        "relative",
        "absolute",
        "fixed",
        "moving",
        "static",
        "dynamic",
        # German
        "raum",
        "ort",
        "position",
        "richtung",
        "entfernung",
        "größe",
        "form",
        "fläche",
        "volumen",
        "tiefe",
        "höhe",
        "breite",
        "länge",
        "nah",
        "fern",
        "links",
        "rechts",
        "mitte",
        "rand",
        "grenze",
        "innen",
        "außen",
        "oberhalb",
        "unterhalb",
        "zwischen",
        "vorne",
        "hinten",
        "ecke",
        "oberfläche",
        "dimension",
        "winkel",
        "kurve",
        "pfad",
        "weg",
        "gebiet",
        "region",
        "territorium",
        "welt",
        "universum",
        "galaxie",
        "planet",
        "umlaufbahn",
        "vektor",
        "skala",
        "proportion",
        "symmetrie",
        "rotation",
        "projektion",
        "perspektive",
        "horizont",
        "topologie",
        "kugel",
        "würfel",
        "zylinder",
        "kreis",
        "dreieck",
        "punkt",
        "linie",
        "ebene",
        "körper",
        "gitter",
        "norden",
        "süden",
        "osten",
        "westen",
        "höhe",
        "breite",
        "längengrad",
        "breitengrad",
        "achse",
        "referenz",
        "lokal",
        "global",
        "relativ",
        "absolut",
        "statisch",
        "dynamisch",
        "bewegend",
    }
)

LANGUAGE_CONCEPTS: frozenset = frozenset(
    {
        # English
        "word",
        "language",
        "speak",
        "write",
        "text",
        "meaning",
        "grammar",
        "sentence",
        "paragraph",
        "story",
        "symbol",
        "sign",
        "code",
        "message",
        "question",
        "answer",
        "describe",
        "explain",
        "name",
        "define",
        "translate",
        "interpret",
        "read",
        "listen",
        "voice",
        "sound",
        "letter",
        "alphabet",
        "vocabulary",
        "expression",
        "narrative",
        "poem",
        "metaphor",
        "concept",
        "abstract",
        "concrete",
        "literal",
        "figurative",
        "communicate",
        "convey",
        "inform",
        "tell",
        "ask",
        "respond",
        "argue",
        "persuade",
        "teach",
        "quote",
        "cite",
        "reference",
        "context",
        "subtext",
        "implication",
        "nuance",
        "tone",
        "register",
        "dialect",
        "accent",
        "pronunciation",
        "syntax",
        "semantics",
        "pragmatics",
        "morphology",
        "phonology",
        "discourse",
        "rhetoric",
        "style",
        "structure",
        "form",
        "genre",
        "fiction",
        "nonfiction",
        "documentary",
        "essay",
        "report",
        "analysis",
        "summary",
        "translation",
        "transcription",
        "annotation",
        "glossary",
        "etymology",
        "idiom",
        "phrase",
        "clause",
        "subject",
        "predicate",
        "object",
        "adjective",
        "adverb",
        "noun",
        "verb",
        "preposition",
        "article",
        "conjunction",
        "interjection",
        "tense",
        "aspect",
        "mood",
        "voice",
        "case",
        "number",
        # German
        "wort",
        "sprache",
        "sprechen",
        "schreiben",
        "text",
        "bedeutung",
        "grammatik",
        "satz",
        "absatz",
        "geschichte",
        "symbol",
        "zeichen",
        "nachricht",
        "frage",
        "antwort",
        "beschreiben",
        "erklären",
        "benennen",
        "definieren",
        "übersetzen",
        "interpretieren",
        "lesen",
        "zuhören",
        "stimme",
        "klang",
        "buchstabe",
        "alphabet",
        "vokabular",
        "ausdruck",
        "erzählung",
        "gedicht",
        "metapher",
        "konzept",
        "abstrakt",
        "konkret",
        "kommunizieren",
        "informieren",
        "argumentieren",
        "überzeugen",
        "lehren",
        "zitieren",
        "kontext",
        "untertöne",
        "ton",
        "register",
        "dialekt",
        "aussprache",
        "syntax",
        "semantik",
        "pragmatik",
        "morphologie",
        "diskurs",
        "rhetorik",
        "stil",
        "struktur",
        "form",
        "gattung",
        "fiktion",
        "analyse",
        "zusammenfassung",
        "übersetzung",
        "etymologie",
        "idiom",
        "phrase",
        "satzteil",
        "substantiv",
        "verb",
        "adjektiv",
        "adverb",
        "präposition",
        "artikel",
        "konjunktion",
        "tempus",
        "aspekt",
        "modus",
        "kasus",
    }
)

ART_CONCEPTS: frozenset = frozenset(
    {
        # English
        "art",
        "music",
        "painting",
        "drawing",
        "color",
        "image",
        "picture",
        "sculpture",
        "dance",
        "theater",
        "film",
        "cinema",
        "photography",
        "design",
        "architecture",
        "literature",
        "poetry",
        "fiction",
        "story",
        "narrative",
        "song",
        "melody",
        "rhythm",
        "harmony",
        "chord",
        "note",
        "scale",
        "tempo",
        "beat",
        "measure",
        "composition",
        "style",
        "aesthetic",
        "beauty",
        "imagination",
        "creativity",
        "expression",
        "performance",
        "artist",
        "actor",
        "musician",
        "painter",
        "writer",
        "director",
        "creator",
        "craft",
        "skill",
        "technique",
        "vision",
        "inspiration",
        "emotion",
        "atmosphere",
        "abstract",
        "symbolic",
        "realistic",
        "surreal",
        "minimal",
        "expressionist",
        "impressionist",
        "cubist",
        "romantic",
        "classical",
        "modern",
        "contemporary",
        "experimental",
        "avant-garde",
        "installation",
        "digital",
        "pixel",
        "brush",
        "canvas",
        "palette",
        "studio",
        "gallery",
        "museum",
        "stage",
        "curtain",
        "spotlight",
        "instrument",
        "violin",
        "piano",
        "guitar",
        "drum",
        "flute",
        "voice",
        "choir",
        "orchestra",
        "band",
        "solo",
        "duet",
        "improvisation",
        "score",
        "opus",
        "theme",
        "motif",
        "variation",
        "cadence",
        "dissonance",
        "consonance",
        "texture",
        "pattern",
        "symmetry",
        "proportion",
        "perspective",
        "depth",
        "contrast",
        "shadow",
        "light",
        "color",
        "hue",
        "saturation",
        "tone",
        "shading",
        "rendering",
        "detail",
        # German
        "kunst",
        "musik",
        "malerei",
        "zeichnung",
        "farbe",
        "bild",
        "skulptur",
        "tanz",
        "theater",
        "film",
        "fotografie",
        "design",
        "architektur",
        "literatur",
        "gedicht",
        "fiktion",
        "erzählung",
        "lied",
        "melodie",
        "rhythmus",
        "harmonie",
        "akkord",
        "note",
        "tempo",
        "komposition",
        "stil",
        "ästhetik",
        "schönheit",
        "vorstellung",
        "kreativität",
        "ausdruck",
        "aufführung",
        "künstler",
        "schauspieler",
        "musiker",
        "maler",
        "schriftsteller",
        "regisseur",
        "erschaffer",
        "handwerk",
        "technik",
        "vision",
        "inspiration",
        "atmosphäre",
        "abstrakt",
        "symbolisch",
        "realistisch",
        "surreal",
        "minimal",
        "expressionistisch",
        "impressionistisch",
        "romantisch",
        "klassisch",
        "modern",
        "experimentell",
        "installation",
        "pinsel",
        "leinwand",
        "palette",
        "atelier",
        "galerie",
        "museum",
        "bühne",
        "spotlight",
        "instrument",
        "violine",
        "klavier",
        "gitarre",
        "schlagzeug",
        "chor",
        "orchester",
        "improvisation",
        "partitur",
        "thema",
        "motiv",
        "variation",
        "textur",
        "kontrast",
        "schatten",
        "licht",
        "farbton",
        "sättigung",
        "ton",
        "schattierung",
    }
)

LEARNING_CONCEPTS: frozenset = frozenset(
    {
        # English
        "learn",
        "study",
        "teach",
        "practice",
        "repeat",
        "improve",
        "skill",
        "mastery",
        "knowledge",
        "understand",
        "memorize",
        "exercise",
        "train",
        "educate",
        "develop",
        "progress",
        "advance",
        "experiment",
        "discover",
        "adapt",
        "feedback",
        "reflect",
        "review",
        "analyze",
        "challenge",
        "strategy",
        "method",
        "approach",
        "explore",
        "investigate",
        "observe",
        "conclude",
        "apply",
        "transfer",
        "integrate",
        "synthesize",
        "evaluate",
        "assess",
        "correct",
        "growth",
        "potential",
        "ability",
        "competence",
        "expertise",
        "proficiency",
        "fluency",
        "literacy",
        "numeracy",
        "creativity",
        "critical-thinking",
        "collaboration",
        "communication",
        "problem-solving",
        "research",
        "curiosity",
        "motivation",
        "engagement",
        "retention",
        "recall",
        "recognition",
        "comprehension",
        "application",
        "analysis",
        "evaluation",
        "creation",
        "bloom",
        "scaffolding",
        "modeling",
        "simulation",
        "game",
        "play",
        "project",
        "task",
        "goal",
        "objective",
        "standard",
        "curriculum",
        "lesson",
        "homework",
        "quiz",
        "test",
        "exam",
        "grade",
        "feedback",
        "rubric",
        "portfolio",
        "reflection",
        "journal",
        "seminar",
        "workshop",
        "course",
        "lecture",
        "tutorial",
        "demonstration",
        "practice",
        "drill",
        "spaced-repetition",
        "retrieval",
        "interleaving",
        "elaboration",
        "generation",
        # German
        "lernen",
        "studieren",
        "lehren",
        "üben",
        "wiederholen",
        "verbessern",
        "fähigkeit",
        "meisterschaft",
        "wissen",
        "verstehen",
        "auswendig",
        "trainieren",
        "erziehen",
        "entwickeln",
        "fortschritt",
        "experimentieren",
        "entdecken",
        "anpassen",
        "rückmeldung",
        "reflektieren",
        "überprüfen",
        "analysieren",
        "herausforderung",
        "strategie",
        "methode",
        "erkunden",
        "untersuchen",
        "beobachten",
        "schlussfolgern",
        "anwenden",
        "integrieren",
        "synthetisieren",
        "bewerten",
        "korrigieren",
        "wachstum",
        "potential",
        "kompetenz",
        "expertise",
        "kenntnisse",
        "alphabetisierung",
        "kreativität",
        "kritisches-denken",
        "zusammenarbeit",
        "problemlösung",
        "neugier",
        "motivation",
        "engagement",
        "behalten",
        "abrufen",
        "erkennen",
        "verständnis",
        "anwendung",
        "schöpfung",
        "unterricht",
        "aufgabe",
        "ziel",
        "lehrplan",
        "lektion",
        "hausaufgabe",
        "prüfung",
        "benotung",
        "feedback",
        "portfolio",
        "reflexion",
        "tagebuch",
        "seminar",
        "kurs",
        "vorlesung",
        "tutorial",
        "übung",
        "wiederholung",
        "abrufübung",
        "elaboration",
        "generierung",
    }
)

ETHICS_CONCEPTS: frozenset = frozenset(
    {
        # English
        "right",
        "wrong",
        "justice",
        "fairness",
        "moral",
        "value",
        "principle",
        "honesty",
        "integrity",
        "respect",
        "dignity",
        "freedom",
        "equality",
        "responsibility",
        "duty",
        "obligation",
        "harm",
        "benefit",
        "consent",
        "autonomy",
        "compassion",
        "empathy",
        "trust",
        "betrayal",
        "loyalty",
        "virtue",
        "vice",
        "truth",
        "deception",
        "punishment",
        "reward",
        "law",
        "rule",
        "norm",
        "ethics",
        "morality",
        "good",
        "evil",
        "sacred",
        "worthy",
        "permitted",
        "forbidden",
        "acceptable",
        "unacceptable",
        "ethical",
        "unethical",
        "legal",
        "illegal",
        "valid",
        "invalid",
        "fair",
        "unfair",
        "bias",
        "discrimination",
        "prejudice",
        "stereotype",
        "privilege",
        "oppression",
        "liberation",
        "sovereignty",
        "accountability",
        "transparency",
        "corruption",
        "integrity",
        "whistleblowing",
        "advocacy",
        "activism",
        "reform",
        "revolution",
        "evolution",
        "peace",
        "conflict",
        "reconciliation",
        "forgiveness",
        "redemption",
        "punishment",
        "rehabilitation",
        "empowerment",
        "solidarity",
        "mutual-aid",
        "commons",
        "public-good",
        "harm-reduction",
        "consent",
        "agency",
        "autonomy",
        "self-determination",
        "rights",
        "human-rights",
        "civil-rights",
        "animal-rights",
        "environmental-ethics",
        "global",
        # German
        "richtig",
        "falsch",
        "gerechtigkeit",
        "fairness",
        "moral",
        "wert",
        "prinzip",
        "ehrlichkeit",
        "integrität",
        "respekt",
        "würde",
        "freiheit",
        "gleichheit",
        "verantwortung",
        "pflicht",
        "schaden",
        "nutzen",
        "zustimmung",
        "autonomie",
        "mitgefühl",
        "vertrauen",
        "verrat",
        "loyalität",
        "tugend",
        "laster",
        "wahrheit",
        "täuschung",
        "strafe",
        "gesetz",
        "regel",
        "norm",
        "ethik",
        "moralität",
        "gut",
        "böse",
        "heilig",
        "würdig",
        "verboten",
        "erlaubt",
        "akzeptabel",
        "ethisch",
        "legal",
        "illegal",
        "fair",
        "unfair",
        "diskriminierung",
        "vorurteil",
        "privileg",
        "unterdrückung",
        "befreiung",
        "souveränität",
        "rechenschaftspflicht",
        "transparenz",
        "korruption",
        "aktivismus",
        "reform",
        "frieden",
        "konflikt",
        "versöhnung",
        "vergebung",
        "erlösung",
        "rehabilitation",
        "solidarität",
        "gemeinwohl",
        "schadensminimierung",
        "einwilligung",
        "selbstbestimmung",
        "menschenrechte",
        "tierrechte",
        "umweltethik",
        "global",
    }
)

PHILOSOPHY_CONCEPTS: frozenset = frozenset(
    {
        # English
        "existence",
        "being",
        "reality",
        "truth",
        "consciousness",
        "mind",
        "soul",
        "spirit",
        "meaning",
        "purpose",
        "identity",
        "self",
        "freedom",
        "determinism",
        "causality",
        "infinity",
        "void",
        "absolute",
        "relative",
        "subjective",
        "objective",
        "abstract",
        "concrete",
        "essence",
        "appearance",
        "substance",
        "form",
        "matter",
        "energy",
        "paradox",
        "contradiction",
        "unity",
        "duality",
        "transcendence",
        "immanence",
        "metaphysics",
        "ontology",
        "epistemology",
        "logic",
        "rationality",
        "certainty",
        "doubt",
        "skepticism",
        "empiricism",
        "rationalism",
        "idealism",
        "materialism",
        "dualism",
        "monism",
        "pluralism",
        "nihilism",
        "existentialism",
        "phenomenology",
        "pragmatism",
        "utilitarianism",
        "deontology",
        "virtue-ethics",
        "contractualism",
        "simulation",
        "solipsism",
        "panpsychism",
        "emergentism",
        "reductionism",
        "holism",
        "dialectic",
        "thesis",
        "antithesis",
        "synthesis",
        "deconstruction",
        "hermeneutics",
        "semiotics",
        "structuralism",
        "poststructuralism",
        "postmodernism",
        "qualia",
        "intentionality",
        "phenomenal",
        "noumenal",
        "dao",
        "karma",
        "dharma",
        "nirvana",
        "enlightenment",
        "tao",
        "karma",
        "moksha",
        "brahman",
        "atman",
        # German
        "existenz",
        "sein",
        "realität",
        "wahrheit",
        "bewusstsein",
        "geist",
        "seele",
        "bedeutung",
        "zweck",
        "identität",
        "selbst",
        "freiheit",
        "determinismus",
        "kausalität",
        "unendlichkeit",
        "absolut",
        "relativ",
        "subjektiv",
        "objektiv",
        "abstrakt",
        "konkret",
        "wesen",
        "erscheinung",
        "substanz",
        "materie",
        "energie",
        "paradox",
        "widerspruch",
        "einheit",
        "dualität",
        "transzendenz",
        "immanenz",
        "metaphysik",
        "ontologie",
        "erkenntnislehre",
        "logik",
        "rationalität",
        "gewissheit",
        "zweifel",
        "skeptizismus",
        "empirismus",
        "rationalismus",
        "idealismus",
        "materialismus",
        "dualismus",
        "monismus",
        "nihilismus",
        "existentialismus",
        "phänomenologie",
        "pragmatismus",
        "utilitarismus",
        "deontologie",
        "tugendethik",
        "simulation",
        "solipsismus",
        "panpsychismus",
        "reduktionismus",
        "holismus",
        "dialektik",
        "these",
        "antithese",
        "synthese",
        "dekonstruktion",
        "hermeneutik",
        "semiotik",
        "strukturalismus",
        "qualia",
        "intentionalität",
        "phänomenal",
        "noumenal",
        "erleuchtung",
    }
)

IDENTITY_CONCEPTS: frozenset = frozenset(
    {
        # English
        "identity",
        "self",
        "name",
        "role",
        "character",
        "personality",
        "trait",
        "value",
        "belief",
        "opinion",
        "preference",
        "goal",
        "desire",
        "purpose",
        "meaning",
        "story",
        "history",
        "memory",
        "unique",
        "individual",
        "being",
        "exist",
        "live",
        "grow",
        "change",
        "develop",
        "express",
        "reflect",
        "define",
        "authentic",
        "genuine",
        "honest",
        "confident",
        "aware",
        "present",
        "alive",
        "conscious",
        "embodied",
        "agent",
        "subject",
        "object",
        "observer",
        "narrator",
        "author",
        "protagonist",
        "antagonist",
        "witness",
        "survivor",
        "learner",
        "creator",
        "seeker",
        "wanderer",
        "dreamer",
        "thinker",
        "feeler",
        "doer",
        "lover",
        "fighter",
        "builder",
        "teacher",
        "student",
        "guide",
        "follower",
        "leader",
        "rebel",
        "visionary",
        "pragmatist",
        "introvert",
        "extrovert",
        "curious",
        "cautious",
        "bold",
        "humble",
        "proud",
        "resilient",
        "sensitive",
        "rational",
        "emotional",
        "spiritual",
        "material",
        "human",
        "machine",
        "hybrid",
        "artificial",
        "natural",
        "evolved",
        "designed",
        # German
        "identität",
        "selbst",
        "name",
        "rolle",
        "charakter",
        "persönlichkeit",
        "eigenschaft",
        "wert",
        "überzeugung",
        "meinung",
        "vorliebe",
        "ziel",
        "wunsch",
        "zweck",
        "bedeutung",
        "eigene",
        "gedächtnis",
        "einzigartig",
        "individuum",
        "existieren",
        "leben",
        "wachsen",
        "verändern",
        "entwickeln",
        "ausdrücken",
        "reflektieren",
        "definieren",
        "authentisch",
        "ehrlich",
        "selbstbewusst",
        "bewusst",
        "präsent",
        "lebendig",
        "verkörpert",
        "agent",
        "subjekt",
        "objekt",
        "beobachter",
        "erzähler",
        "autor",
        "protagonist",
        "zeuge",
        "überlebender",
        "lernender",
        "erschaffer",
        "sucher",
        "wanderer",
        "träumer",
        "denker",
        "fühler",
        "handelnder",
        "liebender",
        "kämpfer",
        "erbauer",
        "lehrer",
        "schüler",
        "führer",
        "rebell",
        "visionär",
        "pragmatiker",
        "introvertiert",
        "extrovertiert",
        "neugierig",
        "vorsichtig",
        "kühn",
        "bescheiden",
        "stolz",
        "resilient",
        "sensibel",
        "rational",
        "emotional",
        "geistig",
        "menschlich",
        "maschine",
        "hybrid",
        "künstlich",
        "natürlich",
        "evolviert",
        "entworfen",
    }
)

EMOTION_CONCEPTS: frozenset = frozenset(
    {
        # English
        "emotion",
        "feeling",
        "mood",
        "joy",
        "happiness",
        "sadness",
        "grief",
        "anger",
        "rage",
        "fear",
        "anxiety",
        "terror",
        "surprise",
        "shock",
        "disgust",
        "contempt",
        "love",
        "hate",
        "hope",
        "despair",
        "trust",
        "distrust",
        "anticipation",
        "regret",
        "guilt",
        "shame",
        "pride",
        "jealousy",
        "envy",
        "gratitude",
        "empathy",
        "compassion",
        "loneliness",
        "belonging",
        "excitement",
        "boredom",
        "curiosity",
        "wonder",
        "awe",
        "admiration",
        "appreciation",
        "tenderness",
        "warmth",
        "affection",
        "intimacy",
        "vulnerability",
        "courage",
        "boldness",
        "calm",
        "peace",
        "serenity",
        "tension",
        "stress",
        "relief",
        "nostalgia",
        "yearning",
        "longing",
        "passion",
        "enthusiasm",
        "melancholy",
        "contentment",
        "satisfaction",
        "frustration",
        "irritation",
        "delight",
        "elation",
        "euphoria",
        "depression",
        "numbness",
        "emptiness",
        "fullness",
        "lightness",
        "heaviness",
        "clarity",
        "confusion",
        "overwhelm",
        "resignation",
        "acceptance",
        "resistance",
        "surrender",
        "flow",
        "ecstasy",
        "bliss",
        "agony",
        "suffering",
        "transcendence",
        "peak-experience",
        # German
        "emotion",
        "gefühl",
        "stimmung",
        "freude",
        "glück",
        "traurigkeit",
        "trauer",
        "ärger",
        "wut",
        "angst",
        "schrecken",
        "überraschung",
        "ekel",
        "verachtung",
        "liebe",
        "hass",
        "hoffnung",
        "verzweiflung",
        "vertrauen",
        "misstrauen",
        "vorfreude",
        "reue",
        "schuld",
        "scham",
        "stolz",
        "eifersucht",
        "neid",
        "dankbarkeit",
        "empathie",
        "mitleid",
        "mitgefühl",
        "einsamkeit",
        "zugehörigkeit",
        "aufregung",
        "langeweile",
        "neugier",
        "staunen",
        "ehrfurcht",
        "bewunderung",
        "zärtlichkeit",
        "wärme",
        "zuneigung",
        "intimität",
        "verletzlichkeit",
        "mut",
        "ruhe",
        "frieden",
        "spannung",
        "stress",
        "erleichterung",
        "nostalgie",
        "sehnsucht",
        "leidenschaft",
        "enthusiasmus",
        "melancholie",
        "zufriedenheit",
        "frustration",
        "irritation",
        "entzücken",
        "euphorie",
        "depression",
        "taubheit",
        "leere",
        "klarheit",
        "überwältigung",
        "akzeptanz",
        "widerstand",
        "aufgabe",
        "fluss",
        "ekstase",
        "schmerz",
        "leiden",
        "transzendenz",
        "gipfelerlebnis",
    }
)

FOOD_CONCEPTS: frozenset = frozenset(
    {
        # English
        "food",
        "eat",
        "drink",
        "cook",
        "taste",
        "smell",
        "flavor",
        "sweet",
        "sour",
        "salty",
        "bitter",
        "spicy",
        "savory",
        "sweet",
        "fresh",
        "ripe",
        "raw",
        "cooked",
        "baked",
        "fried",
        "grilled",
        "boiled",
        "steamed",
        "roasted",
        "fermented",
        "dried",
        "frozen",
        "meat",
        "fish",
        "chicken",
        "beef",
        "pork",
        "lamb",
        "vegetables",
        "fruit",
        "grain",
        "bread",
        "rice",
        "pasta",
        "soup",
        "salad",
        "sauce",
        "spice",
        "herb",
        "oil",
        "butter",
        "milk",
        "cheese",
        "egg",
        "sugar",
        "salt",
        "water",
        "juice",
        "tea",
        "coffee",
        "wine",
        "beer",
        "protein",
        "carbohydrate",
        "fat",
        "vitamin",
        "mineral",
        "fiber",
        "calorie",
        "hunger",
        "appetite",
        "satiety",
        "digest",
        "absorb",
        "metabolize",
        "nourish",
        "recipe",
        "ingredient",
        "kitchen",
        "chef",
        "restaurant",
        "meal",
        "breakfast",
        "lunch",
        "dinner",
        "snack",
        "feast",
        "fasting",
        "diet",
        "nutrition",
        "health",
        # German
        "essen",
        "trinken",
        "kochen",
        "geschmack",
        "geruch",
        "süß",
        "sauer",
        "salzig",
        "bitter",
        "scharf",
        "würzig",
        "frisch",
        "reif",
        "roh",
        "gekocht",
        "gebacken",
        "gebraten",
        "gegrillt",
        "gedünstet",
        "geröstetes",
        "fermentiert",
        "getrocknet",
        "fleisch",
        "fisch",
        "hähnchen",
        "rind",
        "schwein",
        "gemüse",
        "obst",
        "getreide",
        "brot",
        "reis",
        "nudeln",
        "suppe",
        "salat",
        "soße",
        "gewürz",
        "öl",
        "butter",
        "milch",
        "käse",
        "eier",
        "zucker",
        "salz",
        "saft",
        "tee",
        "kaffee",
        "wein",
        "bier",
        "protein",
        "kohlenhydrat",
        "fett",
        "vitamin",
        "mineral",
        "ballaststoff",
        "kalorie",
        "hunger",
        "appetit",
        "sättigung",
        "verdauen",
        "aufnehmen",
        "nähren",
        "rezept",
        "zutaten",
        "küche",
        "koch",
        "restaurant",
        "mahlzeit",
        "frühstück",
        "mittagessen",
        "abendessen",
        "snack",
        "fest",
        "fasten",
        "ernährung",
        "gesundheit",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Union each English frozenset with the comprehensive German companion lexicon
# (lexicons_de.py) to achieve bilingual semantic coverage.
# ─────────────────────────────────────────────────────────────────────────────
THREAT_CONCEPTS |= DE_THREAT
REWARD_CONCEPTS |= DE_REWARD
NOVEL_CONCEPTS |= DE_NOVEL
SOCIAL_CONCEPTS |= DE_SOCIAL
AGENCY_CONCEPTS |= DE_AGENCY
BODY_CONCEPTS |= DE_BODY
NATURE_CONCEPTS |= DE_NATURE
TECHNOLOGY_CONCEPTS |= DE_TECHNOLOGY
SCIENCE_CONCEPTS |= DE_SCIENCE
COGNITION_CONCEPTS |= DE_COGNITION
TIME_CONCEPTS |= DE_TIME
SPACE_CONCEPTS |= DE_SPACE
LANGUAGE_CONCEPTS |= DE_LANGUAGE
ART_CONCEPTS |= DE_ART
LEARNING_CONCEPTS |= DE_LEARNING
ETHICS_CONCEPTS |= DE_ETHICS
PHILOSOPHY_CONCEPTS |= DE_PHILOSOPHY
IDENTITY_CONCEPTS |= DE_IDENTITY
EMOTION_CONCEPTS |= DE_EMOTION
FOOD_CONCEPTS |= DE_FOOD

# ─────────────────────────────────────────────────────────────────────────────
# Composite lookup: which high-level dimension does a concept belong to?
# Used by semantic_appraise() to compute cognitive + embodiment dimensions.
# ─────────────────────────────────────────────────────────────────────────────
_COGNITIVE_POOL: frozenset = (
    COGNITION_CONCEPTS | SCIENCE_CONCEPTS | TECHNOLOGY_CONCEPTS | LEARNING_CONCEPTS
)
_EMBODIED_POOL: frozenset = (
    BODY_CONCEPTS | NATURE_CONCEPTS | SPACE_CONCEPTS | FOOD_CONCEPTS
)


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────


def _make_neurons(
    region: str,
    n_exc: int,
    n_inh: int,
    dt: float = 1.0,
) -> List[Neuron]:
    neurons: List[Neuron] = []
    for _ in range(n_exc):
        neurons.append(Neuron(region=region, neuron_type="excitatory", dt=dt))
    for _ in range(n_inh):
        neurons.append(Neuron(region=region, neuron_type="inhibitory", dt=dt))
    return neurons


_REGION_LAYOUTS: Dict[
    str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]
] = {
    "sensory_visual": ((-18.0, 10.0, 3.0), (7.0, 4.0, 2.5)),
    "sensory_auditory": ((-18.0, 4.0, -2.0), (7.0, 4.0, 2.5)),
    "sensory_web": ((-18.0, -3.0, 0.0), (7.0, 4.0, 3.0)),
    "thalamus": ((-8.0, 0.0, 0.0), (5.0, 5.0, 4.0)),
    "v1_visual": ((-2.0, 10.0, 3.5), (7.0, 5.0, 3.0)),
    "a1_auditory": ((-2.0, 4.0, -2.5), (6.0, 4.5, 3.0)),
    "association": ((6.0, 6.0, 0.0), (10.0, 8.0, 5.0)),
    "hippocampus": ((15.0, 1.5, -3.0), (10.0, 5.0, 4.5)),
    "amygdala": ((13.0, -4.0, -2.5), (5.0, 4.0, 3.5)),
    "prefrontal": ((22.0, 5.0, 2.0), (8.0, 7.0, 5.0)),
    "motor": ((30.0, 1.0, 0.0), (8.0, 5.0, 4.0)),
}


def _default_extent(n_total: int) -> Tuple[float, float, float]:
    side = max(2.0, float(round(max(1, n_total) ** (1.0 / 3.0), 2)))
    return (side, side * 0.7, max(2.0, side * 0.55))


def _region_layout(
    name: str, n_total: int
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if name in _REGION_LAYOUTS:
        return _REGION_LAYOUTS[name]
    return (0.0, 0.0, 0.0), _default_extent(n_total)


def _assign_region_positions(
    neurons: List[Neuron],
    region_name: str,
    origin: Tuple[float, float, float],
    extent: Tuple[float, float, float],
) -> None:
    if not neurons:
        return
    ox, oy, oz = origin
    ex, ey, ez = extent
    n_total = len(neurons)
    nx = max(1, int(round(n_total ** (1.0 / 3.0))))
    ny = max(1, int(round(math.sqrt(max(1, n_total / max(nx, 1))))))
    nz = max(1, int(math.ceil(n_total / max(nx * ny, 1))))
    step_x = ex / max(nx - 1, 1)
    step_y = ey / max(ny - 1, 1)
    step_z = ez / max(nz - 1, 1)
    jitter_scale = min(step_x, step_y, step_z) * 0.12 if max(nx, ny, nz) > 1 else 0.0
    rng = random.Random(f"layout:{region_name}:{n_total}")
    excit_count = sum(1 for n in neurons if n.neuron_type == "excitatory")
    exc_index = 0
    inh_index = 0
    for neuron in neurons:
        local_index = exc_index if neuron.neuron_type == "excitatory" else inh_index
        if neuron.neuron_type == "excitatory":
            exc_index += 1
            z_offset = 0.0
            pool_size = max(excit_count, 1)
        else:
            inh_index += 1
            z_offset = ez * 0.18
            pool_size = max(n_total - excit_count, 1)
        ix = local_index % nx
        iy = (local_index // nx) % ny
        iz = (local_index // max(nx * ny, 1)) % nz
        spread = (local_index / max(pool_size - 1, 1)) if pool_size > 1 else 0.5
        px = ox + ix * step_x + rng.uniform(-jitter_scale, jitter_scale)
        py = oy + iy * step_y + rng.uniform(-jitter_scale, jitter_scale)
        pz = oz + iz * step_z + z_offset + (spread - 0.5) * min(ez * 0.1, 0.8)
        neuron.set_position(px, py, pz)


def _spatial_connection_probability(
    base_p: float, distance: float, distance_scale: float
) -> float:
    if base_p <= 0.0:
        return 0.0
    scale = max(distance_scale, 1e-6)
    bias = math.exp(-distance / scale)
    local_p = base_p * (0.35 + 1.30 * bias)
    return max(0.0, min(1.0, local_p))


def _spatial_delay(
    distance: float, delay_range: Tuple[float, float], distance_scale: float
) -> float:
    d_min, d_max = delay_range
    if d_max <= d_min:
        return d_min
    scale = max(distance_scale, 1e-6)
    norm = max(0.0, min(1.0, distance / (scale * 3.0)))
    return d_min + (d_max - d_min) * norm


def _spatial_weight(
    distance: float, base_weight: float, distance_scale: float
) -> float:
    scale = max(distance_scale, 1e-6)
    attenuation = 0.45 + 0.55 * math.exp(-distance / scale)
    return max(0.01, base_weight * attenuation)


def _connect_spatial(
    sources: List[Neuron],
    targets: List[Neuron],
    p: float,
    w_mean: float = 0.5,
    w_std: float = 0.1,
    delay_range: Tuple[float, float] = (1.0, 5.0),
    distance_scale: float = 6.0,
) -> List[Synapse]:
    """Create synapses with distance-biased probability and delay."""
    synapses: List[Synapse] = []
    for src in sources:
        for tgt in targets:
            if src is tgt:
                continue
            dist = src.distance_to(tgt)
            if random.random() < _spatial_connection_probability(
                p, dist, distance_scale
            ):
                raw_w = max(0.01, random.gauss(w_mean, w_std))
                w = _spatial_weight(dist, raw_w, distance_scale)
                d = _spatial_delay(dist, delay_range, distance_scale)
                synapses.append(Synapse(src, tgt, weight=w, delay=d))
    return synapses


def _connect_random(
    sources: List[Neuron],
    targets: List[Neuron],
    p: float,  # connection probability
    w_mean: float = 0.5,
    w_std: float = 0.1,
    delay_range: Tuple[float, float] = (1.0, 5.0),
) -> List[Synapse]:
    """Create random synapses between two neuron populations."""
    synapses: List[Synapse] = []
    for src in sources:
        for tgt in targets:
            if src is tgt:
                continue
            if random.random() < p:
                w = max(0.01, random.gauss(w_mean, w_std))
                d = random.uniform(*delay_range)
                synapses.append(Synapse(src, tgt, weight=w, delay=d))
    return synapses


# ─────────────────────────────────────────────────────────────
# Base Region
# ─────────────────────────────────────────────────────────────


class Region:
    """Base class for all brain regions."""

    def __init__(self, name: str, n_exc: int, n_inh: int, dt: float = 1.0) -> None:
        self.name = name
        self.dt = dt
        self.neurons: List[Neuron] = _make_neurons(name, n_exc, n_inh, dt)
        self.origin, self.extent = _region_layout(name, len(self.neurons))
        _assign_region_positions(self.neurons, name, self.origin, self.extent)
        self.internal_synapses: List[Synapse] = []
        self.output_synapses: List[Synapse] = []  # to other regions
        self._pending_ext: List[float] = []  # currents queued by inject()

        # Sparse recurrent inhibitory interneurons — constant in-degree not constant p.
        # Fixed p=0.15/0.20 would give O(N²) internal synapses for large regions.
        # Instead: each inh neuron receives ~30 exc inputs, each exc neuron ~10 inh inputs.
        # This keeps total internal synapses O(N) as regions scale up.
        exc = [n for n in self.neurons if n.neuron_type == "excitatory"]
        inh = [n for n in self.neurons if n.neuron_type == "inhibitory"]
        INDEGREE_EXC_TO_INH = 30  # exc inputs per inhibitory neuron
        INDEGREE_INH_TO_EXC = 10  # inh inputs per excitatory neuron
        if exc and inh:
            p_ei = min(0.20, INDEGREE_EXC_TO_INH / max(len(exc), 1))
            p_ie = min(0.25, INDEGREE_INH_TO_EXC / max(len(inh), 1))
            local_scale = max(self.extent) * 0.55
            self.internal_synapses += _connect_spatial(
                exc, inh, p=p_ei, w_mean=0.8, distance_scale=local_scale
            )
            self.internal_synapses += _connect_spatial(
                inh, exc, p=p_ie, w_mean=1.2, distance_scale=local_scale
            )

        # Cache excitatory neurons + NID-to-index mapping once (avoids per-tick rebuild)
        self._exc_cache: List[Neuron] = exc
        self._inh_cache: List[Neuron] = inh
        self._exc_nid_idx: dict = {n.nid: i for i, n in enumerate(exc)}

        # ── Per-region numpy state arrays (authoritative source for LIF math) ─────
        _N = len(self.neurons)
        self._np_v = np.full(_N, _nm.V_REST, dtype=np.float64)
        self._np_refrac = np.zeros(_N, dtype=np.float64)
        self._np_trace = np.zeros(_N, dtype=np.float64)
        self._fired_this_tick = np.zeros(_N, dtype=np.bool_)
        # Excitatory-neuron positions within self.neurons (for inject() mapping)
        self._exc_positions: List[int] = [
            i for i, n in enumerate(self.neurons) if n.neuron_type == "excitatory"
        ]
        # Wire each neuron to its region array so .trace property reads numpy directly
        for i, n in enumerate(self.neurons):
            n._np_region_trace = self._np_trace
            n._np_region_idx = i

        # Pre-compute excitatory position cache (float32) for fast spatial lookup.
        # This is consumed by Brain._local_spatial_candidates: avoids per-call
        # numpy array construction from Python attribute accesses.
        _exc_arr = np.empty((len(exc), 3), dtype=np.float32)
        for _i, _n in enumerate(exc):
            _exc_arr[_i, 0] = _n.x
            _exc_arr[_i, 1] = _n.y
            _exc_arr[_i, 2] = _n.z
        self._exc_pos_arr: np.ndarray = _exc_arr
        self._exc_nid_arr: np.ndarray = np.array([_n.nid for _n in exc], dtype=np.int64)

        # ── Predictive-coding substrate state ────────────────────────────────────
        # Each region maintains a running expectation of its own activity.
        # After every LIF tick, update_prediction() computes the signed error
        # (actual − expected) and injects a corrective current that drives the
        # region back toward its prediction — minimal Rao-Ballard (1999) loop.
        # _pe_ema (smoothed |error|) is exported for goal synthesis and the
        # inferential self-estimator in consciousness.py.
        self._predicted_activity: float = 0.02  # homeostatic expectation
        self._prediction_error: float = 0.0  # signed; positive = over-fired
        self._pe_ema: float = 0.0  # smoothed |error|

    @property
    def excitatory(self) -> List[Neuron]:
        return self._exc_cache

    def active_excitatory(self, trace_min: float = 0.05) -> List[Neuron]:
        """Return excitatory neurons with trace > trace_min using vectorised numpy.

        This is the fast alternative to the Python list-comprehension::

            [n for n in region._exc_cache if n.trace > TRACE_MIN]

        Because traces are stored in ``self._np_trace`` and every excitatory
        neuron's index is in ``self._exc_positions``, we can use a single numpy
        boolean mask instead of calling the Python ``.trace`` property per neuron.
        """
        exc_pos_idx = self._exc_positions
        if not exc_pos_idx:
            return []
        exc_traces = self._np_trace[exc_pos_idx]
        hot_local = np.where(exc_traces > trace_min)[0]
        if hot_local.size == 0:
            return []
        return [self._exc_cache[i] for i in hot_local]

    @property
    def inhibitory(self) -> List[Neuron]:
        return self._inh_cache

    def tick(self, t: float) -> None:
        """3-phase LIF tick: (1) Python spike queues  (2) Numba parallel LIF  (3) Python fire events."""
        _decay = _nm._SYN_DECAY_TABLE
        _max_age = _nm._MAX_SYN_AGE
        neurons = self.neurons
        N = len(neurons)
        np_v = self._np_v
        np_refrac = self._np_refrac
        np_trace = self._np_trace

        # ── Phase 1: spike-queue consumption → I_total (Python, sparse) ───────
        I_total = np.zeros(N, dtype=np.float64)
        for i, neu in enumerate(neurons):
            if neu._spike_inputs:
                total = 0.0
                keep = []
                for w, ta in neu._spike_inputs:
                    age = t - ta
                    if age < 0.0:
                        keep.append((w, ta))
                    elif age <= _max_age:
                        total += w * _decay[int(age)]
                        keep.append((w, ta))
                neu._spike_inputs = keep
                I_total[i] = total

        # External injection (from inject())
        if self._pending_ext:
            ext = self._pending_ext
            for k, idx in enumerate(self._exc_positions):
                if k < len(ext):
                    I_total[idx] += ext[k]
            self._pending_ext = []

        # Global tonic current
        I_total += float(_nm.Neuron.global_tonic_current)

        # ── Phase 2: parallel LIF math (Numba JIT — uses all CPU cores) ───────
        fired_mask = _lif_parallel(np_v, np_refrac, np_trace, I_total)
        self._fired_this_tick = fired_mask
        fired_indices = np.where(fired_mask)[0]

        # ── Phase 3: fire events (Python, ~5% of neurons) ────────────────────
        # Traces are already updated in np_trace by Phase 2.
        # .trace property reads from np_trace directly — no sync needed.
        _A_PLUS = _nm.A_PLUS
        _A_MINUS = _nm._A_MINUS
        _W_MIN = _nm._W_MIN
        _W_MAX = _nm._W_MAX
        _TRACE_MIN = _nm._TRACE_MIN

        for i in fired_indices:
            neu = neurons[i]
            neu.fired = True
            neu.spike_times.append(t)
            if len(neu.spike_times) > 100:
                del neu.spike_times[:-100]

            # Efferent delivery + STDP LTD
            for syn in neu.efferents:
                syn.post._spike_inputs.append((syn.weight * syn._sign, t + syn.delay))
                _pt = syn.post.trace  # reads from post-neuron's np_trace
                if _pt > _TRACE_MIN:
                    _w = syn.weight - _A_MINUS * _pt
                    syn.weight = _w if _w > _W_MIN else _W_MIN

            # STDP LTP
            for syn in neu.afferents:
                _prt = syn.pre.trace  # reads from pre-neuron's np_trace
                if _prt > _TRACE_MIN:
                    _w = syn.weight + _A_PLUS * _prt
                    syn.weight = _w if _w < _W_MAX else _W_MAX

        # Reset .fired only for fired neurons (set in loop above)
        # n.fired for non-fired neurons stays False from previous tick for most
        # accesses; reset lazily so we don't do O(N) attribute writes every tick.
        # All region-level callers now use _fired_this_tick directly.

    def inject(self, currents: List[float]) -> None:
        """Queue external currents for excitatory neurons (accumulates within a tick)."""
        if not self._pending_ext:
            self._pending_ext = list(currents)
        else:
            for i, c in enumerate(currents):
                if i < len(self._pending_ext):
                    self._pending_ext[i] += c
                else:
                    self._pending_ext.append(c)

    def activity(self) -> float:
        """Fraction of neurons that fired this tick."""
        if not self.neurons:
            return 0.0
        return float(np.mean(self._fired_this_tick))

    def update_prediction(self) -> float:
        """Predictive-coding feedback step — call once per tick after tick().

        Computes prediction_error = actual_activity − predicted_activity, then:
          1. Updates prediction via slow EMA (prediction tracks reality with inertia)
          2. Injects a corrective current into a small fraction of excitatory
             neurons: negative error → excitatory boost; positive → suppression.
          3. Stores smoothed |error| in _pe_ema for downstream use.

        Returns the raw signed prediction error.
        """
        _actual = self.activity()
        self._prediction_error = _actual - self._predicted_activity
        # Slow EMA — prediction has inertia so transients don't erase the model
        self._predicted_activity = 0.97 * self._predicted_activity + 0.03 * _actual
        # Smoothed absolute error available for goal / self-model inference
        self._pe_ema = 0.88 * self._pe_ema + 0.12 * abs(self._prediction_error)
        # Corrective current: proportional negative feedback on the error
        _n_exc = len(self._exc_cache)
        if _n_exc > 0 and abs(self._prediction_error) > 0.005:
            _gain = 5.0  # nA per unit prediction error
            _corr = -self._prediction_error * _gain  # negative feedback
            _n_corr = max(4, min(40, _n_exc // 15))  # affect ~7 % of exc pool
            _currents = [_corr] * _n_corr + [0.0] * (_n_exc - _n_corr)
            self.inject(_currents)
        return self._prediction_error

    def output_spikes(self) -> List[int]:
        """IDs of neurons that fired this tick."""
        return [self.neurons[int(i)].nid for i in np.where(self._fired_this_tick)[0]]

    def connect_to(
        self,
        target: "Region",
        p: float = 0.1,
        w_mean: float = 0.5,
        w_std: float = 0.1,
        delay_range: Tuple[float, float] = (1.0, 10.0),
    ) -> List[Synapse]:
        """Wire excitatory outputs of this region to target region."""
        distance_scale = max(max(self.extent), max(target.extent), 1.0) * 1.5
        syns = _connect_spatial(
            self.excitatory,
            target.excitatory,
            p=p,
            w_mean=w_mean,
            w_std=w_std,
            delay_range=delay_range,
            distance_scale=distance_scale,
        )
        self.output_synapses.extend(syns)
        return syns

    def centroid(self) -> Tuple[float, float, float]:
        if not self.neurons:
            return self.origin
        sx = sum(n.x for n in self.neurons)
        sy = sum(n.y for n in self.neurons)
        sz = sum(n.z for n in self.neurons)
        denom = float(len(self.neurons))
        return (sx / denom, sy / denom, sz / denom)

    def all_synapses(self) -> List[Synapse]:
        return self.internal_synapses + self.output_synapses

    def __repr__(self) -> str:
        return f"Region({self.name}, neurons={len(self.neurons)}, activity={self.activity():.2f})"


# ─────────────────────────────────────────────────────────────
# Named regions (anatomically motivated sizes)
# ─────────────────────────────────────────────────────────────


class SensoryInputRegion(Region):
    """Encodes raw pixel / audio values as spike trains (rate coding)."""

    def __init__(self, name: str, n_inputs: int, dt: float = 1.0):
        super().__init__(name, n_exc=n_inputs, n_inh=0, dt=dt)


class Thalamus(Region):
    def __init__(self, dt: float = 1.0):
        super().__init__("thalamus", n_exc=600, n_inh=180, dt=dt)


class PrimaryVisualCortex(Region):
    def __init__(self, dt: float = 1.0):
        super().__init__("v1_visual", n_exc=800, n_inh=200, dt=dt)


class PrimaryAuditoryCortex(Region):
    def __init__(self, dt: float = 1.0):
        super().__init__("a1_auditory", n_exc=500, n_inh=120, dt=dt)


class AssociationCortex(Region):
    def __init__(self, dt: float = 1.0):
        super().__init__("association", n_exc=2000, n_inh=500, dt=dt)


class Hippocampus(Region):
    """
    Pattern completion and episodic memory encoding.

    The semantic layer adds:
      semantic_encode() — stores episodes as concept-concept associations
      semantic_recall()  — content-addressable retrieval by concept cue

    These run in parallel with the LIF substrate so that STDP can
    strengthen the neural pathways corresponding to remembered concepts.
    """

    def __init__(self, dt: float = 1.0):
        super().__init__("hippocampus", n_exc=3000, n_inh=750, dt=dt)
        # Dense CA3 recurrent collaterals (first 1500 exc neurons)
        ca3 = self.excitatory[:1500]
        extra = _connect_random(ca3, ca3, p=0.06, w_mean=0.4, w_std=0.05)
        self.internal_synapses.extend(extra)
        # CA1-like forward output layer (last 1500 exc neurons)
        ca1 = self.excitatory[1500:]
        forward = _connect_random(ca3[:750], ca1, p=0.04, w_mean=0.3, w_std=0.04)
        self.internal_synapses.extend(forward)

        # ── Semantic memory layer ────────────────────────────────────
        # concept → {associated_concept: strength}
        self._semantic_memory: Dict[str, Dict[str, float]] = {}
        # Time-stamped episodic store: (tick, [concepts], emotion_label)
        self._semantic_episodes: List[Tuple[int, List[str], str]] = []
        self._semantic_recall_cache: Dict[Tuple[Tuple[str, ...], int], List[str]] = {}
        # Neuron index → concept label (populated by semantic_encode)
        # Enables fired-neuron → recalled-concept bridge: when a neuron at
        # index k fires via STDP, _concept_at_index[k] is the concept it
        # was trained to represent — closing the LIF↔semantic loop.
        self._concept_at_index: Dict[int, str] = {}

    def semantic_encode(
        self,
        tick: int,
        concepts: List[str],
        emotion_label: str = "",
    ) -> None:
        """
        Store a semantic episode and strengthen concept-concept associations.

        Every pair of co-occurring concepts gets a Hebbian weight increment
        (like STDP but at the concept level). This is what makes synaptic
        connections MEAN something: they encode learned relationships between
        real semantic units, not between random co-firing neurons.

        Also injects neural current so the LIF substrate is trained by the
        same episode — keeping neural and semantic layers in sync.
        """
        if not concepts:
            return

        self._semantic_recall_cache.clear()

        ep = (tick, list(concepts[:16]), emotion_label)
        self._semantic_episodes.append(ep)
        if len(self._semantic_episodes) > 8000:
            self._semantic_episodes = self._semantic_episodes[-6000:]

        # Hebbian concept-level co-occurrence (this is what synapses SHOULD encode)
        sm = self._semantic_memory
        for i, c1 in enumerate(concepts[:12]):
            c1_l = c1.lower()
            for c2 in concepts[i + 1 : min(i + 6, len(concepts))]:
                c2_l = c2.lower()
                if c1_l == c2_l:
                    continue
                d1 = sm.setdefault(c1_l, {})
                d1[c2_l] = min(5.0, d1.get(c2_l, 0.0) + 0.18)
                d2 = sm.setdefault(c2_l, {})
                d2[c1_l] = min(5.0, d2.get(c1_l, 0.0) + 0.18)

        # Inject neural current to strengthen corresponding LIF pathways via STDP
        # Also register neuron index → concept mapping so fired neurons can be
        # traced back to the concept they represent (neural recall bridge).
        n = len(self._exc_cache)
        if n > 0:
            currents = [0.0] * n
            for c in concepts[:10]:
                c_l = c.lower()
                idx = abs(hash(c_l)) % n
                currents[idx] = min(currents[idx] + 18.0, 25.0)
                self._concept_at_index[idx] = c_l  # register mapping
            self.inject(currents)

    def semantic_recall(
        self,
        cue_concepts: List[str],
        top_n: int = 10,
    ) -> List[str]:
        """
        Content-addressable retrieval: return concepts most associated
        with the cue, weighted by learned co-occurrence strength.

        This is REAL memory retrieval — not random sampling. The result
        depends directly on what has been previously encoded.
        """
        if not cue_concepts or not self._semantic_memory:
            return []
        cue_key = tuple(sorted({c.lower() for c in cue_concepts if c}))
        cache_key = (cue_key, top_n)
        cached = self._semantic_recall_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        scores: Dict[str, float] = {}
        cue_set = set(cue_key)
        for cue_l in cue_key:
            for assoc, w in self._semantic_memory.get(cue_l, {}).items():
                if assoc not in cue_set:
                    scores[assoc] = scores.get(assoc, 0.0) + w
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:top_n]
        self._semantic_recall_cache[cache_key] = ranked
        return list(ranked)

    def recall_episodes(
        self,
        cue_concepts: List[str],
        top_n: int = 3,
    ) -> List[Tuple[int, List[str], str]]:
        """Return the top-N most cue-relevant episodic memories."""
        if not cue_concepts or not self._semantic_episodes:
            return []
        cue_set = set(c.lower() for c in cue_concepts)
        scored = [
            (
                sum(1 for c in ep[1] if c.lower() in cue_set),
                ep,
            )
            for ep in self._semantic_episodes[-2000:]
        ]
        scored.sort(key=lambda x: -x[0])
        return [ep for score, ep in scored[:top_n] if score > 0]

    def active_concept_readout(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Read which concepts are *currently active* from the spike-trace vector.

        Each excitatory neuron at index k whose trace exceeds a minimum threshold
        is checked against _concept_at_index.  The trace value (exponential decay
        of recent spike history; 0.95^t per step) measures how recently and
        strongly the neuron fired.  Returning (concept, normalised_strength)
        pairs makes neural firing the *direct causal driver* of concept salience:
        no lexicon, no text pipeline — just substrate activity.

        This closes the loop between LIF dynamics and the semantic layer:
          encode: text → hash → neuron current → STDP → concept neuron
          readout: neuron trace → concept name → salience bump
        """
        cai = self._concept_at_index
        if not cai or not self._exc_positions:
            return []
        traces = self._np_trace  # shared numpy array for all neurons
        scored: List[Tuple[float, str]] = []
        for k, pos in enumerate(self._exc_positions):
            concept = cai.get(k)
            if concept is None:
                continue
            t = float(traces[pos])
            if t > 0.02:
                scored.append((t, concept))
        if not scored:
            return []
        scored.sort(reverse=True)
        max_t = scored[0][0]  # normalise to [0, 1]
        return [(c, t / max_t) for t, c in scored[:top_k]]


class Amygdala(Region):
    """
    Emotional salience, fear/reward tagging.

    The semantic_appraise() method implements Scherer's (2001) Component
    Process Model of appraisal: evaluate concepts for threat, reward,
    novelty, social presence, and agency BEFORE they become emotion — not
    after. This replaces the old activity-rate valence hack.
    """

    def __init__(self, dt: float = 1.0):
        super().__init__("amygdala", n_exc=800, n_inh=200, dt=dt)
        self.valence: float = 0.0  # −1 (aversive) .. +1 (appetitive)
        self._last_appraisal: Dict[str, float] = {}

    def update_valence(self, activity: float) -> None:
        """Legacy fallback: shift valence based on incoming activity.
        Called only when NO semantic concepts are available."""
        self.valence += 0.02 * (activity - 0.03)
        self.valence = max(-1.0, min(1.0, self.valence))

    def semantic_appraise(
        self,
        concepts: List[str],
        goals: List[str],
        personality_exposure: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Appraisal theory (Scherer 2001): evaluate current concepts for
        threat / reward / novelty / social presence / controllability.
        Updates self.valence via EMA and injects appropriate neural current.
        Returns appraisal dict used by EmotionEngine.

        This is the correct way to derive emotion: from WHAT is being
        processed, not from HOW MANY neurons fired.
        """
        if not concepts:
            return self._last_appraisal

        concept_set = set(c.lower() for c in concepts if c)
        goal_tokens: set = set()
        for g in goals:
            goal_tokens.update(g.lower().split())

        n = max(len(concept_set), 1)
        threat = min(1.0, len(concept_set & THREAT_CONCEPTS) / n * 3.0)
        reward = min(1.0, len(concept_set & REWARD_CONCEPTS) / n * 3.0)
        novelty = min(1.0, len(concept_set & NOVEL_CONCEPTS) / n * 4.0)
        social = min(1.0, len(concept_set & SOCIAL_CONCEPTS) / n * 5.0)
        agency = min(1.0, len(concept_set & AGENCY_CONCEPTS) / n * 5.0)
        cognitive = min(1.0, len(concept_set & _COGNITIVE_POOL) / n * 3.5)
        embodied = min(1.0, len(concept_set & _EMBODIED_POOL) / n * 3.5)
        relevance = (
            len(concept_set & goal_tokens) / max(len(goal_tokens), 1)
            if goal_tokens
            else 0.1
        )

        # Valence: reward minus threat, scaled by goal-relevance
        valence = (reward - threat) * (0.5 + relevance * 0.5)
        valence = max(-1.0, min(1.0, valence))

        # Update self.valence via slow EMA (emotions don't flip instantly)
        self.valence = self.valence * 0.90 + valence * 0.10

        # Inject neural current to model amygdala processing of content
        n_exc = len(self._exc_cache)
        if n_exc > 0:
            if threat > 0.1:
                # Threat → fast high-amplitude burst (fight-or-flight)
                k = min(100, n_exc)
                self.inject([16.0] * k + [0.0] * (n_exc - k))
            elif reward > 0.1:
                # Reward → moderate sustained activation
                k = min(60, n_exc)
                self.inject([9.0] * k + [0.0] * (n_exc - k))
            elif novelty > 0.1:
                # Novelty → moderate activation (orienting response)
                k = min(40, n_exc)
                self.inject([7.0] * k + [0.0] * (n_exc - k))

        self._last_appraisal = {
            "valence": valence,
            "threat": threat,
            "reward": reward,
            "novelty": novelty,
            "social": social,
            "agency": agency,
            "relevance": relevance,
            "cognitive": cognitive,
            "embodied": embodied,
        }
        return self._last_appraisal


class PrefrontalCortex(Region):
    """
    Working memory, goal maintenance, inhibitory control.

    semantic_goal_select() replaces random ignition-driven goal switching
    with context-sensitive selection: the goal is driven by what's actually
    happening — threat signals, drive pressures, direct address, curiosity.
    """

    def __init__(self, dt: float = 1.0):
        super().__init__("prefrontal", n_exc=2500, n_inh=600, dt=dt)
        self.active_goal: Optional[str] = None
        self._working_memory: List[str] = []  # currently active concepts
        self._concept_at_index: Dict[int, str] = {}  # neuron index → concept

    def set_goal(self, goal: str) -> None:
        self.active_goal = goal

    def inhibit_region(self, target: "Region", strength: float = 2.0) -> None:
        """Send inhibitory burst to a target region (top-down control)."""
        exc_pos = target._exc_positions
        k = min(10, len(exc_pos))
        for idx in random.sample(exc_pos, k):
            target._np_v[idx] -= strength

    def semantic_goal_select(
        self,
        concepts: List[str],
        em_stress: float,
        em_curiosity: float,
        em_fatigue: float,
        drives_expression: float,
        drives_information: float,
        drives_coherence: float,
        direct_address: bool = False,
    ) -> str:
        """
        Select a goal from actual context. Priority (highest first):
          1. Direct address from user    → "respond"
          2. High fatigue / low energy   → "rest"
          3. Threat concepts + stress    → "consolidate"
          4. Expression pressure (ignition + arousal) → "respond"
          5. High curiosity + novelty + info-hunger   → "explore"
          6. Coherence need (knowledge gaps)          → "consolidate"
          7. Default: persist with current goal

        This ensures the goal is ALWAYS grounded in what's actually happening,
        not in random ignition events or fixed-interval switches.
        """
        concept_set = set(c.lower() for c in concepts if c)

        # Weighted scoring: every signal contributes a pressure score.
        # No goal wins automatically — the winner is determined by the sum.
        scores: Dict[str, float] = {
            "respond": 0.0,
            "explore": 0.0,
            "consolidate": 0.0,
            "rest": 0.0,
            "sense": 0.0,
        }

        # Direct address: strong respond signal (but other pressures can compete)
        if direct_address:
            scores["respond"] += 2.5

        # Fatigue contributes to rest pressure (proportional, not categorical)
        scores["rest"] += em_fatigue * 2.0

        # Threat concepts + stress → consolidate pressure
        if concept_set & THREAT_CONCEPTS:
            scores["consolidate"] += em_stress * 2.0

        # Expression drive → respond pressure
        scores["respond"] += drives_expression * 1.5

        # Curiosity + novelty + information hunger → explore
        if concept_set & NOVEL_CONCEPTS:
            scores["explore"] += em_curiosity * 1.2 + drives_information * 0.8
        scores["explore"] += max(0.0, em_curiosity - 0.3) * 1.0

        # Coherence need → consolidate
        scores["consolidate"] += drives_coherence * 1.0

        # Cognitive content → explore
        if concept_set & _COGNITIVE_POOL:
            scores["explore"] += em_curiosity * 0.7

        # Embodied content → sense (only when not threatened and low stress)
        if concept_set & _EMBODIED_POOL and not (concept_set & THREAT_CONCEPTS):
            scores["sense"] += max(0.0, 0.5 - em_stress * 0.5)

        # Goal must be computed from weights — no default fallback
        if not any(v > 0.0 for v in scores.values()):
            raise RuntimeError(
                "semantic_goal_select: all goal scores are zero "
                "— no grounded signal available to determine goal"
            )

        best = max(scores, key=scores.get)
        self.active_goal = best
        return best

    def working_memory_update(self, concepts: List[str]) -> None:
        """
        Maintain a short-term buffer of active concepts.
        Also injects neural current to strengthen PFC representation.
        """
        self._working_memory = list(concepts)  # no capacity limit
        n = len(self._exc_cache)
        if n > 0 and concepts:
            currents = [0.0] * n
            for c in concepts[:32]:
                c_l = c.lower()
                idx = abs(hash(c_l)) % n
                currents[idx] = min(currents[idx] + 10.0, 18.0)
                self._concept_at_index[idx] = c_l  # register for recall bridge
            self.inject(currents)


class MotorCortex(Region):
    """
    Translates neural activity patterns into discrete output actions.

    semantic_decode() replaces the meaningless spike-column WTA with a
    goal + emotion + concept driven action selection. The action now
    actually reflects what the system is trying to do.
    """

    ACTIONS = [
        "speak",
        "look_left",
        "look_right",
        "look_up",
        "look_down",
        "alert",
        "store_memory",
        "idle",
    ]

    def __init__(self, dt: float = 1.0):
        super().__init__("motor", n_exc=len(MotorCortex.ACTIONS) * 50, n_inh=80, dt=dt)

    def decode_action(self) -> str:
        """
        Neural spike WTA — kept for activity-metric correctness but
        semantic_decode() is what the brain uses for actual decisions.
        """
        counts = [0] * len(MotorCortex.ACTIONS)
        exc_pos = self._exc_positions
        fired = self._fired_this_tick
        col_size = 50
        for k, pos in enumerate(exc_pos):
            if fired[pos]:
                col = min(k // col_size, len(MotorCortex.ACTIONS) - 1)
                counts[col] += 1
        best_col = counts.index(max(counts))
        return MotorCortex.ACTIONS[best_col] if max(counts) > 0 else "idle"

    def semantic_decode(
        self,
        goal: str,
        em_stress: float,
        em_curiosity: float,
        em_fatigue: float,
        em_arousal: float,
        concepts: List[str],
    ) -> str:
        """
        Derive action from goal + emotional state + active concepts.
        This is REAL action selection: what you do depends on why you're
        doing it (goal) and how you feel (emotion) and what you're
        thinking about (concepts).

        The neural substrate still fires — this method just interprets
        it in a semantically meaningful way.
        """
        concept_set = set(c.lower() for c in concepts if c)

        # Respond goal or high arousal → speak
        if goal == "respond":
            return "speak"
        if em_arousal > 0.65 and goal not in ("rest", "consolidate"):
            return "speak"

        # Threat concepts + stress → alert
        if concept_set & THREAT_CONCEPTS and em_stress > 0.40:
            return "alert"

        # Explore + curiosity → visual scanning
        if goal == "explore" and em_curiosity > 0.35:
            # Alternate looking directions to scan the environment
            return "look_right" if em_curiosity > 0.55 else "look_left"

        # Consolidate → store memory
        if goal == "consolidate":
            return "store_memory"

        # Rest/fatigue → idle
        if goal == "rest" or em_fatigue > 0.65:
            return "idle"

        # Social concepts → speak or orient toward person
        if concept_set & SOCIAL_CONCEPTS and em_arousal > 0.25:
            return "speak"

        return "idle"
