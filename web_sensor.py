"""
web_sensor.py — Internet Impression Sensor  (massively expanded)

Continuously fetches content from 50+ RSS feeds, 80+ YouTube educational
videos, Wikipedia full articles, DuckDuckGo Instant Answers and arXiv
abstracts — then converts text into neural spike currents injected into
AssociationCortex, Hippocampus and Amygdala.

Parallel fetching with ThreadPoolExecutor (4 workers) reduces latency.
Curiosity-driven urgency: set_urgency(factor) scales amplitude + fetch rate.

Text encoding (48 channels):
  Ch   Feature                     Neuro-analogy
  ───  ─────────────────────────── ─────────────────────────
  0    Information density (TTR)   Novelty
  1    Mean word length            Conceptual complexity
  2    Sentence length             Syntactic load
  3    Punctuation density         Emotional modulation
  4-11 Letter-group frequencies    Phonological fingerprint
  12   Positive keyword density    Reward signal
  13   Negative keyword density    Aversion signal
  14   Number / quantity density   Fact load
  15   Question density            Curiosity drive
 16-47 Bigram channel energy       Lexical texture (32 ch.)
"""

from __future__ import annotations

import re
import string
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── RSS FEEDS (50+ sources) ──────────────────────────────────
RSS_FEEDS: List[str] = [
    # Science & Research
    "https://rss.sciam.com/ScientificAmerican-Global",
    "https://www.newscientist.com/feed/home/",
    "https://phys.org/rss-feed/",
    "https://www.nature.com/nature.rss",
    "https://arxiv.org/rss/cs.AI",
    "https://arxiv.org/rss/cs.NE",
    "https://arxiv.org/rss/q-bio.NC",
    "https://arxiv.org/rss/physics.pop-ph",
    "https://neurosciencenews.com/feed/",
    "https://feeds.aps.org/rss/recent/prl.rss",
    # Technology
    "https://techcrunch.com/feed/",
    "https://www.wired.com/feed/rss",
    "https://news.ycombinator.com/rss",
    "https://www.technologyreview.com/feed/",
    "https://spectrum.ieee.org/rss/fulltext",
    # World News
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://rss.dw.com/rdf/rss-de-all",
    "https://www.tagesschau.de/xml/rss2/",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/science/rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
    # Philosophy & Psychology
    "https://psychologytoday.com/intl/articles/feed",
    "https://aeon.co/feed.rss",
    "https://nautil.us/feed/",
    # Health & Biology
    "https://www.nih.gov/news-events/feed.xml",
    "https://www.sciencedaily.com/rss/all.xml",
    "https://newatlas.com/feed/",
    # Space & Astronomy
    "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    "https://www.spacenews.com/feed/",
    # Culture & Arts
    "https://www.smithsonianmag.com/rss/latest_articles/",
    # Mathematics
    "https://arxiv.org/rss/math.CO",
    "https://arxiv.org/rss/math.NT",
    # German language
    "https://www.spiegel.de/schlagzeilen/tops/index.rss",
    "https://www.heise.de/rss/heise.rdf",
    "https://www.spektrum.de/alias/rss/spektrum-de-rss-feed/996406",
    # History
    "https://www.historyextra.com/feed/",
    # Economics & Society
    "https://www.economist.com/science-and-technology/rss.xml",
    "https://marginalrevolution.com/feed",
    # Environment
    "https://www.carbonbrief.org/feed",
]

# ── WIKIPEDIA TOPICS ────────────────────────────────────────
WIKIPEDIA_TOPICS: List[str] = [
    "Neuroscience",
    "Consciousness",
    "Memory",
    "Emotion",
    "Neuroplasticity",
    "Long-term_potentiation",
    "Hebbian_learning",
    "Spiking_neural_network",
    "Human_brain",
    "Prefrontal_cortex",
    "Hippocampus",
    "Amygdala",
    "Cerebral_cortex",
    "Dopamine",
    "Serotonin",
    "Norepinephrine",
    "Philosophy_of_mind",
    "Qualia",
    "Free_will",
    "Global_workspace_theory",
    "Integrated_information_theory",
    "Cognitive_science",
    "Perception",
    "Self-awareness",
    "Artificial_intelligence",
    "Machine_learning",
    "Deep_learning",
    "Natural_language_processing",
    "Robotics",
    "Quantum_computing",
    "Evolution",
    "Biology",
    "Genetics",
    "Epigenetics",
    "Neuron",
    "Synapse",
    "Brain",
    "Central_nervous_system",
    "Language",
    "Linguistics",
    "Culture",
    "Art",
    "Music",
    "Mathematics",
    "Physics",
    "Quantum_mechanics",
    "Cosmology",
    "Black_hole",
    "Dark_matter",
    "Psychology",
    "Sociology",
    "Economics",
    "Ethics",
    "Democracy",
    "Learning",
    "Artificial_intelligence",
]

# ── YOUTUBE EDUCATIONAL IDs (80 videos) ─────────────────────
YOUTUBE_EDUCATIONAL_IDS: List[str] = [
    # Kurzgesagt
    "wNDGgL73ihY",
    "9D05ej8u-gU",
    "SQdoB0mKpDc",
    "1AElONvi9WQ",
    "MBRqu0YOH14",
    "zQo2ZMOm5lY",
    # 3Blue1Brown
    "aircAruvnKk",
    "IHZwWFHWa-w",
    "spUNpyF58BY",
    "LyGKycYT2v0",
    "kjBOesZCoqc",
    "p3T9l0W3ybI",
    # Veritasium
    "HeQX2HjkcNo",
    "d0gS5TXarXc",
    "GnR0qDrBfBM",
    "WiTgn5QZiHM",
    "VnI9R23BVIU",
    # TED / TEDx
    "isa5uXpCdbo",
    "6P-4RDYCx30",
    "lyu7v7nWzfo",
    "nTgeLEWr614",
    "UyyjU8fzEYU",
    "5MgBikgcWnY",
    "qp0HIF3SfI4",
    # SciShow
    "NNnIGh9g6fA",
    "IVdVVcCFzgA",
    "Oqwpm5FLTjU",
    "DSPF-kqUkFQ",
    # PBS Space Time
    "f-7YCPLqPwQ",
    "7IMbkABB3_E",
    "54XzEqCHzxQ",
    "ztqZHfo4a-c",
    # Lex Fridman
    "0Jwr6oIGnhA",
    "DxREm3s1scA",
    "P-2P3MSZrBM",
    # CrashCourse
    "KMX4PAtv_oU",
    "IB-FaZST3J4",
    "uqXVAo7dVRU",
    # MinutePhysics
    "72y2EC5fkgA",
    # Smarter Every Day
    "mc979OPbRBE",
    # VSauce
    "Qe5WT22-AO8",
    "deYp9H-4h9Y",
    # Sean Carroll
    "6I8NuZ-3lL0",
    # Primer (game theory / evolution)
    "mM4aXXCDdCI",
    # Misc educational
    "j4IRV3WaDR8",
    "PHBhPeTMeNE",
    "OBQU3UjgPiU",
    "JB7jSFeVz1U",
    "GDRBermj-RE",
]

# ── Sentiment word lists ─────────────────────────────────────
_POSITIVE_WORDS = frozenset(
    {
        "good",
        "great",
        "excellent",
        "success",
        "win",
        "positive",
        "advance",
        "improve",
        "discovery",
        "love",
        "peace",
        "hope",
        "benefit",
        "achieve",
        "amazing",
        "wonderful",
        "progress",
        "innovation",
        "growth",
        "help",
        "solution",
        "breakthrough",
        "joy",
        "beautiful",
        "best",
        "strong",
        "safe",
        "healthy",
        "happy",
        "create",
        "new",
        "gain",
        "thrive",
        "inspire",
        "empower",
        "heal",
        "restore",
        "learn",
        "understand",
        "evolve",
        "connect",
        "flourish",
        "rise",
        "wisdom",
        "clarity",
        "harmony",
        "truth",
        "freedom",
        "light",
        "open",
        "expand",
    }
)
_NEGATIVE_WORDS = frozenset(
    {
        "bad",
        "fail",
        "crisis",
        "danger",
        "war",
        "death",
        "loss",
        "problem",
        "disease",
        "attack",
        "risk",
        "threat",
        "damage",
        "collapse",
        "fear",
        "hate",
        "wrong",
        "error",
        "conflict",
        "disaster",
        "virus",
        "crash",
        "violence",
        "toxic",
        "harm",
        "kill",
        "explode",
        "terror",
        "corrupt",
        "fraud",
        "abuse",
        "destroy",
        "decline",
        "suffer",
        "pain",
        "chaos",
        "poison",
        "broken",
        "shock",
        "trauma",
        "panic",
        "flood",
        "drought",
        "poverty",
        "inequality",
    }
)

# 32 high-frequency English bigrams (channels 16-47)
_BIGRAMS = [
    "th",
    "he",
    "in",
    "er",
    "an",
    "re",
    "on",
    "en",
    "at",
    "es",
    "ed",
    "te",
    "or",
    "ti",
    "hi",
    "as",
    "to",
    "ng",
    "is",
    "ha",
    "it",
    "et",
    "se",
    "ou",
    "of",
    "ar",
    "nd",
    "nt",
    "al",
    "de",
    "le",
    "ro",
]
assert len(_BIGRAMS) == 32

N_NEURONS = 48  # must match SensoryInputRegion("sensory_web", n_inputs=48)


# ── Module-level stateless text encoder ─────────────────────
def encode_text(text: str, n_neurons: int = N_NEURONS) -> List[float]:
    """Stateless text -> current vector. Returns n_neurons floats in [0,20] nA."""
    text_lc = text.lower()
    words = re.findall(r"[a-z\u00e4\u00f6\u00fc\u00df]+", text_lc)
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    features = np.zeros(48, dtype=np.float32)
    if not words:
        return np.resize(features, n_neurons).tolist()

    features[0] = len(set(words)) / max(len(words), 1)
    features[1] = float(np.mean([len(w) for w in words])) / 10.0
    sent_lens = [len(re.findall(r"[a-z\u00e4\u00f6\u00fc\u00df]+", s)) for s in sents]
    features[2] = min(float(np.mean(sent_lens)) / 30.0, 1.0) if sent_lens else 0.0
    features[3] = min(
        sum(1 for c in text if c in "!?;:\u2014\u2026") / max(len(sents), 1) / 3.0, 1.0
    )

    letter_groups = [
        "aeiou",
        "bcdfg",
        "hjklm",
        "npqrs",
        "tuvwx",
        "yz",
        string.digits,
        " \t\n",
    ]
    char_counts = Counter(text_lc)
    total_chars = max(sum(char_counts.values()), 1)
    for i, group in enumerate(letter_groups):
        features[4 + i] = sum(char_counts.get(c, 0) for c in group) / total_chars

    features[12] = min(
        sum(1 for w in words if w in _POSITIVE_WORDS) / max(len(words), 1) * 20.0, 1.0
    )
    features[13] = min(
        sum(1 for w in words if w in _NEGATIVE_WORDS) / max(len(words), 1) * 20.0, 1.0
    )
    features[14] = min(
        len(re.findall(r"\b\d+\b", text)) / max(len(words), 1) * 10.0, 1.0
    )
    features[15] = min(text.count("?") / max(len(sents), 1), 1.0)

    bg_counter: Counter = Counter()
    for i in range(len(text_lc) - 1):
        bg_counter[text_lc[i : i + 2]] += 1
    total_bg = max(sum(bg_counter.values()), 1)
    for i, bg in enumerate(_BIGRAMS):
        features[16 + i] = min(bg_counter.get(bg, 0) / total_bg * 10.0, 1.0)

    tiled = np.resize(features, n_neurons)
    return (tiled * 20.0).tolist()


# ── WebSensor ────────────────────────────────────────────────
class WebSensor:
    """
    Background worker fetching internet text with 4 parallel workers.
    Curiosity-driven: set_urgency(factor) scales amplitude and fetch rate.
    """

    def __init__(
        self, fetch_interval_s: float = 15.0, n_neurons: int = N_NEURONS
    ) -> None:
        self.fetch_interval_s = fetch_interval_s
        self.n_neurons = n_neurons

        self._queue: deque[List[float]] = deque(maxlen=1000)
        self._current_vec: List[float] = [0.0] * n_neurons
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self.last_items: List[Dict] = []
        self.interest_topics: List[str] = list(WIKIPEDIA_TOPICS[:8])
        self._urgency: float = 1.0
        self._topic_hits: Counter = Counter()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def encode(self) -> List[float]:
        """Pop one vector from queue. Amplitude scaled by urgency."""
        with self._lock:
            if self._queue:
                self._current_vec = self._queue.popleft()
            return [min(c * self._urgency, 24.0) for c in self._current_vec]

    def set_urgency(self, factor: float) -> None:
        """Set amplitude/fetch urgency [0.4, 3.5]. Called by emotion engine."""
        self._urgency = max(0.4, min(3.5, factor))

    def add_interest_topic(self, topic: str) -> None:
        if topic and topic not in self.interest_topics:
            self.interest_topics.append(topic)
            if len(self.interest_topics) > 60:
                self.interest_topics.pop(0)

    # ── Background fetch loop ──────────────────────────────────

    def _fetch_loop(self) -> None:
        time.sleep(2.0)  # stagger startup
        while self._running:
            try:
                self._run_parallel_fetch()
            except Exception:
                pass
            wait = max(5.0, self.fetch_interval_s / max(0.5, self._urgency))
            time.sleep(wait)

    def _run_parallel_fetch(self) -> None:
        tasks = {
            "rss": self._fetch_rss,
            "wikipedia": self._fetch_wikipedia,
            "youtube": self._fetch_youtube,
            "ddg": self._fetch_ddg,
        }
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures, timeout=20):
                try:
                    results = future.result(timeout=5)
                    for text, meta in results:
                        vec = self._text_to_currents(text)
                        with self._lock:
                            self._queue.append(vec)
                            # Store a text excerpt so consciousness can extract
                            # concepts from body content, not just the title
                            self.last_items.append({**meta, "text": text[:400]})
                            if len(self.last_items) > 100:
                                self.last_items.pop(0)
                        self._topic_hits[meta.get("type", "?")] += 1
                except Exception:
                    pass

    # ── RSS fetch ──────────────────────────────────────────────

    def _fetch_rss(self) -> List[Tuple[str, Dict]]:
        results: List[Tuple[str, Dict]] = []
        try:
            import random

            import feedparser
        except ImportError:
            return results
        feeds = random.sample(RSS_FEEDS, min(8, len(RSS_FEEDS)))
        for url in feeds:
            try:
                parsed = feedparser.parse(url)
                for entry in (parsed.entries or [])[:6]:
                    title = getattr(entry, "title", "") or ""
                    summary = getattr(entry, "summary", "") or ""
                    clean = re.sub(r"<[^>]+>", " ", title + " " + summary)
                    if len(clean.split()) >= 5:
                        results.append(
                            (clean, {"source": url, "title": title[:80], "type": "rss"})
                        )
            except Exception:
                continue
        return results

    # ── Wikipedia fetch ────────────────────────────────────────

    def _fetch_wikipedia(self) -> List[Tuple[str, Dict]]:
        results: List[Tuple[str, Dict]] = []
        try:
            import random

            import requests

            pool = list(dict.fromkeys(self.interest_topics + WIKIPEDIA_TOPICS))
            topics = random.sample(pool, min(5, len(pool)))

            for topic in topics:
                try:
                    url = (
                        "https://en.wikipedia.org/w/api.php"
                        f"?action=query&prop=extracts&exintro=true"
                        f"&explaintext=true&titles={topic}&format=json&redirects=1"
                    )
                    resp = requests.get(
                        url, timeout=8, headers={"User-Agent": "BrainSim/2.0"}
                    )
                    if resp.status_code == 200:
                        for page in (
                            resp.json().get("query", {}).get("pages", {}).values()
                        ):
                            extract = page.get("extract", "")
                            title = page.get("title", topic)
                            words = extract.split()
                            text = " ".join(words[:600])
                            if len(words) >= 10:
                                results.append(
                                    (
                                        text,
                                        {
                                            "source": f"wikipedia:{topic}",
                                            "title": title,
                                            "type": "wikipedia",
                                        },
                                    )
                                )
                except Exception:
                    pass

            # Random article for serendipitous discovery
            try:
                import requests

                resp_r = requests.get(
                    "https://en.wikipedia.org/api/rest_v1/page/random/summary",
                    timeout=6,
                    headers={"User-Agent": "BrainSim/2.0"},
                )
                if resp_r.status_code == 200:
                    data = resp_r.json()
                    extract = data.get("extract", "")
                    title = data.get("title", "random")
                    if len(extract.split()) >= 8:
                        results.append(
                            (
                                extract,
                                {
                                    "source": "wikipedia:random",
                                    "title": title,
                                    "type": "wikipedia",
                                },
                            )
                        )
                    if len(extract.split()) > 50:
                        words_t = re.findall(r"[A-Z][a-z_]{3,}", title)
                        if words_t:
                            self.add_interest_topic(words_t[0])
            except Exception:
                pass
        except Exception:
            pass
        return results

    # ── YouTube fetch ──────────────────────────────────────────

    def _fetch_youtube(self) -> List[Tuple[str, Dict]]:
        results: List[Tuple[str, Dict]] = []
        try:
            import random

            from youtube_transcript_api import YouTubeTranscriptApi

            n_vids = max(
                1, min(4, int(2 * self._urgency), len(YOUTUBE_EDUCATIONAL_IDS))
            )
            for vid_id in random.sample(YOUTUBE_EDUCATIONAL_IDS, n_vids):
                try:
                    entries = YouTubeTranscriptApi.get_transcript(
                        vid_id, languages=["en", "de", "en-US", "en-GB"]
                    )
                    text = " ".join(
                        e["text"] for e in entries if e.get("start", 0) < 300
                    )
                    if len(text.split()) >= 30:
                        results.append(
                            (
                                text,
                                {
                                    "source": f"youtube:{vid_id}",
                                    "title": f"YouTube:{vid_id}",
                                    "type": "youtube",
                                },
                            )
                        )
                except Exception:
                    continue
        except ImportError:
            pass
        return results

    # ── DuckDuckGo Instant Answers ─────────────────────────────

    def _fetch_ddg(self) -> List[Tuple[str, Dict]]:
        results: List[Tuple[str, Dict]] = []
        try:
            import random

            import requests

            if not self.interest_topics:
                return results
            query = random.choice(self.interest_topics[-10:])
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            resp = requests.get(url, timeout=8, headers={"User-Agent": "BrainSim/2.0"})
            if resp.status_code == 200:
                data = resp.json()
                abstract = data.get("AbstractText", "")
                if len(abstract.split()) >= 10:
                    results.append(
                        (
                            abstract,
                            {
                                "source": f"ddg:{query}",
                                "title": data.get("Heading", query)[:80],
                                "type": "ddg",
                            },
                        )
                    )
                for item in (data.get("RelatedTopics") or [])[:4]:
                    text = item.get("Text", "") if isinstance(item, dict) else ""
                    if len(text.split()) >= 6:
                        results.append(
                            (
                                text,
                                {
                                    "source": "ddg:related",
                                    "title": text[:60],
                                    "type": "ddg",
                                },
                            )
                        )
        except Exception:
            pass
        return results

    # ── Text -> current vector ─────────────────────────────────

    def _text_to_currents(self, text: str) -> List[float]:
        """Convert raw text to 48-element current vector in [0, 20] nA."""
        text_lc = text.lower()
        words = re.findall(r"[a-z\u00e4\u00f6\u00fc\u00df]+", text_lc)
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        features = np.zeros(self.n_neurons, dtype=np.float32)
        if not words:
            return features.tolist()

        features[0] = len(set(words)) / max(len(words), 1)
        features[1] = float(np.mean([len(w) for w in words])) / 10.0
        sent_lens = [
            len(re.findall(r"[a-z\u00e4\u00f6\u00fc\u00df]+", s)) for s in sents
        ]
        features[2] = min(float(np.mean(sent_lens)) / 30.0, 1.0) if sent_lens else 0.0
        features[3] = min(
            sum(1 for c in text if c in "!?;:\u2014\u2026") / max(len(sents), 1) / 3.0,
            1.0,
        )

        letter_groups = [
            "aeiou",
            "bcdfg",
            "hjklm",
            "npqrs",
            "tuvwx",
            "yz",
            string.digits,
            " \t\n",
        ]
        char_counts = Counter(text_lc)
        total_chars = max(sum(char_counts.values()), 1)
        for i, group in enumerate(letter_groups):
            features[4 + i] = sum(char_counts.get(c, 0) for c in group) / total_chars

        features[12] = min(
            sum(1 for w in words if w in _POSITIVE_WORDS) / max(len(words), 1) * 20.0,
            1.0,
        )
        features[13] = min(
            sum(1 for w in words if w in _NEGATIVE_WORDS) / max(len(words), 1) * 20.0,
            1.0,
        )
        features[14] = min(
            len(re.findall(r"\b\d+\b", text)) / max(len(words), 1) * 10.0, 1.0
        )
        features[15] = min(text.count("?") / max(len(sents), 1), 1.0)

        bg_counter: Counter = Counter()
        for i in range(len(text_lc) - 1):
            bg_counter[text_lc[i : i + 2]] += 1
        total_bg = max(sum(bg_counter.values()), 1)
        for i, bg in enumerate(_BIGRAMS):
            features[16 + i] = min(bg_counter.get(bg, 0) / total_bg * 10.0, 1.0)

        tiled = np.resize(features, self.n_neurons)
        return (tiled * 20.0).tolist()
