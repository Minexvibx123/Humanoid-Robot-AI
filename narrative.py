"""
narrative.py — Autobiographical Narrative Engine

Builds a running narrative of the system's life through:
  • NarrativeChapter: compressed episode covering a time span
  • NarrativeThread: continuous storyline with chapter formation
  • Turning point detection and self-interpretation

Unlike EpisodicMemory (raw event listing) and AutobiographicalIdentity
(statistical summary), this module creates MEANING from experience:
what happened, why it mattered, and how it changed the system.

Integration:
  - consciousness.py: SelfModel accesses narrative for self-description
  - identity_arc.py: chapters inform identity dimension updates
  - persistence.py: serialises chapters to SQLite
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# NarrativeChapter — one chapter of the system's story
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NarrativeChapter:
    """A compressed autobiographical chapter spanning a time range."""

    start_tick: int
    end_tick: int
    title: str
    dominant_emotion: str
    conflict: str  # what challenge or tension defined this period
    resolution: str  # how the conflict was (or wasn't) resolved
    lessons: List[str]  # what was learned
    key_concepts: List[str]  # most important concepts in this chapter
    persons_involved: List[str]  # people present during this chapter
    turning_point: bool = False  # was this a significant change moment?
    chapter_type: str = "normal"  # "learning", "social", "conflict", "growth", "rest"

    def describe(self) -> str:
        tp = " [TURNING POINT]" if self.turning_point else ""
        persons = (
            f" persons: {', '.join(self.persons_involved[:3])}"
            if self.persons_involved
            else ""
        )
        return (
            f"[Ch {self.start_tick}-{self.end_tick}{tp}] "
            f"'{self.title}' ({self.chapter_type}) "
            f"emotion={self.dominant_emotion}{persons} "
            f"conflict={self.conflict[:40]} → {self.resolution[:40]}"
        )

    def to_dict(self) -> Dict:
        return {
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "title": self.title,
            "dominant_emotion": self.dominant_emotion,
            "conflict": self.conflict,
            "resolution": self.resolution,
            "lessons": self.lessons,
            "key_concepts": self.key_concepts,
            "persons_involved": self.persons_involved,
            "turning_point": self.turning_point,
            "chapter_type": self.chapter_type,
        }

    @staticmethod
    def from_dict(d: Dict) -> "NarrativeChapter":
        return NarrativeChapter(
            start_tick=d.get("start_tick", 0),
            end_tick=d.get("end_tick", 0),
            title=d.get("title", ""),
            dominant_emotion=d.get("dominant_emotion", ""),
            conflict=d.get("conflict", ""),
            resolution=d.get("resolution", ""),
            lessons=d.get("lessons", []),
            key_concepts=d.get("key_concepts", []),
            persons_involved=d.get("persons_involved", []),
            turning_point=d.get("turning_point", False),
            chapter_type=d.get("chapter_type", "normal"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# RelationshipArc — per-person narrative history
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RelationshipArc:
    """Narrative history of the relationship with one person."""

    person_id: str
    first_met_tick: int = 0
    total_chapters: int = 0
    trust_trajectory: List[float] = field(default_factory=list)
    key_moments: List[str] = field(default_factory=list)
    dominant_pattern: str = "neutral"  # "cooperative", "conflictual", "distant", etc.
    last_interaction: int = 0

    def record_interaction(self, tick: int, trust: float, event_type: str) -> None:
        self.trust_trajectory.append(trust)
        if len(self.trust_trajectory) > 100:
            self.trust_trajectory = self.trust_trajectory[-100:]
        self.last_interaction = tick
        if event_type in ("conflict", "betrayal", "repair", "breakthrough"):
            self.key_moments.append(f"[t={tick}] {event_type}")
            if len(self.key_moments) > 20:
                self.key_moments = self.key_moments[-20:]

    def trust_trend(self) -> str:
        if len(self.trust_trajectory) < 3:
            return "insufficient_data"
        recent = self.trust_trajectory[-5:]
        avg = sum(recent) / len(recent)
        if avg > 0.7:
            return "high_trust"
        if avg < 0.3:
            return "low_trust"
        # Check slope
        early = sum(self.trust_trajectory[:5]) / max(len(self.trust_trajectory[:5]), 1)
        if avg > early + 0.1:
            return "improving"
        if avg < early - 0.1:
            return "declining"
        return "stable"

    def to_dict(self) -> Dict:
        return {
            "person_id": self.person_id,
            "first_met_tick": self.first_met_tick,
            "total_chapters": self.total_chapters,
            "trust_trajectory": self.trust_trajectory[-50:],
            "key_moments": self.key_moments[-10:],
            "dominant_pattern": self.dominant_pattern,
            "last_interaction": self.last_interaction,
        }

    @staticmethod
    def from_dict(d: Dict) -> "RelationshipArc":
        ra = RelationshipArc(
            person_id=d.get("person_id", ""),
            first_met_tick=d.get("first_met_tick", 0),
            total_chapters=d.get("total_chapters", 0),
            dominant_pattern=d.get("dominant_pattern", "neutral"),
            last_interaction=d.get("last_interaction", 0),
        )
        ra.trust_trajectory = d.get("trust_trajectory", [])
        ra.key_moments = d.get("key_moments", [])
        return ra


# ─────────────────────────────────────────────────────────────────────────────
# NarrativeThread — the continuous lifecycle story
# ─────────────────────────────────────────────────────────────────────────────


class NarrativeThread:
    """
    Continuous narrative engine that forms chapters from experience.

    Chapter formation triggers:
      1. Enough ticks elapsed (CHAPTER_INTERVAL)
      2. Emotional tone shift detected
      3. Major event (turning point)

    The narrative provides:
      - Self-description over time (not just current state)
      - Relationship histories per person
      - Turning point awareness for identity development
    """

    CHAPTER_INTERVAL = 5_000  # ticks between automatic chapter consolidation
    MAX_CHAPTERS = 200

    def __init__(self) -> None:
        self._chapters: Deque[NarrativeChapter] = deque(maxlen=self.MAX_CHAPTERS)
        self._relationship_arcs: Dict[str, RelationshipArc] = {}
        self._current_chapter_start: int = 0
        self._accumulator: _ChapterAccumulator = _ChapterAccumulator()
        self._turning_points: Deque[str] = deque(maxlen=50)
        self._last_dominant_emotion: str = ""

    def observe_tick(
        self,
        tick: int,
        goal: str,
        emotion: str,
        concepts: List[str],
        success: bool,
        person_ids: Optional[List[str]] = None,
        event_type: str = "",
    ) -> None:
        """Accumulate observations for chapter formation."""
        self._accumulator.add(
            goal, emotion, concepts, success, person_ids or [], event_type
        )

        # Track relationships
        for pid in person_ids or []:
            if pid not in self._relationship_arcs:
                self._relationship_arcs[pid] = RelationshipArc(
                    person_id=pid, first_met_tick=tick
                )

    def observe_skill_event(
        self,
        tick: int,
        skill_name: str,
        status: str,
        goal_intent: str,
        person_id: str = "",
    ) -> None:
        """Accumulate per-step skill events for finer narrative texture.

        Instead of only recording goal-level success/failure, the narrative
        can now say 'fixate_person succeeded, then express_emotion failed'
        rather than just 'greet_person failed'.
        """
        _skill_events = getattr(self._accumulator, "_skill_events", None)
        if _skill_events is None:
            self._accumulator._skill_events = []
            _skill_events = self._accumulator._skill_events
        _skill_events.append((tick, skill_name, status, goal_intent))
        if len(_skill_events) > 200:
            self._accumulator._skill_events = _skill_events[-200:]

        # Track relationship: person-involved skill events
        if person_id and person_id not in self._relationship_arcs:
            self._relationship_arcs[person_id] = RelationshipArc(
                person_id=person_id, first_met_tick=tick
            )

    def try_close_chapter(self, tick: int) -> Optional[NarrativeChapter]:
        """
        Attempt to close the current chapter and start a new one.
        Returns the chapter if one was formed, None otherwise.
        """
        elapsed = tick - self._current_chapter_start
        if elapsed < self.CHAPTER_INTERVAL:
            return None

        acc = self._accumulator
        if acc.tick_count < 100:
            return None

        # Determine chapter properties
        dom_emotion = acc.dominant_emotion()
        dom_goal = acc.dominant_goal()
        top_concepts = acc.top_concepts(5)
        all_persons = acc.all_persons()

        # Detect turning point: emotion shift or major failure/success pattern
        is_turning = False
        if self._last_dominant_emotion and dom_emotion != self._last_dominant_emotion:
            is_turning = True
        if acc.success_rate() < 0.3 and acc.tick_count > 200:
            is_turning = True  # failure crisis
        if acc.success_rate() > 0.9 and acc.tick_count > 200:
            is_turning = True  # breakthrough

        # Generate chapter title
        title = self._generate_title(dom_goal, dom_emotion, top_concepts, is_turning)

        # Conflict/resolution
        conflict = self._infer_conflict(acc)
        resolution = self._infer_resolution(acc, is_turning)

        # Lessons
        lessons = self._extract_lessons(acc, dom_goal, dom_emotion)

        # Chapter type
        ch_type = self._classify_chapter(dom_goal, dom_emotion, acc, all_persons)

        chapter = NarrativeChapter(
            start_tick=self._current_chapter_start,
            end_tick=tick,
            title=title,
            dominant_emotion=dom_emotion,
            conflict=conflict,
            resolution=resolution,
            lessons=lessons,
            key_concepts=top_concepts,
            persons_involved=all_persons,
            turning_point=is_turning,
            chapter_type=ch_type,
        )

        self._chapters.append(chapter)
        self._last_dominant_emotion = dom_emotion
        self._current_chapter_start = tick
        self._accumulator = _ChapterAccumulator()

        if is_turning:
            self._turning_points.append(f"[t={tick}] {title} ({dom_emotion}→{ch_type})")

        return chapter

    def _generate_title(
        self, goal: str, emotion: str, concepts: List[str], turning: bool
    ) -> str:
        if turning:
            prefix = "Turning point: "
        else:
            prefix = ""
        if concepts:
            topic = concepts[0]
            return f"{prefix}{emotion.capitalize()} {goal} — {topic}"
        return f"{prefix}{emotion.capitalize()} {goal} phase"

    def _infer_conflict(self, acc: "_ChapterAccumulator") -> str:
        if acc.success_rate() < 0.4:
            return "repeated failure eroding confidence"
        if (
            "stress" in acc.emotion_counts
            and acc.emotion_counts["stress"] > acc.tick_count * 0.3
        ):
            return "sustained stress pressure"
        if acc.tick_count > 0 and len(acc.unique_concepts) < 3:
            return "semantic stagnation"
        return "maintaining coherence under normal load"

    def _infer_resolution(self, acc: "_ChapterAccumulator", turning: bool) -> str:
        if turning and acc.success_rate() > 0.7:
            return "breakthrough through persistence"
        if turning and acc.success_rate() < 0.3:
            return "unresolved — strategy shift needed"
        if acc.success_rate() > 0.6:
            return "steady progress maintained"
        return "ongoing — no clear resolution"

    def _extract_lessons(
        self, acc: "_ChapterAccumulator", goal: str, emotion: str
    ) -> List[str]:
        lessons = []
        if acc.success_rate() < 0.4:
            lessons.append(f"Approach to {goal} needs revision")
        if acc.success_rate() > 0.8:
            lessons.append(f"Current {goal} strategy is effective")
        if "stress" in acc.emotion_counts and acc.emotion_counts.get("calm", 0) > 0:
            stress_ratio = acc.emotion_counts["stress"] / max(
                acc.emotion_counts.get("calm", 1), 1
            )
            if stress_ratio > 2:
                lessons.append("Need better stress management")
        if len(acc.unique_concepts) > 10:
            lessons.append("Broad conceptual exploration achieved")
        return lessons[:3]

    def _classify_chapter(
        self, goal: str, emotion: str, acc: "_ChapterAccumulator", persons: List[str]
    ) -> str:
        if persons and len(persons) > 0:
            return "social"
        if goal == "explore" and acc.success_rate() > 0.5:
            return "learning"
        if acc.success_rate() < 0.3:
            return "conflict"
        if goal == "rest":
            return "rest"
        if acc.success_rate() > 0.7:
            return "growth"
        return "normal"

    def record_relationship_event(
        self, tick: int, person_id: str, trust: float, event_type: str
    ) -> None:
        if person_id not in self._relationship_arcs:
            self._relationship_arcs[person_id] = RelationshipArc(
                person_id=person_id, first_met_tick=tick
            )
        self._relationship_arcs[person_id].record_interaction(tick, trust, event_type)

    def relationship_summary(self, person_id: str) -> str:
        arc = self._relationship_arcs.get(person_id)
        if arc is None:
            return f"No relationship history with {person_id}"
        trend = arc.trust_trend()
        moments = "; ".join(arc.key_moments[-3:]) if arc.key_moments else "none"
        return (
            f"Relationship with {person_id}: "
            f"pattern={arc.dominant_pattern}, trust_trend={trend}, "
            f"key_moments=[{moments}]"
        )

    def story_so_far(self, n_chapters: int = 5) -> str:
        """Return a condensed narrative of recent chapters."""
        chapters = list(self._chapters)[-n_chapters:]
        if not chapters:
            return "My story has just begun."
        parts = []
        for ch in chapters:
            tp = " *" if ch.turning_point else ""
            parts.append(f"'{ch.title}'{tp}: {ch.resolution}")
        return " → ".join(parts)

    def turning_point_count(self) -> int:
        return sum(1 for ch in self._chapters if ch.turning_point)

    def recent_chapters(self, n: int = 3) -> List[NarrativeChapter]:
        return list(self._chapters)[-n:]

    def to_dict(self) -> Dict:
        return {
            "chapters": [ch.to_dict() for ch in self._chapters],
            "relationships": {
                k: v.to_dict() for k, v in self._relationship_arcs.items()
            },
            "turning_points": list(self._turning_points),
            "current_chapter_start": self._current_chapter_start,
            "last_dominant_emotion": self._last_dominant_emotion,
        }

    def from_dict(self, data: Dict) -> None:
        for cd in data.get("chapters", []):
            self._chapters.append(NarrativeChapter.from_dict(cd))
        for pid, rd in data.get("relationships", {}).items():
            self._relationship_arcs[pid] = RelationshipArc.from_dict(rd)
        for tp in data.get("turning_points", []):
            self._turning_points.append(tp)
        self._current_chapter_start = data.get("current_chapter_start", 0)
        self._last_dominant_emotion = data.get("last_dominant_emotion", "")


# ─────────────────────────────────────────────────────────────────────────────
# Internal accumulator for chapter building
# ─────────────────────────────────────────────────────────────────────────────


class _ChapterAccumulator:
    def __init__(self) -> None:
        self.goal_counts: Dict[str, int] = {}
        self.emotion_counts: Dict[str, int] = {}
        self.concept_counts: Dict[str, int] = {}
        self.unique_concepts: set = set()
        self.success_count: int = 0
        self.failure_count: int = 0
        self.tick_count: int = 0
        self.person_set: set = set()
        self.events: List[str] = []

    def add(
        self,
        goal: str,
        emotion: str,
        concepts: List[str],
        success: bool,
        person_ids: List[str],
        event_type: str,
    ) -> None:
        self.tick_count += 1
        self.goal_counts[goal] = self.goal_counts.get(goal, 0) + 1
        self.emotion_counts[emotion] = self.emotion_counts.get(emotion, 0) + 1
        for c in concepts[-3:]:
            self.concept_counts[c] = self.concept_counts.get(c, 0) + 1
            self.unique_concepts.add(c)
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.person_set.update(person_ids)
        if event_type:
            self.events.append(event_type)

    def dominant_emotion(self) -> str:
        if not self.emotion_counts:
            return "neutral"
        return max(self.emotion_counts, key=self.emotion_counts.get)

    def dominant_goal(self) -> str:
        if not self.goal_counts:
            return "explore"
        return max(self.goal_counts, key=self.goal_counts.get)

    def top_concepts(self, n: int = 5) -> List[str]:
        ranked = sorted(self.concept_counts.items(), key=lambda x: -x[1])
        return [c for c, _ in ranked[:n]]

    def all_persons(self) -> List[str]:
        return list(self.person_set)

    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5
