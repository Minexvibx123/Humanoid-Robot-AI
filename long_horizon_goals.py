"""
long_horizon_goals.py — Long-Horizon Goal Management

Extends the goal system with:
  • GoalCommitment: persistent, multi-session goals that survive restarts
  • ReviewPoint: scheduled self-evaluation checkpoints
  • GoalStack: hierarchical goal tracker with commitment and review

Features:
  - Goals persist across save/load cycles
  - Periodic review points trigger self-evaluation
  - Commitment tracking prevents premature abandonment
  - Progress estimation with milestone tracking

Integration:
  - task_executive.py: long-horizon goals decompose into short-term goals
  - consciousness.py: review points trigger metacognitive reflection
  - persistence.py: full serialisation of multi-session goal state
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Milestone:
    """A concrete sub-objective within a long-horizon goal."""

    description: str
    target_tick: int = 0  # expected completion tick (0 = no estimate)
    completed: bool = False
    completed_tick: int = 0
    # Phase 6 — executable bridging to TaskExecutive
    milestone_id: str = ""  # unique ID for cross-system matching
    executable_intent: str = ""  # maps to TaskExecutive operative/intent

    def to_dict(self) -> Dict:
        return {
            "description": self.description[:100],
            "target_tick": self.target_tick,
            "completed": self.completed,
            "completed_tick": self.completed_tick,
            "milestone_id": self.milestone_id,
            "executable_intent": self.executable_intent,
        }

    @staticmethod
    def from_dict(d: Dict) -> "Milestone":
        return Milestone(
            description=d.get("description", ""),
            target_tick=d.get("target_tick", 0),
            completed=d.get("completed", False),
            completed_tick=d.get("completed_tick", 0),
            milestone_id=d.get("milestone_id", ""),
            executable_intent=d.get("executable_intent", ""),
        )


@dataclass
class ReviewPoint:
    """Scheduled self-evaluation checkpoint."""

    review_tick: int  # when to review
    criteria: str  # what to evaluate
    completed: bool = False
    outcome: str = ""  # result of the review

    def to_dict(self) -> Dict:
        return {
            "review_tick": self.review_tick,
            "criteria": self.criteria[:120],
            "completed": self.completed,
            "outcome": self.outcome[:120],
        }

    @staticmethod
    def from_dict(d: Dict) -> "ReviewPoint":
        return ReviewPoint(
            review_tick=d.get("review_tick", 0),
            criteria=d.get("criteria", ""),
            completed=d.get("completed", False),
            outcome=d.get("outcome", ""),
        )


@dataclass
class GoalCommitment:
    """
    A long-horizon goal with commitment tracking.

    Unlike short-term goals (task_executive.Goal), these persist across
    sessions and have built-in review schedules and milestone tracking.
    """

    goal_id: str
    description: str
    category: str = "personal"  # personal, social, skill, exploration
    priority: int = 5  # 1-10
    created_tick: int = 0
    created_time: float = 0.0  # time.time() for cross-session tracking
    # ── Progress ────────────────────────────────────────────
    milestones: List[Milestone] = field(default_factory=list)
    review_points: List[ReviewPoint] = field(default_factory=list)
    progress: float = 0.0  # [0, 1]
    # ── Status ──────────────────────────────────────────────
    status: str = "active"  # active, paused, completed, abandoned
    commitment: float = 1.0  # [0, 1] — how committed AI is
    abandoned_reason: str = ""
    completed_tick: int = 0
    # ── Context ─────────────────────────────────────────────
    related_values: List[str] = field(default_factory=list)  # value_learning keys
    journal: List[str] = field(default_factory=list)  # brief log entries
    # ── Social / interruption tracking ──────────────────────
    person_id: Optional[int] = None     # set for social obligations (person who matters)
    interrupted_tick: int = 0           # tick when status set to 'paused'
    interrupt_reason: str = ""          # brief reason for interruption

    def add_milestone(
        self,
        description: str,
        target_tick: int = 0,
        executable_intent: str = "",
        milestone_id: str = "",
    ) -> None:
        _mid = milestone_id or f"{self.goal_id}_ms{len(self.milestones)}"
        self.milestones.append(
            Milestone(
                description,
                target_tick,
                milestone_id=_mid,
                executable_intent=executable_intent,
            )
        )

    def complete_milestone(self, index: int, tick: int) -> None:
        if 0 <= index < len(self.milestones):
            self.milestones[index].completed = True
            self.milestones[index].completed_tick = tick
            self._update_progress()

    def schedule_review(self, tick: int, criteria: str) -> None:
        self.review_points.append(ReviewPoint(tick, criteria))

    def due_reviews(self, current_tick: int) -> List[ReviewPoint]:
        return [
            rp
            for rp in self.review_points
            if not rp.completed and rp.review_tick <= current_tick
        ]

    def complete_review(self, review_tick: int, outcome: str) -> None:
        for rp in self.review_points:
            if rp.review_tick == review_tick and not rp.completed:
                rp.completed = True
                rp.outcome = outcome
                break

    def add_journal(self, entry: str) -> None:
        self.journal.append(entry[:120])
        if len(self.journal) > 30:
            self.journal = self.journal[-30:]

    def _update_progress(self) -> None:
        if self.milestones:
            done = sum(1 for m in self.milestones if m.completed)
            self.progress = done / len(self.milestones)
        # Auto-complete the goal
        if self.progress >= 1.0 and self.status == "active":
            self.status = "completed"

    def erode_commitment(self, amount: float = 0.01) -> None:
        """Gradually reduce commitment if no progress is being made."""
        self.commitment = max(0.0, self.commitment - amount)
        if self.commitment < 0.1 and self.status == "active":
            self.status = "abandoned"
            self.abandoned_reason = "commitment eroded below threshold"

    def reinforce_commitment(self, amount: float = 0.05) -> None:
        """Reinforce commitment when progress is made."""
        self.commitment = min(1.0, self.commitment + amount)

    def describe(self) -> str:
        ms = f"{sum(1 for m in self.milestones if m.completed)}/{len(self.milestones)}"
        return (
            f"[{self.goal_id}] {self.description[:40]} "
            f"p={self.priority} prog={self.progress:.0%} "
            f"commit={self.commitment:.2f} ms={ms} "
            f"status={self.status}"
        )

    def to_dict(self) -> Dict:
        return {
            "goal_id": self.goal_id,
            "description": self.description[:200],
            "category": self.category,
            "priority": self.priority,
            "created_tick": self.created_tick,
            "created_time": self.created_time,
            "milestones": [m.to_dict() for m in self.milestones[:20]],
            "review_points": [r.to_dict() for r in self.review_points[:20]],
            "progress": self.progress,
            "status": self.status,
            "commitment": self.commitment,
            "abandoned_reason": self.abandoned_reason,
            "completed_tick": self.completed_tick,
            "related_values": self.related_values[:10],
            "journal": self.journal[-20:],
            "person_id": self.person_id,
            "interrupted_tick": self.interrupted_tick,
            "interrupt_reason": self.interrupt_reason,
        }

    @staticmethod
    def from_dict(d: Dict) -> "GoalCommitment":
        gc = GoalCommitment(
            goal_id=d.get("goal_id", ""),
            description=d.get("description", ""),
            category=d.get("category", "personal"),
            priority=d.get("priority", 5),
            created_tick=d.get("created_tick", 0),
            created_time=d.get("created_time", 0.0),
            progress=d.get("progress", 0.0),
            status=d.get("status", "active"),
            commitment=d.get("commitment", 1.0),
            abandoned_reason=d.get("abandoned_reason", ""),
            completed_tick=d.get("completed_tick", 0),
            related_values=d.get("related_values", []),
            journal=d.get("journal", []),
            person_id=d.get("person_id", None),
            interrupted_tick=d.get("interrupted_tick", 0),
            interrupt_reason=d.get("interrupt_reason", ""),
        )
        gc.milestones = [Milestone.from_dict(m) for m in d.get("milestones", [])]
        gc.review_points = [
            ReviewPoint.from_dict(r) for r in d.get("review_points", [])
        ]
        return gc


# ─────────────────────────────────────────────────────────────────────────────
# GoalStack — long-horizon goal manager
# ─────────────────────────────────────────────────────────────────────────────


class GoalStack:
    """
    Manager for long-horizon, multi-session goals.

    Features:
      - Persistent across save/load
      - Auto-scheduled review points
      - Commitment erosion when stalled
      - Priority-based ordering
    """

    MAX_GOALS = 20
    REVIEW_INTERVAL = 10_000  # ticks between auto-reviews
    STALL_THRESHOLD = 5_000  # ticks without progress → commitment erodes

    def __init__(self) -> None:
        self._goals: Dict[str, GoalCommitment] = {}
        self._next_id: int = 1

    def add_goal(
        self,
        description: str,
        category: str = "personal",
        priority: int = 5,
        tick: int = 0,
        milestones: Optional[List[str]] = None,
        related_values: Optional[List[str]] = None,
    ) -> GoalCommitment:
        """Create and add a new long-horizon goal."""
        if len(self._goals) >= self.MAX_GOALS:
            # Remove lowest-priority completed/abandoned
            removable = [
                g
                for g in self._goals.values()
                if g.status in ("completed", "abandoned")
            ]
            if removable:
                worst = min(removable, key=lambda g: g.priority)
                del self._goals[worst.goal_id]
            else:
                # Remove lowest-commitment active goal
                worst = min(self._goals.values(), key=lambda g: g.commitment)
                del self._goals[worst.goal_id]

        goal_id = f"LH-{self._next_id:04d}"
        self._next_id += 1

        gc = GoalCommitment(
            goal_id=goal_id,
            description=description,
            category=category,
            priority=priority,
            created_tick=tick,
            created_time=time.time(),
            related_values=related_values or [],
        )
        if milestones:
            for ms in milestones:
                gc.add_milestone(ms)

        # Auto-schedule first review
        gc.schedule_review(
            tick + self.REVIEW_INTERVAL, f"Review progress on: {description[:50]}"
        )

        self._goals[goal_id] = gc
        return gc

    def tick(self, current_tick: int) -> List[str]:
        """
        Periodic maintenance. Returns list of action descriptions
        (reviews due, goals abandoned, etc.).
        """
        actions: List[str] = []
        for gc in list(self._goals.values()):
            if gc.status not in ("active", "paused"):
                continue

            # Check for due reviews
            due = gc.due_reviews(current_tick)
            for rp in due:
                actions.append(f"REVIEW_DUE:{gc.goal_id}:{rp.criteria}")

            # Check for stalls
            last_progress_tick = gc.created_tick
            for m in gc.milestones:
                if m.completed and m.completed_tick > last_progress_tick:
                    last_progress_tick = m.completed_tick

            if (
                current_tick - last_progress_tick > self.STALL_THRESHOLD
                and gc.status == "active"
            ):
                gc.erode_commitment(0.005)
                if gc.status == "abandoned":
                    actions.append(f"ABANDONED:{gc.goal_id}:{gc.description[:40]}")

        return actions

    def active_goals(self) -> List[GoalCommitment]:
        return sorted(
            [g for g in self._goals.values() if g.status == "active"],
            key=lambda g: -g.priority,
        )

    def get_goal(self, goal_id: str) -> Optional[GoalCommitment]:
        return self._goals.get(goal_id)

    def record_social_obligation(
        self,
        person_id: int,
        description: str,
        tick: int,
        priority: int = 8,
    ) -> GoalCommitment:
        """Create a persistent social obligation tied to a specific person.

        Social obligations are high-priority goals that survive session
        boundaries.  When that person reappears, resume_candidates() surfaces
        them so the conversation can pick up where it left off.
        """
        # Dedup: don't create a duplicate for the same person+description
        desc_lower = description.lower()[:80]
        for g in self._goals.values():
            if (
                g.category == "social"
                and g.person_id == person_id
                and g.status in ("active", "paused")
                and desc_lower in g.description.lower()
            ):
                # Reinforce commitment of the existing obligation
                g.reinforce_commitment(0.1)
                g.add_journal(f"t={tick} obligation re-confirmed")
                return g

        gc = self.add_goal(
            description[:200],
            category="social",
            priority=priority,
            tick=tick,
            milestones=[f"Fulfill: {description[:60]}"],
        )
        gc.person_id = person_id
        gc.add_journal(f"t={tick} social obligation towards person {person_id}")
        return gc

    def mark_interrupted(
        self, goal_id: str, tick: int, reason: str = "context_switch"
    ) -> bool:
        """Mark a goal as paused (interrupted).

        Paused goals retain all progress and are surfaced by resume_candidates()
        when conditions for resumption are met.  Social obligations paused while
        a person was absent resume when that person reappears.
        """
        gc = self._goals.get(goal_id)
        if gc is None or gc.status != "active":
            return False
        gc.status = "paused"
        gc.interrupted_tick = tick
        gc.interrupt_reason = reason[:120]
        gc.add_journal(f"t={tick} interrupted: {reason[:60]}")
        return True

    def resume_goal(self, goal_id: str, tick: int) -> bool:
        """Resume a paused goal."""
        gc = self._goals.get(goal_id)
        if gc is None or gc.status != "paused":
            return False
        gc.status = "active"
        gc.reinforce_commitment(0.05)
        gc.add_journal(f"t={tick} resumed after {tick - gc.interrupted_tick} ticks")
        return True

    def resume_candidates(
        self,
        current_tick: int,
        person_id: Optional[int] = None,
        min_pause_ticks: int = 0,
    ) -> List[GoalCommitment]:
        """Return paused goals that are candidates for resumption.

        Args:
            current_tick:     current tick count
            person_id:        if set, also include social obligations for this person
            min_pause_ticks:  only include goals paused for at least this many ticks
        """
        candidates = []
        for g in self._goals.values():
            if g.status == "paused":
                pause_dur = current_tick - g.interrupted_tick
                if pause_dur >= min_pause_ticks:
                    candidates.append(g)
            elif (
                g.status == "active"
                and g.category == "social"
                and person_id is not None
                and g.person_id == person_id
            ):
                # Active social obligations for this person are always relevant
                candidates.append(g)
        # Social obligations to this person sort first
        candidates.sort(
            key=lambda g: (
                0 if (g.person_id == person_id and g.category == "social") else 1,
                -g.priority,
            )
        )
        return candidates

    def next_executable_milestone(
        self, goal_id: str
    ) -> Optional["Milestone"]:
        """Return the first incomplete milestone with an executable_intent.

        Used by the tick loop to drive short-term tasks from long-horizon goals.
        """
        gc = self._goals.get(goal_id)
        if gc is None or gc.status != "active":
            return None
        for ms in gc.milestones:
            if not ms.completed and ms.executable_intent:
                return ms
        return None

    def project_summary_for_prompt(self, n: int = 3) -> str:
        """Return a short human-readable summary of the top-n active projects.

        Format: semicolon-separated list of goal descriptions with progress.
        Safe to include in LLM system prompts as one line.
        """
        active = self.active_goals()[:n]
        if not active:
            return ""
        parts = []
        for g in active:
            pct = f"{g.progress:.0%}" if g.progress > 0 else "neu"
            person_note = f" (Person {g.person_id})" if g.person_id is not None else ""
            parts.append(f"{g.description[:50]}{person_note} [{pct}]")
        return "; ".join(parts)

    def social_obligations_for(self, person_id: int) -> List[GoalCommitment]:
        """Return all active or paused social obligations for a given person."""
        return [
            g for g in self._goals.values()
            if g.category == "social"
            and g.person_id == person_id
            and g.status in ("active", "paused")
        ]

    def summary(self) -> str:
        active = [g for g in self._goals.values() if g.status == "active"]
        if not active:
            return "Long-horizon: no active goals"
        parts = [
            f"{g.goal_id}({g.progress:.0%})"
            for g in sorted(active, key=lambda g: -g.priority)[:3]
        ]
        return "Long-horizon: " + ", ".join(parts)

    # ── Pattern-based goal generation ─────────────────────────────────

    PATTERN_THRESHOLD = 3  # repeated occurrences before spawning a goal
    PATTERN_CHECK_INTERVAL = 5000  # ticks between pattern checks

    def __init_pattern_tracker(self) -> None:
        if not hasattr(self, "_pattern_counts"):
            self._pattern_counts: Dict[str, int] = {}
            self._last_pattern_check: int = 0

    def record_pattern(self, pattern_type: str, detail: str = "") -> None:
        """Record a recurring pattern observation. Called from consciousness tick."""
        self.__init_pattern_tracker()
        key = f"{pattern_type}:{detail[:30]}" if detail else pattern_type
        self._pattern_counts[key] = self._pattern_counts.get(key, 0) + 1

    def detect_and_spawn_goals(self, tick: int) -> List[str]:
        """
        Check accumulated patterns and auto-generate goals when thresholds
        are reached. Returns list of descriptions of newly spawned goals.

        Pattern types that trigger goals:
          - knowledge_gap:<topic>     → exploration project
          - social_failure:<person>   → relationship repair project
          - continuity_alarm          → self-stabilization project
          - turning_point:<emotion>   → identity development project
          - repeated_failure:<goal>   → strategy revision project
        """
        self.__init_pattern_tracker()
        if tick - self._last_pattern_check < self.PATTERN_CHECK_INTERVAL:
            return []
        self._last_pattern_check = tick

        spawned: List[str] = []
        existing_descs = {
            g.description.lower() for g in self._goals.values() if g.status == "active"
        }

        _PATTERN_MAP = {
            "knowledge_gap": ("exploration", "Explore and learn about {detail}", 7),
            "social_failure": ("social", "Repair social dynamics with {detail}", 6),
            "continuity_alarm": (
                "personal",
                "Strengthen self-continuity and stability",
                8,
            ),
            "turning_point": ("personal", "Process identity shift: {detail}", 5),
            "repeated_failure": ("skill", "Revise strategy for {detail}", 7),
        }

        for key, count in list(self._pattern_counts.items()):
            if count < self.PATTERN_THRESHOLD:
                continue
            parts = key.split(":", 1)
            ptype = parts[0]
            detail = parts[1] if len(parts) > 1 else ""
            mapping = _PATTERN_MAP.get(ptype)
            if mapping is None:
                continue

            category, desc_template, priority = mapping
            desc = (
                desc_template.format(detail=detail)
                if detail
                else desc_template.replace(" {detail}", "")
            )

            if desc.lower() in existing_descs:
                continue

            active_count = len(
                [g for g in self._goals.values() if g.status == "active"]
            )
            if active_count >= self.MAX_GOALS - 2:
                break

            self.add_goal(
                desc,
                category=category,
                priority=priority,
                tick=tick,
                milestones=[f"Initial assessment of {detail or ptype}"],
            )
            spawned.append(desc)
            # Reset pattern count so it takes fresh evidence to re-trigger
            self._pattern_counts[key] = 0

        return spawned

    def to_dict(self) -> Dict:
        self.__init_pattern_tracker()
        return {
            "goals": {k: v.to_dict() for k, v in self._goals.items()},
            "next_id": self._next_id,
            "pattern_counts": dict(self._pattern_counts),
        }

    def from_dict(self, data: Dict) -> None:
        self._goals.clear()
        for k, v in data.get("goals", {}).items():
            self._goals[k] = GoalCommitment.from_dict(v)
        self._next_id = data.get("next_id", len(self._goals) + 1)
        self._pattern_counts = data.get("pattern_counts", {})
        self._last_pattern_check = 0
