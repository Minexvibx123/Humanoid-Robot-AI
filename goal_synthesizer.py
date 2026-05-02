"""
goal_synthesizer.py — Open Goal Space: Self-Generated Motivations

Replaces the hard-coded four-goal set (explore/consolidate/respond/rest)
with a dynamic pool of GoalAtoms that the system can:
  • Generate  – from prediction errors, state conflicts, self-model
                 inconsistencies, and social interactions.
  • Mutate    – change priority or description in response to evidence.
  • Reject    – remove goals that consistently fail or become irrelevant.
  • Organise  – form parent/child hierarchies from recurring sub-goals.

Core goal generation sources:
  1. Prediction Error   → "resolve <surprise_entity>"
  2. State Conflict     → "reduce <drive_A>/<drive_B> tension"
  3. Self-Model Inconsistency → "reconcile self:<aspect>"
  4. Social Signals     → "understand <person>"
  5. Knowledge Gap      → "investigate <concept>"
  6. Narrative Pattern  → born from recurring chapter types

Integration:
  goal_synth = GoalSynthesizer()

  # each tick / every N ticks, synthesize candidates
  new_goals = goal_synth.synthesize(brain, em, consciousness_core)

  # retrieve top-priority goal atoms for selection
  top = goal_synth.top_goals(n=5)

  # after outcome:
  goal_synth.record_outcome(goal_name, success, reward)

  # get current goal pool for export to ConsciousnessCore
  names = goal_synth.active_goal_names()
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from brain import Brain


# ─── Constants ────────────────────────────────────────────────────────────────
MAX_POOL_SIZE = 40  # maximum live goal atoms
MIN_SALIENCE_TO_KEEP = 0.05  # below this → candidate for pruning
SALIENCE_DECAY_RATE = 0.008  # per-tick decay of goal salience
GOAL_FULFILLMENT_THRESHOLD = 0.80  # goal auto-closes at this fulfillment
MUTATION_INTERVAL = 200  # ticks between priority recalculation
GENERATION_INTERVAL = 100  # ticks between new goal synthesis
FAIL_STREAK_CUTOFF = 4  # consecutive failures → reject goal
HIERARCHY_MIN_RECUR = 3  # min recurrences before sub-goal merges to parent

# Built-in base goals that cannot be removed (but CAN have their priority mutated)
BASE_GOALS = {"explore", "consolidate", "respond", "rest"}


# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class GoalAtom:
    """A single goal in the dynamic pool."""

    name: str
    source: str  # "base" | "prediction_error" | "conflict" |
    # "self_inconsistency" | "social" | "gap" |
    # "narrative" | "mutation"
    description: str
    priority: float  # 0..2, can mutate
    salience: float  # decays; refreshed on evidence
    created_tick: int
    last_active_tick: int
    parent: Optional[str] = None  # parent goal name
    children: List[str] = field(default_factory=list)
    evidence: str = ""
    fulfillment: float = 0.0  # 0..1
    success_count: int = 0
    fail_count: int = 0
    expected_reward: float = 0.5
    # Closed / rejected flags
    closed: bool = False
    closed_reason: str = ""


@dataclass
class GoalSynthesisEvent:
    """Records that a goal was synthesized, mutated, or rejected."""

    tick: int
    event_type: str  # "born" | "mutated" | "rejected" | "fulfilled"
    goal_name: str
    reason: str


# ─── Main class ───────────────────────────────────────────────────────────────


class GoalSynthesizer:
    """
    Dynamic, open goal space.  The system generates its own motivations
    rather than selecting from a fixed menu.

    The four base goals (explore, consolidate, respond, rest) are seeded
    at init and kept permanently; synthesized goals are added, mutated,
    and pruned autonomously.
    """

    def __init__(self) -> None:
        self._tick: int = 0
        self._pool: Dict[str, GoalAtom] = {}
        self._event_log: Deque[GoalSynthesisEvent] = deque(maxlen=500)
        self._last_synthesis_tick: int = 0
        self._last_mutation_tick: int = 0

        # Seed base goals
        for g in BASE_GOALS:
            self._pool[g] = GoalAtom(
                name=g,
                source="base",
                description=f"Base goal: {g}",
                priority=1.0,
                salience=0.5,
                created_tick=0,
                last_active_tick=0,
            )

    # ── Main tick-level synthesis API ────────────────────────────────────────

    def synthesize(
        self,
        tick: int,
        prediction_error: float,
        surprise_entity: str,
        drives: Any,  # IntrinsicDrives
        em: Any,  # EmotionalState
        meta_gaps: List[str],
        self_tensions: List[str],
        social_person: Optional[str],
        narrative_pattern: Optional[str],
        world_entities: int,
    ) -> List[GoalAtom]:
        """
        Synthesize new goal atoms from internal state signals.

        Returns list of newly created goals (may be empty).
        """
        self._tick = tick

        if tick - self._last_synthesis_tick < GENERATION_INTERVAL:
            # Decay saliences even when not synthesizing
            self._decay()
            return []

        self._last_synthesis_tick = tick
        new_goals: List[GoalAtom] = []

        # ── Source 1: Prediction error ────────────────────────────────
        if prediction_error > 0.45 and surprise_entity:
            name = f"resolve_{surprise_entity.replace(' ', '_')[:20]}"
            if name not in self._pool:
                atom = self._make_goal(
                    name=name,
                    source="prediction_error",
                    description=f"Resolve surprise about '{surprise_entity}'",
                    priority=0.9 + prediction_error * 0.5,
                    salience=prediction_error,
                    evidence=f"pred_error={prediction_error:.2f}",
                )
                new_goals.append(atom)

        # ── Source 2: Drive/emotion conflict ─────────────────────────
        _drive_info_hunger = getattr(drives, "information_hunger", 0.0)
        _drive_rest_need = getattr(drives, "rest_need", 0.0)
        _drive_expr_press = getattr(drives, "expression_pressure", 0.0)
        _em_fatigue = getattr(em, "fatigue", 0.0)

        if _drive_info_hunger > 0.7 and _drive_rest_need > 0.6:
            name = "resolve_explore_rest_conflict"
            if name not in self._pool:
                atom = self._make_goal(
                    name=name,
                    source="conflict",
                    description="Resolve tension: information_hunger vs rest_need",
                    priority=0.8,
                    salience=(_drive_info_hunger + _drive_rest_need) * 0.5,
                    evidence=f"info={_drive_info_hunger:.2f} rest={_drive_rest_need:.2f}",
                )
                new_goals.append(atom)

        if _drive_expr_press > 0.8 and _em_fatigue > 0.7:
            name = "resolve_speak_vs_fatigue"
            if name not in self._pool:
                atom = self._make_goal(
                    name=name,
                    source="conflict",
                    description="Resolve tension: expression_pressure vs fatigue",
                    priority=0.75,
                    salience=(_drive_expr_press + _em_fatigue) * 0.5,
                    evidence=f"expr={_drive_expr_press:.2f} fatigue={_em_fatigue:.2f}",
                )
                new_goals.append(atom)

        # ── Source 3: Self-model inconsistency ────────────────────────
        for tension in self_tensions[:2]:
            name = f"reconcile_{tension.replace(' ', '_')[:25]}"
            if name not in self._pool and len(name) > 12:
                atom = self._make_goal(
                    name=name,
                    source="self_inconsistency",
                    description=f"Reconcile self-model tension: {tension}",
                    priority=0.85,
                    salience=0.6,
                    evidence=f"tension:{tension[:40]}",
                )
                new_goals.append(atom)

        # ── Source 4: Knowledge gaps ──────────────────────────────────
        for gap in meta_gaps[:2]:
            name = f"investigate_{gap.replace(' ', '_')[:22]}"
            if name not in self._pool:
                atom = self._make_goal(
                    name=name,
                    source="gap",
                    description=f"Investigate knowledge gap: {gap}",
                    priority=0.70,
                    salience=0.55,
                    evidence=f"gap:{gap[:40]}",
                    parent="explore",
                )
                new_goals.append(atom)

        # ── Source 5: Social signals ──────────────────────────────────
        if social_person:
            name = f"understand_{social_person[:18]}"
            if name not in self._pool:
                atom = self._make_goal(
                    name=name,
                    source="social",
                    description=f"Understand interlocutor '{social_person}'",
                    priority=0.80,
                    salience=0.65,
                    evidence=f"social:{social_person}",
                    parent="respond",
                )
                new_goals.append(atom)

        # ── Source 6: Narrative patterns ─────────────────────────────
        if narrative_pattern:
            name = f"address_{narrative_pattern.replace(' ', '_')[:22]}"
            if name not in self._pool:
                atom = self._make_goal(
                    name=name,
                    source="narrative",
                    description=f"Address recurring pattern: {narrative_pattern}",
                    priority=0.75,
                    salience=0.5,
                    evidence=f"narrative:{narrative_pattern[:40]}",
                )
                new_goals.append(atom)

        # ── Register all new goals ────────────────────────────────────
        for atom in new_goals:
            self._pool[atom.name] = atom
            self._log(tick, "born", atom.name, atom.evidence)

        # ── Prune / mutate ────────────────────────────────────────────
        self._decay()
        if tick - self._last_mutation_tick >= MUTATION_INTERVAL:
            self._mutate_and_prune(tick)
            self._last_mutation_tick = tick

        return new_goals

    # ── Outcome feedback ─────────────────────────────────────────────────────

    def record_outcome(
        self,
        goal_name: str,
        success: bool,
        reward: float,
        tick: Optional[int] = None,
    ) -> None:
        """Update goal statistics from real execution outcome."""
        t = tick or self._tick
        atom = self._pool.get(goal_name)
        if atom is None:
            return
        atom.last_active_tick = t
        if success:
            atom.success_count += 1
            atom.fulfillment = min(1.0, atom.fulfillment + 0.15)
            atom.expected_reward = atom.expected_reward * 0.85 + reward * 0.15
            atom.salience = min(1.0, atom.salience + 0.1)
        else:
            atom.fail_count += 1
            atom.fulfillment = max(0.0, atom.fulfillment - 0.05)
            atom.salience = max(0.0, atom.salience - 0.05)

        # Auto-close on success threshold
        if atom.fulfillment >= GOAL_FULFILLMENT_THRESHOLD and atom.source != "base":
            self._close(atom, "fulfilled", t)

    # ── Queries ──────────────────────────────────────────────────────────────

    def top_goals(self, n: int = 8) -> List[GoalAtom]:
        """Return top-N active goals sorted by priority × salience."""
        active = [g for g in self._pool.values() if not g.closed]
        active.sort(key=lambda g: g.priority * g.salience, reverse=True)
        return active[:n]

    def active_goal_names(self) -> List[str]:
        return [g.name for g in self.top_goals()]

    def get_atom(self, name: str) -> Optional[GoalAtom]:
        return self._pool.get(name)

    def pool_summary(self) -> str:
        active = [g for g in self._pool.values() if not g.closed]
        closed = [g for g in self._pool.values() if g.closed]
        top = self.top_goals(3)
        top_str = ", ".join(f"{g.name}({g.priority:.1f})" for g in top)
        return (
            f"GOALS: pool={len(active)} active / {len(closed)} closed "
            f"| top=[{top_str}]"
        )

    # ── Hierarchy query ───────────────────────────────────────────────────────

    def children_of(self, parent_name: str) -> List[GoalAtom]:
        return [
            g for g in self._pool.values() if g.parent == parent_name and not g.closed
        ]

    def parent_of(self, child_name: str) -> Optional[GoalAtom]:
        atom = self._pool.get(child_name)
        if atom and atom.parent:
            return self._pool.get(atom.parent)
        return None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _make_goal(
        self,
        name: str,
        source: str,
        description: str,
        priority: float,
        salience: float,
        evidence: str = "",
        parent: Optional[str] = None,
    ) -> GoalAtom:
        return GoalAtom(
            name=name,
            source=source,
            description=description,
            priority=min(2.0, max(0.0, priority)),
            salience=min(1.0, max(0.0, salience)),
            created_tick=self._tick,
            last_active_tick=self._tick,
            evidence=evidence,
            parent=parent,
        )

    def _decay(self) -> None:
        for atom in self._pool.values():
            if not atom.closed and atom.source != "base":
                atom.salience = max(0.0, atom.salience - SALIENCE_DECAY_RATE)

    def _mutate_and_prune(self, tick: int) -> None:
        to_remove = []
        for name, atom in list(self._pool.items()):
            if atom.closed or atom.source == "base":
                continue

            # Bump priority for goals with recent success
            total = atom.success_count + atom.fail_count
            if total > 0:
                success_rate = atom.success_count / total
                new_priority = atom.priority * 0.9 + success_rate * 1.5 * 0.1
                if abs(new_priority - atom.priority) > 0.05:
                    atom.priority = max(0.1, min(2.0, new_priority))
                    self._log(
                        tick,
                        "mutated",
                        name,
                        f"priority→{atom.priority:.2f} "
                        f"({success_rate:.0%} success rate)",
                    )

            # Reject consistently failing goals
            if atom.fail_count >= FAIL_STREAK_CUTOFF and atom.success_count == 0:
                to_remove.append((name, "fail_streak"))
                continue

            # Prune stale low-salience goals
            age = tick - atom.last_active_tick
            if atom.salience < MIN_SALIENCE_TO_KEEP and age > 400:
                to_remove.append((name, "stale"))

        for name, reason in to_remove:
            atom = self._pool[name]
            self._close(atom, reason, tick)

        # Enforce pool size limit
        active = [g for g in self._pool.values() if not g.closed and g.source != "base"]
        if len(active) > MAX_POOL_SIZE - len(BASE_GOALS):
            active.sort(key=lambda g: g.priority * g.salience)
            for excess in active[: len(active) - (MAX_POOL_SIZE - len(BASE_GOALS))]:
                self._close(excess, "pool_cap", tick)

    def _close(self, atom: GoalAtom, reason: str, tick: int) -> None:
        atom.closed = True
        atom.closed_reason = reason
        self._log(
            tick,
            "fulfilled" if reason == "fulfilled" else "rejected",
            atom.name,
            reason,
        )

    def _log(self, tick: int, event_type: str, name: str, reason: str) -> None:
        self._event_log.append(
            GoalSynthesisEvent(
                tick=tick, event_type=event_type, goal_name=name, reason=reason
            )
        )

    def recent_events(self, n: int = 10) -> List[GoalSynthesisEvent]:
        return list(self._event_log)[-n:]

    # ── Goal-space probe for test suite ──────────────────────────────────────

    def open_goal_space_probe(self) -> Tuple[bool, str]:
        """
        Test: verify that the system can synthesize goals beyond the four
        base goals (i.e., that the goal space is actually open).
        """
        non_base = [
            g for g in self._pool.values() if not g.closed and g.source != "base"
        ]
        n_base = len(
            [g for g in self._pool.values() if g.source == "base" and not g.closed]
        )

        # Manually trigger synthesis with artificial signals
        self.synthesize(
            tick=self._tick + GENERATION_INTERVAL,
            prediction_error=0.8,
            surprise_entity="test_entity",
            drives=None,
            em=None,
            meta_gaps=["test_gap"],
            self_tensions=["agency_inconsistency"],
            social_person="test_person",
            narrative_pattern="failure_loop",
            world_entities=5,
        )
        non_base_after = [
            g for g in self._pool.values() if not g.closed and g.source != "base"
        ]

        new_count = len(non_base_after) - len(non_base)
        if new_count > 0 or len(non_base) > 0:
            return True, (
                f"PASS: goal space is open — {new_count} new goals synthesized "
                f"(total non-base active: {len(non_base_after)}, "
                f"base={n_base})"
            )
        return False, (
            f"FAIL: goal space is closed — no synthesized goals exist "
            f"(only base goals: {n_base})"
        )
