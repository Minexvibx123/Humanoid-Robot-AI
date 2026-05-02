"""
task_executive.py — Goal-Stack Executive Layer (Schicht D)

Bridges high-level consciousness goals to concrete skill sequences.
Responsibilities:
  • Goal decomposition   (intent → state-space planning OR recipe fallback)
  • Skill sequencing     (precondition chains, fallbacks)
  • Progress monitoring  (success evaluation per step)
  • Recovery             (re-plan on failure, escalate on repeated fail)
  • Priority arbitration (urgent interrupts override ongoing plans)
  • State-space planner  (plans from current predicates → goal predicates)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from causal_graph import TransitionRecord

if TYPE_CHECKING:
    from body_schema import BodySchema
    from safety_supervisor import SafetySupervisor
    from skill_library import Skill, SkillLibrary, SkillResult, SkillStatus
    from world_state import WorldState


# ─────────────────────────────────────────────────────────────
# Goal lifecycle
# ─────────────────────────────────────────────────────────────


class GoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GoalPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3  # interrupts current goal
    CRITICAL = 4  # safety / immediate


@dataclass
class GoalStep:
    """One step in a goal's skill plan."""

    skill_name: str
    skill_kwargs: Dict = field(default_factory=dict)
    required: bool = True  # if False, failure is tolerable
    max_retries: int = 1
    attempts: int = 0
    succeeded: bool = False
    failed: bool = False


@dataclass
class Goal:
    """A high-level goal with an ordered skill plan."""

    intent: str  # e.g. "greet_person", "give_object"
    context: str = ""  # free text situation context
    priority: GoalPriority = GoalPriority.NORMAL
    status: GoalStatus = GoalStatus.PENDING
    steps: List[GoalStep] = field(default_factory=list)
    current_step: int = 0
    created_tick: int = 0
    started_tick: int = 0
    ended_tick: int = 0
    result_msg: str = ""
    _retries: int = 0
    target_person: str = ""  # bound person reference
    target_object: str = ""  # bound object reference


@dataclass
class SkillEvent:
    """Per-step outcome: what happened during one skill execution.

    Produced by TaskExecutive every time a skill starts, completes,
    or fails.  Consumed tick-by-tick by Consciousness for fine-grained
    learning, narrative accumulation, and identity observation.
    """

    tick: int = 0
    goal_intent: str = ""  # parent goal intent
    skill_name: str = ""  # which skill ran
    step_index: int = 0  # position in goal step list
    status: str = ""  # started | succeeded | failed | retry
    success_score: float = 0.0  # [0,1] from SkillResult
    message: str = ""
    target_person: str = ""  # bound person id or ""
    target_object: str = ""  # bound object id or ""
    pre_predicates: Dict = field(default_factory=dict)
    post_predicates: Dict = field(default_factory=dict)


@dataclass
class ExecutiveOutcome:
    """Shared truth object: what actually happened when a goal was executed.

    Produced by TaskExecutive on goal completion/failure.
    Consumed by ConsciousnessCore, IdentityArc, NarrativeThread,
    ValueModel, CausalGraph, and LongHorizon as the single source
    of real-world action consequences.
    """

    tick: int = 0
    intent: str = ""  # goal intent that was attempted
    success: bool = False
    failure_cause: str = ""
    executed_skill: str = ""  # last skill that actually ran
    duration_ticks: int = 0
    steps_attempted: int = 0
    steps_succeeded: int = 0
    reward: float = 0.0  # [-1, 1] scalar outcome signal
    context_summary: str = ""
    target_person: str = ""  # bound person id
    target_object: str = ""  # bound object id


@dataclass
class Postmortem:
    """Post-hoc analysis of a completed/failed goal."""

    intent: str = ""
    tick: int = 0
    success: bool = False
    duration_ticks: int = 0
    steps_attempted: int = 0
    steps_succeeded: int = 0
    failure_cause: str = ""
    correction_rule: str = ""  # what to do differently next time
    context_summary: str = ""


# ─────────────────────────────────────────────────────────────
# Goal decomposition recipes
# ─────────────────────────────────────────────────────────────

GOAL_RECIPES: Dict[str, List[GoalStep]] = {
    "greet_person": [
        GoalStep("fixate_person"),
        GoalStep("express_emotion", {"emotion": "joy"}),
        GoalStep("set_pose", {"pose_name": "attend"}),
    ],
    "attend_speaker": [
        GoalStep("fixate_person"),
        GoalStep("express_emotion", {"emotion": "calm"}),
    ],
    "offer_handshake": [
        GoalStep("fixate_person"),
        GoalStep("set_pose", {"pose_name": "offer_right"}),
        GoalStep("open_hand", {"side": "right"}),
    ],
    "express_joy": [
        GoalStep("express_emotion", {"emotion": "joy"}),
        GoalStep("set_pose", {"pose_name": "gesture_ready"}, required=False),
    ],
    "express_surprise": [
        GoalStep("express_emotion", {"emotion": "surprise"}),
    ],
    "express_sadness": [
        GoalStep("express_emotion", {"emotion": "sadness"}),
    ],
    "idle_pose": [
        GoalStep("set_pose", {"pose_name": "idle"}),
        GoalStep("express_emotion", {"emotion": "calm"}),
    ],
    "mirror_human": [
        GoalStep("fixate_person"),
        GoalStep("mirror_gesture", {"gesture": "neutral", "intensity": 0.5}),
    ],
    "look_around": [
        GoalStep("orient_head", {"yaw": 60.0, "pitch": 90.0}),
        GoalStep("orient_head", {"yaw": 120.0, "pitch": 90.0}),
        GoalStep("orient_head", {"yaw": 90.0, "pitch": 90.0}),
    ],
    "release_object": [
        GoalStep("open_hand", {"side": "right"}),
        GoalStep("set_pose", {"pose_name": "attend"}),
    ],
    "grasp_object": [
        GoalStep("set_pose", {"pose_name": "offer_right"}),
        GoalStep("close_hand", {"side": "right"}),
    ],
    # Social meta-goals (Phase 4)
    "comfort_person": [
        GoalStep("fixate_person"),
        GoalStep("express_emotion", {"emotion": "calm"}),
        GoalStep("set_pose", {"pose_name": "attend"}),
    ],
    "inquire_person": [
        GoalStep("fixate_person"),
        GoalStep("express_emotion", {"emotion": "curiosity"}, required=False),
    ],
    "cooperate": [
        GoalStep("fixate_person"),
        GoalStep("set_pose", {"pose_name": "gesture_ready"}),
        GoalStep("express_emotion", {"emotion": "joy"}, required=False),
    ],
    "create_distance": [
        GoalStep("set_pose", {"pose_name": "idle"}),
        GoalStep("express_emotion", {"emotion": "calm"}),
    ],
}


# ─────────────────────────────────────────────────────────────
# Goal predicate targets (for state-space planner)
# ─────────────────────────────────────────────────────────────

GOAL_PREDICATES: Dict[str, Dict[str, object]] = {
    "greet_person": {
        "head_aligned": True,
        "person_attentive": True,
        "gesture_active": True,
    },
    "attend_speaker": {
        "head_aligned": True,
        "person_attentive": True,
    },
    "offer_handshake": {
        "head_aligned": True,
        "person_attentive": True,
        "hand_free_right": True,
        "gesture_active": True,
    },
    "grasp_object": {
        "holds_object_right": True,
        "hand_free_right": False,
    },
    "release_object": {
        "hand_free_right": True,
        "holds_object_right": False,
    },
    "idle_pose": {
        "body_idle": True,
        "gesture_active": False,
    },
    "mirror_human": {
        "head_aligned": True,
        "gesture_active": True,
    },
    "comfort_person": {
        "person_attentive": True,
        "gesture_active": True,
    },
    "inquire_person": {
        "head_aligned": True,
        "person_attentive": True,
    },
    "cooperate": {
        "head_aligned": True,
        "person_attentive": True,
        "gesture_active": True,
    },
    "create_distance": {
        "body_idle": True,
        "gesture_active": False,
    },
}


# ─────────────────────────────────────────────────────────────
# State-Space Planner
# ─────────────────────────────────────────────────────────────


class StatePlanner:
    """
    Plans skill sequences from world predicates → goal predicates.
    Uses forward search: finds skills whose effects move current state
    closer to the goal state. Falls back to GOAL_RECIPES if no plan found.
    """

    MAX_PLAN_DEPTH = 8

    def __init__(
        self, skill_lib: "SkillLibrary", causal_graph: "Optional[Any]" = None
    ) -> None:
        self._skill_lib = skill_lib
        self._causal_graph = causal_graph

    def plan(
        self,
        intent: str,
        world: "WorldState",
        capability_confidence=None,
        target_id: str = "",
    ) -> Optional[List[GoalStep]]:
        """
        Generate a skill sequence to achieve the goal predicates.
        Returns None if no plan found (caller should use recipe fallback).
        capability_confidence: optional callable(skill_name) → [0,1]
        """
        goal_preds = GOAL_PREDICATES.get(intent)
        if goal_preds is None:
            return None

        current = self._read_predicates(world)
        unsatisfied = {k: v for k, v in goal_preds.items() if current.get(k) != v}
        if not unsatisfied:
            return []  # goal already met

        plan: List[GoalStep] = []
        used_skills: set = set()
        for _depth in range(self.MAX_PLAN_DEPTH):
            if not unsatisfied:
                break
            best_skill = self._find_best_skill(
                unsatisfied,
                current,
                used_skills,
                intent,
                capability_confidence,
                target_id=target_id,
            )
            if best_skill is None:
                break
            plan.append(
                GoalStep(
                    skill_name=best_skill.name,
                    skill_kwargs=self._default_kwargs(best_skill.name, intent),
                )
            )
            for k, v in best_skill.effects.items():
                current[k] = v
                if k in unsatisfied and unsatisfied[k] == v:
                    del unsatisfied[k]
            used_skills.add(best_skill.name)

        if unsatisfied:
            return None
        return plan

    def _find_best_skill(
        self,
        unsatisfied: Dict[str, object],
        current: Dict[str, object],
        used: set,
        intent: str = "",
        capability_confidence=None,
        target_id: str = "",
    ) -> Optional["Skill"]:
        """Find the unused skill that best resolves unsatisfied predicates.
        Uses outcome-learned cost, contextual success, capability model
        confidence, and causal graph skill-tier predictions for adaptive planning."""
        best = None
        best_score = 0.0
        for name in self._skill_lib.available_skills():
            if name in used:
                continue
            skill = self._skill_lib.create(name)
            if skill is None:
                continue
            preconds_met = all(
                current.get(k) == v for k, v in skill.preconditions.items()
            )
            if not preconds_met:
                continue
            score = sum(
                1
                for k, v in skill.effects.items()
                if k in unsatisfied and unsatisfied[k] == v
            )
            if score <= 0:
                continue
            # Adaptive scoring: learned success, cost, and self-confidence
            ctx_success = self._skill_lib.contextual_success(name, intent)
            cost = self._skill_lib.learned_cost(name)
            hist_bonus = ctx_success * 0.4
            cost_penalty = min(0.3, cost * 3.0)
            risk_penalty = skill.risk_score * 0.2
            # Capability model: boost skills we know we're good at,
            # penalise ones we've failed at before
            cap_mod = 0.0
            if capability_confidence is not None:
                cap_conf = capability_confidence(name)
                cap_mod = (cap_conf - 0.5) * 0.3  # [-0.15, +0.15]
            # ── Causal graph: skill-tier prediction ──
            # Query person- and context-specific causal experience
            causal_mod = 0.0
            if self._causal_graph is not None:
                try:
                    _state_sig = f"{intent}:{target_id}" if target_id else intent
                    _causal_pred = self._causal_graph.predict_skill_success(
                        _state_sig, name
                    )
                    if _causal_pred is not None:
                        # _causal_pred is (success_rate, n_samples)
                        _c_rate, _c_n = _causal_pred
                        # Weight by sample count (diminishing returns)
                        import math

                        _c_weight = min(0.35, 0.15 + 0.05 * math.log1p(_c_n))
                        causal_mod = (_c_rate - 0.5) * _c_weight
                except Exception:
                    pass
            total = (
                score + hist_bonus - cost_penalty - risk_penalty + cap_mod + causal_mod
            )
            if total > best_score:
                best_score = total
                best = skill
        return best

    def _read_predicates(self, world: "WorldState") -> Dict[str, object]:
        """Read current world predicates as a flat dict."""
        p = world.predicates
        return {
            "person_visible": p.person_visible,
            "person_attentive": p.person_attentive,
            "person_speaking": p.person_speaking,
            "hand_free_left": p.hand_free_left,
            "hand_free_right": p.hand_free_right,
            "holds_object_left": p.holds_object_left,
            "holds_object_right": p.holds_object_right,
            "distance_safe": p.distance_safe,
            "object_reachable": p.object_reachable,
            "head_aligned": p.head_aligned,
            "body_idle": p.body_idle,
            "gesture_active": p.gesture_active,
            "robot_speaking": p.robot_speaking,
            "greeting_done": p.greeting_done,
        }

    def _default_kwargs(self, skill_name: str, intent: str) -> Dict:
        """Provide sensible default kwargs based on intent context."""
        if skill_name == "express_emotion":
            emotion_map = {
                "greet_person": "joy",
                "comfort_person": "calm",
                "express_joy": "joy",
                "express_sadness": "sadness",
                "express_surprise": "surprise",
                "inquire_person": "curiosity",
            }
            return {"emotion": emotion_map.get(intent, "calm")}
        if skill_name == "set_pose":
            pose_map = {
                "greet_person": "attend",
                "offer_handshake": "offer_right",
                "idle_pose": "idle",
                "mirror_human": "gesture_ready",
                "comfort_person": "attend",
                "cooperate": "gesture_ready",
                "create_distance": "idle",
            }
            return {"pose_name": pose_map.get(intent, "attend")}
        if skill_name == "open_hand":
            return {"side": "right"}
        if skill_name == "close_hand":
            return {"side": "right"}
        if skill_name == "mirror_gesture":
            return {"gesture": "neutral", "intensity": 0.5}
        return {}


# ─────────────────────────────────────────────────────────────
# Reflexive behaviours (always-on, bypass goal stack)
# ─────────────────────────────────────────────────────────────


class ReflexLayer:
    """Immediate reactive behaviours outside conscious control."""

    def __init__(self, skill_lib: "SkillLibrary") -> None:
        self._skill_lib = skill_lib
        self._last_reflex_tick = 0
        self._REFLEX_COOLDOWN = 40  # ticks between reflexes

    def evaluate(
        self,
        tick: int,
        body: "BodySchema",
        world: "WorldState",
        safety: "SafetySupervisor",
    ) -> Optional[str]:
        """Return a reflex goal intent, or None."""
        if (tick - self._last_reflex_tick) < self._REFLEX_COOLDOWN:
            return None

        # E-stop → freeze (handled by safety, no skill needed)
        if safety.state.estop_active:
            return None

        # Person suddenly very close → attend
        person = world.nearest_person()
        if person and person.distance_cm < 40 and not person.face_visible:
            self._last_reflex_tick = tick
            return "attend_speaker"

        return None


# ─────────────────────────────────────────────────────────────
# Task Executive
# ─────────────────────────────────────────────────────────────


class TaskExecutive:
    """
    Goal-stack executive:
      • Accepts goals from consciousness (submit_goal)
      • Decomposes into skill sequences using GOAL_RECIPES
      • Sequences and monitors skill execution
      • Handles failure with retry/fallback
      • Provides status back to consciousness
    """

    MAX_GOAL_QUEUE = 10
    MAX_HISTORY = 200

    def __init__(self, skill_lib: "SkillLibrary") -> None:
        self._skill_lib = skill_lib
        self._reflex = ReflexLayer(skill_lib)
        self._planner = StatePlanner(skill_lib)
        self._goal_queue: deque[Goal] = deque(maxlen=self.MAX_GOAL_QUEUE)
        self._active_goal: Optional[Goal] = None
        self._history: List[Goal] = []
        self._plans_generated: int = 0  # counter for planner-generated plans
        self._recipes_used: int = 0  # counter for recipe fallbacks
        self._postmortems: List[Postmortem] = []
        self.MAX_POSTMORTEMS = 100
        self._capability_fn = None  # optional: callable(skill_name) → [0,1]
        self._causal_graph = None  # set by brain after consciousness init
        # Outcome queue: real action results for consciousness consumption
        self._outcome_queue: List[ExecutiveOutcome] = []
        # Step event queue: per-skill fine-grained events
        self._step_event_queue: List[SkillEvent] = []
        # Snapshot of world predicates before current skill started
        self._pre_step_predicates: Dict = {}

    def set_capability_model(self, fn) -> None:
        """Set a capability confidence function from consciousness."""
        self._capability_fn = fn

    # ── Public API ────────────────────────────────────────────

    def submit_goal(
        self,
        intent: str,
        context: str = "",
        priority: GoalPriority = GoalPriority.NORMAL,
        tick: int = 0,
        world: Optional["WorldState"] = None,
        capability_confidence=None,
        target_person: str = "",
        target_object: str = "",
    ) -> bool:
        """Submit a new goal. Uses state planner first, recipe fallback.
        capability_confidence: optional callable(skill_name) → [0,1]."""
        # ── Deduplication: reject if same intent+target already queued/active ──
        if (
            self._active_goal
            and self._active_goal.intent == intent
            and self._active_goal.target_person == target_person
            and self._active_goal.target_object == target_object
        ):
            return False
        for qg in self._goal_queue:
            if (
                qg.intent == intent
                and qg.target_person == target_person
                and qg.target_object == target_object
            ):
                return False
        # Try state-space planner first
        steps = None
        _cap_fn = capability_confidence or self._capability_fn
        if world is not None:
            # Sync causal graph reference into planner
            self._planner._causal_graph = self._causal_graph
            _target = target_person or target_object or ""
            planned = self._planner.plan(intent, world, _cap_fn, target_id=_target)
            if planned is not None:
                steps = planned
                self._plans_generated += 1

        # Fallback to recipes
        if steps is None:
            if intent not in GOAL_RECIPES:
                return False
            steps = [
                GoalStep(s.skill_name, dict(s.skill_kwargs), s.required, s.max_retries)
                for s in GOAL_RECIPES[intent]
            ]
            self._recipes_used += 1

        goal = Goal(
            intent=intent,
            context=context,
            priority=priority,
            created_tick=tick,
            steps=steps,
            target_person=target_person,
            target_object=target_object,
        )
        # Urgent/critical goals preempt current
        if priority.value >= GoalPriority.URGENT.value and self._active_goal:
            self._cancel_active("preempted by " + intent)
        self._goal_queue.append(goal)
        self._sort_queue()
        return True

    def cancel_active(self, reason: str = "user") -> None:
        self._cancel_active(reason)

    @property
    def active_goal(self) -> Optional[Goal]:
        return self._active_goal

    @property
    def is_busy(self) -> bool:
        return self._active_goal is not None

    @property
    def queue_size(self) -> int:
        return len(self._goal_queue)

    def available_goals(self) -> List[str]:
        return list(set(list(GOAL_RECIPES.keys()) + list(GOAL_PREDICATES.keys())))

    def current_skill_name(self) -> str:
        """Return the name of the currently executing skill, or ''."""
        sk = self._skill_lib.active_skill
        if sk is not None:
            return sk.name
        return ""

    def drain_step_events(self) -> List["SkillEvent"]:
        """Return and clear all pending SkillEvent objects."""
        out = list(self._step_event_queue)
        self._step_event_queue.clear()
        return out

    def _snap_predicates(self, world: "WorldState") -> Dict:
        """Snapshot current world predicates as a flat dict."""
        p = world.predicates
        return {
            "person_visible": p.person_visible,
            "person_attentive": p.person_attentive,
            "head_aligned": p.head_aligned,
            "gesture_active": p.gesture_active,
            "body_idle": p.body_idle,
            "distance_safe": p.distance_safe,
        }

    def _check_target_alive(
        self, goal: Goal, tick: int, world: "WorldState"
    ) -> Optional[str]:
        """If the goal's bound target has disappeared, fail the goal."""
        if goal.target_person and goal.target_person not in world.persons:
            return self._fail_goal(goal, tick, f"target_lost:{goal.target_person}")
        return None

    # ── Tick ──────────────────────────────────────────────────

    def tick(
        self,
        tick: int,
        body: "BodySchema",
        world: "WorldState",
        safety: "SafetySupervisor",
    ) -> Optional[str]:
        """
        Advance executive by one tick. Returns status message or None.
        """
        # 1) Safety gate
        if safety.state.estop_active:
            if self._active_goal:
                self._cancel_active("estop")
            return "estop_active"

        # 2) Check reflexes (bypass goal stack)
        reflex = self._reflex.evaluate(tick, body, world, safety)
        if reflex and not self.is_busy:
            self.submit_goal(reflex, "reflex", GoalPriority.HIGH, tick)

        # 3) If no active goal, pick from queue
        if not self._active_goal:
            if not self._goal_queue:
                return None
            # Purge stale-target goals from queue before picking
            while self._goal_queue:
                cand = self._goal_queue[0]
                if cand.target_person and cand.target_person not in world.persons:
                    self._goal_queue.popleft()
                    cand.status = GoalStatus.FAILED
                    cand.result_msg = f"target_lost:{cand.target_person}"
                    self._archive(cand)
                    self._outcome_queue.append(
                        ExecutiveOutcome(
                            tick=tick,
                            intent=cand.intent,
                            success=False,
                            failure_cause=cand.result_msg,
                            reward=-0.3,
                            target_person=cand.target_person,
                            target_object=cand.target_object,
                            context_summary=cand.context[:80],
                        )
                    )
                    continue
                break
            if not self._goal_queue:
                return None
            self._active_goal = self._goal_queue.popleft()
            self._active_goal.status = GoalStatus.ACTIVE
            self._active_goal.started_tick = tick

        goal = self._active_goal

        # 3b) Stale target check on active goal
        _stale = self._check_target_alive(goal, tick, world)
        if _stale is not None:
            return _stale

        # 4) If skill library is busy, let it run
        result = self._skill_lib.tick(tick, body, world, safety)

        # 5) Handle skill completion → emit SkillEvent
        if result is not None:
            step = goal.steps[goal.current_step]
            step.attempts += 1
            _post_preds = self._snap_predicates(world)
            from skill_library import SkillStatus as SS

            # 3.10 — Failure type taxonomy: classify failure before retry logic
            # Unrecoverable types skip all retries; safety types abort entire goal.
            _UNRECOVERABLE = {"person_lost", "object_lost"}
            _SAFETY_ABORT = {"collision", "human_too_close"}
            # Look up failure_types from the skill class (active_skill is None by now)
            _skill_cls = self._skill_lib._registry.get(step.skill_name)
            _failure_types: set = set(
                getattr(_skill_cls, "failure_types", [])
                if _skill_cls is not None
                else []
            )
            _msg_lower = result.message.lower()
            _detected_type = ""
            for _ft in _failure_types:
                if _ft.replace("_", " ") in _msg_lower or _ft in _msg_lower:
                    _detected_type = _ft
                    break

            if result.status == SS.SUCCEEDED:
                step.succeeded = True
                self._step_event_queue.append(
                    SkillEvent(
                        tick=tick,
                        goal_intent=goal.intent,
                        skill_name=step.skill_name,
                        step_index=goal.current_step,
                        status="succeeded",
                        success_score=result.success_score,
                        message=result.message,
                        target_person=goal.target_person,
                        target_object=goal.target_object,
                        pre_predicates=dict(self._pre_step_predicates),
                        post_predicates=_post_preds,
                    )
                )
                goal.current_step += 1
            elif _detected_type in _SAFETY_ABORT:
                # Safety failure: abort entire goal immediately, no retry
                step.failed = True
                self._step_event_queue.append(
                    SkillEvent(
                        tick=tick,
                        goal_intent=goal.intent,
                        skill_name=step.skill_name,
                        step_index=goal.current_step,
                        status="failed",
                        message=f"safety_abort:{_detected_type}",
                        target_person=goal.target_person,
                        target_object=goal.target_object,
                        pre_predicates=dict(self._pre_step_predicates),
                        post_predicates=_post_preds,
                    )
                )
                return self._fail_goal(
                    goal, tick, f"safety:{_detected_type} in {step.skill_name}"
                )
            elif _detected_type in _UNRECOVERABLE and step.required:
                # Unrecoverable: skip retries, emit target_lost if person/object gone
                step.failed = True
                _fail_reason = f"target_lost:{_detected_type}:{step.skill_name}"
                self._step_event_queue.append(
                    SkillEvent(
                        tick=tick,
                        goal_intent=goal.intent,
                        skill_name=step.skill_name,
                        step_index=goal.current_step,
                        status="failed",
                        message=_fail_reason,
                        target_person=goal.target_person,
                        target_object=goal.target_object,
                        pre_predicates=dict(self._pre_step_predicates),
                        post_predicates=_post_preds,
                    )
                )
                return self._fail_goal(goal, tick, _fail_reason)
            else:
                # ── World-model cost gate: low-confidence skills get fewer retries ──
                _eff_retries = step.max_retries
                _srate = self._skill_lib.success_rate(step.skill_name)
                if _srate < 0.3 and _eff_retries > 1:
                    _eff_retries = max(1, _eff_retries - 1)  # give up one retry sooner
                if step.attempts < _eff_retries:
                    self._step_event_queue.append(
                        SkillEvent(
                            tick=tick,
                            goal_intent=goal.intent,
                            skill_name=step.skill_name,
                            step_index=goal.current_step,
                            status="retry",
                            message=result.message,
                            target_person=goal.target_person,
                            target_object=goal.target_object,
                        )
                    )
                elif step.required:
                    step.failed = True
                    self._step_event_queue.append(
                        SkillEvent(
                            tick=tick,
                            goal_intent=goal.intent,
                            skill_name=step.skill_name,
                            step_index=goal.current_step,
                            status="failed",
                            success_score=result.success_score,
                            message=result.message,
                            target_person=goal.target_person,
                            target_object=goal.target_object,
                            pre_predicates=dict(self._pre_step_predicates),
                            post_predicates=_post_preds,
                        )
                    )
                    return self._fail_goal(
                        goal, tick, f"step {step.skill_name} failed: {result.message}"
                    )
                else:
                    step.failed = True
                    self._step_event_queue.append(
                        SkillEvent(
                            tick=tick,
                            goal_intent=goal.intent,
                            skill_name=step.skill_name,
                            step_index=goal.current_step,
                            status="failed",
                            message=f"optional skip: {result.message}",
                            target_person=goal.target_person,
                            target_object=goal.target_object,
                        )
                    )
                    goal.current_step += 1  # skip optional step

        # 6) If all steps done → succeed
        if goal.current_step >= len(goal.steps):
            return self._succeed_goal(goal, tick)

        # 7) Start next skill if library is idle → emit SkillEvent(started)
        if not self._skill_lib.is_busy:
            step = goal.steps[goal.current_step]
            skill = self._skill_lib.create(step.skill_name, **step.skill_kwargs)
            if skill is None:
                return self._fail_goal(goal, tick, f"unknown skill {step.skill_name}")
            if not skill.can_start(body, world, safety):
                step.attempts += 1
                if step.attempts >= step.max_retries + 3:
                    if step.required:
                        return self._fail_goal(
                            goal, tick, f"preconditions not met for {step.skill_name}"
                        )
                    else:
                        goal.current_step += 1
                return f"waiting:{step.skill_name}"

            # Snapshot predicates before skill begins
            self._pre_step_predicates = self._snap_predicates(world)
            self._skill_lib.start_skill(skill, tick, goal.intent, goal.context)
            self._step_event_queue.append(
                SkillEvent(
                    tick=tick,
                    goal_intent=goal.intent,
                    skill_name=step.skill_name,
                    step_index=goal.current_step,
                    status="started",
                    target_person=goal.target_person,
                    target_object=goal.target_object,
                    pre_predicates=dict(self._pre_step_predicates),
                )
            )

        return f"running:{goal.intent}:{goal.current_step}/{len(goal.steps)}"

    # ── Internal ──────────────────────────────────────────────

    def _succeed_goal(self, goal: Goal, tick: int) -> str:
        goal.status = GoalStatus.SUCCEEDED
        goal.ended_tick = tick
        goal.result_msg = "all steps completed"
        self._archive(goal)
        self._generate_postmortem(goal, tick, True)
        # Build ExecutiveOutcome for consciousness
        _duration = tick - goal.started_tick if goal.started_tick > 0 else 0
        _last_skill = goal.steps[-1].skill_name if goal.steps else ""
        _steps_ok = sum(1 for s in goal.steps if s.succeeded)
        self._outcome_queue.append(
            ExecutiveOutcome(
                tick=tick,
                intent=goal.intent,
                success=True,
                executed_skill=_last_skill,
                duration_ticks=_duration,
                steps_attempted=len(goal.steps),
                steps_succeeded=_steps_ok,
                reward=1.0,
                context_summary=goal.context[:80],
                target_person=goal.target_person,
                target_object=goal.target_object,
            )
        )
        # Record success into causal graph
        if self._causal_graph is not None:
            self._causal_graph.record_goal_transition(
                TransitionRecord(
                    tick=tick,
                    state_signature=f"goal:{goal.intent}",
                    action_kind=goal.intent,
                    action_args={"context": goal.context[:40]},
                    predicted_outcome="success",
                    observed_outcome="success",
                    reward=1.0,
                    surprise=0.0,
                    success=True,
                )
            )
        self._active_goal = None
        return f"goal_done:{goal.intent}"

    def _fail_goal(self, goal: Goal, tick: int, reason: str) -> str:
        goal.status = GoalStatus.FAILED
        goal.ended_tick = tick
        goal.result_msg = reason
        self._archive(goal)
        self._generate_postmortem(goal, tick, False, reason)
        # Build ExecutiveOutcome for consciousness
        _duration = tick - goal.started_tick if goal.started_tick > 0 else 0
        _last_skill = (
            goal.steps[goal.current_step].skill_name
            if goal.current_step < len(goal.steps)
            else ""
        )
        _steps_ok = sum(1 for s in goal.steps if s.succeeded)
        self._outcome_queue.append(
            ExecutiveOutcome(
                tick=tick,
                intent=goal.intent,
                success=False,
                failure_cause=reason,
                executed_skill=_last_skill,
                duration_ticks=_duration,
                steps_attempted=sum(1 for s in goal.steps if s.attempts > 0),
                steps_succeeded=_steps_ok,
                reward=-0.5,
                context_summary=goal.context[:80],
                target_person=goal.target_person,
                target_object=goal.target_object,
            )
        )
        # Record failure into causal graph
        if self._causal_graph is not None:
            self._causal_graph.record_goal_transition(
                TransitionRecord(
                    tick=tick,
                    state_signature=f"goal:{goal.intent}",
                    action_kind=goal.intent,
                    action_args={"reason": reason[:40]},
                    predicted_outcome="success",
                    observed_outcome=f"failed:{reason[:30]}",
                    reward=-0.5,
                    surprise=0.8,
                    success=False,
                )
            )
        self._active_goal = None
        return f"goal_failed:{goal.intent}:{reason}"

    def _generate_postmortem(
        self, goal: Goal, tick: int, success: bool, failure_cause: str = ""
    ) -> None:
        """Create a postmortem analysis for consciousness consumption."""
        steps_ok = sum(1 for s in goal.steps if s.succeeded)
        steps_tried = sum(1 for s in goal.steps if s.attempts > 0)
        duration = tick - goal.started_tick if goal.started_tick > 0 else 0

        # Derive correction rule from failure pattern
        correction = ""
        if not success:
            if "preconditions" in failure_cause:
                correction = "check_preconditions_before_planning"
            elif "unknown skill" in failure_cause:
                correction = "verify_skill_availability"
            elif "timeout" in failure_cause.lower():
                correction = "increase_patience_or_simplify"
            elif steps_ok > 0:
                correction = "partial_success:refine_later_steps"
            else:
                correction = "reconsider_approach"

        pm = Postmortem(
            intent=goal.intent,
            tick=tick,
            success=success,
            duration_ticks=duration,
            steps_attempted=steps_tried,
            steps_succeeded=steps_ok,
            failure_cause=failure_cause,
            correction_rule=correction,
            context_summary=goal.context[:80],
        )
        self._postmortems.append(pm)
        if len(self._postmortems) > self.MAX_POSTMORTEMS:
            self._postmortems = self._postmortems[-self.MAX_POSTMORTEMS :]

    def recent_postmortems(self, n: int = 5) -> List[Postmortem]:
        """Return most recent postmortems for consciousness."""
        return self._postmortems[-n:]

    def drain_outcomes(self) -> List[ExecutiveOutcome]:
        """Return and clear all pending ExecutiveOutcome objects."""
        out = list(self._outcome_queue)
        self._outcome_queue.clear()
        return out

    def _cancel_active(self, reason: str) -> None:
        if self._active_goal:
            self._active_goal.status = GoalStatus.CANCELLED
            self._active_goal.result_msg = reason
            self._archive(self._active_goal)
            self._skill_lib.abort_active(reason)
            self._active_goal = None

    def _archive(self, goal: Goal) -> None:
        self._history.append(goal)
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY :]

    def _sort_queue(self) -> None:
        items = list(self._goal_queue)
        items.sort(key=lambda g: g.priority.value, reverse=True)
        self._goal_queue.clear()
        self._goal_queue.extend(items)

    # ── Status ────────────────────────────────────────────────

    def describe(self) -> str:
        active = (
            f"{self._active_goal.intent} "
            f"step {self._active_goal.current_step}/"
            f"{len(self._active_goal.steps)}"
            if self._active_goal
            else "idle"
        )
        return (
            f"exec active={active} queue={len(self._goal_queue)} "
            f"done={len(self._history)} planned={self._plans_generated} "
            f"recipes={self._recipes_used}"
        )

    def recent_goals(self, n: int = 5) -> List[Dict]:
        """Return recent goal summaries for consciousness."""
        out = []
        for g in self._history[-n:]:
            out.append(
                {
                    "intent": g.intent,
                    "status": g.status.value,
                    "steps": len(g.steps),
                    "msg": g.result_msg,
                }
            )
        return out
