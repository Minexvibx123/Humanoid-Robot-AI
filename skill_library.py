"""
skill_library.py — Parametric Skill Layer (Schicht C)

Atomic, verifiable behaviours with:
  • Typed preconditions  (what must be true before starting)
  • Motor plan           (sequence of joint targets + timing)
  • Success metrics      (measurable completion criteria)
  • Abort conditions     (when to bail out)
  • Risk score           (safety cost estimate)
  • Episodic trace       (full log for learning)

Skills are the bridge between high-level intent (task_executive) and
low-level joint control (body_schema + safety_supervisor).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from body_schema import BodySchema, ProprioceptiveSummary
    from safety_supervisor import SafetySupervisor
    from telemetry_bus import TelemetryBus
    from world_state import WorldState


# ─────────────────────────────────────────────────────────────
# Skill lifecycle
# ─────────────────────────────────────────────────────────────


class SkillStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class SkillTrace:
    """Full episode record of one skill execution."""

    skill_name: str = ""
    intent: str = ""
    context: str = ""
    start_tick: int = 0
    end_tick: int = 0
    status: str = "idle"
    success_score: float = 0.0  # [0,1] measured completion
    risk_score: float = 0.0  # [0,1] assessed risk
    cost: float = 0.0  # energy/load cost
    abort_reason: str = ""
    notes: str = ""


@dataclass
class SkillResult:
    """Returned when a skill completes or aborts."""

    status: SkillStatus
    success_score: float = 0.0
    message: str = ""
    trace: Optional[SkillTrace] = None


# ─────────────────────────────────────────────────────────────
# Base Skill
# ─────────────────────────────────────────────────────────────


class Skill:
    """
    Abstract parametric skill.  Subclass and implement:
      _check_preconditions() → bool
      _plan()                → set up target sequence
      _step()                → advance one tick, return True when done
      _check_success()       → float [0,1]
      _check_abort()         → None or reason string

    Planning interface (for state-space planner):
      preconditions  → dict of {predicate: required_value}
      effects        → dict of {predicate: resulting_value}
      failure_types  → list of possible failure modes
    """

    name: str = "base_skill"
    risk_score: float = 0.1
    # Formal state-change interface for planner
    preconditions: Dict[str, object] = {}
    effects: Dict[str, object] = {}
    failure_types: List[str] = []

    def __init__(self) -> None:
        self.status: SkillStatus = SkillStatus.IDLE
        self._trace: SkillTrace = SkillTrace()
        self._start_tick: int = 0
        self._max_ticks: int = 200  # timeout

    def can_start(
        self, body: "BodySchema", world: "WorldState", safety: "SafetySupervisor"
    ) -> bool:
        """Check preconditions without side effects."""
        if not safety.is_safe_to_move():
            return False
        return self._check_preconditions(body, world, safety)

    def start(self, tick: int, intent: str = "", context: str = "") -> None:
        self.status = SkillStatus.RUNNING
        self._start_tick = tick
        self._trace = SkillTrace(
            skill_name=self.name,
            intent=intent,
            context=context,
            start_tick=tick,
            risk_score=self.risk_score,
        )
        self._plan()

    def tick(
        self,
        tick: int,
        body: "BodySchema",
        world: "WorldState",
        safety: "SafetySupervisor",
    ) -> Optional[SkillResult]:
        """Advance skill by one tick. Returns result when done."""
        if self.status != SkillStatus.RUNNING:
            return None

        # Check abort conditions
        abort = self._check_abort(body, world, safety)
        if abort:
            return self._finish(tick, SkillStatus.ABORTED, 0.0, abort)

        # Timeout
        if (tick - self._start_tick) > self._max_ticks:
            return self._finish(tick, SkillStatus.FAILED, 0.0, "timeout")

        # Step the skill
        done = self._step(tick, body, world, safety)
        if done:
            score = self._check_success(body, world)
            status = SkillStatus.SUCCEEDED if score > 0.4 else SkillStatus.FAILED
            return self._finish(tick, status, score, "")

        return None

    def _finish(
        self, tick: int, status: SkillStatus, score: float, reason: str
    ) -> SkillResult:
        self.status = status
        self._trace.end_tick = tick
        self._trace.status = status.value
        self._trace.success_score = score
        self._trace.abort_reason = reason
        self._trace.cost = (tick - self._start_tick) * 0.001  # simple cost model
        return SkillResult(
            status=status, success_score=score, message=reason, trace=self._trace
        )

    # ── Overridable hooks ────────────────────────────────────

    def _check_preconditions(
        self, body: "BodySchema", world: "WorldState", safety: "SafetySupervisor"
    ) -> bool:
        return True

    def _plan(self) -> None:
        pass

    def _step(
        self,
        tick: int,
        body: "BodySchema",
        world: "WorldState",
        safety: "SafetySupervisor",
    ) -> bool:
        return True  # done immediately

    def _check_success(self, body: "BodySchema", world: "WorldState") -> float:
        return 1.0

    def _check_abort(
        self, body: "BodySchema", world: "WorldState", safety: "SafetySupervisor"
    ) -> str:
        if safety.state.estop_active:
            return "estop"
        return ""


# ─────────────────────────────────────────────────────────────
# Concrete InMoov skills
# ─────────────────────────────────────────────────────────────


class FixatePerson(Skill):
    """Orient head + eyes toward the nearest detected person."""

    name = "fixate_person"
    risk_score = 0.05
    preconditions = {"person_visible": True}
    effects = {"head_aligned": True, "person_attentive": True}
    failure_types = ["person_lost", "head_stall"]

    def __init__(self) -> None:
        super().__init__()
        self._max_ticks = 60
        self._target_yaw = 90.0
        self._target_pitch = 90.0

    def _check_preconditions(self, body, world, safety):
        return world.zone.n_persons_visible > 0

    def _plan(self) -> None:
        pass  # target computed per tick from world state

    def _step(self, tick, body, world, safety):
        person = world.nearest_person()
        if person is None:
            return True  # no target → done
        desired_yaw = 90.0 + (person.center_x - 0.5) * 70.0
        desired_pitch = 90.0 + (0.5 - person.center_y) * 44.0
        body.joints["head_yaw"].set_target(desired_yaw)
        body.joints["head_pitch"].set_target(desired_pitch)
        body.joints["eye_yaw"].set_target(desired_yaw)
        body.joints["eye_pitch"].set_target(desired_pitch)
        # Done when error is small
        return body.chains["head"].total_error() < 0.05

    def _check_success(self, body, world):
        person = world.nearest_person()
        if person is None:
            return 0.3
        err = body.chains["head"].total_error()
        return max(0.0, 1.0 - err * 5.0)


class OrientHead(Skill):
    """Move head to a specific yaw/pitch target."""

    name = "orient_head"
    risk_score = 0.05
    preconditions = {}
    effects = {"head_aligned": True}
    failure_types = ["head_stall"]

    def __init__(self, yaw: float = 90.0, pitch: float = 90.0) -> None:
        super().__init__()
        self._max_ticks = 40
        self._yaw = yaw
        self._pitch = pitch

    def _plan(self) -> None:
        pass

    def _step(self, tick, body, world, safety):
        body.joints["head_yaw"].set_target(self._yaw)
        body.joints["head_pitch"].set_target(self._pitch)
        return body.chains["head"].total_error() < 0.03

    def _check_success(self, body, world):
        return max(0.0, 1.0 - body.chains["head"].total_error() * 10.0)


class SetPose(Skill):
    """Transition to a named full-body pose."""

    name = "set_pose"
    risk_score = 0.15
    preconditions = {"distance_safe": True}
    effects = {"body_idle": False, "gesture_active": True}
    failure_types = ["collision", "stall", "human_too_close"]

    POSES: Dict[str, Dict[str, float]] = {
        "idle": {
            "left_omoplate": 15,
            "right_omoplate": 15,
            "left_shoulder": 20,
            "right_shoulder": 20,
            "left_elbow": 30,
            "right_elbow": 30,
            "left_wrist": 90,
            "right_wrist": 90,
        },
        "attend": {
            "left_omoplate": 20,
            "right_omoplate": 20,
            "left_shoulder": 30,
            "right_shoulder": 30,
            "left_elbow": 45,
            "right_elbow": 45,
            "left_wrist": 90,
            "right_wrist": 90,
        },
        "gesture_ready": {
            "left_omoplate": 32,
            "right_omoplate": 32,
            "left_shoulder": 48,
            "right_shoulder": 48,
            "left_elbow": 58,
            "right_elbow": 58,
            "left_wrist": 90,
            "right_wrist": 90,
        },
        "offer_right": {
            "left_omoplate": 15,
            "right_omoplate": 40,
            "left_shoulder": 20,
            "right_shoulder": 60,
            "left_elbow": 30,
            "right_elbow": 70,
            "left_wrist": 90,
            "right_wrist": 100,
        },
    }

    def __init__(self, pose_name: str = "idle") -> None:
        super().__init__()
        self._pose_name = pose_name
        self._max_ticks = 80
        self._targets: Dict[str, float] = {}

    def _check_preconditions(self, body, world, safety):
        return self._pose_name in self.POSES

    def _plan(self) -> None:
        self._targets = dict(self.POSES.get(self._pose_name, {}))

    def _step(self, tick, body, world, safety):
        gated = safety.gate_targets(self._targets, body)
        for name, deg in gated.items():
            if name in body.joints:
                body.joints[name].set_target(deg)
        # Done when all arm joints are close to target
        errors = []
        for name in self._targets:
            j = body.joints.get(name)
            if j:
                errors.append(j.error)
        return all(e < 0.05 for e in errors) if errors else True

    def _check_success(self, body, world):
        errors = [body.joints[n].error for n in self._targets if n in body.joints]
        if not errors:
            return 1.0
        return max(0.0, 1.0 - sum(errors) / len(errors) * 5.0)


class OpenHand(Skill):
    """Open one or both hands."""

    name = "open_hand"
    risk_score = 0.05
    preconditions = {}
    effects = {"hand_free_right": True, "holds_object_right": False}
    failure_types = ["stall"]

    def __init__(self, side: str = "both") -> None:
        super().__init__()
        self._side = side
        self._max_ticks = 30

    def _plan(self) -> None:
        pass

    def _step(self, tick, body, world, safety):
        if self._side in ("left", "both"):
            body.joints["left_thumb"].set_target(20.0)
            body.joints["left_index"].set_target(20.0)
        if self._side in ("right", "both"):
            body.joints["right_thumb"].set_target(20.0)
            body.joints["right_index"].set_target(20.0)
        return True  # immediate

    def _check_success(self, body, world):
        return 1.0


class CloseHand(Skill):
    """Close one or both hands (grasp)."""

    name = "close_hand"
    risk_score = 0.1
    preconditions = {"hand_free_right": True, "object_reachable": True}
    effects = {"hand_free_right": False, "holds_object_right": True}
    failure_types = ["stall", "object_lost"]

    def __init__(self, side: str = "both") -> None:
        super().__init__()
        self._side = side
        self._max_ticks = 30

    def _plan(self) -> None:
        pass

    def _step(self, tick, body, world, safety):
        if self._side in ("left", "both"):
            body.joints["left_thumb"].set_target(130.0)
            body.joints["left_index"].set_target(130.0)
        if self._side in ("right", "both"):
            body.joints["right_thumb"].set_target(130.0)
            body.joints["right_index"].set_target(130.0)
        return True

    def _check_success(self, body, world):
        return 1.0


class ExpressEmotion(Skill):
    """Express an emotion through face/head movements."""

    name = "express_emotion"
    risk_score = 0.05
    preconditions = {}
    effects = {"gesture_active": True}
    failure_types = []

    _EXPRESSIONS: Dict[str, Dict[str, float]] = {
        "joy": {
            "jaw": 28.0,  # slight smile
            "left_upper_lid": 100.0,
            "right_upper_lid": 80.0,  # eyes wide
            "left_lower_lid": 80.0,
            "right_lower_lid": 100.0,
        },
        "surprise": {
            "jaw": 38.0,
            "left_upper_lid": 108.0,
            "right_upper_lid": 72.0,
            "left_lower_lid": 72.0,
            "right_lower_lid": 108.0,
        },
        "sadness": {
            "jaw": 16.0,
            "left_upper_lid": 85.0,
            "right_upper_lid": 95.0,
            "left_lower_lid": 95.0,
            "right_lower_lid": 85.0,
            "head_pitch": 95.0,  # look down slightly
        },
        "calm": {
            "jaw": 18.0,
            "left_upper_lid": 88.0,
            "right_upper_lid": 92.0,
            "left_lower_lid": 92.0,
            "right_lower_lid": 88.0,
        },
    }

    def __init__(self, emotion: str = "calm") -> None:
        super().__init__()
        self._emotion = emotion
        self._max_ticks = 50
        self._targets: Dict[str, float] = {}

    def _plan(self) -> None:
        self._targets = dict(
            self._EXPRESSIONS.get(self._emotion, self._EXPRESSIONS["calm"])
        )

    def _step(self, tick, body, world, safety):
        for name, deg in self._targets.items():
            if name in body.joints:
                body.joints[name].set_target(deg)
        return (tick - self._start_tick) > 15  # hold for 15 ticks minimum

    def _check_success(self, body, world):
        return 1.0  # expression is always "successful"


class MirrorGesture(Skill):
    """Mirror a detected human gesture with arms."""

    name = "mirror_gesture"
    risk_score = 0.2
    preconditions = {"person_visible": True, "distance_safe": True}
    effects = {"gesture_active": True}
    failure_types = ["human_too_close", "person_lost", "stall"]

    def __init__(self, gesture: str = "neutral", intensity: float = 0.5) -> None:
        super().__init__()
        self._gesture = gesture
        self._intensity = max(0.0, min(1.0, intensity))
        self._max_ticks = 60

    def _check_preconditions(self, body, world, safety):
        return safety.is_safe_to_move("left_shoulder")

    def _plan(self) -> None:
        i = self._intensity
        self._targets = {
            "left_omoplate": 22 + i * 18,
            "right_omoplate": 22 + i * 18,
            "left_shoulder": 30 + i * 35,
            "right_shoulder": 30 + i * 35,
            "left_elbow": 28 + i * 48,
            "right_elbow": 28 + i * 48,
        }

    def _step(self, tick, body, world, safety):
        gated = safety.gate_targets(self._targets, body)
        for name, deg in gated.items():
            if name in body.joints:
                body.joints[name].set_target(deg)
        return (tick - self._start_tick) > 30

    def _check_success(self, body, world):
        return max(0.0, 1.0 - body.chains["left_arm"].total_error() * 3.0)

    def _check_abort(self, body, world, safety):
        base = super()._check_abort(body, world, safety)
        if base:
            return base
        if safety.state.human_too_close:
            return "human_too_close"
        return ""


# ─────────────────────────────────────────────────────────────
# Skill Library (registry)
# ─────────────────────────────────────────────────────────────


class SkillLibrary:
    """
    Registry of all available skills.
    Executive layer requests skills by name with parameters.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, type] = {
            "fixate_person": FixatePerson,
            "orient_head": OrientHead,
            "set_pose": SetPose,
            "open_hand": OpenHand,
            "close_hand": CloseHand,
            "express_emotion": ExpressEmotion,
            "mirror_gesture": MirrorGesture,
        }
        self._active_skill: Optional[Skill] = None
        self._history: List[SkillTrace] = []
        self._success_stats: Dict[str, List[float]] = {}
        # Outcome-learned cost model: skill → EMA of actual tick cost
        self._learned_cost: Dict[str, float] = {}
        # Contextual success: (skill, intent) → success EMA
        self._contextual_success: Dict[tuple, float] = {}

    def available_skills(self) -> List[str]:
        return list(self._registry.keys())

    def create(self, name: str, **kwargs) -> Optional[Skill]:
        """Instantiate a skill by name with parameters."""
        cls = self._registry.get(name)
        if cls is None:
            return None
        return cls(**kwargs)

    @property
    def active_skill(self) -> Optional[Skill]:
        return self._active_skill

    @property
    def is_busy(self) -> bool:
        return (
            self._active_skill is not None
            and self._active_skill.status == SkillStatus.RUNNING
        )

    def start_skill(
        self, skill: Skill, tick: int, intent: str = "", context: str = ""
    ) -> bool:
        """Start a skill if no other is running."""
        if self.is_busy:
            return False
        self._active_skill = skill
        skill.start(tick, intent, context)
        return True

    def tick(
        self,
        tick: int,
        body: "BodySchema",
        world: "WorldState",
        safety: "SafetySupervisor",
    ) -> Optional[SkillResult]:
        """Advance the active skill. Returns result when done."""
        if self._active_skill is None:
            return None
        result = self._active_skill.tick(tick, body, world, safety)
        if result is not None:
            # Record history
            if result.trace:
                self._history.append(result.trace)
                if len(self._history) > 500:
                    self._history = self._history[-500:]
                # Update success stats
                name = result.trace.skill_name
                self._success_stats.setdefault(name, []).append(result.success_score)
                if len(self._success_stats[name]) > 100:
                    self._success_stats[name] = self._success_stats[name][-100:]
                # Outcome learning: update cost model
                actual_cost = result.trace.cost
                prev_cost = self._learned_cost.get(name, actual_cost)
                self._learned_cost[name] = prev_cost * 0.8 + actual_cost * 0.2
                # Contextual success: (skill, intent) pair
                ctx_key = (name, result.trace.intent)
                prev_ctx = self._contextual_success.get(ctx_key, result.success_score)
                self._contextual_success[ctx_key] = (
                    prev_ctx * 0.85 + result.success_score * 0.15
                )
            self._active_skill = None
        return result

    def abort_active(self, reason: str = "executive") -> Optional[SkillResult]:
        """Force-abort the currently running skill."""
        if self._active_skill and self._active_skill.status == SkillStatus.RUNNING:
            result = self._active_skill._finish(0, SkillStatus.ABORTED, 0.0, reason)
            self._active_skill = None
            if result.trace:
                self._history.append(result.trace)
            return result
        return None

    def success_rate(self, skill_name: str) -> float:
        """Average success score for a skill [0,1]."""
        scores = self._success_stats.get(skill_name, [])
        return sum(scores) / max(len(scores), 1)

    def learned_cost(self, skill_name: str) -> float:
        """EMA of actual tick cost for a skill. Returns 0.01 if unknown."""
        return self._learned_cost.get(skill_name, 0.01)

    def contextual_success(self, skill_name: str, intent: str) -> float:
        """Success rate for a specific (skill, intent) pair. 0.5 if unknown."""
        return self._contextual_success.get((skill_name, intent), 0.5)

    def last_traces(self, n: int = 10) -> List[SkillTrace]:
        return self._history[-n:]

    def describe(self) -> str:
        active = self._active_skill.name if self._active_skill else "none"
        status = self._active_skill.status.value if self._active_skill else "idle"
        rates = (
            " ".join(
                f"{name}:{self.success_rate(name):.2f}"
                for name in self._registry
                if name in self._success_stats
            )
            or "no data"
        )
        return f"skill active={active}({status}) history={len(self._history)} rates=[{rates}]"
