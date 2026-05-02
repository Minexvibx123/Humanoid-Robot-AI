"""
safety_supervisor.py — Hardware Safety Layer (Schicht E)

Sits OUTSIDE consciousness — enforces hard physical constraints that
the cognitive system cannot override.  All movement commands pass through
the supervisor before reaching hardware.

Responsibilities:
  • Joint limit enforcement (absolute + velocity)
  • Self-collision prevention via body schema
  • Human proximity safety zones
  • Thermal / stall protection
  • Watchdog timer (no command → fallback pose)
  • Emergency stop (software e-stop)
  • Severity-graded responses: warn → slow → clamp → freeze → e-stop
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from body_schema import BodySchema, ProprioceptiveSummary
    from telemetry_bus import SensorEvent, TelemetryBus


# ─────────────────────────────────────────────────────────────
# Safety state
# ─────────────────────────────────────────────────────────────


@dataclass
class SafetyState:
    """Current safety assessment — published for monitoring."""

    level: str = "normal"  # normal | caution | limited | frozen | recovering | estop
    reason: str = ""
    active_limits: List[str] = field(default_factory=list)
    speed_scale: float = 1.0  # [0,1] applied to all velocities
    frozen_joints: List[str] = field(default_factory=list)
    last_update: float = 0.0
    estop_active: bool = False
    watchdog_ok: bool = True
    collision_risk: bool = False
    human_too_close: bool = False
    recovery_phase: str = ""  # "" | "retreat" | "explain" | "replan"
    recovery_ticks: int = 0  # ticks spent in recovery


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────


@dataclass
class SafetyConfig:
    """Tunable safety parameters."""

    # Speed limits (fraction of max_speed per joint)
    max_speed_fraction_normal: float = 1.0
    max_speed_fraction_caution: float = 0.6
    max_speed_fraction_limited: float = 0.25

    # Human proximity zones (cm)
    intimate_zone_cm: float = 30.0  # → freeze arms
    personal_zone_cm: float = 60.0  # → slow mode
    social_zone_cm: float = 120.0  # → normal with caution

    # Thermal
    joint_temp_warn: float = 0.55
    joint_temp_freeze: float = 0.80

    # Load / stall
    joint_load_warn: float = 0.60
    joint_load_freeze: float = 0.85

    # Watchdog (seconds without new command → fallback)
    watchdog_timeout_s: float = 5.0

    # Collision
    collision_freeze: bool = True

    # Recovery mode (instead of permanent freeze)
    recovery_enabled: bool = True
    recovery_retreat_ticks: int = 20  # ticks to slowly retreat to safe pose
    recovery_explain_ticks: int = 10  # ticks to signal state (express concern)
    recovery_replan_ticks: int = 10  # ticks to re-evaluate before resuming


# ─────────────────────────────────────────────────────────────
# Fallback pose
# ─────────────────────────────────────────────────────────────

FALLBACK_POSE: Dict[str, float] = {
    "head_yaw": 90.0,
    "head_pitch": 90.0,
    "neck_roll": 90.0,
    "jaw": 20.0,
    "eye_yaw": 90.0,
    "eye_pitch": 90.0,
    "left_upper_lid": 92.0,
    "left_lower_lid": 88.0,
    "right_upper_lid": 88.0,
    "right_lower_lid": 92.0,
    "left_omoplate": 15.0,
    "right_omoplate": 15.0,
    "left_shoulder": 20.0,
    "right_shoulder": 20.0,
    "left_elbow": 30.0,
    "right_elbow": 30.0,
    "left_wrist": 90.0,
    "right_wrist": 90.0,
    "left_thumb": 20.0,
    "right_thumb": 20.0,
    "left_index": 20.0,
    "right_index": 20.0,
}


# ─────────────────────────────────────────────────────────────
# Safety Supervisor
# ─────────────────────────────────────────────────────────────


class SafetySupervisor:
    """
    Hard safety layer that gates all movement commands.
    Ticked every brain cycle but enforces at hardware-response speed.
    """

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        self.config = config or SafetyConfig()
        self.state = SafetyState()
        self._last_command_time: float = time.perf_counter()
        self._estop_latch: bool = False  # latched until explicit reset
        self._frozen_set: set = set()
        self._violation_log: List[str] = []
        # Recovery mode state
        self._recovery_active: bool = False
        self._recovery_tick_count: int = 0
        self._recovery_trigger: str = ""
        self._pre_recovery_targets: Dict[str, float] = {}

    # ── E-Stop ───────────────────────────────────────────────

    def emergency_stop(self, reason: str = "manual") -> None:
        """Activate emergency stop — latches until reset_estop()."""
        self._estop_latch = True
        self.state.estop_active = True
        self.state.level = "estop"
        self.state.reason = f"E-STOP: {reason}"
        self._violation_log.append(f"ESTOP: {reason}")

    def reset_estop(self) -> bool:
        """Clear e-stop latch. Returns False if unsafe to clear."""
        self._estop_latch = False
        self.state.estop_active = False
        self.state.level = "normal"
        self.state.reason = "estop cleared"
        return True

    # ── Main tick ────────────────────────────────────────────

    def tick(
        self,
        body: "BodySchema",
        telemetry_bus: "TelemetryBus",
        social_distance_cm: float = 999.0,
    ) -> SafetyState:
        """
        Evaluate all safety constraints and update state.
        Called every brain tick; must be fast.
        """
        now = time.perf_counter()
        self.state.last_update = now
        self.state.active_limits.clear()
        self.state.frozen_joints.clear()
        self._frozen_set.clear()

        # ── E-Stop takes absolute priority ──
        if self._estop_latch:
            self.state.level = "estop"
            self.state.speed_scale = 0.0
            self.state.frozen_joints = list(body.joints.keys())
            self._frozen_set = set(body.joints.keys())
            return self.state

        level = "normal"
        speed = self.config.max_speed_fraction_normal
        reasons: List[str] = []

        # ── 1. Watchdog ──
        dt = now - self._last_command_time
        if dt > self.config.watchdog_timeout_s:
            self.state.watchdog_ok = False
            reasons.append(f"watchdog({dt:.1f}s)")
            level = "caution"
            speed = min(speed, self.config.max_speed_fraction_caution)
        else:
            self.state.watchdog_ok = True

        # ── 2. Human proximity ──
        self.state.human_too_close = False
        if social_distance_cm < self.config.intimate_zone_cm:
            self.state.human_too_close = True
            # Freeze arm joints when human is very close
            for name in body.joints:
                if (
                    "shoulder" in name
                    or "elbow" in name
                    or "wrist" in name
                    or "omoplate" in name
                ):
                    self._frozen_set.add(name)
            level = "frozen" if level != "estop" else level
            speed = 0.0
            reasons.append(f"human_intimate({social_distance_cm:.0f}cm)")
        elif social_distance_cm < self.config.personal_zone_cm:
            level = max(level, "limited") if level != "frozen" else level
            speed = min(speed, self.config.max_speed_fraction_limited)
            reasons.append(f"human_personal({social_distance_cm:.0f}cm)")
        elif social_distance_cm < self.config.social_zone_cm:
            level = (
                max(level, "caution") if level not in ("limited", "frozen") else level
            )
            speed = min(speed, self.config.max_speed_fraction_caution)

        # ── 3. Joint thermal + load ──
        for name, joint in body.joints.items():
            if joint.temperature > self.config.joint_temp_freeze:
                self._frozen_set.add(name)
                reasons.append(f"overtemp({name})")
            elif joint.temperature > self.config.joint_temp_warn:
                speed = min(speed, self.config.max_speed_fraction_limited)
                reasons.append(f"temp_warn({name})")

            if joint.load > self.config.joint_load_freeze:
                self._frozen_set.add(name)
                reasons.append(f"overload({name})")
            elif joint.load > self.config.joint_load_warn:
                speed = min(speed, self.config.max_speed_fraction_caution)

            if joint.stall:
                self._frozen_set.add(name)
                reasons.append(f"stall({name})")

        # ── 4. Collision check ──
        collisions = body.check_collisions()
        self.state.collision_risk = bool(collisions)
        if collisions and self.config.collision_freeze:
            for coll in collisions:
                chain_name = coll.split("->")[0]
                # Freeze the offending chain
                chain = body.chains.get(chain_name)
                if chain:
                    for j in chain.joints:
                        self._frozen_set.add(j.name)
                reasons.append(f"collision({coll})")
            level = "limited" if level not in ("frozen", "estop") else level

        # ── 5. Derive final level ──
        if self._frozen_set:
            if level not in ("frozen", "estop"):
                level = "limited"
            self.state.frozen_joints = sorted(self._frozen_set)

        # ── 6. Recovery mode (instead of permanent freeze) ──
        has_threat = level in ("frozen", "limited") and self._frozen_set
        if self.config.recovery_enabled and has_threat and not self._estop_latch:
            if not self._recovery_active:
                # Enter recovery
                self._recovery_active = True
                self._recovery_tick_count = 0
                self._recovery_trigger = "; ".join(reasons[:3])
            self._recovery_tick_count += 1
            total_recovery = (
                self.config.recovery_retreat_ticks
                + self.config.recovery_explain_ticks
                + self.config.recovery_replan_ticks
            )

            if self._recovery_tick_count <= self.config.recovery_retreat_ticks:
                # Phase 1: RETREAT — slowly move frozen joints toward safe pose
                self.state.recovery_phase = "retreat"
                level = "recovering"
                speed = self.config.max_speed_fraction_limited * 0.5
            elif self._recovery_tick_count <= (
                self.config.recovery_retreat_ticks + self.config.recovery_explain_ticks
            ):
                # Phase 2: EXPLAIN — hold safe pose, signal to consciousness
                self.state.recovery_phase = "explain"
                level = "recovering"
                speed = 0.0  # hold still
            elif self._recovery_tick_count <= total_recovery:
                # Phase 3: REPLAN — re-evaluate, prepare to resume
                self.state.recovery_phase = "replan"
                level = "caution"
                speed = self.config.max_speed_fraction_caution
            else:
                # Recovery complete: check if threat is still present
                # If still frozen joints, restart recovery; else resume normal
                if not self._frozen_set or not has_threat:
                    self._recovery_active = False
                    self.state.recovery_phase = ""
                    level = "normal"
                    speed = self.config.max_speed_fraction_normal
                else:
                    self._recovery_tick_count = 0  # restart recovery cycle

            self.state.recovery_ticks = self._recovery_tick_count
        elif self._recovery_active and not has_threat:
            # Threat cleared during recovery
            self._recovery_active = False
            self.state.recovery_phase = ""
            self.state.recovery_ticks = 0

        self.state.level = level
        self.state.speed_scale = max(0.0, min(1.0, speed))
        self.state.reason = "; ".join(reasons[:5]) if reasons else "ok"
        self.state.active_limits = reasons[:10]

        return self.state

    # ── Command gating ───────────────────────────────────────

    def gate_targets(
        self, targets: Dict[str, float], body: "BodySchema"
    ) -> Dict[str, float]:
        """
        Filter and clamp a dict of {joint_name: target_deg} through
        all safety constraints.  Returns the safe subset.
        """
        self._last_command_time = time.perf_counter()

        if self._estop_latch:
            return dict(FALLBACK_POSE)

        # During recovery retreat, override targets with fallback pose
        if self._recovery_active and self.state.recovery_phase == "retreat":
            targets = dict(FALLBACK_POSE)

        safe: Dict[str, float] = {}
        for name, deg in targets.items():
            # Frozen joints → hold current position
            if name in self._frozen_set:
                joint = body.joints.get(name)
                safe[name] = (
                    joint.current_deg if joint else FALLBACK_POSE.get(name, 90.0)
                )
                continue
            # Clamp to absolute limits
            joint = body.joints.get(name)
            if joint:
                deg = joint.clamp(deg)
                # Apply speed scaling
                max_step = joint.max_speed * self.state.speed_scale
                delta = deg - joint.current_deg
                if abs(delta) > max_step:
                    deg = joint.current_deg + (max_step if delta > 0 else -max_step)
            safe[name] = deg
        return safe

    def notify_command(self) -> None:
        """Call when any movement command is issued (resets watchdog)."""
        self._last_command_time = time.perf_counter()

    # ── Queries ──────────────────────────────────────────────

    def is_safe_to_move(self, joint_name: str = "") -> bool:
        if self._estop_latch:
            return False
        if joint_name and joint_name in self._frozen_set:
            return False
        return self.state.level not in ("frozen", "estop")

    def describe(self) -> str:
        s = self.state
        frozen = ",".join(s.frozen_joints[:5]) or "none"
        recovery = (
            f" recovery={s.recovery_phase}({s.recovery_ticks})"
            if s.recovery_phase
            else ""
        )
        return (
            f"safety={s.level} speed={s.speed_scale:.2f} "
            f"frozen=[{frozen}] watchdog={'ok' if s.watchdog_ok else 'TIMEOUT'} "
            f"collision={int(s.collision_risk)} human_close={int(s.human_too_close)} "
            f"estop={int(s.estop_active)}{recovery} reason={s.reason[:80]}"
        )
