"""
body_schema.py — Kinematic Body Model & Proprioceptive Self-Representation

Schicht B: Full body schema with joint graph, kinematic chains, workspace
envelopes, collision zones, and a proprioceptive self-model that bridges
the gap between servo targets (robot_controller.py) and conscious body
awareness (consciousness.py SelfModel / InteroceptiveBody).

Design:
  • JointNode: single joint with limits, current/target angle, load, error
  • KinematicChain: ordered sequence of joints forming a limb
  • BodySchema: complete body graph — all chains + collision zones + reach
  • ProprioceptiveSummary: compact snapshot for consciousness integration

All angles in degrees, distances in cm (InMoov-scale).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Joint definition
# ─────────────────────────────────────────────────────────────


@dataclass
class JointNode:
    """Single rotary joint with physical limits and live state."""

    name: str
    parent: str  # parent joint name ("root" for torso)
    axis: str  # "yaw" | "pitch" | "roll"
    min_deg: float
    max_deg: float
    rest_deg: float
    max_speed: float  # deg/tick at normal speed
    link_len_cm: float  # physical link length to next joint

    # ── Live state (updated each tick) ──
    current_deg: float = 0.0
    target_deg: float = 0.0
    load: float = 0.0  # normalised motor load [0,1]
    temperature: float = 0.2  # normalised thermal [0,1]
    error: float = 0.0  # |target - current| / range
    stall: bool = False  # motor stalled flag

    def __post_init__(self) -> None:
        self.current_deg = self.rest_deg
        self.target_deg = self.rest_deg

    @property
    def range_deg(self) -> float:
        return max(1.0, self.max_deg - self.min_deg)

    @property
    def normalised(self) -> float:
        """Current position as [0,1] within range."""
        return (self.current_deg - self.min_deg) / self.range_deg

    def clamp(self, deg: float) -> float:
        return max(self.min_deg, min(self.max_deg, deg))

    def set_target(self, deg: float) -> None:
        self.target_deg = self.clamp(deg)

    def step(self, dt: float = 1.0) -> None:
        """Move current toward target respecting speed limit."""
        delta = self.target_deg - self.current_deg
        max_step = self.max_speed * dt
        if abs(delta) <= max_step:
            self.current_deg = self.target_deg
        else:
            self.current_deg += max_step if delta > 0 else -max_step
        self.error = abs(self.target_deg - self.current_deg) / self.range_deg


# ─────────────────────────────────────────────────────────────
# Kinematic chain
# ─────────────────────────────────────────────────────────────


@dataclass
class KinematicChain:
    """Ordered sequence of joints forming a limb."""

    name: str  # e.g. "left_arm", "head", "right_arm"
    joints: List[JointNode] = field(default_factory=list)

    def endpoint_cm(self) -> Tuple[float, float, float]:
        """Approximate 3D endpoint via cumulative forward kinematics (planar approx)."""
        x, y, z = 0.0, 0.0, 0.0
        angle_h = 0.0  # horizontal cumulative angle (yaw)
        angle_v = 0.0  # vertical cumulative angle (pitch)
        for j in self.joints:
            if j.axis == "yaw":
                angle_h += math.radians(j.current_deg - j.rest_deg)
            elif j.axis == "pitch":
                angle_v += math.radians(j.current_deg - j.rest_deg)
            L = j.link_len_cm
            x += L * math.cos(angle_v) * math.sin(angle_h)
            y += L * math.sin(angle_v)
            z += L * math.cos(angle_v) * math.cos(angle_h)
        return (x, y, z)

    def reach_cm(self) -> float:
        """Maximum reach = sum of all link lengths."""
        return sum(j.link_len_cm for j in self.joints)

    def total_error(self) -> float:
        if not self.joints:
            return 0.0
        return sum(j.error for j in self.joints) / len(self.joints)

    def total_load(self) -> float:
        if not self.joints:
            return 0.0
        return max(j.load for j in self.joints)

    def any_stall(self) -> bool:
        return any(j.stall for j in self.joints)


# ─────────────────────────────────────────────────────────────
# Collision zone
# ─────────────────────────────────────────────────────────────


@dataclass
class CollisionZone:
    """Spherical keep-out volume (self-collision or obstacle)."""

    name: str
    center_cm: Tuple[float, float, float]
    radius_cm: float
    severity: float = 1.0  # 0=soft warning, 1=hard stop

    def contains(self, point_cm: Tuple[float, float, float]) -> bool:
        dx = point_cm[0] - self.center_cm[0]
        dy = point_cm[1] - self.center_cm[1]
        dz = point_cm[2] - self.center_cm[2]
        return (dx * dx + dy * dy + dz * dz) <= self.radius_cm**2


# ─────────────────────────────────────────────────────────────
# Proprioceptive summary (for consciousness)
# ─────────────────────────────────────────────────────────────


@dataclass
class ProprioceptiveSummary:
    """Compact body-state snapshot consumed by consciousness."""

    body_pose: str = "idle"  # current posture label
    overall_load: float = 0.0  # [0,1] max load across all joints
    overall_error: float = 0.0  # [0,1] mean tracking error
    reachability_left: float = 0.0  # [0,1] left arm reach fraction
    reachability_right: float = 0.0  # [0,1] right arm reach fraction
    balance_confidence: float = 1.0  # [0,1] stability estimate
    pain_level: float = 0.0  # [0,1] integrity-derived discomfort
    any_stall: bool = False  # a motor is stalled
    head_yaw: float = 90.0
    head_pitch: float = 90.0
    left_hand_open: bool = True
    right_hand_open: bool = True
    social_distance_cm: float = 999.0  # distance to nearest human
    active_human: str = "none"  # label of tracked person
    last_successful_skill: str = "none"


# ─────────────────────────────────────────────────────────────
# Full body schema
# ─────────────────────────────────────────────────────────────


class BodySchema:
    """
    Complete kinematic model of an InMoov-class humanoid.
    Reads targets from RobotController.targets / .state and
    produces ProprioceptiveSummary for consciousness.
    """

    def __init__(self) -> None:
        # ── Build joint graph ────────────────────────────────
        self.joints: Dict[str, JointNode] = {}
        self.chains: Dict[str, KinematicChain] = {}
        self.collision_zones: List[CollisionZone] = []
        self._build_inmoov()
        self.summary = ProprioceptiveSummary()

    # ── InMoov skeleton definition ──────────────────────────

    def _build_inmoov(self) -> None:
        # Head chain
        head_joints = [
            JointNode("head_yaw", "torso", "yaw", 35, 145, 90, 4.0, 8.0),
            JointNode("head_pitch", "head_yaw", "pitch", 55, 125, 90, 3.5, 12.0),
            JointNode("neck_roll", "head_yaw", "roll", 65, 115, 90, 2.5, 5.0),
        ]
        self.chains["head"] = KinematicChain("head", head_joints)

        # Jaw + eyes (no kinematic chain — independent actuators)
        face_joints = [
            JointNode("jaw", "head_pitch", "pitch", 10, 55, 20, 6.0, 3.0),
            JointNode("eye_yaw", "head_pitch", "yaw", 60, 120, 90, 5.0, 2.0),
            JointNode("eye_pitch", "head_pitch", "pitch", 70, 110, 90, 4.0, 2.0),
            JointNode("left_upper_lid", "head_pitch", "pitch", 70, 120, 92, 6.0, 0.5),
            JointNode("left_lower_lid", "head_pitch", "pitch", 60, 110, 88, 6.0, 0.5),
            JointNode("right_upper_lid", "head_pitch", "pitch", 60, 110, 88, 6.0, 0.5),
            JointNode("right_lower_lid", "head_pitch", "pitch", 70, 120, 92, 6.0, 0.5),
        ]
        self.chains["face"] = KinematicChain("face", face_joints)

        # Left arm chain
        left_arm_joints = [
            JointNode("left_omoplate", "torso", "roll", 0, 80, 15, 3.0, 12.0),
            JointNode("left_shoulder", "left_omoplate", "pitch", 0, 85, 20, 3.0, 28.0),
            JointNode("left_elbow", "left_shoulder", "pitch", 5, 95, 30, 4.5, 25.0),
            JointNode("left_wrist", "left_elbow", "roll", 20, 160, 90, 5.0, 8.0),
        ]
        self.chains["left_arm"] = KinematicChain("left_arm", left_arm_joints)

        # Left hand
        left_hand_joints = [
            JointNode("left_thumb", "left_wrist", "pitch", 0, 180, 20, 8.0, 5.0),
            JointNode("left_index", "left_wrist", "pitch", 0, 180, 20, 8.0, 7.0),
        ]
        self.chains["left_hand"] = KinematicChain("left_hand", left_hand_joints)

        # Right arm chain
        right_arm_joints = [
            JointNode("right_omoplate", "torso", "roll", 0, 80, 15, 3.0, 12.0),
            JointNode(
                "right_shoulder", "right_omoplate", "pitch", 0, 85, 20, 3.0, 28.0
            ),
            JointNode("right_elbow", "right_shoulder", "pitch", 5, 95, 30, 4.5, 25.0),
            JointNode("right_wrist", "right_elbow", "roll", 20, 160, 90, 5.0, 8.0),
        ]
        self.chains["right_arm"] = KinematicChain("right_arm", right_arm_joints)

        # Right hand
        right_hand_joints = [
            JointNode("right_thumb", "right_wrist", "pitch", 0, 180, 20, 8.0, 5.0),
            JointNode("right_index", "right_wrist", "pitch", 0, 180, 20, 8.0, 7.0),
        ]
        self.chains["right_hand"] = KinematicChain("right_hand", right_hand_joints)

        # Index all joints by name
        for chain in self.chains.values():
            for j in chain.joints:
                self.joints[j.name] = j

        # ── Collision zones (self-collision prevention) ──
        self.collision_zones = [
            CollisionZone("torso_front", (0.0, 0.0, 10.0), 18.0, 0.8),
            CollisionZone("head_volume", (0.0, 25.0, 5.0), 12.0, 1.0),
            CollisionZone("left_hip", (-15.0, -5.0, 0.0), 10.0, 0.6),
            CollisionZone("right_hip", (15.0, -5.0, 0.0), 10.0, 0.6),
        ]

    # ── Sync from RobotController ────────────────────────────

    def sync_from_controller(
        self, controller_or_targets: object, state: object = None
    ) -> None:
        """Read current + target positions from RobotController dataclasses.
        Accepts either (controller) or (targets, state) for backwards compat."""
        if state is None:
            # Called with a RobotController object
            targets = controller_or_targets.targets
            state = controller_or_targets.state
        else:
            targets = controller_or_targets
        for name, joint in self.joints.items():
            target_val = getattr(targets, name, None)
            if target_val is not None:
                joint.target_deg = joint.clamp(float(target_val))
            current_val = getattr(state, name, None)
            if current_val is not None:
                joint.current_deg = joint.clamp(float(current_val))
            joint.error = abs(joint.target_deg - joint.current_deg) / joint.range_deg
        # Update visual-servo tracking error from controller telemetry
        telem = getattr(controller_or_targets, "telemetry", None)
        if telem is not None:
            self._tracking_confidence = getattr(telem, "tracking_confidence", 0.0)
            self._alignment_error = getattr(telem, "alignment_error", 0.0)
            self._engaged_target = getattr(telem, "engaged_target", "none")
            self._target_center_x = getattr(telem, "selected_target_center_x", 0.5)
            self._target_center_y = getattr(telem, "selected_target_center_y", 0.5)

    # ── Telemetry integration ────────────────────────────────

    # ── Robot command feedback ───────────────────────────────

    def update_from_robot_command(self, kind: str, args: dict) -> None:
        """
        Perception-action feedback: apply a motor command's intended effect to
        the body schema so that subsequent proprioceptive snapshots reflect the
        commanded state.  This closes the loop: action → internal state update
        → perceptual consequence.
        """
        if kind == "look_at":
            yaw = float(args.get("yaw", 0.0))
            pitch = float(args.get("pitch", 0.0))
            h_yaw = self.joints.get("head_yaw")
            h_pitch = self.joints.get("head_pitch")
            e_yaw = self.joints.get("eye_yaw")
            e_pitch = self.joints.get("eye_pitch")
            if h_yaw:
                h_yaw.set_target(h_yaw.target_deg + yaw)
            if h_pitch:
                h_pitch.set_target(h_pitch.target_deg + pitch)
            if e_yaw:
                e_yaw.set_target(e_yaw.target_deg + yaw * 1.5)
            if e_pitch:
                e_pitch.set_target(e_pitch.target_deg + pitch * 1.2)

        elif kind == "set_pose":
            pose = str(args.get("pose", "idle"))
            arms = str(args.get("arms", "parked"))
            hands = str(args.get("hands", "open"))
            # Record in summary so consciousness can observe the change
            if self.summary is not None:
                self.summary.body_pose = pose
                self.summary.left_hand_open = hands in ("open", "both")
                self.summary.right_hand_open = hands in ("open", "both")

        elif kind == "mirror_gesture":
            gesture = str(args.get("gesture", "neutral"))
            intensity = float(args.get("intensity", 0.5))
            # Propagate muscle load proportional to intensity
            for j in self.joints.values():
                if "arm" in j.name or "hand" in j.name:
                    j.load = min(1.0, intensity * 0.4)

        elif kind == "track_person":
            mode = str(args.get("mode", "soft"))
            # Soft tracking → small gaze saccades; hard → larger range
            gain = 0.5 if mode == "soft" else 1.2
            h_yaw = self.joints.get("head_yaw")
            if h_yaw:
                h_yaw.set_target(h_yaw.target_deg + gain)

    def update_from_telemetry(self, telemetry) -> None:
        """Apply sensor feedback from TelemetryFrame or flat dict."""
        # Support both TelemetryFrame dataclass and flat dict
        if hasattr(telemetry, "joint_loads"):
            # TelemetryFrame dataclass
            for name, joint in self.joints.items():
                joint.load = telemetry.joint_loads.get(name, joint.load)
                joint.temperature = telemetry.joint_temps.get(name, joint.temperature)
                joint.stall = telemetry.joint_stalls.get(name, joint.stall)
                pos = telemetry.joint_positions.get(name)
                if pos is not None:
                    joint.current_deg = joint.clamp(float(pos))
                joint.error = (
                    abs(joint.target_deg - joint.current_deg) / joint.range_deg
                )
        else:
            # Flat dict fallback
            for name, joint in self.joints.items():
                joint.load = float(telemetry.get(f"{name}_load", joint.load))
                joint.temperature = float(
                    telemetry.get(f"{name}_temp", joint.temperature)
                )
                joint.stall = bool(telemetry.get(f"{name}_stall", joint.stall))

    # ── Collision check ──────────────────────────────────────

    def check_collisions(self) -> List[str]:
        """Return list of collision zone names that an endpoint penetrates."""
        violations: List[str] = []
        for chain_name in ("left_arm", "right_arm"):
            chain = self.chains.get(chain_name)
            if chain is None:
                continue
            ep = chain.endpoint_cm()
            for zone in self.collision_zones:
                if zone.contains(ep):
                    violations.append(f"{chain_name}->{zone.name}")
        return violations

    # ── Reachability ─────────────────────────────────────────

    def reachability(self, chain_name: str) -> float:
        """How much of max reach is currently available [0,1]."""
        chain = self.chains.get(chain_name)
        if chain is None:
            return 0.0
        ep = chain.endpoint_cm()
        dist = math.sqrt(ep[0] ** 2 + ep[1] ** 2 + ep[2] ** 2)
        max_r = chain.reach_cm()
        return min(1.0, dist / max(1.0, max_r))

    # ── Visual-Proprioceptive Servo Correction ──────────────

    def visual_servo_correction(self) -> Dict[str, float]:
        """Compute corrective joint deltas from visual feedback.

        Closed sensorimotor loop: compares where we expect the target
        to be (from current head angles) with where we actually see it
        (from camera detection). Returns {joint: correction_deg}.
        """
        corrections: Dict[str, float] = {}
        if not hasattr(self, "_engaged_target") or self._engaged_target == "none":
            return corrections
        if not hasattr(self, "_tracking_confidence") or self._tracking_confidence < 0.1:
            return corrections

        # Visual error: deviation of target from image center (0.5, 0.5)
        err_x = self._target_center_x - 0.5  # positive = target is right
        err_y = 0.5 - self._target_center_y  # positive = target is up

        # Proportional gain (scaled by tracking confidence)
        K_yaw = 3.5 * self._tracking_confidence
        K_pitch = 2.5 * self._tracking_confidence

        # Only correct if error exceeds dead zone (prevents jitter)
        if abs(err_x) > 0.03:
            corrections["head_yaw"] = err_x * K_yaw
            corrections["eye_yaw"] = err_x * K_yaw * 1.5  # eyes lead head
        if abs(err_y) > 0.03:
            corrections["head_pitch"] = err_y * K_pitch
            corrections["eye_pitch"] = err_y * K_pitch * 1.2

        return corrections

    def apply_visual_corrections(self) -> None:
        """Apply visual servo corrections to joint targets in-place."""
        corrections = self.visual_servo_correction()
        for name, delta in corrections.items():
            joint = self.joints.get(name)
            if joint:
                joint.set_target(joint.target_deg + delta)

    # ── Step all joints ──────────────────────────────────────

    def step(self, dt: float = 1.0) -> None:
        """Advance all joints toward targets."""
        # Apply visual servo corrections before stepping
        self.apply_visual_corrections()
        for joint in self.joints.values():
            joint.step(dt)

    # ── Produce proprioceptive summary ───────────────────────

    def proprioceptive_snapshot(
        self,
        posture: str = "idle",
        social_distance_cm: float = 999.0,
        active_human: str = "none",
        left_hand_open: bool = True,
        right_hand_open: bool = True,
        last_skill: str = "none",
    ) -> ProprioceptiveSummary:
        """Build a compact summary for consciousness integration."""
        loads = [j.load for j in self.joints.values()]
        errors = [j.error for j in self.joints.values()]
        max_load = max(loads) if loads else 0.0
        mean_error = sum(errors) / max(len(errors), 1)

        head = self.chains.get("head")
        head_yaw = self.joints.get(
            "head_yaw", JointNode("_", "_", "yaw", 0, 180, 90, 1, 1)
        ).current_deg
        head_pitch = self.joints.get(
            "head_pitch", JointNode("_", "_", "pitch", 0, 180, 90, 1, 1)
        ).current_deg

        stall = any(j.stall for j in self.joints.values())
        pain = max(
            0.0, max_load * 0.3 + (1.0 if stall else 0.0) * 0.5 + mean_error * 0.2
        )
        pain = min(1.0, pain)

        # Balance: head + torso error → instability signal
        balance = max(0.0, 1.0 - mean_error * 2.0 - (0.3 if stall else 0.0))

        self.summary = ProprioceptiveSummary(
            body_pose=posture,
            overall_load=max_load,
            overall_error=mean_error,
            reachability_left=self.reachability("left_arm"),
            reachability_right=self.reachability("right_arm"),
            balance_confidence=balance,
            pain_level=pain,
            any_stall=stall,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            left_hand_open=left_hand_open,
            right_hand_open=right_hand_open,
            social_distance_cm=social_distance_cm,
            active_human=active_human,
            last_successful_skill=last_skill,
        )
        return self.summary

    def describe(self) -> str:
        s = self.summary
        return (
            f"pose={s.body_pose} load={s.overall_load:.2f} err={s.overall_error:.2f} "
            f"balance={s.balance_confidence:.2f} pain={s.pain_level:.2f} "
            f"reach_L={s.reachability_left:.2f} reach_R={s.reachability_right:.2f} "
            f"stall={int(s.any_stall)} human={s.active_human} dist={s.social_distance_cm:.0f}cm"
        )
