from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class ServoSpec:
    channel: str
    min_deg: float
    max_deg: float
    rest_deg: float
    max_speed_deg: float


@dataclass
class JointTargets:
    head_yaw: float = 90.0
    head_pitch: float = 90.0
    neck_roll: float = 90.0
    jaw: float = 20.0
    eye_yaw: float = 90.0
    eye_pitch: float = 90.0
    left_upper_lid: float = 92.0
    left_lower_lid: float = 88.0
    right_upper_lid: float = 88.0
    right_lower_lid: float = 92.0
    left_omoplate: float = 15.0
    right_omoplate: float = 15.0
    left_shoulder: float = 20.0
    right_shoulder: float = 20.0
    left_elbow: float = 30.0
    right_elbow: float = 30.0
    left_wrist: float = 90.0
    right_wrist: float = 90.0
    left_thumb: float = 20.0
    right_thumb: float = 20.0
    left_index: float = 20.0
    right_index: float = 20.0


@dataclass
class JointState:
    head_yaw: float = 90.0
    head_pitch: float = 90.0
    neck_roll: float = 90.0
    jaw: float = 20.0
    eye_yaw: float = 90.0
    eye_pitch: float = 90.0
    left_upper_lid: float = 92.0
    left_lower_lid: float = 88.0
    right_upper_lid: float = 88.0
    right_lower_lid: float = 92.0
    left_omoplate: float = 15.0
    right_omoplate: float = 15.0
    left_shoulder: float = 20.0
    right_shoulder: float = 20.0
    left_elbow: float = 30.0
    right_elbow: float = 30.0
    left_wrist: float = 90.0
    right_wrist: float = 90.0
    left_thumb: float = 20.0
    right_thumb: float = 20.0
    left_index: float = 20.0
    right_index: float = 20.0


@dataclass
class HandTargets:
    left: str = "open"
    right: str = "open"


@dataclass
class RobotTelemetry:
    engaged_target: str = "none"
    interaction_zone: str = "public"
    posture: str = "idle"
    gesture_mode: str = "neutral"
    locomotion_locked: bool = True
    imitation_active: bool = False
    command_success: float = 0.5
    alignment_error: float = 0.0
    tracking_confidence: float = 0.0
    selected_target_label: str = "none"
    selected_target_center_x: float = 0.5
    selected_target_center_y: float = 0.5
    selected_target_area: float = 0.0
    target_priority_mode: str = "largest_face"
    last_update_tick: int = 0
    notes: str = ""
    feedback: str = ""


@dataclass
class PolicyValue:
    expected_success: float = 0.5
    observations: int = 0


@dataclass
class HeadServoConfig:
    code: str
    joint_name: str
    channel: int
    min_deg: int
    max_deg: int
    min_pulse: int
    max_pulse: int

    def as_dict(self) -> Dict[str, int | str]:
        return {
            "code": self.code,
            "joint_name": self.joint_name,
            "channel": self.channel,
            "min_deg": self.min_deg,
            "max_deg": self.max_deg,
            "min_pulse": self.min_pulse,
            "max_pulse": self.max_pulse,
        }


class RobotController:
    """Simulated InMoov-facing control layer with continuous servo regulation."""

    HEAD_ONLY_KEYS = (
        "head_yaw",
        "head_pitch",
        "neck_roll",
        "jaw",
        "eye_yaw",
        "eye_pitch",
        "left_upper_lid",
        "left_lower_lid",
        "right_upper_lid",
        "right_lower_lid",
    )
    HEAD_PRESETS: Dict[str, Dict[str, float]] = {
        "Center": {
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
        },
        "Left Scan": {
            "head_yaw": 125.0,
            "head_pitch": 92.0,
            "neck_roll": 96.0,
            "jaw": 20.0,
            "eye_yaw": 112.0,
            "eye_pitch": 92.0,
            "left_upper_lid": 94.0,
            "left_lower_lid": 84.0,
            "right_upper_lid": 90.0,
            "right_lower_lid": 94.0,
        },
        "Right Scan": {
            "head_yaw": 55.0,
            "head_pitch": 92.0,
            "neck_roll": 84.0,
            "jaw": 20.0,
            "eye_yaw": 68.0,
            "eye_pitch": 92.0,
            "left_upper_lid": 90.0,
            "left_lower_lid": 94.0,
            "right_upper_lid": 94.0,
            "right_lower_lid": 84.0,
        },
        "Speak": {
            "head_yaw": 90.0,
            "head_pitch": 88.0,
            "neck_roll": 90.0,
            "jaw": 34.0,
            "eye_yaw": 90.0,
            "eye_pitch": 88.0,
            "left_upper_lid": 96.0,
            "left_lower_lid": 82.0,
            "right_upper_lid": 84.0,
            "right_lower_lid": 96.0,
        },
        "Attend": {
            "head_yaw": 90.0,
            "head_pitch": 82.0,
            "neck_roll": 90.0,
            "jaw": 18.0,
            "eye_yaw": 90.0,
            "eye_pitch": 84.0,
            "left_upper_lid": 102.0,
            "left_lower_lid": 78.0,
            "right_upper_lid": 78.0,
            "right_lower_lid": 102.0,
        },
        "Rest": {
            "head_yaw": 90.0,
            "head_pitch": 98.0,
            "neck_roll": 90.0,
            "jaw": 14.0,
            "eye_yaw": 90.0,
            "eye_pitch": 94.0,
            "left_upper_lid": 82.0,
            "left_lower_lid": 98.0,
            "right_upper_lid": 98.0,
            "right_lower_lid": 82.0,
        },
    }

    SERVO_MAP: Dict[str, ServoSpec] = {
        "head_yaw": ServoSpec("i01.head.rothead", 35.0, 145.0, 90.0, 4.0),
        "head_pitch": ServoSpec("i01.head.neck", 55.0, 125.0, 90.0, 3.5),
        "neck_roll": ServoSpec("i01.head.rollNeck", 65.0, 115.0, 90.0, 2.5),
        "jaw": ServoSpec("i01.head.jaw", 10.0, 55.0, 20.0, 6.0),
        "eye_yaw": ServoSpec("i01.head.eyeX", 60.0, 120.0, 90.0, 5.0),
        "eye_pitch": ServoSpec("i01.head.eyeY", 70.0, 110.0, 90.0, 4.0),
        "left_upper_lid": ServoSpec("i01.head.eyelidLeftUpper", 70.0, 120.0, 92.0, 6.0),
        "left_lower_lid": ServoSpec("i01.head.eyelidLeftLower", 60.0, 110.0, 88.0, 6.0),
        "right_upper_lid": ServoSpec(
            "i01.head.eyelidRightUpper", 60.0, 110.0, 88.0, 6.0
        ),
        "right_lower_lid": ServoSpec(
            "i01.head.eyelidRightLower", 70.0, 120.0, 92.0, 6.0
        ),
        "left_omoplate": ServoSpec("i01.leftArm.omoplate", 0.0, 80.0, 15.0, 3.0),
        "right_omoplate": ServoSpec("i01.rightArm.omoplate", 0.0, 80.0, 15.0, 3.0),
        "left_shoulder": ServoSpec("i01.leftArm.shoulder", 0.0, 85.0, 20.0, 3.0),
        "right_shoulder": ServoSpec("i01.rightArm.shoulder", 0.0, 85.0, 20.0, 3.0),
        "left_elbow": ServoSpec("i01.leftArm.bicep", 5.0, 95.0, 30.0, 4.5),
        "right_elbow": ServoSpec("i01.rightArm.bicep", 5.0, 95.0, 30.0, 4.5),
        "left_wrist": ServoSpec("i01.leftHand.wrist", 20.0, 160.0, 90.0, 5.0),
        "right_wrist": ServoSpec("i01.rightHand.wrist", 20.0, 160.0, 90.0, 5.0),
        "left_thumb": ServoSpec("i01.leftHand.thumb", 0.0, 180.0, 20.0, 8.0),
        "right_thumb": ServoSpec("i01.rightHand.thumb", 0.0, 180.0, 20.0, 8.0),
        "left_index": ServoSpec("i01.leftHand.index", 0.0, 180.0, 20.0, 8.0),
        "right_index": ServoSpec("i01.rightHand.index", 0.0, 180.0, 20.0, 8.0),
    }

    def __init__(self) -> None:
        self.targets = JointTargets()
        self.state = JointState()
        self.hands = HandTargets()
        self.telemetry = RobotTelemetry()
        self._history: list[str] = []
        self._policy_memory: Dict[str, PolicyValue] = {}
        self._active_action_kind: str = ""
        self._active_context_key: str = ""
        self._frame_seq: int = 0
        self._head_config: Dict[str, HeadServoConfig] = {
            "HY": HeadServoConfig("HY", "head_yaw", 0, 35, 145, 110, 510),
            "HP": HeadServoConfig("HP", "head_pitch", 1, 55, 125, 120, 500),
            "NR": HeadServoConfig("NR", "neck_roll", 2, 65, 115, 135, 475),
            "JW": HeadServoConfig("JW", "jaw", 3, 10, 55, 130, 320),
            "EX": HeadServoConfig("EX", "eye_yaw", 4, 60, 120, 165, 430),
            "EY": HeadServoConfig("EY", "eye_pitch", 5, 70, 110, 180, 410),
            "LU": HeadServoConfig("LU", "left_upper_lid", 6, 70, 120, 180, 430),
            "LL": HeadServoConfig("LL", "left_lower_lid", 7, 60, 110, 170, 410),
            "RU": HeadServoConfig("RU", "right_upper_lid", 8, 60, 110, 170, 410),
            "RL": HeadServoConfig("RL", "right_lower_lid", 9, 70, 120, 180, 430),
        }
        self._head_joint_to_code: Dict[str, str] = {
            config.joint_name: code for code, config in self._head_config.items()
        }
        # ── Intent-before-motion state (G) ───────────────────
        self._intent_state: str = "none"      # none | gaze | prepare | execute
        self._intent_target: str = ""         # human-readable target description
        self._intent_motion_fn: Any = None    # callable() to execute on phase 3
        self._intent_ticks_remaining: int = 0

    def apply_action(self, kind: str, args: Dict, tick: int) -> str:
        handler = getattr(self, f"_apply_{kind}", None)
        if handler is None:
            msg = f"ROBOT_CTRL unsupported action={kind}"
            self._remember(msg, tick)
            return msg
        self._active_action_kind = kind
        msg = handler(args, tick)
        self._remember(msg, tick)
        return msg

    def _remember(self, msg: str, tick: int) -> None:
        self.telemetry.last_update_tick = tick
        self._history.append(msg)
        if len(self._history) > 50:
            self._history = self._history[-50:]

    # ── Intent-before-motion (Feature G) ─────────────────────────────────

    def intent_move_to(
        self,
        target: str,
        motion_fn: Callable[[], None],
        tick: int,
    ) -> None:
        """
        Execute a 3-phase intent sequence before running `motion_fn`.

        Phase 1 (3 ticks) — gaze toward target (look first)
        Phase 2 (2 ticks) — brief preparatory pause/lean
        Phase 3 (immediate) — call motion_fn()

        If an intent sequence is already in progress, the new one is
        queued by overwriting (latest intent wins — one at a time).
        """
        self._intent_target = target
        self._intent_motion_fn = motion_fn
        self._intent_state = "gaze"
        self._intent_ticks_remaining = 3
        # Phase 1: gaze toward target (reuse gaze_at_person logic as approximation)
        self.gaze_at_person()

    def tick_intent(self, tick: int) -> None:
        """
        Advance the intent-before-motion state machine.
        Call once per robot tick (brain.py).
        """
        if self._intent_state == "none" or self._intent_motion_fn is None:
            return

        self._intent_ticks_remaining -= 1

        if self._intent_state == "gaze":
            if self._intent_ticks_remaining <= 0:
                # Transition to prepare phase: slight lean / pause
                self._intent_state = "prepare"
                self._intent_ticks_remaining = 2
                # Small preparatory head tilt (readiness cue)
                self._set_target("head_pitch", 86.0)

        elif self._intent_state == "prepare":
            if self._intent_ticks_remaining <= 0:
                # Phase 3: execute motion
                self._intent_state = "execute"
                try:
                    self._intent_motion_fn()
                except Exception:
                    pass
                self._intent_state = "none"
                self._intent_motion_fn = None
                self._intent_target = ""

    def _clamp(self, joint_name: str, value: float) -> float:
        head_code = self._head_joint_to_code.get(joint_name)
        if head_code:
            cfg = self._head_config[head_code]
            return max(cfg.min_deg, min(cfg.max_deg, value))
        spec = self.SERVO_MAP[joint_name]
        return max(spec.min_deg, min(spec.max_deg, value))

    def update_head_servo_config(
        self,
        code: str,
        *,
        channel: int | None = None,
        min_deg: int | None = None,
        max_deg: int | None = None,
        min_pulse: int | None = None,
        max_pulse: int | None = None,
    ) -> bool:
        cfg = self._head_config.get(code)
        if cfg is None:
            return False
        if channel is not None:
            cfg.channel = max(0, min(15, int(channel)))
        if min_deg is not None:
            cfg.min_deg = int(min_deg)
        if max_deg is not None:
            cfg.max_deg = int(max_deg)
        if cfg.max_deg <= cfg.min_deg:
            cfg.max_deg = cfg.min_deg + 1
        current = getattr(self.targets, cfg.joint_name)
        setattr(self.targets, cfg.joint_name, self._clamp(cfg.joint_name, current))
        current_state = getattr(self.state, cfg.joint_name)
        setattr(self.state, cfg.joint_name, self._clamp(cfg.joint_name, current_state))
        if min_pulse is not None:
            cfg.min_pulse = max(50, int(min_pulse))
        if max_pulse is not None:
            cfg.max_pulse = max(cfg.min_pulse + 1, int(max_pulse))
        return True

    def set_manual_head_pose(
        self,
        *,
        yaw_deg: float | None = None,
        pitch_deg: float | None = None,
        roll_deg: float | None = None,
        jaw_deg: float | None = None,
    ) -> str:
        values = {
            "head_yaw": yaw_deg,
            "head_pitch": pitch_deg,
            "neck_roll": roll_deg,
            "jaw": jaw_deg,
        }
        self.set_head_targets(values)
        self.telemetry.engaged_target = "manual"
        self.telemetry.posture = "manual_head_control"
        self.telemetry.notes = "manual head pose"
        return (
            f"ROBOT_CTRL head_manual yaw={self.targets.head_yaw:.1f} pitch={self.targets.head_pitch:.1f} "
            f"roll={self.targets.neck_roll:.1f} jaw={self.targets.jaw:.1f}"
        )

    def set_head_targets(self, joint_values: Dict[str, float | None]) -> str:
        updated = []
        for joint_name, value in joint_values.items():
            if joint_name not in self.HEAD_ONLY_KEYS or value is None:
                continue
            self._set_target(joint_name, float(value))
            updated.append(f"{joint_name}={getattr(self.targets, joint_name):.1f}")
        if updated:
            self.telemetry.engaged_target = "manual"
            self.telemetry.posture = "manual_head_control"
            self.telemetry.notes = "manual head pose"
        return "ROBOT_CTRL head_targets " + " ".join(updated)

    def apply_head_preset(self, preset_name: str) -> str:
        preset = self.HEAD_PRESETS.get(preset_name)
        if preset is None:
            return f"ROBOT_CTRL unknown_head_preset={preset_name}"
        return self.set_manual_head_pose(
            yaw_deg=preset.get("head_yaw"),
            pitch_deg=preset.get("head_pitch"),
            roll_deg=preset.get("neck_roll"),
            jaw_deg=preset.get("jaw"),
        )

    def head_presets_snapshot(self) -> Dict[str, Dict[str, float]]:
        return {name: dict(values) for name, values in self.HEAD_PRESETS.items()}

    def head_config_snapshot(self) -> Dict[str, Dict[str, int | str]]:
        return {code: cfg.as_dict() for code, cfg in self._head_config.items()}

    def _set_target(self, joint_name: str, value: float) -> None:
        setattr(self.targets, joint_name, self._clamp(joint_name, value))

    def control_step(self, tick: int) -> None:
        for joint_name, spec in self.SERVO_MAP.items():
            current = getattr(self.state, joint_name)
            target = getattr(self.targets, joint_name)
            delta = target - current
            if abs(delta) <= spec.max_speed_deg:
                current = target
            else:
                current += spec.max_speed_deg if delta > 0 else -spec.max_speed_deg
            setattr(self.state, joint_name, self._clamp(joint_name, current))
        self.telemetry.last_update_tick = tick

    def _select_target(
        self, detection_targets: list[Dict[str, object]], labels: tuple[str, ...]
    ) -> Dict[str, object]:
        candidates = [t for t in detection_targets if str(t.get("label", "")) in labels]
        if not candidates:
            return {}
        face_candidates = [t for t in candidates if str(t.get("label", "")) == "face"]
        if face_candidates:
            candidates = face_candidates
        return max(
            candidates,
            key=lambda item: (
                float(item.get("width", 0.0)) * float(item.get("height", 0.0)) * 3.0
                + float(item.get("score", 0.0))
                - abs(float(item.get("center_x", 0.5)) - 0.5) * 0.15
            ),
        )

    def observe_world(
        self,
        detections: list[str],
        detection_targets: list[Dict[str, object]],
        focus_region: str,
        task_phase: str,
        tick: int,
    ) -> None:
        sees_person = any(
            d in ("person", "face") or d.startswith("face") for d in detections
        )
        gestures = [d.split(":", 1)[1] for d in detections if d.startswith("hand:")]
        sees_gesture = bool(gestures)
        person_target = self._select_target(detection_targets, ("face", "person"))
        if person_target:
            self.telemetry.selected_target_label = str(
                person_target.get("label", "person")
            )
            self.telemetry.selected_target_center_x = float(
                person_target.get("center_x", 0.5)
            )
            self.telemetry.selected_target_center_y = float(
                person_target.get("center_y", 0.5)
            )
            self.telemetry.selected_target_area = float(
                person_target.get("width", 0.0)
            ) * float(person_target.get("height", 0.0))
        else:
            self.telemetry.selected_target_label = "none"
            self.telemetry.selected_target_center_x = 0.5
            self.telemetry.selected_target_center_y = 0.5
            self.telemetry.selected_target_area = 0.0

        if self.telemetry.engaged_target == "person":
            if person_target:
                center_x = float(person_target.get("center_x", 0.5))
                center_y = float(person_target.get("center_y", 0.5))
                desired_yaw = 90.0 + (center_x - 0.5) * 70.0
                desired_pitch = 90.0 + (0.5 - center_y) * 44.0
                desired_roll = 90.0 + (center_x - 0.5) * 10.0
            else:
                desired_yaw = 82.0 if (tick // 18) % 2 == 0 else 98.0
                desired_pitch = 90.0
                desired_roll = 90.0
            self._set_target("head_yaw", desired_yaw)
            self._set_target("head_pitch", desired_pitch)
            self._set_target("neck_roll", desired_roll)

        target_ok = (
            (self.telemetry.engaged_target == "person" and bool(person_target))
            or (
                self.telemetry.engaged_target not in ("person", "none")
                and bool(detections)
            )
            or (self.telemetry.engaged_target == "internal" and not detections)
        )
        desired_attention = self.targets.head_yaw
        desired_pitch = self.targets.head_pitch
        yaw_error = abs(self.state.head_yaw - desired_attention) / max(
            1.0, self.SERVO_MAP["head_yaw"].max_deg - self.SERVO_MAP["head_yaw"].min_deg
        )
        pitch_error = abs(self.state.head_pitch - desired_pitch) / max(
            1.0,
            self.SERVO_MAP["head_pitch"].max_deg - self.SERVO_MAP["head_pitch"].min_deg,
        )
        self.telemetry.alignment_error = (yaw_error + pitch_error) * 0.5
        base_track = (
            0.88
            if person_target and self.telemetry.engaged_target == "person"
            else 0.25 if sees_person else 0.1
        )
        self.telemetry.tracking_confidence = max(
            0.0, min(1.0, base_track - self.telemetry.alignment_error * 0.75)
        )

        success = 0.35
        if target_ok:
            success += 0.30
        success += self.telemetry.tracking_confidence * 0.20
        if self.telemetry.imitation_active and sees_gesture:
            success += 0.15
        if task_phase == "stabilise":
            success -= 0.10
        self.telemetry.command_success = max(0.0, min(1.0, success))

        if self.telemetry.imitation_active and not sees_gesture:
            self.telemetry.feedback = "gesture target lost"
            self.telemetry.imitation_active = False
        elif sees_person and self.telemetry.tracking_confidence < 0.35:
            self.telemetry.feedback = "tracking drift"
        elif target_ok:
            self.telemetry.feedback = "aligned"
        else:
            self.telemetry.feedback = "awaiting target"

        self._active_context_key = self._build_context_key(
            zone=self.telemetry.interaction_zone,
            focus=focus_region,
            gesture=gestures[0] if gestures else self.telemetry.gesture_mode,
            target=self.telemetry.engaged_target,
        )
        self._learn_from_feedback(
            self._active_action_kind,
            self._active_context_key,
            self.telemetry.command_success,
        )
        self.telemetry.last_update_tick = tick

    def _build_context_key(
        self, zone: str, focus: str, gesture: str, target: str
    ) -> str:
        return f"zone={zone}|focus={focus or '-'}|gesture={gesture or '-'}|target={target or '-'}"

    def _learn_from_feedback(
        self, action_kind: str, context_key: str, success: float
    ) -> None:
        if not action_kind or not context_key:
            return
        key = f"{action_kind}|{context_key}"
        entry = self._policy_memory.get(key)
        if entry is None:
            self._policy_memory[key] = PolicyValue(
                expected_success=success, observations=1
            )
            return
        entry.expected_success = entry.expected_success * 0.88 + success * 0.12
        entry.observations += 1

    def best_policy(self) -> Dict[str, object]:
        if not self._policy_memory:
            return {
                "context": "",
                "recommended_action": "",
                "expected_success": 0.0,
                "observations": 0,
            }
        key, entry = max(
            self._policy_memory.items(),
            key=lambda item: item[1].expected_success * max(1, item[1].observations),
        )
        action_kind, context = key.split("|", 1)
        return {
            "context": context,
            "recommended_action": action_kind,
            "expected_success": round(entry.expected_success, 3),
            "observations": entry.observations,
        }

    def _resolve_deg(self, value: float, scale: float, default: float) -> float:
        return value if abs(value) > 5.0 else default + value * scale

    def _apply_look_at(self, args: Dict, tick: int) -> str:
        yaw = float(
            args.get(
                "yaw_deg", self._resolve_deg(float(args.get("yaw", 0.0)), 40.0, 90.0)
            )
        )
        pitch = float(
            args.get(
                "pitch_deg",
                self._resolve_deg(float(args.get("pitch", 0.0)), 25.0, 90.0),
            )
        )
        self._set_target("head_yaw", yaw)
        self._set_target("head_pitch", pitch)
        self.telemetry.engaged_target = str(args.get("target", "none"))
        self.telemetry.posture = "attentive"
        self.telemetry.notes = "gaze alignment"
        return f"ROBOT_CTRL look_at target={self.telemetry.engaged_target} head_target=({self.targets.head_yaw:.1f},{self.targets.head_pitch:.1f})"

    def _apply_set_pose(self, args: Dict, tick: int) -> str:
        pose = str(args.get("pose", "idle"))
        arms = str(args.get("arms", "parked"))
        hands = str(args.get("hands", "open"))
        self.telemetry.posture = pose
        self.telemetry.notes = f"pose={pose} arms={arms}"
        if arms == "parked":
            self._set_target("left_omoplate", 15.0)
            self._set_target("right_omoplate", 15.0)
            self._set_target("left_shoulder", 20.0)
            self._set_target("right_shoulder", 20.0)
            self._set_target("left_elbow", 30.0)
            self._set_target("right_elbow", 30.0)
        elif arms == "gesture_ready":
            self._set_target("left_omoplate", 32.0)
            self._set_target("right_omoplate", 32.0)
            self._set_target("left_shoulder", 48.0)
            self._set_target("right_shoulder", 48.0)
            self._set_target("left_elbow", 58.0)
            self._set_target("right_elbow", 58.0)
        else:
            self._set_target("left_omoplate", 24.0)
            self._set_target("right_omoplate", 24.0)
            self._set_target("left_shoulder", 34.0)
            self._set_target("right_shoulder", 34.0)
            self._set_target("left_elbow", 42.0)
            self._set_target("right_elbow", 42.0)
        finger = 20.0 if hands == "open" else 110.0
        self._set_target("left_thumb", finger)
        self._set_target("right_thumb", finger)
        self._set_target("left_index", finger)
        self._set_target("right_index", finger)
        self.hands.left = hands
        self.hands.right = hands
        self.telemetry.locomotion_locked = pose in ("protective_idle", "idle")
        return f"ROBOT_CTRL set_pose pose={pose} arms={arms} hands={hands}"

    def _apply_mirror_gesture(self, args: Dict, tick: int) -> str:
        gesture = str(args.get("gesture", "neutral"))
        intensity = max(0.0, min(1.0, float(args.get("intensity", 0.5))))
        self.telemetry.gesture_mode = gesture
        self.telemetry.imitation_active = True
        self.telemetry.posture = "mirroring"
        self.telemetry.notes = f"mirror={gesture}"
        self._set_target("left_omoplate", 22.0 + intensity * 18.0)
        self._set_target("right_omoplate", 22.0 + intensity * 18.0)
        self._set_target("left_shoulder", 30.0 + intensity * 35.0)
        self._set_target("right_shoulder", 30.0 + intensity * 35.0)
        self._set_target("left_elbow", 28.0 + intensity * 48.0)
        self._set_target("right_elbow", 28.0 + intensity * 48.0)
        self._set_target("left_wrist", 90.0 + intensity * 18.0)
        self._set_target("right_wrist", 90.0 - intensity * 18.0)
        self._set_target("jaw", 18.0 + intensity * 8.0)
        finger = 45.0 if gesture in ("open", "victory") else 110.0
        thumb = 35.0 if gesture in ("thumbs_up", "open") else 115.0
        self._set_target("left_thumb", thumb)
        self._set_target("right_thumb", thumb)
        self._set_target("left_index", finger)
        self._set_target("right_index", finger)
        self.hands.left = (
            gesture if gesture in ("open", "victory", "thumbs_up") else "gesture"
        )
        self.hands.right = (
            gesture if gesture in ("open", "victory", "thumbs_up") else "gesture"
        )
        return f"ROBOT_CTRL mirror gesture={gesture} intensity={intensity:.2f}"

    # ── Jaw sync for TTS ─────────────────────────────────────
    def jaw_open(self) -> None:
        """Open jaw for speech sync (called by SpeechOutput on_start)."""
        self._set_target("jaw", 38.0)

    def jaw_close(self) -> None:
        """Close jaw after speech (called by SpeechOutput on_end)."""
        self._set_target("jaw", 20.0)

    # ── Embodied motor cues from UtterancePlan ──────────────
    def nod_head(self) -> None:
        """Quick head nod gesture (small pitch dip + return)."""
        _cur = self.targets.head_pitch
        self._set_target("head_pitch", _cur + 6.0)
        # The regular servo loop will return to current pitch over time

    def gaze_at_person(self) -> None:
        """Direct gaze toward the primary interlocutor (center front)."""
        self._set_target("head_yaw", 90.0)
        self._set_target("head_pitch", 88.0)
        self._set_target("eye_yaw", 90.0)
        self._set_target("eye_pitch", 88.0)
        self.telemetry.engaged_target = "person"
        self.telemetry.posture = "attentive"

    def _apply_track_person(self, args: Dict, tick: int) -> str:
        mode = str(args.get("mode", "soft"))
        zone = str(args.get("zone", "social"))
        self.telemetry.engaged_target = "person"
        self.telemetry.interaction_zone = zone
        self.telemetry.posture = "tracking"
        self.telemetry.notes = f"track={mode}"
        self._set_target("head_yaw", 93.0 if mode == "soft" else 97.0)
        self._set_target("head_pitch", 90.0)
        return f"ROBOT_CTRL track_person mode={mode} zone={zone}"

    def summary(self) -> str:
        policy = self.best_policy()
        policy_text = "none"
        if policy.get("recommended_action"):
            policy_text = (
                f"{policy['recommended_action']}:{policy['expected_success']:.2f}"
            )
        return (
            f"ctrl target={self.telemetry.engaged_target} posture={self.telemetry.posture} "
            f"zone={self.telemetry.interaction_zone} hands={self.hands.left}/{self.hands.right} "
            f"head=({self.state.head_yaw:.1f},{self.state.head_pitch:.1f}) "
            f"ok={self.telemetry.command_success:.2f} track={self.telemetry.tracking_confidence:.2f} "
            f"focus={self.telemetry.selected_target_label}@({self.telemetry.selected_target_center_x:.2f},{self.telemetry.selected_target_center_y:.2f}) "
            f"fb={self.telemetry.feedback or 'none'} imitate={int(self.telemetry.imitation_active)} policy={policy_text}"
        )

    def export_mrl_lines(self) -> list[str]:
        return [
            f"{spec.channel}.moveTo({getattr(self.targets, name):.1f})"
            for name, spec in self.SERVO_MAP.items()
        ]

    def _arduino_field_map(self) -> Dict[str, int]:
        aliases = {
            "head_yaw": "HY",
            "head_pitch": "HP",
            "neck_roll": "NR",
            "jaw": "JW",
            "eye_yaw": "EX",
            "eye_pitch": "EY",
            "left_upper_lid": "LU",
            "left_lower_lid": "LL",
            "right_upper_lid": "RU",
            "right_lower_lid": "RL",
            "left_omoplate": "LO",
            "right_omoplate": "RO",
            "left_shoulder": "LS",
            "right_shoulder": "RS",
            "left_elbow": "LE",
            "right_elbow": "RE",
            "left_wrist": "LW",
            "right_wrist": "RW",
            "left_thumb": "LT",
            "right_thumb": "RT",
            "left_index": "LI",
            "right_index": "RI",
        }
        return {
            aliases[name]: int(round(getattr(self.targets, name)))
            for name in self.SERVO_MAP
        }

    def _arduino_head_field_map(self) -> Dict[str, int]:
        return {
            code: int(round(getattr(self.targets, cfg.joint_name)))
            for code, cfg in self._head_config.items()
        }

    def _checksum_hex(self, payload: str) -> str:
        checksum = 0
        for ch in payload.encode("ascii", errors="ignore"):
            checksum ^= ch
        return f"{checksum:02X}"

    def export_arduino_packet(self) -> str:
        parts = [f"{key}={value}" for key, value in self._arduino_field_map().items()]
        return "SERVO|" + "|".join(parts)

    def export_arduino_head_packet(self) -> str:
        parts = [
            f"{key}={value}" for key, value in self._arduino_head_field_map().items()
        ]
        return "HEAD|PCA9685|" + "|".join(parts)

    def export_arduino_serial_frame(
        self, tick: int | None = None, advance_seq: bool = True
    ) -> str:
        if advance_seq:
            self._frame_seq = (self._frame_seq + 1) % 100000
        frame_tick = self.telemetry.last_update_tick if tick is None else tick
        fields = self._arduino_field_map()
        payload_parts = ["AIBOT", f"SEQ={self._frame_seq}", f"T={int(frame_tick)}"]
        payload_parts.extend(f"{key}={value}" for key, value in fields.items())
        payload = ",".join(payload_parts)
        checksum = self._checksum_hex(payload)
        return f"${payload}*{checksum}\r\n"

    def export_arduino_head_serial_frame(
        self, tick: int | None = None, advance_seq: bool = True
    ) -> str:
        if advance_seq:
            self._frame_seq = (self._frame_seq + 1) % 100000
        frame_tick = self.telemetry.last_update_tick if tick is None else tick
        fields = self._arduino_head_field_map()
        payload_parts = [
            "AIBOT",
            f"SEQ={self._frame_seq}",
            f"T={int(frame_tick)}",
            "PROFILE=HEAD_PCA9685",
        ]
        payload_parts.extend(f"{key}={value}" for key, value in fields.items())
        payload = ",".join(payload_parts)
        checksum = self._checksum_hex(payload)
        return f"${payload}*{checksum}\r\n"

    def export_head_config_serial_frame(self, advance_seq: bool = True) -> str:
        if advance_seq:
            self._frame_seq = (self._frame_seq + 1) % 100000
        payload_parts = [
            "AIBOTCFG",
            f"SEQ={self._frame_seq}",
            "PROFILE=HEAD_PCA9685",
        ]
        for code, cfg in self._head_config.items():
            payload_parts.extend(
                [
                    f"{code}_CH={cfg.channel}",
                    f"{code}_MIND={cfg.min_deg}",
                    f"{code}_MAXD={cfg.max_deg}",
                    f"{code}_MINP={cfg.min_pulse}",
                    f"{code}_MAXP={cfg.max_pulse}",
                ]
            )
        payload = ",".join(payload_parts)
        checksum = self._checksum_hex(payload)
        return f"${payload}*{checksum}\r\n"

    def snapshot(self) -> Dict[str, object]:
        return {
            "servo_map": {
                name: {
                    "channel": spec.channel,
                    "min_deg": spec.min_deg,
                    "max_deg": spec.max_deg,
                    "rest_deg": spec.rest_deg,
                    "max_speed_deg": spec.max_speed_deg,
                }
                for name, spec in self.SERVO_MAP.items()
            },
            "target_joints": dict(self.targets.__dict__),
            "current_joints": dict(self.state.__dict__),
            "hands": dict(self.hands.__dict__),
            "head_config": self.head_config_snapshot(),
            "head_presets": self.head_presets_snapshot(),
            "telemetry": dict(self.telemetry.__dict__),
            "learned_policy": self.best_policy(),
            "exports": {
                "myrobotlab": self.export_mrl_lines()[:8],
                "arduino": self.export_arduino_packet(),
                "arduino_head": self.export_arduino_head_packet(),
                "arduino_frame": self.export_arduino_serial_frame(
                    self.telemetry.last_update_tick, advance_seq=False
                ),
                "arduino_head_frame": self.export_arduino_head_serial_frame(
                    self.telemetry.last_update_tick, advance_seq=False
                ),
                "head_config_frame": self.export_head_config_serial_frame(
                    advance_seq=False
                ),
            },
            "summary": self.summary(),
        }


# ─────────────────────────────────────────────────────────────
# GazeDynamics — tick-driven natural gaze behaviour
# ─────────────────────────────────────────────────────────────


class GazeDynamics:
    """
    Tick-driven state machine for natural gaze behaviour.

    States:
        ATTEND      — direct eye contact with interlocutor
        GLANCE_AWAY — brief look-away (thought / recall simulation)
        BLINK       — rapid lid close-open

    Rules (rough empirical values from human gaze studies):
        • During LISTENING: ATTEND ~80 % of time, short glances ~20 %
        • During SPEAKING:  ATTEND ~60 % of time, glances ~40 %
        • Blink every 3-6 s (randomised)
        • Glances last 0.6-1.8 s; attend windows 2-5 s

    Usage (in brain.py tick loop)::

        self._gaze = GazeDynamics()
        # each tick:
        self._gaze.tick(self.tick_count, is_speaking, self._robot_controller)
    """

    ATTEND = "attend"
    GLANCE = "glance"
    BLINK = "blink"

    # Approximate ticks per second (brain runs ~41 Hz; GazeDynamics assumes ~40)
    _TICK_HZ: int = 40

    def __init__(self) -> None:
        self._state: str = self.ATTEND
        self._state_ticks: int = 0          # ticks spent in current state
        self._next_switch: int = 0          # tick at which to consider switching
        self._blink_next: int = 0           # tick for next blink
        self._glance_yaw_offset: float = 0.0
        self._last_tick: int = 0
        # ── Idle-life micro-motions ───────────────────────────
        self._breath_tick: int = 0          # accumulated ticks for breathing sine
        self._breath_base_pitch: float = 89.0
        self._idle_sway_tick: int = 0       # accumulated ticks for idle yaw sway

    # ── Public API ────────────────────────────────────────────

    def tick(self, tick: int, is_speaking: bool, rc: "RobotController") -> None:
        """Advance gaze state machine and apply to RobotController targets."""
        if tick == self._last_tick:
            return  # guard against double-call
        self._last_tick = tick
        self._state_ticks += 1

        # ── Blink check (independent of gaze state) ───────────
        if self._blink_next == 0:
            self._blink_next = tick + self._rand_blink_interval()
        if tick >= self._blink_next and self._state != self.BLINK:
            self._enter_blink(rc)
            self._blink_next = tick + self._rand_blink_interval()
            return  # skip gaze switch this tick

        # ── State machine ─────────────────────────────────────
        if self._state == self.BLINK:
            # Blink lasts ~3 ticks (~75 ms)
            if self._state_ticks >= 3:
                self._enter_attend(rc)

        elif self._state == self.ATTEND:
            if tick >= self._next_switch:
                # During speaking: look away 40%; listening: 20%
                _glance_prob = 0.40 if is_speaking else 0.20
                if random.random() < _glance_prob:
                    self._enter_glance(rc, is_speaking)
                else:
                    # Extend attend window
                    self._next_switch = tick + self._rand_attend_ticks(is_speaking)

        elif self._state == self.GLANCE:
            if tick >= self._next_switch:
                self._enter_attend(rc)

        # ── Idle-life: breathing micro-pitch (F) ──────────────
        # Only during non-blink states; ±0.5° sine wave ~4-5 s cycle
        if self._state != self.BLINK:
            import math as _math
            self._breath_tick += 1
            _breath_phase = self._breath_tick / (4.5 * self._TICK_HZ)
            _breath_offset = 0.5 * _math.sin(2 * _math.pi * _breath_phase)
            _new_pitch = self._breath_base_pitch + _breath_offset
            rc._set_target("head_pitch", _new_pitch)

            # Idle sway: very slow ±1° yaw drift every ~3 s
            self._idle_sway_tick += 1
            if self._idle_sway_tick % (3 * self._TICK_HZ) == 0:
                _sway_offset = random.uniform(-1.0, 1.0)
                rc._set_target("head_yaw", 90.0 + _sway_offset)

    # ── State transitions ─────────────────────────────────────

    def _enter_attend(self, rc: "RobotController") -> None:
        self._state = self.ATTEND
        self._state_ticks = 0
        self._next_switch = self._last_tick + self._rand_attend_ticks(False)
        # Eyes straight ahead at interlocutor
        rc._set_target("eye_yaw", 90.0)
        rc._set_target("eye_pitch", 89.0)
        rc._set_target("head_yaw", 90.0)
        rc._set_target("head_pitch", 89.0)
        # Eyes open (normal lid position)
        rc._set_target("left_upper_lid", 94.0)
        rc._set_target("left_lower_lid", 86.0)
        rc._set_target("right_upper_lid", 86.0)
        rc._set_target("right_lower_lid", 94.0)

    def _enter_glance(self, rc: "RobotController", is_speaking: bool) -> None:
        self._state = self.GLANCE
        self._state_ticks = 0
        _dur_s = random.uniform(0.6, 1.8)
        self._next_switch = self._last_tick + int(_dur_s * self._TICK_HZ)
        # Slight yaw/pitch offset — look slightly to one side (cognitive recall)
        _yaw_off = random.choice([-1, 1]) * random.uniform(8.0, 22.0)
        _pitch_off = random.uniform(-4.0, 8.0)  # slight upward often = recall
        rc._set_target("eye_yaw", max(68.0, min(112.0, 90.0 + _yaw_off)))
        rc._set_target("eye_pitch", max(78.0, min(100.0, 90.0 + _pitch_off)))

    def _enter_blink(self, rc: "RobotController") -> None:
        self._state = self.BLINK
        self._state_ticks = 0
        # Close lids (towards each other)
        rc._set_target("left_upper_lid", 82.0)   # upper moves down
        rc._set_target("left_lower_lid", 98.0)   # lower moves up
        rc._set_target("right_upper_lid", 98.0)
        rc._set_target("right_lower_lid", 82.0)

    # ── Randomised timing ─────────────────────────────────────

    def _rand_attend_ticks(self, is_speaking: bool) -> int:
        # Speaking: 2-4 s; listening: 2.5-5 s
        if is_speaking:
            return int(random.uniform(2.0, 4.0) * self._TICK_HZ)
        return int(random.uniform(2.5, 5.0) * self._TICK_HZ)

    def _rand_blink_interval(self) -> int:
        return int(random.uniform(3.0, 6.5) * self._TICK_HZ)
