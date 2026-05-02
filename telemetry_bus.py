"""
telemetry_bus.py — Structured Sensorimotor Feedback Bus

Collects and distributes hardware telemetry from the Arduino link and
sensor stack into structured events that feed the body schema, safety
supervisor, world state, emotion system, and episodic memory.

Design:
  • TelemetryFrame: one snapshot of all hardware signals
  • SensorEvent: typed perception event (person_seen, touch, joint_error …)
  • TelemetryBus: aggregates raw data, emits events, keeps rolling history

The bus replaces ad-hoc string injection with structured data flow.
Consumers register for event types and receive callbacks.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Set

# ─────────────────────────────────────────────────────────────
# Event types
# ─────────────────────────────────────────────────────────────

# Perception events
EVENT_PERSON_SEEN = "person_seen"
EVENT_PERSON_LOST = "person_lost"
EVENT_SPEAKER_ACTIVE = "speaker_active"
EVENT_OBJECT_DETECTED = "object_detected"
EVENT_GESTURE_DETECTED = "gesture_detected"
EVENT_FACE_DETECTED = "face_detected"

# Body events
EVENT_JOINT_ERROR_HIGH = "joint_error_high"
EVENT_JOINT_STALL = "joint_stall"
EVENT_MOTOR_OVERTEMP = "motor_overtemp"
EVENT_TOUCH = "touch"
EVENT_IMU_TILT = "imu_tilt"
EVENT_COLLISION_RISK = "collision_risk"

# System events
EVENT_BATTERY_LOW = "battery_low"
EVENT_SERIAL_LOST = "serial_lost"
EVENT_SERIAL_CONNECTED = "serial_connected"
EVENT_SKILL_STARTED = "skill_started"
EVENT_SKILL_COMPLETED = "skill_completed"
EVENT_SKILL_FAILED = "skill_failed"
EVENT_SAFETY_STOP = "safety_stop"


@dataclass
class SensorEvent:
    """Single typed perception/body/system event."""

    kind: str  # EVENT_* constant
    tick: int = 0
    timestamp: float = 0.0  # perf_counter
    source: str = ""  # originating subsystem
    data: Dict[str, object] = field(default_factory=dict)
    severity: float = 0.0  # [0,1] urgency

    def __str__(self) -> str:
        d = " ".join(f"{k}={v}" for k, v in self.data.items())
        return f"[{self.kind}] t={self.tick} sev={self.severity:.2f} {d}"


# ─────────────────────────────────────────────────────────────
# Telemetry frame (raw hardware snapshot)
# ─────────────────────────────────────────────────────────────


@dataclass
class TelemetryFrame:
    """One snapshot of all measurable hardware signals."""

    tick: int = 0
    timestamp: float = 0.0

    # Joint feedback (joint_name → value)
    joint_positions: Dict[str, float] = field(default_factory=dict)
    joint_loads: Dict[str, float] = field(default_factory=dict)
    joint_temps: Dict[str, float] = field(default_factory=dict)
    joint_stalls: Dict[str, bool] = field(default_factory=dict)

    # IMU
    imu_pitch: float = 0.0
    imu_roll: float = 0.0
    imu_yaw: float = 0.0
    imu_accel_g: float = 0.0  # total acceleration

    # Touch / force
    touch_sensors: Dict[str, float] = field(default_factory=dict)  # location → pressure

    # Power
    battery_voltage: float = 12.0
    total_current_a: float = 0.0

    # Serial link state
    serial_connected: bool = True
    serial_latency_ms: float = 0.0


# ─────────────────────────────────────────────────────────────
# Telemetry Bus
# ─────────────────────────────────────────────────────────────

EventCallback = Callable[[SensorEvent], None]


class TelemetryBus:
    """
    Central event bus for all sensorimotor data.

    Flow:
      1. Hardware adapters call push_frame() with raw telemetry
      2. Vision/audio subsystems call push_event() for perception events
      3. Bus computes deltas, detects anomalies, emits SensorEvents
      4. Registered consumers receive events via callbacks
      5. Rolling history kept for learning / replay
    """

    _HISTORY_LEN = 500

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[EventCallback]] = {}
        self._event_log: Deque[SensorEvent] = collections.deque(
            maxlen=self._HISTORY_LEN
        )
        self._latest_frame: Optional[TelemetryFrame] = None
        self._prev_frame: Optional[TelemetryFrame] = None
        self._pending_events: Deque[SensorEvent] = collections.deque(maxlen=200)

        # Tracking state for delta detection
        self._person_visible: bool = False
        self._serial_was_connected: bool = True

        # Configurable thresholds
        self.joint_error_threshold: float = 0.25  # normalised error → event
        self.joint_temp_threshold: float = 0.7  # normalised temp → event
        self.imu_tilt_threshold: float = 15.0  # degrees from vertical
        self.battery_low_voltage: float = 10.5  # volts

    # ── Subscription ─────────────────────────────────────────

    def subscribe(self, event_kind: str, callback: EventCallback) -> None:
        self._subscribers.setdefault(event_kind, []).append(callback)

    def subscribe_all(self, callback: EventCallback) -> None:
        """Receive every event type."""
        self._subscribers.setdefault("*", []).append(callback)

    # ── Ingestion ────────────────────────────────────────────

    def push_frame(self, frame: TelemetryFrame) -> None:
        """Accept a new hardware telemetry frame and derive events."""
        self._prev_frame = self._latest_frame
        self._latest_frame = frame
        self._derive_body_events(frame)

    def push_event(self, event: SensorEvent) -> None:
        """Accept an externally generated event (perception, skill, …)."""
        if event.timestamp == 0.0:
            event.timestamp = time.perf_counter()
        self._pending_events.append(event)

    def push_detection_events(
        self,
        detections: List[str],
        detection_targets: List[Dict[str, object]],
        tick: int,
    ) -> None:
        """Convert raw vision detection strings into structured events."""
        now = time.perf_counter()
        sees_person = False
        for d in detections:
            if d in ("person", "face") or d.startswith("face"):
                sees_person = True
                # Find matching target data
                target_data = {}
                for t in detection_targets:
                    if str(t.get("label", "")) in ("face", "person"):
                        target_data = dict(t)
                        break
                self.push_event(
                    SensorEvent(
                        kind=EVENT_FACE_DETECTED if "face" in d else EVENT_PERSON_SEEN,
                        tick=tick,
                        timestamp=now,
                        source="vision",
                        data=target_data,
                        severity=0.3,
                    )
                )
            elif d.startswith("hand:"):
                gesture = d.split(":", 1)[1]
                self.push_event(
                    SensorEvent(
                        kind=EVENT_GESTURE_DETECTED,
                        tick=tick,
                        timestamp=now,
                        source="vision",
                        data={"gesture": gesture},
                        severity=0.2,
                    )
                )
            else:
                self.push_event(
                    SensorEvent(
                        kind=EVENT_OBJECT_DETECTED,
                        tick=tick,
                        timestamp=now,
                        source="vision",
                        data={"label": d},
                        severity=0.1,
                    )
                )

        # Person presence transitions
        if sees_person and not self._person_visible:
            self.push_event(
                SensorEvent(
                    kind=EVENT_PERSON_SEEN,
                    tick=tick,
                    timestamp=now,
                    source="vision",
                    data={"transition": "appeared"},
                    severity=0.5,
                )
            )
        elif not sees_person and self._person_visible:
            self.push_event(
                SensorEvent(
                    kind=EVENT_PERSON_LOST,
                    tick=tick,
                    timestamp=now,
                    source="vision",
                    data={"transition": "lost"},
                    severity=0.4,
                )
            )
        self._person_visible = sees_person

    def push_speech_event(self, text: str, tick: int) -> None:
        """Emit a speaker_active event from speech recognition."""
        if text.strip():
            self.push_event(
                SensorEvent(
                    kind=EVENT_SPEAKER_ACTIVE,
                    tick=tick,
                    timestamp=time.perf_counter(),
                    source="speech",
                    data={"text": text},
                    severity=0.4,
                )
            )

    # ── Flush & distribute ───────────────────────────────────

    def flush(self) -> List[SensorEvent]:
        """Distribute all pending events to subscribers. Returns the flushed list."""
        flushed: List[SensorEvent] = []
        while self._pending_events:
            event = self._pending_events.popleft()
            self._event_log.append(event)
            flushed.append(event)
            # Dispatch to kind-specific subscribers
            for cb in self._subscribers.get(event.kind, []):
                try:
                    cb(event)
                except Exception:
                    pass
            # Dispatch to wildcard subscribers
            for cb in self._subscribers.get("*", []):
                try:
                    cb(event)
                except Exception:
                    pass
        return flushed

    # ── Queries ──────────────────────────────────────────────

    @property
    def latest_frame(self) -> Optional[TelemetryFrame]:
        return self._latest_frame

    def recent_events(
        self, n: int = 20, kind: Optional[str] = None
    ) -> List[SensorEvent]:
        if kind is None:
            return list(self._event_log)[-n:]
        return [e for e in self._event_log if e.kind == kind][-n:]

    def has_recent(
        self, kind: str, within_ticks: int = 50, current_tick: int = 0
    ) -> bool:
        cutoff = current_tick - within_ticks
        return any(e.kind == kind and e.tick >= cutoff for e in self._event_log)

    # ── Internal event derivation ────────────────────────────

    def _derive_body_events(self, frame: TelemetryFrame) -> None:
        """Detect anomalies in hardware telemetry and emit events."""
        now = time.perf_counter()

        # Joint errors
        if self._prev_frame and frame.joint_positions:
            for name, target in frame.joint_positions.items():
                load = frame.joint_loads.get(name, 0.0)
                if load > self.joint_error_threshold:
                    self.push_event(
                        SensorEvent(
                            kind=EVENT_JOINT_ERROR_HIGH,
                            tick=frame.tick,
                            timestamp=now,
                            source="body",
                            data={"joint": name, "load": load},
                            severity=min(1.0, load),
                        )
                    )

        # Stalls
        for name, stalled in frame.joint_stalls.items():
            if stalled:
                self.push_event(
                    SensorEvent(
                        kind=EVENT_JOINT_STALL,
                        tick=frame.tick,
                        timestamp=now,
                        source="body",
                        data={"joint": name},
                        severity=0.8,
                    )
                )

        # Temperature
        for name, temp in frame.joint_temps.items():
            if temp > self.joint_temp_threshold:
                self.push_event(
                    SensorEvent(
                        kind=EVENT_MOTOR_OVERTEMP,
                        tick=frame.tick,
                        timestamp=now,
                        source="body",
                        data={"joint": name, "temp": temp},
                        severity=0.7,
                    )
                )

        # IMU tilt
        tilt = max(abs(frame.imu_pitch), abs(frame.imu_roll))
        if tilt > self.imu_tilt_threshold:
            self.push_event(
                SensorEvent(
                    kind=EVENT_IMU_TILT,
                    tick=frame.tick,
                    timestamp=now,
                    source="body",
                    data={
                        "pitch": frame.imu_pitch,
                        "roll": frame.imu_roll,
                        "tilt": tilt,
                    },
                    severity=min(1.0, tilt / 45.0),
                )
            )

        # Touch
        for location, pressure in frame.touch_sensors.items():
            if pressure > 0.1:
                self.push_event(
                    SensorEvent(
                        kind=EVENT_TOUCH,
                        tick=frame.tick,
                        timestamp=now,
                        source="body",
                        data={"location": location, "pressure": pressure},
                        severity=min(1.0, pressure),
                    )
                )

        # Battery
        if frame.battery_voltage < self.battery_low_voltage:
            self.push_event(
                SensorEvent(
                    kind=EVENT_BATTERY_LOW,
                    tick=frame.tick,
                    timestamp=now,
                    source="power",
                    data={"voltage": frame.battery_voltage},
                    severity=0.9,
                )
            )

        # Serial state transitions
        if not frame.serial_connected and self._serial_was_connected:
            self.push_event(
                SensorEvent(
                    kind=EVENT_SERIAL_LOST,
                    tick=frame.tick,
                    timestamp=now,
                    source="serial",
                    severity=0.9,
                )
            )
        elif frame.serial_connected and not self._serial_was_connected:
            self.push_event(
                SensorEvent(
                    kind=EVENT_SERIAL_CONNECTED,
                    tick=frame.tick,
                    timestamp=now,
                    source="serial",
                    severity=0.2,
                )
            )
        self._serial_was_connected = frame.serial_connected
