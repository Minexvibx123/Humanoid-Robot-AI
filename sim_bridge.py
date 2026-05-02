"""
sim_bridge.py — Simulation / Real-Hardware Bridge

Provides a single BodyInterface that abstracts between:
  A) Real InMoov body  → commands go over ArduinoSerialLink
  B) Simulated body    → commands update in-memory model only

Both modes expose an identical API so the entire cognitive stack
(brain, consciousness, task_executive, skill_library) never knows
which backend is active.  Switching is done at startup.

Also provides a lightweight physics step for simulation mode:
  • Joint inertia (smooth movement toward targets)
  • Gravity bias (arms droop without commands)
  • Noise (sensor jitter)
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from body_schema import BodySchema
    from robot_controller import JointState, JointTargets
    from robot_serial import ArduinoSerialLink
    from safety_supervisor import SafetySupervisor
    from telemetry_bus import TelemetryBus, TelemetryFrame


class BackendMode(Enum):
    REAL = "real"
    SIMULATED = "simulated"


@dataclass
class SimJoint:
    """Simulated joint with inertia, gravity, contact and delay."""

    name: str
    current_deg: float = 90.0
    target_deg: float = 90.0
    speed: float = 4.0  # deg/tick
    gravity_bias: float = 0.0  # downward drift deg/tick
    noise_std: float = 0.15  # sensor noise
    load: float = 0.0
    temperature: float = 28.0
    stall: bool = False
    # Contact / interaction state
    contact_force: float = 0.0  # contact load [0,1]
    command_delay: int = 0  # ticks of command latency remaining

    _pending_target: float = 90.0
    _delay_ticks: int = 2  # base command delay (realistic servo lag)

    def set_target(self, deg: float) -> None:
        """Set target with realistic command delay."""
        self._pending_target = deg
        self.command_delay = self._delay_ticks

    def step(self) -> None:
        """One tick of simulation physics."""
        # Apply command delay
        if self.command_delay > 0:
            self.command_delay -= 1
        else:
            self.target_deg = self._pending_target

        diff = self.target_deg - self.current_deg
        max_step = self.speed

        # Load-dependent speed reduction (loaded joints move slower)
        effective_speed = max_step * max(0.3, 1.0 - self.contact_force * 0.6)

        if abs(diff) > effective_speed:
            self.current_deg += effective_speed * (1 if diff > 0 else -1)
        else:
            self.current_deg = self.target_deg

        # Gravity bias (arms droop slightly)
        self.current_deg += self.gravity_bias

        # Sensor noise
        self.current_deg += random.gauss(0, self.noise_std)

        # Load model: higher when far from rest + contact force
        base_load = min(1.0, abs(diff) / 60.0 * 0.3)
        self.load = min(1.0, base_load + self.contact_force * 0.5)

        # Stall detection: high load + no movement
        self.stall = self.load > 0.8 and abs(diff) > 2.0

        # Thermal model (slow heat-up under load, faster under stall)
        heat_rate = 0.008 + (0.015 if self.stall else 0.0)
        self.temperature += self.load * heat_rate - 0.004
        self.temperature = max(22.0, min(65.0, self.temperature))

        # Contact force decays naturally
        self.contact_force = max(0.0, self.contact_force - 0.02)


# Gravity biases for arm joints (droop when unpowered)
_GRAVITY_MAP = {
    "left_shoulder": 0.05,
    "right_shoulder": 0.05,
    "left_elbow": 0.08,
    "right_elbow": 0.08,
    "left_omoplate": 0.03,
    "right_omoplate": 0.03,
}


class SimulatedBody:
    """Lightweight in-memory body simulation with contact and perturbation."""

    def __init__(self) -> None:
        self.joints: Dict[str, SimJoint] = {}
        self.battery_pct: float = 100.0
        self.imu_roll: float = 0.0
        self.imu_pitch: float = 0.0
        self._tick_count = 0
        self._contact_events: List[Dict[str, object]] = []

    def init_from_schema(self, body: "BodySchema") -> None:
        """Create sim joints matching the body_schema layout."""
        for name, jnode in body.joints.items():
            grav = _GRAVITY_MAP.get(name, 0.0)
            self.joints[name] = SimJoint(
                name=name,
                current_deg=jnode.current_deg,
                target_deg=jnode.target_deg,
                speed=jnode.max_speed,
                gravity_bias=grav,
            )

    def set_targets(self, targets: Dict[str, float]) -> None:
        for name, deg in targets.items():
            j = self.joints.get(name)
            if j:
                j.set_target(deg)

    def inject_contact(self, joint_name: str, force: float = 0.5) -> None:
        """Simulate external contact/load spike on a joint."""
        j = self.joints.get(joint_name)
        if j:
            j.contact_force = min(1.0, j.contact_force + force)
            self._contact_events.append(
                {"joint": joint_name, "force": force, "tick": self._tick_count}
            )

    def inject_perturbation(self, joint_name: str, deg_offset: float) -> None:
        """Simulate external disturbance (bump, push) as position offset."""
        j = self.joints.get(joint_name)
        if j:
            j.current_deg += deg_offset

    def step(self) -> None:
        """Advance simulation by one tick."""
        self._tick_count += 1
        for j in self.joints.values():
            j.step()
        # IMU drift
        self.imu_roll = random.gauss(0, 0.3)
        self.imu_pitch = random.gauss(0, 0.3)
        # Battery drain
        total_load = sum(j.load for j in self.joints.values())
        self.battery_pct = max(0.0, self.battery_pct - 0.0001 - total_load * 0.00005)
        # Prune old contact events
        self._contact_events = [
            e for e in self._contact_events if self._tick_count - e["tick"] < 30
        ]

    def read_positions(self) -> Dict[str, float]:
        return {name: j.current_deg for name, j in self.joints.items()}

    def read_loads(self) -> Dict[str, float]:
        return {name: j.load for name, j in self.joints.items()}

    def read_temps(self) -> Dict[str, float]:
        return {name: j.temperature for name, j in self.joints.items()}


# ─────────────────────────────────────────────────────────────
# Body Interface (unified API)
# ─────────────────────────────────────────────────────────────


class BodyInterface:
    """
    Unified interface for sim or real body.
    The brain/consciousness only interact through this.
    """

    def __init__(
        self,
        mode: BackendMode = BackendMode.SIMULATED,
        body_schema: Optional["BodySchema"] = None,
        serial_link: Optional["ArduinoSerialLink"] = None,
        safety: Optional["SafetySupervisor"] = None,
    ) -> None:
        self.mode = mode
        self._body = body_schema
        self._serial = serial_link
        self._safety = safety
        self._sim: Optional[SimulatedBody] = None

        if mode == BackendMode.SIMULATED:
            self._sim = SimulatedBody()
            if body_schema:
                self._sim.init_from_schema(body_schema)

    @property
    def is_real(self) -> bool:
        return self.mode == BackendMode.REAL

    @property
    def is_simulated(self) -> bool:
        return self.mode == BackendMode.SIMULATED

    # ── Command ──────────────────────────────────────────────

    def send_targets(self, targets: Dict[str, float]) -> bool:
        """Send joint targets. Safety gating applied if available."""
        gated = targets
        if self._safety and self._body:
            gated = self._safety.gate_targets(targets, self._body)

        if self.mode == BackendMode.REAL:
            return self._send_real(gated)
        elif self._sim:
            self._sim.set_targets(gated)
            return True
        return False

    def send_frame(self, frame: str) -> bool:
        """Send a raw serial frame (real mode only)."""
        if self.mode == BackendMode.REAL and self._serial:
            return self._serial.send_frame(frame)
        return self.mode == BackendMode.SIMULATED  # no-op in sim

    # ── Sense ────────────────────────────────────────────────

    def read_joint_positions(self) -> Dict[str, float]:
        if self._sim:
            return self._sim.read_positions()
        # Real: positions come from body_schema (updated via controller)
        if self._body:
            return {n: j.current_deg for n, j in self._body.joints.items()}
        return {}

    def read_joint_loads(self) -> Dict[str, float]:
        if self._sim:
            return self._sim.read_loads()
        if self._body:
            return {n: j.load for n, j in self._body.joints.items()}
        return {}

    def read_stalls(self) -> Dict[str, bool]:
        if self._sim:
            return {name: j.stall for name, j in self._sim.joints.items()}
        if self._body:
            return {n: j.stall for n, j in self._body.joints.items()}
        return {}

    def read_joint_temps(self) -> Dict[str, float]:
        if self._sim:
            return self._sim.read_temps()
        if self._body:
            return {n: j.temperature for n, j in self._body.joints.items()}
        return {}

    def battery_pct(self) -> float:
        if self._sim:
            return self._sim.battery_pct
        return 100.0  # real: would come from telemetry

    def imu(self) -> tuple:
        if self._sim:
            return (self._sim.imu_roll, self._sim.imu_pitch)
        return (0.0, 0.0)

    # ── Tick ─────────────────────────────────────────────────

    def tick(self) -> None:
        """Advance body by one tick (only does work in sim mode)."""
        if self._sim:
            self._sim.step()

    def build_telemetry_frame(self) -> Optional["TelemetryFrame"]:
        """Build a TelemetryFrame from current body state."""
        from telemetry_bus import TelemetryFrame

        return TelemetryFrame(
            joint_positions=self.read_joint_positions(),
            joint_loads=self.read_joint_loads(),
            joint_temps=self.read_joint_temps(),
            joint_stalls=self.read_stalls(),
            imu_roll=self.imu()[0],
            imu_pitch=self.imu()[1],
            battery_voltage=self.battery_pct() * 0.12,  # ~12V at 100%
            serial_connected=(
                self._serial._state.connected if self._serial else self.is_simulated
            ),
        )

    # ── Real backend ─────────────────────────────────────────

    def _send_real(self, targets: Dict[str, float]) -> bool:
        """Apply targets to the real body via serial."""
        if not self._serial:
            return False
        # Targets are applied via robot_controller → serial frame
        # This path is used for direct body interface commands
        if self._body:
            for name, deg in targets.items():
                j = self._body.joints.get(name)
                if j:
                    j.set_target(deg)
        return True

    # ── Status ───────────────────────────────────────────────

    def describe(self) -> str:
        if self._sim:
            return (
                f"sim_body joints={len(self._sim.joints)} "
                f"batt={self._sim.battery_pct:.1f}%"
            )
        conn = ""
        if self._serial:
            conn = "connected" if self._serial._state.connected else "disconnected"
        return f"real_body serial={conn}"
