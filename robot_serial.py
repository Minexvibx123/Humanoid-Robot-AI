from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SerialState:
    available: bool = False
    connected: bool = False
    port: str = ""
    baudrate: int = 115200
    last_error: str = ""
    last_frame: str = ""


class ArduinoSerialLink:
    def __init__(self) -> None:
        self._serial_mod = None
        self._serial = None
        self._state = SerialState()
        try:
            import serial  # type: ignore

            self._serial_mod = serial
            self._state.available = True
        except Exception as exc:
            self._state.last_error = f"pyserial unavailable: {exc}"

    def list_ports(self) -> List[str]:
        if not self._serial_mod:
            return []
        try:
            from serial.tools import list_ports  # type: ignore

            return [port.device for port in list_ports.comports()]
        except Exception:
            return []

    def connect(self, port: str, baudrate: int = 115200, timeout: float = 0.02) -> bool:
        self.disconnect()
        if not self._serial_mod:
            self._state.last_error = "pyserial unavailable"
            return False
        try:
            self._serial = self._serial_mod.Serial(
                port=port, baudrate=baudrate, timeout=timeout, write_timeout=timeout
            )
            self._state.connected = True
            self._state.port = port
            self._state.baudrate = baudrate
            self._state.last_error = ""
            return True
        except Exception as exc:
            self._serial = None
            self._state.connected = False
            self._state.port = port
            self._state.baudrate = baudrate
            self._state.last_error = str(exc)
            return False

    def disconnect(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self._state.connected = False

    def send_frame(self, frame: str) -> bool:
        self._state.last_frame = frame.strip()
        if not self._serial or not self._state.connected:
            return False
        try:
            self._serial.write(frame.encode("ascii", errors="ignore"))
            self._serial.flush()
            self._state.last_error = ""
            return True
        except Exception as exc:
            self._state.connected = False
            self._state.last_error = str(exc)
            return False

    def snapshot(self) -> Dict[str, object]:
        return {
            "available": self._state.available,
            "connected": self._state.connected,
            "port": self._state.port,
            "baudrate": self._state.baudrate,
            "last_error": self._state.last_error,
            "last_frame": self._state.last_frame,
            "ports": self.list_ports(),
        }
