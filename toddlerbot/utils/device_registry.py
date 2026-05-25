"""Per-robot USB device registry.

Maps logical device names (e.g. "fsr_board", "dynamixel_bus") to physical
USB serial devices identified by VID/PID/serial_number. Replaces hardcoded
device paths that are not portable across machines or operating systems.

Configuration lives at:
    toddlerbot/descriptions/<robot_name>/devices.json

Populate it interactively via:
    python -m toddlerbot.tools.setup_devices --robot <robot_name>
"""

import json
import os
from typing import Dict, Optional

import serial.tools.list_ports as list_ports

KIND_DYNAMIXEL_BUS = "dynamixel_bus"
KIND_FSR_BOARD = "fsr_board"

DEVICES_FILENAME = "devices.json"
SCHEMA_VERSION = 1


def _description_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "descriptions"
    )


def devices_config_path(robot_name: str) -> str:
    return os.path.join(_description_dir(), robot_name, DEVICES_FILENAME)


def load_registry(robot_name: str) -> Dict[str, Dict]:
    """Returns {kind -> {vid, pid, serial_number, description}}, empty if absent."""
    path = devices_config_path(robot_name)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("devices", {})


def save_registry(robot_name: str, devices: Dict[str, Dict]) -> str:
    path = devices_config_path(robot_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"version": SCHEMA_VERSION, "devices": devices}, f, indent=2)
    return path


def _hex_to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(value, 16)


def find_device_path(robot_name: str, kind: str) -> str:
    """Returns the live device path for a registered device kind.

    Matches by (VID, PID, serial_number). Falls back to (VID, PID) only if no
    serial_number was recorded (some clones omit it). Raises if the kind is
    not registered or no live USB device matches.
    """
    registry = load_registry(robot_name)
    if kind not in registry:
        raise KeyError(
            f"Device '{kind}' not registered for robot '{robot_name}'. "
            f"Run: python -m toddlerbot.tools.setup_devices --robot {robot_name}"
        )

    spec = registry[kind]
    vid = _hex_to_int(spec.get("vid"))
    pid = _hex_to_int(spec.get("pid"))
    sn = spec.get("serial_number")

    matches = []
    for p in list_ports.comports():
        if p.vid != vid or p.pid != pid:
            continue
        if sn is not None and p.serial_number != sn:
            continue
        matches.append(p.device)

    if not matches:
        raise ConnectionError(
            f"Device '{kind}' for robot '{robot_name}' not found "
            f"(VID={spec.get('vid')}, PID={spec.get('pid')}, SN={sn}). "
            "Is it plugged in?"
        )
    return sorted(matches)[0]
