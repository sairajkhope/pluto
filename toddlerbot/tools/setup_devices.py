"""Interactive setup: register USB devices for a robot.

Run once per robot (or whenever USB hardware changes):
    python -m toddlerbot.tools.setup_devices --robot toddlerbot_2xc

Lists currently-connected USB serial devices, prompts the user to identify
each known device kind, and writes the result to:
    toddlerbot/descriptions/<robot>/devices.json
"""

import argparse
from typing import Dict, List

import serial.tools.list_ports as list_ports
from serial.tools.list_ports_common import ListPortInfo

from toddlerbot.utils.device_registry import (
    KIND_DYNAMIXEL_BUS,
    KIND_FSR_BOARD,
    devices_config_path,
    load_registry,
    save_registry,
)

KNOWN_KINDS = [KIND_DYNAMIXEL_BUS, KIND_FSR_BOARD]


def _list_usb_serial_ports() -> List[ListPortInfo]:
    return [p for p in list_ports.comports() if p.vid is not None]


def _format_port(idx: int, p: ListPortInfo) -> str:
    return (
        f"[{idx}] {p.device}\n"
        f"    VID={hex(p.vid)} PID={hex(p.pid)}\n"
        f"    serial_number={p.serial_number}\n"
        f"    description={p.description}"
    )


def _port_to_spec(p: ListPortInfo) -> Dict:
    return {
        "vid": hex(p.vid),
        "pid": hex(p.pid),
        "serial_number": p.serial_number,
        "description": p.description,
    }


def _prompt_for_kind(kind: str, ports: List[ListPortInfo]) -> Dict:
    """Asks user to identify which port is the given device kind.

    Returns the spec dict to record, or {} to skip, or {"__remove__": True}
    to delete an existing entry.
    """
    while True:
        choice = input(
            f"Which port is the '{kind}'? "
            "(integer, 's' to skip, 'r' to remove existing): "
        ).strip().lower()

        if choice == "s":
            return {}
        if choice == "r":
            return {"__remove__": True}
        try:
            idx = int(choice)
            return _port_to_spec(ports[idx])
        except (ValueError, IndexError):
            print(f"  invalid input '{choice}', try again.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register USB devices for a robot."
    )
    parser.add_argument(
        "--robot",
        required=True,
        help="Robot name (must match a directory in toddlerbot/descriptions/).",
    )
    args = parser.parse_args()

    ports = _list_usb_serial_ports()
    if not ports:
        print("No USB serial devices detected. Plug in your hardware and try again.")
        return

    print(f"\nDetected {len(ports)} USB serial device(s):\n")
    for i, p in enumerate(ports):
        print(_format_port(i, p))
        print()

    devices = dict(load_registry(args.robot))
    for kind in KNOWN_KINDS:
        if kind in devices:
            print(f"(currently registered: '{kind}' -> {devices[kind].get('description')})")
        result = _prompt_for_kind(kind, ports)
        if not result:
            continue
        if result.get("__remove__"):
            devices.pop(kind, None)
            print(f"  removed '{kind}'.")
            continue
        devices[kind] = result
        print(f"  registered '{kind}' -> {result['description']}")

    path = save_registry(args.robot, devices)
    print(f"\nSaved registry to {path}")
    print(f"({devices_config_path(args.robot)})")


if __name__ == "__main__":
    main()
