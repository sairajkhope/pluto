"""Tool for calibrating robot joint zero positions."""

import argparse
import os
import platform
from typing import List

import numpy as np
import yaml

from toddlerbot.actuation import dynamixel_controller as dynamixel_cpp
from toddlerbot.sim.robot import Robot

# This script is used to calibrate the zero points of the Dynamixel motors.


def main(robot: Robot, parts: List[str]):
    """Calibrates the robot's motors based on specified parts and updates the configuration.

    This function prompts the user to confirm the installation of calibration parts,
    calibrates the Dynamixel motors if present, and updates the motor configuration
    file with the initial positions of the specified parts.

    Args:
        robot (Robot): The robot instance containing configuration and joint attributes.
        parts (List[str]): A list of parts to calibrate. Can include specific parts
            like 'left_arm', 'right_arm', or 'all' to calibrate all parts.

    Raises:
        ValueError: If an invalid part is specified in the `parts` list.
        FileNotFoundError: If the motor configuration file is not found.
    """
    while True:
        response = input("Have you installed the calibration parts? (y/n) > ")
        response = response.strip().lower()
        if response == "y" or response[0] == "y":
            break
        if response == "n" or response[0] == "n":
            return

        print("Please answer 'yes' or 'no'.")

    # On Mac, need full path; on Linux, just device name
    port_name = "/dev/tty.usbserial-FTAK8D39" if platform.system() == "Darwin" else "ttyUSB0"
    
    controllers = dynamixel_cpp.create_controllers(
        port_name,
        robot.motor_kp_real,
        robot.motor_kd_real,
        np.zeros(robot.nu, dtype=np.float32),
        ["extended_position"] * robot.nu,
        2000000,
        1,
    )
    dynamixel_cpp.initialize(controllers)

    motor_ids = dynamixel_cpp.get_motor_ids(controllers)
    motor_state = dynamixel_cpp.get_motor_states(controllers, -1)

    if len(motor_state) == 0:
        raise RuntimeError("No motors found. Please check the connections.")

    all_motor_ids = []
    all_motor_pos = []
    for key in sorted(motor_ids.keys()):
        ids = motor_ids[key]
        pos = motor_state[key]["pos"]
        # Append all ids and corresponding data
        all_motor_ids.extend(ids)
        all_motor_pos.extend(pos)

    # Convert to numpy arrays
    all_motor_ids = np.array(all_motor_ids)
    motor_pos = np.array(all_motor_pos, dtype=np.float32)
    # Sort everything by motor_ids
    sort_idx = np.argsort(all_motor_ids)
    motor_pos = motor_pos[sort_idx]

    zero_pos = []
    for i, (name, pos) in enumerate(zip(robot.motor_ordering, motor_pos)):
        zero_pos.append(pos)

    dynamixel_cpp.close(controllers)

    # Generate the motor mask based on the specified parts
    all_parts = {
        "left_arm": [16, 17, 18, 19, 20, 21, 22],
        "right_arm": [23, 24, 25, 26, 27, 28, 29],
        "left_gripper": [30],
        "right_gripper": [31],
        "hip": [2, 3, 4, 5, 6, 10, 11, 12],
        "knee": [7, 13],
        "left_ankle": [8, 9],
        "right_ankle": [14, 15],
        "neck": [0, 1],
    }
    if "all" in parts:
        motor_mask = robot.motor_ids
    else:
        motor_mask = []
        for part in parts:
            if part not in all_parts:
                raise ValueError(f"Invalid part: {part}")

            motor_mask.extend(all_parts[part])

    motor_config_path = os.path.join(
        "toddlerbot", "descriptions", robot.name, "motors.yml"
    )
    if os.path.exists(motor_config_path):
        motor_config = yaml.safe_load(open(motor_config_path, "r"))
        if motor_config is None:
            motor_config = {"motors": {}}
    else:
        motor_config = {"motors": {}}

    max_name_len = max(
        len(name)
        for id, name in zip(robot.motor_ids, robot.motor_ordering)
        if id in motor_mask
    )
    print(f"Updating motor zero point at {motor_config_path}...")
    for id, name, pos in zip(robot.motor_ids, robot.motor_ordering, zero_pos):
        if id in motor_mask:
            if name not in motor_config["motors"]:
                motor_config["motors"][name] = {}
                motor_config["motors"][name]["zero_pos"] = 0.0

            pos = round(float(pos), 6)
            print(
                f"{name:<{max_name_len}} : {motor_config['motors'][name]['zero_pos']:<6} -> {pos}"
            )
            motor_config["motors"][name]["zero_pos"] = pos

    with open(motor_config_path, "w") as f:
        yaml.dump(motor_config, f, indent=4, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the zero point calibration.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xc",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--parts",
        type=str,
        default="all",
        help="Specify parts to calibrate. Use 'all' or a subset of [left_arm, right_arm, left_gripper, right_gripper, hip, knee, left_ankle, right_ankle, neck], split by space.",
    )
    args = parser.parse_args()

    # Parse parts into a list
    parts = args.parts.split(" ") if args.parts != "all" else ["all"]

    robot = Robot(args.robot)

    main(robot, parts)
