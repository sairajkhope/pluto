"""Test motion reference systems in MuJoCo simulation.

This module tests various motion reference systems (walking, cartwheel, get_up)
with optional joystick control for interactive testing.
"""

import argparse
import time
from typing import List

import numpy as np
from tqdm import tqdm

from toddlerbot.locomotion.mjx_env import get_env_config
from toddlerbot.reference.cartwheel_ref import CartwheelReference
from toddlerbot.reference.get_up_prone_ref import GetUpProneReference
from toddlerbot.reference.get_up_roll_ref import GetUpRollReference
from toddlerbot.reference.motion_ref import MotionReference
from toddlerbot.reference.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


def test_ref_motion(
    sim: MuJoCoSim,
    motion_ref: MotionReference,
    command_range: List[List[float]],
    vis_type: str,
    loop: bool = True,
):
    """Test motion reference in simulation with optional joystick control."""
    joystick = None
    try:
        joystick = Joystick()
    except ValueError as e:
        print(f"Joystick initialization failed: {e}")

    state_ref = motion_ref.get_default_state()
    # pose_command = np.random.uniform(-1, 1, 5)

    step_idx = 0
    sample_interval = 2

    max_steps = motion_ref.n_frames if hasattr(motion_ref, "n_frames") else 3000
    p_bar = tqdm(total=max_steps, desc="Running the test")
    last_command = None

    contact_pairs = []
    try:
        while loop or step_idx < max_steps:
            if joystick is None:
                if step_idx % (sample_interval / sim.control_dt) == 0:
                    if np.random.rand() < 0.5:
                        control_inputs = {
                            "walk_x": np.random.uniform(-1, 1),
                            "walk_y": np.random.uniform(-1, 1),
                            "walk_turn": 0.0,
                        }
                    else:
                        control_inputs = {
                            "walk_x": 0.0,
                            "walk_y": 0.0,
                            "walk_turn": np.random.uniform(-1, 1),
                        }
                # control_inputs = {"walk_x": 0.0, "walk_y": 1.0, "walk_turn": 0.0}
            else:
                control_inputs = joystick.get_controller_input()

            command = np.zeros(len(command_range), dtype=np.float32)

            if "walk" in motion_ref.name:
                for task, input in control_inputs.items():
                    axis = None
                    if task == "walk_x":
                        axis = 5
                    elif task == "walk_y":
                        axis = 6
                    elif task == "walk_turn":
                        axis = 7

                    if axis is not None:
                        command[axis] = np.interp(
                            input,
                            [-1, 0, 1],
                            [command_range[axis][1], 0.0, command_range[axis][0]],
                        )
            elif "get_up" in motion_ref.name:
                if last_command is None:
                    command = np.random.randint(
                        0, len(motion_ref.get_up_mode_list), (1,)
                    )
                    last_command = command
                else:
                    command = last_command

            # print(f"Command: {command}")
            time_curr = step_idx * sim.control_dt
            state_ref = motion_ref.get_state_ref(time_curr, command, state_ref)

            # sim.set_motor_angles(state_ref["motor_pos"])
            # sim.set_joint_angles(state_ref["joint_pos"])
            sim.set_qpos(state_ref["qpos"])
            sim.forward()

            # sim.set_motor_target(state_ref["motor_pos"])
            # sim.step()

            if sim.data.ncon > 0:
                for i in range(sim.data.ncon):
                    contact = sim.data.contact[i]
                    geom1_id = contact.geom1
                    geom2_id = contact.geom2
                    geom1_name = sim.model.geom(geom1_id).name
                    geom2_name = sim.model.geom(geom2_id).name
                    if (geom1_name, geom2_name) not in contact_pairs and (
                        geom2_name,
                        geom1_name,
                    ) not in contact_pairs:
                        contact_pairs.append((geom1_name, geom2_name))

            step_idx += 1

            p_bar_steps = int(1 / sim.control_dt)
            if step_idx % p_bar_steps == 0:
                p_bar.update(p_bar_steps)

            if vis_type == "view":
                time.sleep(sim.control_dt)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping the simulation...")

    finally:
        print("\nContact pairs:")
        for pair in sorted(contact_pairs):
            print(f'    ["{pair[0]}", "{pair[1]}"],')

        if vis_type == "render" and hasattr(sim, "save_recording"):
            assert isinstance(sim, MuJoCoSim)
            sim.save_recording(".", sim.control_dt, 2, f"{motion_ref.name}.mp4")

        sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xc",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="mujoco",
        help="The simulator to use.",
    )
    parser.add_argument(
        "--vis",
        type=str,
        default="view",
        help="The visualization type.",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="walk_zmp",
        help="The name of the task.",
    )
    parser.add_argument(
        "--no-loop",
        dest="loop",
        action="store_false",
        help="Run for a fixed number of steps instead of looping indefinitely.",
    )
    parser.set_defaults(loop=True)

    args = parser.parse_args()

    robot = Robot(args.robot)
    if args.sim == "mujoco":
        sim = MuJoCoSim(robot, vis_type=args.vis, fixed_base="fixed" in args.robot)
    else:
        raise ValueError("Unknown simulator")

    motion_ref: MotionReference | None = None

    if "walk" in args.ref:
        walk_cfg, _ = get_env_config("walk")
        command_range = walk_cfg.commands.command_range

        motion_ref = WalkZMPReference(
            robot,
            walk_cfg.sim.timestep * walk_cfg.action.n_frames,
            walk_cfg.action.cycle_time,
        )
    elif "cartwheel" in args.ref:
        cartwheel_cfg, _ = get_env_config("cartwheel")
        command_range = cartwheel_cfg.commands.command_range

        motion_ref = CartwheelReference(
            robot, cartwheel_cfg.sim.timestep * cartwheel_cfg.action.n_frames
        )
    elif "get_up_prone" in args.ref:
        get_up_prone_cfg, _ = get_env_config("get_up_prone")
        command_range = get_up_prone_cfg.commands.command_range

        motion_ref = GetUpProneReference(
            robot, get_up_prone_cfg.sim.timestep * get_up_prone_cfg.action.n_frames
        )
    elif "get_up_roll" in args.ref:
        get_up_roll_cfg, _ = get_env_config("get_up_roll")
        command_range = get_up_roll_cfg.commands.command_range

        motion_ref = GetUpRollReference(
            robot, get_up_roll_cfg.sim.timestep * get_up_roll_cfg.action.n_frames
        )
    else:
        raise ValueError("Unknown ref motion")

    test_ref_motion(sim, motion_ref, command_range, args.vis, args.loop)
