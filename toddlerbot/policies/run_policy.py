"""Policy execution framework with visualization and logging capabilities.

This module provides the main execution framework for running policies on robots,
including dynamic policy loading, data logging, plotting, and experiment management.
"""

import argparse
import importlib
import os
import time
from itertools import product
from typing import Any, Dict, List, Type

import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

import wandb
from toddlerbot.policies import BasePolicy
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.comm_utils import ZMQNode, sync_time
from toddlerbot.utils.misc_utils import dump_profiling_data  # , profile
from toddlerbot.visualization.vis_plot import (
    plot_joint_tracking,
    plot_joint_tracking_frequency,
    plot_joint_tracking_single,
    plot_line_graph,
    plot_loop_time,
    plot_motor_vel_tor_mapping,
)


def get_policy_class(policy_name: str) -> Type[BasePolicy]:
    """Dynamically imports and returns the policy class for the given name."""
    module_name_list = [
        f"toddlerbot.policies.{policy_name}_policy",
        f"toddlerbot.policies.{policy_name}",
    ]

    def first_upper(s: str) -> str:
        return s[0].upper() + s[1:] if s else s

    class_name_list = []
    words = policy_name.split("_")
    for style_combo in product(["first_upper", "upper"], repeat=len(words)):
        styled_words = []
        for word, style in zip(words, style_combo):
            if style == "upper":
                styled_words.append(word.upper())
            elif style == "first_upper":
                styled_words.append(first_upper(word))
        class_name = "".join(styled_words) + "Policy"
        class_name_list.append(class_name)

    errors = []
    for module_name in module_name_list:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            errors.append(f"Module '{module_name}' not found")
            continue

        for class_name in class_name_list:
            try:
                cls = getattr(module, class_name)
                if not issubclass(cls, BasePolicy):
                    raise TypeError(f"{class_name} is not a subclass of BasePolicy")
                return cls
            except AttributeError:
                errors.append(
                    f"Class '{class_name}' not found in module '{module_name}'"
                )
            except TypeError as e:
                errors.append(str(e))

    raise ValueError(
        f"Policy '{policy_name}' could not be loaded. Tried combinations:\n"
        + "\n".join(errors)
    )


def plot_results(
    robot: Robot,
    loop_time_list: List[List[float]],
    obs_list: List[Obs],
    control_inputs_list: List[Dict[str, float]],
    action_list: List[npt.NDArray[np.float32]],
    exp_folder_path: str,
    blocking: bool,
):
    """Generates and saves various plots to visualize the performance and behavior of a robot during an experiment.

    Args:
        robot (Robot): The robot object containing information about the robot's configuration and state.
        loop_time_list (List[List[float]]): A list of lists containing timing information for each loop iteration.
        obs_list (List[Obs]): A list of observations recorded during the experiment.
        control_inputs_list (List[Dict[str, float]]): A list of dictionaries containing control inputs applied to the robot.
        motor_angles_list (List[Dict[str, float]]): A list of dictionaries containing motor angles recorded during the experiment.
        exp_folder_path (str): The path to the folder where the plots will be saved.
    """
    time_obs_list: List[float] = []
    ang_vel_obs_list: List[npt.NDArray[np.float32]] = []
    pos_obs_list: List[npt.NDArray[np.float32]] = []
    euler_obs_list: List[npt.NDArray[np.float32]] = []
    cur_total_list: List[float] = []
    time_seq_dict: Dict[str, List[float]] = {}
    time_seq_ref_dict: Dict[str, List[float]] = {}
    motor_pos_dict: Dict[str, List[float]] = {}
    motor_vel_dict: Dict[str, List[float]] = {}
    motor_acc_dict: Dict[str, List[float]] = {}
    motor_tor_dict: Dict[str, List[float]] = {}
    motor_cur_dict: Dict[str, List[float]] = {}
    for i, obs in enumerate(obs_list):
        time_obs_list.append(obs.time)
        # lin_vel_obs_list.append(obs.lin_vel)
        ang_vel_obs_list.append(obs.ang_vel if obs.ang_vel is not None else np.zeros(3))
        pos_obs_list.append(obs.pos if obs.pos is not None else np.zeros(3))
        euler_obs_list.append(obs.rot.as_euler("xyz", degrees=False) if obs.rot is not None else np.zeros(3))
        if obs.motor_cur is not None:
            cur_total_list.append(sum(abs(obs.motor_cur)))

        for j, motor_name in enumerate(robot.motor_ordering):
            if motor_name not in time_seq_dict:
                time_seq_ref_dict[motor_name] = []
                time_seq_dict[motor_name] = []
                motor_pos_dict[motor_name] = []
                motor_vel_dict[motor_name] = []
                motor_acc_dict[motor_name] = []
                motor_tor_dict[motor_name] = []
                if obs.motor_cur is not None:
                    motor_cur_dict[motor_name] = []

            # Assume the state fetching is instantaneous
            time_seq_dict[motor_name].append(float(obs.time))
            time_seq_ref_dict[motor_name].append(float(obs.time))
            # time_seq_ref_dict[motor_name].append(i * policy.control_dt)
            motor_pos_dict[motor_name].append(obs.motor_pos[j])
            motor_vel_dict[motor_name].append(obs.motor_vel[j])
            if obs.motor_acc is not None:
                motor_acc_dict[motor_name].append(obs.motor_acc[j])
            motor_tor_dict[motor_name].append(obs.motor_tor[j])
            if obs.motor_cur is not None:
                motor_cur_dict[motor_name].append(obs.motor_cur[j])

    action_dict: Dict[str, List[float]] = {}
    for motor_target in action_list:
        motor_angles = dict(zip(robot.motor_ordering, motor_target))
        for motor_name, motor_angle in motor_angles.items():
            if motor_name not in action_dict:
                action_dict[motor_name] = []
            action_dict[motor_name].append(motor_angle)

    # control_inputs_dict: Dict[str, List[float]] = {}
    # for control_inputs in control_inputs_list:
    #     for control_name, control_input in control_inputs.items():
    #         if control_name not in control_inputs_dict:
    #             control_inputs_dict[control_name] = []
    #         control_inputs_dict[control_name].append(control_input)

    plt.switch_backend("Agg")

    plot_loop_time(loop_time_list, exp_folder_path, blocking=True)

    if "sysID" in robot.name and "joint_0" in motor_vel_dict and len(motor_vel_dict["joint_0"]) > 0:
        plot_motor_vel_tor_mapping(
            motor_vel_dict["joint_0"],
            motor_tor_dict["joint_0"],
            save_path=exp_folder_path,
            blocking=blocking,
        )

    if len(cur_total_list) > 0:
        plot_line_graph(
            cur_total_list,
            time_obs_list,
            legend_labels=["Current (mA)"],
            title="Total Current  Over Time",
            x_label="Time (s)",
            y_label="Current (mA)",
            save_config=True,
            save_path=exp_folder_path,
            file_name="total_cur_tracking",
            blocking=True,
        )()

    plot_line_graph(
        np.array(ang_vel_obs_list).T,
        time_obs_list,
        legend_labels=["Roll (X)", "Pitch (Y)", "Yaw (Z)"],
        title="Angular Velocities Over Time",
        x_label="Time (s)",
        y_label="Angular Velocity (rad/s)",
        save_config=True,
        save_path=exp_folder_path,
        file_name="ang_vel_tracking",
        blocking=True,
    )()
    plot_line_graph(
        np.array(euler_obs_list).T,
        time_obs_list,
        legend_labels=["Roll (X)", "Pitch (Y)", "Yaw (Z)"],
        title="Euler Angles Over Time",
        x_label="Time (s)",
        y_label="Euler Angles (rad)",
        save_config=True,
        save_path=exp_folder_path,
        file_name="euler_tracking",
        blocking=True,
    )()
    plot_joint_tracking(
        time_seq_dict,
        time_seq_ref_dict,
        motor_pos_dict,
        action_dict,
        save_path=exp_folder_path,
        blocking=blocking,
    )

    if len(motor_acc_dict) > 0:
        plot_joint_tracking_single(
            time_seq_dict,
            motor_acc_dict,
            save_path=exp_folder_path,
            y_label="Acceleration (rad/s^2)",
            file_name="motor_acc_tracking",
            blocking=blocking,
        )

    plot_joint_tracking_single(
        time_seq_dict,
        motor_tor_dict,
        save_path=exp_folder_path,
        y_label="Torque (Nm)",
        file_name="motor_tor_tracking",
        blocking=blocking,
    )

    if len(motor_cur_dict) > 0:
        plot_joint_tracking_single(
            time_seq_dict,
            motor_cur_dict,
            save_path=exp_folder_path,
            y_label="Current (mA)",
            file_name="motor_cur_tracking",
            blocking=blocking,
        )

    plot_joint_tracking_single(
        time_seq_dict,
        motor_vel_dict,
        save_path=exp_folder_path,
        blocking=blocking,
    )
    plot_joint_tracking_frequency(
        time_seq_dict,
        time_seq_ref_dict,
        motor_pos_dict,
        action_dict,
        save_path=exp_folder_path,
        blocking=blocking,
    )


# @profile()
def run_policy(
    sim: BaseSim,
    robot: Robot,
    policy: BasePolicy,
    vis_type: str,
    plot: bool,
    record: bool,
    note: str,
    recorder_ip: str = "192.168.0.5",
):
    """Executes a control policy on a robot within a simulation environment, logging data and optionally visualizing results.

    Args:
        robot (Robot): The robot instance to control.
        sim (BaseSim): The simulation environment in which the robot operates.
        policy (BasePolicy): The control policy to execute.
        vis_type (str): The type of visualization to use ('view', 'render', etc.).
        plot (bool): Whether to plot the results after execution.
    """

    exp_name = f"{robot.name}_{policy.name}_{sim.name}"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{exp_name}_{time_str}"

    os.makedirs(exp_folder_path, exist_ok=True)

    if record:
        project = "toddlerbot"
        entity = "toddlerbot"
        run_name = f"{exp_name}_{time_str}"
        run = wandb.init(
            project=project, entity=entity, name=run_name, notes=note, job_type="eval"
        )

        if "real" in sim.name:
            sender = ZMQNode(ip=recorder_ip)
            time.sleep(1)  # Allow subscriber to connect
            data = {
                "signal": 1,
                "project": project,
                "entity": entity,
                "run_name": run_name,
                "run_id": run.id,
                "note": note,
            }
            sender.send_msg(data)

    loop_time_list: List[List[float]] = []
    obs_list: List[Obs] = []
    control_inputs_list: List[Dict[str, float]] = []
    action_list: List[npt.NDArray[np.float32]] = []

    n_steps_total = (
        float("inf")
        if "real" in sim.name and "fixed" not in policy.name
        else policy.n_steps_total
    )
    contact_pairs = set()
    p_bar = tqdm(total=n_steps_total, desc="Running the policy")
    start_time = time.monotonic()
    step_idx = 0
    time_until_next_step = 0.0
    try:
        while step_idx < n_steps_total:
            step_start = time.monotonic()

            # Get the latest state from the queue
            obs = sim.get_observation()
            obs.time -= start_time

            if "real" not in sim.name and vis_type != "view":
                obs.time += time_until_next_step

            obs_time = time.monotonic()
            control_inputs, motor_target = policy.step(obs, sim)

            inference_time = time.monotonic()
            sim.set_motor_target(motor_target)
            set_action_time = time.monotonic()

            sim.step()

            if "real" not in sim.name:
                contact_pairs.update(sim.check_self_collisions())

            sim_step_time = time.monotonic()

            obs_list.append(obs)
            control_inputs_list.append(control_inputs)
            action_list.append(motor_target)

            step_idx += 1

            p_bar_steps = int(1 / policy.control_dt)
            if step_idx % p_bar_steps == 0:
                p_bar.update(p_bar_steps)

            step_end = time.monotonic()

            time_until_next_step = max(
                start_time + policy.control_dt * step_idx - step_end, 0
            )
            loop_time_list.append(
                [
                    step_start,
                    obs_time,
                    inference_time,
                    set_action_time,
                    sim_step_time,
                    step_end,
                    time_until_next_step,
                ]
            )

            # print(f"time_until_next_step: {time_until_next_step * 1000:.2f} ms")
            if ("real" in sim.name or vis_type == "view") and time_until_next_step > 0:
                time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("KeyboardInterrupt recieved. Closing...")

    finally:
        if record and "real" in sim.name:
            sender.send_msg({"signal": 0})

        p_bar.close()

        if vis_type == "render" and hasattr(sim, "save_recording"):
            assert isinstance(sim, MuJoCoSim)
            sim.save_recording(exp_folder_path, policy.control_dt, 2)

        try:
            sim.close()
        except Exception as e:
            print(f"Error closing simulation: {e}")

        if len(contact_pairs) > 0:
            print("\nContact pairs detected during the simulation:")
            for pair in sorted(contact_pairs):
                print(f'    ["{pair[0]}", "{pair[1]}"],')

        log_data_dict: Dict[str, Any] = {
            "obs_list": obs_list,
            "control_inputs_list": control_inputs_list,
            "action_list": action_list,
        }
        log_data_path = os.path.join(exp_folder_path, "log_data.lz4")
        joblib.dump(log_data_dict, log_data_path, compress="lz4")

        prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
        dump_profiling_data(prof_path)

        policy.close(exp_folder_path)

        plot_results(
            robot,
            loop_time_list,
            obs_list,
            control_inputs_list,
            action_list,
            exp_folder_path,
            blocking=plot,
        )

        if record:
            artifact = wandb.Artifact(name="rollout", type="rollout", metadata={})
            artifact.add_dir(exp_folder_path)
            wandb.log_artifact(
                artifact, aliases=["latest", os.path.basename(exp_folder_path)]
            )


def main(args=None):
    """Executes a policy for a specified robot and simulator configuration.

    This function parses command-line arguments to configure and run a policy for a robot. It supports different robots, simulators, visualization types, and tasks. The function initializes the appropriate simulation environment and policy based on the provided arguments and executes the policy.

    Args:
        args (list, optional): List of command-line arguments. If None, defaults to sys.argv.

    Raises:
        ValueError: If an unknown simulator is specified.
        AssertionError: If the teleop leader policy is used with an unsupported robot or simulator.
    """
    parser = argparse.ArgumentParser(description="Run a policy.")
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
        help="The name of the simulator to use.",
        choices=["mujoco", "real"],
    )
    parser.add_argument(
        "--vis",
        type=str,
        default="none",
        help="The visualization type.",
        choices=["render", "view", "none"],
    )
    parser.add_argument(
        "--policy", type=str, default="stand", help="The name of the task."
    )
    parser.add_argument(
        "--command",
        type=str,
        default="",
        help="The policy checkpoint to load for RL policies.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Anything that needs to be loaded from this path.",
    )
    parser.add_argument(
        "--ip", type=str, default="", help="The ip address of the follower."
    )
    parser.add_argument("--task", type=str, default="", help="The name of the task.")
    parser.add_argument(
        "--plot", action="store_true", default=False, help="Skip the plot functions."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Record the experiment with wandb.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="A note to add to the wandb run.",
    )
    args = parser.parse_args(args)

    if "teleop_leader" in args.policy:
        assert "fixed" in args.policy, (
            "The teleop leader policy only supports fixed base. "
            "Please use the teleop_leader_fixed."
        )

    robot = Robot(args.robot)

    sim: BaseSim | None = None
    if args.sim == "mujoco":
        sim = MuJoCoSim(robot, vis_type=args.vis, fixed_base="fixed" in args.policy)
        init_motor_pos = sim.get_observation().motor_pos

    elif args.sim == "real":
        from toddlerbot.sim.real_world import RealWorld

        sim = RealWorld(robot)
        init_motor_pos = sim.get_observation(retries=-1).motor_pos

    else:
        raise ValueError("Unknown simulator")

    PolicyClass = get_policy_class(args.policy.replace("_fixed", ""))

    kwargs = {
        "name": args.policy,
        "robot": robot,
        "init_motor_pos": init_motor_pos,
    }
    # Common extras
    if hasattr(args, "path") and args.path:
        kwargs["path"] = args.path
    if hasattr(args, "ip") and args.ip:
        kwargs["ip"] = args.ip
        sync_time(args.ip)
    if hasattr(args, "task") and args.task:
        kwargs["task"] = args.task
    if hasattr(args, "command") and args.command:
        kwargs["fixed_command"] = np.array(args.command.split(" "), dtype=np.float32)

    # Create policy
    policy = PolicyClass(**kwargs)

    run_policy(sim, robot, policy, args.vis, args.plot, args.record, args.note)


if __name__ == "__main__":
    main()
