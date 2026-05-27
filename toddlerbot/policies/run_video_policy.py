import argparse
import sys
import time
from typing import Dict

import numpy as np
import numpy.typing as npt

from toddlerbot.actuation import dynamixel_cpp
from toddlerbot.policies import BasePolicy
from toddlerbot.policies.run_policy import get_policy_class
from toddlerbot.sim import BaseSim
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode

PREP_DURATION = 1.0
MIN_STANDBY_DURATION = 0.0
INITIAL_STAND_PREP_DURATION = 7.0
INITIAL_STAND_MIN_STANDBY_DURATION = 5.0


POLICY_CONFIGS = {
    "get_off_cart": {
        "policy": "get_off_cart",
        "skill": "get_off_cart",
        "type": "transition",
    },
    "get_up": {
        "policy": "get_up_open",
        "skill": "get_up_open",
        "type": "transition",
    },
    "stand": {
        "policy": "stand",
        "skill": "stand",
        "type": "transition",
    },
    "recovery": {
        "policy": "get_up_ground",
        "skill": "get_up_ground",
        "type": "transition",
    },
    "get_down": {
        "policy": "get_down_open",
        "skill": "get_down_open",
        "type": "transition",
    },
}

POLICY_ORDER = list(POLICY_CONFIGS.keys())

SKILL_TO_KEY = {config["skill"]: key for key, config in POLICY_CONFIGS.items()}


class PolicySwitchingState:
    """Thread-safe state for policy switching"""

    def __init__(self):
        self.current_policy = None
        self.switch_requested = False
        self.new_policy = None
        self.prep_mode = True
        self.pending_target_policy = None
        self.in_safe_transition_from_walk = False


def load_policies(
    robot: Robot, init_motor_pos: npt.NDArray[np.float32]
) -> Dict[str, BasePolicy]:
    """Load all configured policies"""
    loaded_policies: Dict[str, BasePolicy] = {}
    policy_instances: Dict[tuple, BasePolicy] = {}

    print("Loading all policies...")

    for key, policy_cfg in POLICY_CONFIGS.items():
        try:
            if "checkpoint_path" in policy_cfg and policy_cfg["checkpoint_path"]:
                instance_key = (policy_cfg["policy"], policy_cfg["checkpoint_path"])
                if instance_key in policy_instances:
                    loaded_policies[key] = policy_instances[instance_key]
                    mode_str = (
                        f" (mode: {policy_cfg['mode']})" if "mode" in policy_cfg else ""
                    )
                    cmd_str = (
                        f" (command: {policy_cfg.get('command', 'default')})"
                        if "command" in policy_cfg
                        else ""
                    )
                    print(
                        f"  ✓ Reusing {policy_cfg['policy']} instance{mode_str}{cmd_str}"
                    )
                    continue

            # Handle different policy loading strategies
            if policy_cfg["policy"] == "stand":
                kwargs = {
                    "name": policy_cfg["policy"],
                    "robot": robot,
                    "init_motor_pos": init_motor_pos.copy(),
                }
                policy_class = get_policy_class(policy_cfg["policy"])
                policy = policy_class(**kwargs)
                print(f"  ✓ Loaded {policy_cfg['policy']} policy")

            elif policy_cfg["policy"] in [
                "get_off_cart",
                "crawl_only_demo",
                "push_up_crawl",
                "pull_up",
                "get_up_ground",
                "get_down_open",
                "get_up_open",
            ]:
                kwargs = {
                    "name": policy_cfg["policy"],
                    "robot": robot,
                    "init_motor_pos": init_motor_pos.copy(),
                }
                policy_class = get_policy_class(policy_cfg["policy"])
                policy = policy_class(**kwargs)
                print(
                    f"  ✓ Loaded {policy_cfg['policy']} backup policy (embedded motion)"
                )

            elif "checkpoint_path" in policy_cfg and policy_cfg["checkpoint_path"]:
                kwargs = {
                    "name": policy_cfg["policy"],
                    "robot": robot,
                    "init_motor_pos": init_motor_pos.copy(),
                    "path": policy_cfg["checkpoint_path"],
                }

                if policy_cfg.get("command", None) is not None:
                    command = np.array(
                        policy_cfg["command"].split(" "), dtype=np.float32
                    )
                    kwargs["fixed_command"] = command

                policy_class = get_policy_class(policy_cfg["policy"])
                policy = policy_class(**kwargs)

                instance_key = (policy_cfg["policy"], policy_cfg["checkpoint_path"])
                policy_instances[instance_key] = policy

                print(f"  ✓ Loaded {policy_cfg['policy']} policy (checkpoint)")

            else:
                print(f"  ! No valid path found for {policy_cfg['policy']}")
                continue

            if hasattr(policy, "prep_duration"):
                policy.prep_duration = PREP_DURATION
                policy.min_standby_duration = MIN_STANDBY_DURATION

            loaded_policies[key] = policy

        except Exception as e:
            print(f"  ✗ Failed to load {policy_cfg['policy']} policy: {e}")
            continue

    print(f"Successfully loaded {len(loaded_policies)} policies.")
    return loaded_policies


def create_simulation(args: argparse.Namespace, robot: Robot):
    """Create simulation environment"""
    if args.sim == "mujoco":
        sim = MuJoCoSim(
            robot,
            vis_type=args.vis,
            fixed_base=False,
            xml_path=f"toddlerbot/descriptions/{robot.name}/scene_cart.xml",
        )

        return sim

    elif args.sim == "real":
        from toddlerbot.sim.real_world import RealWorld

        sim = RealWorld(robot)
        return sim
    else:
        raise ValueError(f"Unknown simulator: {args.sim}")


def run_multiple_policy(
    robot: Robot,
    sim: BaseSim,
    loaded_policies: Dict[str, BasePolicy],
    args: argparse.Namespace,
):
    """Run multiple policies with switching logic"""
    switch_state = PolicySwitchingState()
    switch_state.prep_mode = not args.no_prep
    switch_state.new_policy = args.init

    obs_list = []
    control_inputs_list = []
    action_list = []
    loop_time_list = []

    initial_policy = loaded_policies.get(args.init)
    recovery_policy = loaded_policies.get("recovery")

    if not initial_policy:
        print("\n✗ ERROR: No policy found at key '0' in POLICY_CONFIGS.")
        print("Key '0' is required as the initial policy to start with.")
        print("Please ensure POLICY_CONFIGS has a valid policy at key '0'.")
        sys.exit(1)

    if hasattr(initial_policy, "prep_duration"):
        initial_policy.prep_duration = INITIAL_STAND_PREP_DURATION
        initial_policy.min_standby_duration = INITIAL_STAND_MIN_STANDBY_DURATION
        initial_policy.is_prepared = False

    current_policy = initial_policy
    current_policy_key = args.init
    previous_policy_key = args.init

    zmq_sender = ZMQNode(type="sender")
    zmq_receiver = ZMQNode(type="receiver", port=5556)
    has_pulled_up = False

    def speak(words: str):
        zmq_sender.send_msg(
            ZMQMessage(time=time.monotonic(), text=f"[REPEAT AFTER ME] {words}")
        )

    print("Multiple Policy Runner initialized.")
    print(f"\nRunning {POLICY_CONFIGS['get_off_cart']['policy']} policy.")
    print("Press any key to switch policies or 'esc' to quit.")
    print("Available policies:")
    for key, config in POLICY_CONFIGS.items():
        print(f"  {key}: {config['policy']} (skill: {config['skill']})")
    print("  esc: quit")
    print("-" * 50)

    try:
        step_count = 0
        policy_start_time = time.monotonic()
        loop_start_time = time.monotonic()
        global_heading_ref = None

        if "get_off_cart" in args.init:
            speak("Oops, let me get off this cart first.")

        while True:
            step_start_time = time.monotonic()
            obs = sim.get_observation()

            body_forward_vec = obs.rot.apply(np.array([1, 0, 0], dtype=np.float32))
            body_left_vec = obs.rot.apply(np.array([0, 1, 0], dtype=np.float32))
            if "recovery" not in current_policy_key:
                if abs(body_forward_vec[2]) > 0.5:
                    if (
                        "crawl" not in current_policy_key
                        and "get_off_cart" not in current_policy_key
                        and "push_up" not in current_policy_key
                        and "get_down" not in current_policy_key
                        and "get_up" not in current_policy_key
                    ) or abs(body_left_vec[2]) > 0.6:
                        recovery_policy.fall_down_counter += 1

                if recovery_policy.fall_down_counter > 20:
                    if body_forward_vec[2] < -0.5:
                        recovery_policy.motion_ref = recovery_policy.get_up_prone_ref
                        left_shoulder_pitch_pos = obs.motor_pos[
                            recovery_policy.left_shoulder_pitch_idx
                        ]
                        right_shoulder_pitch_pos = obs.motor_pos[
                            recovery_policy.right_shoulder_pitch_idx
                        ]
                        is_left_arm_down = (
                            left_shoulder_pitch_pos > -np.pi / 2
                            and left_shoulder_pitch_pos < np.pi / 2
                        )
                        is_right_arm_down = (
                            right_shoulder_pitch_pos > -np.pi / 2
                            and right_shoulder_pitch_pos < np.pi / 2
                        )
                        if is_left_arm_down and is_right_arm_down:
                            recovery_policy.get_up_mode = 0
                        elif not is_left_arm_down and not is_right_arm_down:
                            recovery_policy.get_up_mode = 1
                        elif is_left_arm_down and not is_right_arm_down:
                            recovery_policy.get_up_mode = 2
                        else:
                            recovery_policy.get_up_mode = 3

                    elif body_forward_vec[2] > 0.5:
                        recovery_policy.motion_ref = recovery_policy.get_up_roll_ref
                        if abs(body_left_vec[2]) < 0.1736:
                            recovery_policy.get_up_mode = 0
                        elif body_left_vec[2] >= 0.1736:
                            recovery_policy.get_up_mode = 1
                        else:
                            recovery_policy.get_up_mode = 2

                    recovery_policy.fall_down_counter = 0
                    recovery_policy.is_prepared = False
                    recovery_policy.is_done = False

                    current_policy = recovery_policy
                    previous_policy_key = current_policy_key
                    current_policy_key = "recovery"

                    speak("Ouch! My feet slipped! Don't worry, I'll get back up.")

                    if "real" in sim.name:
                        dynamixel_cpp.enable_motors(sim.controllers)

                    print("\nSwitching to recovery policy...")

            obs.time = obs.time - policy_start_time

            if (
                switch_state.in_safe_transition_from_walk
                and switch_state.pending_target_policy
            ):
                if (
                    current_policy_key == "walk_in_place"
                    and hasattr(current_policy, "is_standing")
                    and current_policy.is_standing
                ):
                    switch_state.new_policy = switch_state.pending_target_policy
                    switch_state.switch_requested = True
                    switch_state.pending_target_policy = None
                    switch_state.in_safe_transition_from_walk = False

                elif current_policy_key == "walk_in_place":
                    switch_state.switch_requested = False

            if switch_state.switch_requested:
                new_policy_key = switch_state.new_policy

                if (
                    current_policy_key == "walk"
                    and new_policy_key != "walk"
                    and not switch_state.in_safe_transition_from_walk
                ):
                    if (
                        "walk_in_place" in loaded_policies
                        and current_policy_key != "walk_in_place"
                    ):
                        switch_state.pending_target_policy = new_policy_key
                        switch_state.new_policy = "walk_in_place"
                        new_policy_key = switch_state.new_policy
                        switch_state.in_safe_transition_from_walk = True

                current_policy = loaded_policies[new_policy_key]
                if (
                    "recovery" not in current_policy_key
                    and "get_down" not in current_policy_key
                ):
                    previous_policy_key = current_policy_key

                current_policy_key = new_policy_key

                if "real" in sim.name:
                    dynamixel_cpp.enable_motors(sim.controllers)

                print(f"\nSwitching to {new_policy_key} policy.")

                if "command" in POLICY_CONFIGS[new_policy_key] and hasattr(
                    current_policy, "fixed_command"
                ):
                    command = np.array(
                        POLICY_CONFIGS[new_policy_key]["command"].split(" "),
                        dtype=np.float32,
                    )
                    current_policy.fixed_command = command
                    if command[-1] != 0.0:
                        current_policy.target_torso_yaw = obs.rot.as_euler("xyz")[2]
                    if command[-1] == 0.0:
                        if command[-3] >= 0.0:
                            current_policy.target_torso_yaw = 0.0
                        else:
                            current_policy.target_torso_yaw = np.pi

                if hasattr(current_policy, "base_torso_rot_inv"):
                    if (
                        (
                            POLICY_CONFIGS[new_policy_key]["policy"] == "walk"
                            or POLICY_CONFIGS[new_policy_key]["type"] == "locomotion"
                            or POLICY_CONFIGS[new_policy_key]["policy"]
                            == "climb_up_box_crawl"
                        )
                        and global_heading_ref is not None
                        and current_policy.base_torso_rot_inv is None
                    ):
                        current_policy.base_torso_rot_inv = global_heading_ref
                    elif (
                        POLICY_CONFIGS[new_policy_key]["policy"] != "walk"
                        and POLICY_CONFIGS[new_policy_key]["type"] != "locomotion"
                    ):
                        current_policy.base_torso_rot_inv = None

                if hasattr(current_policy, "reset"):
                    current_policy.reset()

                if hasattr(current_policy, "is_done"):
                    current_policy.is_done = False

                if new_policy_key != args.init and hasattr(current_policy, "set_qpos"):
                    print(
                        f"Resetting qpos for policy: {POLICY_CONFIGS[new_policy_key]['policy']}."
                    )
                    current_policy.set_qpos = True

                policy_start_time = time.monotonic()

                obs.time = 0.0

                if switch_state.prep_mode:
                    current_policy.init_motor_pos = obs.motor_pos.copy()
                    current_policy.is_prepared = False
                else:
                    current_policy.is_prepared = True
                    current_policy.prep_duration = 0.0
                    if hasattr(current_policy, "time_start"):
                        current_policy.time_start = 0.0

                switch_state.switch_requested = False

            obs_time = time.monotonic()

            control_inputs, action = current_policy.step(obs, sim)
            if "stand" in current_policy_key:
                action[1] = 0.0

            inference_time = time.monotonic()

            sim.set_motor_target(action)

            set_action_time = time.monotonic()

            sim.step()

            sim_step_time = time.monotonic()

            if "stand" in current_policy_key and not has_pulled_up:
                msg = zmq_receiver.get_msg()  # Clear any existing messages
                if msg and msg.text and "pull_up" in msg.text.lower():
                    speak(
                        "That's my favorite! Haochen, give me your hands. Now watch me pull up!"
                    )
                    switch_state.switch_requested = True
                    switch_state.new_policy = "pull_up"
                    has_pulled_up = True

            if hasattr(current_policy, "is_done") and current_policy.is_done:
                if "get_off_cart" in current_policy_key:
                    speak(
                        "Alright, I'm off the cart now. Watch me crawl a few steps foreward!"
                    )
                    switch_state.switch_requested = True
                    switch_state.new_policy = "get_up"
                elif "get_up" in current_policy_key:
                    speak("Phew! That was a nice stretch—now let's stand up tall!")
                    switch_state.switch_requested = True
                    switch_state.new_policy = "stand"
                elif "recovery" in current_policy_key:
                    if abs(body_forward_vec[2]) < 0.5:
                        speak("That hurt! But I'm okay now, let's keep going.")
                        switch_state.switch_requested = True
                        next_policy_key = POLICY_ORDER[
                            POLICY_ORDER.index(previous_policy_key) + 1
                        ]
                        if "crawl" in next_policy_key or "push_up" in next_policy_key:
                            switch_state.new_policy = "get_down"
                        else:
                            switch_state.new_policy = "stand"
                elif "get_down" in current_policy_key:
                    switch_state.switch_requested = True
                    switch_state.new_policy = previous_policy_key

            step_count += 1
            step_end_time = time.monotonic()
            target_time = loop_start_time + step_count * current_policy.control_dt
            time_until_next_step = target_time - time.monotonic()

            obs_for_logging = obs.__class__(
                time=obs_time - loop_start_time,
                motor_pos=obs.motor_pos,
                motor_vel=obs.motor_vel,
                motor_acc=obs.motor_acc,
                motor_tor=obs.motor_tor,
                motor_cur=obs.motor_cur,
                lin_vel=obs.lin_vel,
                ang_vel=obs.ang_vel,
                pos=obs.pos,
                rot=obs.rot,
            )
            obs_list.append(obs_for_logging)
            control_inputs_list.append(control_inputs)
            action_list.append(action)

            loop_time_list.append(
                [
                    step_start_time,
                    obs_time,
                    inference_time,
                    set_action_time,
                    sim_step_time,
                    step_end_time,
                    time_until_next_step,
                ]
            )

            if ("real" in sim.name or args.vis == "view") and time_until_next_step > 0:
                time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping...")
    finally:
        pass


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Multiple Policy Runner.")
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
        default="view",
        help="The visualization type.",
        choices=["render", "view", "none"],
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Enable blocking plot display (requires --log).",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="A note to add to the wandb run.",
    )
    parser.add_argument(
        "--no-prep",
        action="store_true",
        default=False,
        help="Disable preparation mode between policy switches (prep mode enabled by default).",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="get_off_cart",
        help="The initial policy to start with.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MULTIPLE POLICY RUNNER")
    print("=" * 60)
    print(f"Robot: {args.robot}")
    print(f"Simulator: {args.sim}")
    print("=" * 60)

    robot = Robot(args.robot)

    sim = create_simulation(args, robot)

    if "real" in sim.name:
        init_motor_pos = sim.get_observation(retries=-1).motor_pos
    else:
        init_motor_pos = sim.get_observation().motor_pos

    loaded_policies = load_policies(robot, init_motor_pos)

    try:
        run_multiple_policy(
            robot=robot, sim=sim, loaded_policies=loaded_policies, args=args
        )
    finally:
        sim.close()


if __name__ == "__main__":
    main()
