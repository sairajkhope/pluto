"""MJX-based neural policy for reinforcement learning control.

This module implements a policy that uses trained neural networks via ONNX Runtime
to control the robot using observations and commands with frame stacking and action buffering.
"""

import json
import os
from typing import Any

import jax
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

import wandb
from toddlerbot.policies import BasePolicy
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.math_utils import get_action_traj, interpolate_action

device = "cpu"


def load_wandb_policy(
    name: str, project="ToddlerBot", entity="toddlerbot"
) -> dict[str, Any]:
    """Load a policy from WandB artifacts."""
    root = os.path.join("ckpts", name)
    ckpt_path = os.path.join(root, "model_best.onnx")
    if not os.path.exists(ckpt_path):
        if wandb.run:
            run = wandb.run
        else:
            run = wandb.init(project=project, entity=entity, name=name, job_type="eval")

        artifact = run.use_artifact(f"{entity}/{project}/policy:{name}", type="model")
        os.makedirs(root, exist_ok=True)
        artifact.download(root=root)

    return root


class MJXPolicy(BasePolicy):
    """Policy for controlling the robot using the MJX model."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        path: str,
        joystick: Joystick | None = None,
        fixed_command: npt.NDArray[np.float32] | None = None,
    ):
        """Initializes the class with configuration and state parameters for controlling a robot.

        Args:
            name (str): The name of the robot controller.
            robot (Robot): The robot instance to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions.
            ckpt (str): Path to the checkpoint file for loading model parameters.
            joystick (Optional[Joystick]): Joystick instance for manual control, if available.
            fixed_command (Optional[npt.NDArray[np.float32]]): Fixed command array, if any.
            cfg (Optional[MJXConfig]): Configuration object containing control parameters.
            motion_ref (Optional[MotionReference]): Reference for motion planning.

        Raises:
            AssertionError: If `cfg` is not provided.
        """
        super().__init__(name, robot, init_motor_pos)

        print(f"Loading policy from {path}...")

        exp_folder_path = load_wandb_policy(path)
        ckpt_path = os.path.join(exp_folder_path, "model_best.onnx")
        sess_options = ort.SessionOptions()
        # reduces thread jitter
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # avoid thread contention or spikes caused by thread pool delays
        sess_options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            ckpt_path, sess_options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.env_cfg = json.load(open(os.path.join(exp_folder_path, "env_config.json")))

        self.command_obs_indices = self.env_cfg["commands"]["command_obs_indices"]
        self.command_range = np.array(
            self.env_cfg["commands"]["command_range"], dtype=np.float32
        )
        self.num_commands = len(self.command_range)

        if fixed_command is None:
            self.fixed_command = np.zeros(self.num_commands, dtype=np.float32)
        else:
            self.fixed_command = fixed_command

        self.obs_size = self.env_cfg["obs"]["num_single_obs"]
        self.privileged_obs_size = self.env_cfg["obs"]["num_single_privileged_obs"]
        self.frame_stack = self.env_cfg["obs"]["frame_stack"]
        self.num_obs_history = self.frame_stack * self.obs_size
        self.num_privileged_obs_history = self.frame_stack * self.privileged_obs_size
        self.obs_history = np.zeros(self.num_obs_history, dtype=np.float32)
        self.obs_scales = self.env_cfg["obs_scales"]

        self.action_scale = self.env_cfg["action"]["action_scale"]
        self.n_steps_delay = self.env_cfg["action"]["n_steps_delay"]
        self.action_parts = self.env_cfg["action"]["action_parts"]
        self.motor_limits = np.array(list(self.robot.motor_limits.values()))

        action_mask: list[jax.Array] = []
        for part_name in self.action_parts:
            if part_name == "neck":
                action_mask.append(self.neck_motor_indices)
            elif part_name == "waist":
                action_mask.append(self.waist_motor_indices)
            elif part_name == "leg":
                action_mask.append(self.leg_motor_indices)
            elif part_name == "arm":
                action_mask.append(self.arm_motor_indices)

        self.action_mask = np.sort(np.concatenate(action_mask))
        self.num_action = self.action_mask.shape[0]
        self.default_action = self.default_motor_pos[self.action_mask]
        self.ref_motor_pos = self.default_motor_pos.copy()

        # # Set neck position in ref_motor_pos based on fixed_command
        # if hasattr(self.robot, "neck_joint_limits") and len(self.fixed_command) > 1:
        #     # Interpolate neck positions from command (same logic as walk_zmp_ref.py)
        #     neck_yaw_pos = np.interp(
        #         self.fixed_command[0],
        #         np.array([-1, 0, 1]),
        #         np.array(
        #             [
        #                 self.robot.neck_joint_limits[0, 0],
        #                 0.0,
        #                 self.robot.neck_joint_limits[1, 0],
        #             ]
        #         ),
        #     )
        #     neck_pitch_pos = np.interp(
        #         self.fixed_command[1],
        #         np.array([-1, 0, 1]),
        #         np.array(
        #             [
        #                 self.robot.neck_joint_limits[0, 1],
        #                 0.0,
        #                 self.robot.neck_joint_limits[1, 1],
        #             ]
        #         ),
        #     )

        #     # Convert joint positions to motor positions using robot's IK
        #     neck_joint_pos = np.array([neck_yaw_pos, neck_pitch_pos])
        #     neck_motor_pos = self.robot.neck_ik(neck_joint_pos)

        #     # Update ref_motor_pos with neck positions
        #     self.ref_motor_pos[self.neck_motor_indices] = neck_motor_pos

        self.base_torso_rot_inv = None
        self.target_torso_yaw = 0.0
        self.yaw_corr_gain = 0.5

        self.joystick = joystick
        if joystick is None:
            try:
                self.joystick = Joystick()
            except Exception:
                pass

        self.control_inputs: dict[str, float] = {}
        self.is_prepared = False
        self.prep_duration = 7.0
        self.min_standby_duration = (
            5.0  # Minimum time to hold final pose before starting
        )
        self.is_done = False

        self.reset()

    def reset(self):
        """Resets the internal state of the policy to its initial configuration.

        This method clears the observation history, phase signal, command list, and action buffer. It also sets the standing state to True and initializes the last action and current step counter to zero.
        """
        print(f"Resetting the {self.name} policy...")
        self.obs_history = np.zeros(self.num_obs_history, dtype=np.float32)
        self.phase_signal = np.zeros(2, dtype=np.float32)
        self.is_standing = True
        self.command_list = []
        self.last_action = np.zeros(self.num_action, dtype=np.float32)
        self.action_buffer = np.zeros(
            ((self.n_steps_delay + 1) * self.num_action), dtype=np.float32
        )
        self.step_curr = 0

    def get_phase_signal(self, time_curr: float) -> npt.NDArray[np.float32]:
        """Get the phase signal at the current time.

        Args:
            time_curr (float): The current time for which the phase signal is requested.

        Returns:
            npt.NDArray[np.float32]: An array containing the phase signal as a float32 value.
        """
        return np.zeros(1, dtype=np.float32)

    def get_command(
        self, obs: Obs, control_inputs: dict[str, float]
    ) -> npt.NDArray[np.float32]:
        """Returns a fixed command as a NumPy array.

        Args:
            control_inputs (Dict[str, float]): A dictionary of control inputs, where keys are input names and values are their respective float values.

        Returns:
            npt.NDArray[np.float32]: A fixed command represented as a NumPy array of float32 values.
        """
        return self.fixed_command

    # @profile()
    def step(
        self, obs: Obs, sim: BaseSim
    ) -> tuple[dict[str, float], npt.NDArray[np.float32]]:
        """Processes a single step in the control loop, updating the system's state and generating motor target positions.

        Args:
            obs (Obs): The current observation containing motor positions, velocities, and other sensor data.
            is_real (bool, optional): Indicates if the system is operating in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing the control inputs and the target motor positions.
        """
        is_real = "real" in sim.name
        if not self.is_prepared:
            self.is_prepared = True
            self.time_start = self.prep_duration
            self.prep_time, self.prep_action = get_action_traj(
                0.0,
                self.init_motor_pos,
                self.ref_motor_pos,
                self.prep_duration,
                self.control_dt,
                end_time=self.min_standby_duration,
            )

        if obs.time < self.time_start:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action
        elif self.base_torso_rot_inv is None:
            self.base_torso_rot_inv = R.from_euler(
                "z", obs.rot.as_euler("xyz")[2]
            ).inv()

        time_curr = self.step_curr * self.control_dt

        control_inputs: dict[str, float] = {}
        if len(self.control_inputs) > 0:
            control_inputs = self.control_inputs
        elif self.joystick is not None:
            control_inputs = self.joystick.get_controller_input()

        self.phase_signal = self.get_phase_signal(time_curr)
        motor_pos_delta = obs.motor_pos - self.default_motor_pos
        motor_vel = obs.motor_vel

        if self.robot.has_gripper:
            motor_pos_delta = motor_pos_delta[:-2]
            motor_vel = motor_vel[:-2]

        obs.rot = self.base_torso_rot_inv * obs.rot

        # if "walk" in self.name:
        #     if self.fixed_command[-3] < 0:
        #         x_axis = obs.rot.as_matrix()[:, 0]
        #         yaw = np.pi
        #         yaw_rot = R.from_rotvec(-yaw * np.array([0, 0, 1]))
        #         obs.rot = yaw_rot * obs.rot

        # NOTE: Ignore arm and neck motor positions for walking
        if "walk" in self.name:
            motor_pos_delta[23:30] = 0.0  # Sets right arm to default positions
            motor_pos_delta[0:2] = 0.0  # neck
            motor_pos_delta[16:23] = 0.0  # left arm

        obs_quat = obs.rot.as_quat(scalar_first=True)
        if obs_quat[0] < 0:
            obs_quat = -obs_quat

        # This needs to go after obs.rot is changed
        command = self.get_command(obs, control_inputs)

        obs_arr = np.concatenate(
            [
                self.phase_signal,
                command[self.command_obs_indices],
                motor_pos_delta * self.obs_scales["dof_pos"],
                motor_vel * self.obs_scales["dof_vel"],
                self.last_action,
                # obs.lin_vel * self.obs_scales.lin_vel,
                obs.ang_vel * self.obs_scales["ang_vel"],
                obs_quat * self.obs_scales["quat"],  # quaternion in wxyz format
            ]
        )

        self.obs_history[obs_arr.size :] = self.obs_history[: -obs_arr.size]
        self.obs_history[: obs_arr.size] = obs_arr

        action = self.session.run(
            [self.output_name], {self.input_name: self.obs_history[None]}
        )[0].squeeze()
        # action = np.zeros(self.num_action, dtype=np.float32)
        if is_real:
            delayed_action = action
        else:
            self.action_buffer = np.roll(self.action_buffer, action.size)
            self.action_buffer[: action.size] = action
            delayed_action = self.action_buffer[-self.num_action :]

        action_target = self.default_action + self.action_scale * delayed_action

        motor_target = self.default_motor_pos.copy()
        motor_target[self.action_mask] = action_target
        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        self.command_list.append(command)
        self.last_action = delayed_action
        self.step_curr += 1

        return control_inputs, motor_target
