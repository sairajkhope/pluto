"""Pull-up exercise policy using vision-guided grasping.

This module implements a pull-up policy that uses computer vision to detect
targets and perform coordinated arm movements for pull-up exercises.
"""

import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import get_action_traj, interpolate_action


class PullUpPolicy(BasePolicy):
    """Policy for pulling up the robot."""

    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        """Initializes the object with specified parameters and sets up camera and motion data.

        Args:
            name (str): The name of the object.
            robot (Robot): The robot instance associated with this object.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions.

        Raises:
            ValueError: If required motion data files are not found.
        """
        super().__init__(name, robot, init_motor_pos)

        # assert robot.has_gripper, "PullUpPolicy requires a robot with a gripper."

        self.left_eye = None
        self.right_eye = None
        try:
            self.left_eye = Camera("left")
            self.right_eye = Camera("right")
        except Exception:
            pass

        self.root_to_left_sho_pitch = np.array(
            [-0.0035, 0.07, 0.1042], dtype=np.float32
        )
        self.root_to_right_sho_pitch = np.array(
            [-0.0035, -0.07, 0.1042], dtype=np.float32
        )
        self.elbow_roll_to_sho_pitch = 0.0876
        self.wrist_pitch_to_elbow_roll = 0.0806
        self.ee_center_to_wrist_pitch = 0.045

        self.root_to_left_eye_t = np.array([0.032, 0.017, 0.19], dtype=np.float32)
        self.root_to_neck_t = np.array([0.016, 0.0, 0.1419], dtype=np.float32)
        self.waist_motor_indices = np.array(
            [
                robot.motor_ordering.index("waist_act_1"),
                robot.motor_ordering.index("waist_act_2"),
            ]
        )
        self.neck_pitch_idx = robot.motor_ordering.index("neck_pitch_act")
        self.neck_pitch_vel = np.pi / 16
        self.neck_pitch_act_pos = 0.0
        self.tag_pose_avg: Optional[npt.NDArray[np.float32]] = None

        robot_suffix = "_2xc" if "_2xc" in robot.name else "_2xm"
        grasp_motion_path = os.path.join("motion", f"pull_up_grasp{robot_suffix}.lz4")
        if os.path.exists(grasp_motion_path):
            grasp_data_dict = joblib.load(grasp_motion_path)
        else:
            raise ValueError(f"No data files found in {grasp_motion_path}")

        self.grasp_time_arr = np.array(grasp_data_dict["time"], dtype=np.float32)
        self.grasp_action_arr = np.array(grasp_data_dict["action"], dtype=np.float32)

        self.prepared_time = 0.0

        self.last_action = np.zeros(robot.nu, dtype=np.float32)
        self.grasped_count = 0
        self.grasped_time = 0.0
        self.grasped_action = np.zeros(robot.nu, dtype=np.float32)
        self.is_done = False

        pull_motion_path = os.path.join("motion", f"pull_up_pull{robot_suffix}.lz4")
        if os.path.exists(pull_motion_path):
            pull_data_dict = joblib.load(pull_motion_path)
        else:
            raise ValueError(f"No data files found in {pull_motion_path}")

        self.pull_time_arr = np.array(pull_data_dict["time"])
        self.pull_action_arr = np.array(pull_data_dict["action"], dtype=np.float32)

        self.step_curr = 0

        self.is_prepared = False

    def step(
        self, obs: Obs, sim: BaseSim
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a step in the control loop, determining the appropriate action based on the current observation and state.

        Args:
            obs (Obs): The current observation containing time and other relevant data.
            is_real (bool, optional): Flag indicating whether the operation is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and the computed action as a NumPy array.
        """
        is_real = "real" in sim.name
        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 7.0 if is_real else 2.0
            self.prep_time, self.prep_action = get_action_traj(
                0.0,
                self.init_motor_pos,
                self.default_motor_pos,
                self.prep_duration,
                self.control_dt,
                end_time=5.0 if is_real else 0.0,
            )

        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        left_eye_transform = np.eye(4, dtype=np.float32)
        left_eye_transform[:3, 3] = self.root_to_left_eye_t
        tag_pose_avg = np.eye(4, dtype=np.float32)
        tag_pose_avg[:3, 3] = np.array([0.05, -0.017, 0.05], dtype=np.float32)
        self.tag_pose_avg = left_eye_transform @ tag_pose_avg

        if self.grasped_count < 1 / self.control_dt:
            if self.prepared_time == 0:
                self.prepared_time = obs.time

            # Calculate the interpolation factor
            time_elapsed = obs.time - self.prepared_time
            # Find the closest pull action index
            curr_idx = np.argmin(np.abs(self.grasp_time_arr - time_elapsed))
            grasp_action = self.grasp_action_arr[curr_idx]
            grasp_action[self.neck_pitch_idx] = self.neck_pitch_act_pos
            grasp_action[self.waist_motor_indices] = 0.0

            if curr_idx == len(self.grasp_time_arr) - 1:
                self.grasped_count += 1

            self.last_action = grasp_action

            return {}, grasp_action

        else:
            if self.grasped_time == 0:
                self.grasped_time = obs.time
                self.grasped_action = self.last_action

            # Calculate the interpolation factor
            time_elapsed = obs.time - self.grasped_time
            interp_duration = 1.0  # Duration in seconds for the transition
            interp_factor = min(time_elapsed / interp_duration, 1.0)  # Clamp to [0, 1]
            # Find the closest pull action index
            curr_idx = np.argmin(np.abs(self.pull_time_arr - time_elapsed))
            pull_action = (
                1 - interp_factor
            ) * self.grasped_action + interp_factor * self.pull_action_arr[curr_idx]
            pull_action[self.waist_motor_indices] = 0.0

            self.last_action = pull_action

            if curr_idx == len(self.pull_action_arr) - 1:
                self.is_done = True

            return {}, pull_action
