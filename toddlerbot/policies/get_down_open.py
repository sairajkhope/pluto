import os
from typing import Dict, Tuple

import joblib
import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import get_action_traj, interpolate_action


class GetDownOpenPolicy(BasePolicy):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        """Initializes the GetDownOpenPolicy with specific parameters."""
        super().__init__(name, robot, init_motor_pos)

        robot_suffix = "_2xc" if "2xc" in robot.name else "_2xm"
        motion_file_path = os.path.join("motion", f"get_down{robot_suffix}.lz4")
        data_dict = joblib.load(motion_file_path)

        self.time_arr = np.array(data_dict["time"], dtype=np.float32)
        self.action_arr = np.array(data_dict["action"], dtype=np.float32)

        # start_idx = 0
        # for idx, action in enumerate(self.action_arr):
        #     if np.allclose(self.default_motor_pos, action, atol=0.05):
        #         start_idx = idx
        #     elif start_idx != 0:
        #         print(f"Truncating dataset at index {start_idx}...")
        #         break

        # self.time_arr = self.time_arr[start_idx:]
        # self.time_arr -= self.time_arr[0]
        # self.action_arr = self.action_arr[start_idx:]

        self.step_curr = 0
        self.is_prepared = False
        self.time_start = None
        self.is_done = False

        self.prep_duration = 7.0
        self.min_standby_duration = 5.0  # Minimum time to hold final pose

    def step(
        self, obs: Obs, sim: BaseSim
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a single step in the simulation or real environment, returning the current action.

        This function determines the appropriate action to take based on the current observation and whether the environment is real or simulated. It handles the preparation phase if necessary and updates the action based on the current time and keyboard inputs.

        Args:
            obs (Obs): The current observation containing the time and other relevant data.
            is_real (bool, optional): Indicates if the environment is real. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and the action array for the current step.
        """
        if not self.is_prepared:
            self.is_prepared = True
            self.time_start = self.prep_duration
            self.prep_time, self.prep_action = get_action_traj(
                0.0,
                obs.motor_pos,
                self.action_arr[0],
                self.prep_duration,
                self.control_dt,
                end_time=self.min_standby_duration,
            )

        if obs.time < self.time_start:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        # Use pre-constructed looped motion arrays
        time_delta = obs.time - self.time_start
        curr_idx = int(time_delta / self.control_dt)
        curr_idx = np.clip(curr_idx, 0, len(self.action_arr) - 1)
        action = self.action_arr[curr_idx]

        if curr_idx == len(self.action_arr) - 1:
            self.is_done = True

        self.step_curr += 1

        return {}, action
