"""Motion replay policy for executing pre-recorded movements.

This module implements a replay policy that can playback keyframe animations
or recorded motion data with proper interpolation and synchronization.
"""

from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import get_action_traj, interpolate_action


class ReplayPolicy(BasePolicy):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        path: str,
    ):
        """Initializes the class with motion data and configuration.

        Args:
            name (str): The name of the instance.
            robot (Robot): The robot object associated with this instance.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions.
            run_name (str): The name of the run, used to determine the motion file path.

        Raises:
            ValueError: If no data files are found for the specified run name.

        This constructor loads motion data from a specified file based on the `run_name`. If the run name includes "cuddle" or "push_up", it loads a motion file. Otherwise, it attempts to load data from a dataset or pickle file. The method also initializes various attributes related to motion timing and actions, and sets up a keyboard listener for saving keyframes.
        """
        super().__init__(name, robot, init_motor_pos)

        data_dict = joblib.load(path)
        self.action_arr: Optional[npt.NDArray[np.float32]] = None
        self.qpos_arr: Optional[npt.NDArray[np.float32]] = None

        if "time" in data_dict and "action" in data_dict:
            self.time_arr = np.array(data_dict["time"])
            self.action_arr = self._normalize_action_arr(data_dict["action"])
            if "qpos" in data_dict:
                self.qpos_arr = np.array(data_dict["qpos"], dtype=np.float32)

        elif "obs_list" in data_dict and "action_list" in data_dict:
            self.time_arr = np.array([obs.time for obs in data_dict["obs_list"]])
            self.action_arr = np.array(data_dict["action_list"], dtype=np.float32)
        else:
            raise ValueError(
                f"Unsupported replay data format in {path}. "
                "Expected {time, action} or {obs_list, action_list}."
            )

        if self.action_arr is not None:
            self.action_arr = self._align_action_dim(self.action_arr)
            if len(self.time_arr) != len(self.action_arr):
                min_len = min(len(self.time_arr), len(self.action_arr))
                self.time_arr = self.time_arr[:min_len]
                self.action_arr = self.action_arr[:min_len]

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

        self.set_qpos = False

    def _normalize_action_arr(
        self, action_data: object
    ) -> Optional[npt.NDArray[np.float32]]:
        if action_data is None:
            return None

        action_arr = np.array(action_data, dtype=np.float32)
        if action_arr.ndim == 2:
            return action_arr
        if action_arr.ndim == 1 and action_arr.size > 0:
            return action_arr[None, :]
        return None

    def _align_action_dim(
        self, action_arr: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        target_dim = len(self.robot.motor_ordering)
        if action_arr.shape[1] == target_dim:
            return action_arr

        if action_arr.shape[1] < target_dim:
            if self.robot.has_gripper:
                padding_dim = target_dim - action_arr.shape[1]
                return np.concatenate(
                    [action_arr, np.zeros((action_arr.shape[0], padding_dim))], axis=1
                )
            raise ValueError(
                f"Replay action width {action_arr.shape[1]} is smaller than "
                f"robot motor width {target_dim}."
            )

        # If the source has extra dimensions (e.g. gripper joints for no-gripper robot),
        # keep the first motor dimensions only.
        return action_arr[:, :target_dim]

    def _ensure_action_arr(self, sim: BaseSim):
        if self.action_arr is not None:
            return
        if self.qpos_arr is None:
            raise ValueError(
                "Replay motion has no valid action array and no qpos fallback."
            )
        if not hasattr(sim, "q_start_idx") or not hasattr(sim, "motor_indices"):
            raise ValueError(
                "Replay motion has no valid action array. qpos fallback requires "
                "a simulator with `q_start_idx` and `motor_indices`."
            )

        q_start_idx = int(getattr(sim, "q_start_idx"))
        motor_indices = np.asarray(getattr(sim, "motor_indices"))
        self.action_arr = self.qpos_arr[:, q_start_idx + motor_indices]
        self.action_arr = self._align_action_dim(self.action_arr)
        if len(self.time_arr) != len(self.action_arr):
            min_len = min(len(self.time_arr), len(self.action_arr))
            self.time_arr = self.time_arr[:min_len]
            self.action_arr = self.action_arr[:min_len]

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
        self._ensure_action_arr(sim)
        assert self.action_arr is not None

        is_real = "real" in sim.name
        if not is_real and not self.set_qpos and self.qpos_arr is not None:
            sim.data.qpos = self.qpos_arr[0]
            sim.forward()
            self.set_qpos = True
            self.is_prepared = True
            self.time_start = 0.0

        if not self.is_prepared:
            self.is_prepared = True
            self.time_start = self.prep_duration
            self.prep_time, self.prep_action = get_action_traj(
                obs.time,
                obs.motor_pos,
                self.action_arr[0],
                self.prep_duration,
                self.control_dt,
                end_time=self.min_standby_duration,
            )

        if is_real:
            if obs.time < self.time_start:
                action = np.asarray(
                    interpolate_action(obs.time, self.prep_time, self.prep_action)
                )
                return {}, action
        else:
            if self.step_curr * self.control_dt < self.time_start:
                action = np.asarray(
                    interpolate_action(obs.time, self.prep_time, self.prep_action)
                )
                self.step_curr += 1
                return {}, action

        # During simulation, obs.time can be much faster than simulation time
        if is_real:
            time_delta = obs.time - self.time_start
        else:
            time_delta = self.step_curr * self.control_dt - self.time_start
        # TODO: Choose action not based on integer curr_idx but float curr_idx and interpolate
        curr_idx = int(time_delta / self.control_dt)
        curr_idx = np.clip(curr_idx, 0, len(self.action_arr) - 1)
        action = self.action_arr[curr_idx]

        if curr_idx == len(self.action_arr) - 1:
            self.is_done = True

        self.step_curr += 1

        return {}, action
