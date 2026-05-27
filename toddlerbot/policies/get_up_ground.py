from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.reference.get_up_prone_ref import GetUpProneReference
from toddlerbot.reference.get_up_roll_ref import GetUpRollReference
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import get_action_traj, interpolate_action

# This script replays a keyframe animation or recorded motion data.


class GetUpGroundPolicy(BasePolicy):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
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

        self.get_up_prone_ref = GetUpProneReference(robot, self.control_dt)
        self.get_up_roll_ref = GetUpRollReference(robot, self.control_dt)

        self.left_shoulder_pitch_idx = self.robot.motor_ordering.index(
            "left_shoulder_pitch"
        )
        self.right_shoulder_pitch_idx = self.robot.motor_ordering.index(
            "right_shoulder_pitch"
        )

        self.motion_ref = None
        self.last_state_ref = None

        self.is_prepared = False
        self.is_done = True
        self.time_start = None

        self.reset()

    def reset(self):
        self.step_curr = 0
        self.get_up_mode = None
        self.fall_down_counter = 0

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
        body_forward_vec = obs.rot.apply(np.array([1, 0, 0], dtype=np.float32))
        # print(f"Body forward vector: {body_forward_vec}")

        if self.get_up_mode is None and self.is_done:
            if abs(body_forward_vec[2]) > 0.5:  # Robot is tilted significantly
                self.fall_down_counter += 1

        if self.fall_down_counter > 20:
            if body_forward_vec[2] < -0.5:  # Robot is face down (prone)
                self.motion_ref = self.get_up_prone_ref
                left_shoulder_pitch_pos = obs.motor_pos[self.left_shoulder_pitch_idx]
                right_shoulder_pitch_pos = obs.motor_pos[self.right_shoulder_pitch_idx]
                is_left_arm_down = (
                    left_shoulder_pitch_pos > -np.pi / 2
                    and left_shoulder_pitch_pos < np.pi / 2
                )
                is_right_arm_down = (
                    right_shoulder_pitch_pos > -np.pi / 2
                    and right_shoulder_pitch_pos < np.pi / 2
                )
                if is_left_arm_down and is_right_arm_down:
                    self.get_up_mode = 0
                elif not is_left_arm_down and not is_right_arm_down:
                    self.get_up_mode = 1
                elif is_left_arm_down and not is_right_arm_down:
                    self.get_up_mode = 2
                else:  # not is_left_arm_down and is_right_arm_down
                    self.get_up_mode = 3

                print(f"Get up prone mode: {self.get_up_mode}")

            elif body_forward_vec[2] > 0.5:  # Robot is on back (supine)
                self.motion_ref = self.get_up_roll_ref
                body_left_vec = obs.rot.apply(np.array([0, 1, 0], dtype=np.float32))
                if (
                    abs(body_left_vec[2]) < 0.1736
                ):  # cos(80°) ≈ 0.1736, more restrictive than π/8
                    self.get_up_mode = 0
                elif body_left_vec[2] >= 0.1736:
                    self.get_up_mode = 1
                else:  # roll_dot_product <= -0.1736
                    self.get_up_mode = 2

                print(f"Get up roll mode: {self.get_up_mode}")

            self.fall_down_counter = 0
            self.is_prepared = False

        if self.fall_down_counter > 0:
            return {}, obs.motor_pos

        if not self.is_prepared:
            self.is_prepared = True

            if self.get_up_mode is None:
                prep_duration = 2.0
                self.prep_time, self.prep_action = get_action_traj(
                    obs.time,
                    obs.motor_pos,
                    self.default_motor_pos,
                    prep_duration,
                    self.control_dt,
                    end_time=0.0,
                )

            else:
                state_ref = self.motion_ref.get_default_state()
                state_ref = self.motion_ref.get_state_ref(
                    0, [self.get_up_mode], state_ref
                )
                ref_motor_pos = state_ref["motor_pos"].copy()
                self.last_state_ref = state_ref

                prep_duration = 1.5
                self.prep_time, self.prep_action = get_action_traj(
                    obs.time,
                    obs.motor_pos,
                    ref_motor_pos,
                    prep_duration,
                    self.control_dt,
                    end_time=0.5,
                )

            self.time_start = obs.time + prep_duration

        if obs.time < self.time_start:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        if self.get_up_mode is None:
            return {}, self.default_motor_pos

        state_ref = self.motion_ref.get_state_ref(
            self.step_curr * self.control_dt,
            [self.get_up_mode],
            self.last_state_ref,
        )
        self.last_state_ref = state_ref

        action = state_ref["motor_pos"].copy()

        self.step_curr += 1

        self.is_done = self.step_curr >= self.motion_ref.n_frames
        if self.is_done:
            self.reset()

        return {}, action
