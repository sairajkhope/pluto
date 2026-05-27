"""Simple standing policy for maintaining default posture.

This module implements a basic standing policy that keeps the robot
in its default motor position configuration.
"""

from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action


class StandPolicy(BasePolicy):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        """Initializes the object with a name, a robot instance, and initial motor positions.

        Args:
            name (str): The name of the object.
            robot (Robot): An instance of the Robot class.
            init_motor_pos (npt.NDArray[np.float32]): An array of initial motor positions.
        """
        super().__init__(name, robot, init_motor_pos)

        self.is_prepared = False
        self.prep_duration = 2.0
        self.min_standby_duration = 0.0  # Minimum time to hold final pose

        self.is_done = False

    def step(
        self, obs: Obs, sim: BaseSim
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Generates the next action based on the current observation and preparation phase.

        Args:
            obs (Obs): The current observation containing the time and other relevant data.
            is_real (bool, optional): Flag indicating if the step is being executed in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and the next action as a numpy array. If the current time is within the preparation duration, the action is interpolated based on the preparation time and action; otherwise, the default motor position is returned.
        """
        # Regenerate prep trajectory if not prepared (like MJXPolicy and ReplayPolicy)
        if not self.is_prepared:
            self.is_prepared = True
            from toddlerbot.utils.math_utils import get_action_traj

            # NOTE: Neck pitch can be configured here if needed, but currently it's set to default position
            motor_pos = self.default_motor_pos.copy()
            # neck_motor_pos = self.robot.neck_ik(
            #     np.array([0.0, 0.85])
            # )  # yaw=0, pitch=0.85
            # motor_pos[self.neck_motor_indices] = neck_motor_pos

            self.prep_time, self.prep_action = get_action_traj(
                0.0,
                self.init_motor_pos,
                motor_pos,
                self.prep_duration,
                self.control_dt,
                end_time=self.min_standby_duration,
            )

        # TODO: obs.time can be much faster than simulation time
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        motor_pos = self.default_motor_pos.copy()
        # NOTE: Neck pitch can be configured here if needed, but currently it's set to default position
        neck_motor_pos = self.robot.neck_ik(np.array([0.0, 0.0]))
        # neck_motor_pos = self.robot.neck_ik(np.array([0.0, 0.85]))  # yaw=0, pitch=0.85
        motor_pos[self.neck_motor_indices] = neck_motor_pos

        return {}, motor_pos
