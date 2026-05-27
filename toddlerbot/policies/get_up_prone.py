"""Get up from prone position policy.

This module implements a get-up-from-prone policy that extends MJXPolicy to perform
recovery movements from a prone (face-down) position.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.reference.get_up_prone_ref import GetUpProneReference
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class GetUpPronePolicy(MJXPolicy):
    """Get up from prone position policy for the toddlerbot robot."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        path: str,
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        super().__init__(name, robot, init_motor_pos, path, joystick, fixed_command)

        motion_ref = GetUpProneReference(robot, self.control_dt)
        self.n_frames = motion_ref.n_frames
        # self.fixed_command = np.random.randint(
        #     0, len(motion_ref.get_up_mode_list), (1,)
        # )
        self.fixed_command = self.fixed_command.astype(np.int32)
        state_ref = motion_ref.get_default_state()
        state_ref = motion_ref.get_state_ref(0.0, self.fixed_command, state_ref)
        self.default_action = np.mean(
            np.array(list(self.robot.motor_limits.values())), axis=-1
        )
        self.ref_qpos = state_ref["qpos"].copy()
        self.ref_motor_pos = state_ref["motor_pos"].copy()

        self.set_qpos = False

    def get_phase_signal(self, time_curr: float):
        """Get the phase signal for the current time."""
        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.control_dt).astype(np.int32)
        # Calculate phase based on total_idx
        phase = (
            (time_idx / self.n_frames) * 2 * np.pi
            if time_idx < self.n_frames
            else 2 * np.pi
        )
        phase_signal = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)
        return phase_signal

    def step(self, obs: Obs, sim: BaseSim):
        is_real = "real" in sim.name
        if not is_real and not self.set_qpos:
            sim.data.qpos = self.ref_qpos
            sim.forward()
            self.set_qpos = True

        return super().step(obs, sim)
