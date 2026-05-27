from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.reference.climb_wall_ref import ClimbWallReference
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class ClimbWallPolicy(MJXPolicy):
    """Climb wall policy for the toddlerbot robot."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        path: str,
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        """Initializes the ClimbWallPolicy with specific parameters."""
        super().__init__(name, robot, init_motor_pos, path, joystick, fixed_command)

        motion_ref = ClimbWallReference(robot, self.control_dt)
        state_ref = motion_ref.get_default_state()
        state_ref = motion_ref.get_state_ref(0.0, np.zeros(3), state_ref)

        self.default_action = state_ref["motor_pos"][self.action_mask]
        self.ref_motor_pos = state_ref["motor_pos"].copy()

        self.num_frames = motion_ref.full_episode_length
        self.count = 0
        self.is_done = False

    def reset(self):
        super().reset()
        self.count = 0
        self.is_done = False

    def get_phase_signal(self, time_curr: float, init_idx: int = 0, num_frames=None):
        """Get the phase signal for the current time."""
        # Use self.num_frames if num_frames not provided
        if num_frames is None:
            num_frames = self.num_frames

        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.control_dt).astype(np.int32)
        total_idx = (init_idx + time_idx) % num_frames

        # Calculate phase based on total_idx
        phase = (total_idx / num_frames) * 2 * np.pi
        phase_signal = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

        self.count += 1
        return phase_signal

    def get_command(
        self, obs: Obs, control_inputs: Dict[str, float]
    ) -> npt.NDArray[np.float32]:
        """Generates a command array based on control inputs."""
        command = np.zeros(self.num_commands, dtype=np.float32)
        return command

    def step(
        self, obs: Obs, sim: BaseSim
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a control step based on the observed state."""
        control_inputs, motor_target = super().step(obs, sim)
        # if self.count == self.num_frames:
        if self.count == self.num_frames - 300:
            self.ctrl_inputs, self.mtr_target = control_inputs, motor_target
            self.is_done = True
        if self.is_done:
            return self.ctrl_inputs, self.mtr_target
        return control_inputs, motor_target
