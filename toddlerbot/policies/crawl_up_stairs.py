from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.reference.crawl_up_stairs_ref import CrawlUpStairsReference
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class CrawlUpStairsPolicy(MJXPolicy):
    """Crawl up stairs policy for the toddlerbot robot."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        path: str,
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        """Initializes the CrawlUpStairsPolicy with specific parameters."""
        super().__init__(name, robot, init_motor_pos, path, joystick, fixed_command)

        motion_ref = CrawlUpStairsReference(robot, self.control_dt)
        state_ref = motion_ref.get_default_state()
        state_ref = motion_ref.get_state_ref(0.0, np.zeros(3), state_ref)
        self.default_action = state_ref["motor_pos"][self.action_mask]
        self.ref_motor_pos = state_ref["motor_pos"].copy()

        self.num_frames = motion_ref.full_episode_length
        self.count = 0
        self.loop_count = 0  # Track number of completed loops
        self.is_done = False
        self.ctrl_inputs = {}
        self.mtr_target = None

    def reset(self):
        super().reset()
        self.count = 0
        self.loop_count = 0
        self.is_done = False
        self.ctrl_inputs = {}
        self.mtr_target = None

    def get_phase_signal(self, time_curr: float, init_idx: int = 0, num_frames=None):
        """Get the phase signal for the current time."""
        # Use self.num_frames if num_frames not provided
        if num_frames is None:
            num_frames = self.num_frames

        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.control_dt).astype(np.int32)

        # Loop between frames 200-650 after reaching frame 650
        loop_start = 200
        loop_end = 650
        loop_length = loop_end - loop_start

        total_idx = init_idx + time_idx

        if total_idx < loop_end:
            clamped_idx = total_idx
        else:
            # In looping phase
            loop_time_idx = total_idx - loop_end
            current_loop = loop_time_idx // loop_length
            loop_idx = loop_time_idx % loop_length

            # Check if we completed a new loop
            if current_loop > self.loop_count:
                self.loop_count = current_loop
                # Stop after 1 loop (total 2 passes through loop section)
                if self.loop_count >= 1:
                    self.is_done = True

            clamped_idx = loop_start + loop_idx

        # Calculate phase based on clamped_idx
        phase = (clamped_idx / num_frames) * 2 * np.pi
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
        # If done, freeze at last position
        if self.is_done and self.mtr_target is not None:
            return self.ctrl_inputs, self.mtr_target

        control_inputs, motor_target = super().step(obs, sim)

        # Save last outputs for freezing when done
        self.ctrl_inputs = control_inputs
        self.mtr_target = motor_target

        return control_inputs, motor_target
