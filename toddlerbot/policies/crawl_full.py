from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.reference.crawl_full_ref import CrawlFullReference
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class CrawlFullPolicy(MJXPolicy):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        path: str,
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        """Initializes the CrawlFullPolicy with specific parameters."""
        super().__init__(name, robot, init_motor_pos, path, joystick, fixed_command)

        # "get_down_chair", "crawl_chair", "get_up_chair"
        self.mode = ""
        self.mode_start_time = 0.0  # Time when current mode started
        self._last_mode = None  # Track mode changes
        self.get_down_end = 340
        self.crawl_start = 375
        self.crawl_end = 500
        self.get_up_start = 775

        motion_ref = CrawlFullReference(robot, self.control_dt)
        state_ref = motion_ref.get_default_state()
        state_ref = motion_ref.get_state_ref(0.0, np.zeros(3), state_ref)

        self.default_action = state_ref["motor_pos"][self.action_mask]
        self.ref_motor_pos = state_ref["motor_pos"].copy()

        self.num_frames = motion_ref.full_episode_length

        # NOTE: For get_up mode since the obs_history is not exactly the same in the beginning, action is unstable
        # if self.mode == "get_up":
        #     state_ref = motion_ref.get_default_state()
        #     state_ref = motion_ref.get_state_ref(12.0, np.zeros(3), state_ref)
        #     self.ref_motor_pos = state_ref["motor_pos"].copy()

        #     # Initialize last_action to match frame 600 action to prevent jumps
        #     self.last_action = state_ref["motor_pos"][self.action_mask]

        self.is_done = False
        self.prev_control_inputs = None
        self.prev_motor_target = None

    def get_phase_signal(
        self, time_curr: float, init_idx: int = 0, num_frames: int = None
    ):
        """Get the phase signal for the current time."""
        if num_frames is None:
            num_frames = self.num_frames

        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.control_dt).astype(np.int32)

        if self.mode == "get_down_chair":
            # Reset mode_start_time if this is first call after mode change
            if self._last_mode != self.mode:
                self.mode_start_time = time_curr
                self._last_mode = self.mode
                self.prev_control_inputs = None
                self.prev_motor_target = None

            elapsed_time = time_curr - self.mode_start_time
            elapsed_frames = np.floor(elapsed_time / self.control_dt).astype(int)

            # NOTE: After get down, policy is not overfit to phase signal so it starts to crawl if not using prev actions to stop
            total_idx = init_idx + elapsed_frames
            # Clamp to get_down motion only
            clamped_idx = np.minimum(elapsed_frames, self.get_down_end)
            phase = (clamped_idx / num_frames) * 2 * np.pi
            # print(clamped_idx)
            if clamped_idx >= self.get_down_end:
                self.is_done = True
        elif self.mode == "crawl_chair":
            crawl_loop_length = self.crawl_end - self.crawl_start

            # Reset mode_start_time if this is first call after mode change
            if self._last_mode != self.mode:
                self.mode_start_time = time_curr
                self._last_mode = self.mode
                self.prev_control_inputs = None
                self.prev_motor_target = None

            # Use time-based indexing from mode start
            elapsed_time = time_curr - self.mode_start_time
            elapsed_frames = np.floor(elapsed_time / self.control_dt).astype(int)

            # In crawling loop phase
            loop_idx = elapsed_frames % crawl_loop_length
            clamped_idx = self.crawl_start + loop_idx

            phase = (clamped_idx / num_frames) * 2 * np.pi
        elif self.mode == "get_up_chair":
            # Reset mode_start_time if this is first call after mode change
            if self._last_mode != self.mode:
                self.mode_start_time = time_curr
                self._last_mode = self.mode
                self.prev_control_inputs = None
                self.prev_motor_target = None

            # Use time-based indexing from mode start
            elapsed_time = time_curr - self.mode_start_time
            elapsed_frames = np.floor(elapsed_time / self.control_dt).astype(int)
            total_idx = self.get_up_start + elapsed_frames
            clamped_idx = np.minimum(total_idx, self.num_frames)
            phase = (clamped_idx / num_frames) * 2 * np.pi
            if clamped_idx >= self.num_frames:
                self.is_done = True
        else:
            # Default behavior
            total_idx = (init_idx + time_idx) % num_frames
            phase = (total_idx / num_frames) * 2 * np.pi

        phase_signal = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)
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
        # NOTE: Return prev actions when done before super().step to avoide unnecessary observation history update
        if (
            self.prev_control_inputs is not None
            and self.prev_motor_target is not None
            and self.is_done
        ):
            return self.prev_control_inputs, self.prev_motor_target
        control_inputs, motor_target = super().step(obs, sim)
        self.prev_control_inputs = control_inputs.copy()
        self.prev_motor_target = motor_target.copy()
        return control_inputs, motor_target
