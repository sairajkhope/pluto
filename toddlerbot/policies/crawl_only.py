from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.reference.crawl_only_ref import CrawlOnlyReference
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class CrawlOnlyPolicy(MJXPolicy):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        path: str,
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        """Initializes the CrawlOnlyPolicy with specific parameters."""
        super().__init__(name, robot, init_motor_pos, path, joystick, fixed_command)

        motion_ref = CrawlOnlyReference(robot, self.control_dt)
        state_ref = motion_ref.get_default_state()
        state_ref = motion_ref.get_state_ref(0.0, np.zeros(3), state_ref)

        self.default_action = state_ref["motor_pos"][self.action_mask]
        self.ref_motor_pos = state_ref["motor_pos"].copy()

        # Set neck position in ref_motor_pos based on fixed_command
        if hasattr(self.robot, "neck_joint_limits") and len(self.fixed_command) > 1:
            # Interpolate neck positions from command (same logic as walk_zmp_ref.py)
            neck_yaw_pos = np.interp(
                self.fixed_command[0],
                np.array([-1, 0, 1]),
                np.array(
                    [
                        self.robot.neck_joint_limits[0, 0],
                        0.0,
                        self.robot.neck_joint_limits[1, 0],
                    ]
                ),
            )
            neck_pitch_pos = np.interp(
                self.fixed_command[1],
                np.array([-1, 0, 1]),
                np.array(
                    [
                        self.robot.neck_joint_limits[0, 1],
                        0.0,
                        self.robot.neck_joint_limits[1, 1],
                    ]
                ),
            )

            # Convert joint positions to motor positions using robot's IK
            neck_joint_pos = np.array([neck_yaw_pos, neck_pitch_pos])
            neck_motor_pos = self.robot.neck_ik(neck_joint_pos)

            # Update ref_motor_pos with neck positions
            self.ref_motor_pos[self.neck_motor_indices] = neck_motor_pos

        self.num_frames = motion_ref.full_episode_length

    def get_phase_signal(
        self, time_curr: float, init_idx: int = 0, num_frames: int = None
    ):
        """Get the phase signal for the current time."""
        if num_frames is None:
            num_frames = self.num_frames
        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.control_dt).astype(np.int32)

        # NOTE: Hardcoded start and end frames
        crawl_start = 175
        crawl_end = 300
        crawl_loop_length = crawl_end - crawl_start

        total_idx = init_idx + time_idx

        if total_idx <= crawl_start:
            clamped_idx = total_idx
        else:
            # In crawling loop phase
            crawl_time_idx = total_idx - crawl_start
            loop_idx = crawl_time_idx % crawl_loop_length
            clamped_idx = crawl_start + loop_idx

        # print(total_idx, clamped_idx)
        phase = (clamped_idx / num_frames) * 2 * np.pi

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
        control_inputs, motor_target = super().step(obs, sim)

        # Set neck pitch based on command
        if obs.time >= self.prep_duration:
            motor_target[self.neck_motor_indices] = self.ref_motor_pos[
                self.neck_motor_indices
            ]
        return control_inputs, motor_target
