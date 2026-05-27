from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.reference.crawl_only_ref import CrawlOnlyReference
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class CrawlRotatePolicy(MJXPolicy):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        path: str,
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        """Initializes the CrawlRotatePolicy with specific parameters."""
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

        # Rotation tracking for completion detection
        self.initial_heading = None  # Set on first step after prep
        self.total_rotation = 0.0  # Accumulated rotation in radians
        self.last_heading = None  # Previous heading for delta calculation
        self.crossed_180 = False  # Track if we've crossed the ±180° boundary
        self.rotation_threshold = np.pi * 0.98  # (slightly less than 180°)
        self.is_done = False
        self.ctrl_inputs = None
        self.mtr_target = None

    def reset(self):
        """Reset rotation tracking state."""
        super().reset()
        self.initial_heading = None
        self.total_rotation = 0.0
        self.last_heading = None
        self.crossed_180 = False
        self.is_done = False
        self.ctrl_inputs = None
        self.mtr_target = None

    def get_phase_signal(
        self, time_curr: float, init_idx: int = 0, num_frames: int = None
    ):
        """Get the phase signal for the current time."""
        if num_frames is None:
            num_frames = self.num_frames
        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.control_dt).astype(np.int32)
        # time_idx = np.floor(time_curr / self.control_dt).astype(np.int32) + 175

        # NOTE: Hardcoded start and end frames
        crawl_start = 175
        crawl_end = 300
        # crawl_start = 0
        # crawl_end = 425
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
        # phase = (clamped_idx / num_frames) * 2 * np.pi

        phase_signal = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

        return phase_signal

    def get_command(
        self, obs: Obs, control_inputs: Dict[str, float]
    ) -> npt.NDArray[np.float32]:
        """Generates a command array based on control inputs."""
        command = np.zeros(self.num_commands, dtype=np.float32)
        return command

    def _get_heading_from_obs(self, obs: Obs) -> float:
        """Calculate robot heading from torso→head vector (avoids 180° ambiguity).

        Since torso and head are rigidly connected (no joint between them), we define
        a fixed torso→head vector in the torso's local frame, rotate it to world frame
        using the quaternion, then project onto XY plane to get heading.

        Args:
            obs: Observation containing torso quaternion

        Returns:
            float: Current heading angle in radians [-π, π]
        """
        # Define torso→head vector in torso's local frame
        # Since head side is higher than lower body, use upward direction (+Z)
        # This vector is constant in the torso's body frame
        torso_to_head_local = np.array([0.0, 0.0, 1.0])

        # Rotate to world frame using torso quaternion
        torso_to_head_world = obs.rot.apply(torso_to_head_local)

        # Project onto XY plane to get heading direction
        heading_vec_xy = torso_to_head_world[:2]  # [x, y] components

        # Compute heading angle from projected vector
        heading = np.arctan2(heading_vec_xy[1], heading_vec_xy[0])

        return heading

    def step(
        self, obs: Obs, sim: BaseSim
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a control step based on the observed state."""
        # If already done, return last action
        if self.is_done:
            return self.ctrl_inputs, self.mtr_target

        control_inputs, motor_target = super().step(obs, sim)

        # Set neck pitch based on command
        if obs.time >= self.prep_duration:
            motor_target[self.neck_motor_indices] = self.ref_motor_pos[
                self.neck_motor_indices
            ]

            # Track rotation after prep period
            current_heading = self._get_heading_from_obs(obs)

            # Initialize heading tracking on first step after prep
            if self.initial_heading is None:
                self.initial_heading = current_heading
                self.last_heading = current_heading

            # Accumulate rotation using properly wrapped deltas
            # This handles crossing ±180° boundary correctly
            if self.last_heading is not None:
                delta_heading = current_heading - self.last_heading
                # Wrap delta to [-π, π] to handle boundary crossing
                delta_heading = np.arctan2(np.sin(delta_heading), np.cos(delta_heading))
                # Accumulate signed rotation (preserves direction)
                self.total_rotation += delta_heading
                self.last_heading = current_heading

            # Check if rotation magnitude exceeds threshold
            # Use abs() to handle both CW and CCW rotations
            if abs(self.total_rotation) >= self.rotation_threshold:
                self.ctrl_inputs = control_inputs
                self.mtr_target = motor_target
                self.is_done = True
                print(
                    f"CrawlRotate completed: rotated {np.degrees(self.total_rotation):.1f}°"
                )

        return control_inputs, motor_target
