"""Get up motion reference implementation for toddlerbot.

Provides get up motion references using precomputed motion data.
"""

import os
from typing import Dict, Tuple

import jax
import joblib

from toddlerbot.reference.motion_ref import MotionReference
from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np


class GetUpRollReference(MotionReference):
    """Motion reference for get up movements using precomputed motion data."""

    def __init__(self, robot, dt: float, fixed_base: bool = False):
        """Initialize get up motion reference.

        Args:
            robot: Robot instance for motion generation.
            dt: Time step for motion reference.
            fixed_base: Whether to use fixed base mode.
        """
        super().__init__("get_up_roll", "keyframe", robot, dt, fixed_base)

        # Load get up motion data
        robot_suffix = "_2xc" if "2xc" in robot.name else "_2xm"

        get_up_roll_file_path = os.path.join("motion", f"get_up_roll{robot_suffix}.lz4")
        get_up_roll_left_file_path = os.path.join(
            "motion", f"get_up_roll_left{robot_suffix}.lz4"
        )
        get_up_roll_right_file_path = os.path.join(
            "motion", f"get_up_roll_right{robot_suffix}.lz4"
        )
        get_up_roll_ref = joblib.load(get_up_roll_file_path)
        get_up_roll_left_ref = joblib.load(get_up_roll_left_file_path)
        get_up_roll_right_ref = joblib.load(get_up_roll_right_file_path)

        motion_data = {
            "roll": get_up_roll_ref,
            "roll_left": get_up_roll_left_ref,
            "roll_right": get_up_roll_right_ref,
        }
        self.get_up_mode_list = list(motion_data.keys())

        n_frames_max = 0.0
        mode_max = ""
        # print("\n=== MOTION DATA ===")
        for mode, ref in motion_data.items():
            # assert not ref["is_robot_relative_frame"], (
            #     f"{mode} is not in the global frame"
            # )

            for key in ["keyframes", "timed_sequence"]:
                if key in ref:
                    del ref[key]

            n_frames = ref["qpos"].shape[0]
            if n_frames_max < n_frames:
                n_frames_max = n_frames
                mode_max = mode

        self.time_arr = np.array(motion_data[mode_max]["time"])
        self.n_frames = n_frames_max

        field_list = [
            "qpos",
            "action",
            "body_pos",
            "body_quat",
            "body_lin_vel",
            "body_ang_vel",
            "site_pos",
            "site_quat",
        ]
        # Pad shorter motions with their last frame
        # print(f"\n=== PADDING TO {n_frames_max} FRAMES ===")
        for mode, ref in motion_data.items():
            n_frames = ref["qpos"].shape[0]
            if n_frames < n_frames_max:
                pad_frames = n_frames_max - n_frames
                # print(f"  - Padding {mode}: {n_frames} -> {n_frames_max} frames")

                for field in field_list:
                    if field in ref:
                        last_frame = ref[field][-1:]  # Keep dimensions
                        # Repeat last frame pad_frames times
                        padding = np.repeat(last_frame, pad_frames, axis=0)
                        # Concatenate original data with padding
                        motion_data[mode][field] = np.concatenate(
                            [ref[field], padding], axis=0
                        )
                        # print(f"    - {field}: {ref[field].shape}")
            else:
                pass
                # print(f"  - {mode}: Already {n_frames_max} frames")

        self.is_robot_relative_frame = True

        # For non-JAX mode, keep original structure
        self.motion_ref = {}
        for field in field_list:
            self.motion_ref[field] = np.stack(
                [ref[field] for ref in motion_data.values()]
            )

        if self.use_jax:
            # Keep large arrays on CPU, only convert small arrays to JAX
            self.time_arr = jax.device_put(self.time_arr)
            self.motion_ref = jax.device_put(self.motion_ref)

    def get_phase_signal(self, time_curr: float, init_idx: int = 0) -> ArrayType:
        """Get the phase signal for the current time."""
        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.dt).astype(np.int32)
        total_idx = (init_idx + time_idx) % self.n_frames

        # Calculate phase based on total_idx
        phase = (total_idx / self.n_frames) * 2 * np.pi
        phase_signal = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

        return phase_signal

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        """Get the desired linear and angular velocities."""
        lin_vel = np.zeros(3, dtype=np.float32)  # No linear velocity
        ang_vel = np.zeros(3, dtype=np.float32)  # No rotation
        return lin_vel, ang_vel

    def get_state_ref(
        self,
        time_curr: float,
        command: ArrayType,
        last_state: Dict[str, ArrayType],
        init_idx: int = 0,
    ) -> Dict[str, ArrayType]:
        """Get the reference state for the current time. Supports RIS if fed init_idx

        Args:
            time_curr (float): The current time.
            command (ArrayType): Command inputs for the robot's movement.
            last_state (Dict[str, ArrayType]): The last state of the robot.
            init_idx (int, optional): Starting initial state index for RIS. Defaults to 0.

        Returns:
            Dict[str, ArrayType]: A dictionary containing the path state, motor positions, joint positions, body poses, and other reference data.
        """
        # Calculate the index based on time and init_idx
        time_idx = np.floor(time_curr / self.dt).astype(np.int32)
        # Cartwheel motion is not periodic
        curr_idx = np.min(np.array([init_idx + time_idx, self.n_frames - 1]))

        get_up_mode_idx = command[0]

        # Get reference qpos from keyframes using JAX-compatible indexing
        qpos = self.motion_ref["qpos"][get_up_mode_idx, curr_idx]
        if self.fixed_base:
            qpos = qpos[7:]  # Skip first 7 elements for fixed base

        joint_pos = qpos[self.q_start_idx + self.mj_joint_indices]
        # Get motor positions from action data
        motor_pos = self.motion_ref["action"][get_up_mode_idx, curr_idx]

        # OPTIMIZATION: Use direct indexing for body poses (like walk_zmp_ref)
        body_pos = self.motion_ref["body_pos"][get_up_mode_idx, curr_idx]
        body_quat = self.motion_ref["body_quat"][get_up_mode_idx, curr_idx]
        body_lin_vel = self.motion_ref["body_lin_vel"][get_up_mode_idx, curr_idx]
        body_ang_vel = self.motion_ref["body_ang_vel"][get_up_mode_idx, curr_idx]

        # Get reference site poses
        site_pos = self.motion_ref["site_pos"][get_up_mode_idx, curr_idx]
        site_quat = self.motion_ref["site_quat"][get_up_mode_idx, curr_idx]
        stance_mask = np.ones(2, dtype=np.float32)

        return {
            "motor_pos": motor_pos,
            "joint_pos": joint_pos,
            "qpos": qpos,
            "body_pos": body_pos,
            "body_quat": body_quat,
            "body_lin_vel": body_lin_vel,
            "body_ang_vel": body_ang_vel,
            "site_pos": site_pos,
            "site_quat": site_quat,
            "stance_mask": stance_mask,
        }
