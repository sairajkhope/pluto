import os
from typing import Dict, Tuple

import jax
import joblib

from toddlerbot.reference.motion_ref import MotionReference
from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as jnp

# Define a threshold for determining ground contact. If a foot's Z-height is below this, it's considered to be in stance.
STANCE_HEIGHT_THRESHOLD = 0.02


class ClimbDownBoxReference(MotionReference):
    def __init__(
        self, robot, dt: float, fixed_base: bool = False, playback_speed: float = 1.0
    ):
        super().__init__("climb_down_box", "keyframe", robot, dt, fixed_base)
        robot_suffix = "_2xc" if "2xc" in robot.name else "_2xm"
        motion_file_path = os.path.join("motion", f"climb_down_box{robot_suffix}.lz4")
        loaded_motion_ref = joblib.load(motion_file_path)

        # Remove incompatible data types
        del loaded_motion_ref["keyframes"]
        del loaded_motion_ref["timed_sequence"]

        original_n_frames = len(loaded_motion_ref["time"])

        # Create new indices by stepping through the original data at the desired speed
        new_indices = jnp.arange(0, original_n_frames, playback_speed)

        # The new number of frames is the length of our resampled indices
        self.n_frames = len(new_indices)

        self.motion_ref = {}
        # Use interpolation to resample each data array
        for key, value in loaded_motion_ref.items():
            # Skip non-array or metadata entries
            if not isinstance(value, type(loaded_motion_ref["qpos"])):
                self.motion_ref[key] = value
                continue

            if key == "time":
                continue

            original_data = jnp.array(value)
            # Flatten all feature dimensions for easier interpolation
            original_feature_shape = original_data.shape[1:]
            data_flat = original_data.reshape(original_n_frames, -1)

            # Create a list of interpolation functions, one for each feature
            interp_funcs = [
                lambda x, col=col: jnp.interp(x, jnp.arange(original_n_frames), col)
                for col in data_flat.T
            ]

            # Apply interpolation for each feature and stack results, then reshape
            resampled_flat = jnp.array([func(new_indices) for func in interp_funcs]).T
            resampled_data = resampled_flat.reshape(
                (self.n_frames,) + original_feature_shape
            )

            # If the current key is for quaternions, renormalize them
            if "quat" in key:
                # Calculate the norm along the last axis (the 4 components of the quaternion)
                norms = jnp.linalg.norm(resampled_data, axis=-1, keepdims=True)
                # Avoid division by zero for zero-quaternions
                resampled_data = jnp.where(
                    norms != 0, resampled_data / norms, jnp.zeros_like(resampled_data)
                )

            self.motion_ref[key] = resampled_data

        self.time_arr = jnp.arange(0, self.n_frames) * dt
        self.motion_ref["time"] = self.time_arr

        self.full_episode_length = self.n_frames

        self.is_robot_relative_frame = self.motion_ref.get(
            "is_robot_relative_frame", False
        )

        # Track availability of optional motion data for get_state_ref
        self.has_action = self.motion_ref.get("action") is not None
        self.has_motor_vel = self.motion_ref.get("motor_vel") is not None
        self.has_joint_vel = self.motion_ref.get("joint_vel") is not None

        print("\n=== CLIMB DOWN BOX MOTION DATA ===")
        print(f"  - path     : {motion_file_path}")
        print(f"  - playback speed: {playback_speed}x")
        print(f"  - qpos     : {self.motion_ref['qpos'].shape}")
        if self.has_action:
            print(f"  - action   : {self.motion_ref['action'].shape}")
        else:
            print("  - action   : Not provided in motion data (will extract from qpos)")
        if self.has_motor_vel:
            print(f"  - motor_vel: {self.motion_ref['motor_vel'].shape}")
        else:
            print("  - motor_vel: Not provided in motion data (will use zeros)")
        if self.has_joint_vel:
            print(f"  - joint_vel: {self.motion_ref['joint_vel'].shape}")
        else:
            print("  - joint_vel: Not provided in motion data (will use zeros)")
        print(f"  - is_robot_relative_frame: {self.is_robot_relative_frame}")
        print(f"  - body_pos : {self.motion_ref['body_pos'].shape}")
        print(f"  - body_quat: {self.motion_ref['body_quat'].shape}")
        print(f"  - site_pos : {self.motion_ref['site_pos'].shape}")
        print(f"  - site_quat: {self.motion_ref['site_quat'].shape}")
        print(f"  - body_lin_vel: {self.motion_ref['body_lin_vel'].shape}")
        print(f"  - body_ang_vel: {self.motion_ref['body_ang_vel'].shape}")

        if self.use_jax:
            self.time_arr = jax.device_put(self.time_arr)
            self.motion_ref = jax.device_put(self.motion_ref)

    def get_phase_signal(self, time_curr: float) -> ArrayType:
        time_idx = jnp.floor(time_curr / self.dt).astype(jnp.int32)
        phase_idx = time_idx % self.n_frames
        phase = (phase_idx / self.n_frames) * 2 * jnp.pi
        return jnp.array([jnp.sin(phase), jnp.cos(phase)], dtype=jnp.float32)

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        return jnp.zeros(3, dtype=jnp.float32), jnp.zeros(3, dtype=jnp.float32)

    def get_state_ref(
        self,
        time_curr: float,
        command: ArrayType,
        last_state: Dict[str, ArrayType],
        init_idx: int = 0,
    ) -> Dict[str, ArrayType]:
        time_idx = jnp.floor(time_curr / self.dt).astype(jnp.int32)
        curr_idx = (init_idx + time_idx) % self.n_frames

        qpos = self.motion_ref["qpos"][curr_idx]
        if self.fixed_base:
            qpos = qpos[7:]

        joint_pos = qpos[self.q_start_idx + self.mj_joint_indices]

        # Use action from motion file if available, otherwise extract motor positions from qpos
        if self.has_action:
            motor_pos = self.motion_ref["action"][curr_idx]
        else:
            motor_pos = qpos[self.q_start_idx + self.mj_motor_indices]

        # Use motor_vel from motion file if available, otherwise return zeros
        if self.has_motor_vel:
            motor_vel = self.motion_ref["motor_vel"][curr_idx]
        else:
            motor_vel = jnp.zeros(len(self.mj_motor_indices), dtype=jnp.float32)

        # Use joint_vel from motion file if available, otherwise return zeros
        if self.has_joint_vel:
            joint_vel = self.motion_ref["joint_vel"][curr_idx]
        else:
            joint_vel = jnp.zeros(len(self.mj_joint_indices), dtype=jnp.float32)

        # # The site order is [left_hand, right_hand, left_foot, right_foot]
        # foot_positions = self.motion_ref["site_pos"][curr_idx, 2:4]
        # foot_z_heights = foot_positions[:, 2]  # Extract just the Z-coordinate
        # # A foot is in stance if its height is below the threshold
        # stance_mask = (foot_z_heights < STANCE_HEIGHT_THRESHOLD).astype(jnp.float32)
        stance_mask = jnp.ones(2, dtype=jnp.float32)

        return {
            **self.integrate_path_state(command, last_state),
            "motor_pos": motor_pos,
            "motor_vel": motor_vel,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "qpos": qpos,
            "body_pos": self.motion_ref["body_pos"][curr_idx],
            "body_quat": self.motion_ref["body_quat"][curr_idx],
            "body_lin_vel": self.motion_ref["body_lin_vel"][curr_idx],
            "body_ang_vel": self.motion_ref["body_ang_vel"][curr_idx],
            "site_pos": self.motion_ref["site_pos"][curr_idx],
            "site_quat": self.motion_ref["site_quat"][curr_idx],
            "stance_mask": stance_mask,
        }
