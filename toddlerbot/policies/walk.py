"""Walking locomotion policy with phase-based gait control.

This module implements a walking policy that generates coordinated
bipedal locomotion using cyclic phase signals and command interpretation.
"""

from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.sim import BaseSim, Obs


class WalkPolicy(MJXPolicy):
    """Walking policy for the toddlerbot robot."""

    # NOTE: Uncomment if using a checkpoint trained with the new crouched default_motor_pos.
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.default_motor_pos = np.array(
    #         [
    #             0.0,
    #             0.85,
    #             0.0,
    #             0.0,  # neck_yaw, neck_pitch, waist_1, waist_2
    #             -0.67,
    #             0.0,
    #             0.0,
    #             -1.02,
    #             0.0,
    #             -0.49,  # left leg
    #             0.67,
    #             0.0,
    #             0.0,
    #             1.02,
    #             0.0,
    #             0.49,  # right leg
    #             0.174533,
    #             0.087266,
    #             1.570796,
    #             -0.523599,
    #             -1.570796,
    #             -1.36,
    #             0.0,  # left arm
    #             -0.174533,
    #             0.087266,
    #             -1.570796,
    #             -0.523599,
    #             1.570796,
    #             1.36,
    #             0.0,  # right arm
    #         ],
    #         dtype=np.float32,
    #     )
    #     self.default_action = self.default_motor_pos[self.action_mask]
    #     self.ref_motor_pos = self.default_motor_pos.copy()

    def get_phase_signal(self, time_curr: float):
        """Calculate the phase signal as a 2D vector for a given time.

        Args:
            time_curr (float): The current time for which to calculate the phase signal.

        Returns:
            np.ndarray: A 2D vector containing the sine and cosine components of the phase signal, with dtype np.float32.
        """
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * time_curr / self.env_cfg["action"]["cycle_time"]),
                np.cos(2 * np.pi * time_curr / self.env_cfg["action"]["cycle_time"]),
            ],
            dtype=np.float32,
        )
        return phase_signal

    def get_command(
        self, obs: Obs, control_inputs: Dict[str, float]
    ) -> npt.NDArray[np.float32]:
        """Generates a command array based on control inputs for walking.

        Args:
            control_inputs (Dict[str, float]): A dictionary containing control inputs with keys 'walk_x', 'walk_y', and 'walk_turn'.

        Returns:
            npt.NDArray[np.float32]: A numpy array representing the command, with the first five elements as zeros and the remaining elements scaled by the command discount factor.
        """
        if len(control_inputs) == 0:
            command = self.fixed_command.copy()
        else:
            command = np.zeros(self.num_commands, dtype=np.float32)
            command[5:] = np.array(
                [
                    control_inputs["walk_x"],
                    control_inputs["walk_y"],
                    control_inputs["walk_turn"],
                ]
            )

        self.target_torso_yaw += command[-1] * self.control_dt

        x_axis = obs.rot.as_matrix()[:, 0]
        torso_yaw = np.arctan2(x_axis[1], x_axis[0])
        # torso_yaw = obs.rot.as_euler("xyz")[2]

        # Wrap the error to [-pi, pi]
        yaw_error = self.target_torso_yaw - torso_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # Fixed yaw correction: deadzone within 0.2 rad
        if abs(yaw_error) <= 0.2:
            command[-1] = 0.0
        else:
            if command[-1] > 0:
                pass
            elif command[-1] < 0:
                pass
            else:
                command[-1] = 0.8 if yaw_error > 0 else -0.2
                # command[-1] = 0.8 if yaw_error > 0 else -0.6
                command[-2] = 0
                command[-3] = 0
        # command[-1] = 0

        # Walk in place for first 3 seconds to stabilize
        # if obs.time < 3.0:
        #     command[-3:] = 0.0
        #     self.target_torso_yaw -= command[-1] * self.control_dt

        # print(f"walk_command: {command[5:]}")
        return command

    def step(
        self, obs: Obs, sim: BaseSim
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a control step based on the observed state and updates the standing status.

        Args:
            obs (Obs): The current observation of the system state.
            is_real (bool, optional): Flag indicating whether the step is being executed in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing the control inputs as a dictionary and the motor target as a NumPy array.
        """
        control_inputs, motor_target = super().step(obs, sim)

        # Set neck pitch based on command
        if obs.time >= self.prep_duration:
            motor_target[self.neck_motor_indices] = self.ref_motor_pos[
                self.neck_motor_indices
            ]

        if len(self.command_list) >= int(1 / self.control_dt):
            last_commands = self.command_list[-int(1 / self.control_dt) :]
            # NOTE: Check if walk velocities are zero (x, y) and yaw correction is small
            # Index 5: walk_x, Index 6: walk_y should be exactly 0
            # Index 7: walk_turn + yaw_correction should be small (threshold for yaw correction drift)
            yaw_threshold = 0.1  # Allow small yaw corrections
            all_zeros = all(
                command[5] == 0 and command[6] == 0 and abs(command[7]) < yaw_threshold
                for command in last_commands
            )
            # self.is_standing = all_zeros and abs(self.phase_signal[1]) > 1 - 1e-6
            self.is_standing = all_zeros
        else:
            self.is_standing = False

        return control_inputs, motor_target
