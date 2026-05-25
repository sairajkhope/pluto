"""Motor control implementations for ToddlerBot simulation.

Provides PD controllers with asymmetric saturation models for both
Dynamixel motor control and basic position control.
"""

from typing import Dict, Optional

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np


class MotorController:
    """A class for controlling the Dynamixel motors of a robot."""

    def __init__(self, robot: Robot):
        """Initializes the control parameters for a robot's joints using attributes specific to "dynamixel" type actuators.

        Args:
            robot (Robot): An instance of the Robot class from which joint attributes are retrieved.
        """
        self.kp = np.array(robot.motor_kp_sim, dtype=np.float32)
        self.kd = np.array(robot.motor_kd_sim, dtype=np.float32)
        self.tau_max = np.array(robot.motor_tau_max, dtype=np.float32)
        self.q_dot_max = np.array(robot.motor_q_dot_max, dtype=np.float32)
        self.tau_q_dot_max = np.array(robot.motor_tau_q_dot_max, dtype=np.float32)
        self.q_dot_tau_max = np.array(robot.motor_q_dot_tau_max, dtype=np.float32)
        self.tau_brake_max = np.array(robot.motor_tau_brake_max, dtype=np.float32)
        self.kd_min = np.array(robot.motor_kd_min, dtype=np.float32)
        self.passive_active_ratio = robot.passive_active_ratio

    def step(
        self,
        q: ArrayType,
        q_dot: ArrayType,
        q_dot_dot: ArrayType,
        a: ArrayType,
        noise: Dict[str, ArrayType] = {},
    ):
        """
        Compute torque commands for Dynamixel motors with an asymmetric
        saturation model, implemented in JAX.

        The controller produces a PD torque (`kp * (a - q) - kd_min * q_dot`)
        and then clamps it by two different limits:

        **Acceleration-side limit** — a piece-wise linear envelope
        constant `tau_max` until `|q_dot| > q_dot_tau_max`
        linearly tapers to `tau_q_dot_max` at `|q_dot| = q_dot_max`
        zero beyond `q_dot_max`.

        **Brake-side limit** — fixed at `tau_brake_max`.

        Positive joint velocity uses **[ -tau_brake_max, +tau_acc_limit ]**;
        negative velocity uses **[ -tau_acc_limit, +tau_brake_max ]**.

        Any optional input left as **None** defaults to the corresponding
        attribute initialised in `__init__`; otherwise the provided value
        (scalar or vector) is used.

        Args:
            q (ArrayType): Joint positions (rad or m).
            q_dot (ArrayType): Joint velocities (rad/s or m/s).
            a (ArrayType): Desired action (reference position or torque proxy).
            noise (Optional[ArrayType]): Additive noise applied to
                `tau_acc_limit` (useful for system-ID perturbations).

        Returns:
            ArrayType: Torque command after asymmetric saturation.
        """
        kp = self.kp * noise.get("kp", 1.0)
        kd = self.kd * noise.get("kd", 1.0)
        tau_max = self.tau_max * noise.get("tau_max", 1.0)
        q_dot_tau_max = self.q_dot_tau_max * noise.get("q_dot_tau_max", 1.0)
        q_dot_max = self.q_dot_max * noise.get("q_dot_max", 1.0)
        kd_min = self.kd_min * noise.get("kd_min", 1.0)
        tau_brake_max = self.tau_brake_max * noise.get("tau_brake_max", 1.0)
        tau_q_dot_max = self.tau_q_dot_max * noise.get("tau_q_dot_max", 1.0)
        passive_active_ratio = self.passive_active_ratio * noise.get(
            "passive_active_ratio", 1.0
        )

        error = a - q
        real_kp = np.where(q_dot_dot * error < 0, kp * passive_active_ratio, kp)
        tau_m = real_kp * error - (kd_min + kd) * q_dot
        abs_q_dot = np.abs(q_dot)

        # linear taper between (q_dot_tau_max, tau_max) and q_dot_max, tau_q_dot_max)
        slope = (tau_q_dot_max - tau_max) / (q_dot_max - q_dot_tau_max)
        taper_limit = tau_max + slope * (abs_q_dot - q_dot_tau_max)

        tau_acc_limit = np.where(abs_q_dot <= q_dot_tau_max, tau_max, taper_limit)
        tau_m_clamped = np.where(
            np.logical_and(abs_q_dot > q_dot_max, q_dot * a > 0),
            # The following line simulates the dynamixel's self-protection
            np.where(
                q_dot > 0,
                np.ones_like(tau_m) * -tau_brake_max,
                np.ones_like(tau_m) * tau_brake_max,
            ),
            np.where(
                q_dot > 0,
                np.clip(tau_m, -tau_brake_max, tau_acc_limit),
                np.clip(tau_m, -tau_acc_limit, tau_brake_max),
            ),
        )
        return tau_m_clamped


class PositionController:
    """Passes position targets through to MuJoCo `ctrl` for `<position>` actuators."""

    def step(
        self,
        q: ArrayType,
        q_dot: ArrayType,
        q_dot_dot: ArrayType,
        a: ArrayType,
        noise: Optional[Dict[str, ArrayType]] = None,
    ):
        """Return motor position targets as control inputs (must match `MotorController.step` arity).

        Args:
            q: Current motor positions (unused; MuJoCo applies the actuator law).
            q_dot: Current motor velocities (unused).
            q_dot_dot: Current accelerations (unused; kept for API parity with `MotorController`).
            a: Target motor positions (same ordering as `robot.motor_ordering`).

        Returns:
            Control vector for MuJoCo position actuators (target positions).
        """
        del q, q_dot, q_dot_dot, noise
        return a
