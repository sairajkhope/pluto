"""Test script for motor actuation model behavior."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toddlerbot.sim.motor_control import MotorController
from toddlerbot.utils.array_utils import array_lib as np


# Dummy robot definition with required fields
class DummyRobot:
    """Dummy robot for testing motor control parameters."""

    def __init__(self, dof: int = 1):
        self.motor_kp_sim = np.ones(dof) * 10.0
        self.motor_kd_sim = np.ones(dof) * 0.0
        self.motor_tau_max = np.ones(dof) * 0.68
        self.motor_q_dot_max = np.ones(dof) * 6.52
        self.motor_tau_q_dot_max = np.ones(dof) * 0.49
        self.motor_q_dot_tau_max = np.ones(dof) * 1.0
        self.motor_tau_brake_max = np.ones(dof) * 1.54
        self.motor_kd_min = np.ones(dof) * 0.341
        self.passive_active_ratio = 1.0


def main():
    """Test motor controller torque output vs velocity for different action values."""
    dof = 1
    robot = DummyRobot(dof)
    controller = MotorController(robot)

    q_test = 0.0

    q_dot_vals = np.linspace(-20, 20, 200)
    q = np.ones((200, dof)) * q_test
    q_dot = q_dot_vals[:, None]

    q_dot_dot = np.zeros((200, dof))

    # Evaluate for a = -0.5
    a_neg = np.ones((200, dof)) * -0.5
    tau_neg = controller.step(q, q_dot, q_dot_dot, a_neg)

    # Evaluate for a = 0.5
    a_pos = np.ones((200, dof)) * 0.5
    tau_pos = controller.step(q, q_dot, q_dot_dot, a_pos)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(q_dot_vals, tau_neg[:, 0], label="a = -0.5", linestyle="-")
    plt.plot(q_dot_vals, tau_pos[:, 0], label="a = 0.5", linestyle="--")

    plt.title("Torque vs q_dot for a Single DOF")
    plt.xlabel("q_dot (rad/s)")
    plt.ylabel("Torque Output")
    plt.grid(True)
    plt.legend()

    # Annotation
    plt.text(
        0.05,
        0.95,
        f"q = {q_test}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("actuation_model.png")


if __name__ == "__main__":
    main()
