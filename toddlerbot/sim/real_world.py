"""Real-world robot interface for hardware integration.

Provides RealWorld class to interface with physical ToddlerBot hardware,
including Dynamixel motor control and IMU sensor integration.
"""

import platform
import time
from typing import Dict

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

# Use Python SDK wrapper on macOS for better serial support
if platform.system() == "Darwin":
    from toddlerbot.actuation import dynamixel_py_wrapper as dynamixel_cpp
else:
    from toddlerbot.actuation import dynamixel_cpp

from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot


class RealWorld(BaseSim):
    """Real-world robot interface class."""

    def __init__(self, robot: Robot):
        """Initializes the real-world robot interface.

        Args:
            robot (Robot): An instance of the Robot class containing configuration details.

        Attributes:
            has_imu (bool): Indicates if the robot is equipped with an Inertial Measurement Unit (IMU).
            has_dynamixel (bool): Indicates if the robot uses Dynamixel motors.
            negated_motor_names (List[str]): A list of motor names that require direction negation due to URDF configuration issues.
        """
        super().__init__("real_world")

        self.robot = robot

        # Check if IMU is needed based on robot configuration
        has_imu_config = self.robot.config.get("general", {}).get("has_imu", True)
        
        self.imu = None
        if has_imu_config:
            try:
                from toddlerbot.sensing.IMU import ThreadedIMU
                self.imu = ThreadedIMU()
                self.imu.start()
            except Exception as e:
                print(f"IMU not found: {e}")
        else:
            print("IMU disabled in robot configuration (has_imu: false)")

        self.controllers = []
        try:
            # On Mac, need full path; on Linux, just device name
            port_name = "/dev/tty.usbserial-FTAK8D39" if platform.system() == "Darwin" else "ttyUSB0"
            self.controllers = dynamixel_cpp.create_controllers(
                port_name,
                robot.motor_kp_real,
                robot.motor_kd_real,
                robot.motor_zero_pos,
                ["extended_position"] * robot.nu,
                2000000,
                1,
            )
            dynamixel_cpp.initialize(self.controllers)
            self.motor_ids = dynamixel_cpp.get_motor_ids(self.controllers)
            motor_ids_flat = np.array(
                sum((self.motor_ids[key] for key in sorted(self.motor_ids)), [])
            )
            self.motor_sort_idx = np.argsort(motor_ids_flat)
            self.motor_unsort_idx = motor_ids_flat
            self.motor_lens = [
                len(self.motor_ids[key]) for key in sorted(self.motor_ids)
            ]

        except Exception as e:
            print(f"Dynamixel controller not found: {e}")

        # Warm up the observation retrieval
        if self.imu:
            imu_data = self.imu.get_latest_state()
            counter = 0
            while not imu_data:
                counter += 1
                print(
                    f"\rWaiting for real-world observation data... [{counter}]",
                    end="",
                    flush=True,
                )
                imu_data = self.imu.get_latest_state()

            print("\nData received.")

    def step(self):
        """Perform a simulation step (no-op for real world interface)."""
        pass

    def motor_cur_to_tor(
        self, motor_cur: npt.NDArray[np.float32], motor_vel: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Convert motor current to torque using motor characteristics.

        Args:
            motor_cur: Motor current values in milliamps.
            motor_vel: Motor velocity values.

        Returns:
            Calculated motor torque values.
        """
        motor_cur_amp = motor_cur / 1000
        motor_tor = (
            np.maximum(np.abs(motor_cur_amp) - self.robot.motor_bi, 0)
            / self.robot.motor_ki
            * np.sign(motor_cur_amp)
        )
        motor_tor[motor_cur_amp * motor_vel < 0] *= 3
        motor_tor = np.where(
            self.robot.cur_sensor_mask == 1,
            motor_tor,
            motor_cur_amp * self.robot.motor_tau_max,
        )
        return motor_tor

    # @profile()
    def get_observation(self, retries: int = 0):
        """Retrieve and process sensor observations asynchronously.

        This method collects data from available sensors, such as Dynamixel motors and IMU, using asynchronous calls. It processes the collected data to generate a comprehensive observation object.

        Args:
            retries (int, optional): The number of retry attempts for obtaining motor state data. Defaults to 0.

        Returns:
            An observation object containing processed sensor data, including motor states and, if available, IMU angular velocity and Euler angles.
        """
        motor_state = dynamixel_cpp.get_motor_states(self.controllers, 0)

        all_motor_pos = []
        all_motor_vel = []
        all_motor_cur = []

        for key in sorted(self.motor_ids.keys()):
            data = motor_state[key]
            pos = data["pos"]
            vel = data["vel"]
            cur = data["cur"]

            # Append all ids and corresponding data
            all_motor_pos.extend(pos)
            all_motor_vel.extend(vel)
            all_motor_cur.extend(cur)

        # Convert to numpy arrays
        motor_pos = np.array(all_motor_pos, dtype=np.float32)[self.motor_sort_idx]
        motor_vel = np.array(all_motor_vel, dtype=np.float32)[self.motor_sort_idx]
        motor_cur = np.array(all_motor_cur, dtype=np.float32)[self.motor_sort_idx]
        motor_tor = self.motor_cur_to_tor(motor_cur, motor_vel)

        # Get the IMU data (optional)
        quat, ang_vel = None, None
        if self.imu:
            quat, _, ang_vel = self.imu.get_latest_state()

        return Obs(
            time=time.monotonic(),
            motor_pos=motor_pos,
            motor_vel=motor_vel,
            motor_cur=motor_cur,
            motor_tor=motor_tor,
            ang_vel=np.array(ang_vel, dtype=np.float32)
            if ang_vel is not None
            else None,
            rot=R.from_quat(quat, scalar_first=True) if quat is not None else None,
        )

    # @profile()
    def set_motor_target(
        self, motor_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        """Sets the target angles for the robot's motors, adjusting for any negated motor directions and updating the positions of Dynamixel motors if present.

        Args:
            motor_angles (Dict[str, float]): A dictionary mapping motor names to their target angles in degrees.
        """
        if isinstance(motor_angles, dict):
            motor_pos = np.array(list(motor_angles.values()), dtype=np.float32)
        else:
            motor_pos = np.array(motor_angles, dtype=np.float32)

        reordered_pos = motor_pos[self.motor_unsort_idx]
        splits = np.split(reordered_pos, np.cumsum(self.motor_lens)[:-1])
        pos_vecs = [x.tolist() for x in splits]
        dynamixel_cpp.set_motor_pos(self.controllers, pos_vecs)

    def set_motor_kps(self, motor_kps: Dict[str, float]):
        """Sets the proportional gain (Kp) values for motors of type 'dynamixel'.

        If the robot has Dynamixel motors, this method updates their Kp values based on the provided dictionary. If a motor's Kp is not specified in the dictionary, it defaults to the value in the robot's configuration.

        Args:
            motor_kps (Dict[str, float]): A dictionary mapping motor names to their desired Kp values.
        """
        dynamixel_kps = np.array(self.robot.motor_kp_real, dtype=np.float32)
        for i, k in enumerate(self.robot.motor_ordering):
            if k in motor_kps:
                dynamixel_kps[i] = motor_kps[k]

        # Reorder and split gains for each controller
        reordered_kps = dynamixel_kps[self.motor_unsort_idx]
        splits = np.split(reordered_kps, np.cumsum(self.motor_lens)[:-1])
        kp_vecs = [x.tolist() for x in splits]
        dynamixel_cpp.set_motor_kps(self.controllers, kp_vecs)

    def close(self):
        """Closes all active components.

        This method checks for active components such as Dynamixel motors and IMU sensors. It safely closes the connections to these components, ensuring that resources are released properly.
        """
        if self.imu:
            self.imu.close()

        dynamixel_cpp.close(self.controllers)
