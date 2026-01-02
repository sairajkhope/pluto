"""Test Dynamixel motor control frequency and communication latency."""

import time
from typing import List

import numpy as np

from toddlerbot.actuation import dynamixel_controller as dynamixel_cpp


# @profile()
def main():
    """Test motor communication latency and control frequency."""
    controllers = dynamixel_cpp.create_controllers(
        "/dev/ttyUSB*",  # "/dev/cu.usbserial-FTAK8D39",
        # "ttyUSB0",
        [900] * 30,
        [0.0] * 30,
        [0.0] * 30,
        ["extended_position"] * 30,
        2000000,
        1,
    )
    dynamixel_cpp.initialize(controllers)
    step_idx = 0
    step_time_list: List[float] = []

    try:
        while True:
            step_start = time.monotonic()
            motor_state = dynamixel_cpp.get_motor_states(controllers, 0)
            step_time = time.monotonic() - step_start
            step_time_list.append(step_time)
            step_idx += 1
            print(f"Step time: {step_time * 1000:.2f} ms")
            print(f"Motor State: {motor_state}")

    except KeyboardInterrupt:
        pass

    finally:
        time.sleep(1)

        dynamixel_cpp.close(controllers)

        print(
            f"Average Latency: {np.mean(step_time_list) * 1000:.2f} +- {np.std(step_time_list) * 1000:.2f} ms"
        )


if __name__ == "__main__":
    main()
