#!/usr/bin/env python3
"""
Standalone script to send a pull-up signal to the demo policy runner.
"""

import time

from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode


def main():
    """Send pull-up signal to the demo policy runner."""
    print("Sending pull-up signal to demo policy runner...")

    # Create ZMQ sender to connect to port 5556 where the demo policy is listening
    zmq_sender = ZMQNode(type="sender", port=5556)

    # Send the pull-up signal
    try:
        while True:
            message = ZMQMessage(time=time.monotonic(), text="pull_up")
            zmq_sender.send_msg(message)
            time.sleep(0.1)  # Send every 100ms to ensure the message is received
    except KeyboardInterrupt:
        pass

    print("Pull-up signal sent successfully!")
    print("The robot should now transition to pull-up mode when it's standing.")


if __name__ == "__main__":
    main()
