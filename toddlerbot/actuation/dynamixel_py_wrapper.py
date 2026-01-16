"""Python Dynamixel SDK wrapper for macOS compatibility.

This module provides a minimal interface matching the C++ dynamixel_cpp module,
using the Python Dynamixel SDK which has better macOS support through PySerial.
"""

import math
import time
from typing import Dict, List, Tuple
import numpy as np
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite


# Control table addresses (Protocol 2.0)
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POS_VEL_CUR = 126  # Start of contiguous block

# Data lengths
LEN_TORQUE_ENABLE = 1
LEN_OPERATING_MODE = 1
LEN_PID_GAIN = 2
LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4
LEN_PRESENT_VELOCITY = 4
LEN_PRESENT_CURRENT = 2
LEN_PRESENT_POS_VEL_CUR = 10  # cur(2) + vel(4) + pos(4)

# Scales
DEFAULT_POS_SCALE = 2.0 * math.pi / 4096.0
DEFAULT_VEL_SCALE = 0.229 * 2.0 * math.pi / 60.0
PROTOCOL_VERSION = 2.0


def unsigned_to_signed(value: int, num_bytes: int) -> int:
    """Convert unsigned integer to signed based on byte length."""
    max_val = 2 ** (num_bytes * 8)
    if value >= max_val / 2:
        return value - max_val
    return value


class DynamixelController:
    """Single port controller for Dynamixel motors."""
    
    def __init__(
        self,
        port_name: str,
        motor_ids: List[int],
        kp: List[float],
        kd: List[float],
        zero_pos: List[float],
        control_mode: List[str],
        baudrate: int,
        return_delay: int,
    ):
        self.port_name = port_name
        self.motor_ids = motor_ids
        self.zero_pos = np.array(zero_pos, dtype=np.float32)
        self.baudrate = baudrate
        
        # Initialize handlers
        self.port_handler = PortHandler(port_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        
        # Will be set up during initialization
        self.sync_read = None
        self.sync_write = None
        self._connected = False
        
    def connect(self):
        """Open port and set baud rate."""
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.port_name}")
        
        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate to {self.baudrate}")
        
        self._connected = True
        print(f"Connected to {self.port_name} at {self.baudrate} baud")
        
    def initialize(self):
        """Initialize motors: set operating mode and enable torque."""
        if not self._connected:
            self.connect()
        
        # Set up sync read for bulk reading pos/vel/cur
        self.sync_read = GroupSyncRead(
            self.port_handler,
            self.packet_handler,
            ADDR_PRESENT_POS_VEL_CUR,
            LEN_PRESENT_POS_VEL_CUR
        )
        
        # Set up sync write for goal positions
        self.sync_write = GroupSyncWrite(
            self.port_handler,
            self.packet_handler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION
        )
        
        # Add parameters for each motor
        for motor_id in self.motor_ids:
            self.sync_read.addParam(motor_id)
        
        # Set operating mode to extended position (4) and enable torque
        for motor_id in self.motor_ids:
            # Disable torque first
            self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 0
            )
            
            # Set extended position mode
            self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, ADDR_OPERATING_MODE, 4
            )
            
            # Enable torque
            self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 1
            )
        
        print(f"Initialized {len(self.motor_ids)} motors: {self.motor_ids}")
    
    def get_state(self, retries: int = 0) -> Dict[str, List[float]]:
        """Read position, velocity, and current from all motors."""
        attempt = 0
        while attempt <= max(0, retries):
            # Perform sync read
            comm_result = self.sync_read.txRxPacket()
            
            if comm_result == 0:  # COMM_SUCCESS
                positions = []
                velocities = []
                currents = []
                
                for motor_id in self.motor_ids:
                    # Check if data is available
                    if not self.sync_read.isAvailable(motor_id, ADDR_PRESENT_POS_VEL_CUR, LEN_PRESENT_POS_VEL_CUR):
                        break
                    
                    # Read current (2 bytes)
                    cur_raw = self.sync_read.getData(motor_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT)
                    cur_signed = unsigned_to_signed(cur_raw, 2)
                    currents.append(cur_signed * 2.69)  # mA, from datasheet
                    
                    # Read velocity (4 bytes)
                    vel_raw = self.sync_read.getData(motor_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
                    vel_signed = unsigned_to_signed(vel_raw, 4)
                    velocities.append(vel_signed * DEFAULT_VEL_SCALE)
                    
                    # Read position (4 bytes)
                    pos_raw = self.sync_read.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    pos_signed = unsigned_to_signed(pos_raw, 4)
                    positions.append(pos_signed * DEFAULT_POS_SCALE)
                
                if len(positions) == len(self.motor_ids):
                    return {
                        "pos": positions,
                        "vel": velocities,
                        "cur": currents,
                    }
            
            attempt += 1
            if attempt <= retries:
                time.sleep(0.001)
        
        # Return zeros if all retries failed
        return {
            "pos": [0.0] * len(self.motor_ids),
            "vel": [0.0] * len(self.motor_ids),
            "cur": [0.0] * len(self.motor_ids),
        }
    
    def set_goal_pos(self, goal_positions: List[float]):
        """Set goal positions for all motors."""
        self.sync_write.clearParam()
        
        for motor_id, goal_pos in zip(self.motor_ids, goal_positions):
            # Convert to motor units
            goal_raw = int(goal_pos / DEFAULT_POS_SCALE)
            
            # Convert to 4-byte array (little endian)
            goal_bytes = [
                goal_raw & 0xFF,
                (goal_raw >> 8) & 0xFF,
                (goal_raw >> 16) & 0xFF,
                (goal_raw >> 24) & 0xFF,
            ]
            
            self.sync_write.addParam(motor_id, goal_bytes)
        
        self.sync_write.txPacket()
    
    def close(self):
        """Close the port."""
        if self._connected:
            self.port_handler.closePort()
            self._connected = False
            print(f"Closed port {self.port_name}")


def create_controllers(
    port_pattern: str,
    motor_ids: List[int],
    kp: List[float],
    kd: List[float],
    zero_pos: List[float],
    control_mode: List[str],
    baudrate: int,
    return_delay: int,
) -> List[DynamixelController]:
    """Create controller(s) matching the C++ interface.
    
    For macOS, port_pattern should be the full device path like:
    "/dev/tty.usbserial-FTAK8D39"
    
    For Linux, it would be like:
    "ttyUSB0" or "/dev/ttyUSB0"
    """
    # Ensure full path
    if not port_pattern.startswith("/dev/"):
        port_pattern = f"/dev/{port_pattern}"
    
    # Create single controller
    controller = DynamixelController(
        port_name=port_pattern,
        motor_ids=motor_ids,
        kp=kp,
        kd=kd,
        zero_pos=zero_pos,
        control_mode=control_mode,
        baudrate=baudrate,
        return_delay=return_delay,
    )
    
    return [controller]


def initialize(controllers: List[DynamixelController]):
    """Initialize all controllers."""
    for controller in controllers:
        controller.initialize()


def get_obs(controllers: List[DynamixelController], retries: int = 0) -> Dict[str, np.ndarray]:
    """Get observations from all controllers."""
    all_pos = []
    all_vel = []
    all_cur = []
    
    for controller in controllers:
        state = controller.get_state(retries)
        all_pos.extend(state["pos"])
        all_vel.extend(state["vel"])
        all_cur.extend(state["cur"])
    
    return {
        "pos": np.array(all_pos, dtype=np.float32),
        "vel": np.array(all_vel, dtype=np.float32),
        "cur": np.array(all_cur, dtype=np.float32),
    }


def set_goal_pos(controllers: List[DynamixelController], goal_positions: np.ndarray):
    """Set goal positions across all controllers."""
    offset = 0
    for controller in controllers:
        num_motors = len(controller.motor_ids)
        controller_goals = goal_positions[offset:offset + num_motors].tolist()
        controller.set_goal_pos(controller_goals)
        offset += num_motors


def close(controllers: List[DynamixelController]):
    """Close all controllers."""
    for controller in controllers:
        controller.close()
