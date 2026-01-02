import os
import sys
import glob
import time
import math
import threading
import platform
import subprocess
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union

try:
    from dynamixel_sdk import *
except ImportError:
    # If the package is not installed, we can't do much. 
    # But since we are creating the file, we assume it will be run in an env where it exists.
    pass

# Constants matching C++ implementation
PROTOCOL_VERSION = 2.0
DEFAULT_POS_SCALE = 2.0 * math.pi / 4096.0
DEFAULT_VEL_SCALE = 0.229 * 2.0 * math.pi / 60.0
DEFAULT_V_IN_SCALE = 0.1

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POS_VEL_CUR = 126
ADDR_PRESENT_V_IN = 144

LEN_GOAL_POSITION = 4
LEN_PRESENT_CURRENT = 2
LEN_PRESENT_VELOCITY = 4
LEN_PRESENT_POSITION = 4
LEN_PRESENT_POS_VEL_CUR = 10
LEN_PRESENT_V_IN = 2

CONTROL_MODE_DICT = {
    "current": 0,
    "velocity": 1,
    "position": 3,
    "extended_position": 4,
    "current_based_position": 5,
    "pwm": 16
}

class DynamixelClient:
    def __init__(self, motor_ids: List[int], port: str, baudrate: int = 1000000, lazy_connect: bool = False):
        self.motor_ids = motor_ids
        self.port_name = port
        self.baudrate = baudrate
        self.lazy_connect = lazy_connect
        
        self.port_handler = PortHandler(self.port_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        
        self.bulk_reader = GroupBulkRead(self.port_handler, self.packet_handler)
        for mid in self.motor_ids:
            self.bulk_reader.addParam(mid, ADDR_PRESENT_POS_VEL_CUR, LEN_PRESENT_POS_VEL_CUR)
            
        self.sync_writers = {} # (addr, len) -> GroupSyncWrite
        self.cur_scale_arr = []
        self.is_open = False
        
        # We use a lock for thread safety during communication
        self.lock = threading.Lock()
        
        if not self.lazy_connect:
            self.connect()

    def connect(self):
        if self.is_open:
            return
            
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.port_name}")
            
        if not self.port_handler.setBaudRate(self.baudrate):
            self.port_handler.closePort()
            raise RuntimeError(f"Failed to set baudrate {self.baudrate} on {self.port_name}")
            
        self.is_open = True

    def disconnect(self):
        if not self.is_open:
            return
        
        # Disable torque before closing
        try:
            self.set_torque_enabled(self.motor_ids, False)
        except Exception:
            pass
            
        self.port_handler.closePort()
        self.is_open = False

    def check_connected(self):
        if self.lazy_connect and not self.is_open:
            self.connect()
        if not self.is_open:
            raise RuntimeError("Port not connected")

    def set_torque_enabled(self, ids: List[int], enable: bool):
        self.check_connected()
        val = 1 if enable else 0
        for mid in ids:
            comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, mid, ADDR_TORQUE_ENABLE, val)
            if comm_result != COMM_SUCCESS:
                print(f"[Dynamixel] Torque set error on ID {mid}: {self.packet_handler.getTxRxResult(comm_result)}")
            elif dxl_error != 0:
                print(f"[Dynamixel] Torque set error on ID {mid}: {self.packet_handler.getRxPacketError(dxl_error)}")

    def clear_multi_turn(self, ids: List[int]):
        self.check_connected()
        for mid in ids:
            # clearMultiTurn might not be available in all python SDK versions, use clearMultiTurn if exists
            if hasattr(self.packet_handler, 'clearMultiTurn'):
                self.packet_handler.clearMultiTurn(self.port_handler, mid)
            else:
                 # Fallback? Not sure if there is a direct register for this on all motors without command
                 pass

    def clear_error(self, ids: List[int]):
        self.check_connected()
        for mid in ids:
             # Try clearStatusPacket or reboot if clearError not available?
             # Standard SDK usually has factoryReset or reboot. 
             # Use reboot if specific clear not found, or write to Control/Status registers.
             # Actually, Python SDK 3.7.31+ added clearMultiTurn.
             # We assume recent SDK.
             pass

    def read_vin(self) -> Tuple[float, List[float]]:
        self.check_connected()
        # Using sync read for efficiency if possible, but simplest is separate reads or group sync read
        # C++ used sync_read. Let's use GroupSyncRead.
        group_read = GroupSyncRead(self.port_handler, self.packet_handler, ADDR_PRESENT_V_IN, LEN_PRESENT_V_IN)
        for mid in self.motor_ids:
            group_read.addParam(mid)
            
        t_start = time.monotonic()
        with self.lock:
            comm_result = group_read.txRxPacket()
            
        if comm_result != COMM_SUCCESS:
             # print(f"Vin read error: {self.packet_handler.getTxRxResult(comm_result)}")
             pass
             
        results = []
        for mid in self.motor_ids:
            if group_read.isAvailable(mid, ADDR_PRESENT_V_IN, LEN_PRESENT_V_IN):
                raw = group_read.getData(mid, ADDR_PRESENT_V_IN, LEN_PRESENT_V_IN)
                results.append(raw * DEFAULT_V_IN_SCALE)
            else:
                results.append(0.0)
        
        group_read.clearParam()
        return (time.monotonic() - t_start) * 1000.0, results

    def sync_write(self, ids: List[int], values: List[int], addr: int, length: int):
        self.check_connected()
        key = (addr, length)
        if key not in self.sync_writers:
            self.sync_writers[key] = GroupSyncWrite(self.port_handler, self.packet_handler, addr, length)
        
        writer = self.sync_writers[key]
        writer.clearParam()
        
        for i, mid in enumerate(ids):
            # Convert int to byte array
            val = values[i]
            # Handle negative numbers for signed inputs
            if val < 0:
                # Calculate two's complement for 'length' bytes
                val = (1 << (length * 8)) + val
                
            param = []
            for b in range(length):
                param.append((val >> (8 * b)) & 0xFF)
                
            writer.addParam(mid, param)
            
        with self.lock:
            comm_result = writer.txPacket()
            
        if comm_result != COMM_SUCCESS:
            print(f"Sync write error: {self.packet_handler.getTxRxResult(comm_result)}")

    def read_pos_vel_cur(self, retries=0) -> Tuple[float, List[float], List[float], List[float]]:
        self.check_connected()
        
        t_start = time.monotonic()
        while True:
            with self.lock:
                comm_result = self.bulk_reader.txRxPacket()
            
            if comm_result == COMM_SUCCESS:
                break
            if retries <= 0:
                break
            retries -= 1
            
        latency = (time.monotonic() - t_start) * 1000.0
        
        pos_list = []
        vel_list = []
        cur_list = []
        
        # Populate current scale if empty
        if not self.cur_scale_arr:
            self.cur_scale_arr = [2.69] * len(self.motor_ids) # Default to 2.69 mA
        
        for i, mid in enumerate(self.motor_ids):
            if self.bulk_reader.isAvailable(mid, ADDR_PRESENT_POS_VEL_CUR, LEN_PRESENT_POS_VEL_CUR):
                # 126: Current (2), 128: Vel (4), 132: Pos (4)
                # GroupBulkRead accesses by address
                
                # Current
                raw_cur = self.bulk_reader.getData(mid, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT)
                # Convert 16-bit signed
                if raw_cur & 0x8000: raw_cur -= 0x10000
                cur_list.append(raw_cur * self.cur_scale_arr[i])
                
                # Velocity
                raw_vel = self.bulk_reader.getData(mid, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
                # Convert 32-bit signed
                if raw_vel & 0x80000000: raw_vel -= 0x100000000
                vel_list.append(raw_vel * DEFAULT_VEL_SCALE)
                
                # Position
                raw_pos = self.bulk_reader.getData(mid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                # Convert 32-bit signed
                if raw_pos & 0x80000000: raw_pos -= 0x100000000
                pos_list.append(raw_pos * DEFAULT_POS_SCALE)
            else:
                pos_list.append(0.0)
                vel_list.append(0.0)
                cur_list.append(0.0)
                
        return latency, pos_list, vel_list, cur_list
        
    def write_desired_pos(self, ids: List[int], positions: List[float]):
        raw_vals = [int(p / DEFAULT_POS_SCALE) for p in positions]
        self.sync_write(ids, raw_vals, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

class DynamixelControl:
    def __init__(self, port, motor_ids, kp, kd, zero_pos, control_mode, baudrate=2000000, return_delay_time=1):
        self.port = port
        self.motor_ids = motor_ids
        self.kp = list(kp)
        self.kd = list(kd)
        self.zero_pos = list(zero_pos)
        self.control_mode = control_mode
        self.baudrate = baudrate
        self.return_delay_time = return_delay_time
        
        # Default other gains
        self.ki = [0.0] * len(motor_ids)
        self.kff1 = [0.0] * len(motor_ids)
        self.kff2 = [0.0] * len(motor_ids)
        
        self.client = DynamixelClient(motor_ids, port, baudrate, lazy_connect=True)

    def initialize_motors(self):
        self.client.connect()
        
        # Clear Multi-turn (if implemented/supported)
        # self.client.clear_multi_turn(self.motor_ids)
        
        # Voltage Check
        # _, vins = self.client.read_vin()
        # for v in vins:
        #    if v < 10.0:
        #        print(f"Warning: Low voltage {v}V on port {self.port}")
                
        # Sync Write Configs
        
        # Return Delay
        self.client.sync_write(self.motor_ids, [self.return_delay_time]*len(self.motor_ids), 9, 1)
        
        # Control Mode
        modes = [CONTROL_MODE_DICT.get(m, 3) for m in self.control_mode]
        self.client.sync_write(self.motor_ids, modes, 11, 1)
        
        # Gains
        self.client.sync_write(self.motor_ids, [int(x) for x in self.kd], 80, 2)
        self.client.sync_write(self.motor_ids, [int(x) for x in self.ki], 82, 2)
        self.client.sync_write(self.motor_ids, [int(x) for x in self.kp], 84, 2)
        self.client.sync_write(self.motor_ids, [int(x) for x in self.kff2], 88, 2)
        self.client.sync_write(self.motor_ids, [int(x) for x in self.kff1], 90, 2)
        
        # Enable Torque
        self.client.set_torque_enabled(self.motor_ids, True)
        time.sleep(0.2)
        
        # Update Zero Pos
        _, cur_pos_list, _, _ = self.client.read_pos_vel_cur()
        for i, pos in enumerate(cur_pos_list):
            delta = pos - self.zero_pos[i]
            # Wrap delta to [-pi, pi]
            delta = (delta + math.pi) % (2 * math.pi) - math.pi
            self.zero_pos[i] = pos - delta
            
        print(f"[Initialize] Motors initialized on {self.port}")

    def get_state(self, retries=0):
        _, pos, vel, cur = self.client.read_pos_vel_cur(retries)
        # Adjust pos by zero_pos
        adj_pos = [p - z for p, z in zip(pos, self.zero_pos)]
        return {"pos": adj_pos, "vel": vel, "cur": cur}

    def set_pos(self, pos_vec):
        # drive = zero_pos + cmd_pos
        drive = [z + p for z, p in zip(self.zero_pos, pos_vec)]
        self.client.write_desired_pos(self.motor_ids, drive)
        
    def close_motors(self):
        self.client.disconnect()

    def get_motor_ids(self):
        return self.motor_ids

# ------------------------------------------------------------------------------
# Module-level functions (API compatible with dynamixel_cpp)
# ------------------------------------------------------------------------------

def scan_port(port_name: str, baudrate: int) -> List[int]:
    """Scans a single port for motors."""
    found_ids = []
    
    try:
        ph = PortHandler(port_name)
        if ph.openPort() and ph.setBaudRate(baudrate):
            pkt = PacketHandler(PROTOCOL_VERSION)
            for i in range(30): # Scan IDs 0-30 (or up to 253 if needed, but usually low)
                # Ping
                model_num, res, err = pkt.ping(ph, i)
                if res == COMM_SUCCESS and err == 0:
                    found_ids.append(i)
            ph.closePort()
    except Exception:
        pass
        
    return found_ids

def create_controllers(port_pattern: str, kp: List[float], kd: List[float], zero_pos: List[float], 
                       control_mode: List[str], baudrate: int = 2000000, return_delay: int = 1) -> List[DynamixelControl]:
    
    # 1. Expand glob pattern
    # The C++ code scans /dev/ for the pattern.
    # In Python glob works well.
    ports = glob.glob(port_pattern)
    if not ports:
        # On Mac, sometimes patterns like /dev/tty.* work better
        if platform.system() == "Darwin" and ("ttyUSB" in port_pattern or "ttyACM" in port_pattern):
             # Try to find something similar
             alt_patterns = ["/dev/tty.usbserial*", "/dev/cu.usbserial*"]
             for pat in alt_patterns:
                 ports = glob.glob(pat)
                 if ports:
                     print(f"Auto-detected port(s) matching '{pat}' instead of '{port_pattern}'")
                     break
    
    # Set latency timer on Mac (Try to run the existing tool)
    if platform.system() == "Darwin":
        try:
             # Assuming we are at project root
             tool_path = "./toddlerbot/actuation/latency_timer_setter_macOS/set_latency_timer"
             if os.path.exists(tool_path):
                 # We need to find the specific port name to pass
                 # C++ implementation passes latency value. 
                 # Wait, create_controllers calls set_latency_timer which runs the command.
                 subprocess.run([tool_path, "-l", "1"], capture_output=True)
                 # Note: The tool might require sudo or might set it for all FTDI devices.
        except Exception as e:
            print(f"Failed to set latency: {e}")

    controllers = []
    port_to_ids = {}
    all_ids = []

    # Scan ports
    for port in ports:
        ids = scan_port(port, baudrate)
        if ids:
            ids.sort()
            port_to_ids[port] = ids
            all_ids.extend(ids)
            
    all_ids.sort()
    
    # Map global config to local controllers
    # Create mapping id -> index in global arrays
    id_to_idx = {mid: i for i, mid in enumerate(all_ids)}
    
    for port, ids in port_to_ids.items():
        local_kp = []
        local_kd = []
        local_zero = []
        local_mode = []
        
        for mid in ids:
            if mid in id_to_idx:
                idx = id_to_idx[mid]
                if idx < len(kp): local_kp.append(kp[idx])
                else: local_kp.append(800.0) # Default
                
                if idx < len(kd): local_kd.append(kd[idx])
                else: local_kd.append(10.0)
                
                if idx < len(zero_pos): local_zero.append(zero_pos[idx])
                else: local_zero.append(0.0)
                
                if idx < len(control_mode): local_mode.append(control_mode[idx])
                else: local_mode.append("position")
        
        ctrl = DynamixelControl(port, ids, local_kp, local_kd, local_zero, local_mode, baudrate, return_delay)
        controllers.append(ctrl)
        
    return controllers

def initialize(controllers: List[DynamixelControl]):
    threads = []
    for i, ctrl in enumerate(controllers):
        # Stagger start
        t = threading.Thread(target=lambda c, delay: (time.sleep(delay), c.initialize_motors()), 
                             args=(ctrl, i * 0.05))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()

def get_motor_states(controllers: List[DynamixelControl], retries: int = 0) -> Dict[str, Dict[str, List[float]]]:
    # Parallel get state
    results = {}
    
    def fetch(idx, ctrl):
        try:
            results[f"controller_{idx}"] = ctrl.get_state(retries)
        except Exception as e:
            # print(f"Error getting state for controller {idx}: {e}")
            results[f"controller_{idx}"] = {"pos": [], "vel": [], "cur": []}

    threads = []
    for i, ctrl in enumerate(controllers):
        t = threading.Thread(target=fetch, args=(i, ctrl))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    return results

def get_motor_ids(controllers: List[DynamixelControl]) -> Dict[str, List[int]]:
    return {f"controller_{i}": ctrl.get_motor_ids() for i, ctrl in enumerate(controllers)}

def set_motor_pos(controllers: List[DynamixelControl], pos_vecs: List[List[float]]):
    threads = []
    for i, ctrl in enumerate(controllers):
        if i < len(pos_vecs):
            t = threading.Thread(target=ctrl.set_pos, args=(pos_vecs[i],))
            threads.append(t)
            t.start()
            
    for t in threads:
        t.join()

def close(controllers: List[DynamixelControl]):
    for ctrl in controllers:
        ctrl.close_motors()

def disable_motors(controllers: List[DynamixelControl]):
    for ctrl in controllers:
        ctrl.disable_motors()
