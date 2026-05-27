import argparse
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import joblib
import mujoco
import numpy as np
import numpy.typing as npt
import zmq
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import wandb
from toddlerbot.actuation import dynamixel_cpp
from toddlerbot.policies import BasePolicy
from toddlerbot.policies.run_policy import get_policy_class, plot_results
from toddlerbot.sim import BaseSim
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import Robot
from toddlerbot.sim.terrain.generate_terrain import create_terrain_spec
from toddlerbot.sim.terrain.get_depth import (
    add_data_collection_cameras,
    add_stereo_cameras_to_robot,
    cleanup_depth_capture,
    setup_depth_capture,
)
from toddlerbot.sim.terrain.get_elevation import add_elevation_map_markers
from toddlerbot.sim.terrain.get_skill import add_skill_map_markers, get_robot_skill
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.utils.misc_utils import dump_profiling_data

# === Visualization Config (Simulation Only) ===
VISUALIZE_ELEVATION_MAP = (
    False  # Whether to visualize elevation map spheres (press 4 to toggle)
)
ELEVATION_MAP_SIZE = 0.001  # Size of elevation map in meters (square)
ELEVATION_MAP_RESOLUTION = 0.0005  # Resolution in meters (grid spacing)
ELEVATION_MAP_FORWARD_OFFSET = 0.0  # Offset center forward in robot frame (meters)

VISUALIZE_SKILL_MAP = False  # Whether to visualize skill map bounds with colored spheres (press 4 to toggle)

# === Depth Map Capture Config ===
# Affects data collection, skill classifier testing, and depth display
# DEPTH_CAPTURE_INTERVAL_MIN = 0.5  # minimum seconds between depth captures
# DEPTH_CAPTURE_INTERVAL_MAX = 1.0  # maximum seconds between depth captures
# For skill classifier testing, similar frequency to foundation stereo
DEPTH_CAPTURE_INTERVAL_MIN = 0.3  # minimum seconds between depth captures
DEPTH_CAPTURE_INTERVAL_MAX = 0.33  # maximum seconds between depth captures

# === Depth Config (Simulation Only) ===
DISPLAY_DEPTH_REALTIME = False  # Whether to display depth in real-time OpenCV window
# When True: Shows depth continuously unless using skill classifier or data collection,
# in which case it shows depth at their intervals so you can see what they're processing

# === Control Config (Both Sim and Real) ===
# NOTE: For --collect-depth-data, increase this to capture more transition frames
# PREP_DURATION = 2.0  # Total preparation duration for policies in seconds
PREP_DURATION = 2.0  # Total preparation duration for policies in seconds
MIN_STANDBY_DURATION = 0.0  # Minimum time to hold final pose before policy starts
INITIAL_STAND_PREP_DURATION = (
    7.0  # Initial stand total preparation time (both sim and real)
)
INITIAL_STAND_MIN_STANDBY_DURATION = 5.0  # Initial stand minimum standby duration


# Policy configuration with skill and type fields
# NOTE: Key "0" is REQUIRED - it's used as the initial policy at startup (typically stand for stabilization)
# NOTE: "walk_in_place" is used for safe transitions from walking to other policies and not selectable from keyboard
# NOTE: All skill names must be unique across all configs - no duplicates allowed!
WALK_POLICY = "toddlerbot_2xc_walk_rsl_20251226_114612"
POLICY_CONFIGS = {
    "0": {  # REQUIRED: Initial policy - executed first when the system starts
        "policy": "stand",
        "skill": "stand",
        "type": "transition",
        "path": "",
    },
    "recovery": {
        "policy": "get_up_ground",
        "skill": "get_up_ground",
        "type": "transition",
        "path": "",
    },
    "walk_in_place": {
        "policy": "walk",
        "skill": "walk_in_place",
        "type": "locomotion",
        "checkpoint_path": WALK_POLICY,
        "command": "0 0.609 0 0 0 0 0 0",  # neck_pitch=0.85 rad
    },
    "w": {
        "policy": "walk",
        "skill": "walk",
        "type": "locomotion",  # Keep checking skill map
        "checkpoint_path": WALK_POLICY,
        "command": "0 0 0 0 0 0.05 -0.03 0",  # neck_pitch=0.85 rad
    },
    "a": {
        "policy": "walk",
        "skill": "turn_ccw",
        "type": "locomotion",
        "checkpoint_path": WALK_POLICY,
        "command": "0 0.609 0 0 0 0 0 0.8",  # neck_pitch=0.85 rad
    },
    "s": {
        "policy": "walk",
        "skill": "walk_back",
        "type": "locomotion",
        "checkpoint_path": WALK_POLICY,
        "command": "0 0.609 0 0 0 -0.03 -0.01 0",  # neck_pitch=0.85 rad
    },
    "q": {
        "policy": "walk",
        "skill": "walk_left_sideways",
        "type": "locomotion",  # Keep checking skill map
        "checkpoint_path": WALK_POLICY,
        "command": "0 0.609 0 0 0 0.05 0.07 0",  # neck_pitch=0.85 rad
    },
    "e": {
        "policy": "walk",
        "skill": "walk_right_sideways",
        "type": "locomotion",  # Keep checking skill map
        "checkpoint_path": WALK_POLICY,
        "command": "0 0.609 0 0 0 0.05 -0.1 0",  # neck_pitch=0.85 rad
    },
    "1": {
        "policy": "crawl_full",
        "skill": "get_down_chair",
        "type": "transition",
        "checkpoint_path": "toddlerbot_2xc_crawl_full_rsl_20260214_145000",
        "mode": "get_down_chair",
    },
    "2": {
        "policy": "crawl_full",
        "skill": "crawl_chair",
        "type": "locomotion",
        "checkpoint_path": "toddlerbot_2xc_crawl_full_rsl_20260214_145000",
        "mode": "crawl_chair",
    },
    "3": {
        "policy": "get_up",
        "skill": "get_up_chair",
        "type": "transition",
        "checkpoint_path": "toddlerbot_2xc_get_up_rsl_20260214_145222",
    },
    "4": {
        "policy": "climb_wall",
        "skill": "climb_wall",
        "type": "transition",
        "checkpoint_path": "toddlerbot_2xc_climb_wall_rsl_20250913_174540",
        "chain": {
            "get_up": None,
        },
    },
    "5": {
        "policy": "get_up",
        "skill": "get_up",
        "type": "transition",
        "checkpoint_path": "toddlerbot_2xc_get_up_rsl_20260214_145222",
        "command": "0 -0.327 0 0 0 0 0 0",  # neck_pitch=-0.2 rad
    },
    "6": {
        "policy": "climb_up_box",
        "skill": "climb_up_box",
        "type": "transition",
        "checkpoint_path": "toddlerbot_2xc_climb_up_box_rsl_20250907_201202",
    },
    "7": {
        "policy": "crawl_only",
        "skill": "crawl_box",
        "type": "locomotion",
        "checkpoint_path": "toddlerbot_2xc_crawl_only_rsl_20260214_145735",
        "command": "0 -0.327 0 0 0 0 0 0",  # neck_pitch=-0.2 rad
    },
    "8": {
        "policy": "crawl_rotate",
        "skill": "rotate_box",
        "type": "transition",
        "checkpoint_path": "toddlerbot_2xc_crawl_rotate_rsl_20251109_195839",
        "command": "0 -0.327 0 0 0 0 0 0",
        # NOTE: Use skill names in the chain
        # NOTE: Duration at the end of the chain is ignored and set to None
        "chain": {
            # "crawl_back_box": 2.0,  # 2 seconds max
            "climb_down_box_backward": None,  # Run until is_done
            # "walk_back": 1.0,
            "turn_ccw": 1.0,
            "stand": 1.0,  # NOTE: stand does not have an episode length, so setting the duration to None makes it run indefinitely.
        },
    },
    "9": {
        "policy": "climb_down_box",
        "skill": "climb_down_box_backward",
        "type": "transition",
        "checkpoint_path": "toddlerbot_2xc_climb_down_box_rsl_20250914_014343",
    },
    "z": {
        "policy": "crawl_up_stairs",
        "skill": "crawl_up_stairs",
        "type": "locomotion",
        "checkpoint_path": "toddlerbot_2xc_crawl_up_stairs_rsl_20260217_211836",
        "chain": {
            "crawl_down_stairs": None,
            "crawl_only": 17.0,
            # "crawl_only": 12.0,
            "get_up": None,
        },
    },
    "x": {
        "policy": "crawl_down_stairs",
        "skill": "crawl_down_stairs",
        "type": "locomotion",
        "checkpoint_path": "toddlerbot_2xc_crawl_down_stairs_rsl_20260215_094639",
    },
    "c": {
        "policy": "crawl_only",
        "skill": "crawl_only",
        "type": "locomotion",
        "checkpoint_path": "toddlerbot_2xc_crawl_only_rsl_20260214_145735",
        "command": "0 -0.327 0 0 0 0 0 0",  # neck_pitch=-0.2 rad
    },
}

# Create skill-to-key reverse lookup for skill map integration
SKILL_TO_KEY = {config["skill"]: key for key, config in POLICY_CONFIGS.items()}


class FoundationStereoSubscriber:
    """Non-blocking ZMQ subscriber for Foundation Stereo depth frames.

    Receives processed depth frames from run_foundation_stereo.py server.
    Designed to never block the control loop - always returns immediately.
    """

    def __init__(self, port: int = 5555, depth_shape: tuple = None):
        """Initialize ZMQ subscriber.

        Args:
            port: ZMQ port to subscribe to (must match server port)
            depth_shape: Expected depth frame shape after process_depth_map cropping (height, width)
                        Should match skill classifier's expected resolution after 10px left crop
        """
        if depth_shape is None:
            raise ValueError(
                "depth_shape must be provided (height, width) after 10px crop"
            )
        self.depth_shape = depth_shape
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        # Subscribe to all messages (empty filter)
        self.socket.subscribe(b"")

        # Set high water mark to 1 - only keep latest frame
        self.socket.setsockopt(zmq.RCVHWM, 1)

        # Enable conflate to keep only most recent message
        self.socket.setsockopt(zmq.CONFLATE, 1)

        # Connect to server
        self.socket.connect(f"tcp://localhost:{port}")

        print(f"Foundation Stereo subscriber connected to port {port}")

    def get_latest_frame(self) -> Optional[npt.NDArray[np.float32]]:
        """Get latest depth frame without blocking.

        Returns:
            Preprocessed depth frame as float32 in [0, 1.0] range (meters, ready for skill classifier),
            or None if no NEW frame since last call
            This matches simulation behavior where depth is only captured at intervals
        """
        try:
            # Non-blocking receive
            # If queue has a message -> returns it (new frame!)
            # If queue is empty -> raises zmq.Again (no new frame)
            data = self.socket.recv(zmq.NOBLOCK)

            # Deserialize: float32 preprocessed depth in [0, 1.0] range
            depth_preprocessed = np.frombuffer(data, dtype=np.float32).reshape(
                self.depth_shape
            )
            # print("\nReceived new depth frame from Foundation Stereo")
            return depth_preprocessed

        except zmq.Again:
            # No new frame available - return None to match simulation behavior
            # This prevents calling skill classifier multiple times with same frame
            # print("\nNo new depth frame available")
            return None

    def cleanup(self):
        """Cleanup ZMQ resources."""
        self.socket.close()
        self.context.term()


class PolicySwitchingState:
    """Thread-safe state for policy switching"""

    def __init__(self):
        self.lock = threading.Lock()
        self.current_policy = None
        self.switch_requested = False
        self.new_policy = None
        self.prep_mode = True  # Default to prep mode enabled
        self.policy_start_step = 0  # Step when current policy started
        self.policy_num_frames = None  # How many frames this policy should run
        self.paused = False  # Pause state
        self.pause_start_time = None  # When pause started
        self.total_pause_duration = 0.0  # Total time spent paused
        # NOTE: Walk safety transition - stores final target when transitioning through walk_in_place
        self.pending_target_policy = None  # Final target after walk_in_place transition
        self.in_safe_transition_from_walk = (
            False  # Flag indicating we're in walk safety transition
        )
        # Chain execution state
        self.chain = []  # List of (policy_name, duration) remaining in chain
        self.chain_duration = (
            None  # Duration for current chain step (None = until is_done)
        )

    def get_pausable_time(self, base_time):
        """Get time adjusted for pauses (time stops when paused)"""
        with self.lock:
            if self.paused and self.pause_start_time is not None:
                # Return the time when pause started, adjusted for previous pauses
                return self.pause_start_time - self.total_pause_duration
            else:
                # Return current time minus total pause duration
                return base_time - self.total_pause_duration


class RawInputManager:
    """Manages raw terminal input state"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        self.raw_mode = False

    def enable_raw_mode(self):
        if not self.raw_mode:
            tty.setraw(self.fd)
            self.raw_mode = True

    def disable_raw_mode(self):
        if self.raw_mode:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            self.raw_mode = False

    def get_char_nonblocking(self) -> Optional[str]:
        """Get character input without blocking, returns None if no input"""
        self.enable_raw_mode()

        # Check if input is available without blocking
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            ch = sys.stdin.read(1)
            # Flush any additional input to prevent multiple rapid presses
            while select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                sys.stdin.read(1)
            return ch
        return None

    def cleanup(self):
        self.disable_raw_mode()


# Global input manager
input_manager = RawInputManager()


def handle_keyboard_input(
    switch_state: PolicySwitchingState,
    loaded_policies: Dict[str, BasePolicy],
    stop_event: threading.Event,
    skill_map_mode: bool = False,
) -> bool:
    """Keyboard input handler with pause functionality"""
    char = input_manager.get_char_nonblocking()
    if char is None:
        return False

    # Prevent rapid switching by checking if a switch is already in progress
    with switch_state.lock:
        if switch_state.switch_requested:
            return False  # Ignore input if switch already requested

    if char == "\x1b":  # ESC key
        input_manager.disable_raw_mode()  # Restore terminal before quitting
        print("\nQuitting...")
        stop_event.set()
        cleanup_depth_capture(display_realtime=DISPLAY_DEPTH_REALTIME)
        return True
    elif char == " ":  # Spacebar to pause
        with switch_state.lock:
            if not switch_state.paused:
                # Starting pause
                switch_state.paused = True
                switch_state.pause_start_time = time.monotonic()
                status = "PAUSED"
            else:
                # Resuming from pause
                switch_state.paused = False
                if switch_state.pause_start_time is not None:
                    # Add the pause duration to total
                    switch_state.total_pause_duration += (
                        time.monotonic() - switch_state.pause_start_time
                    )
                    switch_state.pause_start_time = None
                status = "RESUMED"

            # Temporarily disable raw mode for clean output
            input_manager.disable_raw_mode()
            print(f"\r\n{status} - Press space again to toggle")
            sys.stdout.flush()
            input_manager.enable_raw_mode()
        return False
    elif char in POLICY_CONFIGS:
        # Block manual policy switching when skill map mode is enabled
        if skill_map_mode:
            # Temporarily disable raw mode for clean output
            input_manager.disable_raw_mode()
            print("\nManual policy switching disabled in skill-map mode.")
            print("The robot will switch policies automatically based on terrain.")
            print("Press 'esc' to quit or spacebar to pause/resume.")
            sys.stdout.flush()
            input_manager.enable_raw_mode()
            return False

        with switch_state.lock:
            # Policy availability check considering skill name
            policy_available = False
            if char in loaded_policies:
                policy_available = True
            elif char in POLICY_CONFIGS and "mode" in POLICY_CONFIGS[char]:
                # Handle mode-based policies: multiple keys can share the same policy instance
                # (e.g., keys "4" and "5" both use crawl_chair policy with different modes)
                target_policy_name = POLICY_CONFIGS[char]["policy"]
                for loaded_key in loaded_policies:
                    if POLICY_CONFIGS[loaded_key]["policy"] == target_policy_name:
                        policy_available = True
                        break

            if policy_available:
                switch_state.new_policy = char
                # Clear any active chain when user manually overrides
                switch_state.chain = []
                switch_state.chain_duration = None
            else:
                print(
                    f"\nPolicy {POLICY_CONFIGS[char]['policy']} not available (failed to load)"
                )
                return False
            switch_state.switch_requested = True

        # Temporarily disable raw mode for clean output
        input_manager.disable_raw_mode()
        policy_info = POLICY_CONFIGS[char]
        mode_str = f", mode: {policy_info['mode']}" if "mode" in policy_info else ""
        print(
            f"\nSwitching to policy: {policy_info['policy']} (skill: {policy_info['skill']}, type: {policy_info['type']}{mode_str})"
        )
        print("Press any key to switch policies or 'esc' to quit.")
        print("Available policies:")
        for key, config in POLICY_CONFIGS.items():
            print(f"  {key}: {config['policy']} (skill: {config['skill']})")
        print("  esc: quit")
        print("-" * 50)  # Visual separator
        sys.stdout.flush()
        input_manager.enable_raw_mode()
    return False


def load_policies(
    robot: Robot, init_motor_pos: npt.NDArray[np.float32], is_real_robot: bool = False
) -> Dict[str, BasePolicy]:
    """Policy loader supporting configuration system with instance reuse"""
    loaded_policies: Dict[str, BasePolicy] = {}
    # Track policy instances by their unique loading parameters
    # Key: (policy_name, checkpoint_path or motion_file_path), Value: policy instance
    policy_instances: Dict[tuple, BasePolicy] = {}

    print("\nLoading all policies...")

    for key, policy_cfg in tqdm(POLICY_CONFIGS.items(), desc="Loading policies"):
        try:
            # Handle different policy loading strategies
            if "checkpoint_path" in policy_cfg and policy_cfg["checkpoint_path"]:
                # Check if we can reuse an existing policy instance
                instance_key = (policy_cfg["policy"], policy_cfg["checkpoint_path"])
                if instance_key in policy_instances:
                    # Reuse existing instance
                    loaded_policies[key] = policy_instances[instance_key]
                    mode_str = (
                        f" (mode: {policy_cfg['mode']})" if "mode" in policy_cfg else ""
                    )
                    cmd_str = (
                        f" (command: {policy_cfg.get('command', 'default')})"
                        if "command" in policy_cfg
                        else ""
                    )
                    print(
                        f"  ✓ Reusing {policy_cfg['policy']} instance{mode_str}{cmd_str}"
                    )
                    continue

                # Load new policy from checkpoint
                kwargs = {
                    "name": policy_cfg["policy"],
                    "robot": robot,
                    "init_motor_pos": init_motor_pos.copy(),
                    "path": policy_cfg["checkpoint_path"],
                }

                # NOTE: For shared instances, using first encountered command as fixed_command
                if policy_cfg.get("command", None) is not None:
                    command = np.array(
                        policy_cfg["command"].split(" "), dtype=np.float32
                    )
                    kwargs["fixed_command"] = command

                policy_class = get_policy_class(policy_cfg["policy"])
                policy = policy_class(**kwargs)

                # Store in instances dict for potential reuse
                policy_instances[instance_key] = policy

                print(f"  ✓ Loaded {policy_cfg['policy']} policy (checkpoint)")

            elif "motion_file_path" in policy_cfg and policy_cfg["motion_file_path"]:
                # Load motion file policy (ReplayPolicy)
                motion_path = os.path.join("motion", policy_cfg["motion_file_path"])

                if not os.path.exists(motion_path):
                    print(f"  ✗ Motion file not found: {motion_path}")
                    continue

                # Use ReplayPolicy (for sim.step())
                policy_class = get_policy_class("replay")
                policy = policy_class(
                    name=policy_cfg["policy"],
                    robot=robot,
                    init_motor_pos=init_motor_pos.copy(),
                    path=motion_path,
                )
                print(
                    f"  ✓ Loaded {policy_cfg['policy']} policy (replay: {policy_cfg['motion_file_path']})"
                )

            else:
                # Load policies with no checkpoint/motion file needed
                # Examples: stand, rotate_box_backup, stair_crawl_backup, get_up_ground, etc.
                kwargs = {
                    "name": policy_cfg["policy"],
                    "robot": robot,
                    "init_motor_pos": init_motor_pos.copy(),
                }
                policy_class = get_policy_class(policy_cfg["policy"])
                policy = policy_class(**kwargs)
                print(f"  ✓ Loaded {policy_cfg['policy']} policy")

            # Set prep timing configurations
            if hasattr(policy, "prep_duration"):
                policy.prep_duration = PREP_DURATION
                # Add min_standby_duration for policies to use
                policy.min_standby_duration = MIN_STANDBY_DURATION

            loaded_policies[key] = policy

        except Exception as e:
            print(f"  ✗ Failed to load {policy_cfg['policy']} policy: {e}")
            continue

    print(f"Successfully loaded {len(loaded_policies)} policies.")
    return loaded_policies


def create_simulation(args: argparse.Namespace, robot: Robot):
    """Create simulation environment"""
    if args.sim == "mujoco":
        # TODO: Use hardcoded environment settings
        hardcoded_env_config = {
            "terrain": {
                "manual_map": [["testbed"]],  # Use testbed for multiple policies
                "tile_width": 1.0,
                "tile_length": 1.0,
                "resolution_per_meter": 32,
                "scene": "",  # Scene loading not supported yet - use terrain generation only
            }
        }

        terrain_cfg = hardcoded_env_config.get("terrain", {})

        print(f"Generating terrain: {terrain_cfg.get('manual_map', 'N/A')}")
        spec, _, safe_spawns, elevation_info, skill_map = create_terrain_spec(
            tile_width=terrain_cfg.get("tile_width", 1.0),
            tile_length=terrain_cfg.get("tile_length", 1.0),
            pixels_per_meter=terrain_cfg.get("resolution_per_meter", 32),
            terrain_map=terrain_cfg.get("manual_map", [["testbed"]]),
            robot_xml_path=os.path.join(
                "toddlerbot", "descriptions", robot.name, f"{robot.name}.xml"
            ),
        )

        # Add visualizations if enabled
        if VISUALIZE_ELEVATION_MAP:
            print("Adding elevation map visualization...")
            add_elevation_map_markers(
                spec, ELEVATION_MAP_SIZE, ELEVATION_MAP_RESOLUTION
            )
        else:
            print("Elevation map visualization disabled.")

        if VISUALIZE_SKILL_MAP and skill_map:
            print(f"Adding skill map visualization for {len(skill_map)} skills...")
            add_skill_map_markers(spec, skill_map, elevation_info)
        elif skill_map:
            print(
                f"Skill map visualization disabled ({len(skill_map)} skills available)."
            )
        else:
            print("No skill map available for visualization.")

        # TODO: Hardcoded initial robot position
        initial_robot_position = [0.0, 0.0, 0.315053]
        if safe_spawns:
            initial_robot_position = list(safe_spawns[6])
            initial_robot_position[2] += 0.315053
            initial_robot_position[0] += 0.0

        # Add multiple tracking cameras for better visualization
        camera_configs = [
            ("perspective", [0.7, -0.7, 0.7], [1, 1, 0, -1, 1, 3]),
            ("side", [0, -1.0, 0.6], [1, 0, 0, 0, 1, 3]),
            ("top", [0, 0, 1.0], [0, 1, 0, -1, 0, 0]),
            ("front", [1.0, 0, 0.6], [0, 1, 0, -1, 0, 3]),
        ]

        for name, pos, xyaxes in camera_configs:
            # Only add x,y position from initial_robot_position to maintain proper camera angles
            camera_pos = np.array(pos)
            camera_pos[0] += initial_robot_position[0]  # Add x offset
            camera_pos[1] += initial_robot_position[1]  # Add y offset
            # Keep original z position for proper camera perspective
            spec.worldbody.add_camera(
                name=name,
                pos=camera_pos,
                xyaxes=xyaxes,
                mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
            )

        # Always add stereo cameras first for consistent head_cam_left access
        print("Adding stereo cameras to robot...")
        add_stereo_cameras_to_robot(spec)

        # Add data collection camera grid if enabled
        data_collection_camera_names = []
        if args.collect_depth_data:
            print("Adding data collection camera grid...")
            data_collection_camera_names = add_data_collection_cameras(
                spec,
                grid_size_yz=(0.04, 0.02),  # 4cm x 2cm grid area
                grid_spacing=0.01,  # 1cm spacing between cameras
                x_positions=[0.0],  # Only at baseline center
                yaw_angles=[-0.05, 0.0, 0.05],  # Look left, forward, right
                roll_angles=[-0.1, 0.0, 0.1],  # Roll left, center, right
            )

        # Set robot position in spec
        torso = spec.worldbody.first_body()
        torso.pos = initial_robot_position

        model = spec.compile()

        sim = MuJoCoSim(
            robot,
            vis_type=args.vis,
            fixed_base=False,
            xml_path="",
            model=model,
        )

        # Set custom friction if specified
        if args.friction is not None:
            print(
                f"Setting ground friction to {args.friction} (default training range: [0.4, 1.0])"
            )
            # Set friction for all contact pairs
            # pair_friction[:, 0] = tangential friction 1
            # pair_friction[:, 1] = tangential friction 2
            # pair_friction[:, 2] = torsional friction (rolling)
            sim.model.pair_friction[:, :2] = args.friction

        # Pass elevation info to sim
        sim.elevation_info = elevation_info
        sim.visualize_elevation = VISUALIZE_ELEVATION_MAP
        sim.elevation_map_size = ELEVATION_MAP_SIZE
        sim.elevation_map_resolution = ELEVATION_MAP_RESOLUTION
        sim.elevation_map_forward_offset = ELEVATION_MAP_FORWARD_OFFSET

        # Setup depth rendering if any depth functionality is needed
        # TODO: (display and skill classifier always use head_cam_left, data collection uses data_collection_cameras)
        if DISPLAY_DEPTH_REALTIME or args.collect_depth_data or args.skill_classifier:
            print(
                f"Setting up depth capture (resolution: {tuple(args.depth_resolution)}, display: {DISPLAY_DEPTH_REALTIME}, intervals: {DEPTH_CAPTURE_INTERVAL_MIN}-{DEPTH_CAPTURE_INTERVAL_MAX}s)"
            )
            sim.depth_renderer, _, _ = setup_depth_capture(
                sim.model,
                resolution=tuple(args.depth_resolution),
                display_realtime=DISPLAY_DEPTH_REALTIME,
            )

        # Setup RGB rendering for data collection (640×480, matches real-world camera resolution)
        if args.collect_depth_data:
            print("Setting up RGB capture (640×480) for data collection")
            sim.rgb_renderer = mujoco.Renderer(sim.model, width=640, height=480)

        return sim, skill_map, data_collection_camera_names

    elif args.sim == "real":
        sim = RealWorld(robot)
        return sim, None, []
    else:
        raise ValueError(f"Unknown simulator: {args.sim}")


def run_multiple_policy(
    robot: Robot,
    sim: BaseSim,
    skill_map: Optional[Dict],
    data_collection_camera_names: List[str],
    skill_classifier: Optional[Any],
    depth_subscriber: Optional[FoundationStereoSubscriber],
    loaded_policies: Dict[str, BasePolicy],
    args: argparse.Namespace,
):
    """Main execution logic for multiple policy runner"""
    # Initialize policy switching state
    switch_state = PolicySwitchingState()
    switch_state.prep_mode = not args.no_prep
    stop_event = threading.Event()

    # Initialize logging if enabled
    exp_folder_path = None
    wandb_run = None
    obs_list = []
    control_inputs_list = []
    action_list = []
    loop_time_list = []

    if args.log:
        exp_name = f"{robot.name}_multiple_policy_{sim.name}"
        time_str = time.strftime("%Y%m%d_%H%M%S")
        exp_folder_path = f"results/{exp_name}_{time_str}"
        os.makedirs(exp_folder_path, exist_ok=True)
        print(f"Logging enabled. Data will be saved to: {exp_folder_path}")

        if args.record:
            project = "toddlerbot"
            entity = "toddlerbot"
            run_name = f"{exp_name}_{time_str}"
            wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                notes=args.note,
                job_type="eval",
            )
            print(f"Wandb recording enabled. Run: {run_name}")

            # Send signal to real robot if needed
            if "real" in sim.name:
                recorder_ip = "192.168.0.5"  # Default IP
                sender = ZMQNode(ip=recorder_ip)
                time.sleep(1)  # Allow subscriber to connect
                data = {
                    "signal": 1,
                    "project": project,
                    "entity": entity,
                    "run_name": run_name,
                    "run_id": wandb_run.id,
                    "note": args.note,
                }
                sender.send_msg(data)

    # Initialize initial policy with custom prep duration
    initial_policy = loaded_policies.get("0")  # Key "0" is used as the initial policy

    if not initial_policy:
        print("\n✗ ERROR: No policy found at key '0' in POLICY_CONFIGS.")
        print("Key '0' is required as the initial policy to start with.")
        print("Please ensure POLICY_CONFIGS has a valid policy at key '0'.")
        sys.exit(1)

    if hasattr(initial_policy, "prep_duration"):
        initial_policy.prep_duration = INITIAL_STAND_PREP_DURATION
        initial_policy.min_standby_duration = INITIAL_STAND_MIN_STANDBY_DURATION
        # Regenerate prep trajectory with new duration
        initial_policy.is_prepared = False

    current_policy = initial_policy  # Always start with initial policy (key "0")
    current_policy_key = "0"
    last_action = None
    policy_was_done = False  # Track if policy was done in previous step

    print("Multiple Policy Runner initialized.")
    print(f"\nRunning {POLICY_CONFIGS['0']['policy']} policy.")
    print("Press any key to switch policies or 'esc' to quit.")
    print("Available policies:")
    for key, config in POLICY_CONFIGS.items():
        print(f"  {key}: {config['policy']} (skill: {config['skill']})")
    print("  esc: quit")
    print("-" * 50)  # Visual separator

    try:
        step_count = 0
        policy_start_time = time.monotonic()  # Track when current policy started
        loop_start_time = time.monotonic()  # For timing synchronization
        global_heading_ref = None  # Global heading reference after initial stand

        # Initialize depth capture timing (start after initial stand prep completes)
        last_depth_capture_time = None
        next_capture_interval = np.random.uniform(
            DEPTH_CAPTURE_INTERVAL_MIN, DEPTH_CAPTURE_INTERVAL_MAX
        )

        # Initialize depth data collection buffer
        depth_data_buffer = []
        capture_counter = 0

        # Delay warning tracking
        delay_warning_count = 0
        delay_warning_interval = 500  # Warn every 500 steps if delayed

        # Fall recovery tracking
        fall_down_counter = 0
        recovery_policy = loaded_policies.get("recovery")

        while not stop_event.is_set():
            step_start_time = time.monotonic()  # For logging: start of entire loop
            obs = sim.get_observation()

            # Set obs.time using pausable clock (time stops when paused)
            current_time = time.monotonic()
            pausable_time = switch_state.get_pausable_time(current_time)
            obs.time = pausable_time - policy_start_time

            # Capture depth data for both skill classifier and data collection
            head_cam_depth_frame = None

            # Real robot: Get depth from Foundation Stereo via ZMQ (non-blocking)
            if (
                "real" in sim.name
                and args.skill_classifier
                and depth_subscriber
                and (step_count * 0.02) >= INITIAL_STAND_PREP_DURATION
            ):
                try:
                    # Non-blocking fetch - returns None if no new frame
                    input_manager.disable_raw_mode()  # Ensure raw mode for non-blocking input
                    head_cam_depth_frame = depth_subscriber.get_latest_frame()
                    input_manager.enable_raw_mode()  # Re-enable raw mode

                except Exception as e:
                    print(f"Error fetching depth from Foundation Stereo: {e}")
                    head_cam_depth_frame = None

            # Simulation: Render depth from cameras
            elif (
                "real" not in sim.name  # Simulation only
                and hasattr(sim, "depth_renderer")
                and sim.depth_renderer
                and (
                    args.skill_classifier
                    or args.collect_depth_data
                    or DISPLAY_DEPTH_REALTIME
                )
                and (step_count * 0.02) >= INITIAL_STAND_PREP_DURATION
            ):
                try:
                    # Determine capture timing based on intervals for skill classifier and data collection
                    should_capture_now = False

                    if args.skill_classifier or args.collect_depth_data:
                        # Use interval-based capture for skill classifier and data collection
                        if last_depth_capture_time is None:
                            should_capture_now = True
                            last_depth_capture_time = current_time
                        elif (
                            current_time - last_depth_capture_time
                            >= next_capture_interval
                        ):
                            should_capture_now = True
                            last_depth_capture_time = current_time
                            # Randomize next interval for data collection variety
                            next_capture_interval = np.random.uniform(
                                DEPTH_CAPTURE_INTERVAL_MIN, DEPTH_CAPTURE_INTERVAL_MAX
                            )
                    elif DISPLAY_DEPTH_REALTIME:
                        # Continuous capture for real-time display
                        should_capture_now = True

                    if should_capture_now:
                        # Get PREPROCESSED depth from get_depth_map (cropped, clipped)
                        from toddlerbot.sim.terrain.get_depth import (
                            create_depth_visualization,
                            get_depth_map,
                        )

                        head_cam_depth_frame = get_depth_map(
                            sim.data,
                            sim.depth_renderer,
                            camera_name="head_cam_left",
                            add_noise=False,
                            max_depth=1.0,
                        )

                        # Display depth visualization if enabled
                        if DISPLAY_DEPTH_REALTIME:
                            depth_vis = (255 * head_cam_depth_frame / 1.0).astype(
                                np.uint8
                            )
                            depth_display = create_depth_visualization(
                                head_cam_depth_frame, depth_vis, max_depth=1.0
                            )
                            cv2.imshow("Depth View", depth_display)
                            cv2.waitKey(1)

                        # Data collection uses multiple cameras for training data variety
                        # NOTE: Works with both manual and automatic (--skill-map) policy switching
                        # but --skill-map is recommended for consistent training data since manual
                        # switching timing can vary between runs
                        # NOTE: Data collection continues until manually stopped - press 'esc' when done
                        # NOTE: Skip data collection for initial stand policy (key "0") as it's just for stabilization
                        # NOTE: Skip data collection during sequential chain execution (predetermined sequence, not vision-based)
                        # BUT: DO capture for the initial policy that starts the chain (e.g., rotate_box_backup)
                        current_skill = POLICY_CONFIGS[current_policy_key].get(
                            "skill", ""
                        )
                        is_chain_step = (
                            switch_state.chain
                            and len(switch_state.chain) > 0
                            and not POLICY_CONFIGS[current_policy_key].get(
                                "chain", None
                            )
                        )
                        if (
                            args.collect_depth_data
                            and current_policy
                            and POLICY_CONFIGS[current_policy_key]["policy"] != "stand"
                            and not is_chain_step  # Skip only if currently executing a chain step
                        ):
                            # Determine if we should capture based on policy type
                            policy_config = POLICY_CONFIGS.get(current_policy_key, {})
                            policy_type = policy_config.get("type", "")
                            current_skill = policy_config.get("skill", "unknown")
                            policy_time = obs.time  # Time since policy started

                            should_capture_data = False
                            if policy_type == "transition":
                                # For transition policies: capture during prep phase only
                                should_capture_data = (
                                    policy_time < current_policy.prep_duration
                                )
                            elif policy_type == "locomotion":
                                # For locomotion policies: capture after prep phase (during actual locomotion)
                                should_capture_data = (
                                    policy_time >= current_policy.prep_duration
                                )

                            if should_capture_data:
                                # Global time for unique timestamps
                                global_time = time.monotonic() - loop_start_time

                                for cam_idx, camera_name in enumerate(
                                    data_collection_camera_names
                                ):
                                    # Capture RGB at 640×480
                                    sim.rgb_renderer.update_scene(
                                        sim.data, camera=camera_name
                                    )
                                    rgb_frame = (
                                        sim.rgb_renderer.render()
                                    )  # Returns (480, 640, 3) uint8 in RGB format

                                    # Capture RAW depth at args.depth_resolution (BEFORE process_depth_map)
                                    sim.depth_renderer.enable_depth_rendering()
                                    sim.depth_renderer.update_scene(
                                        sim.data, camera=camera_name
                                    )
                                    depth_raw = (
                                        sim.depth_renderer.render()
                                    )  # RAW float32 depth
                                    sim.depth_renderer.disable_depth_rendering()

                                    if rgb_frame is not None and depth_raw is not None:
                                        # Determine skill label
                                        if switch_state.in_safe_transition_from_walk:
                                            target_key = (
                                                switch_state.pending_target_policy
                                            )
                                            target_config = POLICY_CONFIGS.get(
                                                target_key, {}
                                            )
                                            skill_label = target_config.get(
                                                "skill", "unknown"
                                            )
                                        else:
                                            skill_label = current_skill

                                        # Add to buffer
                                        depth_data_buffer.append(
                                            {
                                                "rgb_frame": rgb_frame.copy(),
                                                "depth_raw": depth_raw.copy(),
                                                "timestamp": global_time,
                                                "camera_idx": cam_idx,
                                                "camera_name": camera_name,
                                                "skill": skill_label,
                                                "capture_id": capture_counter,
                                            }
                                        )

                                # Only increment counter after successful capture batch
                                if (
                                    len(
                                        [
                                            d
                                            for d in depth_data_buffer
                                            if d["capture_id"] == capture_counter
                                        ]
                                    )
                                    > 0
                                ):
                                    input_manager.disable_raw_mode()
                                    if switch_state.in_safe_transition_from_walk:
                                        print(
                                            f"\nCaptured RGB+depth batch #{capture_counter} during walk safety transition to skill '{skill_label}'"
                                        )
                                    else:
                                        print(
                                            f"Captured RGB+depth batch #{capture_counter} for skill '{skill_label}'"
                                        )
                                    input_manager.enable_raw_mode()
                                    capture_counter += 1

                except Exception as e:
                    print(f"Depth capture failed: {e}")
                    head_cam_depth_frame = None

            # Fall detection and recovery (uses both IMU orientation and depth)
            if recovery_policy is not None and "recovery" not in current_policy_key:
                body_forward_vec = obs.rot.apply(np.array([1, 0, 0], dtype=np.float32))
                body_left_vec = obs.rot.apply(np.array([0, 1, 0], dtype=np.float32))

                # Check orientation and depth to detect fall
                is_falling = False
                depth_max_thresh = 0.15  # Max depth threshold for face-down (meters)
                depth_min_thresh = 0.8  # Min depth threshold for on-back (meters)

                # Fall detection requires depth to distinguish falling from crawling
                if head_cam_depth_frame is not None:
                    if body_forward_vec[2] < -0.5:  # Tilted face down
                        # Check depth: if max depth is very small, camera sees floor up close
                        if np.max(head_cam_depth_frame) < depth_max_thresh:
                            fall_down_counter += 1
                            is_falling = True

                    elif body_forward_vec[2] > 0.5:  # Tilted on back
                        # Check depth: if min depth is very large, camera sees sky/ceiling
                        if np.min(head_cam_depth_frame) > depth_min_thresh:
                            fall_down_counter += 1
                            is_falling = True

                if not is_falling:
                    fall_down_counter = 0

                if fall_down_counter > 20:
                    # Determine get-up mode based on body orientation
                    if body_forward_vec[2] < -0.5:  # Face down (prone)
                        recovery_policy.motion_ref = recovery_policy.get_up_prone_ref
                        left_shoulder_pitch_pos = obs.motor_pos[
                            recovery_policy.left_shoulder_pitch_idx
                        ]
                        right_shoulder_pitch_pos = obs.motor_pos[
                            recovery_policy.right_shoulder_pitch_idx
                        ]
                        is_left_arm_down = (
                            left_shoulder_pitch_pos > -np.pi / 2
                            and left_shoulder_pitch_pos < np.pi / 2
                        )
                        is_right_arm_down = (
                            right_shoulder_pitch_pos > -np.pi / 2
                            and right_shoulder_pitch_pos < np.pi / 2
                        )
                        if is_left_arm_down and is_right_arm_down:
                            recovery_policy.get_up_mode = 0
                        elif not is_left_arm_down and not is_right_arm_down:
                            recovery_policy.get_up_mode = 1
                        elif is_left_arm_down and not is_right_arm_down:
                            recovery_policy.get_up_mode = 2
                        else:
                            recovery_policy.get_up_mode = 3

                    elif body_forward_vec[2] > 0.5:  # On back (supine)
                        recovery_policy.motion_ref = recovery_policy.get_up_roll_ref
                        if abs(body_left_vec[2]) < 0.1736:
                            recovery_policy.get_up_mode = 0
                        elif body_left_vec[2] >= 0.1736:
                            recovery_policy.get_up_mode = 1
                        else:
                            recovery_policy.get_up_mode = 2

                    fall_down_counter = 0
                    recovery_policy.is_prepared = False
                    recovery_policy.is_done = False

                    current_policy = recovery_policy
                    current_policy_key = "recovery"

                    if "real" in sim.name:
                        dynamixel_cpp.enable_motors(sim.controllers)

                    input_manager.disable_raw_mode()
                    print("\nFall detected! Switching to recovery policy...")
                    input_manager.enable_raw_mode()

            # TODO: Timing adjustment for simulation render mode (follows run_policy.py pattern)
            # Real robot and viewer mode run at real-time so no adjustment needed,
            # but render mode runs faster and needs timing correction for policies
            # This should behave differently depending on prep mode or not and for logging
            # if "real" not in sim.name and args.vis != "view":
            #     obs.time += time_until_next_step

            # Handle keyboard input for policy switching and pause
            # Block policy switching during initial stand preparation (allow quit and pause)
            if (step_count * 0.02) >= INITIAL_STAND_PREP_DURATION:
                if global_heading_ref is None:
                    input_manager.disable_raw_mode()
                    print(
                        "\nInitial stand preparation complete. Policy switching enabled."
                    )
                    input_manager.enable_raw_mode()
                    global_heading_ref = R.from_euler(
                        "z", obs.rot.as_euler("xyz")[2]
                    ).inv()

                if handle_keyboard_input(
                    switch_state, loaded_policies, stop_event, args.skill_map
                ):
                    break  # Exit if quit was requested
            else:
                # During initial stand, only allow quit (not policy switching)
                char = input_manager.get_char_nonblocking()
                if char == "\x1b":  # ESC key
                    input_manager.disable_raw_mode()
                    print("\nQuitting...")
                    stop_event.set()
                    cleanup_depth_capture(display_realtime=DISPLAY_DEPTH_REALTIME)
                    break

                if step_count % 50 == 0:
                    input_manager.disable_raw_mode()
                    print(
                        f"\rInitial stand in preparation... ({INITIAL_STAND_PREP_DURATION - (step_count * 0.02):.1f}s remaining)",
                        end="",
                    )
                    sys.stdout.flush()
                    input_manager.enable_raw_mode()

            # Handle skill map for automatic policy switching (simulation only)
            # Use simulation time based on step count (each step = 0.02s control frequency)
            if (
                args.skill_map
                and skill_map
                and current_policy
                and (step_count * 0.02) >= INITIAL_STAND_PREP_DURATION
                and not switch_state.in_safe_transition_from_walk  # NOTE: Skip during walk safety transition
                and not switch_state.chain  # NOTE: Skip during chain execution (predetermined sequence)
            ):
                with switch_state.lock:
                    if (
                        not switch_state.switch_requested
                    ):  # Only check if no switch pending
                        current_config = POLICY_CONFIGS[current_policy_key]

                        # Only check skill map if current policy is locomotion OR
                        # if current policy is transition and it's done OR
                        # if it's the initial stand policy (special case)
                        should_check_skill_map = False
                        if current_config["type"] == "locomotion":
                            # Locomotion policies: always check skill map
                            should_check_skill_map = True
                        elif current_config["type"] == "transition":
                            # Special case: initial stand policy can switch after prep time
                            if current_policy_key == "0":  # Initial stand
                                should_check_skill_map = True
                            # Other transition policies: only check if done
                            elif current_policy and hasattr(current_policy, "is_done"):
                                should_check_skill_map = current_policy.is_done

                        if should_check_skill_map:
                            robot_x, robot_y = obs.pos[0], obs.pos[1]
                            current_skill = current_config["skill"]

                            # Get required skill from position
                            required_skill = get_robot_skill(
                                robot_x, robot_y, skill_map, current_skill
                            )

                            # Default to walk if no specific skill found
                            if required_skill is None:
                                required_skill = "walk"

                            # Check if we need to switch skills
                            if required_skill != current_skill:
                                # Look up policy key for this skill
                                if required_skill in SKILL_TO_KEY:
                                    target_key = SKILL_TO_KEY[required_skill]

                                    # Switch immediately since we already checked if we should
                                    switch_state.new_policy = target_key
                                    switch_state.switch_requested = True
                                    input_manager.disable_raw_mode()
                                    print(
                                        f"\nSkill map triggered switch: {current_skill} → {required_skill}"
                                    )
                                    input_manager.enable_raw_mode()

            # Handle skill classifier for automatic policy switching (simulation only)
            # Same logic as skill map but using vision-based prediction
            # If depth map is None keeps running current policy
            if (
                args.skill_classifier
                and skill_classifier
                and head_cam_depth_frame is not None
                and current_policy
                and (step_count * 0.02) >= INITIAL_STAND_PREP_DURATION
                and not switch_state.in_safe_transition_from_walk  # NOTE: Skip during walk safety transition
                and not switch_state.chain  # NOTE: Skip during chain execution (predetermined sequence)
            ):
                with switch_state.lock:
                    if (
                        not switch_state.switch_requested
                    ):  # Only check if no switch pending
                        current_config = POLICY_CONFIGS[current_policy_key]

                        # Only check skill classifier if current policy is locomotion OR
                        # if current policy is transition and it's done OR
                        # if it's the initial stand policy (special case)
                        should_check_skill_classifier = False
                        if current_config["type"] == "locomotion":
                            # Locomotion policies: always check skill classifier
                            should_check_skill_classifier = True
                        elif current_config["type"] == "transition":
                            # Special case: initial stand policy can switch after prep time
                            if current_policy_key == "0":  # Initial stand
                                should_check_skill_classifier = True
                            # Other transition policies: only check if done
                            elif current_policy and hasattr(current_policy, "is_done"):
                                should_check_skill_classifier = current_policy.is_done

                        if should_check_skill_classifier:
                            current_skill = current_config["skill"]

                            input_manager.disable_raw_mode()
                            # Get required skill from vision
                            required_skill = skill_classifier.get_robot_skill(
                                head_cam_depth_frame, fallback_skill=current_skill
                            )
                            print(
                                f"Skill returned from skill classifier: {required_skill}"
                            )
                            input_manager.enable_raw_mode()

                            # Check if we need to switch skills
                            if required_skill and required_skill != current_skill:
                                # Look up policy key for this skill
                                if required_skill in SKILL_TO_KEY:
                                    target_key = SKILL_TO_KEY[required_skill]

                                    # Switch immediately since we already checked if we should
                                    switch_state.new_policy = target_key
                                    switch_state.switch_requested = True
                                    input_manager.disable_raw_mode()
                                    print(
                                        f"\nSkill classifier triggered switch: {current_skill} → {required_skill}"
                                    )
                                    input_manager.enable_raw_mode()
                                else:
                                    print(f"Unknown required skill: {required_skill}")

            # NOTE: Walk safety transition - check if we're in walk_in_place and ready to complete transition
            # This happens AFTER we've switched to walk_in_place and are waiting for standing position
            if (
                switch_state.in_safe_transition_from_walk
                and switch_state.pending_target_policy
            ):
                with switch_state.lock:
                    # Check if current policy is walk_in_place and is_standing
                    if (
                        current_policy_key == "walk_in_place"  # We're in walk_in_place
                        and hasattr(current_policy, "is_standing")
                        and current_policy.is_standing  # Robot is standing
                    ):
                        # NOTE: Robot is now safely standing, complete transition to final target
                        switch_state.new_policy = switch_state.pending_target_policy
                        switch_state.switch_requested = True
                        switch_state.pending_target_policy = None
                        switch_state.in_safe_transition_from_walk = False

                        input_manager.disable_raw_mode()
                        print(
                            f"\nStanding achieved, completing transition to {POLICY_CONFIGS[switch_state.new_policy]['policy']}"
                        )
                        input_manager.enable_raw_mode()
                    elif current_policy_key == "walk_in_place":
                        # NOTE: If not standing yet, override any skill switching requests
                        switch_state.switch_requested = False  # Wait longer
                        input_manager.disable_raw_mode()
                        print(
                            "\rWaiting to achieve safe pose from walk_in_place...",
                            end="",
                        )
                        sys.stdout.flush()
                        input_manager.enable_raw_mode()

            # Handle policy switching
            with switch_state.lock:
                if switch_state.switch_requested:
                    new_policy_key = switch_state.new_policy

                    if "real" in sim.name:
                        dynamixel_cpp.enable_motors(sim.controllers)

                    # NOTE: Walk safety transition - when switching FROM walk TO non-walk policies,
                    # we need to ensure robot is standing (both feet on ground) for safe transition.
                    # This prevents robot from falling when switching mid-stride.
                    if (
                        POLICY_CONFIGS[current_policy_key]["policy"] == "walk"
                        and POLICY_CONFIGS[new_policy_key]["policy"] != "walk"
                        and not switch_state.in_safe_transition_from_walk  # Not already in safe transition
                    ):
                        # Check if walk_in_place is available (key "walk_in_place")
                        if (
                            "walk_in_place" in loaded_policies
                            and current_policy_key != "walk_in_place"
                        ):
                            # NOTE: Initiate automatic two-step transition:
                            # 1. First switch to walk_in_place to stop forward motion
                            # 2. Once standing detected, switch to final target
                            switch_state.pending_target_policy = (
                                new_policy_key  # Save final target
                            )
                            switch_state.new_policy = (
                                "walk_in_place"  # Override to walk_in_place first
                            )
                            new_policy_key = switch_state.new_policy
                            switch_state.in_safe_transition_from_walk = True

                            input_manager.disable_raw_mode()
                            print(
                                f"Initiating safe transition: walk → walk_in_place → {POLICY_CONFIGS[switch_state.pending_target_policy]['skill']}"
                            )
                            input_manager.enable_raw_mode()
                            # Continue with switch to walk_in_place
                        # If walk_in_place not available, proceed directly (less safe but functional)

                    # Check if this is a mode change or a policy change
                    is_mode_change = False

                    # Handle mode-based policies (same policy instance, different modes)
                    if "mode" in POLICY_CONFIGS[new_policy_key]:
                        # Find the actual policy instance
                        target_policy_name = POLICY_CONFIGS[new_policy_key]["policy"]
                        for loaded_key, loaded_policy in loaded_policies.items():
                            if (
                                POLICY_CONFIGS[loaded_key]["policy"]
                                == target_policy_name
                            ):
                                # Check if we're already using this policy
                                if current_policy == loaded_policy:
                                    is_mode_change = True
                                current_policy = loaded_policy
                                # Set the mode if policy supports it
                                if hasattr(current_policy, "mode"):
                                    current_policy.mode = POLICY_CONFIGS[
                                        new_policy_key
                                    ]["mode"]
                                    current_policy.is_done = False
                                    # Handle heading for mode changes (crawl_chair specific)
                                    if hasattr(current_policy, "base_torso_rot_inv"):
                                        if (
                                            POLICY_CONFIGS[new_policy_key]["type"]
                                            == "locomotion"
                                        ):
                                            if global_heading_ref is not None:
                                                # Locomotion mode: use global heading
                                                current_policy.base_torso_rot_inv = (
                                                    global_heading_ref
                                                )
                                                # NOTE: Reset heading to current heading
                                                # current_policy.base_torso_rot_inv = R.from_euler(
                                                #     "z", obs.rot.as_euler("xyz")[2]
                                                # ).inv()
                                        else:
                                            # Transition mode: reset to use current heading
                                            current_policy.base_torso_rot_inv = None
                                break
                    else:
                        # Check if switching to a different config of the same policy instance such as walking with different commands
                        if current_policy == loaded_policies[new_policy_key]:
                            is_mode_change = (
                                True  # Treat command change like mode change
                            )
                        current_policy = loaded_policies[new_policy_key]

                    # Update command if specified in the new policy config
                    if "command" in POLICY_CONFIGS[new_policy_key] and hasattr(
                        current_policy, "fixed_command"
                    ):
                        command = np.array(
                            POLICY_CONFIGS[new_policy_key]["command"].split(" "),
                            dtype=np.float32,
                        )
                        current_policy.fixed_command = command
                        # NOTE: Set target torso yaw to the current heading to avoid sudden turns
                        if command[-1] != 0.0:
                            current_policy.target_torso_yaw = obs.rot.as_euler("xyz")[2]
                            # pass
                        # NOTE: Set target torso yaw always to 0 when command is straight and to pi when backward
                        if command[-1] == 0.0:
                            if command[-3] >= 0.0:
                                current_policy.target_torso_yaw = 0.0  # Face forward
                            else:
                                current_policy.target_torso_yaw = np.pi  # Face backward
                                # current_policy.target_torso_yaw = 0.0

                    current_policy_key = new_policy_key

                    # NOTE: Heading handling for walk, locomotion, and transition policies
                    # For walk and locomotion policies, we want to maintain a consistent global heading reference
                    # For transition policies, we want to reset heading reference to current orientation
                    # TODO: Add more transition policy trained with randomized initial poses to the global heading assignment
                    if hasattr(current_policy, "base_torso_rot_inv"):
                        if (
                            (
                                POLICY_CONFIGS[new_policy_key]["policy"] == "walk"
                                or POLICY_CONFIGS[new_policy_key]["type"]
                                == "locomotion"
                                or POLICY_CONFIGS[new_policy_key]["policy"]
                                == "climb_up_box"
                            )
                            and global_heading_ref is not None
                            # and current_policy.base_torso_rot_inv is None
                        ):
                            current_policy.base_torso_rot_inv = global_heading_ref
                            # current_policy.base_torso_rot_inv = R.from_euler(
                            #     "z", obs.rot.as_euler("xyz")[2]
                            # ).inv()
                        elif (
                            POLICY_CONFIGS[new_policy_key]["policy"] != "walk"
                            and POLICY_CONFIGS[new_policy_key]["type"] != "locomotion"
                        ):
                            current_policy.base_torso_rot_inv = None

                    # Temporarily disable raw mode for all output operations
                    input_manager.disable_raw_mode()

                    # Reset completion state
                    if hasattr(current_policy, "is_done"):
                        current_policy.is_done = False
                        if (
                            POLICY_CONFIGS[current_policy_key]["policy"]
                            == "get_up_ground"
                        ):
                            current_policy.is_done = True

                    # Reset policy state (skip for mode changes)
                    if not is_mode_change:
                        # Always reset the policy to start from beginning
                        if hasattr(current_policy, "reset"):
                            current_policy.reset()  # This may print "Resetting the X policy..."

                        # Reset qpos flag for rotate_box_backup, stair_crawl_backup, and replay policy
                        if hasattr(current_policy, "set_qpos"):
                            current_policy.set_qpos = True

                        # Always reset policy timing and pause tracking for new policy
                        policy_start_time = time.monotonic()
                        switch_state.total_pause_duration = 0.0
                        switch_state.pause_start_time = None

                        # Reset obs.time to 0 for the new policy (critical for ReplayPolicy and others)
                        obs.time = 0.0

                        # Handle prep mode separately (during prep mode policy switching can be still triggered)
                        if switch_state.prep_mode:
                            # Update init_motor_pos to current position for smooth transition
                            current_policy.init_motor_pos = obs.motor_pos.copy()
                            current_policy.is_prepared = False  # Force re-preparation
                        else:
                            # No prep mode - just ensure policy is marked as prepared (not resetting obs.time plays the main role in avoiding prep mode)
                            current_policy.is_prepared = True
                            current_policy.prep_duration = 0.0
                            # For ReplayPolicy, ensure time_start is set when skipping prep
                            if hasattr(current_policy, "time_start"):
                                current_policy.time_start = 0.0

                    switch_state.switch_requested = False
                    switch_state.policy_start_step = step_count
                    policy_was_done = False  # Reset completion tracking for new policy

                    # Initialize chain if the new policy has one (but preserve existing chain state)
                    if "chain" in POLICY_CONFIGS[new_policy_key]:
                        chain_config = POLICY_CONFIGS[new_policy_key]["chain"]
                        if chain_config:  # Only start chain if not empty
                            switch_state.chain = list(
                                chain_config.items()
                            )  # Convert dict to list of (name, duration)
                            switch_state.chain_duration = (
                                None  # Will be set when first chain step starts
                            )
                        else:
                            switch_state.chain = []
                            switch_state.chain_duration = None
                    elif not switch_state.chain:  # Only reset if no active chain
                        switch_state.chain = []
                        switch_state.chain_duration = None
                    # If we have an active chain, preserve chain_duration (we're in a chain step)

                    # Print appropriate message based on what changed
                    if is_mode_change:
                        if "mode" in POLICY_CONFIGS[new_policy_key]:
                            mode_str = f", mode: {POLICY_CONFIGS[new_policy_key].get('mode', 'default')}"
                            print(
                                f"Mode changed: {POLICY_CONFIGS[new_policy_key]['policy']}{mode_str}"
                            )
                        elif "command" in POLICY_CONFIGS[new_policy_key]:
                            # Command change for shared instance
                            print(
                                f"Command changed: {POLICY_CONFIGS[new_policy_key]['policy']} (key {new_policy_key})"
                            )
                    else:
                        print(
                            f"Switched to policy: {POLICY_CONFIGS[new_policy_key]['policy']}"
                        )
                    sys.stdout.flush()
                    input_manager.enable_raw_mode()

            # Get policy action or use frozen action when paused
            obs_time = time.monotonic()  # For logging: before policy execution
            with switch_state.lock:
                # TODO: Pause works during the prep phase, but switching policies while paused resets pause tracking and causes the prep phase to pass.
                if switch_state.paused and last_action is not None:
                    # Freeze robot actions: keep robot at last position while simulation continues
                    action = last_action
                    control_inputs = {}
                else:
                    # Get normal policy action
                    control_inputs, action = current_policy.step(obs, sim)
                    last_action = (
                        action.copy()
                    )  # Update last action for potential pause

            inference_time = time.monotonic()  # For logging: after policy step

            # Apply action to simulation
            sim.set_motor_target(action)
            set_action_time = (
                time.monotonic()
            )  # For logging: after setting motor targets

            # Step the simulation
            sim.step()

            sim_step_time = time.monotonic()  # For logging: after simulation step

            # Check if current chain step should timeout
            if (
                switch_state.chain_duration is not None
                and not switch_state.in_safe_transition_from_walk
            ):
                steps_elapsed = step_count - switch_state.policy_start_step
                time_elapsed = steps_elapsed * current_policy.control_dt
                if time_elapsed >= switch_state.chain_duration:
                    if hasattr(current_policy, "is_done"):
                        current_policy.is_done = True  # Force completion

            # Check for policy completion (print only once when it transitions from not done to done)
            policy_is_done = (
                hasattr(current_policy, "is_done") and current_policy.is_done
            )
            if policy_is_done and not policy_was_done:
                # Pop completed chain step FIRST (if in chain)
                # This keeps chain non-empty during execution for tracking (skill-map, data collection)
                if switch_state.chain:
                    # Check if current policy's skill matches the first chain item
                    # If it matches, we just completed that chain step, so pop it
                    # If it doesn't match, it's the initial policy that started the chain (not in chain list)
                    current_skill = POLICY_CONFIGS[current_policy_key].get("skill", "")
                    if current_skill == switch_state.chain[0][0]:
                        switch_state.chain.pop(0)  # Remove completed step

                policy_type = POLICY_CONFIGS[current_policy_key].get(
                    "type", "locomotion"
                )
                if policy_type == "transition":
                    input_manager.disable_raw_mode()
                    print(
                        f"Transition policy {POLICY_CONFIGS[current_policy_key]['policy']} completed"
                    )
                    input_manager.enable_raw_mode()
                elif (
                    policy_type == "locomotion"
                    and switch_state.chain_duration is not None
                ):
                    # Locomotion policy completed due to chain timeout
                    input_manager.disable_raw_mode()
                    print(
                        f"Locomotion policy {POLICY_CONFIGS[current_policy_key]['policy']} timed out"
                    )
                    input_manager.enable_raw_mode()

                # Check if there's a next chain step to execute
                if switch_state.chain:
                    next_skill_name, next_duration = switch_state.chain[
                        0
                    ]  # Peek at next step
                    # Add prep duration to chain duration if prep mode is enabled
                    if next_duration is not None and switch_state.prep_mode:
                        switch_state.chain_duration = next_duration + PREP_DURATION
                    else:
                        switch_state.chain_duration = next_duration

                    # Find the policy key for this skill name
                    next_policy_key = None
                    for key, config in POLICY_CONFIGS.items():
                        if config["skill"] == next_skill_name:  # Match by skill name
                            next_policy_key = key
                            break

                    if next_policy_key:
                        switch_state.new_policy = next_policy_key
                        switch_state.switch_requested = True
                        input_manager.disable_raw_mode()
                        duration_str = (
                            f"{next_duration}s" if next_duration else "until done"
                        )
                        print(f"Chain: Starting {next_skill_name} ({duration_str})")
                        input_manager.enable_raw_mode()
                    else:
                        input_manager.disable_raw_mode()
                        print(f"Chain: Skill '{next_skill_name}' not found, skipping")
                        input_manager.enable_raw_mode()

                # Handle recovery policy completion
                elif "recovery" in current_policy_key:
                    body_forward_vec = obs.rot.apply(
                        np.array([1, 0, 0], dtype=np.float32)
                    )
                    if abs(body_forward_vec[2]) < 0.5:  # Robot is upright
                        input_manager.disable_raw_mode()
                        print("Recovery complete. Querying skill classifier...")
                        input_manager.enable_raw_mode()
                        # Don't switch directly - let the skill classifier decide
                        # Set current_policy_key to stand temporarily so skill classifier runs
                        current_policy_key = "0"
                        current_policy = loaded_policies["0"]
                        current_config = POLICY_CONFIGS["0"]
                        # Reset the stand policy
                        if hasattr(current_policy, "reset"):
                            current_policy.reset()

            policy_was_done = policy_is_done

            # Timing synchronization - sleep to maintain correct loop rate
            step_count += 1
            step_end_time = time.monotonic()  # For logging: end of entire loop
            target_time = loop_start_time + step_count * current_policy.control_dt
            current_time = time.monotonic()
            time_until_next_step = target_time - current_time

            # Log all data if enabled
            if args.log:
                # Create copy of obs with global continuous time for logging
                obs_for_logging = obs.__class__(
                    time=obs_time - loop_start_time,  # Continuous global time
                    motor_pos=obs.motor_pos,
                    motor_vel=obs.motor_vel,
                    motor_acc=obs.motor_acc,
                    motor_tor=obs.motor_tor,
                    motor_cur=obs.motor_cur,
                    lin_vel=obs.lin_vel,
                    ang_vel=obs.ang_vel,
                    pos=obs.pos,
                    rot=obs.rot,
                )
                obs_list.append(obs_for_logging)
                control_inputs_list.append(control_inputs)
                action_list.append(action)

                # Add timing data
                loop_time_list.append(
                    [
                        step_start_time,
                        obs_time,
                        inference_time,
                        set_action_time,
                        sim_step_time,
                        step_end_time,
                        time_until_next_step,
                    ]
                )

            # Sleep if we're running faster than target rate (common in simulation)
            # TODO: Both Sim and Real work well when running faster than the desired control frequency, but when slower this script ignores actual simulation time and the real policy still outputs based only on its internal counter.
            if ("real" in sim.name or args.vis == "view") and time_until_next_step > 0:
                time.sleep(time_until_next_step)
            elif (
                time_until_next_step < -current_policy.control_dt * 0.5
            ):  # Significantly delayed
                delay_warning_count += 1
                if delay_warning_count % delay_warning_interval == 0:
                    delay_ms = abs(time_until_next_step) * 1000
                    input_manager.disable_raw_mode()
                    # TODO: Running with the MuJoCo Viewer often causes delays but this script uses wall-clock time (obs.time) instead of simulation time.
                    print(
                        f"Warning: Loop running {delay_ms:.1f}ms behind target rate (step {step_count})"
                    )
                    input_manager.enable_raw_mode()

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping...")
        stop_event.set()
    finally:
        # Save collected depth data if enabled
        if args.collect_depth_data and depth_data_buffer:
            print(f"\nSaving {len(depth_data_buffer)} RGB+depth samples...")

            # Create output directory with timestamp
            import json
            from datetime import datetime

            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(f"results/collect_sim_skill_data_{time_str}")
            rgb_dir = results_dir / "rgb"
            depth_resolution_str = (
                f"{args.depth_resolution[0]}x{args.depth_resolution[1]}"
            )
            depth_dir = results_dir / "depth" / depth_resolution_str

            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)

            # Save metadata
            metadata = {
                "dataset_type": "simulation",
                "collection_timestamp": time_str,
                "depth_resolution": args.depth_resolution,
                "rgb_resolution": [640, 480],
                "skill_classes": list(set(d["skill"] for d in depth_data_buffer)),
                "total_samples": len(depth_data_buffer),
            }
            with open(rgb_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Open timestamps file
            timestamps_file = open(rgb_dir / "timestamps.txt", "w")
            timestamps_file.write("# frame_index timestamp_s camera_idx skill_label\n")

            # Save all RGB and depth with progress bar
            for idx, data in enumerate(
                tqdm(depth_data_buffer, desc="Saving RGB+depth")
            ):
                # Save RGB as JPG: {index}_cam_{cam_idx}_rectified_{skill}.jpg
                rgb_filename = (
                    f"{idx:08d}_cam_{data['camera_idx']}_rectified_{data['skill']}.jpg"
                )
                cv2.imwrite(
                    str(rgb_dir / rgb_filename),
                    cv2.cvtColor(data["rgb_frame"], cv2.COLOR_RGB2BGR),
                )

                # Save raw depth as NPY: {index}_cam_{cam_idx}_depth_{skill}.npy
                depth_filename = (
                    f"{idx:08d}_cam_{data['camera_idx']}_depth_{data['skill']}.npy"
                )
                np.save(str(depth_dir / depth_filename), data["depth_raw"])

                # Write timestamp
                timestamps_file.write(
                    f"{idx} {data['timestamp']:.6f} {data['camera_idx']} {data['skill']}\n"
                )

            timestamps_file.close()

            print(f"✓ Data saved to {results_dir}")
            print(f"  RGB images: {rgb_dir}")
            print(f"  Depth maps: {depth_dir}")
            print(f"  Total samples: {len(depth_data_buffer)}")

            # Print summary of captured skills
            skill_counts = {}
            for data in depth_data_buffer:
                skill = data["skill"]
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
            print("  Samples per skill:")
            for skill, count in sorted(skill_counts.items()):
                print(f"    - {skill}: {count} samples")

        # Save logging data if enabled
        if args.log and exp_folder_path:
            print(f"Saving logged data to {exp_folder_path}...")
            log_data_dict = {
                "obs_list": obs_list,
                "control_inputs_list": control_inputs_list,
                "action_list": action_list,
                "loop_time_list": loop_time_list,
            }
            log_data_path = os.path.join(exp_folder_path, "log_data.lz4")
            joblib.dump(log_data_dict, log_data_path, compress="lz4")

            # Save profiling data
            prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
            dump_profiling_data(prof_path)

            # Upload to wandb if recording
            if args.record and wandb_run:
                print("Uploading to wandb...")
                artifact = wandb.Artifact(name="rollout", type="rollout", metadata={})
                artifact.add_dir(exp_folder_path)
                wandb.log_artifact(
                    artifact, aliases=["latest", os.path.basename(exp_folder_path)]
                )

                # Send stop signal to real robot if needed
                if "real" in sim.name:
                    sender.send_msg({"signal": 0})

            # Generate plots when logging (always save plots, --plot controls display)
            if len(obs_list) > 0:
                print("Generating plots...")
                plot_results(
                    robot=robot,
                    loop_time_list=loop_time_list,
                    obs_list=obs_list,
                    control_inputs_list=control_inputs_list,
                    action_list=action_list,
                    exp_folder_path=exp_folder_path,
                    blocking=args.plot,  # Only display/show plots if --plot is used
                )

        # Cleanup
        input_manager.cleanup()
        if hasattr(sim, "depth_renderer") and sim.depth_renderer:
            cleanup_depth_capture(display_realtime=DISPLAY_DEPTH_REALTIME)


def main():
    """Main function for multiple policy runner"""
    parser = argparse.ArgumentParser(description="Multiple Policy Runner.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xc",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="mujoco",
        help="The name of the simulator to use.",
        choices=["mujoco", "real"],
    )
    parser.add_argument(
        "--vis",
        type=str,
        default="view",
        help="The visualization type.",
        choices=["render", "view", "none"],
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="Enable experiment logging and data collection.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Enable blocking plot display (requires --log).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Record the experiment with wandb (requires --log).",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="A note to add to the wandb run.",
    )
    parser.add_argument(
        "--skill-map",
        action="store_true",
        default=False,
        help="Enable automatic policy switching based on robot position and terrain skill map.",
    )
    parser.add_argument(
        "--skill-classifier",
        type=str,
        default=None,
        help="Enable vision-based skill classification. Provide path to trained skill classifier model.",
    )
    parser.add_argument(
        "--depth-resolution",
        type=int,
        nargs=2,
        default=[256, 192],
        metavar=("WIDTH", "HEIGHT"),
        help="Depth capture resolution (width, height). Overridden by skill classifier's expected resolution if --skill-classifier is provided.",
    )
    parser.add_argument(
        "--collect-depth-data",
        action="store_true",
        default=False,
        help="Enable depth data collection for training skill classifier (simulation only).",
    )
    parser.add_argument(
        "--no-prep",
        action="store_true",
        default=False,
        help="Disable preparation mode between policy switches (prep mode enabled by default).",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=None,
        help="Ground friction coefficient (default: XML value 1, training uses [0.4, 1.0]). Lower=slippery.",
    )

    args = parser.parse_args()

    # Validate argument dependencies
    if args.record and not args.log:
        parser.error("--record requires --log")
    if args.plot and not args.log:
        print("Warning: --plot has no effect without --log (no data to plot)")
    if args.skill_classifier and args.skill_map:
        parser.error(
            "--skill-classifier and --skill-map cannot be used together (choose one automatic switching method)"
        )
    if args.sim == "real" and args.skill_map:
        parser.error("--skill-map is not supported with real robot (simulation only)")
    if args.collect_depth_data and args.no_prep:
        parser.error(
            "--collect-depth-data requires prep mode for transition policies (cannot use with --no-prep)"
        )
    if args.collect_depth_data and args.sim == "real":
        parser.error(
            "--collect-depth-data is only supported with simulation (use --sim mujoco)"
        )

    print("=" * 60)
    print("MULTIPLE POLICY RUNNER")
    print("=" * 60)
    print(f"Robot: {args.robot}")
    print(f"Simulator: {args.sim}")
    print(f"Logging: {'ON' if args.log else 'OFF'}")
    print(f"Recording: {'ON' if args.record else 'OFF'}")
    print(f"Depth Data Collection: {'ON' if args.collect_depth_data else 'OFF'}")
    if args.collect_depth_data:
        print("  → Press 'esc' to stop and save collected data")
    print(f"Depth Display Real-time: {'ON' if DISPLAY_DEPTH_REALTIME else 'OFF'}")
    print(f"Skill Classifier: {'ON' if args.skill_classifier else 'OFF'}")
    print(f"Skill Map: {'ON' if args.skill_map else 'OFF'}")
    print(f"Prep Mode: {'OFF' if args.no_prep else 'ON'}")
    print("=" * 60)

    robot = Robot(args.robot)

    # Override depth resolution if skill classifier is provided (must happen before create_simulation)
    if args.skill_classifier:
        from toddlerbot.skill_classifier.inference.predictor import SkillPredictor
        from toddlerbot.skill_classifier.inference.skill_mapper import (
            _resolve_model_path,
        )

        try:
            model_path = _resolve_model_path(args.skill_classifier)
            temp_predictor = SkillPredictor(model_path=model_path, device="cpu")
            expected_res = temp_predictor.expected_depth_resolution
            if tuple(args.depth_resolution) != expected_res:
                print(
                    f"Overriding depth resolution from {args.depth_resolution} to {expected_res} (from ONNX model)"
                )
                args.depth_resolution = list(expected_res)
            del temp_predictor  # Clean up
        except Exception as e:
            print(
                f"Warning: Could not extract depth resolution from skill classifier: {e}"
            )
            print(f"Using default depth resolution: {args.depth_resolution}")

    # Create simulation environment
    sim, skill_map, data_collection_camera_names = create_simulation(args, robot)

    # Initialize vision skill classifier if enabled
    skill_classifier = None
    depth_subscriber = None
    if args.skill_classifier:
        print(
            f"Initializing vision skill classifier with model: {args.skill_classifier}"
        )
        from toddlerbot.skill_classifier.inference.skill_mapper import (
            initialize_vision_skill_map,
        )

        try:
            skill_classifier = initialize_vision_skill_map(args.skill_classifier)
            print("Vision skill classifier initialized successfully")
        except Exception as e:
            print(f"Failed to initialize vision skill classifier: {e}")
            print("Falling back to regular operation")

        # Initialize Foundation Stereo ZMQ subscriber for real robot
        if "real" in args.sim:
            try:
                print("Initializing Foundation Stereo ZMQ subscriber...")
                # Calculate expected depth shape: (height, width_after_crop)
                # args.depth_resolution is [width, height], crop 10px from left
                depth_shape = (args.depth_resolution[1], args.depth_resolution[0] - 10)
                depth_subscriber = FoundationStereoSubscriber(
                    port=5555, depth_shape=depth_shape
                )
                print("Foundation Stereo subscriber initialized successfully")
                print(
                    f"Expecting depth shape: {depth_shape} (height, width after 10px crop)"
                )
                print("Note: Make sure run_foundation_stereo.py server is running!")
            except Exception as e:
                print(f"Failed to initialize Foundation Stereo subscriber: {e}")
                print("Skill classifier will not work without depth frames")

    # Get initial motor positions (use retries for real world)
    if "real" in sim.name:
        init_motor_pos = sim.get_observation(retries=-1).motor_pos
    else:
        init_motor_pos = sim.get_observation().motor_pos

    # Load all policies
    is_real_robot = args.sim == "real"
    loaded_policies = load_policies(robot, init_motor_pos, is_real_robot)

    # Run the main control loop
    try:
        run_multiple_policy(
            robot=robot,
            sim=sim,
            skill_map=skill_map,
            data_collection_camera_names=data_collection_camera_names,
            skill_classifier=skill_classifier,
            depth_subscriber=depth_subscriber,
            loaded_policies=loaded_policies,
            args=args,
        )
    finally:
        # Cleanup
        if depth_subscriber:
            depth_subscriber.cleanup()
        sim.close()


if __name__ == "__main__":
    main()
