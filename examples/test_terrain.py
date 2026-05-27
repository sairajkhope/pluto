"""
test_terrain.py

Test script for Toddlerbot simulation with procedural terrain and elevation mapping.

This script:
- Defines terrain grid configuration
- Generates terrain with elevation info using create_terrain_spec()
- Provides real-time elevation mapping and visualization
- Interactive robot control with keyboard input
"""

import os

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from toddlerbot.sim.terrain.generate_terrain import create_terrain_spec
from toddlerbot.sim.terrain.get_depth import (
    add_stereo_cameras_to_robot,
    capture_depth_frame,
    cleanup_depth_capture,
    get_fovy,
    setup_depth_capture,
)
from toddlerbot.sim.terrain.get_elevation import (
    add_elevation_map_markers,
    get_elevation_map,
    get_robot_ground_height,
    update_elevation_visualization,
)

# === Terrain config ===
TILE_WIDTH = 4.0
TILE_LENGTH = 4.0
PIXELS_PER_METER = 32
TIMESTEP = 0.004
TERRAIN_MAP = [
    ["boxes", "stairs", "bumps"],
    ["flat", "slope", "rough"],
]

# === Elevation map config ===
ELEVATION_MAP_SIZE = 0.5  # Size of elevation map in meters (square)
ELEVATION_MAP_RESOLUTION = 0.02  # Resolution in meters (grid spacing)
ELEVATION_MAP_FORWARD_OFFSET = 0.25  # Offset center forward in robot frame (meters)
VISUALIZE_ELEVATION_MAP = (
    False  # Whether to visualize elevation map in simulation by default
)

# === Depth capture config ===
# os.environ["MUJOCO_GL"] = "glfw"
CAPTURE_DEPTH_VIDEO = True  # Whether to capture depth video from head camera
DISPLAY_DEPTH_REALTIME = True  # Whether to display depth in real-time OpenCV window
DEPTH_MAX_RANGE = 1.0  # Maximum depth range in meters
DEPTH_RESOLUTION = (128, 96)  # Resolution for depth video
DEPTH_VIDEO_FPS = 30  # FPS for depth video output


class ToddlerbotSimulator:
    """Interactive ToddlerBot simulator with keyboard controls."""

    def __init__(
        self,
        mj_model,
        elevation_info=None,
        tile_width=4.0,
        tile_length=4.0,
        capture_depth=False,
        display_depth=False,
        depth_resolution=(128, 96),
    ):
        self.model = mj_model
        self.data = mujoco.MjData(self.model)

        # Reset to home keyframe (which now has neck pitch at -0.65 rad)
        home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key_id)

        self.tile_width = tile_width
        self.tile_length = tile_length

        # Depth capture setup
        self.capture_depth = capture_depth
        self.display_depth = display_depth
        self.depth_resolution = depth_resolution
        self.depth_renderer = None
        self.depth_writer = None
        if self.capture_depth or self.display_depth:
            # Create timestamped output directory for depth video under results
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                os.path.dirname(__file__), "..", "results", f"test_terrain_{timestamp}"
            )
            self.depth_renderer, self.depth_writer, _ = setup_depth_capture(
                model=self.model,
                resolution=self.depth_resolution,
                capture_video=self.capture_depth,
                display_realtime=self.display_depth,
                video_fps=DEPTH_VIDEO_FPS,
                output_dir=output_dir,
            )

        self.MOVE_DISTANCE = 0.05
        self.MOVE_DURATION = 0.05
        self.MOVE_STEPS = int(self.MOVE_DURATION / self.model.opt.timestep)
        self.VELOCITY = self.MOVE_DISTANCE / self.MOVE_DURATION
        self.motion_queue = []

        self.JOINT_MOVE_AMOUNT = 0.4
        self.JOINT_MOVE_DURATION = 0.05
        self.JOINT_MOVE_STEPS = int(self.JOINT_MOVE_DURATION / self.model.opt.timestep)
        self.joint_motion_queue = []

        self.ROTATION_ANGLE = np.deg2rad(10)

        self.KEY_MOVES = {
            glfw.KEY_UP: np.array([1, 0, 0]),
            glfw.KEY_DOWN: np.array([-1, 0, 0]),
            glfw.KEY_LEFT: "turn_left",
            glfw.KEY_RIGHT: "turn_right",
            glfw.KEY_E: np.array([0, 0, 1]),
            glfw.KEY_R: np.array([0, 0, -1]),
        }

        self._init_joint_indices()

        # Store elevation info for terrain queries
        self.elevation_info = elevation_info

    def _init_joint_indices(self):
        """Initialize joint indices for head movement control."""
        yaw_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw_drive"
        )
        pitch_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch_act"
        )
        yaw_qpos_adr = self.model.jnt_qposadr[yaw_id]
        pitch_qpos_adr = self.model.jnt_qposadr[pitch_id]

        self.JOINT_KEYS = {
            glfw.KEY_B: (yaw_qpos_adr, +1),
            glfw.KEY_C: (yaw_qpos_adr, -1),
            glfw.KEY_F: (pitch_qpos_adr, +1),
            glfw.KEY_V: (pitch_qpos_adr, -1),
        }

    def key_callback(self, key):
        """Handle keyboard input for robot movement and head control."""
        if key in self.KEY_MOVES:
            move = self.KEY_MOVES[key]
            if isinstance(move, str):
                quat = self.data.qpos[3:7]
                rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
                delta_rot = R.from_euler(
                    "z",
                    self.ROTATION_ANGLE
                    if move == "turn_left"
                    else -self.ROTATION_ANGLE,
                )
                new_quat = (delta_rot * rot).as_quat()
                self.data.qpos[3:7] = [
                    new_quat[3],
                    new_quat[0],
                    new_quat[1],
                    new_quat[2],
                ]
            else:
                self.motion_queue.append((move, self.MOVE_STEPS))

        elif key in self.JOINT_KEYS:
            joint_idx, direction = self.JOINT_KEYS[key]
            delta = direction * self.JOINT_MOVE_AMOUNT
            self.joint_motion_queue.append((joint_idx, delta, self.JOINT_MOVE_STEPS))

    def _apply_base_motion(self, motion):
        direction, steps_left = motion
        quat = self.data.qpos[3:7]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        world_dir = rot.apply(direction)
        step = world_dir * self.VELOCITY * self.model.opt.timestep
        self.data.qpos[:3] += step

        return (direction, steps_left - 1) if steps_left > 1 else None

    def _apply_joint_motion(self, motion):
        idx, delta_per_step, steps_left = motion
        self.data.qpos[idx] += delta_per_step
        return (idx, delta_per_step, steps_left - 1) if steps_left > 1 else None

    def run(self):
        """Start the interactive simulation with keyboard controls."""
        print("Arrow keys to move, E/R up/down, F/C/V/B for head, ESC to quit")
        print("Press '4' to toggle elevation map visualization")
        current_motion = None
        current_joint_motion = None
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.key_callback
        ) as viewer:
            while viewer.is_running():
                if current_motion is None and self.motion_queue:
                    current_motion = self.motion_queue.pop(0)
                if current_motion:
                    current_motion = self._apply_base_motion(current_motion)

                if current_joint_motion is None and self.joint_motion_queue:
                    joint_idx, total_delta, steps = self.joint_motion_queue.pop(0)
                    delta_per_step = total_delta / steps
                    current_joint_motion = (joint_idx, delta_per_step, steps)
                if current_joint_motion:
                    current_joint_motion = self._apply_joint_motion(
                        current_joint_motion
                    )

                mujoco.mj_forward(self.model, self.data)

                robot_x, robot_y = self.data.qpos[:2]
                robot_ground_height = get_robot_ground_height(
                    robot_x, robot_y, self.elevation_info
                )

                global_map, relative_map, x_coords, y_coords = get_elevation_map(
                    self.data,
                    self.elevation_info,
                    map_size=ELEVATION_MAP_SIZE,
                    map_resolution=ELEVATION_MAP_RESOLUTION,
                    forward_offset=ELEVATION_MAP_FORWARD_OFFSET,
                )

                print(
                    f"\nRobot: ({robot_x:.3f}, {robot_y:.3f}) | Height: {robot_ground_height:.3f}m",
                )

                # Update elevation visualization spheres
                if VISUALIZE_ELEVATION_MAP:
                    update_elevation_visualization(
                        self.data, self.model, global_map, x_coords, y_coords
                    )

                # Capture depth frame if enabled
                if self.capture_depth or self.display_depth:
                    capture_depth_frame(
                        data=self.data,
                        renderer=self.depth_renderer,
                        video_writer=self.depth_writer if self.capture_depth else None,
                        display_realtime=self.display_depth,
                        camera_name="head_cam_left",
                        max_depth=DEPTH_MAX_RANGE,
                    )

                viewer.sync()

        # Cleanup depth capture when simulation ends
        cleanup_depth_capture(
            video_writer=self.depth_writer, display_realtime=self.display_depth
        )


def main():
    """Set up terrain and run interactive ToddlerBot simulation."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    robot_path = os.path.join(
        this_dir, "../toddlerbot/descriptions/toddlerbot_2xc/toddlerbot_2xc.xml"
    )

    # Generate terrain + robot
    spec, _, _, elevation_info, _ = create_terrain_spec(
        tile_width=TILE_WIDTH,
        tile_length=TILE_LENGTH,
        terrain_map=TERRAIN_MAP,
        robot_xml_path=robot_path,
        timestep=TIMESTEP,
        pixels_per_meter=PIXELS_PER_METER,
        # robot_collision_geom_names=[
        #     "left_ankle_roll_link_collision",
        #     "right_ankle_roll_link_collision",
        # ],
        # self_contact_pairs=[["left_hand_collision", "right_hand_collision"]],
    )

    # Add stereo cameras to robot head
    add_stereo_cameras_to_robot(spec)

    torso = spec.worldbody.first_body()

    spec.worldbody.add_camera(
        name="perspective",
        pos=np.array(torso.pos) + np.array([1.0, -1.0, 0.4]),
        xyaxes=[1, 1, 0, -1, 1, 3],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
    )

    # Add elevation visualization markers before compiling
    if VISUALIZE_ELEVATION_MAP:
        add_elevation_map_markers(spec, ELEVATION_MAP_SIZE, ELEVATION_MAP_RESOLUTION)

    model = spec.compile()

    # Use the compiled model to find the correct qpos index and modify keyframe
    neck_pitch_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch"
    )

    if neck_pitch_joint_id >= 0:
        neck_pitch_qpos_idx = model.jnt_qposadr[neck_pitch_joint_id]
        print(f"Found neck_pitch joint at qpos index: {neck_pitch_qpos_idx}")

        # Now modify the keyframe in the compiled model
        home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id >= 0:
            print(f"Home keyframe ID: {home_key_id}")
            # The keyframe arrays are indexed differently
            # model.key_qpos shape is (nkey * nq,) so we need the correct offset
            keyframe_qpos_offset = home_key_id * model.nq
            target_idx = keyframe_qpos_offset + neck_pitch_qpos_idx
            print(
                f"Trying to access key_qpos[{target_idx}] (offset {keyframe_qpos_offset} + joint idx {neck_pitch_qpos_idx})"
            )

            # The keyframe arrays are 2D: (nkey, nq) and (nkey, nu)
            model.key_qpos[home_key_id, neck_pitch_qpos_idx] = 0.85
            print("Set neck pitch qpos to -0.65 rad in home keyframe")

    # Debug: Print complete elevation info
    if elevation_info:
        print("\n=== ELEVATION INFO ===")

        # Base terrain info
        base_info = elevation_info["base"]
        print("Base Terrain:")
        print(
            f"  Bounds: X[{base_info['bounds'][0]:.3f}, {base_info['bounds'][1]:.3f}] Y[{base_info['bounds'][2]:.3f}, {base_info['bounds'][3]:.3f}]"
        )
        print(f"  Resolution: {base_info['resolution']} pixels/meter")
        if base_info["data"] is not None:
            hmap = base_info["data"]
            print(f"  Heightmap shape: {hmap.shape}")
            print(f"  Height range: [{hmap.min():.3f}, {hmap.max():.3f}] meters")
        # Obstacle info
        obstacles = elevation_info["obstacles"]
        print(f"\nObstacles: {len(obstacles)}")
        for i, obstacle in enumerate(obstacles):
            bounds = obstacle["bounds"]
            print(f"  {i}: {obstacle['type']} | Height: {obstacle['height']:.3f}m")
            print(
                f"      X[{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}] Y[{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}]"
            )
            print(
                f"      Base: {obstacle['base_height']:.3f}m | Collidable: {obstacle['collidable']}"
            )

    get_fovy(model)  # Print camera FOVY for debugging

    sim = ToddlerbotSimulator(
        mj_model=model,
        elevation_info=elevation_info,
        tile_width=TILE_WIDTH,
        tile_length=TILE_LENGTH,
        capture_depth=CAPTURE_DEPTH_VIDEO,
        display_depth=DISPLAY_DEPTH_REALTIME,
        depth_resolution=DEPTH_RESOLUTION,
    )
    sim.run()


if __name__ == "__main__":
    main()
