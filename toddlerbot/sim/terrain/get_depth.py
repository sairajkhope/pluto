"""Depth perception utilities for stereo camera setup on ToddlerBot head."""

import os

import cv2
import mediapy as media
import mujoco
import numpy as np


def load_camera_intrinsics(
    rectification_path: str = None,
    resolution: tuple = (640, 480),
):
    """Load camera intrinsic parameters from rectification file.

    Args:
        rectification_path: Path to rectification.npz file (default: toddlerbot/depth/params/rectification.npz)
        resolution: Camera resolution (width, height) for calculating principal pixel offset

    Returns:
        dict with keys:
            - focal_pixel: (fx, fy) from P1 projection matrix
            - principal_pixel: (cx_offset, cy_offset) from P1 projection matrix
            - sensor_size: Fixed values required by MuJoCo (actual values don't affect rendering)
    """
    if rectification_path is None:
        # Default path relative to this file's location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rectification_path = os.path.join(
            current_dir, "..", "..", "depth", "params", "rectification.npz"
        )

    rect = np.load(rectification_path, allow_pickle=True)
    P1 = rect["P1"]

    # Extract focal lengths from P1 projection matrix
    # P1 format: [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]]
    fx = P1[0, 0]
    fy = P1[1, 1]
    cx_rect = P1[0, 2]
    cy_rect = P1[1, 2]

    # Calculate principal pixel offset from default center
    # Default center for MuJoCo: (width/2 - 0.5, height/2 - 0.5)
    width, height = resolution
    cx_default = width / 2 - 0.5  # 319.5 for 640x480
    cy_default = height / 2 - 0.5  # 239.5 for 640x480

    cx_offset = cx_default - cx_rect
    cy_offset = cy_default - cy_rect

    return {
        "focal_pixel": [fx, fy],
        "principal_pixel": [cx_offset, cy_offset],
        # Sensor size values don't seem to matter but are required to apply resolution, focal_pixel, and principal_pixel
        "sensor_size": [0.0048, 0.0036],
    }


def add_stereo_cameras_to_robot(spec: mujoco.MjSpec) -> None:
    """Add stereo cameras to the robot head for depth perception.

    Args:
        spec: MuJoCo model specification to modify

    Example usage:
        spec, _, _, _, _ = create_terrain_spec(...)
        add_stereo_cameras_to_robot(spec)
        model = spec.compile()
    """
    # Find the head body in the robot specification
    head_body = None
    torso_body = spec.worldbody.first_body()  # This should be the torso

    if torso_body is None:
        raise ValueError("Could not find torso body in robot specification")

    # Search for head body in the robot specification
    # Let's traverse all bodies in the spec to find the head
    head_body = None

    def search_all_bodies(body):
        nonlocal head_body
        if body.name == "head":
            head_body = body
            return
        # Search child bodies
        child = body.first_body()
        while child and head_body is None:
            search_all_bodies(child)
            # Move to next sibling - need to check if there's a next body
            try:
                next_child = body.next_body(child)
                child = next_child
            except Exception:
                # No more siblings
                break

    search_all_bodies(torso_body)

    if head_body is None:
        raise ValueError("Could not find head body in robot specification")

    # Load camera intrinsics from calibration files
    intrinsics = load_camera_intrinsics()

    # Add left stereo camera
    head_body.add_camera(
        name="head_cam_left",
        # TODO: Calculate camera position based on head geometry
        pos=[0.01, -0.004 + 0.033 / 2, 0.048],
        euler=[1.57, -1.57, 0],
        resolution=[640, 480],
        focal_pixel=intrinsics["focal_pixel"],
        principal_pixel=intrinsics["principal_pixel"],
        sensor_size=intrinsics["sensor_size"],
    )

    # Add right stereo camera
    head_body.add_camera(
        name="head_cam_right",
        pos=[0.01, -0.004 - 0.033 / 2, 0.048],
        euler=[1.57, -1.57, 0],
        resolution=[640, 480],
        focal_pixel=intrinsics["focal_pixel"],
        principal_pixel=intrinsics["principal_pixel"],
        sensor_size=intrinsics["sensor_size"],
    )

    print("Added stereo cameras to robot head:")
    print("  - Left camera: head_cam_left")
    print("  - Right camera: head_cam_right")


def add_data_collection_cameras(
    spec: mujoco.MjSpec,
    grid_size_yz: tuple = (0.04, 0.03),
    grid_spacing: float = 0.01,
    x_spacing: float = 0.01,
    x_positions: list = [0.0],  # Only at baseline center by default
    yaw_angles: list = [-0.1, 0.0, 0.1],  # Controls up/down camera tilt
    roll_angles: list = [-0.1, 0.0, 0.1],  # Controls left/right camera roll
) -> list:
    """Add a dense grid of cameras around the stereo baseline for comprehensive data collection.

    Args:
        spec: MuJoCo model specification to modify
        grid_size_yz: Size of the rectangular grid area in (y, z) dimensions
        grid_spacing: Spacing between grid points (meters)
        x_spacing: Spacing between x positions (meters)
        x_positions: List of x offsets from baseline center
        yaw_angles: List of yaw angles (rad) for camera orientations (up/down)
        roll_angles: List of roll angles (rad) for camera orientations (left/right roll)

    Creates cameras at:
    - Multiple x positions (e.g., -0.01, 0.01 from baseline center)
    - Grid positions in YZ plane around stereo baseline center
    - Multiple orientations (pitch/yaw combinations) at each grid position
    """
    # Find the head body
    head_body = None
    torso_body = spec.worldbody.first_body()

    def search_all_bodies(body):
        nonlocal head_body
        if body.name == "head":
            head_body = body
            return
        child = body.first_body()
        while child and head_body is None:
            search_all_bodies(child)
            try:
                next_child = body.next_body(child)
                child = next_child
            except Exception:
                break

    search_all_bodies(torso_body)

    if head_body is None:
        raise ValueError("Could not find head body in robot specification")

    # Load camera intrinsics from calibration files
    intrinsics = load_camera_intrinsics()

    # Stereo baseline center (based on current stereo camera positions)
    baseline_center = [0.01, -0.004, 0.048]

    # Calculate grid positions
    y_size, z_size = grid_size_yz
    y_positions = np.arange(-y_size / 2, y_size / 2 + grid_spacing / 2, grid_spacing)
    z_positions = np.arange(-z_size / 2, z_size / 2 + grid_spacing / 2, grid_spacing)

    camera_count = 0
    camera_names = []

    for x_offset in x_positions:
        for y_offset in y_positions:
            for z_offset in z_positions:
                for yaw in yaw_angles:
                    for roll in roll_angles:
                        # Calculate camera position
                        cam_pos = [
                            baseline_center[0] + x_offset,
                            baseline_center[1] + y_offset,
                            baseline_center[2] + z_offset,
                        ]

                        # Calculate euler angles with correct naming
                        # Base orientation is [1.57, -1.57, 0] for forward-facing
                        # yaw_angles -> up/down, roll_angles -> left/right roll
                        euler = [1.57 + roll, -1.57 + yaw, 0]

                        # Create unique camera name with integer indices to avoid floating point duplicates
                        x_idx = int(x_offset * 1000)  # Convert to mm as integer
                        y_idx = int(y_offset * 1000)
                        z_idx = int(z_offset * 1000)
                        r_idx = int(roll * 1000)  # Convert to milliradians as integer
                        w_idx = int(yaw * 1000)
                        camera_name = f"data_cam_x{x_idx:+04d}_y{y_idx:+04d}_z{z_idx:+04d}_r{r_idx:+04d}_w{w_idx:+04d}"
                        camera_names.append(camera_name)

                        # Add camera
                        head_body.add_camera(
                            name=camera_name,
                            pos=cam_pos,
                            euler=euler,
                            resolution=[640, 480],
                            focal_pixel=intrinsics["focal_pixel"],
                            principal_pixel=intrinsics["principal_pixel"],
                            sensor_size=intrinsics["sensor_size"],
                        )

                        camera_count += 1

    print(f"Added {camera_count} data collection cameras to robot head")
    print(f"  - Grid size: {y_size:.3f}m x {z_size:.3f}m (Y x Z)")
    print(f"  - Grid spacing: {grid_spacing:.3f}m")
    print(f"  - X positions: {x_positions}")
    print(
        f"  - Orientations: {len(roll_angles)} roll × {len(yaw_angles)} yaw = {len(roll_angles) * len(yaw_angles)} per position"
    )

    return camera_names


def setup_depth_capture(
    model: mujoco.MjModel,
    resolution=(128, 96),
    capture_video: bool = False,
    display_realtime: bool = False,
    video_fps: int = 30,
    output_dir: str = "",
):
    """Setup depth capture system with renderer and optional video writer.

    Args:
        model: MuJoCo model containing the head cameras
        capture_video: Whether to save depth video to file
        display_realtime: Whether to show real-time depth in OpenCV window
        video_fps: FPS for video output
        output_dir: Directory to save video file

    Returns:
        Tuple of (renderer, video_writer, camera_resolution)
        video_writer will be None if capture_video=False

    Example usage:
        renderer, writer, resolution = setup_depth_capture(model, capture_video=True)
        # ... use renderer for depth capture
        if writer:
            writer.release()
    """
    width, height = resolution[0], resolution[1]

    # Initialize renderer with camera resolution
    renderer = mujoco.Renderer(model, width=width, height=height)

    # Setup video recording using mediapy (stores frames in memory)
    video_writer = None
    if capture_video:
        if len(output_dir) > 0:
            os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, "depth_video.mp4")
        # Use a dictionary to store video metadata and frames
        video_writer = {
            "frames": [],
            "fps": video_fps,
            "output_path": video_path,
        }
        print(f"Depth video recording enabled - frames will be saved to {video_path}")

    # Setup OpenCV window for real-time display (only if display is enabled)
    if display_realtime:
        cv2.namedWindow("Depth View", cv2.WINDOW_AUTOSIZE)
        print(f"Real-time depth display enabled - resolution: {width}x{height}")

    print(f"Depth system setup complete - resolution: {width}x{height}")

    return renderer, video_writer, (width, height)


def process_depth_map(
    depth_map: np.ndarray, add_noise: bool = False, max_depth=1.0
) -> np.ndarray:
    """Process raw depth map with optional noise augmentation.

    Args:
        depth_map: Raw depth map from renderer
        add_noise: If True, applies local noise + blur for domain gap simulation (default: False)
        max_depth: Maximum depth value to clip (default: 1.0m)

    Returns:
        Processed depth map with cropping, optional noise, and clipping applied
    """
    # Crop the leftmost pixels like in Foundation Stereo.
    processed_depth_map = depth_map[:, 10:]

    if add_noise:
        # Add domain gap simulation to match real Foundation Stereo characteristics
        # 1. Generate smooth, spatially coherent noise like neural networks
        h, w = processed_depth_map.shape
        smooth_noise = np.random.normal(0, 0.02, (h // 96, w // 96))
        smooth_noise = cv2.resize(smooth_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        processed_depth_map += smooth_noise

        # 2. Slight blur to simulate neural model smoothing
        processed_depth_map = cv2.GaussianBlur(processed_depth_map, (5, 5), 15)

        # NOTE: Temporal flickering kept disabled - too unstable for training
        # # 3. Temporal flickering simulation (random global offset)
        # flicker_offset = np.random.normal(0, 0.01)  # 0.01m global flicker
        # processed_depth_map += flicker_offset

        # Ensure no negative depths
        processed_depth_map = np.maximum(processed_depth_map, 0)

    # Clip depth values to max_depth and return
    processed_depth_map = np.clip(processed_depth_map, 0, max_depth)

    return processed_depth_map


def get_depth_map(
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera_name: str = "head_cam_left",
    add_noise: bool = False,
    max_depth: float = 1.0,
) -> np.ndarray:
    """Get depth image from the specified camera.

    Args:
        data: MuJoCo data
        renderer: MuJoCo renderer (should be initialized with appropriate resolution)
        camera_name: Name of the camera to use for depth rendering
        add_noise: If True, applies local noise + blur augmentation (default: False for clean data collection)
        max_depth: Maximum depth value in meters, values beyond this are clipped (default: 1.0m)

    Returns:
        Depth map as numpy array with values clipped to [0, max_depth] range and in meter unit.
    """
    # Enable depth rendering
    renderer.enable_depth_rendering()

    # Update scene with specified camera
    renderer.update_scene(data, camera=camera_name)

    # Render depth image
    depth = renderer.render()

    # Disable depth rendering
    renderer.disable_depth_rendering()

    processed_depth_map = process_depth_map(depth, add_noise, max_depth)

    return processed_depth_map


def create_depth_visualization(
    depth_map: np.ndarray, depth_vis: np.ndarray, max_depth: float = 1.0
) -> np.ndarray:
    """Create enhanced depth visualization with legend and center value.

    Args:
        depth_map: Raw depth map in meters (values 0 to max_depth)
        depth_vis: 8-bit depth visualization (values 0-255)
        max_depth: Maximum depth range for legend labels

    Returns:
        Enhanced grayscale visualization with legend, crosshair, and center depth value
    """
    # Use original depth_vis directly (grayscale, no scaling)
    depth_display = depth_vis.copy()

    # Get center depth value
    center_y, center_x = depth_map.shape[0] // 2, depth_map.shape[1] // 2
    center_depth = depth_map[center_y, center_x]

    # Draw center crosshair (using different intensities for visibility)
    cv2.line(
        depth_display,
        (center_x - 5, center_y),
        (center_x + 5, center_y),
        128,  # Gray crosshair
        1,
    )
    cv2.line(
        depth_display,
        (center_x, center_y - 5),
        (center_x, center_y + 5),
        128,  # Gray crosshair
        1,
    )

    # Create a larger canvas with space for legend and text area below (grayscale)
    canvas_width = depth_display.shape[1] + 120  # Extra space for legend
    text_area_height = 20  # Space for center depth text below image
    canvas_height = depth_display.shape[0] + text_area_height
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # Place depth image on canvas
    canvas[: depth_display.shape[0], : depth_display.shape[1]] = depth_display

    # Create depth legend/colorbar
    legend_x = depth_display.shape[1] + 10
    legend_width = 15
    legend_height = depth_display.shape[0] - 20
    legend_y_start = 10

    # Create vertical gradient for legend
    for i in range(legend_height):
        intensity = int(255 * (legend_height - i) / legend_height)
        cv2.rectangle(
            canvas,
            (legend_x + 60, legend_y_start + i),
            (legend_x + legend_width + 60, legend_y_start + i + 1),
            intensity,
            -1,
        )

    # Add legend border
    cv2.rectangle(
        canvas,
        (legend_x + 60, legend_y_start),
        (legend_x + legend_width + 60, legend_y_start + legend_height),
        255,
        1,
    )

    # Add grayscale text labels (no color conversion)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_color = 255  # White text for grayscale
    font_thickness = 1

    # Max depth label (top)
    cv2.putText(
        canvas,
        f"{max_depth:.1f}m",
        (legend_x + legend_width + 65, legend_y_start + 10),
        font,
        font_scale,
        font_color,
        font_thickness,
    )

    # Min depth label (bottom)
    cv2.putText(
        canvas,
        "0.0m",
        (legend_x + legend_width + 65, legend_y_start + legend_height - 5),
        font,
        font_scale,
        font_color,
        font_thickness,
    )

    # Center depth value display in separate text area below (like Foundation Stereo)
    center_text = f"Depth at center: {center_depth:.3f} m"
    text_position = (10, depth_display.shape[0] + text_area_height - 7)
    cv2.putText(canvas, center_text, text_position, font, 0.4, 255, 1)

    return canvas


def capture_depth_frame(
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    video_writer=None,
    display_realtime: bool = False,
    camera_name: str = "head_cam_left",
    add_noise: bool = False,
    max_depth: float = 1.0,
):
    """Capture and optionally display/save a depth frame.

    Args:
        data: MuJoCo data
        renderer: MuJoCo renderer for depth capture
        video_writer: Optional OpenCV video writer for saving frames
        display_realtime: Whether to display in OpenCV window
        camera_name: Name of camera to capture from
        add_noise: If True, applies local noise + blur augmentation (default: False for clean data collection)
        max_depth: Maximum depth range in meters

    Returns:
        depth_map: Raw depth map in meters, or None if capture failed
    """
    if renderer is None:
        return None

    try:
        # Get depth map from camera
        depth_map = get_depth_map(data, renderer, camera_name, add_noise, max_depth)

        # Convert to 8-bit for visualization (0-255 range)
        depth_vis = (255 * depth_map / max_depth).astype(np.uint8)

        # Display in real-time OpenCV window
        if display_realtime:
            depth_display = create_depth_visualization(depth_map, depth_vis, max_depth)
            cv2.imshow("Depth View", depth_display)
            cv2.waitKey(1)  # Non-blocking update

        # Write frame to video (collect in memory using mediapy format)
        if video_writer is not None:
            # Convert single-channel grayscale to BGR for proper encoding
            depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            # Append frame to list (mediapy expects RGB or BGR array of shape (H, W, 3))
            video_writer["frames"].append(depth_bgr)

        # Return the visualized depth display
        return depth_vis

    except Exception as e:
        print(f"Error processing depth frame: {e}")
        return None


def cleanup_depth_capture(video_writer=None, display_realtime: bool = False):
    """Cleanup depth capture resources.

    Args:
        video_writer: Optional dict containing video metadata and frames to save
        display_realtime: Whether real-time display was enabled
    """
    if video_writer is not None and isinstance(video_writer, dict):
        if video_writer["frames"]:
            frames_array = np.array(video_writer["frames"])
            output_path = video_writer["output_path"]
            fps = video_writer["fps"]
            try:
                media.write_video(output_path, frames_array, fps=fps)
                print(
                    f"Depth video saved to {output_path} ({len(video_writer['frames'])} frames)"
                )
            except Exception as e:
                print(f"ERROR: Failed to save depth video: {e}")
        else:
            print("WARNING: No frames were captured for depth video")

    if display_realtime:
        cv2.destroyAllWindows()
        print("Depth display window closed")


def get_fovy(model: mujoco.MjModel) -> float:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam_left"
    )
    fovy_deg = model.cam_fovy[cam.fixedcamid]
    print(f"Camera FOVY: {fovy_deg}")
    return fovy_deg
