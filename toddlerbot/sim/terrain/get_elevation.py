"""
Elevation querying and mapping utilities for terrain systems.

This module provides:
- Terrain height queries at any world position
- Robot ground height retrieval
- Elevation map generation (both global and relative)
- Terminal printing of elevation maps
- Visual elevation markers update for MuJoCo (toggleable with key '4')

Used by simulation environments, policy runners, and testing scripts.
Compatible with both JAX and NumPy through array_utils.
"""

import mujoco
from scipy.spatial.transform import Rotation as R

from toddlerbot.utils.array_utils import array_lib as jnp_or_np
from toddlerbot.utils.array_utils import conditional_update


def _interpolate_heightfield(x, y, hmap, bounds):
    """
    Helper function to interpolate height from a heightfield at position (x, y).
    JAX-compatible version using array_utils.

    Args:
        x (float): World X coordinate
        y (float): World Y coordinate
        hmap (np.ndarray): Heightfield data array
        bounds (tuple): (x_min, x_max, y_min, y_max)

    Returns:
        float: Interpolated height at the position
    """
    # Convert to JAX-compatible arrays
    hmap_array = jnp_or_np.array(hmap)
    x_min, x_max, y_min, y_max = bounds

    # Clamp position to heightfield bounds
    x_clamped = jnp_or_np.clip(x, x_min, x_max)
    y_clamped = jnp_or_np.clip(y, y_min, y_max)

    # Convert world coordinates to heightfield indices
    nrow, ncol = hmap_array.shape

    # Map from world coordinates to pixel coordinates
    col_float = (x_clamped - x_min) / (x_max - x_min) * (ncol - 1)
    row_float = (y_clamped - y_min) / (y_max - y_min) * (nrow - 1)

    # Bilinear interpolation
    col_low = jnp_or_np.floor(col_float).astype(int)
    col_high = jnp_or_np.minimum(col_low + 1, ncol - 1)
    row_low = jnp_or_np.floor(row_float).astype(int)
    row_high = jnp_or_np.minimum(row_low + 1, nrow - 1)

    # Get interpolation weights
    col_weight = col_float - col_low
    row_weight = row_float - row_low

    # Get the four corner values
    h00 = hmap_array[row_low, col_low]
    h01 = hmap_array[row_low, col_high]
    h10 = hmap_array[row_high, col_low]
    h11 = hmap_array[row_high, col_high]

    # Bilinear interpolation
    h_bottom = h00 * (1 - col_weight) + h01 * col_weight
    h_top = h10 * (1 - col_weight) + h11 * col_weight
    height = h_bottom * (1 - row_weight) + h_top * row_weight

    return height


def get_elevation_at_position(x, y, elevation_info):
    """
    Get the terrain elevation at world position (x, y).
    JAX-compatible version using array_utils.

    Args:
        x (float): World X coordinate
        y (float): World Y coordinate
        elevation_info (dict): Elevation info from create_terrain_spec

    Returns:
        float: Terrain height at the position
    """
    if elevation_info is None:
        return 0.0

    # Check if position is inside any obstacle first
    epsilon = 1e-6
    for obstacle in elevation_info["obstacles"]:
        bounds = obstacle["bounds"]
        x_in_bounds = (bounds["x"][0] - epsilon <= x) & (x <= bounds["x"][1] + epsilon)
        y_in_bounds = (bounds["y"][0] - epsilon <= y) & (y <= bounds["y"][1] + epsilon)

        if x_in_bounds & y_in_bounds:
            # Inside obstacle - get height from obstacle
            if "hmap" in obstacle:
                # For slopes: interpolate from heightfield
                obstacle_bounds = (
                    bounds["x"][0],
                    bounds["x"][1],
                    bounds["y"][0],
                    bounds["y"][1],
                )
                interpolated_height = _interpolate_heightfield(
                    x, y, obstacle["hmap"], obstacle_bounds
                )
                return interpolated_height + obstacle["base_height"]
            else:
                # For fixed-height obstacles
                return obstacle["height"]

    # Not inside any obstacle - get height from base terrain
    base_info = elevation_info["base"]
    base_bounds = base_info["bounds"]
    return _interpolate_heightfield(x, y, base_info["data"], base_bounds)


def get_robot_ground_height(robot_x, robot_y, elevation_info):
    """
    Get the terrain height directly under the robot torso.

    Args:
        robot_x (float): Robot X position
        robot_y (float): Robot Y position
        elevation_info (dict): Elevation info from create_terrain_spec

    Returns:
        float: Terrain height under robot
    """
    return get_elevation_at_position(robot_x, robot_y, elevation_info)


def get_elevation_map(
    data,
    elevation_info,
    map_size=0.5,
    map_resolution=0.02,
    forward_offset=0.25,
):
    """
    Generate elevation maps around robot position.
    JAX-compatible version using array_utils.

    Args:
        data: MuJoCo data object (robot state extracted automatically)
        elevation_info (dict): Elevation info from create_terrain_spec
        map_size (float): Size of the square elevation map in meters
        map_resolution (float): Grid resolution in meters
        forward_offset (float): Offset map center forward in robot frame (meters)

    Returns:
        tuple: (global_elevation_map, relative_elevation_map, x_coords, y_coords)
            - global_elevation_map: 2D array of absolute terrain heights
            - relative_elevation_map: 2D array of heights relative to robot
            - x_coords: 2D array of X coordinates for each grid point
            - y_coords: 2D array of Y coordinates for each grid point
    """

    center_x, center_y = data.qpos[:2]
    quat = data.qpos[3:7]
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    robot_yaw = rot.as_euler("xyz")[2]

    # Calculate grid parameters
    half_size = map_size / 2
    grid_points = int(map_size / map_resolution) + 1

    # Generate local grid coordinates (robot body frame)
    local_coords = jnp_or_np.linspace(-half_size, half_size, grid_points)

    # Initialize arrays for world coordinates
    x_coords = jnp_or_np.zeros((grid_points, grid_points))
    y_coords = jnp_or_np.zeros((grid_points, grid_points))
    global_elevation_map = jnp_or_np.zeros((grid_points, grid_points))

    # Handle rotation if provided
    if robot_yaw is not None:
        cos_yaw = jnp_or_np.cos(robot_yaw)
        sin_yaw = jnp_or_np.sin(robot_yaw)
        # Apply forward offset
        offset_center_x = center_x + cos_yaw * forward_offset
        offset_center_y = center_y + sin_yaw * forward_offset
    else:
        cos_yaw = 1.0
        sin_yaw = 0.0
        offset_center_x = center_x + forward_offset  # Forward in X direction
        offset_center_y = center_y

    # Create coordinate grids for vectorized computation
    i_indices, j_indices = jnp_or_np.meshgrid(
        jnp_or_np.arange(grid_points), jnp_or_np.arange(grid_points), indexing="ij"
    )

    # Get local coordinates for all grid points
    local_y_grid = local_coords[i_indices]
    local_x_grid = local_coords[j_indices]

    # Transform from robot body frame to world frame for all points at once
    world_x_grid = offset_center_x + cos_yaw * local_x_grid - sin_yaw * local_y_grid
    world_y_grid = offset_center_y + sin_yaw * local_x_grid + cos_yaw * local_y_grid

    # Store coordinate grids
    x_coords = world_x_grid
    y_coords = world_y_grid

    # For JAX compatibility, vectorize the elevation calculation
    def get_elevation_vectorized(world_x, world_y):
        return get_elevation_at_position(world_x, world_y, elevation_info)

    # Apply elevation function to all grid points
    global_elevation_map = jnp_or_np.vectorize(get_elevation_vectorized)(
        world_x_grid, world_y_grid
    )

    # Calculate relative elevation map using robot's current position as reference
    robot_elevation = get_robot_ground_height(center_x, center_y, elevation_info)
    relative_elevation_map = global_elevation_map - robot_elevation

    return global_elevation_map, relative_elevation_map, x_coords, y_coords


def print_elevation_map(elevation_map):
    """
    Print elevation map to terminal in robot-centric view.
    Grid orientation: forward=right, left=up.

    Args:
        elevation_map: 2D array from get_elevation_map()
    """
    for i in reversed(range(elevation_map.shape[0])):
        for j in range(elevation_map.shape[1]):
            print(f"{elevation_map[i, j]:5.3f}", end=" ")
        print()
    print()


def add_elevation_map_markers(spec, map_size, map_resolution):
    """
    Add visualization spheres to MjSpec for elevation map display.
    Must be called before spec.compile() and before update_elevation_visualization().

    Args:
        spec: MjSpec object to add markers to
        map_size (float): Size of elevation map in meters
        map_resolution (float): Grid resolution in meters
    """
    grid_points = int(map_size / map_resolution) + 1

    for i in range(grid_points):
        for j in range(grid_points):
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0.005, 0.005],  # Smaller spheres for grid
                pos=[0, 0, 0],  # Will be updated dynamically
                rgba=[0.0, 1.0, 0.0, 1.0],
                name=f"elevation_grid_{i}_{j}",
                group=4,  # Visual only group
                contype=0,  # No collision
                conaffinity=0,  # No collision
            )


def update_elevation_visualization(data, model, global_map, x_coords, y_coords):
    """
    Update elevation map visualization markers.
    Requires add_elevation_map_markers() called before using this method.

    Args:
        data: MuJoCo data object
        model: MuJoCo model object
        global_map: 2D elevation map from get_elevation_map()
        x_coords: 2D X coordinate array from get_elevation_map()
        y_coords: 2D Y coordinate array from get_elevation_map()
    """

    # Update each marker
    for i in range(global_map.shape[0]):
        for j in range(global_map.shape[1]):
            marker_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"elevation_grid_{i}_{j}"
            )
            # marker_id = model.geom(f"elevation_grid_{i}_{j}").id

            # Position at global terrain elevation
            x = float(x_coords[i, j])
            y = float(y_coords[i, j])
            global_elevation = float(global_map[i, j])

            data.geom_xpos[marker_id] = jnp_or_np.array([x, y, global_elevation])

            # TODO: Set color based on height
            # color = [0.0, 1.0, 0.0, 1.0]  # Green
            # model.geom_rgba[marker_id] = color
