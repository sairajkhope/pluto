"""
Skill query system for terrain-based policy switching.

This module provides functions to determine which skill/policy should be active
based on the robot's current position and the terrain's skill map.

Assumptions:
- No two skills have overlapping bounds (first match wins)
- If no skill found, default to walk_policy
- JAX-compatible for use in RL training pipeline using array_utils
"""

import mujoco

from toddlerbot.sim.terrain.get_elevation import get_elevation_at_position
from toddlerbot.utils.array_utils import array_lib as jnp_or_np


def get_skill_at_position(x, y, skill_map, current_skill=None):
    """
    Get the skill that should be active at a given world position.
    JAX-compatible version using array_utils.

    Args:
        x (float): Robot's X position in world coordinates
        y (float): Robot's Y position in world coordinates
        skill_map (List[dict]): List of skill regions with bounds, skill, and transition requirements
        current_skill (str, optional): Currently active skill name for transition checking

    Returns:
        str or None: Skill name that should be active, or None if no skill (defaults to walk skill)
    """
    if not skill_map:
        return None

    # Convert inputs to JAX-compatible arrays
    x = jnp_or_np.array(x)
    y = jnp_or_np.array(y)

    # Use epsilon for robust boundary checking (same as elevation system)
    epsilon = 1e-6

    for skill in skill_map:
        bounds = skill["bounds"]

        # Check if position is within skill bounds with epsilon tolerance (JAX-compatible)
        x_in_bounds = (bounds["x"][0] - epsilon <= x) & (x <= bounds["x"][1] + epsilon)
        y_in_bounds = (bounds["y"][0] - epsilon <= y) & (y <= bounds["y"][1] + epsilon)

        if x_in_bounds & y_in_bounds:
            # Check if previous skill requirement is satisfied
            prev_skill_required = skill.get("prev_skill_required")
            if prev_skill_required is None:
                return skill["skill"]
            elif isinstance(prev_skill_required, list):
                # Multiple previous skills allowed
                if current_skill in prev_skill_required:
                    return skill["skill"]
            else:
                # Single previous skill required
                if prev_skill_required == current_skill:
                    return skill["skill"]

    # No skill found, default to walk policy
    return None


def get_robot_skill(robot_x, robot_y, skill_map, current_skill=None):
    """
    Get the skill that should be active for the robot's current position.

    Args:
        robot_x: Robot's X position (can be JAX array or NumPy array)
        robot_y: Robot's Y position (can be JAX array or NumPy array)
        skill_map (List[dict]): List of skill regions
        current_skill (str, optional): Currently active skill name

    Returns:
        str or None: Skill name that should be active, or None for default walk skill
    """
    return get_skill_at_position(robot_x, robot_y, skill_map, current_skill)


def get_skill_color(skill_name):
    """
    Get a distinct color for each skill type.

    Args:
        skill_name (str): Name of the skill

    Returns:
        List[float]: RGBA color [r, g, b, a] with values in [0, 1]
    """
    # Define distinct colors for each skill
    skill_colors = {
        # Box-related skills (yellow/orange family)
        "climb_up_box": [1.0, 0.8, 0.0, 0.8],  # Golden yellow
        "crawl_box": [1.0, 0.6, 0.0, 0.8],  # Orange
        "rotate_box": [1.0, 1.0, 0.0, 0.8],  # Dark orange
        # Chair-related skills (blue family)
        "get_down_chair": [0.0, 0.8, 1.0, 0.8],  # Light blue
        "crawl_chair": [0.0, 0.6, 1.0, 0.8],  # Medium blue
        "get_up_chair": [0.0, 0.4, 1.0, 0.8],  # Dark blue
        # Wall skills (red family)
        "climb_wall": [1.0, 0.0, 0.0, 0.8],  # Red
        # Stairs skills (green family)
        "crawl_up_stairs": [0.0, 1.0, 0.4, 0.8],  # Green
        "crawl_down_stairs": [0.0, 0.8, 0.2, 0.8],  # Dark green
        "get_up_stairs": [0.0, 0.6, 0.0, 0.8],  # Forest green
    }

    return skill_colors.get(skill_name, [1.0, 0.0, 1.0, 0.8])  # Default to magenta


def add_skill_map_markers(spec, skill_map, elevation_info):
    """
    Add visualization spheres to MjSpec for skill map display.
    Must be called before spec.compile().

    Args:
        spec: MjSpec object to add markers to
        skill_map (List[dict]): List of skill regions to visualize
        elevation_info: Elevation information for height positioning
    """

    for skill_idx, skill in enumerate(skill_map):
        bounds = skill["bounds"]
        skill_name = skill["skill"]

        # Add corner markers for skill bounds
        corners = [
            (bounds["x"][0], bounds["y"][0]),  # bottom-left
            (bounds["x"][1], bounds["y"][0]),  # bottom-right
            (bounds["x"][1], bounds["y"][1]),  # top-right
            (bounds["x"][0], bounds["y"][1]),  # top-left
        ]

        # Get distinct color for this skill
        color = get_skill_color(skill_name)

        for corner_idx, (corner_x, corner_y) in enumerate(corners):
            height = get_elevation_at_position(corner_x, corner_y, elevation_info)
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0.01, 0.01],  # Visible spheres for skill bounds
                pos=[corner_x, corner_y, height],  # Slightly above ground
                rgba=color,
                name=f"skill_{skill_idx}_{skill_name}_corner_{corner_idx}",
                group=4,  # Visual only group
                contype=0,  # No collision
                conaffinity=0,  # No collision
            )
