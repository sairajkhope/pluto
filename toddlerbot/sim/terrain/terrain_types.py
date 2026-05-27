"""
Defines functions to procedurally generate different terrain tiles.
Each function returns a heightmap patch (2D numpy array) and its maximum height value.

Supported tile types:
- Bumps: Perlin-based smooth noise with edge fade and flat center.
- Rough: Higher-frequency Perlin noise, rougher than bumps.
- Slope: Radial frustum shape with a flat top.
- Stairs: Step-like concentric square levels increasing in height.
- Boxes: Random box-shaped bumps, avoiding the center spawn area.

Depends on terrain_utils.py for helper utilities.
"""

import numpy as np

from toddlerbot.utils.terrain_utils import (
    center_flat_mask,
    edge_slope,
    frustum_with_flat_top,
    perlin,
)


def generate_bumps_patch(size):
    """
    Generate a smooth Perlin noise bump patch with edge fade and flat center.

    Args:
        size (int): Resolution of the square heightmap.

    Returns:
        Tuple[np.ndarray, float]: Heightmap array and its max height.
    """
    bump_frequency = 8
    raw = perlin((size, size), (bump_frequency, bump_frequency))
    raw = (raw - np.min(raw)) / (np.max(raw) - np.min(raw))  # Normalize to [0, 1]

    fade = edge_slope(size, border_width=10, blur_iterations=100)
    center_mask = center_flat_mask(size, flat_radius_frac=0.04)

    heightmap = raw * fade * center_mask
    max_roughness = 0.12
    heightmap *= max_roughness

    return heightmap, heightmap.max()


def generate_rough_patch(size):
    """
    Generate a rougher version of Perlin terrain with smaller bumps.

    Args:
        size (int): Resolution of the square heightmap.

    Returns:
        Tuple[np.ndarray, float]: Heightmap array and its max height.
    """
    bump_frequency = 32  # Higher frequency => smaller rough bumps
    raw = perlin((size, size), (bump_frequency, bump_frequency))
    raw = (raw - np.min(raw)) / (np.max(raw) - np.min(raw))

    center_mask = center_flat_mask(size, flat_radius_frac=0.01)
    heightmap = raw * center_mask

    max_roughness = 0.02
    heightmap *= max_roughness

    return heightmap, heightmap.max()


def generate_slope_patch(size, peak=0.2, flat_ratio=0.15):
    """
    Generate a radial frustum-shaped slope with a flat top.

    Args:
        size (int): Size of the heightmap (square).
        peak (float): Maximum height at the center.
        flat_ratio (float): Ratio of flat top region.

    Returns:
        Tuple[np.ndarray, float]: Heightmap and max height.
    """
    edge_flat_width = 0
    hmap = frustum_with_flat_top(
        size, peak=peak, flat_ratio=flat_ratio, edge_flat_width=edge_flat_width
    )
    return hmap, peak


# TODO: Slopes don't look realistic yet due to limitied resolution of heightmap.
def generate_up_slope_patch(size, peak=0.13):
    """
    Generate a linear slope going up from left to right.

    Args:
        size (int): Size of the heightmap (square).
        peak (float): Maximum height at the right edge.

    Returns:
        Tuple[np.ndarray, float]: Heightmap and max height.
    """
    hmap = np.zeros((size, size), dtype=np.float32)
    x_coords = np.linspace(0, 1, size)
    hmap[:, :] = x_coords * peak
    return hmap, peak


def generate_down_slope_patch(size, peak=0.13):
    """
    Generate a linear slope going down from left to right.

    Args:
        size (int): Size of the heightmap (square).
        peak (float): Maximum height at the left edge.

    Returns:
        Tuple[np.ndarray, float]: Heightmap and max height.
    """
    hmap = np.zeros((size, size), dtype=np.float32)
    x_coords = np.linspace(0, 1, size)
    hmap[:, :] = (1 - x_coords) * peak
    return hmap, peak


def generate_stairs_patch(size, num_steps=5, peak_height=0.1):
    """
    Generate a stair-like concentric step pattern.

    Args:
        size (int): Size of the square heightmap.
        num_steps (int): Number of concentric step levels.
        peak_height (float): Maximum height at the innermost step.

    Returns:
        Tuple[np.ndarray, float]: Heightmap and max height.
    """
    hmap = np.zeros((size, size), dtype=np.float32)

    step_thickness = size // (2 * num_steps)

    for s in range(num_steps):
        level_height = peak_height * (s + 1) / num_steps
        pad = s * step_thickness

        if pad >= size // 2:
            break  # Prevent invalid indexing when steps get too small

        # Create a square ring at this level
        hmap[pad : size - pad, pad : size - pad] = level_height

    return hmap, peak_height


def generate_boxes_patch(
    size, num_boxes=40, box_height=0.01, box_size_ratio=0.1, center_ratio=0.1
):
    """
    Generate a sparse grid with randomly placed square boxes.

    Avoids placing boxes in a central region to ensure safe spawning.

    Args:
        size (int): Size of the grid (square).
        num_boxes (int): Number of boxes to place.
        box_height (float): Height of each box bump.
        box_size_ratio (float): Fraction of the grid taken up by one box.
        center_ratio (float): Size of protected center area as fraction of grid.

    Returns:
        Tuple[np.ndarray, float]: Heightmap with boxes and max height.
    """
    hmap = np.zeros((size, size), dtype=np.float32)

    box_size = max(1, int(size * box_size_ratio))
    center_size = int(size * center_ratio)
    center_start = (size - center_size) // 2
    center_end = center_start + center_size

    rng = np.random.default_rng()

    for _ in range(num_boxes):
        attempts = 0
        while attempts < 100:
            x = rng.integers(0, size - box_size)
            y = rng.integers(0, size - box_size)

            # Check if box overlaps protected center
            if (x + box_size < center_start or x > center_end) or (
                y + box_size < center_start or y > center_end
            ):
                hmap[y : y + box_size, x : x + box_size] = box_height
                break

            attempts += 1

    return hmap, box_height


def generate_box_patch(size):
    """
    Generate a flat patch. The box obstacle is a separate geom.
    """
    hmap = np.zeros((size, size), dtype=np.float32)
    return hmap, 0.0


def generate_crawl_patch(size):
    """
    Generate a flat patch for the crawl task. The chair is added later.

    Args:
        size (int): Resolution of the square heightmap.

    Returns:
        Tuple[np.ndarray, float]: Heightmap array and its max height (0.0).
    """
    hmap = np.zeros((size, size), dtype=np.float32)
    return hmap, 0.0


def generate_wall_patch(size):
    """
    Generate a flat patch for the wall obstacle. The wall geom is added later.

    Args:
        size (int): Resolution of the square heightmap.

    Returns:
        Tuple[np.ndarray, float]: Heightmap array and its max height (0.0).
    """
    hmap = np.zeros((size, size), dtype=np.float32)
    return hmap, 0.0
