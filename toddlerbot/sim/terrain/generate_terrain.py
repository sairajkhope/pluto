"""
Main terrain generator that creates procedural terrain from modular patches.

Each tile in the 2D `terrain_map` specifies a terrain type ("flat", "bumps", "slope", etc.).
The generator creates base terrain heightfields and places obstacle geoms, returning an
MjSpec model with structured elevation info for accurate terrain queries.

For flat terrain, uses an efficient plane geom with checker texture.
For mixed terrain, creates heightfields with separate obstacle tracking.

Used during simulation setup and RL environment creation.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from toddlerbot.sim.terrain.terrain_types import (
    generate_box_patch,
    generate_boxes_patch,
    generate_bumps_patch,
    generate_crawl_patch,
    generate_down_slope_patch,
    generate_rough_patch,
    generate_slope_patch,
    generate_stairs_patch,
    generate_up_slope_patch,
    generate_wall_patch,
)

# === Supported terrain types ===
TERRAIN_TYPES = ["flat", "rough", "bump", "slope", "stairs", "boxes", "testbed"]


def generate_groundplane(spec):
    """
    Generate a flat groundplane with checker texture for flat terrain.

    Args:
        spec (MjSpec): The MuJoCo spec to add the groundplane to.

    Returns:
        str: Name of the floor geom created.
    """
    # Workaround: MjSpec doesn't support built-in textures via API,
    # spec.add_texture(
    #     name="groundplane",
    #     type=mujoco.mjtTexture.mjTEXTURE_2D,
    #     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    #     rgb1=[0.2, 0.3, 0.4],
    #     rgb2=[0.1, 0.2, 0.3],
    #     mark=mujoco.mjtMark.mjMARK_EDGE,
    #     markrgb=[0.8, 0.8, 0.8],
    #     width=300,
    #     height=300,
    # )
    # spec.add_material(
    #     name="groundplane",
    #     textures=["groundplane"],
    #     texuniform=True,
    #     texrepeat=[5, 5],
    #     reflectance=0.0,
    # )

    # Create checker texture using MJCF snippet
    checker_snippet = """
    <mujoco>
        <asset>
            <texture name="groundplane" type="2d" builtin="checker"
                    width="300" height="300"
                    rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
                    mark="edge" markrgb="0.8 0.8 0.8"/>

            <material name="groundplane" texture="groundplane"
                    texuniform="true" texrepeat="5 5" reflectance="0.0"/>
        </asset>
    </mujoco>
    """

    # Parse MJCF snippet and extract assets
    patch_spec = mujoco.MjSpec.from_string(checker_snippet)

    # Add textures
    for tex in patch_spec.textures:
        spec.add_texture(
            name=tex.name,
            type=tex.type,
            builtin=tex.builtin,
            width=tex.width,
            height=tex.height,
            rgb1=tex.rgb1,
            rgb2=tex.rgb2,
            mark=tex.mark,
            markrgb=tex.markrgb,
        )

    # Add materials
    for mat in patch_spec.materials:
        spec.add_material(
            name=mat.name,
            textures=mat.textures,
            texuniform=mat.texuniform,
            texrepeat=mat.texrepeat,
            reflectance=mat.reflectance,
        )

    floor_name = "floor"
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material="groundplane",
        name=floor_name,
        condim=3,  # default is 1
    )

    return floor_name


def create_terrain_spec(
    tile_width,
    tile_length,
    terrain_map,
    robot_xml_path=None,
    robot_position=(0.0, 0.0, 0.315053),
    timestep=0.004,
    pixels_per_meter=16,
    robot_collision_geom_names=None,
    self_contact_pairs=None,
):
    """
    Creates a MuJoCo MjSpec model with procedurally generated terrain and an optional robot.

    Terrain is built from a 2D grid of tile names (e.g., "flat", "slope", "stairs").
    If all tiles are "flat", a single plane geom is used instead of a heightfield for performance.

    If a robot is loaded from XML, its initial pose is set, and contact pairs are added between
    terrain geoms and the robot's collision geoms. If robot_collision_geom_names is not provided,
    it is automatically inferred from the geom names appearing in the MJCF <pair> tags.

    Args:
        tile_width (float): Width of each terrain tile in meters.
        tile_length (float): Length of each terrain tile in meters.
        terrain_map (List[List[str]]): 2D list specifying terrain type per tile.
        robot_xml_path (str): Path to MJCF XML file describing the robot (optional).
        robot_position (Tuple[float, float, float]): Initial robot base position.
        timestep (float): Physics simulation timestep.
        pixels_per_meter (int): Resolution scale for the terrain heightfield.
        robot_collision_geom_names (List[str] or None): Names of robot geoms to register
            for terrain contact. If None, inferred from MJCF <pair> entries and filtered
            to geoms defined under the robot body.
        self_contact_pairs (List[List[str]] or None): A list of custom contact pairs (geom1, geom2)
            to be added explicitly to the MuJoCo scene. If defined, this overrides and removes the original
            <contact> section from the robot MJCF.

    Returns:
        spec (MjSpec): Compiled MuJoCo scene specification.
        terrain_geom_names (List[str]): Names of the geoms used for terrain.
        safe_spawns (List[Tuple[float, float, float]]): List of spawn positions per tile.
        elevation_info (dict): Structured terrain information for elevation queries.
        skill_map (List[dict]): List of skill regions with bounds, policy, and transition requirements.
    """
    is_testbed_mode = (
        len(terrain_map) == 1
        and len(terrain_map[0]) == 1
        and terrain_map[0][0] == "testbed"
    )
    if any("testbed" in row for row in terrain_map) and not is_testbed_mode:
        raise ValueError(
            "Testbed mode requires a single tile with 'testbed' terrain type."
        )
    if is_testbed_mode:
        print("\n=== TESTBED MODE ACTIVATED ===")
        # Define possible obstacle courses for the middle row
        obstacle_courses = [
            # box and stairs should have a flat in between
            # chair and stairs should have a flat in between
            # Cases where 'flat' is placed between 'box' and 'full_stairs'
            ["flat", "crawl", "wall", "box", "flat", "full_stairs"],
            # ["flat", "full_stairs", "flat", "box", "wall", "crawl"],
            # ["flat", "wall", "full_stairs", "flat", "crawl", "box"],
            # ["flat", "box", "crawl", "flat", "full_stairs", "wall"],
            # ["flat", "crawl", "flat", "full_stairs", "wall", "box"],
            # TODO: up_slope and down_slope are not supported as base is groundplane for testbed
        ]

        # Build terrain map with all courses separated by flat rows
        terrain_map = []
        max_cols = max(len(course) for course in obstacle_courses)

        # Add a flat row at the beginning
        terrain_map.append(["flat"] * max_cols)

        # Add each obstacle course as a row, padding with "flat" if needed
        for course in obstacle_courses:
            padded_course = course + ["flat"] * (max_cols - len(course))
            terrain_map.append(padded_course)
            # Add flat row after each course for separation
            terrain_map.append(["flat"] * max_cols)

        # Override tile dimensions to a fixed 1x1 meter for the testbed
        tile_width = 1.0
        tile_length = 1.0
        print("  - Using fixed 1.0m x 1.0m tiles for testbed terrain.")

    # Check if all tiles are flat
    all_flat = all(t == "flat" for row in terrain_map for t in row)

    # Flip vertically so row 0 appears at the top visually
    terrain_map = terrain_map[::-1]

    rows = len(terrain_map)
    cols = len(terrain_map[0])

    total_width = cols * tile_width
    total_length = rows * tile_length

    nrow = int(total_length * pixels_per_meter)
    ncol = int(total_width * pixels_per_meter)

    print("\n=== TERRAIN MAP ===")
    for row in terrain_map:
        print("  - " + " ".join(f"{cell:2}" for cell in row))

    print("\n=== TERRAIN STATS ===")
    print(f"  - Total width  : {total_width:.2f} m")
    print(f"  - Total length : {total_length:.2f} m")
    print(f"  - Pixels per m : {pixels_per_meter}")
    print(f"  - HField size  : {nrow} rows x {ncol} cols")

    # === Create MjSpec model ===
    if robot_xml_path:
        # If adding user configured pairs, parse the XML, remove the <contact> section,
        # and load the spec from the modified string.
        if self_contact_pairs is not None:
            # Load the raw XML string from the file
            with open(robot_xml_path, "r") as f:
                xml_string = f.read()

            tree = ET.fromstring(xml_string)
            contact_element = tree.find("contact")
            if contact_element is not None:
                tree.remove(contact_element)

            modified_xml_string = ET.tostring(tree, encoding="unicode")
            spec = mujoco.MjSpec.from_string(modified_xml_string)
            meshdir = "/".join(robot_xml_path.split("/")[:-1] + ["assets"])

            # Set the mesh directory again since it's loaded from a modified XML string
            spec.meshdir = meshdir

            # Add the custom inter-robot contact pairs
            print("\n=== INTER-ROBOT COLLISION GEOMS ===")
            for pair in self_contact_pairs:
                if len(pair) == 2:
                    spec.add_pair(geomname1=pair[0], geomname2=pair[1])
                    print(f"{pair[0]} <-> {pair[1]}")
                else:
                    raise ValueError(
                        f"Malformed inter-robot contact pair: {pair}. "
                        "Each pair must contain exactly two geom names."
                    )
        else:
            # If not clearing, load the original XML directly.
            spec = mujoco.MjSpec.from_file(robot_xml_path)
    else:
        spec = mujoco.MjSpec()
    spec.option.timestep = timestep

    # --- Place robot (if loaded) ---
    torso = spec.worldbody.first_body()
    torso.pos = list(robot_position)

    # === Add a material for the terrain ===
    spec.add_material(name="terrain_material", rgba=[0.5, 0.5, 0.5, 1])
    spec.add_material(name="obstacle_material", rgba=[0.5, 0.5, 0.5, 0.5])

    # === Add a mesh for the testbed obstacles (if needed) ===
    if any("crawl" in row for row in terrain_map):
        spec.add_mesh(
            name="chair_mesh",
            file="../../../sim/terrain/assets/chair.stl",
            # scale=[0.65, 0.65, 0.65],
            scale=[0.62785388127, 0.66818774445, 0.62408581179],
        )

    # === Lighting and visual config ===
    spec.worldbody.add_light(
        pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )

    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]

    spec.visual.global_.azimuth = 160
    spec.visual.global_.elevation = -20
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    spec.add_texture(
        name="skybox_gradient",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )

    # === Initialize base height map and spawn data ===
    base_hmap = np.zeros((nrow, ncol), dtype=np.float32)
    safe_spawns = []

    # === Initialize elevation info structure ===
    elevation_info = {
        "base": {
            "data": None,  # Will be set to base_hmap at the end
            "bounds": (
                -total_width / 2,
                total_width / 2,
                -total_length / 2,
                total_length / 2,
            ),
            "resolution": pixels_per_meter,
        },
        "obstacles": [],
    }

    terrain_geom_names = []
    obstacle_geom_names = []
    skill_map = []

    if all_flat:
        floor_name = generate_groundplane(spec)

        # Safe spawns at centers of each tile
        safe_spawns = [
            (
                -total_width / 2 + (j + 0.5) * tile_width,
                -total_length / 2 + (i + 0.5) * tile_length,
                0.0,
            )
            for i in range(rows)
            for j in range(cols)
        ]

        terrain_geom_names = [floor_name]

        # Zero heightmap for flat terrain (boundary clamping handled in mjx_env.py)
        base_hmap = np.zeros((nrow, ncol), dtype=np.float32)
    else:
        # === Loop through each tile and fill the corresponding patch ===
        for i, row in enumerate(terrain_map):
            for j, terrain in enumerate(row):
                start_row = int(i * tile_length * pixels_per_meter)
                end_row = start_row + int(tile_length * pixels_per_meter)
                start_col = int(j * tile_width * pixels_per_meter)
                end_col = start_col + int(tile_width * pixels_per_meter)

                # Convert tile center to world XY coordinates
                x_center = -total_width / 2 + (j + 0.5) * tile_width
                y_center = -total_length / 2 + (i + 0.5) * tile_length

                # --- Generate the patch ---
                if terrain == "flat":
                    patch = np.zeros(
                        (end_row - start_row, end_col - start_col), dtype=np.float32
                    )

                elif terrain == "bumps":
                    patch, h = generate_bumps_patch(end_row - start_row)

                elif terrain == "rough":
                    patch, h = generate_rough_patch(end_row - start_row)

                elif terrain == "slope":
                    patch, h = generate_slope_patch(end_row - start_row)

                elif terrain == "stairs":
                    patch, h = generate_stairs_patch(end_row - start_row)

                elif terrain == "boxes":
                    patch, h = generate_boxes_patch(end_row - start_row)

                elif terrain == "box":
                    patch, h = generate_box_patch(end_row - start_row)

                    # # Add a box obstacle at the center of the tile
                    # # TODO: Align or merge the box collision geom and mesh geom together, resize box, regenereate mesh for the box as it's not totally flat from Hitem3D
                    box_height = 0.13
                    # box_size = [0.47856409002 / 2, 0.36284246575 / 2, box_height / 2]
                    box_size = [0.62 / 2, 0.64 / 2, box_height / 2]
                    box_geom = spec.worldbody.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=box_size,
                        pos=[x_center, y_center, box_size[2]],
                        material="obstacle_material",
                        name=f"box_obstacle_{i}_{j}",
                        # group=3,
                        group=0,
                        condim=3,
                    )
                    terrain_geom_names.append(box_geom.name)

                    # box_height = 0.13
                    # width = 0.62
                    # length = 0.64
                    # box_size = [
                    #     width / 2,
                    #     length / 2,
                    #     box_height / 2,
                    # ]  # Keep same dimensions

                    # # Create heightfield for box with tunable resolution per meter
                    # box_resolution_per_meter = 32
                    # box_width_pixels = int(width * box_resolution_per_meter)
                    # box_length_pixels = int(length * box_resolution_per_meter)

                    # # Create box heightfield data
                    # box_hfield_data = np.ones(
                    #     (box_length_pixels, box_width_pixels), dtype=np.float32
                    # )

                    # # Add the box heightfield
                    # spec.add_hfield(
                    #     name=f"hfield_box_{i}_{j}",
                    #     size=[
                    #         box_size[0],  # X half-width
                    #         box_size[1],  # Y half-width
                    #         1,
                    #         box_height,  # Base thickness
                    #     ],
                    #     nrow=box_length_pixels,
                    #     ncol=box_width_pixels,
                    #     userdata=box_hfield_data.flatten(),
                    # )

                    # # Add geom for the box heightfield
                    # box_hfield_geom = spec.worldbody.add_geom(
                    #     type=mujoco.mjtGeom.mjGEOM_HFIELD,
                    #     hfieldname=f"hfield_box_{i}_{j}",
                    #     pos=[x_center, y_center, box_height],
                    #     material="obstacle_material",
                    #     name=f"box_hfield_{i}_{j}",
                    #     condim=3,
                    # )
                    # obstacle_geom_names.append(box_hfield_geom.name)

                    # Add box to elevation_info
                    elevation_info["obstacles"].append(
                        {
                            "type": "box",
                            "bounds": {
                                "x": [x_center - box_size[0], x_center + box_size[0]],
                                "y": [y_center - box_size[1], y_center + box_size[1]],
                            },
                            "height": box_height,
                            "base_height": 0.0,
                            "collidable": True,
                        }
                    )

                    # Add skill bounds for box climbing (positioned in front of box)
                    # Box dimensions: width=0.8m (x), length=1.2m (y)
                    # Skill bounds positioned at the edge of the box in negative x direction
                    x_margin = 0.02
                    y_margin = 0.15
                    skill_map.append(
                        {
                            "skill": "climb_up_box",
                            "bounds": {
                                "x": [
                                    x_center - box_size[0] - 0.075 - x_margin,
                                    x_center - box_size[0] - 0.075 + x_margin,
                                ],  # At box edge - 0.1m
                                "y": [
                                    y_center - box_size[1] + y_margin,
                                    y_center + box_size[1] - y_margin,
                                ],  # Box length + margin
                            },
                            "prev_skill_required": "walk",
                        }
                    )

                    crawl_end = 0.12
                    skill_map.append(
                        {
                            "skill": "crawl_box",
                            "bounds": {
                                "x": [
                                    x_center - box_size[0],
                                    x_center + crawl_end,
                                ],
                                "y": [y_center - box_size[1], y_center + box_size[1]],
                            },
                            "prev_skill_required": ["climb_up_box", "crawl_box"],
                        }
                    )

                    skill_map.append(
                        {
                            "skill": "rotate_box",
                            "bounds": {
                                "x": [
                                    x_center + crawl_end,
                                    x_center + box_size[0],
                                ],
                                "y": [y_center - box_size[1], y_center + box_size[1]],
                            },
                            "prev_skill_required": ["crawl_box"],
                        }
                    )

                    # # Add climb_down_box_forward skill at the opposite end (positive x direction)
                    # skill_map.append(
                    #     {
                    #         "skill": "get_up_box",
                    #         "bounds": {
                    #             "x": [
                    #                 x_center + box_size[0] - 0.05 - x_margin,
                    #                 x_center + box_size[0] - 0.05 + x_margin,
                    #             ],  # At box edge + 0.1m
                    #             "y": [
                    #                 y_center - box_size[1] + y_margin,
                    #                 y_center + box_size[1] - y_margin,
                    #             ],  # Box length + margin
                    #         },
                    #         "prev_skill_required": "climb_up_box_crawl",
                    #     }
                    # )

                    # spec.worldbody.add_body(
                    #     name=f"box_{i}_{j}",
                    #     pos=[x_center, y_center, box_size[2]],
                    #     euler=[1.56, -1.56, 0],
                    # ).add_geom(
                    #     type=mujoco.mjtGeom.mjGEOM_MESH,
                    #     # name="box_mesh",
                    #     meshname="box_mesh",
                    #     mass=0,
                    #     group=2,
                    # )
                    # terrain_geom_names.append("box_mesh")

                elif terrain == "up_slope":
                    patch = np.zeros(
                        (end_row - start_row, end_col - start_col), dtype=np.float32
                    )

                    slope_resolution_per_meter = 32
                    slope_peak = 0.13
                    slope_width_pixels = int(tile_width * slope_resolution_per_meter)
                    slope_length_pixels = int(tile_length * slope_resolution_per_meter)
                    slope_hmap, _ = generate_up_slope_patch(
                        slope_length_pixels, slope_peak
                    )

                    # Add the slope heightfield
                    spec.add_hfield(
                        name=f"hfield_up_slope_{i}_{j}",
                        size=[
                            tile_width / 2,  # X half-width
                            tile_length / 2,  # Y half-width
                            slope_peak,
                            0.1,  # Base thickness
                        ],
                        nrow=slope_length_pixels,
                        ncol=slope_width_pixels,
                        userdata=slope_hmap.flatten(),
                    )

                    # Add geom for the slope heightfield
                    base_height = 0.0
                    slope_hfield_geom = spec.worldbody.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_HFIELD,
                        hfieldname=f"hfield_up_slope_{i}_{j}",
                        pos=[x_center, y_center, base_height],
                        material="obstacle_material",
                        name=f"up_slope_hfield_{i}_{j}",
                        condim=3,
                    )
                    obstacle_geom_names.append(slope_hfield_geom.name)

                    # Add slope to elevation_info
                    elevation_info["obstacles"].append(
                        {
                            "type": "up_slope",
                            "bounds": {
                                "x": [
                                    x_center - tile_width / 2,
                                    x_center + tile_width / 2,
                                ],
                                "y": [
                                    y_center - tile_length / 2,
                                    y_center + tile_length / 2,
                                ],
                            },
                            "height": slope_peak,
                            "hmap": slope_hmap,
                            "base_height": base_height,
                            "collidable": True,
                        }
                    )

                elif terrain == "down_slope":
                    patch = np.zeros(
                        (end_row - start_row, end_col - start_col), dtype=np.float32
                    )

                    slope_resolution_per_meter = 32
                    slope_peak = 0.13
                    slope_width_pixels = int(tile_width * slope_resolution_per_meter)
                    slope_length_pixels = int(tile_length * slope_resolution_per_meter)
                    slope_hmap, _ = generate_down_slope_patch(
                        slope_length_pixels, slope_peak
                    )

                    # Add the slope heightfield
                    spec.add_hfield(
                        name=f"hfield_down_slope_{i}_{j}",
                        size=[
                            tile_width / 2,  # X half-width
                            tile_length / 2,  # Y half-width
                            slope_peak,
                            0.1,  # Base thickness
                        ],
                        nrow=slope_length_pixels,
                        ncol=slope_width_pixels,
                        userdata=slope_hmap.flatten(),
                    )

                    # Add geom for the slope heightfield
                    base_height = 0.0
                    slope_hfield_geom = spec.worldbody.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_HFIELD,
                        hfieldname=f"hfield_down_slope_{i}_{j}",
                        pos=[x_center, y_center, base_height],
                        material="obstacle_material",
                        name=f"down_slope_hfield_{i}_{j}",
                        condim=3,
                    )
                    obstacle_geom_names.append(slope_hfield_geom.name)

                    # Add slope to elevation_info
                    elevation_info["obstacles"].append(
                        {
                            "type": "down_slope",
                            "bounds": {
                                "x": [
                                    x_center - tile_width / 2,
                                    x_center + tile_width / 2,
                                ],
                                "y": [
                                    y_center - tile_length / 2,
                                    y_center + tile_length / 2,
                                ],
                            },
                            "height": slope_peak,
                            "hmap": slope_hmap,
                            "base_height": base_height,
                            "collidable": True,
                        }
                    )

                elif terrain == "crawl":
                    patch, h = generate_crawl_patch(end_row - start_row)

                    # Add a chair obstacle at the center of the tile (visual only)
                    spec.worldbody.add_body(
                        name=f"chair_{i}_{j}",
                        pos=[x_center, y_center, 0.435],
                        euler=[1.56, -1.56, 0],
                    ).add_geom(
                        type=mujoco.mjtGeom.mjGEOM_MESH,
                        # name="chair_mesh",
                        meshname="chair_mesh",
                        mass=0,
                        group=2,
                    )
                    # terrain_geom_names.append("chair_mesh")

                    # # Add chair to elevation_info (estimated dimensions for skillmap)
                    # chair_footprint_size = 0.325  # Same as we calculated before
                    # chair_height_for_planning = 0.55  # High enough to be avoided
                    # elevation_info["obstacles"].append(
                    #     {
                    #         "type": "chair",
                    #         "bounds": {
                    #             "x": [
                    #                 x_center - chair_footprint_size / 2,
                    #                 x_center + chair_footprint_size / 2,
                    #             ],
                    #             "y": [
                    #                 y_center - chair_footprint_size / 2,
                    #                 y_center + chair_footprint_size / 2,
                    #             ],
                    #         },
                    #         "height": chair_height_for_planning,
                    #         "base_height": 0.0,
                    #         "collidable": False,  # Visual only, not for collision checking
                    #     }
                    # )

                    wall_height = 0.45
                    wall_size = [0.3, 0.02, wall_height / 2]  # half-extents for MuJoCo
                    wall_geom_left = spec.worldbody.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=wall_size,
                        pos=[
                            x_center,
                            y_center + 0.22,
                            wall_size[2],
                        ],  # place at ground + half height
                        material="obstacle_material",
                        name=f"crawl_wall_obstacle_left_{i}_{j}",
                        group=3,
                        condim=3,
                    )
                    obstacle_geom_names.append(wall_geom_left.name)

                    # Add wall to elevation_info
                    elevation_info["obstacles"].append(
                        {
                            "type": "crawl_wall_left",
                            "bounds": {
                                "x": [
                                    x_center - wall_size[0],
                                    x_center + wall_size[0],
                                ],
                                "y": [
                                    y_center + 0.22 - wall_size[1],
                                    y_center + 0.22 + wall_size[1],
                                ],
                            },
                            "height": wall_height,
                            "base_height": 0.0,
                            "collidable": True,
                        }
                    )

                    wall_geom_right = spec.worldbody.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=wall_size,
                        pos=[
                            x_center,
                            y_center - 0.22 - 0.01,
                            wall_size[2],
                        ],  # place at ground + half height
                        material="obstacle_material",
                        name=f"crawl_wall_obstacle_right_{i}_{j}",
                        group=3,
                        condim=3,
                    )
                    obstacle_geom_names.append(wall_geom_right.name)

                    # Add wall to elevation_info
                    elevation_info["obstacles"].append(
                        {
                            "type": "crawl_wall_right",
                            "bounds": {
                                "x": [
                                    x_center - wall_size[0],
                                    x_center + wall_size[0],
                                ],
                                "y": [
                                    y_center - 0.22 - 0.01 - wall_size[1],
                                    y_center - 0.22 - 0.01 + wall_size[1],
                                ],
                            },
                            "height": wall_height,
                            "base_height": 0.0,
                            "collidable": True,
                        }
                    )

                    # Add skill bounds for crawl sequence
                    x_margin = 0.03
                    y_margin = 0.1
                    crawl_width = (
                        wall_size[0] * 2
                    )  # wall_size[0] is half-width in x direction

                    # 1. get_down skill (approach from negative x side)
                    get_down_lower_x_bound = x_center - crawl_width / 2 - 0.1 - x_margin
                    get_down_upper_x_bound = x_center - crawl_width / 2 - 0.1 + x_margin
                    skill_map.append(
                        {
                            "skill": "get_down_chair",
                            "bounds": {
                                "x": [
                                    get_down_lower_x_bound,
                                    get_down_upper_x_bound,
                                ],
                                "y": [
                                    y_center - 0.1,
                                    y_center + 0.1,
                                ],
                            },
                            "prev_skill_required": "walk",
                        }
                    )

                    get_up_lower_x_bound = x_center + crawl_width / 2 + 0.1 - x_margin
                    # 2. crawl skill (between the walls) - starts right after get_down
                    skill_map.append(
                        {
                            "skill": "crawl_chair",
                            "bounds": {
                                "x": [
                                    get_down_lower_x_bound,  # Start right after get_down
                                    get_up_lower_x_bound,  # End right before get_up
                                ],
                                "y": [
                                    y_center - 0.2,  # Match get_down y bounds
                                    y_center + 0.2,
                                ],
                            },
                            "prev_skill_required": [
                                "get_down",
                                "get_down_and_crawl_chair",
                                "crawl_only_chair_long",
                                "get_down_chair",
                                "crawl_chair",
                            ],
                        }
                    )

                    # 3. get_up skill (exit from positive x side) - starts right after crawl
                    skill_map.append(
                        {
                            "skill": "get_up_chair",
                            "bounds": {
                                "x": [
                                    get_up_lower_x_bound,  # Start right after crawl
                                    x_center
                                    + crawl_width / 2
                                    + 0.2
                                    + x_margin,  # Extended exit area
                                ],
                                "y": [
                                    y_center - 0.2,  # Match crawl y bounds
                                    y_center + 0.2,
                                ],
                            },
                            "prev_skill_required": [
                                "crawl_only_chair_long",
                                "crawl_chair",
                            ],
                        }
                    )

                elif terrain == "full_stairs":
                    # Create full staircase: 4 steps up + 3 steps down (7 steps total)
                    patch = np.zeros(
                        (end_row - start_row, end_col - start_col), dtype=np.float32
                    )

                    # Exact dimensions from scene_mjx_stairs_crawl.xml
                    # Box 0: size="0.0725 0.305 0.02125" pos="0.145 0 0.02125"
                    # Box 1: size="0.0725 0.305 0.02125" pos="0.29  0 0.06375"
                    # Box 2: size="0.0725 0.305 0.02125" pos="0.435 0 0.10625"
                    # Box 3: size="0.0725 0.305 0.02125" pos="0.58  0 0.14875"

                    box_size = [0.0725, 0.305, 0.02125]  # Half-extents for all boxes

                    # Full staircase spans wider: 7 steps * 0.145m = ~1.015m total
                    base_x_offset = (
                        -0.5075 - 0.0725
                    )  # Center the full staircase in the tile

                    # Up stairs (4 steps) - same as original half_stairs
                    up_positions = [0.145, 0.29, 0.435, 0.58]
                    up_heights = [0.02125, 0.06375, 0.10625, 0.14875]

                    for idx, (x_pos, height) in enumerate(
                        zip(up_positions, up_heights)
                    ):
                        box_geom = spec.worldbody.add_geom(
                            type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=box_size,
                            pos=[x_center + base_x_offset + x_pos, y_center, height],
                            material="obstacle_material",
                            name=f"full_stairs_up_{idx}_{i}_{j}",
                            group=0,
                            condim=3,
                        )
                        obstacle_geom_names.append(box_geom.name)

                        # Add to elevation_info
                        elevation_info["obstacles"].append(
                            {
                                "type": f"full_stairs_up_{idx}",
                                "bounds": {
                                    "x": [
                                        x_center + base_x_offset + x_pos - box_size[0],
                                        x_center + base_x_offset + x_pos + box_size[0],
                                    ],
                                    "y": [
                                        y_center - box_size[1],
                                        y_center + box_size[1],
                                    ],
                                },
                                "height": height + box_size[2],  # Top of the box
                                "base_height": 0.0,
                                "collidable": True,
                            }
                        )

                    # Down stairs (3 steps going back down to ground level)
                    down_positions = [
                        0.725,
                        0.87,
                        1.015,
                    ]  # Continue from where up stairs end
                    down_heights = [
                        0.10625,
                        0.06375,
                        0.02125,
                    ]  # Go back down in 3 steps

                    for idx, (x_pos, height) in enumerate(
                        zip(down_positions, down_heights)
                    ):
                        box_geom = spec.worldbody.add_geom(
                            type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=box_size,
                            pos=[x_center + base_x_offset + x_pos, y_center, height],
                            material="obstacle_material",
                            name=f"full_stairs_down_{idx}_{i}_{j}",
                            group=0,
                            condim=3,
                        )
                        obstacle_geom_names.append(box_geom.name)

                        # Add to elevation_info
                        elevation_info["obstacles"].append(
                            {
                                "type": f"full_stairs_down_{idx}",
                                "bounds": {
                                    "x": [
                                        x_center + base_x_offset + x_pos - box_size[0],
                                        x_center + base_x_offset + x_pos + box_size[0],
                                    ],
                                    "y": [
                                        y_center - box_size[1],
                                        y_center + box_size[1],
                                    ],
                                },
                                "height": height + box_size[2],  # Top of the box
                                "base_height": 0.0,
                                "collidable": True,
                            }
                        )

                    skill_map.append(
                        {
                            "skill": "crawl_up_stairs",
                            "bounds": {
                                "x": [
                                    x_center + base_x_offset - 0.03,
                                    x_center + base_x_offset + 0.01,
                                ],
                                "y": [y_center - box_size[1], y_center + box_size[1]],
                            },
                            "prev_skill_required": "walk",
                        }
                    )
                    # skill_map.append(
                    #     {
                    #         "skill": "crawl_down_stairs",
                    #         "bounds": {
                    #             "x": [
                    #                 x_center,
                    #                 x_center
                    #                 + base_x_offset
                    #                 + 0.20
                    #                 + down_positions[-1],
                    #             ],
                    #             "y": [y_center - box_size[1], y_center + box_size[1]],
                    #         },
                    #         "prev_skill_required": [
                    #             "crawl_up_stairs",
                    #             "crawl_down_stairs",
                    #         ],
                    #     }
                    # )
                    # skill_map.append(
                    #     {
                    #         "skill": "get_up_stairs",
                    #         "bounds": {
                    #             "x": [
                    #                 x_center
                    #                 + base_x_offset
                    #                 + 0.20
                    #                 + down_positions[-1],
                    #                 x_center
                    #                 + base_x_offset
                    #                 + 0.20
                    #                 + down_positions[-1]
                    #                 + 0.1,
                    #             ],
                    #             "y": [y_center - box_size[1], y_center + box_size[1]],
                    #         },
                    #         "prev_skill_required": "crawl_down_stairs",
                    #     }
                    # )

                elif terrain == "wall":
                    patch, h = generate_wall_patch(end_row - start_row)

                    # Add a thin wall obstacle at the center of the tile
                    # Based on your example: size="0.025 0.3 0.13" (thin, wide, tall)
                    wall_height = 0.123
                    wall_size = [
                        0.0195,
                        0.303,
                        wall_height / 2,
                    ]  # half-extents for MuJoCo
                    wall_geom = spec.worldbody.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=wall_size,
                        pos=[
                            x_center,
                            y_center,
                            wall_size[2],
                        ],  # place at ground + half height
                        material="obstacle_material",
                        name=f"wall_obstacle_{i}_{j}",
                        group=0,
                        condim=3,
                    )
                    obstacle_geom_names.append(wall_geom.name)

                    # Add wall to elevation_info
                    elevation_info["obstacles"].append(
                        {
                            "type": "wall",
                            "bounds": {
                                "x": [x_center - wall_size[0], x_center + wall_size[0]],
                                "y": [y_center - wall_size[1], y_center + wall_size[1]],
                            },
                            "height": wall_height,
                            "base_height": 0.0,
                            "collidable": True,
                        }
                    )

                    # Add skill bounds for wall climbing (positioned in front of wall)
                    # Wall is at x_center, skill bounds are -0.15 to -0.05 from wall center
                    # Y bounds extend 0.2m beyond wall width on each side
                    wall_width = wall_size[1] * 2  # wall_size[1] is half-width
                    x_margin = 0.02
                    y_margin = 0.15
                    skill_map.append(
                        {
                            "skill": "climb_wall",
                            "bounds": {
                                "x": [
                                    x_center - 0.09 - x_margin,
                                    x_center - 0.09 + x_margin,
                                ],
                                "y": [
                                    y_center - wall_width / 2 + y_margin,
                                    y_center + wall_width / 2 - y_margin,
                                ],
                            },
                            "prev_skill_required": "walk",
                        }
                    )

                else:
                    raise ValueError(f"Unknown terrain type: '{terrain}'")

                base_hmap[start_row:end_row, start_col:end_col] = patch

                # --- Compute safe spawn point at tile center ---
                # TODO: Currently safe spawns assume default pose of the robot.
                if terrain not in [
                    "crawl",
                    "up_slope",
                    "down_slope",
                    "wall",
                    "half_stairs",
                    "box",
                ]:
                    # For most terrains, spawn at tile center at the appropriate height
                    if terrain == "flat":
                        spawn_z = 0.0
                    # elif terrain == "box":
                    #     # Spawn on top of obstacle for box terrains
                    #     spawn_z = 0.13  # box_height
                    else:
                        # For other terrains (stairs, slope, etc), use base_hmap
                        tile_center_row = start_row + (end_row - start_row) // 2
                        tile_center_col = start_col + (end_col - start_col) // 2
                        spawn_z = float(base_hmap[tile_center_row, tile_center_col])

                    safe_spawns.append((x_center, y_center, spawn_z))

        # Testbed mode: use flat groundplane with heightfield bounds for elevation queries
        if is_testbed_mode:
            floor = generate_groundplane(spec)
            terrain_geom_names.append(floor)
        else:
            # MuJoCo's hfield expects data in [0, 1] and scales it by size[2].
            hfield_min = base_hmap.min()
            hfield_max = base_hmap.max()

            hfield_range = hfield_max - hfield_min
            if hfield_range <= 1e-6:
                hfield_range = 1.0  # Avoid division by zero

            relative_hmap = base_hmap - hfield_min
            normalized_hmap = relative_hmap / hfield_range

            # === Define the terrain heightfield ===
            spec.add_hfield(
                name="hfield_terrain",
                size=[
                    total_width / 2,  # X radius
                    total_length / 2,  # Y radius
                    hfield_range,
                    0.1,  # Base Z (thickness)
                ],
                nrow=nrow,
                ncol=ncol,
                userdata=normalized_hmap.flatten(),
            )

            hfield_terrain_geom_name = "hfield_terrain_geom"
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_HFIELD,
                hfieldname="hfield_terrain",
                pos=[0, 0, hfield_min],
                material="terrain_material",
                name=hfield_terrain_geom_name,
                condim=3,
            )

            terrain_geom_names.append(hfield_terrain_geom_name)

    if robot_xml_path:
        if robot_collision_geom_names is None:
            # Extract all geoms used in contact pairs
            robot_collision_geom_names = set()

            for pair in spec.pairs:
                if pair.geomname1:
                    robot_collision_geom_names.add(pair.geomname1)
                if pair.geomname2:
                    robot_collision_geom_names.add(pair.geomname2)

    # Add terrain–robot contact pairs
    print("\n=== TERRAIN COLLISION GEOMS ===")
    for terrain_geom in terrain_geom_names:
        for robot_geom in robot_collision_geom_names:
            spec.add_pair(geomname1=terrain_geom, geomname2=robot_geom)
            # spec.add_pair(geomname1=terrain_geom, geomname2=robot_geom, condim=4)
            print(f"  - {robot_geom} <-> {terrain_geom}")
    print("\n=== OBSTACLE COLLISION GEOMS ===")
    for obstacle_geom in obstacle_geom_names:
        for robot_geom in robot_collision_geom_names:
            spec.add_pair(geomname1=obstacle_geom, geomname2=robot_geom)
            # spec.add_pair(geomname1=obstacle_geom, geomname2=robot_geom, condim=4)
            print(f"  - {robot_geom} <-> {obstacle_geom}")

    # Store base terrain heightfield (excludes obstacles) for elevation queries
    elevation_info["base"]["data"] = base_hmap

    return spec, terrain_geom_names, safe_spawns, elevation_info, skill_map
