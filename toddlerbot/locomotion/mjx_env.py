"""Base MJX environment for ToddlerBot locomotion tasks.

This module provides the core MJXEnv class that serves as the base for all locomotion
environments in ToddlerBot. It handles physics simulation, observation generation,
reward computation, and environment management using MuJoCo and JAX.
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import asdict
from typing import Any

import gin
import jax
import jax.numpy as jnp
import mujoco
import numpy
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax.scipy.spatial.transform import Rotation as R
from mujoco import mjx
from mujoco.mjx._src import support

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.ppo_config import PPOConfig
from toddlerbot.reference.motion_ref import MotionReference
from toddlerbot.sim.motor_control import MotorController
from toddlerbot.sim.robot import Robot
from toddlerbot.sim.terrain.generate_terrain import (
    create_terrain_spec,
)
from toddlerbot.sim.terrain.get_elevation import get_robot_ground_height
from toddlerbot.utils.math_utils import get_local_vec

# from toddlerbot.utils.misc_utils import profile

# Global registry to store env names and their corresponding classes
env_registry: dict[str, type["MJXEnv"]] = {}


def add_contact_limits_to_xml(xml_path: str, max_contacts: int, max_pairs: int) -> str:
    """Add contact limits to XML in memory (does not modify files on disk).

    Returns modified XML string with MuJoCo size element for contact limits.
    """
    with open(xml_path) as f:
        xml_string = f.read()

    root = ET.fromstring(xml_string)

    # Add size element with contact limits
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")

    size.set("nconmax", str(max_contacts))
    size.set("njmax", str(max_pairs * 10))

    return ET.tostring(root, encoding="unicode")


def add_obstacle_copies_to_xml(
    xml_string: str,
    heights: list[float],
    spawn_y_offsets: list[float],
    scene_name: str = "",
    collision_geoms: list[str] | None = None,
) -> str:
    """Add copies of obstacles at different Y positions and heights.

    Auto-detects obstacle type based on scene name:
    - Stairs scenes (scene name contains 'stairs'): Creates stair sets where each
      step has the specified absolute height. The Z positions adjust cumulatively.
    - Other scenes: Creates single obstacles with specified absolute heights.

    Args:
        xml_string: The XML string to modify.
        heights: Absolute heights in meters. For stairs, this is the step height
                 (e.g., [0.0075, 0.025, 0.0425, 0.06, 0.0775] for 0.75-7.75cm steps).
        spawn_y_offsets: List of Y-axis positions for each obstacle variant.
        scene_name: Scene name used to detect obstacle type (e.g., 'scene_mjx_crawl_up_stairs').
        collision_geoms: List of robot geom names that should collide with obstacles.
            If None, uses default collision geoms from original XML.

    Returns:
        Modified XML string with multiple obstacles.
    """
    root = ET.fromstring(xml_string)
    worldbody = root.find("worldbody")
    contact = root.find("contact")

    if worldbody is None:
        return xml_string

    # Detect if this is a stairs scene
    is_stairs = "stairs" in scene_name.lower()

    if is_stairs:
        return _add_stairs_copies_to_xml(
            root, worldbody, contact, heights, spawn_y_offsets
        )
    else:
        return _add_single_obstacle_copies_to_xml(
            root, worldbody, contact, heights, spawn_y_offsets
        )


def _add_single_obstacle_copies_to_xml(
    root: ET.Element,
    worldbody: ET.Element,
    contact: ET.Element | None,
    heights: list[float],
    spawn_y_offsets: list[float],
) -> str:
    """Add copies of a single obstacle (box/wall) at different Y positions."""
    original_box = worldbody.find(".//geom[@name='box']")
    if original_box is None:
        return ET.tostring(root, encoding="unicode")

    # Get original box properties
    original_size = original_box.get("size", "0.0195 0.303 0.123").split()
    original_pos = original_box.get("pos", "0.1 0 0").split()
    box_x = float(original_pos[0])
    box_half_width = float(original_size[0])
    box_half_length = float(original_size[1])

    # Collect collision geoms from original contact pairs
    original_collision_geoms = set()
    if contact is not None:
        for pair in contact.findall("pair"):
            geom1 = pair.get("geom1")
            geom2 = pair.get("geom2")
            if geom1 == "box" and geom2:
                original_collision_geoms.add(geom2)
            elif geom2 == "box" and geom1:
                original_collision_geoms.add(geom1)

    # Remove original box and its contact pairs
    worldbody.remove(original_box)
    if contact is not None:
        pairs_to_remove = []
        for pair in contact.findall("pair"):
            if pair.get("geom1") == "box" or pair.get("geom2") == "box":
                pairs_to_remove.append(pair)
        for pair in pairs_to_remove:
            contact.remove(pair)

    # Create new obstacles at each Y position
    for i, (height, y_offset) in enumerate(zip(heights, spawn_y_offsets)):
        half_height = height / 2.0
        geom_name = f"box_{i}"

        # Create new geom
        new_geom = ET.SubElement(worldbody, "geom")
        new_geom.set("name", geom_name)
        new_geom.set("type", "box")
        new_geom.set("size", f"{box_half_width} {box_half_length} {half_height}")
        new_geom.set("pos", f"{box_x} {y_offset} {half_height}")
        new_geom.set("condim", "3")

        # Add contact pairs for this obstacle
        if contact is not None:
            for collision_geom in original_collision_geoms:
                pair = ET.SubElement(contact, "pair")
                pair.set("geom1", geom_name)
                pair.set("geom2", collision_geom)

    return ET.tostring(root, encoding="unicode")


def _add_stairs_copies_to_xml(
    root: ET.Element,
    worldbody: ET.Element,
    contact: ET.Element | None,
    step_heights: list[float],
    spawn_y_offsets: list[float],
) -> str:
    """Add copies of stairs at different Y positions with varying step heights.

    Args:
        step_heights: Absolute step heights in meters (e.g., [0.0425, 0.0525, 0.0625, 0.0775]).
        spawn_y_offsets: Y positions for each stair variant.

    For stairs, the height delta from original is computed, then:
    - Each step's half-height increases by delta/2
    - Each step's Z position increases by delta/2 + (step_index × delta) to stack correctly
    """
    # Find all original stair step geoms (box, box1, box2, box3)
    step_names = ["box", "box1", "box2", "box3"]
    original_steps = []
    for name in step_names:
        step = worldbody.find(f".//geom[@name='{name}']")
        if step is not None:
            original_steps.append(step)

    if len(original_steps) == 0:
        return ET.tostring(root, encoding="unicode")

    # Parse original step properties
    step_props = []
    for step in original_steps:
        size = [float(x) for x in step.get("size").split()]
        pos = [float(x) for x in step.get("pos").split()]
        step_props.append({"size": size, "pos": pos})

    # Get original step height from first step (half-height * 2)
    original_step_height = step_props[0]["size"][2] * 2.0

    # Collect collision geoms from original contact pairs
    collision_geoms = set()
    actual_step_names = [s.get("name") for s in original_steps]
    if contact is not None:
        for pair in contact.findall("pair"):
            geom1 = pair.get("geom1")
            geom2 = pair.get("geom2")
            if geom1 in actual_step_names and geom2:
                collision_geoms.add(geom2)
            elif geom2 in actual_step_names and geom1:
                collision_geoms.add(geom1)

    # Remove original steps and their contact pairs
    for step in original_steps:
        worldbody.remove(step)
    if contact is not None:
        pairs_to_remove = []
        for pair in contact.findall("pair"):
            if (
                pair.get("geom1") in actual_step_names
                or pair.get("geom2") in actual_step_names
            ):
                pairs_to_remove.append(pair)
        for pair in pairs_to_remove:
            contact.remove(pair)

    # Create stair sets at each Y position with corresponding step height
    for variant_idx, (step_height, y_offset) in enumerate(
        zip(step_heights, spawn_y_offsets)
    ):
        # Compute delta from original step height
        delta = step_height - original_step_height

        for step_idx, props in enumerate(step_props):
            geom_name = (
                f"box{step_idx}_{variant_idx}" if step_idx > 0 else f"box_{variant_idx}"
            )

            # Calculate new size: half-height increases by delta/2
            new_half_height = props["size"][2] + delta / 2.0
            new_size = f"{props['size'][0]} {props['size'][1]} {new_half_height}"

            # Calculate new position:
            # - X stays the same
            # - Y shifts by spawn_y_offset
            # - Z increases: own height center shift (delta/2) + cumulative from previous steps
            new_z = props["pos"][2] + delta / 2.0 + step_idx * delta
            new_pos = f"{props['pos'][0]} {y_offset} {new_z}"

            # Create new geom
            new_geom = ET.SubElement(worldbody, "geom")
            new_geom.set("name", geom_name)
            new_geom.set("type", "box")
            new_geom.set("size", new_size)
            new_geom.set("pos", new_pos)
            new_geom.set("condim", "3")

            # Add contact pairs
            if contact is not None:
                for collision_geom in collision_geoms:
                    pair = ET.SubElement(contact, "pair")
                    pair.set("geom1", geom_name)
                    pair.set("geom2", collision_geom)

    return ET.tostring(root, encoding="unicode")


def get_env_config(env: str):
    """Retrieves and parses the configuration for a specified environment.

    Args:
        env (str): The name of the environment for which to retrieve the configuration.

    Returns:
        MJXConfig: An instance of MJXConfig initialized with the parsed configuration.

    Raises:
        FileNotFoundError: If the configuration file for the specified environment does not exist.
    """
    gin_file_path = os.path.join(os.path.dirname(__file__), env + ".gin")
    if not os.path.exists(gin_file_path):
        raise FileNotFoundError(f"File {gin_file_path} not found.")

    gin.parse_config_file(gin_file_path)
    return MJXConfig(), PPOConfig()


def get_env_class(env_name: str) -> type["MJXEnv"]:
    """Returns the environment class associated with the given environment name.

    Args:
        env_name (str): The name of the environment to retrieve.

    Returns:
        Type[MJXEnv]: The class of the specified environment.

    Raises:
        ValueError: If the environment name is not found in the registry.
    """
    if env_name not in env_registry:
        raise ValueError(f"Unknown env: {env_name}")

    return env_registry[env_name]


class MJXEnv(PipelineEnv):
    """Base MuJoCo-JAX environment for ToddlerBot locomotion tasks."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        motion_ref: MotionReference,
        fixed_base: bool = False,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        """Initializes the environment with the specified configuration and robot parameters.

        Args:
            name (str): The name of the environment.
            robot (Robot): The robot instance to be used in the environment.
            cfg (MJXConfig): Configuration settings for the environment and simulation.
            motion_ref (MotionReference): Reference for motion planning and execution.
            fixed_base (bool, optional): Whether the robot has a fixed base. Defaults to False.
            add_domain_rand (bool, optional): Whether to add domain randomization. Defaults to True.
            **kwargs (Any): Additional keyword arguments for environment initialization.
        """
        self.name = name
        self.cfg = cfg
        self.robot = robot
        self.motion_ref = motion_ref
        self.fixed_base = fixed_base
        self.add_domain_rand = add_domain_rand
        # Automatically configure episode length based on motion reference, falls back to gin file if not available
        try:
            self.episode_length = max(
                motion_ref.n_frames, get_env_config(name)[-1].episode_length
            )
        except AttributeError:
            self.episode_length = get_env_config(name)[-1].episode_length

        # Get frame type from motion reference
        self.is_robot_relative_frame = getattr(
            motion_ref, "is_robot_relative_frame", False
        )
        # No need to do relative frame conversion for fixed base
        if self.fixed_base:
            self.is_robot_relative_frame = False

        description_dir = os.path.join("toddlerbot", "descriptions")

        if not self.fixed_base and self.cfg.terrain.scene is None:
            xml_path = os.path.join(
                description_dir, robot.name, f"{robot.name}_mjx.xml"
            )
            # === 1. Terrain Setup ===
            terrain_cfg = self.cfg.terrain
            terrain_map = terrain_cfg.manual_map

            spec, _, self.safe_spawns, self.elevation_info, _ = create_terrain_spec(
                tile_width=terrain_cfg.tile_width,
                tile_length=terrain_cfg.tile_length,
                terrain_map=terrain_map,
                robot_xml_path=xml_path,
                timestep=self.cfg.sim.timestep,
                pixels_per_meter=terrain_cfg.resolution_per_meter,
                robot_collision_geom_names=terrain_cfg.robot_collision_geom_names,
                self_contact_pairs=self.cfg.sim.self_contact_pairs,
            )
            self.random_spawn = terrain_cfg.random_spawn

            rows, cols = len(terrain_map), len(terrain_map[0])
            self.total_width = cols * terrain_cfg.tile_width
            self.total_length = rows * terrain_cfg.tile_length

            # === 2. Camera Setup ===
            camera_configs = [
                ("perspective", [0.7, -0.7, 0.7], [1, 1, 0, -1, 1, 3]),
                ("side", [0, -1.0, 0.6], [1, 0, 0, 0, 1, 3]),
                ("top", [0, 0, 1.0], [0, 1, 0, -1, 0, 0]),
                ("front", [1.0, 0, 0.6], [0, 1, 0, -1, 0, 3]),
            ]

            for name, pos, xyaxes in camera_configs:
                spec.worldbody.add_camera(
                    name=name,
                    pos=pos,
                    xyaxes=xyaxes,
                    mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
                )

            # === 3. Compile MJCF model ===
            mj_model = spec.compile()
            sys = mjcf.load_model(mj_model)
        elif not self.fixed_base and self.cfg.terrain.scene is not None:
            # Use pre-defined scene XML file
            xml_path = os.path.join(
                description_dir, robot.name, f"{self.cfg.terrain.scene}.xml"
            )

            # Check if we need to modify the XML (contact limits or obstacle randomization)
            needs_xml_modification = (
                hasattr(self.cfg.sim, "max_contact_points")
                and self.cfg.sim.max_contact_points is not None
                and self.cfg.sim.max_geom_pairs is not None
            ) or self.cfg.obstacle_rand.enabled

            if needs_xml_modification:
                # Read the XML file
                with open(xml_path) as f:
                    modified_xml_string = f.read()

                # Add contact limits if configured
                if (
                    hasattr(self.cfg.sim, "max_contact_points")
                    and self.cfg.sim.max_contact_points is not None
                    and self.cfg.sim.max_geom_pairs is not None
                ):
                    modified_xml_string = add_contact_limits_to_xml(
                        xml_path,
                        self.cfg.sim.max_contact_points,
                        self.cfg.sim.max_geom_pairs,
                    )

                # Add obstacle copies at different Y positions if enabled
                if self.cfg.obstacle_rand.enabled:
                    modified_xml_string = add_obstacle_copies_to_xml(
                        modified_xml_string,
                        self.cfg.obstacle_rand.heights,
                        self.cfg.obstacle_rand.spawn_y_offsets,
                        scene_name=self.cfg.terrain.scene,
                    )

                # Change to XML directory to resolve include paths, then load
                original_dir = os.getcwd()
                xml_dir = os.path.dirname(xml_path)
                os.chdir(xml_dir)
                try:
                    mj_model = mujoco.MjModel.from_xml_string(modified_xml_string)
                    sys = mjcf.load_model(mj_model)
                finally:
                    os.chdir(original_dir)
            else:
                sys = mjcf.load(xml_path)
        else:
            xml_suffix = ""
            if fixed_base:
                xml_suffix += "_fixed"
            xml_path = os.path.join(
                description_dir, robot.name, f"scene_mjx{xml_suffix}.xml"
            )
            # Add contact limits if configured (both must be specified)
            if (
                hasattr(self.cfg.sim, "max_contact_points")
                and self.cfg.sim.max_contact_points is not None
                and self.cfg.sim.max_geom_pairs is not None
            ):
                modified_xml_string = add_contact_limits_to_xml(
                    xml_path,
                    self.cfg.sim.max_contact_points,
                    self.cfg.sim.max_geom_pairs,
                )
                # Change to XML directory to resolve include paths, then load
                original_dir = os.getcwd()
                xml_dir = os.path.dirname(xml_path)
                os.chdir(xml_dir)
                try:
                    mj_model = mujoco.MjModel.from_xml_string(modified_xml_string)
                    sys = mjcf.load_model(mj_model)
                finally:
                    os.chdir(original_dir)
            else:
                sys = mjcf.load(xml_path)

        sys = sys.tree_replace(
            {"opt.timestep": cfg.sim.timestep, "opt.solver": cfg.sim.solver}
        )

        kwargs["n_frames"] = cfg.action.n_frames
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._init_env()
        self._init_reward()

    # Automatic registration of subclasses
    def __init_subclass__(cls, env_name: str = "", **kwargs):
        """Initializes a subclass and optionally registers it in the environment registry.

        Args:
            env_name (str): The name of the environment to register the subclass under. If provided, the subclass is added to the `env_registry` with this name.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        super().__init_subclass__(**kwargs)
        if len(env_name) > 0:
            env_registry[env_name] = cls

    def _init_env(self) -> None:
        """Initializes the environment by setting up various system parameters, colliders, joint indices, motor indices, actuator indices, and action configurations.

        This method configures the environment based on the system and robot specifications, including the number of joints, colliders, and actuators. It identifies and categorizes joint and motor indices for different body parts such as legs, arms, neck, and waist. It also sets up action masks, default actions, and noise scales for the simulation. Additionally, it configures filters and command parameters for controlling the robot's movements and interactions within the environment.
        """
        self.nu = self.sys.nu
        self.nq = self.sys.nq
        self.nv = self.sys.nv

        self.q_start_idx = 0 if self.fixed_base else 7
        self.qd_start_idx = 0 if self.fixed_base else 6

        # Store random initial state indices for use during reset
        # If empty list [], sample uniformly from all frames
        if len(self.cfg.domain_rand.rand_init_state_indices) > 0:
            # Use specified frame indices (e.g., [0] for no randomization)
            self.rand_init_state_indices = jnp.array(
                self.cfg.domain_rand.rand_init_state_indices
            )
        else:
            # Get total number of frames from motion reference
            if hasattr(self.motion_ref, "n_frames"):
                num_frames = self.motion_ref.n_frames
            else:
                num_frames = 1  # No motion reference

            # Uniform random sampling from all frames
            self.rand_init_state_indices = jnp.arange(num_frames)

        # colliders
        pair_geom1 = self.sys.pair_geom1
        pair_geom2 = self.sys.pair_geom2
        self.collider_geom_ids = numpy.unique(
            numpy.concatenate([pair_geom1, pair_geom2])
        )
        self.num_colliders = self.collider_geom_ids.shape[0]
        left_foot_collider_indices: list[int] = []
        right_foot_collider_indices: list[int] = []
        for i, geom_id in enumerate(self.collider_geom_ids):
            geom_name = support.id2name(self.sys, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name is None:
                continue

            if f"left_{self.robot.foot_name}" in geom_name:
                left_foot_collider_indices.append(i)
            elif f"right_{self.robot.foot_name}" in geom_name:
                right_foot_collider_indices.append(i)

        self.left_foot_collider_indices = jnp.array(left_foot_collider_indices)
        self.right_foot_collider_indices = jnp.array(right_foot_collider_indices)

        feet_link_mask = jnp.array(
            numpy.char.find(self.sys.link_names, self.robot.foot_name) >= 0
        )
        self.feet_link_ids = jnp.arange(self.sys.num_links())[feet_link_mask]

        self.contact_force_threshold = self.cfg.action.contact_force_threshold

        # This leads to CPU memory leak
        # self.jit_contact_force = jax.jit(support.contact_force, static_argnums=(2, 3))
        self.jit_contact_force = support.contact_force

        self.joint_indices = jnp.array(
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.joint_ordering
            ]
        )
        if not self.fixed_base:
            # Disregard the free joint
            self.joint_indices -= 1

        joint_groups = numpy.array(self.robot.joint_groups)
        self.leg_joint_indices = self.joint_indices[joint_groups == "leg"]
        self.arm_joint_indices = self.joint_indices[joint_groups == "arm"]
        self.neck_joint_indices = self.joint_indices[joint_groups == "neck"]
        self.waist_joint_indices = self.joint_indices[joint_groups == "waist"]

        hip_pitch_joint_mask = (
            numpy.char.find(self.robot.joint_ordering, "hip_pitch") >= 0
        )
        hip_yaw_joint_mask = numpy.char.find(self.robot.joint_ordering, "hip_yaw") >= 0
        knee_joint_mask = numpy.char.find(self.robot.joint_ordering, "knee") >= 0
        ank_pitch_joint_mask = (
            numpy.char.find(self.robot.joint_ordering, "ankle_pitch") >= 0
        )

        self.hip_pitch_joint_indices = self.joint_indices[hip_pitch_joint_mask]
        self.hip_yaw_joint_indices = self.joint_indices[hip_yaw_joint_mask]
        self.knee_joint_indices = self.joint_indices[knee_joint_mask]
        self.ank_pitch_joint_indices = self.joint_indices[ank_pitch_joint_mask]

        self.motor_indices = jnp.array(
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.motor_ordering
            ]
        )
        if not self.fixed_base:
            # Disregard the free joint
            self.motor_indices -= 1

        motor_groups = numpy.array(self.robot.motor_groups)
        self.leg_motor_indices = self.motor_indices[motor_groups == "leg"]
        self.arm_motor_indices = self.motor_indices[motor_groups == "arm"]
        self.neck_motor_indices = self.motor_indices[motor_groups == "neck"]
        self.waist_motor_indices = self.motor_indices[motor_groups == "waist"]

        self.hip_motor_indices = self.motor_indices[
            numpy.char.find(self.robot.motor_ordering, "hip") >= 0
        ]

        self.actuator_indices = jnp.arange(len(self.robot.motor_ordering))
        self.leg_actuator_indices = self.actuator_indices[motor_groups == "leg"]
        self.arm_actuator_indices = self.actuator_indices[motor_groups == "arm"]
        self.neck_actuator_indices = self.actuator_indices[motor_groups == "neck"]
        self.waist_actuator_indices = self.actuator_indices[motor_groups == "waist"]

        self.neck_passive_dof_indices = jnp.array(
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.neck_passive_dof_names
            ]
        )
        if not self.fixed_base:
            # Disregard the free joint
            self.neck_passive_dof_indices -= 1

        # default qpos
        self.default_qpos = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        self.default_motor_pos = self.default_qpos[
            self.q_start_idx + self.motor_indices
        ]
        self.default_joint_pos = self.default_qpos[
            self.q_start_idx + self.joint_indices
        ]

        # action
        self.action_parts = self.cfg.action.action_parts

        action_mask: list[jax.Array] = []
        for part_name in self.action_parts:
            if part_name == "neck":
                action_mask.append(self.neck_actuator_indices)
            elif part_name == "waist":
                action_mask.append(self.waist_actuator_indices)
            elif part_name == "leg":
                action_mask.append(self.leg_actuator_indices)
            elif part_name == "arm":
                action_mask.append(self.arm_actuator_indices)

        self.action_mask = jnp.sort(jnp.concatenate(action_mask))
        self.num_action = self.action_mask.shape[0]

        self.motor_limits = jnp.array(list(self.robot.motor_limits.values()))
        self.joint_limits = jnp.array(list(self.robot.joint_limits.values()))
        self.action_limits = self.motor_limits[self.action_mask]

        self.action_scale = self.cfg.action.action_scale
        self.n_steps_delay = self.cfg.action.n_steps_delay

        self.leg_pitch_actuator_indices = self.leg_actuator_indices[
            jnp.array([0, 3, 5, 6, 9, 11])
        ]
        self.leg_pitch_joint_signs = jnp.array([-1, 1, -1, 1, -1, 1])

        self.controller = MotorController(self.robot)

        # commands
        # x vel, y vel, yaw vel, heading
        self.resample_time = self.cfg.commands.resample_time
        self.resample_steps = int(self.resample_time / self.dt)
        self.zero_chance = self.cfg.commands.zero_chance
        self.turn_chance = self.cfg.commands.turn_chance
        self.command_obs_indices = jnp.array(self.cfg.commands.command_obs_indices)
        self.command_range = jnp.array(self.cfg.commands.command_range)
        self.deadzone = (
            jnp.array(self.cfg.commands.deadzone)
            if len(self.cfg.commands.deadzone) > 1
            else (
                self.cfg.commands.deadzone[0]
                if len(self.cfg.commands.deadzone) == 1
                else 0.0
            )
        )
        # observation
        self.ref_start_idx = 7 + 6
        self.num_obs_history = self.cfg.obs.frame_stack
        self.num_privileged_obs_history = self.cfg.obs.c_frame_stack
        self.use_phase_signal = self.cfg.obs.use_phase_signal

        # Verify motion reference has motor_vel if not using phase signal
        if not self.use_phase_signal and self.motion_ref is not None:
            # Check if motion reference has actual motor_vel data (not just zeros fallback)
            if (
                hasattr(self.motion_ref, "has_motor_vel")
                and not self.motion_ref.has_motor_vel
            ):
                raise ValueError(
                    "Motion reference does not have 'motor_vel' data but use_phase_signal=False. "
                    "Either set use_phase_signal=True or regenerate the motion file to include motor_vel data."
                )

        self.obs_size = self.cfg.obs.num_single_obs
        self.privileged_obs_size = self.cfg.obs.num_single_privileged_obs
        self.obs_scales = self.cfg.obs_scales

        if self.robot.has_gripper:
            self.obs_size += 4
            self.privileged_obs_size += 4

        self.backlash_range = self.cfg.domain_rand.backlash_range
        self.backlash_activation = self.cfg.domain_rand.backlash_activation
        self.torso_roll_range = self.cfg.domain_rand.torso_roll_range
        self.torso_pitch_range = self.cfg.domain_rand.torso_pitch_range
        self.persistent_torso_pitch_error_range = (
            self.cfg.domain_rand.persistent_torso_pitch_error_range
        )
        self.persistent_torso_roll_error_range = (
            self.cfg.domain_rand.persistent_torso_roll_error_range
        )
        self.arm_joint_pos_range = self.cfg.domain_rand.arm_joint_pos_range
        self.add_head_pose = self.cfg.domain_rand.add_head_pose
        self.kp_range = self.cfg.domain_rand.kp_range
        self.kd_range = self.cfg.domain_rand.kd_range
        self.tau_max_range = self.cfg.domain_rand.tau_max_range
        self.q_dot_tau_max_range = self.cfg.domain_rand.q_dot_tau_max_range
        self.q_dot_max_range = self.cfg.domain_rand.q_dot_max_range
        self.kd_min_range = self.cfg.domain_rand.kd_min_range
        self.tau_brake_max_range = self.cfg.domain_rand.tau_brake_max_range
        self.tau_q_dot_max_range = self.cfg.domain_rand.tau_q_dot_max_range
        self.passive_active_ratio_range = (
            self.cfg.domain_rand.passive_active_ratio_range
        )

        self.add_push = self.cfg.domain_rand.add_push
        self.push_interval = numpy.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.push_duration = numpy.ceil(self.cfg.domain_rand.push_duration_s / self.dt)
        self.push_torso_range = self.cfg.domain_rand.push_torso_range
        self.push_other_range = self.cfg.domain_rand.push_other_range
        self.rand_pos_offset = self.cfg.domain_rand.rand_pos_offset
        self.rand_yaw_offset = self.cfg.domain_rand.rand_yaw_offset

        self.site_indices = {
            support.id2name(self.sys, mujoco.mjtObj.mjOBJ_SITE, i): i
            for i in range(self.sys.nsite)
        }

        # Key end-effector site indices from robot model
        self.end_effector_site_indices = {
            "left_foot_center": self.site_indices.get("left_foot_center", -1),
            "right_foot_center": self.site_indices.get("right_foot_center", -1),
            "left_hand_center": self.site_indices.get("left_hand_center", -1),
            "right_hand_center": self.site_indices.get("right_hand_center", -1),
        }
        # Mapping from recorded site indices (0-3) to site names
        # Based on the site_pose data structure: 4 sites with [pos, quat] each
        self.recorded_site_mapping = {
            0: "left_hand_center",  # Site 0: left hand
            1: "left_foot_center",  # Site 1: left foot
            2: "right_hand_center",  # Site 2: right hand
            3: "right_foot_center",  # Site 3: right foot
        }
        # TODO: clean this up, remove the two above, vectorize the code
        # self.recorded_site_indices = []
        # for recorded_idx, site_name in self.recorded_site_mapping.items():
        #     robot_site_idx = self.end_effector_site_indices.get(site_name, -1)
        #     self.recorded_site_indices.append(robot_site_idx)

        # self.recorded_site_indices = jnp.array(self.recorded_site_indices)

        # print(f"Initialized site indices: {self.end_effector_site_indices}")
        # print(f"Recorded site mapping: {self.recorded_site_mapping}")

    def _init_reward(self) -> None:
        """Initializes the reward system by filtering and scaling reward components.

        This method processes the reward scales configuration by removing any components with a scale of zero and scaling the remaining components by a time factor. It then prepares a list of reward function names and their corresponding scales, which are stored for later use in reward computation. Additionally, it sets parameters related to health and tracking rewards.
        """
        self.healthy_z_range = self.cfg.rewards.healthy_z_range
        self.pos_tracking_sigma = self.cfg.rewards.pos_tracking_sigma
        self.rot_tracking_sigma = self.cfg.rewards.rot_tracking_sigma
        self.lin_vel_tracking_sigma = self.cfg.rewards.lin_vel_tracking_sigma
        self.ang_vel_tracking_sigma = self.cfg.rewards.ang_vel_tracking_sigma
        self.add_regularization = self.cfg.rewards.add_regularization
        self.use_exp_reward = self.cfg.rewards.use_exp_reward
        # Pose weights: if None, use uniform weights of 1.0 for controlled joints
        if self.cfg.rewards.pose_weights is not None:
            self.pose_weights = jnp.array(self.cfg.rewards.pose_weights)
        else:
            self.pose_weights = jnp.ones(self.num_action)
        # Close feet penalty threshold
        self.close_feet_threshold = self.cfg.rewards.close_feet_threshold

        reward_scale_dict = asdict(self.cfg.reward_scales)
        # Remove zero scales and multiply non-zero ones by dt
        for key in list(reward_scale_dict.keys()):
            if reward_scale_dict[key] == 0:
                reward_scale_dict.pop(key)

        # prepare list of functions
        self.reward_names = []
        self.reward_functions = []
        reward_scales = []
        for _i, (name, scale) in enumerate(reward_scale_dict.items()):
            if not self.add_regularization and any(
                k in name for k in ["action_rate", "action_acc", "energy", "torque"]
            ):
                continue

            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))
            reward_scales.append(scale)

        self.reward_scales = reward_scales

    @property
    def action_size(self) -> int:  # override default action_size
        """Returns the number of possible actions.

        Overrides the default action size to provide the specific number of actions available.

        Returns:
            int: The number of possible actions.
        """
        return self.num_action

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment state and initializes various components for a new episode.

        This function splits the input random number generator (RNG) into multiple streams for different components, initializes the state information dictionary, and sets up the initial positions, velocities, and commands for the environment. It also applies domain randomization if enabled and prepares observation histories.

        Args:
            rng (jax.Array): The random number generator state for initializing the environment.

        Returns:
            State: The initialized state of the environment, including pipeline state, observations, rewards, and other relevant information.
        """
        (
            rng,
            rng_spawn,
            rng_roll,
            rng_pitch,
            # rng_yaw,
            rng_neck,
            rng_arm,
            rng_backlash,
            rng_command,
            rng_kp,
            rng_kd,
            rng_tau_max,
            rng_q_dot_tau_max,
            rng_q_dot_max,
            rng_kd_min,
            rng_tau_brake_max,
            rng_tau_q_dot_max,
            rng_pos_offset,
            rng_quat_offset,
            rng_persistent_pitch,
            rng_persistent_roll,
            rng_obstacle,
        ) = jax.random.split(rng, 21)

        state_info = {
            "rng": rng,
            "contact_forces": jnp.zeros((self.num_colliders, self.num_colliders, 3)),
            "num_contact_points": jnp.zeros((self.num_colliders, self.num_colliders)),
            "feet_air_time": jnp.zeros(2),
            "feet_air_dist": jnp.zeros(2),
            "action_buffer": jnp.zeros((self.n_steps_delay + 1) * self.num_action),
            "last_act": jnp.zeros(self.num_action),
            "rewards": dict.fromkeys(self.reward_names, 0.0),
            "imu_state": (jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3)),
            "torso_height_max": 0,
            "push_step": 0,
            "push_remaining": 0,
            "push_id": 1,
            "push": jnp.zeros(2),
            "done": False,
            "step": 0,
            "global_step": 0,
            "spawn_y_offset": 0.0,
            "spawn_z_offset": 0.0,
        }

        state_ref = self.motion_ref.get_default_state()
        command = self._sample_command(rng_command)

        # Get initial state index
        if len(self.rand_init_state_indices) > 1:
            random_init_idx = self.rand_init_state_indices[
                jax.random.randint(rng, (), 0, len(self.rand_init_state_indices))
            ]
            state_info["init_idx"] = random_init_idx
        else:
            state_info["init_idx"] = self.rand_init_state_indices[0]

        # Get state reference with the correct initial index
        state_ref = self.motion_ref.get_state_ref(
            0.0, command, state_ref, state_info["init_idx"]
        )
        qpos = state_ref["qpos"]

        # Store the original position before applying random offsets
        original_pos = qpos[:3].copy() if not self.fixed_base else None

        motor_pos_old = qpos[self.q_start_idx + self.motor_indices]
        joint_pos_old = qpos[self.q_start_idx + self.joint_indices]
        motor_pos_offset = jnp.zeros(self.nu, dtype=jnp.float32)
        joint_pos_offset = jnp.zeros(self.nu, dtype=jnp.float32)

        # Default neck motor target (from reference pose); may be overwritten by randomization
        state_info["neck_motor_target"] = motor_pos_old[self.neck_actuator_indices]

        if self.add_domain_rand and self.add_head_pose:
            # Neck yaw: ±75° (half of full range), pitch: full range
            # Sample absolute positions (not offsets from default pose)
            rng_neck_yaw, rng_neck_pitch = jax.random.split(rng_neck)
            neck_yaw = jax.random.uniform(
                rng_neck_yaw, (1,), minval=-1.309, maxval=1.309
            )  # ±75°
            neck_pitch = jax.random.uniform(
                rng_neck_pitch,
                (1,),
                minval=self.joint_limits[self.neck_actuator_indices][1, 0],
                maxval=self.joint_limits[self.neck_actuator_indices][1, 1],
            )
            # Absolute target positions for neck
            neck_joint_pos = jnp.concatenate([neck_yaw, neck_pitch])
            neck_motor_pos = self.robot.neck_ik(neck_joint_pos)
            # Compute offset from current pose for qpos initialization
            joint_pos_offset = joint_pos_offset.at[self.neck_actuator_indices].set(
                neck_joint_pos - joint_pos_old[self.neck_actuator_indices]
            )
            motor_pos_offset = motor_pos_offset.at[self.neck_actuator_indices].set(
                neck_motor_pos - motor_pos_old[self.neck_actuator_indices]
            )
            # Store absolute position for use as motor target during step
            # (zero_point_offset for neck is used as absolute target, not offset)
            state_info["neck_motor_target"] = neck_motor_pos
            qpos = qpos.at[self.q_start_idx + self.neck_passive_dof_indices].set(
                jnp.repeat(-neck_joint_pos[1:], len(self.neck_passive_dof_indices))
            )

        if not self.fixed_base and self.add_domain_rand:
            torso_roll = jax.random.uniform(
                rng_roll,
                (1,),
                minval=self.torso_roll_range[0],
                maxval=self.torso_roll_range[1],
            )
            waist_joint_pos = jnp.concatenate([-torso_roll, jnp.zeros(1)])
            waist_motor_pos = self.robot.waist_ik(waist_joint_pos)
            joint_pos_offset = joint_pos_offset.at[self.waist_actuator_indices].set(
                waist_joint_pos
            )
            motor_pos_offset = motor_pos_offset.at[self.waist_actuator_indices].set(
                waist_motor_pos
            )

            torso_pitch = jax.random.uniform(
                rng_pitch,
                (1,),
                minval=self.torso_pitch_range[0],
                maxval=self.torso_pitch_range[1],
            )

            _, rng_hip_pitch, rng_knee_pitch = jax.random.split(rng_pitch, 3)
            hip_pitch_delta = jax.random.uniform(
                rng_hip_pitch, (1,), minval=0.0, maxval=jnp.abs(torso_pitch)
            )
            knee_pitch_delta = jax.random.uniform(
                rng_knee_pitch,
                (1,),
                minval=0.0,
                maxval=jnp.abs(torso_pitch) - hip_pitch_delta,
            )
            ankle_pitch_delta = (
                jnp.abs(torso_pitch) - hip_pitch_delta - knee_pitch_delta
            )
            leg_pitch_delta = jnp.concatenate(
                [
                    hip_pitch_delta,
                    knee_pitch_delta,
                    ankle_pitch_delta,
                    hip_pitch_delta,
                    knee_pitch_delta,
                    ankle_pitch_delta,
                ]
            )
            leg_pitch_joint_delta = (
                leg_pitch_delta * self.leg_pitch_joint_signs * jnp.sign(torso_pitch)
            )

            joint_pos_offset = joint_pos_offset.at[self.leg_pitch_actuator_indices].set(
                leg_pitch_joint_delta
            )
            # By default, leg_motor_pos is the same as leg_joint_pos
            motor_pos_offset = motor_pos_offset.at[self.leg_pitch_actuator_indices].set(
                leg_pitch_joint_delta
            )

            # torso_yaw = jax.random.uniform(rng_yaw, (1,), minval=0.0, maxval=2 * jnp.pi)
            # state_ref["path_rot"] = R.from_euler("z", torso_yaw) * state_ref["path_rot"]

            arm_joint_pos_offset = jax.random.uniform(
                rng_arm,
                (len(self.arm_actuator_indices),),
                minval=self.arm_joint_pos_range[0],
                maxval=self.arm_joint_pos_range[1],
            )
            joint_pos_offset = joint_pos_offset.at[self.arm_actuator_indices].set(
                arm_joint_pos_offset
            )

            arm_motor_pos_offset = self.robot.arm_ik(arm_joint_pos_offset)
            motor_pos_offset = motor_pos_offset.at[self.arm_actuator_indices].set(
                arm_motor_pos_offset
            )

            left_hip_pitch_idx = self.leg_pitch_actuator_indices[0]
            left_knee_pitch_idx = self.leg_pitch_actuator_indices[1]

            offsets = self.robot.config["robot"]
            torso_z_delta = (jnp.cos(torso_pitch) - 1) * offsets["torso_to_hip_z"] + (
                jnp.cos((joint_pos_old + joint_pos_offset)[left_hip_pitch_idx])
                - jnp.cos(joint_pos_old[left_hip_pitch_idx])
            ) * offsets["hip_to_knee_z"]
            +(
                jnp.cos((joint_pos_old + joint_pos_offset)[left_knee_pitch_idx])
                - jnp.cos(joint_pos_old[left_knee_pitch_idx])
            ) * offsets["knee_to_ankle_z"]

            torso_rot = R.from_quat(jnp.concatenate([qpos[4:7], qpos[3:4]]))
            torso_rot_delta = R.from_euler(
                "xyz", jnp.concatenate([torso_roll, torso_pitch, jnp.zeros(1)])
            )
            torso_quat_xyzw_new = (torso_rot_delta * torso_rot).as_quat()
            torso_quat_new = jnp.concatenate(
                [torso_quat_xyzw_new[3:], torso_quat_xyzw_new[:3]]
            )

            qpos = qpos.at[2:3].add(torso_z_delta)
            qpos = qpos.at[3:7].set(torso_quat_new)

        qpos = qpos.at[self.q_start_idx + self.motor_indices].set(
            motor_pos_old + motor_pos_offset
        )
        qpos = qpos.at[self.q_start_idx + self.joint_indices].set(
            joint_pos_old + joint_pos_offset
        )
        state_info["zero_point_offset"] = motor_pos_offset

        # Sample random spawn location if enabled
        if not self.fixed_base and self.cfg.terrain.scene is None:
            if self.random_spawn:
                idx = jax.random.randint(rng_spawn, (), 0, len(self.safe_spawns))
                safe_spawns_jax = jax.tree_util.tree_map(
                    lambda *xs: jnp.stack(xs), *self.safe_spawns
                )
                pos = jax.tree_util.tree_map(lambda x: x[idx], safe_spawns_jax)
            else:
                # Choose the "center-ish" safe spawn with bias to top and left
                rows = len(self.cfg.terrain.manual_map)
                cols = len(self.cfg.terrain.manual_map[0])

                # Compute center row/col with "rounding down" to bias toward top/left
                center_row = (rows - 1) // 2
                center_col = (cols - 1) // 2
                center_index = center_row * cols + center_col

                # Convert to JAX array for compatibility
                pos = jax.tree_util.tree_map(
                    lambda x: jnp.array(x), self.safe_spawns[center_index]
                )
                # pos = jax.tree_util.tree_map(lambda x: jnp.array(x), self.safe_spawns[0])

            qpos = qpos.at[:3].add(pos)

        # Apply obstacle randomization offsets if enabled
        if not self.fixed_base and self.cfg.obstacle_rand.enabled:
            spawn_y_offsets = jnp.array(self.cfg.obstacle_rand.spawn_y_offsets)
            num_obstacles = len(spawn_y_offsets)
            obstacle_idx = jax.random.randint(rng_obstacle, (), 0, num_obstacles)
            spawn_y_offset = spawn_y_offsets[obstacle_idx]
            qpos = qpos.at[1].add(spawn_y_offset)
            state_info["spawn_y_offset"] = spawn_y_offset

            # Apply Z offset if configured (for tasks where robot starts on obstacle)
            if len(self.cfg.obstacle_rand.spawn_z_offsets) > 0:
                spawn_z_offsets = jnp.array(self.cfg.obstacle_rand.spawn_z_offsets)
                spawn_z_offset = spawn_z_offsets[obstacle_idx]
                qpos = qpos.at[2].add(spawn_z_offset)
                state_info["spawn_z_offset"] = spawn_z_offset

        # Apply random position and orientation offsets if domain randomization is enabled
        if not self.fixed_base and self.add_domain_rand:
            # Random position offset in x,y
            if self.rand_pos_offset[0] > 0.0 or self.rand_pos_offset[1] > 0.0:
                rng_pos_x, rng_pos_y = jax.random.split(rng_pos_offset, 2)
                pos_offset_x = jax.random.uniform(
                    rng_pos_x,
                    (),
                    minval=-self.rand_pos_offset[0],
                    maxval=self.rand_pos_offset[0],
                )
                pos_offset_y = jax.random.uniform(
                    rng_pos_y,
                    (),
                    minval=-self.rand_pos_offset[1],
                    maxval=self.rand_pos_offset[1],
                )
                qpos = qpos.at[0].add(pos_offset_x)
                qpos = qpos.at[1].add(pos_offset_y)

            # Random yaw orientation offset
            if self.rand_yaw_offset > 0.0:
                yaw_offset = jax.random.uniform(
                    rng_quat_offset,
                    (),
                    minval=-self.rand_yaw_offset,
                    maxval=self.rand_yaw_offset,
                )
                # Get current torso orientation and add yaw offset
                current_quat_wxyz = qpos[3:7]  # [w,x,y,z] format
                current_quat_xyzw = jnp.concatenate(
                    [current_quat_wxyz[1:], current_quat_wxyz[:1]]
                )
                current_rot = R.from_quat(current_quat_xyzw)
                yaw_rot = R.from_euler("z", yaw_offset)
                new_rot = yaw_rot * current_rot
                new_quat_xyzw = new_rot.as_quat()
                new_quat_wxyz = jnp.concatenate([new_quat_xyzw[3:], new_quat_xyzw[:3]])
                qpos = qpos.at[3:7].set(new_quat_wxyz)

        # Update state_ref if random offsets were applied to ensure consistent tracking
        if not self.fixed_base and self.add_domain_rand and (original_pos is not None):
            if (
                self.rand_pos_offset[0] > 0.0 or self.rand_pos_offset[1] > 0.0
            ) or self.rand_yaw_offset > 0.0:
                # Get the robot's actual position and orientation after randomization
                actual_pos = qpos[:3]
                actual_quat_wxyz = qpos[3:7]
                actual_quat_xyzw = jnp.concatenate(
                    [actual_quat_wxyz[1:], actual_quat_wxyz[:1]]
                )
                actual_rot = R.from_quat(actual_quat_xyzw)

                # Re-compute state_ref with the robot's actual initial pose
                try:
                    state_ref = self.motion_ref.get_state_ref(
                        0.0,
                        command,
                        state_ref,
                        state_info["init_idx"],
                        torso_yaw=None,
                        direction_idx=None,
                        robot_init_pos=actual_pos,
                        robot_init_rot=actual_rot,
                    )
                except TypeError:
                    # Random offsets not used, no changes needed
                    pass

        # Initialize qvel from reference motion for RSI (Reference State Initialization)
        qvel = jnp.zeros(self.nv)
        if not self.fixed_base:
            # For free-floating robot, set root velocities from reference
            torso_lin_vel = state_ref["body_lin_vel"][0]  # Torso linear velocity (3,)
            torso_ang_vel = state_ref["body_ang_vel"][0]  # Torso angular velocity (3,)
            qvel = qvel.at[:3].set(torso_lin_vel)
            qvel = qvel.at[3:6].set(torso_ang_vel)
        # Set motor and joint velocities from reference (if available, else stays zero)
        if getattr(self.motion_ref, "has_motor_vel", False):
            qvel = qvel.at[self.qd_start_idx + self.motor_indices].set(
                state_ref["motor_vel"]
            )
        if getattr(self.motion_ref, "has_joint_vel", False):
            qvel = qvel.at[self.qd_start_idx + self.joint_indices].set(
                state_ref["joint_vel"]
            )

        pipeline_state = self.pipeline_init(qpos, qvel)

        state_info["command"] = command
        # Handle empty command_obs_indices gracefully (e.g., for DeepMimic)
        if len(self.command_obs_indices) > 0:
            state_info["command_obs"] = command[self.command_obs_indices]
        else:
            state_info["command_obs"] = jnp.array([], dtype=jnp.float32)

        state_info["state_ref"] = state_ref
        state_info["first_state_ref"] = {
            k: R.from_quat(v.as_quat()) if isinstance(v, R) else v.copy()
            for k, v in state_ref.items()
        }
        state_info["stance_mask"] = state_ref["stance_mask"]
        state_info["last_stance_mask"] = state_ref["stance_mask"]
        # Get phase signal - try to pass init_idx, fallback if not supported
        try:
            state_info["phase_signal"] = self.motion_ref.get_phase_signal(
                0.0, state_info["init_idx"]
            )
        except TypeError:
            # Fallback for motion references that don't support init_idx parameter
            state_info["phase_signal"] = self.motion_ref.get_phase_signal(0.0)

        state_info["feet_height_init"] = pipeline_state.x.pos[self.feet_link_ids, 2]
        if "get_up_" in self.name:
            state_info["default_action"] = jnp.mean(self.action_limits, axis=1)
        else:
            # Use first frame motor_pos for default_action (not random init frame for RSI)
            first_frame_ref = self.motion_ref.get_state_ref(
                0.0, command, self.motion_ref.get_default_state(), init_idx=0
            )
            state_info["default_action"] = first_frame_ref["motor_pos"][
                self.action_mask
            ].copy()

        state_info["last_action_target"] = state_info["default_action"].copy()
        state_info["actuator_noise"] = {}

        if self.add_domain_rand:
            state_info["backlash"] = jax.random.uniform(
                rng_backlash,
                (self.nu,),
                minval=self.backlash_range[0],
                maxval=self.backlash_range[1],
            )

            # Sample persistent torso calibration errors (affects physics throughout episode)
            persistent_torso_pitch = jax.random.uniform(
                rng_persistent_pitch,
                (),
                minval=self.persistent_torso_pitch_error_range[0],
                maxval=self.persistent_torso_pitch_error_range[1],
            )
            persistent_torso_roll = jax.random.uniform(
                rng_persistent_roll,
                (),
                minval=self.persistent_torso_roll_error_range[0],
                maxval=self.persistent_torso_roll_error_range[1],
            )

            # Torso ROLL → waist motors
            persistent_waist_joint = jnp.array([-persistent_torso_roll, 0.0])
            persistent_waist_motor = self.robot.waist_ik(persistent_waist_joint)
            state_info["persistent_waist_calibration_error"] = persistent_waist_motor

            # Torso PITCH → distribute across leg pitch joints (hip, knee, ankle)
            # Use SAME randomized distribution as initial torso_pitch_range
            _, rng_hip_pitch_pers, rng_knee_pitch_pers = jax.random.split(
                rng_persistent_pitch, 3
            )
            hip_pitch_delta_pers = jax.random.uniform(
                rng_hip_pitch_pers,
                (1,),
                minval=0.0,
                maxval=jnp.abs(persistent_torso_pitch),
            )
            knee_pitch_delta_pers = jax.random.uniform(
                rng_knee_pitch_pers,
                (1,),
                minval=0.0,
                maxval=jnp.abs(persistent_torso_pitch) - hip_pitch_delta_pers,
            )
            ankle_pitch_delta_pers = (
                jnp.abs(persistent_torso_pitch)
                - hip_pitch_delta_pers
                - knee_pitch_delta_pers
            )
            leg_pitch_delta_pers = jnp.concatenate(
                [
                    hip_pitch_delta_pers,
                    knee_pitch_delta_pers,
                    ankle_pitch_delta_pers,
                    hip_pitch_delta_pers,
                    knee_pitch_delta_pers,
                    ankle_pitch_delta_pers,
                ]
            )
            leg_pitch_error_pers = (
                leg_pitch_delta_pers
                * self.leg_pitch_joint_signs
                * jnp.sign(persistent_torso_pitch)
            )
            state_info["persistent_leg_pitch_error"] = leg_pitch_error_pers

            state_info["actuator_noise"] = {
                "kp": jax.random.uniform(
                    rng_kp,
                    (self.nu,),
                    minval=self.kp_range[0],
                    maxval=self.kp_range[1],
                ),
                "kd": jax.random.uniform(
                    rng_kd,
                    (self.nu,),
                    minval=self.kd_range[0],
                    maxval=self.kd_range[1],
                ),
                "tau_max": jax.random.uniform(
                    rng_tau_max,
                    (self.nu,),
                    minval=self.tau_max_range[0],
                    maxval=self.tau_max_range[1],
                ),
                "q_dot_tau_max": jax.random.uniform(
                    rng_q_dot_tau_max,
                    (self.nu,),
                    minval=self.q_dot_tau_max_range[0],
                    maxval=self.q_dot_tau_max_range[1],
                ),
                "q_dot_max": jax.random.uniform(
                    rng_q_dot_max,
                    (self.nu,),
                    minval=self.q_dot_max_range[0],
                    maxval=self.q_dot_max_range[1],
                ),
                "kd_min": jax.random.uniform(
                    rng_kd_min,
                    (self.nu,),
                    minval=self.kd_min_range[0],
                    maxval=self.kd_min_range[1],
                ),
                "tau_brake_max": jax.random.uniform(
                    rng_tau_brake_max,
                    (self.nu,),
                    minval=self.tau_brake_max_range[0],
                    maxval=self.tau_brake_max_range[1],
                ),
                "tau_q_dot_max": jax.random.uniform(
                    rng_tau_q_dot_max,
                    (self.nu,),
                    minval=self.tau_q_dot_max_range[0],
                    maxval=self.tau_q_dot_max_range[1],
                ),
                "passive_active_ratio": jax.random.uniform(
                    rng,
                    (self.nu,),
                    minval=self.passive_active_ratio_range[0],
                    maxval=self.passive_active_ratio_range[1],
                ),
            }
        else:
            # No domain randomization - zero calibration error
            state_info["persistent_waist_calibration_error"] = jnp.zeros(
                len(self.waist_actuator_indices)
            )
            state_info["persistent_leg_pitch_error"] = jnp.zeros(
                len(self.leg_pitch_actuator_indices)
            )

        obs_history = jnp.zeros(self.num_obs_history * self.obs_size)
        privileged_obs_history = jnp.zeros(
            self.num_privileged_obs_history * self.privileged_obs_size
        )
        obs, info = self._get_obs(
            pipeline_state,
            state_info,
            obs_history,
            privileged_obs_history,
        )
        reward, done, zero = jnp.zeros(3)

        metrics: dict[str, Any] = {}
        for k in self.reward_names:
            metrics[k] = zero

        return State(pipeline_state, obs, reward, done, metrics, state_info)

    def pipeline_step(self, state: State, action: jax.Array) -> base.State:
        """Executes a pipeline step by applying a control action to the system state.

        This function iteratively applies a control action to the system's state over a specified number of frames. It uses a controller to compute control signals based on the current state and action, and updates the pipeline state accordingly.

        Args:
            state (State): The current state of the system, containing information required for control computations.
            action (jax.Array): The control action to be applied to the system.

        Returns:
            base.State: The updated state of the system after applying the control action over the specified number of frames.
        """

        def f(pipeline_state, _):
            ctrl = self.controller.step(
                pipeline_state.q[self.q_start_idx + self.motor_indices],
                pipeline_state.qd[self.qd_start_idx + self.motor_indices],
                pipeline_state.qacc[self.qd_start_idx + self.motor_indices],
                action,
                state.info["actuator_noise"],
            )
            return (
                self._pipeline.step(self.sys, pipeline_state, ctrl, self._debug),
                None,
            )

        return jax.lax.scan(f, state.pipeline_state, (), self._n_frames)[0]

    # @profile()
    def step(self, state: State, action: jax.Array) -> State:
        """Advances the simulation by one time step, updating the state based on the given action.

        This function updates the state of the simulation by processing the given action, applying filters, and incorporating domain randomization if enabled. It computes the motor targets, updates the pipeline state, and checks for termination conditions. Additionally, it calculates rewards and updates various state information, including contact forces, stance masks, and command resampling.

        Args:
            state (State): The current state of the simulation, containing information about the system's dynamics and metadata.
            action (jax.Array): The action to be applied at this time step, influencing the system's behavior.

        Returns:
            State: The updated state after applying the action and advancing the simulation by one step.
        """
        rng, cmd_rng, push_torso_rng, push_id_rng, push_theta_rng, push_rng = (
            jax.random.split(state.info["rng"], 6)
        )

        # Check if motion completed in previous step and teleport BEFORE computing anything
        # This ensures obs/rewards are computed from the correct (teleported) state
        should_teleport = (
            state.info["step"] + state.info["init_idx"] >= self.episode_length
        )

        def do_teleport():
            # Get qpos from frame 0 of the motion file
            frame0_ref = self.motion_ref.get_state_ref(
                0.0, state.info["command"], state.info["state_ref"], init_idx=0
            )
            frame0_qpos = frame0_ref["qpos"]
            frame0_qvel = jnp.zeros(self.nv)  # Zero velocity at start of motion
            new_pipeline_state = self.pipeline_init(frame0_qpos, frame0_qvel)
            return (
                new_pipeline_state,
                frame0_ref,
                0,
            )  # pipeline_state, state_ref, init_idx

        def no_teleport():
            return (
                state.pipeline_state,
                state.info["state_ref"],
                state.info["init_idx"],
            )

        pipeline_state_start, state_ref_start, init_idx_start = jax.lax.cond(
            should_teleport, do_teleport, no_teleport
        )

        # Update state with potentially teleported values
        state = state.replace(pipeline_state=pipeline_state_start)
        state.info["state_ref"] = state_ref_start
        state.info["init_idx"] = init_idx_start
        # Reset motion step counter on teleport (motion restarts at frame 0)
        # Note: EpisodeWrapper's episode length tracking uses state.info['steps'] (plural)
        # so resetting our 'step' (singular) doesn't affect episode termination
        state.info["step"] = jnp.where(should_teleport, 0, state.info["step"])
        # Reset torso_height_max on teleport
        state.info["torso_height_max"] = jnp.where(
            should_teleport, 0.0, state.info["torso_height_max"]
        )
        # Clear history buffers on teleport to avoid stale data from previous motion cycle
        zero_action_buffer = jnp.zeros((self.n_steps_delay + 1) * self.num_action)
        state.info["action_buffer"] = jnp.where(
            should_teleport, zero_action_buffer, state.info["action_buffer"]
        )
        state.info["last_act"] = jnp.where(
            should_teleport, jnp.zeros(self.num_action), state.info["last_act"]
        )
        state.info["last_action_target"] = jnp.where(
            should_teleport,
            state.info["default_action"],
            state.info["last_action_target"],
        )
        state.info["feet_air_time"] = jnp.where(
            should_teleport, jnp.zeros(2), state.info["feet_air_time"]
        )
        state.info["feet_air_dist"] = jnp.where(
            should_teleport, jnp.zeros(2), state.info["feet_air_dist"]
        )
        # Clear observation history buffers on teleport
        zero_obs_history = jnp.zeros(self.num_obs_history * self.obs_size)
        zero_privileged_obs_history = jnp.zeros(
            self.num_privileged_obs_history * self.privileged_obs_size
        )
        new_obs = {
            "state": jnp.where(should_teleport, zero_obs_history, state.obs["state"]),
            "privileged_state": jnp.where(
                should_teleport,
                zero_privileged_obs_history,
                state.obs["privileged_state"],
            ),
        }
        state = state.replace(obs=new_obs)

        torso_quat = state.pipeline_state.x.rot[0]
        torso_quat_xyzw = jnp.concatenate([torso_quat[1:], torso_quat[:1]])
        torso_yaw = R.from_quat(torso_quat_xyzw).as_euler("xyz")[2]

        time_curr = state.info["step"] * self.dt
        # Try to pass init_idx, fallback if not supported
        try:
            state_ref = self.motion_ref.get_state_ref(
                time_curr,
                state.info["command"],
                state.info["state_ref"],
                state.info["init_idx"],
                torso_yaw=torso_yaw,
            )
        except TypeError:
            # Fallback for motion references that don't support torso_rot parameter
            state_ref = self.motion_ref.get_state_ref(
                time_curr,
                state.info["command"],
                state.info["state_ref"],
                state.info["init_idx"],
            )

        state.info["state_ref"] = state_ref
        # Get phase signal - try to pass init_idx, fallback if not supported
        try:
            state.info["phase_signal"] = self.motion_ref.get_phase_signal(
                time_curr, state.info["init_idx"]
            )
        except TypeError:
            # Fallback for motion references that don't support init_idx parameter
            # NOTE: Seems like all LBF ref don't support init_idx which means using rsi will break phase signal?
            state.info["phase_signal"] = self.motion_ref.get_phase_signal(time_curr)

        state.info["action_buffer"] = (
            jnp.roll(state.info["action_buffer"], self.num_action)
            .at[: self.num_action]
            .set(action)
        )

        delayed_action = state.info["action_buffer"][-self.num_action :]
        action_target = (
            state.info["default_action"] + self.action_scale * delayed_action
        )
        # Action target hack, only for debugging
        # action_target = (
        #     (self.action_limits[:, 0] + self.action_limits[:, 1])
        #     + (self.action_limits[:, 1] - self.action_limits[:, 0]) * delayed_action
        # ) * 0.5
        # jax.debug.print("action_target: {}", action_target)

        # Nomalization 2
        # action_target = (
        #     state.info["default_action"]
        #     + (self.action_limits[:, 1] - state.info["default_action"])
        #     * jnp.maximum(delayed_action, 0.0)
        #     + (self.action_limits[:, 1] - state.info["default_action"])
        #     * jnp.minimum(delayed_action, 0.0)
        # )

        assert isinstance(action_target, jax.Array)
        state.info["last_action_target"] = action_target.copy()

        motor_target = (
            jnp.asarray(state_ref["motor_pos"]).at[self.action_mask].set(action_target)
        )

        # Ground Truth Trajectory, only for debugging
        # motor_target = state_ref["motor_pos"]

        if self.add_domain_rand and self.add_head_pose:
            motor_target = motor_target.at[self.neck_actuator_indices].set(
                state.info["neck_motor_target"]
            )

        # Apply persistent calibration errors (torso tilt persists throughout episode)
        # Torso roll error → waist motors
        motor_target = motor_target.at[self.waist_actuator_indices].add(
            state.info["persistent_waist_calibration_error"]
        )
        # Torso pitch error → leg pitch joints (hip, knee, ankle)
        motor_target = motor_target.at[self.leg_pitch_actuator_indices].add(
            state.info["persistent_leg_pitch_error"]
        )

        if self.add_push:
            # Check if we should start a new push
            start_new_push = (
                jnp.mod(state.info["push_step"] + 1, self.push_interval) == 0
            )

            # If starting new push, generate new push parameters
            push_torso = jax.random.bernoulli(push_torso_rng, 0.5)
            new_push_id = jnp.where(
                push_torso, 1, jax.random.randint(push_id_rng, (), 2, self.sys.nbody)
            )
            push_theta = jax.random.uniform(push_theta_rng, maxval=2 * jnp.pi)
            push_magnitude = jnp.where(
                push_torso,
                jax.random.uniform(
                    push_rng,
                    minval=self.push_torso_range[0],
                    maxval=self.push_torso_range[1],
                ),
                jax.random.uniform(
                    push_rng,
                    minval=self.push_other_range[0],
                    maxval=self.push_other_range[1],
                ),
            )
            new_push = (
                jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)]) * push_magnitude
            )

            # Update push parameters if starting new push, otherwise keep previous
            push_id = jnp.where(start_new_push, new_push_id, state.info["push_id"])
            push = jnp.where(start_new_push, new_push, state.info["push"])

            # Update duration remaining
            duration_remaining = jnp.where(
                start_new_push,
                self.push_duration,
                jnp.maximum(0, state.info["push_remaining"] - 1),
            )

            # Apply push only if duration remaining > 0
            active_push = jnp.where(duration_remaining > 0, push, jnp.zeros(2))

            xfrc = jnp.zeros_like(state.pipeline_state.xfrc_applied)
            xfrc = xfrc.at[push_id, :2].set(active_push)

            state = state.tree_replace({"pipeline_state.xfrc_applied": xfrc})

            state.info["push_id"] = push_id
            state.info["push"] = active_push
            state.info["push_remaining"] = duration_remaining
            state.info["push_step"] += 1

        motor_target = jnp.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        pipeline_state = self.pipeline_step(state, motor_target)

        if not self.fixed_base:
            (
                contact_forces,
                num_contact_points,
                stance_mask,
            ) = self._solve_contact(pipeline_state)

            state.info["contact_forces"] = contact_forces
            state.info["num_contact_points"] = num_contact_points
            state.info["stance_mask"] = stance_mask

        torso_pos = pipeline_state.x.pos[0]
        torso_height = torso_pos[2]
        if not self.fixed_base and self.cfg.terrain.scene is None:
            # Get terrain elevation at robot position using elevation function
            local_ground = get_robot_ground_height(
                torso_pos[0], torso_pos[1], self.elevation_info
            )
            torso_height -= local_ground

        if "get_up_" in self.name:
            done = jnp.logical_or(
                jnp.logical_or(
                    torso_height - state_ref["qpos"][2] < self.healthy_z_range[0],
                    torso_height - state_ref["qpos"][2] > self.healthy_z_range[1],
                ),
                torso_height - state.info["torso_height_max"] < -0.1,
            )
        else:
            done = jnp.logical_or(
                torso_height < self.healthy_z_range[0],
                torso_height > self.healthy_z_range[1],
            )

        # TODO: Move check_termination to a separate function

        state.info["torso_height_max"] = jnp.maximum(
            torso_height, state.info["torso_height_max"]
        )

        # Prevent done for fixed base
        if self.fixed_base:
            done = jnp.array(False)

        state.info["done"] = done

        obs, info = self._get_obs(
            pipeline_state,
            state.info,
            state.obs["state"],
            state.obs["privileged_state"],
        )

        reward_dict = self._compute_reward(pipeline_state, state.info, action)
        reward = sum(reward_dict.values()) * self.dt
        # reward = jnp.clip(reward, 0.0)

        if not self.fixed_base:
            state.info["last_stance_mask"] = stance_mask.copy()
            state.info["feet_air_time"] += self.dt
            state.info["feet_air_time"] *= 1.0 - stance_mask

            feet_z_delta = (
                pipeline_state.x.pos[self.feet_link_ids, 2]
                - state.info["feet_height_init"]
            )
            state.info["feet_air_dist"] += feet_z_delta
            state.info["feet_air_dist"] *= 1.0 - stance_mask

        state.info["last_act"] = delayed_action.copy()
        state.info["rewards"] = reward_dict
        state.info["rng"] = rng
        state.info["step"] += 1

        # Check if command should be resampled
        should_resample = state.info["step"] % self.resample_steps == 0

        state.info["command"] = jax.lax.cond(
            should_resample,
            lambda: self._sample_command(cmd_rng, state.info["command"]),
            lambda: state.info["command"],
        )
        """
        # Reset path state when command is resampled to prevent tracing old commands
        def reset_path_state():
            # Get current robot position and orientation
            current_pos = pipeline_state.x.pos[0]  # Torso position
            current_quat_wxyz = pipeline_state.x.rot[0]  # Torso quaternion [w,x,y,z]
            current_quat_xyzw = jnp.concatenate(
                [current_quat_wxyz[1:], current_quat_wxyz[:1]]
            )
            current_rot = R.from_quat(current_quat_xyzw)

            # Update state_ref with current robot pose
            updated_state_ref = state.info["state_ref"].copy()
            updated_state_ref["path_pos"] = current_pos
            updated_state_ref["path_rot"] = current_rot
            return updated_state_ref

        state.info["state_ref"] = jax.lax.cond(
            should_resample,
            reset_path_state,
            lambda: state.info["state_ref"],
        )
        """

        # Handle empty command_obs_indices gracefully (e.g., for DeepMimic)
        if len(self.command_obs_indices) > 0:
            state.info["command_obs"] = state.info["command"][self.command_obs_indices]
        else:
            state.info["command_obs"] = jnp.array([], dtype=jnp.float32)

        # Reset state_ref on failure (internal timeout already handled at step start)
        state.info["state_ref"] = jax.tree_map(
            lambda old, new: jnp.where(done, new, old),
            state.info["state_ref"],
            state.info["first_state_ref"],
        )
        # Reset step on failure (internal timeout keeps step counting)
        state.info["step"] = jnp.where(done, 0, state.info["step"])

        state.metrics.update(reward_dict)

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done.astype(jnp.float32),
        )

    def visualize_world_axes(
        self,
        renderer: mujoco.Renderer,
        origin: numpy.ndarray = numpy.zeros(3),
        axis_len: float = 20.0,
        alpha: float = 0.5,
    ):
        colors = {
            "x": [1, 0, 0, alpha],  # red
            "y": [0, 1, 0, alpha],  # green
            "z": [0, 0, 1, alpha],  # blue
        }
        axes = {
            "x": numpy.array([axis_len, 0, 0]),
            "y": numpy.array([0, axis_len, 0]),
            "z": numpy.array([0, 0, axis_len]),
        }
        for key in ["x", "y", "z"]:
            p1 = origin
            p2 = origin + axes[key]
            i = renderer.scene.ngeom
            geom = renderer.scene.geoms[i]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=numpy.zeros(3),
                pos=numpy.zeros(3),
                mat=numpy.eye(3).flatten(),
                rgba=colors[key],
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 5, p1, p2)
            renderer.scene.ngeom += 1

    def visualize_force_arrow(
        self,
        renderer: mujoco.Renderer,
        state: State,
        push_id: int,
        push_force: jax.Array,
        vis_scale: float = 0.05,
    ):
        i = renderer.scene.ngeom
        p1 = state.pipeline_state.xipos[push_id, :3]
        p2 = p1 + push_force * vis_scale
        # print(
        #     f"geom: {i}, p1: {p1}, p2: {p2}, push_id: {push_id}, push: {push_force}"
        # )
        geom = renderer.scene.geoms[i]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=numpy.zeros(3),
            pos=numpy.zeros(3),
            mat=numpy.eye(3).flatten(),
            rgba=[1, 0, 0, 1],  # red arrow for force
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.01, p1, p2)
        renderer.scene.ngeom += 1

    def render(
        self,
        states: list[State],
        height: int = 240,
        width: int = 320,
        camera: str | None = None,
    ):
        renderer = mujoco.Renderer(self.sys.mj_model, height=height, width=width)
        camera = camera or -1

        push_id = states[0].info["push_id"]
        push_force = numpy.concatenate([states[0].info["push"], numpy.zeros(1)])
        image_list = []
        for state in states:
            d = mujoco.MjData(self.sys.mj_model)
            d.qpos, d.qvel = state.pipeline_state.q, state.pipeline_state.qd
            mujoco.mj_forward(self.sys.mj_model, d)
            renderer.update_scene(d, camera=camera)
            if numpy.linalg.norm(state.info["push"]) > 0:
                push_id = state.info["push_id"]
                push_force = numpy.concatenate([state.info["push"], numpy.zeros(1)])

            self.visualize_world_axes(renderer)
            self.visualize_force_arrow(renderer, state, push_id, push_force)
            image_list.append(renderer.render())

        return image_list

    def _sample_command(
        self, rng: jax.Array, last_command: jax.Array | None = None
    ) -> jax.Array:
        raise NotImplementedError

    def _sample_command_uniform(
        self, rng: jax.Array, command_range: jax.Array
    ) -> jax.Array:
        """Generates a uniformly distributed random sample within specified command ranges.

        Args:
            rng (jax.Array): A JAX random number generator array.
            command_range (jax.Array): A 2D array where each row specifies the minimum and maximum values for sampling.

        Returns:
            jax.Array: An array of uniformly distributed random samples, one for each range specified in `command_range`.
        """
        return jax.random.uniform(
            rng,
            (command_range.shape[0],),
            minval=command_range[:, 0],
            maxval=command_range[:, 1],
        )

    def _solve_contact(self, data: mjx.Data):
        """Compute contact forces between colliders and determine foot contact masks.

        This function calculates the contact forces between colliders based on the provided
        simulation data. It also determines whether the left and right foot colliders are in
        contact with the ground by comparing the contact forces against a predefined threshold.

        Args:
            data (mjx.Data): The simulation data containing contact information.

        Returns:
            Tuple[jax.Array, jax.Array, jax.Array]: A tuple containing:
                - A 3D array of shape (num_colliders, num_colliders, 3) representing the global
                  contact forces between colliders.
                - A 1D array indicating whether each left foot collider is in contact.
                - A 1D array indicating whether each right foot collider is in contact.
        """
        # Extract geom1 and geom2 directly
        geom1 = data.contact.geom1
        geom2 = data.contact.geom2

        def get_body_index(geom_id: jax.Array) -> jax.Array:
            return jnp.argmax(self.collider_geom_ids == geom_id)

        # Vectorized computation of body indices for geom1 and geom2
        body_indices_1 = jax.vmap(get_body_index)(geom1)
        body_indices_2 = jax.vmap(get_body_index)(geom2)

        contact_forces_global = jnp.zeros((self.num_colliders, self.num_colliders, 3))
        num_contact_points = jnp.zeros((self.num_colliders, self.num_colliders))
        for i in range(data.ncon):
            contact_force = self.jit_contact_force(self.sys, data, i, True)[:3]
            # Update the contact forces for both body_indices_1 and body_indices_2
            # Add instead of set to accumulate forces from multiple contacts
            contact_forces_global = contact_forces_global.at[
                body_indices_1[i], body_indices_2[i]
            ].add(contact_force)
            contact_forces_global = contact_forces_global.at[
                body_indices_2[i], body_indices_1[i]
            ].add(contact_force)
            is_valid = (jnp.linalg.norm(contact_force) > 0.1).astype(jnp.float32)
            num_contact_points = num_contact_points.at[
                body_indices_1[i], body_indices_2[i]
            ].add(is_valid)
            num_contact_points = num_contact_points.at[
                body_indices_2[i], body_indices_1[i]
            ].add(is_valid)

        left_foot_contact_mask = (
            contact_forces_global[0, self.left_foot_collider_indices, 2]
            > self.contact_force_threshold
        )
        right_foot_contact_mask = (
            contact_forces_global[0, self.right_foot_collider_indices, 2]
            > self.contact_force_threshold
        )

        stance_mask = jnp.array(
            [jnp.any(left_foot_contact_mask), jnp.any(right_foot_contact_mask)]
        ).astype(jnp.float32)

        return contact_forces_global, num_contact_points, stance_mask

    def _get_imu_noise(
        self,
        rng: jax.Array,
        true_gyro: jax.Array,
        true_quat_wxyz: jax.Array,  # (4,) wxyz
        imu_state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        # Gyro AR(1) + bias RW + optional white
        (
            _,
            rng_gyro_ar1,
            rng_gyro_bias,
            rng_gyro_white,
            rng_quat_ar1,
            rng_quat_bias,
            rng_quat_white,
            rng_gyro_amp,
            rng_quat_amp,
        ) = jax.random.split(rng, 9)

        gyro_ar1, gyro_bias, quat_ar1, quat_bias = imu_state

        rho_g = jnp.exp(-2.0 * jnp.pi * self.cfg.noise.gyro_fc * self.dt)
        eps_g_std = self.cfg.noise.gyro_std * jnp.sqrt(
            jnp.maximum(1e-12, 1.0 - rho_g**2)
        )
        gyro_ar1 = rho_g * gyro_ar1 + eps_g_std * jax.random.normal(rng_gyro_ar1, (3,))
        gyro_bias = gyro_bias + self.cfg.noise.gyro_bias_walk_std * jax.random.normal(
            rng_gyro_bias, (3,)
        )
        gyro_white = self.cfg.noise.gyro_white_std * jax.random.normal(
            rng_gyro_white, (3,)
        )

        # Add amplitude variation to gyro noise
        gyro_amp_scale = jax.random.uniform(
            rng_gyro_amp,
            (3,),
            minval=self.cfg.noise.gyro_amp_min,
            maxval=self.cfg.noise.gyro_amp_max,
        )
        gyro_noisy = true_gyro + gyro_amp_scale * (gyro_ar1 + gyro_bias + gyro_white)

        # Orientation: AR(1) rotvec + bias RW + optional white
        rho_q = jnp.exp(-2.0 * jnp.pi * self.cfg.noise.quat_fc * self.dt)
        eps_q_std = self.cfg.noise.quat_std * jnp.sqrt(
            jnp.maximum(1e-12, 1.0 - rho_q**2)
        )
        quat_ar1 = rho_q * quat_ar1 + eps_q_std * jax.random.normal(rng_quat_ar1, (3,))
        quat_bias = quat_bias + self.cfg.noise.quat_bias_walk_std * jax.random.normal(
            rng_quat_bias, (3,)
        )
        quat_white = self.cfg.noise.quat_white_std * jax.random.normal(
            rng_quat_white, (3,)
        )

        # Add amplitude variation to quat noise
        quat_amp_scale = jax.random.uniform(
            rng_quat_amp,
            (3,),
            minval=self.cfg.noise.quat_amp_min,
            maxval=self.cfg.noise.quat_amp_max,
        )
        rotvec_total = quat_amp_scale * (quat_ar1 + quat_bias + quat_white)

        # Compose: R_noise * R_true
        q_true_xyzw = jnp.concatenate([true_quat_wxyz[1:], true_quat_wxyz[:1]])
        R_noise = R.from_rotvec(rotvec_total)
        R_true = R.from_quat(q_true_xyzw)
        q_noisy_xyzw = (R_noise * R_true).as_quat()
        q_noise_wxyz = jnp.concatenate([q_noisy_xyzw[3:], q_noisy_xyzw[:3]])
        quat_noisy = jnp.where(q_noise_wxyz[0] < 0, -q_noise_wxyz, q_noise_wxyz)

        return gyro_noisy, quat_noisy, (gyro_ar1, gyro_bias, quat_ar1, quat_bias)

    def _get_obs(
        self,
        pipeline_state: base.State,
        info: dict[str, Any],
        obs_history: jax.Array,
        privileged_obs_history: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Generates and returns the current and privileged observations for the system.

        Args:
            pipeline_state (base.State): The current state of the pipeline, containing position, velocity, and other dynamics information.
            info (dict[str, Any]): A dictionary containing additional information such as random number generator state, reference states, and other auxiliary data.
            obs_history (jax.Array): An array storing the history of observations for the system.
            privileged_obs_history (jax.Array): An array storing the history of privileged observations for the system.

        Returns:
            Tuple[jax.Array, jax.Array]: A tuple containing the updated observation and privileged observation arrays.
        """
        _, rng_motor_pos, rng_motor_vel, rng_imu = jax.random.split(info["rng"], 4)

        motor_pos = pipeline_state.q[self.q_start_idx + self.motor_indices]
        motor_vel = pipeline_state.qd[self.qd_start_idx + self.motor_indices]

        # Subtract persistent calibration errors from observations
        # Physics has tilt, but encoders read commanded positions (not actual positions)
        # Policy must learn to balance using IMU (torso_quat/ang_vel) instead of motor positions
        motor_pos_corrected = motor_pos.copy()
        motor_pos_corrected = motor_pos_corrected.at[self.waist_actuator_indices].add(
            -info["persistent_waist_calibration_error"]
        )
        motor_pos_corrected = motor_pos_corrected.at[
            self.leg_pitch_actuator_indices
        ].add(-info["persistent_leg_pitch_error"])

        motor_pos_delta = motor_pos_corrected - self.default_motor_pos

        torso_quat = pipeline_state.x.rot[0]
        torso_quat = jnp.where(torso_quat[0] < 0, -torso_quat, torso_quat)
        torso_ang_vel = get_local_vec(pipeline_state.xd.ang[0], torso_quat)

        motor_pos_noisy = motor_pos_corrected.copy()
        motor_vel_noisy = motor_vel.copy()
        torso_quat_noisy = torso_quat.copy()
        torso_ang_vel_noisy = torso_ang_vel.copy()

        if self.add_domain_rand:
            # motor_pos_noisy -= info["zero_point_offset"]
            motor_pos_noisy += (
                0.5
                * info["backlash"]
                * jnp.tanh(
                    pipeline_state.qfrc_actuator[self.qd_start_idx + self.motor_indices]
                    / self.backlash_activation
                )
            )
            motor_pos_noisy += jax.random.uniform(
                rng_motor_pos,
                (self.nu,),
                minval=-self.cfg.noise.dof_pos * self.cfg.noise.level,
                maxval=self.cfg.noise.dof_pos * self.cfg.noise.level,
            )
            motor_vel_noisy += jax.random.uniform(
                rng_motor_vel,
                (self.nu,),
                minval=-self.cfg.noise.dof_vel * self.cfg.noise.level,
                maxval=self.cfg.noise.dof_vel * self.cfg.noise.level,
            )
            # Sample Euler angle noise
            torso_ang_vel_noisy, torso_quat_noisy, info["imu_state"] = (
                self._get_imu_noise(
                    rng_imu, torso_ang_vel, torso_quat, info["imu_state"]
                )
            )

        motor_pos_delta_noisy = motor_pos_noisy - self.default_motor_pos

        if self.use_phase_signal:
            phase_info = info["phase_signal"]  # 2 dims
        else:
            # Reference motor states as motion phase
            phase_info = jnp.concatenate(
                [
                    info["state_ref"]["motor_pos"],
                    info["state_ref"]["motor_vel"],
                ]
            )  # 2*nu dims

        obs = jnp.concatenate(
            [
                phase_info,
                info["command_obs"],
                motor_pos_delta_noisy * self.obs_scales.dof_pos,
                motor_vel_noisy * self.obs_scales.dof_vel,
                info["last_act"],
                # torso_lin_vel * self.obs_scales.lin_vel,
                torso_ang_vel_noisy * self.obs_scales.ang_vel,
                torso_quat_noisy * self.obs_scales.quat,
            ]
        )

        motor_pos_error = motor_pos - info["state_ref"]["motor_pos"]
        torso_lin_vel = get_local_vec(pipeline_state.xd.vel[0], torso_quat)

        # Motion phase for privileged obs (same as actor obs)
        if self.use_phase_signal:
            phase_info_privileged = info["phase_signal"]  # 2 dims
        else:
            # BeyondMimic: reference motor states as motion phase
            phase_info_privileged = jnp.concatenate(
                [
                    info["state_ref"]["motor_pos"],
                    info["state_ref"]["motor_vel"],
                ]
            )  # 2*nu dims

        privileged_obs = jnp.concatenate(
            [
                phase_info_privileged,
                info["command_obs"],
                motor_pos_delta * self.obs_scales.dof_pos,
                motor_vel * self.obs_scales.dof_vel,
                info["last_act"],
                torso_ang_vel * self.obs_scales.ang_vel,
                torso_quat * self.obs_scales.quat,
                motor_pos_error,
                torso_lin_vel * self.obs_scales.lin_vel,
                pipeline_state.actuator_force * self.obs_scales.actuator_force,
                info["stance_mask"],
                info["state_ref"]["stance_mask"],
                # info["push"],
                # info["feet_air_time"],
                # info["feet_air_dist"],
            ]
        )
        # jax.debug.breakpoint()

        # Stack observations through time (controlled by frame_stack config)
        # Note: frame_stack=1 means no history
        if self.num_obs_history > 1:
            obs = jnp.roll(obs_history, obs.size).at[: obs.size].set(obs)
        if self.num_privileged_obs_history > 1:
            privileged_obs = (
                jnp.roll(privileged_obs_history, privileged_obs.size)
                .at[: privileged_obs.size]
                .set(privileged_obs)
            )

        return {"state": obs, "privileged_state": privileged_obs}, info

    def _compute_reward(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Computes a dictionary of rewards based on the current pipeline state, additional information, and the action taken.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): Additional information that may be required for reward computation.
            action (jax.Array): The action taken, which influences the reward calculation.

        Returns:
            Dict[str, jax.Array]: A dictionary where keys are reward names and values are the computed rewards as JAX arrays.
        """
        # Create an array of indices to map over
        indices = jnp.arange(len(self.reward_names))
        global_step = info.get("global_step", 0)

        def resolve_scale(scale):
            if isinstance(scale, dict):
                keys = jnp.array(sorted([int(k) for k in scale.keys()]))
                values = jnp.array([scale[k] for k in sorted(scale.keys())])
                idx = jnp.sum(keys <= global_step) - 1
                return jnp.where(idx >= 0, values[idx], 0.0)
            else:
                return scale

        resolved_scales = jnp.array(
            [resolve_scale(scale) for scale in self.reward_scales]
        )

        reward_arr = jax.lax.map(
            lambda i: (
                jax.lax.switch(i, self.reward_functions, pipeline_state, info, action)
                * resolved_scales[i]
            ),
            indices,
        )

        reward_dict: dict[str, jax.Array] = {}
        for i, name in enumerate(self.reward_names):
            reward_dict[name] = reward_arr[i]

        return reward_dict

    def _tracking_mse(self, x: jax.Array):
        """Calculates the negative mean squared error for tracking rewards.

        Args:
            x (jax.Array): Input array for which to compute the MSE.

        Returns:
            jax.Array: The negative mean squared error.
        """
        return -jnp.mean(x**2)

    def _tracking_exp(self, x: jax.Array, sigma: float = 1.0):
        """Calculates the exponential of the input array.

        Args:
            x (jax.Array): Input array for which to compute the exponential.
            sigma (float): Scaling factor for the exponential calculation.

        Returns:
            jax.Array: The exponential of the input array.
        """
        return jnp.exp(-sigma * jnp.sum(x**2))

    # Generalized reward functions
    def _reward_torso_pos_xy(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates the reward based on the horizontal position of the torso.

        This function computes a reward by comparing the current horizontal position (x,y)
        of the torso to a reference position. The reward is calculated using a Gaussian
        function that penalizes deviations from the reference position in the XY plane.
        Doesn't work for fixed base envs.

        Args:
            pipeline_state (base.State): The current state of the system, containing
                positional information via qpos.
            info (dict[str, Any]): A dictionary containing reference state information,
                specifically the reference qpos containing global torso position.
            action (jax.Array): The action taken, though not used in this reward calculation.

        Returns:
            jax.Array: The computed reward based on the deviation of the torso's horizontal
            position from the reference position.
        """
        torso_pos = pipeline_state.qpos[:2]
        torso_pos_ref = info["state_ref"]["qpos"][:2]
        # Apply spawn Y offset for obstacle randomization
        spawn_y_offset = info.get("spawn_y_offset", 0.0)
        torso_pos_ref = torso_pos_ref.at[1].add(spawn_y_offset)
        error = jnp.linalg.norm(torso_pos - torso_pos_ref, axis=-1)
        reward = jnp.exp(-self.pos_tracking_sigma * error**2)
        return reward

    def _reward_torso_pos_z(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates the reward based on the vertical position of the torso.

        This function computes a reward by comparing the current height (z) of the torso
        to a reference height. The reward is calculated using a Gaussian function that
        penalizes deviations from the reference height.
        Doesn't work for fixed base envs.

        Args:
            pipeline_state (base.State): The current state of the system, containing
                positional information via x.pos for the torso.
            info (dict[str, Any]): A dictionary containing reference state information,
                specifically the reference qpos containing global torso position.
            action (jax.Array): The action taken, though not used in this reward calculation.

        Returns:
            jax.Array: The computed reward based on the deviation of the torso's height
            from the reference height.
        """
        torso_height = pipeline_state.qpos[2]
        torso_height_ref = info["state_ref"]["qpos"][2]
        # Apply spawn Z offset for obstacle randomization
        spawn_z_offset = info.get("spawn_z_offset", 0.0)
        torso_height_ref = torso_height_ref + spawn_z_offset
        error = torso_height - torso_height_ref
        reward = jnp.exp(-self.pos_tracking_sigma * error**2)
        return reward

    def _reward_torso_quat(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates a reward based on the alignment of the torso's quaternion orientation with a reference orientation.

        Args:
            pipeline_state (base.State): The current state of the system, containing the rotation of the torso.
            info (dict[str, Any]): A dictionary containing reference state information, including the reference quaternion and waist joint positions.
            action (jax.Array): The action taken, though not used in this function.

        Returns:
            jax.Array: A reward value computed from the quaternion angle difference between the current and reference torso orientations.
        """
        torso_rot = R.from_quat(
            jnp.concatenate([pipeline_state.x.rot[0][1:], pipeline_state.x.rot[0][:1]])
        )
        torso_rot_ref = R.from_quat(
            jnp.concatenate(
                [info["state_ref"]["qpos"][4:7], info["state_ref"]["qpos"][3:4]]
            )
        )
        rel_rot = torso_rot * torso_rot_ref.inv()
        angle_diff = rel_rot.magnitude()  # returns angle in radians
        reward = jnp.exp(-self.rot_tracking_sigma * angle_diff**2)
        return reward

    def _reward_lin_vel_xy(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates the reward based on the linear velocity in the XY plane.

        Args:
            pipeline_state (base.State): The current state of the system, including position and velocity.
            info (dict[str, Any]): Additional information, including reference state data.
            action (jax.Array): The action taken by the agent.

        Returns:
            jax.Array: The computed reward value, which is a function of the error between the current and reference linear velocities in the XY plane.
        """
        lin_vel_local = get_local_vec(pipeline_state.xd.vel[0], pipeline_state.x.rot[0])
        lin_vel_xy = lin_vel_local[:2]
        lin_vel_xy_ref = info["state_ref"]["lin_vel"][:2]
        error = jnp.linalg.norm(lin_vel_xy - lin_vel_xy_ref, axis=-1)
        reward = jnp.exp(-self.lin_vel_tracking_sigma * error**2)
        return reward

    def _reward_lin_vel_z(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculate the reward based on the vertical component of linear velocity.

        This function computes a reward that measures how closely the vertical component
        of the linear velocity of an object matches a reference value. The reward is
        calculated using a Gaussian function of the error between the actual and
        reference vertical velocities.

        Args:
            pipeline_state (base.State): The current state of the system, containing
                position and velocity information.
            info (dict[str, Any]): A dictionary containing reference state information,
                specifically the reference vertical velocity.
            action (jax.Array): The action taken, not used in this calculation.

        Returns:
            jax.Array: The computed reward based on the vertical velocity tracking error.
        """
        lin_vel_local = get_local_vec(pipeline_state.xd.vel[0], pipeline_state.x.rot[0])
        lin_vel_z = lin_vel_local[2]
        lin_vel_z_ref = info["state_ref"]["lin_vel"][2]
        error = lin_vel_z - lin_vel_z_ref
        reward = jnp.exp(-self.lin_vel_tracking_sigma * error**2)
        return reward

    def _reward_ang_vel_xy(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates a reward based on the angular velocity in the XY plane.

        This function computes the reward by comparing the current angular velocity in the XY plane to a reference value. The reward is calculated using a Gaussian function of the error between the current and reference angular velocities.

        Args:
            pipeline_state (base.State): The current state of the system, containing rotational and angular velocity information.
            info (dict[str, Any]): A dictionary containing reference state information, specifically the target angular velocity in the XY plane.
            action (jax.Array): The action taken, though not directly used in this function.

        Returns:
            jax.Array: The computed reward based on the angular velocity tracking error.
        """
        ang_vel_local = get_local_vec(pipeline_state.xd.ang[0], pipeline_state.x.rot[0])
        ang_vel_xy = ang_vel_local[:2]
        ang_vel_xy_ref = info["state_ref"]["ang_vel"][:2]
        error = jnp.linalg.norm(ang_vel_xy - ang_vel_xy_ref, axis=-1)
        reward = jnp.exp(-self.ang_vel_tracking_sigma * error**2)
        return reward

    def _reward_penalty_ang_vel_xy(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Penalize angular velocity in the XY plane (roll/pitch rate).

        Matches holosoma's penalty_ang_vel_xy: returns negative sum of squared
        angular velocities in xy. Use with positive weight in config.

        Args:
            pipeline_state (base.State): The current state of the system.
            info (dict[str, Any]): Additional state information (unused).
            action (jax.Array): The action taken (unused).

        Returns:
            jax.Array: Negative sum of squared xy angular velocities.
        """
        ang_vel_local = get_local_vec(pipeline_state.xd.ang[0], pipeline_state.x.rot[0])
        return -jnp.sum(jnp.square(ang_vel_local[:2]))

    def _reward_penalty_orientation(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Penalize non-flat base orientation (roll/pitch).

        Matches holosoma's penalty_orientation: projects gravity into base frame
        and penalizes xy components. When upright, gravity in base frame is [0,0,-1].

        Args:
            pipeline_state (base.State): The current state of the system.
            info (dict[str, Any]): Additional state information (unused).
            action (jax.Array): The action taken (unused).

        Returns:
            jax.Array: Negative sum of squared projected gravity xy components.
        """
        gravity_world = jnp.array([0.0, 0.0, -1.0])
        projected_gravity = get_local_vec(gravity_world, pipeline_state.x.rot[0])
        return -jnp.sum(jnp.square(projected_gravity[:2]))

    def _reward_ang_vel_z(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculate the reward based on the z-component of angular velocity.

        This function computes a reward that measures how closely the z-component of the
        angular velocity of a system matches a reference value. The reward is calculated
        using a Gaussian function of the error between the actual and reference angular
        velocities.

        Args:
            pipeline_state (base.State): The current state of the system, including
                angular velocities and orientations.
            info (dict[str, Any]): A dictionary containing reference states, including
                the reference angular velocity.
            action (jax.Array): The action taken, though not used in this calculation.

        Returns:
            jax.Array: The computed reward based on the angular velocity error.
        """
        ang_vel_local = get_local_vec(pipeline_state.xd.ang[0], pipeline_state.x.rot[0])
        ang_vel_z = ang_vel_local[2]
        ang_vel_z_ref = info["state_ref"]["ang_vel"][2]
        error = ang_vel_z - ang_vel_z_ref
        reward = jnp.exp(-self.ang_vel_tracking_sigma * error**2)
        return reward

    def _reward_feet_contact(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates the reward based on the contact of feet with the ground.

        This function computes the reward by comparing the stance mask from the
        `info` dictionary with the reference state, specifically the last two
        elements of the `state_ref` array. The reward is the sum of matches
        between these two arrays, indicating the number of feet in contact with
        the ground as expected.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): A dictionary containing information about the
                current state, including the 'stance_mask' and 'state_ref'.
            action (jax.Array): The action taken, represented as a JAX array.

        Returns:
            jax.numpy.ndarray: The computed reward as a float32 value.
        """
        reward = jnp.sum(
            info["stance_mask"] == info["state_ref"]["stance_mask"]
        ).astype(jnp.float32)
        return reward

    def _reward_contact_number(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        # Count how many contacts each link has with the floor
        num_contact_points = info["num_contact_points"][0]
        # Penalize any link with < 3 contacts
        reward = -jnp.sum(
            jnp.logical_and(num_contact_points > 0, num_contact_points < 3)
        ).astype(jnp.float32)
        return reward

    def _reward_motor_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates a weighted combination of motor position rewards for all body parts.

        This function computes the reward by evaluating the position errors for leg, arm, neck,
        and waist motors separately, then combines them using configurable weights. Body parts
        not included in ActionConfig.action_parts are masked out (weight = 0).

        Args:
            pipeline_state (base.State): The current state of the system, containing the positions of all motors.
            info (dict[str, Any]): A dictionary containing reference state information, including the desired motor positions.
            action (jax.Array): The action taken, though not used in this reward calculation.

        Returns:
            jax.Array: The calculated weighted combination of motor position rewards.
        """
        weighted_reward = 0.0

        # Leg motor position reward
        leg_motor_pos = pipeline_state.q[self.q_start_idx + self.leg_motor_indices]
        leg_motor_pos_ref = info["state_ref"]["motor_pos"][self.leg_actuator_indices]
        leg_error = leg_motor_pos - leg_motor_pos_ref
        leg_reward = jax.lax.cond(
            self.use_exp_reward, self._tracking_exp, self._tracking_mse, leg_error
        )
        # Only add leg reward if legs are in action_parts
        leg_weight = self.cfg.rewards.leg_weight if "leg" in self.action_parts else 0.0
        weighted_reward += leg_weight * leg_reward

        # Arm motor position reward
        arm_motor_pos = pipeline_state.q[self.q_start_idx + self.arm_motor_indices]
        arm_motor_pos_ref = info["state_ref"]["motor_pos"][self.arm_actuator_indices]
        arm_error = arm_motor_pos - arm_motor_pos_ref
        arm_reward = jax.lax.cond(
            self.use_exp_reward, self._tracking_exp, self._tracking_mse, arm_error
        )
        # Only add arm reward if arms are in action_parts
        arm_weight = self.cfg.rewards.arm_weight if "arm" in self.action_parts else 0.0
        weighted_reward += arm_weight * arm_reward

        # Neck motor position reward
        neck_motor_pos = pipeline_state.q[self.q_start_idx + self.neck_motor_indices]
        neck_motor_pos_ref = info["state_ref"]["motor_pos"][self.neck_actuator_indices]
        neck_error = neck_motor_pos - neck_motor_pos_ref
        neck_reward = jax.lax.cond(
            self.use_exp_reward, self._tracking_exp, self._tracking_mse, neck_error
        )
        # Only add neck reward if neck is in action_parts
        neck_weight = (
            self.cfg.rewards.neck_weight if "neck" in self.action_parts else 0.0
        )
        weighted_reward += neck_weight * neck_reward

        # Waist motor position reward
        waist_motor_pos = pipeline_state.q[self.q_start_idx + self.waist_motor_indices]
        waist_motor_pos_ref = info["state_ref"]["motor_pos"][
            self.waist_actuator_indices
        ]
        waist_error = waist_motor_pos - waist_motor_pos_ref
        waist_reward = jax.lax.cond(
            self.use_exp_reward, self._tracking_exp, self._tracking_mse, waist_error
        )
        # Only add waist reward if waist is in action_parts
        waist_weight = (
            self.cfg.rewards.waist_weight if "waist" in self.action_parts else 0.0
        )
        weighted_reward += waist_weight * waist_reward

        return weighted_reward

    def _reward_collision(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates a negative reward based on collision forces in the environment.

        This function computes a penalty for collisions by evaluating the contact forces
        between objects, excluding the floor. If the force exceeds a threshold, it is
        considered a collision, and a negative reward is accumulated for each collision.

        Args:
            pipeline_state (base.State): The current state of the simulation pipeline.
            info (dict[str, Any]): A dictionary containing information about the current
                simulation step, including contact forces.
            action (jax.Array): The action taken by the agent, not used in this function.

        Returns:
            float: A negative reward representing the penalty for collisions.
        """
        collision_forces = jnp.linalg.norm(info["contact_forces"][1:, 1:], axis=-1)
        collision_contact = collision_forces > 0.1
        reward = -jnp.sum(collision_contact.astype(jnp.float32))
        return reward

    def _reward_motor_torque(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on motor torque.

        This function computes a reward by evaluating the squared error of the motor torque
        and returning its negative mean. The reward is designed to penalize high torque values.

        Args:
            pipeline_state (base.State): The current state of the pipeline, containing actuator forces.
            info (dict[str, Any]): Additional information, not used in this function.
            action (jax.Array): The action taken, not used in this function.

        Returns:
            jax.Array: The calculated reward as a negative mean of the squared torque error.
        """
        torque = pipeline_state.qfrc_actuator[self.qd_start_idx + self.motor_indices]
        reward = -jnp.mean(torque**2)
        return reward

    def _reward_energy(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the energy-based reward for a given pipeline state and action.

        This function computes the reward based on the energy consumption of the actuators
        in the system. It calculates the energy as the product of torque and motor velocity,
        then computes the error as the square of the energy. The reward is the negative mean
        of this error, encouraging actions that minimize energy consumption.

        Args:
            pipeline_state (base.State): The current state of the pipeline, containing
                information about actuator forces and velocities.
            info (dict[str, Any]): Additional information that might be used for reward
                calculation (not used in this function).
            action (jax.Array): The action taken, represented as a JAX array.

        Returns:
            jax.Array: The calculated reward as a JAX array, representing the negative mean
            of the squared energy error.
        """
        torque = pipeline_state.qfrc_actuator[self.qd_start_idx + self.motor_indices]
        motor_vel = pipeline_state.qvel[self.qd_start_idx + self.motor_indices]
        energy = torque * motor_vel
        reward = -jnp.mean(energy**2)
        return reward

    def _reward_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the rate of change of actions.

        This function computes a reward by evaluating the squared difference between the current and last actions, averaged over all actuators. The reward is negative, encouraging minimal change in actions.

        Args:
            pipeline_state (base.State): The current state of the pipeline, not used in this function.
            info (dict[str, Any]): A dictionary containing the last action taken, with key "last_act".
            action (jax.Array): The current action array.

        Returns:
            jax.Array: A scalar reward representing the negative mean squared error of action changes.
        """
        error = jnp.square(action - info["last_act"])
        reward = -jnp.mean(error)
        return reward

    def _reward_penalty_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Penalize changes in actions between steps (sum version).

        Matches holosoma's penalty_action_rate: returns negative sum of squared
        action differences. Use with positive weight in config.

        Args:
            pipeline_state (base.State): The current state of the pipeline (unused).
            info (dict[str, Any]): A dictionary containing the last action taken.
            action (jax.Array): The current action array.

        Returns:
            jax.Array: Negative sum of squared action changes.
        """
        return -jnp.sum(jnp.square(action - info["last_act"]))

    def _reward_penalty_pose(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Penalize deviation from default pose for controlled joints.

        Matches holosoma's pose reward but only for action_parts (controlled DOFs).
        Returns negative weighted sum of squared position errors from default pose.
        Weights are configurable via RewardsConfig.pose_weights (defaults to uniform 1.0).

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): A dictionary containing episode information.
            action (jax.Array): The current action array.

        Returns:
            jax.Array: Negative weighted sum of squared pose errors for controlled joints.
        """
        # Get current motor positions for controlled joints only
        motor_pos = pipeline_state.q[self.q_start_idx + self.motor_indices]
        controlled_motor_pos = motor_pos[self.action_mask]

        # Get default positions for controlled joints
        default_controlled_pos = self.default_motor_pos[self.action_mask]

        # Calculate weighted squared errors
        pose_error = jnp.square(controlled_motor_pos - default_controlled_pos)
        weighted_error = pose_error * self.pose_weights

        # Penalize deviation from default
        return -jnp.sum(weighted_error)

    def _reward_penalty_close_feet_xy(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Penalize when feet are too close together in xy plane.

        Matches holosoma's penalty_close_feet_xy: returns -1.0 if lateral distance
        between feet (perpendicular to forward direction) is below threshold.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): A dictionary containing episode information.
            action (jax.Array): The current action array.

        Returns:
            jax.Array: -1.0 if feet too close, 0.0 otherwise.
        """
        # Get feet positions
        left_foot_pos = pipeline_state.x.pos[self.feet_link_ids[0]]
        right_foot_pos = pipeline_state.x.pos[self.feet_link_ids[1]]

        # Get base forward direction from quaternion
        base_quat = pipeline_state.x.rot[0]
        forward_local = jnp.array([1.0, 0.0, 0.0])
        base_forward = R.from_quat(
            jnp.concatenate([base_quat[1:], base_quat[:1]])
        ).apply(forward_local)
        base_yaw = jnp.arctan2(base_forward[1], base_forward[0])

        # Calculate perpendicular (lateral) distance in base-local coordinates
        feet_diff = left_foot_pos[:2] - right_foot_pos[:2]
        feet_lateral_dist = jnp.abs(
            jnp.cos(base_yaw) * feet_diff[1] - jnp.sin(base_yaw) * feet_diff[0]
        )

        # Return penalty when feet are too close
        too_close = feet_lateral_dist < self.close_feet_threshold
        return jnp.where(too_close, -1.0, 0.0)

    def _reward_survival(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates a survival reward based on the pipeline state and action taken.

        The reward is negative if the episode is marked as done before reaching the
        specified number of reset steps, encouraging survival until the reset threshold.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): A dictionary containing episode information, including
                whether the episode is done and the current step count.
            action (jax.Array): The action taken at the current step.

        Returns:
            jax.Array: A float32 array representing the survival reward.
        """
        return -info["done"].astype(jnp.float32)

    def _reward_alive(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Constant reward for staying alive.

        Matches holosoma's alive reward: returns 1.0 every timestep.
        This provides a dense signal encouraging longer episodes.

        Args:
            pipeline_state (base.State): The current state of the pipeline (unused).
            info (dict[str, Any]): A dictionary containing episode information (unused).
            action (jax.Array): The action taken at the current step (unused).

        Returns:
            jax.Array: 1.0 constant reward.
        """
        return jnp.float32(1.0)

    # DeepMimic specific reward functions; make sure to have _init_site_indices() in your custom env to have site tracking
    def _reward_body_quat(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates body quaternion tracking reward with weighted categories. (The "r_pose" in DeepMimic)

        This function computes a reward based on the alignment of body quaternions with reference poses,
        using weighted categories for different body parts (neck, waist, legs, arms). Only body parts
        included in action_parts are considered in the final weighted sum.

        Args:
            pipeline_state (base.State): The current state of the system, containing body rotations.
            info (dict[str, Any]): A dictionary containing reference state information, including body quaternions.
            action (jax.Array): The action taken, though not used in this calculation.

        Returns:
            jax.Array: The computed weighted body quaternion tracking reward.
        """
        # Body quaternion tracking with weighted categories
        body_quat = pipeline_state.x.rot  # Shape: (45,) - excludes world body
        body_quat_ref = info["state_ref"]["body_quat"][
            1:
        ]  # Reference body quaternions, skipping world body

        # Check if reference data is in robot-relative frame
        if self.is_robot_relative_frame:
            # Need to transform current quaternions from world frame to robot-relative frame
            torso_quat_world = pipeline_state.x.rot[
                0
            ]  # Torso quaternion in world frame [w,x,y,z]

            # Convert to xyzw format and create rotation object
            torso_quat_xyzw = jnp.concatenate(
                [torso_quat_world[1:], torso_quat_world[:1]]
            )
            torso_rot_world = R.from_quat(torso_quat_xyzw)
            torso_rot_inv = torso_rot_world.inv()

            # Transform all body quaternions to robot-relative frame
            def transform_quat_to_relative(body_quat_w):
                # Convert to xyzw format for R.from_quat
                body_quat_xyzw = jnp.concatenate([body_quat_w[1:], body_quat_w[:1]])
                body_rot_world = R.from_quat(body_quat_xyzw)
                # Transform to robot-relative: body_relative = body_world * torso_world.inv()
                body_rot_relative = body_rot_world * torso_rot_inv
                # Convert back to [w,x,y,z] format
                quat_xyzw = body_rot_relative.as_quat()
                return jnp.concatenate([quat_xyzw[3:], quat_xyzw[:3]])

            # Apply transformation to all body quaternions using vmap
            body_quat_relative = jax.vmap(transform_quat_to_relative)(body_quat)
        else:
            # Reference data is in global frame, use directly
            body_quat_relative = body_quat

        # Ensure shapes match
        min_bodies = min(body_quat_relative.shape[0], body_quat_ref.shape[0])
        body_quat_relative = body_quat_relative[:min_bodies]
        body_quat_ref = body_quat_ref[:min_bodies]

        neck_bodies = list(
            range(0, 6)
        )  # neck_yaw_gear_drive, neck_yaw_link, head, neck_pitch_plate, neck_rod, neck_rod_2
        waist_bodies = list(
            range(6, 10)
        )  # waist_gears, pelvis_link, waist_gear_drive, waist_gear_drive_2
        leg_bodies = list(
            range(10, 24)
        )  # all hip, knee, ankle links for both legs (14 bodies)
        arm_bodies = list(
            range(24, 44)
        )  # all shoulder, elbow, wrist links and hands for both arms (20 bodies)

        # Helper function to compute pose reward for a body category
        def compute_category_reward(body_indices):
            if len(body_indices) == 0:
                return 0.0

            total_loss = 0.0
            valid_bodies = 0

            for i in body_indices:
                if i < min_bodies:  # Ensure index is valid
                    quat = body_quat_relative[i]
                    quat_ref = body_quat_ref[i]

                    # Normalize quaternions
                    quat_ref = quat_ref / jnp.linalg.norm(quat_ref)
                    quat = quat / jnp.linalg.norm(quat)

                    # Compute normalized quaternion difference using dot product
                    dot_product = jnp.sum(quat * quat_ref)
                    # Ensure we take the shorter rotation (handle quaternion double cover)
                    dot_product = jnp.abs(dot_product)
                    angle_diff = 2.0 * jnp.arccos(jnp.clip(dot_product, 0.0, 1.0))

                    total_loss += angle_diff**2
                    valid_bodies += 1

            if valid_bodies > 0:
                avg_loss = total_loss / valid_bodies
                return jnp.exp(-2.0 * avg_loss)
            else:
                return 0.0

        # Compute rewards for each category
        neck_reward = compute_category_reward(neck_bodies)
        waist_reward = compute_category_reward(waist_bodies)
        leg_reward = compute_category_reward(leg_bodies)
        arm_reward = compute_category_reward(arm_bodies)

        # Combine using weighted sum (same weights as motor tracking)
        # Only include body parts that are in action_parts
        neck_weight = (
            self.cfg.rewards.neck_weight if "neck" in self.action_parts else 0.0
        )
        waist_weight = (
            self.cfg.rewards.waist_weight if "waist" in self.action_parts else 0.0
        )
        leg_weight = self.cfg.rewards.leg_weight if "leg" in self.action_parts else 0.0
        arm_weight = self.cfg.rewards.arm_weight if "arm" in self.action_parts else 0.0

        weighted_reward = (
            neck_weight * neck_reward
            + waist_weight * waist_reward
            + leg_weight * leg_reward
            + arm_weight * arm_reward
        )

        # Normalize by total weight to keep reward magnitude similar
        total_weight = neck_weight + waist_weight + leg_weight + arm_weight

        # Avoid division by zero if no body parts are active
        return jnp.where(total_weight > 0.0, weighted_reward / total_weight, 0.0)

    def _reward_site_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates site-based end-effector position tracking reward. (The "r_ee" in DeepMimic)

        This function computes a reward based on the position tracking of end-effector sites
        using recorded site data.

        Args:
            pipeline_state (base.State): The current state of the system, containing site positions.
            info (dict[str, Any]): A dictionary containing reference state information, including site positions.
            action (jax.Array): The action taken, though not used in this calculation.

        Returns:
            jax.Array: The computed reward based on end-effector position tracking.
        """
        # Site-based end-effector position tracking using recorded site data
        site_pos_world = pipeline_state.site_xpos  # World frame site positions
        site_pos_ref = info["state_ref"]["site_pos"]  # Reference site positions

        # Apply spawn offsets for obstacle randomization (only in global frame mode)
        if not self.is_robot_relative_frame:
            spawn_y_offset = info.get("spawn_y_offset", 0.0)
            spawn_z_offset = info.get("spawn_z_offset", 0.0)
            position_offset = jnp.array([0.0, spawn_y_offset, spawn_z_offset])
            site_pos_ref = site_pos_ref + position_offset

        # Compute position error for the 4 recorded end-effector sites
        total_loss = 0.0
        num_sites = 0

        for recorded_idx, site_name in self.recorded_site_mapping.items():
            robot_site_idx = self.end_effector_site_indices.get(site_name, -1)

            if robot_site_idx >= 0:
                # Current position from robot simulation
                pos_current = site_pos_world[robot_site_idx]

                if self.is_robot_relative_frame:
                    # Need to transform current position from world frame to robot-relative frame
                    torso_pos_world = pipeline_state.x.pos[
                        0
                    ]  # Torso position in world frame
                    torso_quat_world = pipeline_state.x.rot[
                        0
                    ]  # Torso quaternion in world frame [w,x,y,z]

                    # Convert to xyzw format and create rotation object
                    torso_quat_xyzw = jnp.concatenate(
                        [torso_quat_world[1:], torso_quat_world[:1]]
                    )
                    torso_rot_world = R.from_quat(torso_quat_xyzw)
                    torso_rot_inv = torso_rot_world.inv()

                    # Transform to robot-relative frame using R.apply
                    pos_diff = pos_current - torso_pos_world
                    pos_current_relative = torso_rot_inv.apply(pos_diff)
                else:
                    # Reference data is in global frame, use current position directly
                    pos_current_relative = pos_current

                pos_ref = site_pos_ref[recorded_idx, :3]

                # Compute position error
                error = jnp.sum(jnp.square(pos_current_relative - pos_ref))
                total_loss += error
                num_sites += 1

        # Average loss across all tracked end-effector sites
        if num_sites > 0:
            avg_loss = total_loss / num_sites
            reward = jnp.exp(-25.0 * avg_loss)
        else:
            reward = 0.0

        return reward

    def _reward_body_lin_vel(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates body linear velocity tracking reward.

        This function computes a reward based on the linear velocity tracking of all bodies
        with equal weights.

        Args:
            pipeline_state (base.State): The current state of the system, containing body velocities.
            info (dict[str, Any]): A dictionary containing reference state information, including body linear velocities.
            action (jax.Array): The action taken, though not used in this calculation.

        Returns:
            jax.Array: The computed reward based on body linear velocity tracking.
        """
        # Body linear velocity tracking with equal weights for all bodies
        body_vel_world = pipeline_state.xd.vel
        body_vel_ref = info["state_ref"]["body_lin_vel"][1:]

        if self.is_robot_relative_frame:
            # Need to transform current velocities from world frame to robot-relative frame
            torso_quat_world = pipeline_state.x.rot[0]  # [w, x, y, z]
            # Convert to xyzw format and create rotation object
            torso_quat_xyzw = jnp.concatenate(
                [torso_quat_world[1:], torso_quat_world[:1]]
            )
            torso_rot_world = R.from_quat(torso_quat_xyzw)
            torso_rot_inv = torso_rot_world.inv()

            # Transform all body velocities to robot-relative frame
            body_vel_relative = jax.vmap(torso_rot_inv.apply)(body_vel_world)
        else:
            # Reference data is in global frame, use current velocities directly
            body_vel_relative = body_vel_world

        # Compute velocity error for all bodies with equal weight
        total_loss = 0.0
        num_bodies = body_vel_ref.shape[0]
        for i in range(num_bodies):
            vel = body_vel_relative[i]
            vel_ref = body_vel_ref[i]
            error = jnp.sum(jnp.square(vel - vel_ref))
            total_loss += error

        # Average loss across all bodies
        avg_loss = total_loss / num_bodies
        reward = jnp.exp(-0.1 * avg_loss)

        return reward

    def _reward_body_ang_vel(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates body angular velocity tracking reward. (The "r_vel" in DeepMimic)

        This function computes a reward based on the angular velocity tracking of all bodies
        with equal weights.

        Args:
            pipeline_state (base.State): The current state of the system, containing body angular velocities.
            info (dict[str, Any]): A dictionary containing reference state information, including body angular velocities.
            action (jax.Array): The action taken, though not used in this calculation.

        Returns:
            jax.Array: The computed reward based on body angular velocity tracking.
        """
        # Body angular velocity tracking with equal weights for all bodies
        body_ang_vel_world = pipeline_state.xd.ang
        body_ang_vel_ref = info["state_ref"]["body_ang_vel"][1:]
        if self.is_robot_relative_frame:
            # Need to transform current angular velocities from world frame to robot-relative frame
            torso_quat_world = pipeline_state.x.rot[0]  # [w, x, y, z]
            # Convert to xyzw format and create rotation object
            torso_quat_xyzw = jnp.concatenate(
                [torso_quat_world[1:], torso_quat_world[:1]]
            )
            torso_rot_world = R.from_quat(torso_quat_xyzw)
            torso_rot_inv = torso_rot_world.inv()

            # Transform all body angular velocities to robot-relative frame
            body_ang_vel_relative = jax.vmap(torso_rot_inv.apply)(body_ang_vel_world)
        else:
            # Reference data is in global frame, use current angular velocities directly
            body_ang_vel_relative = body_ang_vel_world

        # Compute angular velocity error for all bodies with equal weight
        total_loss = 0.0
        num_bodies = body_ang_vel_ref.shape[0]
        for i in range(num_bodies):
            vel = body_ang_vel_relative[i]
            vel_ref = body_ang_vel_ref[i]
            error = jnp.sum(jnp.square(vel - vel_ref))
            total_loss += error

        # Average loss across all bodies
        avg_loss = total_loss / num_bodies
        reward = jnp.exp(-0.1 * avg_loss)

        return reward
