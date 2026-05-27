from typing import Any, Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy
from brax import base

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.reference.crawl_only_ref import CrawlOnlyReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import R


class CrawlRotateEnv(MJXEnv, env_name="crawl_rotate"):
    """Environment for training robot to rotate in place using crawling motion.

    Uses crawl_only reference motion but adds custom reward to:
    - Encourage rotation around Z axis at target angular velocity
    - Penalize XY position drift from origin
    - Preserve crawl joint patterns through existing robot-relative rewards
    """

    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        fixed_base: bool = False,
        add_domain_rand: bool = True,
        # target_angular_velocity: float = -0.4,  # rad/s
        target_angular_velocity: float = -0.2,  # rad/s
        rotation_start_delay: float = 1.0,  # seconds to wait before starting rotation
        **kwargs: Any,
    ):
        # NOTE: Use the same crawl_only reference motion
        motion_ref = CrawlOnlyReference(
            robot,
            cfg.sim.timestep * cfg.action.n_frames,
            fixed_base=fixed_base,
            playback_speed=cfg.sim.playback_speed,
        )

        self.target_angular_velocity = target_angular_velocity
        self.rotation_start_delay = rotation_start_delay

        # Dynamically set the episode length based on the length of the motion reference and playback speed
        # This variable gets assigned to PPOConfig.episode_length
        self.episode_length = motion_ref.full_episode_length

        super().__init__(
            name,
            robot,
            cfg,
            motion_ref,
            fixed_base=fixed_base,
            add_domain_rand=add_domain_rand,
            **kwargs,
        )

    def _sample_command(
        self, rng: jax.Array, last_command: Optional[jax.Array] = None
    ) -> jax.Array:
        return jnp.zeros(3)

    def reset(self, rng: jax.Array) -> base.State:
        """Override reset to randomize initial heading and store it in info.

        Randomizes robot's initial heading between 0 and π to ensure
        the policy learns to rotate from any starting orientation.
        """
        # Split RNG for yaw randomization
        rng, rng_yaw = jax.random.split(rng)

        # Call parent reset first
        state = super().reset(rng)

        # Randomize initial heading (yaw) between -π and 0 (CW direction)
        random_yaw = jax.random.uniform(rng_yaw, (), minval=-jnp.pi, maxval=0.0)

        # Store initial yaw in state.info for reward calculation
        state.info["initial_yaw"] = random_yaw

        initial_xy = state.pipeline_state.x.pos[0, :2]
        state.info["initial_xy"] = initial_xy

        # Get current qpos and qvel
        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd

        # Get current torso quaternion from reference motion (crawling pose)
        current_quat_w = qpos[3]  # [w, x, y, z]
        current_quat_x = qpos[4]
        current_quat_y = qpos[5]
        current_quat_z = qpos[6]

        # Create yaw rotation quaternion: [cos(θ/2), 0, 0, sin(θ/2)]
        half_yaw = random_yaw * 0.5
        yaw_quat_w = jnp.cos(half_yaw)
        yaw_quat_x = 0.0
        yaw_quat_y = 0.0
        yaw_quat_z = jnp.sin(half_yaw)

        # Multiply quaternions: new_quat = yaw_quat * current_quat
        # Quaternion multiplication formula
        new_quat_w = (
            yaw_quat_w * current_quat_w
            - yaw_quat_x * current_quat_x
            - yaw_quat_y * current_quat_y
            - yaw_quat_z * current_quat_z
        )
        new_quat_x = (
            yaw_quat_w * current_quat_x
            + yaw_quat_x * current_quat_w
            + yaw_quat_y * current_quat_z
            - yaw_quat_z * current_quat_y
        )
        new_quat_y = (
            yaw_quat_w * current_quat_y
            - yaw_quat_x * current_quat_z
            + yaw_quat_y * current_quat_w
            + yaw_quat_z * current_quat_x
        )
        new_quat_z = (
            yaw_quat_w * current_quat_z
            + yaw_quat_x * current_quat_y
            - yaw_quat_y * current_quat_x
            + yaw_quat_z * current_quat_w
        )

        # Update qpos with COMPOSED quaternion (reference pose + yaw rotation)
        qpos = qpos.at[3].set(new_quat_w)
        qpos = qpos.at[4].set(new_quat_x)
        qpos = qpos.at[5].set(new_quat_y)
        qpos = qpos.at[6].set(new_quat_z)

        # CRITICAL: Reinitialize pipeline state with modified qpos
        # (not just replace, need to recompute physics transforms!)
        pipeline_state = self.pipeline_init(qpos, qvel)

        # Recompute observations from the new pipeline_state
        # (parent reset computed obs from OLD pose before yaw rotation)
        obs_history = jnp.zeros(self.num_obs_history * self.obs_size)
        privileged_obs_history = jnp.zeros(
            self.num_privileged_obs_history * self.privileged_obs_size
        )
        obs, info = self._get_obs(
            pipeline_state,
            state.info,
            obs_history,
            privileged_obs_history,
        )

        # Return updated state with new pipeline state AND observations
        return state.replace(pipeline_state=pipeline_state, obs=obs)

    def _reward_heading_tracking(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking target heading (rotation).

        Uses torso→head vector to avoid 180° ambiguity (head can't be in two places).

        Args:
            pipeline_state (base.State): The current state of the system.
            info (dict[str, Any]): Additional information including elapsed time and initial yaw.
            action (jax.Array): The action taken, though not used in this calculation.

        Returns:
            jax.Array: Reward in [0, 1] for matching target heading.
        """
        # Get current heading from torso→head vector (avoids 180° ambiguity)
        # Body indices in pipeline_state.x.pos (world body excluded):
        #   0 = torso (MuJoCo body 1)
        #   3 = head (MuJoCo body 4)
        torso_xy = pipeline_state.x.pos[0, :2]
        head_xy = pipeline_state.x.pos[3, :2]

        # Heading = direction robot is facing (torso → head)
        heading_vec = head_xy - torso_xy
        current_yaw = jnp.arctan2(heading_vec[1], heading_vec[0])

        # Get initial yaw (stored at reset)
        initial_yaw = info.get("initial_yaw", 0.0)

        # Calculate target heading RELATIVE to initial heading
        elapsed_time = info.get("step", 0) * self.dt
        rotation_time = jnp.where(
            elapsed_time < self.rotation_start_delay,
            0.0,  # Don't start rotating yet
            elapsed_time - self.rotation_start_delay,  # Compensate for delay
        )

        # Target yaw accumulates based on angular velocity
        # target_angular_velocity > 0 → CCW rotation (yaw increases)
        # target_angular_velocity < 0 → CW rotation (yaw decreases)
        target_yaw = initial_yaw + (rotation_time * self.target_angular_velocity)

        # Wrap target_yaw to [-π, π] for cleaner values
        target_yaw = jnp.arctan2(jnp.sin(target_yaw), jnp.cos(target_yaw))

        # Compute heading error (shortest angular distance)
        # Wrapping ensures we always take the shortest path
        heading_error = current_yaw - target_yaw
        heading_error = jnp.arctan2(jnp.sin(heading_error), jnp.cos(heading_error))

        # Reward for matching target heading
        reward = jnp.exp(-10.0 * heading_error**2)

        return reward

    def _reward_xy_drift(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Penalty for XY position drift from origin.

        Returns NEGATIVE values to penalize drift (like survival reward).

        Args:
            pipeline_state (base.State): The current state of the system.
            info (dict[str, Any]): Additional information (not used).
            action (jax.Array): The action taken (not used).

        Returns:
            jax.Array: Penalty in [-1, 0] for XY drift from origin.
        """
        # Get XY position
        # xy_position = pipeline_state.qpos[:2]
        # xy_drift = jnp.linalg.norm(xy_position)
        # Current world XY position of robot root body
        current_xy = pipeline_state.x.pos[0, :2]

        # Stored initial position from reset()
        initial_xy = info.get("initial_xy", jnp.zeros(2))

        # Compute drift magnitude
        xy_drift = jnp.linalg.norm(current_xy - initial_xy)

        # Return NEGATIVE penalty (0 when at origin, -1 when far)
        penalty = jnp.exp(-1000.0 * xy_drift**2) - 1.0

        return penalty

    def _reward_angular_velocity(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for matching target angular velocity around Z-axis.

        Provides immediate feedback on rotation rate (complementary to heading tracking).

        Args:
            pipeline_state (base.State): The current state of the system.
            info (dict[str, Any]): Additional information including elapsed time.
            action (jax.Array): The action taken (not used).

        Returns:
            jax.Array: Reward in [0, 1] for matching target angular velocity.
        """
        # Get Z-component of angular velocity in world frame (vertical axis)
        # This measures yaw rate (rotation in XY plane), not rotation around tilted body axis
        current_ang_vel_z = pipeline_state.xd.ang[0, 2]  # World-frame Z component

        # Add delay before rotation starts (match heading tracking)
        elapsed_time = info.get("step", 0) * self.dt
        target_ang_vel = jnp.where(
            elapsed_time < self.rotation_start_delay, 0.0, self.target_angular_velocity
        )

        # Compute error and reward
        ang_vel_error = current_ang_vel_z - target_ang_vel
        reward = jnp.exp(-5.0 * ang_vel_error**2)

        return reward

    def visualize_current_heading(
        self, renderer, torso_pos, head_pos, line_width=8, alpha=1.0
    ):
        """Visualize current robot heading as line from torso to head (ground projection).

        Args:
            renderer: MuJoCo renderer
            torso_pos: Torso XY position (numpy array [x, y])
            head_pos: Head XY position (numpy array [x, y])
            line_width: Width of the line
            alpha: Transparency
        """
        # Project positions to ground level (slightly above for visibility)
        p1 = numpy.array([torso_pos[0], torso_pos[1], 0.05])
        p2 = numpy.array([head_pos[0], head_pos[1], 0.05])

        # Draw cyan line for current heading
        i = renderer.scene.ngeom
        geom = renderer.scene.geoms[i]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=numpy.zeros(3),
            pos=numpy.zeros(3),
            mat=numpy.eye(3).flatten(),
            rgba=[0, 1, 1, alpha],  # cyan
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, line_width, p1, p2)
        renderer.scene.ngeom += 1

    def visualize_target_heading(self, renderer, pos, yaw, axis_len=0.3, alpha=0.8):
        """Visualize target heading coordinate frame at desired XY position.

        Shows RGB axes (X=red, Y=green, Z=blue) oriented at target yaw angle.

        Args:
            renderer: MuJoCo renderer
            pos: XY position (numpy array [x, y]), Z will be set to ground level
            yaw: Target heading angle in radians
            axis_len: Length of axis arrows
            alpha: Transparency of axes
        """
        # Create rotation from yaw angle (rotation around Z-axis)
        # Yaw=0 → facing +X, Yaw=π/2 → facing +Y
        rot = R.from_euler("z", yaw)

        # Set Z to ground level (slightly above for visibility)
        pos_3d = numpy.array([pos[0], pos[1], 0.05])

        colors = {
            "x": [1, 0, 0, alpha],  # red
            "y": [0, 1, 0, alpha],  # green
            "z": [0, 0, 1, alpha],  # blue
        }
        axes = {
            "x": rot.apply(numpy.array([axis_len, 0, 0])),
            "y": rot.apply(numpy.array([0, axis_len, 0])),
            "z": rot.apply(numpy.array([0, 0, axis_len])),
        }

        for key in ["x", "y", "z"]:
            p1 = pos_3d
            p2 = pos_3d + axes[key]
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

    def render(
        self,
        states: list[base.State],
        height: int = 240,
        width: int = 320,
        camera: str | None = None,
    ):
        """Render environment states with target heading visualization.

        Shows target heading as RGB coordinate frame at initial XY position.

        Args:
            states: List of Brax states to render
            height: Image height in pixels
            width: Image width in pixels
            camera: Camera name or ID (-1 for default tracking camera)

        Returns:
            List of rendered images (numpy arrays)
        """
        renderer = mujoco.Renderer(self.sys.mj_model, height=height, width=width)
        camera = camera or -1

        image_list = []
        for state in states:
            d = mujoco.MjData(self.sys.mj_model)
            d.qpos, d.qvel = state.pipeline_state.q, state.pipeline_state.qd
            mujoco.mj_forward(self.sys.mj_model, d)
            renderer.update_scene(d, camera=camera)

            # Extract target yaw from state info
            # Recompute target_yaw using same logic as reward function
            initial_yaw = state.info.get("initial_yaw", 0.0)
            # Compute elapsed time from step count (state.info["time"] doesn't exist!)
            elapsed_time = state.info.get("step", 0) * self.dt

            if elapsed_time < self.rotation_start_delay:
                rotation_time = 0.0
            else:
                rotation_time = elapsed_time - self.rotation_start_delay

            target_yaw = initial_yaw + (rotation_time * self.target_angular_velocity)
            target_yaw = numpy.arctan2(numpy.sin(target_yaw), numpy.cos(target_yaw))

            # Get initial XY position (where robot should stay centered)
            initial_xy = state.info.get("initial_xy", numpy.zeros(2))

            # Visualize target heading at initial XY position
            self.visualize_target_heading(renderer, initial_xy, target_yaw)

            # Visualize current heading (torso→head line on ground)
            # Body indices: 0=torso, 3=head (world body excluded from pipeline_state)
            torso_xy = numpy.array(state.pipeline_state.x.pos[0, :2])
            head_xy = numpy.array(state.pipeline_state.x.pos[3, :2])
            self.visualize_current_heading(renderer, torso_xy, head_xy)

            image_list.append(renderer.render())

        return image_list
