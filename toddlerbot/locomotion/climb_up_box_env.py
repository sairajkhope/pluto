from typing import Any, Optional

import jax
import jax.numpy as jnp

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.reference.climb_up_box_ref import ClimbUpBoxReference
from toddlerbot.sim.robot import Robot


class ClimbUpBoxEnv(MJXEnv, env_name="climb_up_box"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        fixed_base: bool = False,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        motion_ref = ClimbUpBoxReference(
            robot,
            cfg.sim.timestep * cfg.action.n_frames,
            fixed_base=fixed_base,
            playback_speed=cfg.sim.playback_speed,
        )

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

    # TODO: As for climbing up box, the stance_mask needs to be calculated based on the floor or the box.
    # def _reward_feet_slip(
    #     self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    # ) -> jax.Array:
    #     """Penalizes foot velocity for feet that are in contact with the ground."""
    #     feet_speed = pipeline_state.xd.vel[self.feet_link_ids]
    #     # Penalize only horizontal velocity
    #     feet_speed_square = jnp.sum(jnp.square(feet_speed[:, :2]), axis=-1)
    #     # The stance_mask comes from climb_wall_ref.py
    #     reward = -jnp.sum(feet_speed_square * info["stance_mask"])
    #     return reward
