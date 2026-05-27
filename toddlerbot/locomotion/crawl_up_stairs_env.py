from typing import Any, Optional

import jax
import jax.numpy as jnp

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.reference.crawl_up_stairs_ref import CrawlUpStairsReference
from toddlerbot.sim.robot import Robot


class CrawlUpStairsEnv(MJXEnv, env_name="crawl_up_stairs"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        fixed_base: bool = False,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        motion_ref = CrawlUpStairsReference(
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
