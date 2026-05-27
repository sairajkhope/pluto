"""Get up prone locomotion environment for ToddlerBot.

This module provides the GetUpProneEnv class for training ToddlerBot in get up movements from prone position.
The environment extends MJXEnv with get up prone-specific motion references and command sampling.
"""

from typing import Any, Optional

import jax

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.reference.get_up_prone_ref import GetUpProneReference
from toddlerbot.sim.robot import Robot


class GetUpProneEnv(MJXEnv, env_name="get_up_prone"):
    """Environment for training get up locomotion with ToddlerBot."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        fixed_base: bool = False,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        """Initialize the get up prone environment with motion reference setup."""
        motion_ref = GetUpProneReference(
            robot, cfg.sim.timestep * cfg.action.n_frames, fixed_base=fixed_base
        )
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
        """Sample command for get up mode selection."""
        if last_command is None:
            num_modes = len(self.motion_ref.get_up_mode_list)
            get_up_mode = jax.random.randint(rng, (1,), 0, num_modes)
            return get_up_mode

        return last_command
