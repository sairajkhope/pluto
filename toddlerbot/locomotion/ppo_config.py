"""PPO configuration settings for ToddlerBot training.

This module defines the PPOConfig dataclass containing hyperparameters and
configuration settings for Proximal Policy Optimization (PPO) training.
"""

from dataclasses import dataclass

import gin


@gin.configurable
@dataclass
class PPOConfig:
    """Data class for storing PPO hyperparameters."""

    wandb_project: str = "ToddlerBot"
    wandb_entity: str = "toddlerbot"
    policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    use_rnn: bool = False  # specifc to rsl_rl
    rnn_type: str = "lstm"
    rnn_hidden_size: int = 512
    rnn_num_layers: int = 1
    activation: str = "elu"
    distribution_type: str = "normal"
    noise_std_type: str = "log"
    init_noise_std: float = 0.5
    num_timesteps: int = 500_000_000
    num_evals: int = 100
    episode_length: int = 0  # Auto-set by env from motion file; falls back to gin config if unavailable. Specify in gin or defaults to 0
    unroll_length: int = 20
    num_updates_per_batch: int = 4
    discounting: float = 0.97
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    normalize_advantage: bool = True
    normalize_observation: bool = False
    learning_rate: float = 3e-5
    entropy_cost: float = 5e-4  # 1e-3
    clipping_epsilon: float = 0.2
    num_envs: int = 4096  # 4096
    render_nums: int = 20
    batch_size: int = 512  # 512
    num_minibatches: int = 16  # 16
    seed: int = 0
