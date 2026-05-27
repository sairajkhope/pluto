"""Visualize reward functions in MJX locomotion environments.

This module tests and visualizes reward function behavior in JAX-accelerated
MuJoCo environments for locomotion tasks.
"""

import os

os.environ["USE_JAX"] = "true"

import argparse
import importlib
import math
import pkgutil
from typing import Dict, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from brax.base import Contact, Motion, Transform
from brax.envs.base import State
from mujoco import mjx
from tqdm import tqdm

from toddlerbot.locomotion.mjx_env import get_env_class, get_env_config
from toddlerbot.sim.robot import Robot


def dynamic_import_envs(env_package: str):
    """Imports all modules from a specified package.

    This function dynamically imports all modules within a given package, allowing their contents to be accessed programmatically. It is useful for loading environment configurations or plugins from a specified package directory.

    Args:
        env_package (str): The name of the package from which to import all modules.
    """
    package = importlib.import_module(env_package)
    package_path = package.__path__

    # Iterate over all modules in the given package directory
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        full_module_name = f"{env_package}.{module_name}"
        importlib.import_module(full_module_name)


# Call this to import all policies dynamically
dynamic_import_envs("toddlerbot.locomotion")


def main(env, reward_names) -> None:
    """Run reward visualization for specified reward functions."""
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    action = jnp.zeros(env.action_size)

    reward_dict: Dict[str, List[float]] = {name: [] for name in reward_names}
    num_steps = getattr(env.motion_ref, "n_frames", 1000)

    def forward(data):
        data = mjx.forward(env.sys, data)
        q, qd = data.qpos, data.qvel
        x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[env.sys.body_rootid[1:]]
        offset = Transform.create(pos=offset)
        xd = offset.vmap().do(cvel)

        if data.ncon > 0:
            mjx_contact = data._impl.contact if hasattr(data, "_impl") else data.contact
            elasticity = jnp.zeros(mjx_contact.pos.shape[0])
            body1 = jnp.array(env.sys.geom_bodyid)[mjx_contact.geom1] - 1
            body2 = jnp.array(env.sys.geom_bodyid)[mjx_contact.geom2] - 1
            link_idx = (body1, body2)
            contact = Contact(
                link_idx=link_idx, elasticity=elasticity, **mjx_contact.__dict__
            )
            data = data.replace(contact=contact)

        pipeline_state = data.replace(q=q, qd=qd, x=x, xd=xd)
        contact_info = {}
        (
            contact_forces,
            num_contact_points,
            stance_mask,
        ) = env._solve_contact(pipeline_state)

        contact_info["contact_forces"] = contact_forces
        contact_info["num_contact_points"] = num_contact_points
        contact_info["stance_mask"] = stance_mask

        return pipeline_state, contact_info

    forward_fn = jax.jit(forward)

    for step_idx in tqdm(range(num_steps), desc="Running simulation"):
        time_curr = step_idx * env.dt
        info = state.info

        state_ref = env.motion_ref.get_state_ref(
            time_curr,
            info["command"],
            info["state_ref"],
            info.get("init_idx", 0),
        )
        info["state_ref"] = state_ref
        data = state.pipeline_state
        data = data.replace(qpos=state_ref["qpos"])

        pipeline_state, contact_info = forward_fn(data)
        info.update(contact_info)

        state = State(
            pipeline_state, state.obs, state.reward, state.done, state.metrics, info
        )

        # torso_pos = pipeline_state.qpos[:2]
        # torso_pos_ref = info["state_ref"]["qpos"][:2]
        # print(f"torso_pos: {torso_pos}, torso_pos_ref: {torso_pos_ref}")

        for name in args.rewards:
            reward_fn = getattr(env, f"_reward_{name}", None)
            reward = reward_fn(pipeline_state, info, action)
            reward_dict[name].append(float(reward))

        info["step"] += 1
        info["global_step"] += 1

    # Plot rewards
    # Number of plots and columns
    num_plots = len(args.rewards)
    num_cols = 3
    num_rows = math.ceil(num_plots / num_cols)

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 2 * num_rows))
    axes = axes.flatten()  # flatten 2D array to 1D list

    # Plot each reward
    for i, name in enumerate(args.rewards):
        ax = axes[i]
        ax.plot(reward_dict[name])
        ax.set_title(name)
        ax.set_ylabel("Reward")

    # Remove unused axes
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    # Label bottom row
    for ax in axes[-num_cols:]:
        ax.set_xlabel("Step")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize rewards in MJX environments"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="walk",
        help="Name of the locomotion environment to instantiate",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xc",
        help="Robot description name",
    )
    parser.add_argument(
        "--rewards",
        nargs="+",
        required=True,
        help="List of reward names to plot",
    )
    args = parser.parse_args()

    # Instantiate environment
    cfg, _ = get_env_config(args.env)
    env_cls = get_env_class(args.env)
    robot = Robot(args.robot)
    env = env_cls(args.env, robot, cfg, add_domain_rand=False)

    main(env, args.rewards)
