"""Shared helpers for the Start Trek Lunar Lander project.

Used by both dqn.ipynb and ppo.ipynb so that the random baseline and the
evaluation protocol (100 episodes, no exploration) are identical between the
two agents. Algorithm-specific code (replay buffer, epsilon, networks, GAE,
rollout storage) lives inside each notebook.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

ENV_ID = "LunarLander-v3"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python, numpy, torch. Call once at the top of each notebook."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def make_env(render_mode=None, seed: int | None = None):
    """Create LunarLander-v3. Pass seed=None during training for variety.

    Note: rendering is VERY slow; use render_mode=None for training/eval,
    render_mode="human" only when watching a saved model.
    """
    env = gym.make(ENV_ID, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


# --------------------------------------------------------------------------- #
# Episode end classification
# --------------------------------------------------------------------------- #
def end_reason(env, terminated: bool, truncated: bool, final_obs) -> str:
    """Map Gymnasium's terminated/truncated to the 4 reasons we log.

    LunarLander-v3 sets terminated=True when (same order as its source code):
      * game_over (the lander BODY touched the ground) -> crash          (-100)
      * abs(state[0]) >= 1.0 (normalized x)             -> out_of_viewport (-100)
      * not lander.awake (Box2D put it to sleep)        -> at_rest         (+100)
    FIX: the first version tested lander.position.x, which is in WORLD units
    (about 0..20, pad at 10), so almost every episode was wrongly labelled
    out_of_viewport. final_obs[0] is the normalized x (pad at 0, edges at +-1).
    """
    if truncated:
        return "truncated"
    if not env.unwrapped.lander.awake:
        return "at_rest"
    if abs(float(final_obs[0])) >= 1.0:
        return "out_of_viewport"
    return "crash"


# --------------------------------------------------------------------------- #
# Episode running
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    reward: float
    steps: int
    end_reason: str


def run_episode(env, act_fn: Callable[[np.ndarray], int],
                seed: int | None = None) -> EpisodeResult:
    """Play one episode with a fixed (non-training) policy.

    act_fn(state) -> int. For evaluation it must NOT explore
    (DQN: argmax Q; PPO: argmax logits, no sampling).
    """
    state, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = act_fn(state)
        state, r, terminated, truncated, _ = env.step(action)
        total += float(r)
        steps += 1
    return EpisodeResult(total, steps, end_reason(env, terminated, truncated, state))


# --------------------------------------------------------------------------- #
# Evaluation (100 episodes, no exploration)
# --------------------------------------------------------------------------- #
def evaluate(env, act_fn, n_episodes: int = 100,
             seed: int = 0, verbose: bool = True) -> dict:
    """Run n_episodes with a deterministic act_fn and return aggregate stats."""
    results = [run_episode(env, act_fn, seed=seed + i) for i in range(n_episodes)]
    rewards = np.array([r.reward for r in results], dtype=float)
    steps = np.array([r.steps for r in results], dtype=int)
    reasons = Counter(r.end_reason for r in results)
    stats = {
        "rewards": rewards,
        "steps": steps,
        "end_reasons": dict(reasons),
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "min": float(rewards.min()),
        "max": float(rewards.max()),
    }
    if verbose:
        print(f"Eval over {n_episodes} episodes (no exploration):")
        print(f"  mean={stats['mean']:.2f}  std={stats['std']:.2f}  "
              f"min={stats['min']:.2f}  max={stats['max']:.2f}")
        print(f"  end reasons: {stats['end_reasons']}")
    return stats


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def moving_average(x, window: int = 100) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return np.array([])
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_training(rewards, lengths, end_reasons=None,
                  losses: dict | None = None, extras: dict | None = None,
                  window: int = 100, title: str = "Training"):
    """3-panel plot: reward + MA, episode length + MA, losses/extras.

    `losses` and `extras` are dicts {label: list-of-values} so DQN can plot
    {MSE}, PPO can plot {policy_loss, value_loss, entropy}, etc.
    """
    rewards = np.asarray(rewards, dtype=float)
    lengths = np.asarray(lengths, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(title)

    # Reward + moving average
    ax = axes[0]
    ax.plot(rewards, alpha=0.3, label="reward")
    ma = moving_average(rewards, window)
    if len(ma):
        ax.plot(np.arange(window - 1, len(rewards)), ma,
                linewidth=2, label=f"MA{window}")
    ax.set_xlabel("episode"); ax.set_ylabel("total reward")
    ax.set_title("Reward per episode"); ax.legend()

    # Episode length
    ax = axes[1]
    ax.plot(lengths, alpha=0.3, label="length")
    ma_l = moving_average(lengths, window)
    if len(ma_l):
        ax.plot(np.arange(window - 1, len(lengths)), ma_l,
                linewidth=2, label=f"MA{window}")
    ax.set_xlabel("episode"); ax.set_ylabel("steps")
    ax.set_title("Episode length"); ax.legend()

    # Losses / extras
    ax = axes[2]
    if losses:
        for name, values in losses.items():
            ax.plot(values, label=name)
        ax.set_title("Losses"); ax.set_xlabel("update")
    elif extras:
        for name, values in extras.items():
            ax.plot(values, label=name)
        ax.set_title("Extras"); ax.set_xlabel("episode")
    else:
        ax.text(0.5, 0.5, "no losses/extras", ha="center", va="center")
        ax.set_axis_off()
    ax.legend()

    plt.tight_layout()
    plt.show()