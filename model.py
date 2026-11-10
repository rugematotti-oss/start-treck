import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

USAGE = """USAGE
    python model.py ALGO MODEL [EPISODES]

DESCRIPTION
    ALGO        algorithm of the saved model: dqn or ppo
    MODEL       path to the saved model file
    EPISODES    number of episodes to run (default: 1)"""


def fail(msg):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(84)


class QNetwork(nn.Module):
    """DQN: state -> one Q-value per action (8 -> 128 -> 128 -> 4, ReLU).

    Same structure and attribute name (self.net) as dqn.ipynb, so the saved
    weights load without renaming.
    """
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    """PPO: separate actor and critic, 8 -> 64 -> 64, tanh (same structure as ppo.ipynb)."""
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        # ACTOR: state -> one logit per action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        # CRITIC: state -> expected score V(s)
        # (not needed to play, but present in the saved file)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x).squeeze(-1)


def greedy_action(algo, net, state):
    """No exploration: always take the action judged best."""
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        if algo == "ppo":
            logits, _ = net(x)            # most probable action, no sampling
            return int(logits.argmax())
        return int(net(x).argmax())       # DQN: highest Q-value


def main(argv):
    if len(argv) == 1 and argv[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)
    if len(argv) not in (2, 3):
        fail("wrong number of arguments (see -h)")
    algo, path = argv[0], argv[1]
    if algo not in ("dqn", "ppo"):
        fail("ALGO must be dqn or ppo")
    try:
        episodes = int(argv[2]) if len(argv) == 3 else 1
        if episodes < 1:
            raise ValueError
    except ValueError:
        fail("EPISODES must be a positive integer")
    # Missing file: clean error (code 84) before the window opens
    if not os.path.isfile(path):
        fail(f"model file not found: {path}")

    env = gym.make("LunarLander-v3", render_mode="human")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    try:
        net = (QNetwork if algo == "dqn" else ActorCritic)(state_dim, n_actions)
        # weights_only=True: load only the weights (safer)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net.eval()
    except Exception as e:
        env.close()
        fail(f"cannot load model: {e}")

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action = greedy_action(algo, net, state)
            state, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        print(f"Episode {ep}: score {total:.1f}")
    env.close()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except SystemExit:
        raise
    except Exception as e:
        fail(str(e))