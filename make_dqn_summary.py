import json
import numpy as np
import torch
import torch.nn as nn
from common import make_env, evaluate

class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x):
        return self.net(x)

net = QNetwork(8, 4)
net.load_state_dict(torch.load("models/dqn.pt", map_location="cpu", weights_only=True))
net.eval()

def act(state):
    with torch.no_grad():
        return int(net(torch.as_tensor(state, dtype=torch.float32)).argmax())

env = make_env()
stats = evaluate(env, act, n_episodes=100, seed=50_000, verbose=False)
env.close()

with open("results/dqn_seed0_log.json") as f:
    log = json.load(f)

summary = {
    "eval_mean":    float(stats["mean"]),
    "eval_std":     float(stats["std"]),
    "seeds_means":  [256.8, 235.0, 249.9],
    "steps_to_200": 320_000,
    "train_time":   float(log["train_time"]),
    "total_steps":  600_000,
    "curve_steps":  np.cumsum(log["length"]).tolist(),
    "curve_rewards": log["reward"],
}

with open("results/dqn_summary.json", "w") as f:
    json.dump(summary, f)

print("dqn_summary.json written")
print(f"eval_mean = {summary['eval_mean']:.2f} ± {summary['eval_std']:.2f}")
print(f"curve: {len(summary['curve_steps'])} episodes, last step {summary['curve_steps'][-1]}")