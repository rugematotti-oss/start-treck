# Start Trek - Lunar Lander (DQN + PPO)

This project implements Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents to solve the `LunarLander-v3` environment from Gymnasium. 

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt


## Layout

bootstrap.ipynb : Taxi Q-learning warm-up (tabular Q-learning)

dqn.ipynb : DQN training, curves, evaluation, target-network experiment, 3 seeds

ppo.ipynb : PPO training, curves, evaluation, clip-eps experiment, 3 seeds, DQN comparison

model.py : Load a saved model and watch it land

make_dqn_summary.py : Generates the DQN metrics for the final comparison

models/ : Saved weights (dqn.pt, ppo.pt)

results/ : Plots and metrics exported from notebooks

## Run

To watch a trained agent land:

bash
python model.py dqn models/dqn.pt 3
python model.py ppo models/ppo.pt 3


##  Summary

Both algorithms successfully solve the environment (mean score ≥ 200 over 100 episodes).

DQN: 256.8 ± 25.7 (solved in ~320k steps)

PPO: 266.2 ± 21.2 (solved in ~426k steps)

Comparison: PPO achieves a higher final score, 100% clean landings, and is faster in wall-clock time (due to parallel environments). DQN is more sample-efficient, reaching the 200-threshold in fewer environment steps. Full details and plots are in ppo.ipynb Section 12.