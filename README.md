# Project Assignment 2: The Gambler's Dilemma (K-Armed Bandits)

This folder contains a full implementation of a **10-armed bandit** experiment using an **epsilon-greedy** agent.

## LLM Usage
- LLM used: **OpenAI GPT-5 (Codex)**

## What is implemented
- 10-armed stationary bandit environment.
- Reward model per arm: `R ~ Normal(mu_arm, 1)`.
- True arm means sampled once per run: `mu_arm ~ Normal(0, 1)`.
- Epsilon-greedy action selection.
- Two action-value update modes:
  - Constant step-size: `alpha = 0.1`
  - Sample-average: `alpha = 1 / n`
- Required hyperparameter sweep:
  - `epsilon`: `0.0`, `0.01`, `0.1`
  - `Q1`: `0`, `5`
  - `alpha mode`: `constant`, `sample_average`

## Metrics logged
At each step, the experiment records:
- `average_reward`
- `percent_optimal_action`
- `average_cumulative_regret`

## How to run
1. Install dependencies:
   ```bash
   pip install -r Homework2/requirements.txt
   ```
2. Run all experiments (offline WandB mode by default):
   ```bash
   make -C Homework2 run
   ```

Or run directly:
```bash
python3 Homework2/bandit_experiment.py --steps 2000 --runs 200 --wandb-mode offline
```

To log to WandB cloud:
```bash
wandb login
make -C Homework2 run-online
```

## Outputs
- Per-configuration CSV files in `Homework2/results/`
- Aggregate run summary in `Homework2/results/summary.json`

There are 12 CSV files total (3 epsilon values x 2 initializations x 2 alpha modes).
