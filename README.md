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