#!/usr/bin/env python3
"""Run epsilon-greedy experiments on a 10-armed bandit and log metrics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    wandb = None


@dataclass(frozen=True)
class ExperimentConfig:
    epsilon: float
    initial_q: float
    alpha_mode: str  # "constant" or "sample_average"
    alpha_constant: float = 0.1
    k_arms: int = 10
    steps: int = 2000
    runs: int = 200

    def run_name(self) -> str:
        eps_str = str(self.epsilon).replace(".", "p")
        q_str = str(self.initial_q).replace(".", "p")
        return f"eps{eps_str}_q{q_str}_{self.alpha_mode}"


class KArmedBandit:
    def __init__(self, k_arms: int, rng: np.random.Generator) -> None:
        self.k_arms = k_arms
        self.rng = rng
        self.true_means = rng.normal(loc=0.0, scale=1.0, size=k_arms)
        self.optimal_action = int(np.argmax(self.true_means))
        self.optimal_mean = float(self.true_means[self.optimal_action])

    def pull(self, action: int) -> float:
        return float(self.rng.normal(loc=self.true_means[action], scale=1.0))


class EpsilonGreedyAgent:
    def __init__(
        self,
        k_arms: int,
        epsilon: float,
        initial_q: float,
        alpha_mode: str,
        alpha_constant: float,
        rng: np.random.Generator,
    ) -> None:
        self.k_arms = k_arms
        self.epsilon = epsilon
        self.alpha_mode = alpha_mode
        self.alpha_constant = alpha_constant
        self.rng = rng

        self.q_estimates = np.full(k_arms, fill_value=initial_q, dtype=np.float64)
        self.action_counts = np.zeros(k_arms, dtype=np.int64)

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.k_arms))

        max_q = np.max(self.q_estimates)
        greedy_actions = np.flatnonzero(self.q_estimates == max_q)
        return int(self.rng.choice(greedy_actions))

    def update(self, action: int, reward: float) -> None:
        self.action_counts[action] += 1

        if self.alpha_mode == "sample_average":
            alpha = 1.0 / self.action_counts[action]
        elif self.alpha_mode == "constant":
            alpha = self.alpha_constant
        else:
            raise ValueError(f"Unsupported alpha_mode: {self.alpha_mode}")

        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])


def run_single(config: ExperimentConfig, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 1)

    bandit = KArmedBandit(k_arms=config.k_arms, rng=rng_env)
    agent = EpsilonGreedyAgent(
        k_arms=config.k_arms,
        epsilon=config.epsilon,
        initial_q=config.initial_q,
        alpha_mode=config.alpha_mode,
        alpha_constant=config.alpha_constant,
        rng=rng_agent,
    )

    rewards = np.zeros(config.steps, dtype=np.float64)
    optimal_taken = np.zeros(config.steps, dtype=np.float64)
    regrets = np.zeros(config.steps, dtype=np.float64)

    for t in range(config.steps):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)

        rewards[t] = reward
        optimal_taken[t] = float(action == bandit.optimal_action)
        regrets[t] = bandit.optimal_mean - reward

    cumulative_regret = np.cumsum(regrets)
    return rewards, optimal_taken, cumulative_regret


def aggregate_runs(config: ExperimentConfig, seed: int) -> dict[str, np.ndarray]:
    rewards_all = np.zeros((config.runs, config.steps), dtype=np.float64)
    optimal_all = np.zeros((config.runs, config.steps), dtype=np.float64)
    cumulative_regret_all = np.zeros((config.runs, config.steps), dtype=np.float64)

    for run_idx in range(config.runs):
        run_seed = seed + (run_idx * 13)
        rewards, optimal_taken, cumulative_regret = run_single(config=config, seed=run_seed)
        rewards_all[run_idx] = rewards
        optimal_all[run_idx] = optimal_taken
        cumulative_regret_all[run_idx] = cumulative_regret

    return {
        "average_reward": rewards_all.mean(axis=0),
        "percent_optimal_action": optimal_all.mean(axis=0) * 100.0,
        "average_cumulative_regret": cumulative_regret_all.mean(axis=0),
    }


def write_csv(path: Path, metrics: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "average_reward", "percent_optimal_action", "average_cumulative_regret"])
        for idx in range(len(metrics["average_reward"])):
            writer.writerow(
                [
                    idx + 1,
                    float(metrics["average_reward"][idx]),
                    float(metrics["percent_optimal_action"][idx]),
                    float(metrics["average_cumulative_regret"][idx]),
                ]
            )


def maybe_init_wandb(project: str, mode: str, config: ExperimentConfig):
    if wandb is None:
        return None

    return wandb.init(
        project=project,
        name=config.run_name(),
        config=asdict(config),
        mode=mode,
        reinit=True,
    )


def maybe_log_wandb(metrics: dict[str, np.ndarray]) -> None:
    if wandb is None:
        return

    steps = len(metrics["average_reward"])
    for idx in range(steps):
        wandb.log(
            {
                "average_reward": float(metrics["average_reward"][idx]),
                "percent_optimal_action": float(metrics["percent_optimal_action"][idx]),
                "average_cumulative_regret": float(metrics["average_cumulative_regret"][idx]),
            },
            step=idx + 1,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-armed bandit experiment runner")
    parser.add_argument("--steps", type=int, default=2000, help="Number of time steps per simulation run")
    parser.add_argument("--runs", type=int, default=200, help="Number of independent runs per configuration")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--project", type=str, default="cs4880-hw2-kbandit", help="WandB project name")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="offline",
        choices=["online", "offline", "disabled"],
        help="WandB logging mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Homework2/results"),
        help="Directory for CSV and summary outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configs = [
        ExperimentConfig(epsilon=eps, initial_q=q_init, alpha_mode=alpha_mode, steps=args.steps, runs=args.runs)
        for eps, q_init, alpha_mode in itertools.product(
            [0.0, 0.01, 0.1],
            [0.0, 5.0],
            ["constant", "sample_average"],
        )
    ]

    all_summary: list[dict[str, Any]] = []
    if wandb is None:
        print("wandb is not installed; metrics will still be written to CSV files.")

    for idx, config in enumerate(configs):
        run = maybe_init_wandb(project=args.project, mode=args.wandb_mode, config=config)
        metrics = aggregate_runs(config=config, seed=args.seed + idx * 997)

        csv_path = args.output_dir / f"{config.run_name()}.csv"
        write_csv(csv_path, metrics)

        final_row = {
            "config": asdict(config),
            "csv": str(csv_path),
            "final_average_reward": float(metrics["average_reward"][-1]),
            "final_percent_optimal_action": float(metrics["percent_optimal_action"][-1]),
            "final_average_cumulative_regret": float(metrics["average_cumulative_regret"][-1]),
        }
        all_summary.append(final_row)

        maybe_log_wandb(metrics)
        if run is not None:
            wandb.summary.update(final_row)
            run.finish()

        print(
            f"Completed {config.run_name()}: "
            f"reward={final_row['final_average_reward']:.4f}, "
            f"optimal%={final_row['final_percent_optimal_action']:.2f}, "
            f"regret={final_row['final_average_cumulative_regret']:.2f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
