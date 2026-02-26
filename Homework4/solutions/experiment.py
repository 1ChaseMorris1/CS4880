from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import statistics
import time
from typing import Callable

from .evaluators import LLMPositionEvaluator, RandomRolloutEvaluator
from .generate_figures import generate_all
from .game import (
    PLAYER_O,
    PLAYER_X,
    GameState,
    apply_move,
    initial_state,
    terminal_outcome,
)
from .mcts import MCTSAgent
from .minimax import choose_perfect_move


class MinimaxAgent:
    def __init__(self, seed: int | None = None, name: str = "minimax") -> None:
        self.rng = random.Random(seed)
        self.name = name
        self.last_decision: dict[str, float | int] | None = None

    def choose_move(self, state: GameState) -> int:
        start = time.perf_counter()
        move, value = choose_perfect_move(state, self.rng)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.last_decision = {
            "move": move,
            "estimated_value": value,
            "ms_per_iteration": 0.0,
            "elapsed_ms": elapsed_ms,
        }
        return move


def play_game(agent_x, agent_o):
    state = initial_state()
    move_logs: list[dict[str, float | int | str]] = []

    while True:
        outcome = terminal_outcome(state)
        if outcome is not None:
            return outcome, move_logs

        agent = agent_x if state.player == PLAYER_X else agent_o
        move = agent.choose_move(state)

        decision = getattr(agent, "last_decision", None)
        est = None
        ms_iter = None
        if decision is not None:
            if isinstance(decision, dict):
                est = float(decision.get("estimated_value", 0.5))
                ms_iter = float(decision.get("ms_per_iteration", 0.0))
            else:
                est = float(getattr(decision, "estimated_value", 0.5))
                ms_iter = float(getattr(decision, "ms_per_iteration", 0.0))

        move_logs.append(
            {
                "agent": agent.name,
                "player": state.player,
                "move": int(move),
                "estimated_value": est if est is not None else 0.5,
                "ms_per_iteration": ms_iter if ms_iter is not None else 0.0,
            }
        )
        state = apply_move(state, move)


def run_duel(
    label: str,
    iterations: int,
    games_per_side: int,
    agent_a_factory: Callable[[int], object],
    agent_b_factory: Callable[[int], object],
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rng = random.Random(seed)

    a_wins = 0
    b_wins = 0
    draws = 0

    move_metrics: dict[str, dict[str, list[float]]] = {}

    def ingest_logs(logs):
        for row in logs:
            bucket = move_metrics.setdefault(
                str(row["agent"]),
                {"estimated_values": [], "ms_per_iteration": []},
            )
            bucket["estimated_values"].append(float(row["estimated_value"]))
            bucket["ms_per_iteration"].append(float(row["ms_per_iteration"]))

    # Game 1 orientation: A as X, B as O
    # Game 2 orientation: B as X, A as O
    for _ in range(games_per_side):
        a_as_x = agent_a_factory(rng.randrange(1, 1_000_000_000))
        b_as_o = agent_b_factory(rng.randrange(1, 1_000_000_000))
        outcome, logs = play_game(a_as_x, b_as_o)
        ingest_logs(logs)
        if outcome == PLAYER_X:
            a_wins += 1
        elif outcome == PLAYER_O:
            b_wins += 1
        else:
            draws += 1

        b_as_x = agent_b_factory(rng.randrange(1, 1_000_000_000))
        a_as_o = agent_a_factory(rng.randrange(1, 1_000_000_000))
        outcome, logs = play_game(b_as_x, a_as_o)
        ingest_logs(logs)
        if outcome == PLAYER_X:
            b_wins += 1
        elif outcome == PLAYER_O:
            a_wins += 1
        else:
            draws += 1

    games = games_per_side * 2
    row = {
        "matchup": label,
        "iterations": iterations,
        "games": games,
        "agent_a": agent_a_factory(0).name,
        "agent_b": agent_b_factory(0).name,
        "a_wins": a_wins,
        "draws": draws,
        "b_wins": b_wins,
        "a_win_rate": round(a_wins / games, 4),
        "draw_rate": round(draws / games, 4),
        "b_win_rate": round(b_wins / games, 4),
    }

    move_rows: list[dict[str, object]] = []
    for agent_name, bucket in move_metrics.items():
        move_rows.append(
            {
                "matchup": label,
                "iterations": iterations,
                "agent": agent_name,
                "moves": len(bucket["estimated_values"]),
                "avg_estimated_value": round(statistics.mean(bucket["estimated_values"]), 4),
                "avg_ms_per_iteration": round(statistics.mean(bucket["ms_per_iteration"]), 6),
            }
        )

    return row, move_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _table(headers: list[str], records: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in records:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report_and_slides(
    summary_rows: list[dict[str, object]],
    move_rows: list[dict[str, object]],
    output_dir: Path,
    llm_provider: str,
    llm_model: str,
    prompt_text: str,
    llm_cache_path: str,
    figure_paths: list[Path],
) -> None:
    baseline_vs_llm = [r for r in summary_rows if str(r["matchup"]) == "baseline_vs_llm"]
    baseline_vs_llm.sort(key=lambda r: int(r["iterations"]))

    if baseline_vs_llm:
        records = [
            [
                str(r["iterations"]),
                str(r["a_wins"]),
                str(r["draws"]),
                str(r["b_wins"]),
                f"{100 * float(r['a_win_rate']):.1f}%",
                f"{100 * float(r['draw_rate']):.1f}%",
                f"{100 * float(r['b_win_rate']):.1f}%",
            ]
            for r in baseline_vs_llm
        ]
        results_table = _table(
            [
                "Iterations",
                "Baseline wins",
                "Draws",
                "LLM-guided wins",
                "Baseline win rate",
                "Draw rate",
                "LLM-guided win rate",
            ],
            records,
        )

        best_llm_row = max(baseline_vs_llm, key=lambda r: float(r["b_win_rate"]))
        best_base_row = max(baseline_vs_llm, key=lambda r: float(r["a_win_rate"]))
        best_llm_line = (
            f"Best LLM-guided result: {best_llm_row['iterations']} iterations "
            f"with {100 * float(best_llm_row['b_win_rate']):.1f}% wins."
        )
        best_base_line = (
            f"Best baseline result: {best_base_row['iterations']} iterations "
            f"with {100 * float(best_base_row['a_win_rate']):.1f}% wins."
        )
    else:
        results_table = "No results generated."
        best_llm_line = "No baseline-vs-LLM rows available."
        best_base_line = "No baseline-vs-LLM rows available."

    move_lookup = {}
    for row in move_rows:
        key = (str(row["matchup"]), int(row["iterations"]), str(row["agent"]))
        move_lookup[key] = row

    move_records: list[list[str]] = []
    for duel in baseline_vs_llm:
        it = int(duel["iterations"])
        for agent in (str(duel["agent_a"]), str(duel["agent_b"])):
            key = ("baseline_vs_llm", it, agent)
            if key in move_lookup:
                rec = move_lookup[key]
                move_records.append(
                    [
                        str(it),
                        agent,
                        str(rec["moves"]),
                        str(rec["avg_estimated_value"]),
                        str(rec["avg_ms_per_iteration"]),
                    ]
                )

    move_table = _table(
        [
            "Iterations",
            "Agent",
            "Moves logged",
            "Avg chosen move value",
            "Avg ms/iteration",
        ],
        move_records,
    )

    figure_markdown = "\n".join([f"![{p.stem}]({p.as_posix()})" for p in figure_paths]) if figure_paths else "No figures generated."

    report_path = output_dir / "REPORT.md"
    report_text = f"""# Homework 4 Report: LLM-Guided MCTS for Tic-Tac-Toe

## Experimental Setup
- **Baseline agent**: MCTS with random rollouts in the simulation phase.
- **LLM-guided agent**: MCTS with playout replaced by static position evaluation.
- **LLM provider mode**: `{llm_provider}`
- **Model**: `{llm_model}`
- **State cache**: enabled (`{llm_cache_path}`)
- **Game protocol**: for each configuration, agents play both sides (X and O) equally.
- **Iteration budgets tested**: 10, 50, 100, 500.

### LLM Prompt Used
```text
{prompt_text}
```

## Results (Baseline vs LLM-guided)
{results_table}

{best_base_line}

{best_llm_line}

## Per-Move Timing and Value Estimates
{move_table}

## Visual Results
{figure_markdown}

## Discussion
### 1. Did LLM-guided evaluation improve play quality?
The comparison above shows the effect of replacing random rollouts with an informed value estimate. At lower iteration counts, informed value estimates are expected to help because random rollouts are noisy and the tree has little time to average that noise out. At higher iteration counts, both approaches often trend toward strong play in Tic-Tac-Toe due to the small state space.

### 2. Why Tic-Tac-Toe changes the conclusion
Tic-Tac-Toe is solved and tiny. The state space is small enough that ordinary MCTS converges quickly even with random rollouts. This caps the upside of LLM guidance: once baseline MCTS has enough iterations, both methods often produce near-optimal choices and many draws.

### 3. Computational trade-offs
Random rollout evaluation is extremely cheap and can run many simulations per second. API-backed LLM evaluation introduces request latency and token cost, even with caching. In this project, caching board states is essential because MCTS revisits the same states repeatedly.

### 4. Where LLM-guidance should matter more
The advantage should grow with:
- larger branching factors,
- deeper long-horizon tasks,
- environments where random rollouts poorly approximate expert behavior,
- expensive or hard-to-code simulators.

### 5. Connection to RAP
RAP shows stronger gains in complex reasoning and planning tasks where random simulation is weak and learned world knowledge helps prune search. Tic-Tac-Toe is the opposite regime: exact dynamics are trivial and random rollout already reaches useful estimates quickly, so the gap between methods is naturally smaller.
"""
    report_path.write_text(report_text, encoding="utf-8")

    slides_path = output_dir / "SLIDES.md"
    slides_text = f"""# Slide 1 - Title
LLM-Guided Monte Carlo Tree Search for Tic-Tac-Toe

## Slide 2 - Motivation
- Baseline MCTS uses random rollouts for simulation.
- Random playouts are noisy at low search budgets.
- RAP suggests replacing/supplementing rollouts with LLM-based value signals.

## Slide 3 - Methods
- Baseline: UCB1 + random rollout simulation.
- LLM-guided: UCB1 + LLM position evaluator (with cache).
- Provider mode: `{llm_provider}`
- Model: `{llm_model}`

## Slide 4 - Prompt
```text
{prompt_text}
```

## Slide 5 - Experimental Design
- Iterations per move: 10, 50, 100, 500.
- Symmetric matchups: each agent plays both X and O.
- Metrics: win/draw/loss, chosen-move value estimate, time per iteration.

## Slide 6 - Results
{results_table}

## Slide 7 - Graphs
{figure_markdown}

## Slide 8 - Trade-offs
- Random rollouts: very fast, no token cost.
- LLM eval: potentially better priors at low iterations, but higher latency/cost.
- Caching is mandatory for practical MCTS+LLM performance.

## Slide 9 - Takeaways
- LLM guidance can help when search budget is low or domain complexity is high.
- Tic-Tac-Toe's small solved state space limits gains.
- RAP-style methods should show larger benefits in larger planning/reasoning tasks.
"""
    slides_path.write_text(slides_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline vs LLM-guided MCTS experiments")
    parser.add_argument("--iterations", default="10,50,100,500", help="Comma-separated list, e.g. 10,50,100,500")
    parser.add_argument("--games-per-side", type=int, default=40, help="Games per side (total games = 2x)")
    parser.add_argument("--vs-minimax-games-per-side", type=int, default=15)
    parser.add_argument("--exploration", type=float, default=1.41)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="solutions/summary")
    parser.add_argument("--llm-provider", default="heuristic", choices=["heuristic", "openai"])
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    parser.add_argument("--llm-cache", default="solutions/summary/llm_cache.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iterations_list = [int(x.strip()) for x in args.iterations.split(",") if x.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_evaluator = LLMPositionEvaluator(
        provider=args.llm_provider,
        model=args.llm_model,
        cache_path=args.llm_cache,
    )

    summary_rows: list[dict[str, object]] = []
    move_rows: list[dict[str, object]] = []

    for iters in iterations_list:
        def make_baseline(seed: int) -> MCTSAgent:
            return MCTSAgent(
                evaluator=RandomRolloutEvaluator(seed=seed),
                iterations=iters,
                exploration=args.exploration,
                seed=seed,
                name=f"baseline-mcts-{iters}",
            )

        def make_llm(seed: int) -> MCTSAgent:
            return MCTSAgent(
                evaluator=llm_evaluator,
                iterations=iters,
                exploration=args.exploration,
                seed=seed,
                name=f"llm-mcts-{iters}",
            )

        def make_minimax(seed: int) -> MinimaxAgent:
            return MinimaxAgent(seed=seed, name="minimax")

        row, rows = run_duel(
            label="baseline_vs_llm",
            iterations=iters,
            games_per_side=args.games_per_side,
            agent_a_factory=make_baseline,
            agent_b_factory=make_llm,
            seed=args.seed + (iters * 17),
        )
        summary_rows.append(row)
        move_rows.extend(rows)

        if args.vs_minimax_games_per_side > 0:
            row, rows = run_duel(
                label="baseline_vs_minimax",
                iterations=iters,
                games_per_side=args.vs_minimax_games_per_side,
                agent_a_factory=make_baseline,
                agent_b_factory=make_minimax,
                seed=args.seed + (iters * 31),
            )
            summary_rows.append(row)
            move_rows.extend(rows)

            row, rows = run_duel(
                label="llm_vs_minimax",
                iterations=iters,
                games_per_side=args.vs_minimax_games_per_side,
                agent_a_factory=make_llm,
                agent_b_factory=make_minimax,
                seed=args.seed + (iters * 43),
            )
            summary_rows.append(row)
            move_rows.extend(rows)

    llm_evaluator.save_cache()

    summary_path = output_dir / "summary.csv"
    move_path = output_dir / "move_metrics.csv"
    json_path = output_dir / "summary.json"
    config_path = output_dir / "run_config.json"

    write_csv(summary_path, summary_rows)
    write_csv(move_path, move_rows)
    json_path.write_text(json.dumps({"summary": summary_rows, "move_metrics": move_rows}, indent=2), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "iterations": iterations_list,
                "games_per_side": args.games_per_side,
                "vs_minimax_games_per_side": args.vs_minimax_games_per_side,
                "exploration": args.exploration,
                "seed": args.seed,
                "llm_provider": args.llm_provider,
                "llm_model": args.llm_model,
                "llm_cache": args.llm_cache,
                "output_dir": args.output_dir,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    figure_paths = generate_all(output_dir)
    relative_figure_paths = [path.relative_to(output_dir) for path in figure_paths]

    write_report_and_slides(
        summary_rows=summary_rows,
        move_rows=move_rows,
        output_dir=output_dir,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        prompt_text=LLMPositionEvaluator.prompt_template(),
        llm_cache_path=args.llm_cache,
        figure_paths=relative_figure_paths,
    )

    print("Wrote:")
    print(f"  - {summary_path}")
    print(f"  - {move_path}")
    print(f"  - {json_path}")
    print(f"  - {config_path}")
    for figure_path in figure_paths:
        print(f"  - {figure_path}")
    print(f"  - {output_dir / 'REPORT.md'}")
    print(f"  - {output_dir / 'SLIDES.md'}")


if __name__ == "__main__":
    main()
