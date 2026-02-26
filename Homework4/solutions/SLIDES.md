# Slide 1 - Title
LLM-Guided Monte Carlo Tree Search for Tic-Tac-Toe

## Slide 2 - Motivation
- Baseline MCTS uses random rollouts for simulation.
- Random playouts are noisy at low search budgets.
- RAP suggests replacing/supplementing rollouts with LLM-based value signals.

## Slide 3 - Methods
- Baseline: UCB1 + random rollout simulation.
- LLM-guided: UCB1 + LLM position evaluator (with cache).
- Provider mode: `heuristic`
- Model: `gpt-4.1-mini`

## Slide 4 - Prompt
```text
You are given a Tic-Tac-Toe board and the root player perspective. Estimate the probability (0 to 1) that the root player eventually wins assuming reasonably strong play. Return JSON with keys: win_probability (number), reason (short string).
```

## Slide 5 - Experimental Design
- Iterations per move: 10, 50, 100, 500.
- Symmetric matchups: each agent plays both X and O.
- Metrics: win/draw/loss, chosen-move value estimate, time per iteration.

## Slide 6 - Results
| Iterations | Baseline wins | Draws | LLM-guided wins | Baseline win rate | Draw rate | LLM-guided win rate |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 9 | 20 | 51 | 11.2% | 25.0% | 63.7% |
| 50 | 1 | 54 | 25 | 1.2% | 67.5% | 31.2% |
| 100 | 2 | 70 | 8 | 2.5% | 87.5% | 10.0% |
| 500 | 0 | 79 | 1 | 0.0% | 98.8% | 1.2% |

## Slide 7 - Trade-offs
- Random rollouts: very fast, no token cost.
- LLM eval: potentially better priors at low iterations, but higher latency/cost.
- Caching is mandatory for practical MCTS+LLM performance.

## Slide 8 - Takeaways
- LLM guidance can help when search budget is low or domain complexity is high.
- Tic-Tac-Toe's small solved state space limits gains.
- RAP-style methods should show larger benefits in larger planning/reasoning tasks.
