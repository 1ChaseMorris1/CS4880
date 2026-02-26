# Homework 4 Report: LLM-Guided MCTS for Tic-Tac-Toe

## Experimental Setup
- **Baseline agent**: MCTS with random rollouts in the simulation phase.
- **LLM-guided agent**: MCTS with playout replaced by static position evaluation.
- **LLM provider mode**: `heuristic`
- **Model**: `gpt-4.1-mini`
- **State cache**: enabled (`solutions/results/llm_cache.json`)
- **Game protocol**: for each configuration, agents play both sides (X and O) equally.
- **Iteration budgets tested**: 10, 50, 100, 500.

### LLM Prompt Used
```text
You are given a Tic-Tac-Toe board and the root player perspective. Estimate the probability (0 to 1) that the root player eventually wins assuming reasonably strong play. Return JSON with keys: win_probability (number), reason (short string).
```

## Results (Baseline vs LLM-guided)
| Iterations | Baseline wins | Draws | LLM-guided wins | Baseline win rate | Draw rate | LLM-guided win rate |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 9 | 20 | 51 | 11.2% | 25.0% | 63.7% |
| 50 | 1 | 54 | 25 | 1.2% | 67.5% | 31.2% |
| 100 | 2 | 70 | 8 | 2.5% | 87.5% | 10.0% |
| 500 | 0 | 79 | 1 | 0.0% | 98.8% | 1.2% |

Best baseline result: 10 iterations with 11.2% wins.

Best LLM-guided result: 10 iterations with 63.7% wins.

## Per-Move Timing and Value Estimates
| Iterations | Agent | Moves logged | Avg chosen move value | Avg ms/iteration |
| --- | --- | --- | --- | --- |
| 10 | baseline-mcts-10 | 258 | 0.6905 | 0.015944 |
| 10 | llm-mcts-10 | 273 | 0.7734 | 0.007794 |
| 50 | baseline-mcts-50 | 314 | 0.5802 | 0.014541 |
| 50 | llm-mcts-50 | 317 | 0.6539 | 0.00926 |
| 100 | baseline-mcts-100 | 347 | 0.5542 | 0.01503 |
| 100 | llm-mcts-100 | 347 | 0.6364 | 0.010862 |
| 500 | baseline-mcts-500 | 359 | 0.5442 | 0.014897 |
| 500 | llm-mcts-500 | 359 | 0.553 | 0.011844 |

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
