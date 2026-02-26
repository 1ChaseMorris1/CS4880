# Homework 4 Solutions

This folder contains:
- Baseline MCTS (`mcts.py`) using random rollout simulation.
- LLM-guided MCTS (`evaluators.py`) with cached position evaluation.
- Perfect minimax oracle (`minimax.py`) for optional comparison.
- Experiment runner (`experiment.py`) that generates:
  - `summary/summary.csv`
  - `summary/move_metrics.csv`
  - `summary/summary.json`
  - `summary/run_config.json`
  - `summary/REPORT.md`
  - `summary/SLIDES.md`
  - `summary/figures/*.svg`

Run from `Homework4/`:

```bash
make run
```

For OpenAI-backed evaluation:

```bash
# in Homework4/.env
OPENAI_API_KEY=...
LLM_PROVIDER=openai
OPEN_AI_MODEL=gpt-4.1-mini

make run
```
