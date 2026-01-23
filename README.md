# Homework 1 Snake RL (Gymnasium + SB3 + WandB)

Minimal Snake environment compatible with Gymnasium and Stable-Baselines3, plus
a PPO training script with Weights & Biases logging.

## Project layout

```
snake_rl/
  envs/
    snake_env.py
  train.py
  requirements.txt
  README.md
```

## Environment design

- Grid-based observation `(grid_size, grid_size)` with float32 values:
  - `0.0` empty
  - `0.5` snake body
  - `1.0` food
- Actions: `0=up`, `1=right`, `2=down`, `3=left`
- Rewards:
  - Food eaten: `+1`
  - Collision: `-1` and episode terminates
  - Step penalty (dense): `-0.01`, sparse: `0`

## Training

Default algorithm: PPO with `MlpPolicy`. Training uses WandB for metrics.

Run: `python train.py`

Makefile shortcuts: `make install`, `make wandb-login`, `make train`

Optional environment variables:

- `GRID_SIZE` (default `10`)
- `REWARD_TYPE` (`dense` or `sparse`, default `dense`)
- `TOTAL_TIMESTEPS` (default `200000`)
- `ALGORITHMS` (comma-separated, e.g. `ppo,a2c,dqn`)
- `RECORD_VIDEO` (`1` to log videos)
- `VIDEO_DIR` (default `videos`)
- `LOG_VIDEO_EVERY_STEPS` (default `50000`, ~4 videos per 200k steps)
- `VIDEO_SECONDS` (default `60`)
- `VIDEO_LENGTH` (default `0`, computed from `VIDEO_SECONDS` and env FPS)

Outputs:

- WandB run with episode reward and length over timesteps
- PPO loss curves from the default SB3 logs
- Saved model at `snake_<algo>.zip`
- Algorithm comparison in WandB when using `ALGORITHMS`
- Recorded videos synced to WandB

WandB report:

```
https://wandb.ai/cmorris8273-ohio-university/snake-rl/reports/Snake-Game-Benchmark--VmlldzoxNTcyODA0Mg?accessToken=9jlhdn7abc9qfk5qkkb7twtvu90ae35n1d7i4z3t4z5sszycksuntyfelwhwf7x0
```

Included videos (by algorithm):

<table>
  <tr>
    <th>PPO</th>
    <th>A2C</th>
    <th>DQN</th>
  </tr>
  <tr>
    <td><img src="Homework1/video/ppo.gif" alt="PPO video"></td>
    <td><img src="Homework1/video/a2c.gif" alt="A2C video"></td>
    <td><img src="Homework1/video/dqn.gif" alt="DQN video"></td>
  </tr>
</table>

## Tests

Run the test suite with pytest: `pytest -q`

Coverage includes Gymnasium API compliance, reward logic, collisions, food spawning,
determinism, and an SB3 `DummyVecEnv` smoke test.

## Experiments to try

- Dense vs. sparse rewards
- Scaling grid size for difficulty and generalization

## GenAI usage note

Code scaffold generated using OpenAI Codex.
Code created with the assistance of gpt-5.2-codex.
