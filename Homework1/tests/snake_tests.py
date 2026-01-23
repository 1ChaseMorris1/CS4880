import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.snake_env import SnakeEnv


def test_reset_returns_obs_and_info():
    env = SnakeEnv(grid_size=7, reward_type="dense")
    obs, info = env.reset(seed=123)
    assert isinstance(info, dict)
    assert obs.shape == (7, 7)
    assert obs.dtype == np.float32
    assert np.all(np.isin(obs, [0.0, 0.5, 1.0]))


def test_step_signature_and_types():
    env = SnakeEnv(grid_size=5, reward_type="dense")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (5, 5)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_food_not_on_snake_after_reset():
    env = SnakeEnv(grid_size=6, reward_type="dense")
    env.reset(seed=1)
    assert env.food not in env.snake


def test_food_spawn_near_full_grid():
    env = SnakeEnv(grid_size=3, reward_type="dense")
    env.snake = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    env._spawn_food()
    assert env.food == (2, 2)


def test_food_spawn_full_grid_sets_none():
    env = SnakeEnv(grid_size=2, reward_type="dense")
    env.snake = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env._spawn_food()
    assert env.food is None


def test_wall_collision_terminates():
    env = SnakeEnv(grid_size=4, reward_type="dense")
    env.reset(seed=0)
    env.snake = [(0, 0)]
    obs, reward, terminated, truncated, _ = env.step(0)
    assert terminated is True
    assert truncated is False
    assert reward == -1.0


def test_self_collision_terminates():
    env = SnakeEnv(grid_size=4, reward_type="dense")
    env.reset(seed=0)
    env.snake = [(1, 1), (1, 2), (1, 3), (1, 2)]
    obs, reward, terminated, truncated, _ = env.step(1)
    assert terminated is True
    assert truncated is False
    assert reward == -1.0


def test_food_eaten_reward_and_growth():
    env = SnakeEnv(grid_size=5, reward_type="dense")
    env.reset(seed=0)
    env.snake = [(2, 2)]
    env.food = (2, 3)
    _, reward, terminated, _, _ = env.step(1)
    assert terminated is False
    assert reward == 1.0
    assert len(env.snake) == 2


def test_dense_reward_step_penalty():
    env = SnakeEnv(grid_size=5, reward_type="dense")
    env.reset(seed=0)
    env.snake = [(2, 2)]
    env.food = (4, 4)
    _, reward, terminated, _, _ = env.step(1)
    assert terminated is False
    assert reward == -0.01


def test_sparse_reward_no_penalty():
    env = SnakeEnv(grid_size=5, reward_type="sparse")
    env.reset(seed=0)
    env.snake = [(2, 2)]
    env.food = (4, 4)
    _, reward, terminated, _, _ = env.step(1)
    assert terminated is False
    assert reward == 0.0


def test_seed_determinism():
    env1 = SnakeEnv(grid_size=6, reward_type="dense")
    env2 = SnakeEnv(grid_size=6, reward_type="dense")
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    assert np.array_equal(obs1, obs2)
    assert env1.food == env2.food


def test_action_mapping():
    env = SnakeEnv(grid_size=5, reward_type="dense")
    env.reset(seed=0)
    env.snake = [(2, 2)]
    env.food = (4, 4)
    env.step(0)
    assert env.snake[0] == (1, 2)


def test_env_checker():
    env = SnakeEnv(grid_size=5, reward_type="dense")
    check_env(env, skip_render_check=True)


def test_sb3_dummy_vec_env_smoke():
    def _make():
        return SnakeEnv(grid_size=5, reward_type="dense")

    env = DummyVecEnv([_make])
    obs = env.reset()
    assert obs.shape[-1] == 5
    action = [env.action_space.sample()]
    obs, reward, terminated, info = env.step(action)
    assert obs is not None
