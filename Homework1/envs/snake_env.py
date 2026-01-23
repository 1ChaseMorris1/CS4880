from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


Position = Tuple[int, int]


@dataclass
class SnakeConfig:
    grid_size: int = 10
    reward_type: str = "dense"


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(
        self,
        grid_size: int = 10,
        reward_type: str = "dense",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = SnakeConfig(grid_size=grid_size, reward_type=reward_type)
        if self.config.reward_type not in {"dense", "sparse"}:
            raise ValueError("reward_type must be 'dense' or 'sparse'")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.config.grid_size, self.config.grid_size),
            dtype=np.float32,
        )

        self.np_random = np.random.default_rng()
        self.snake: List[Position] = []
        self.food: Position | None = None

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        center = self.config.grid_size // 2
        self.snake = [(center, center)]
        self._spawn_food()
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action: int):
        direction = self._action_to_delta(action)
        head_x, head_y = self.snake[0]
        new_head = (head_x + direction[0], head_y + direction[1])

        hit_wall = not (0 <= new_head[0] < self.config.grid_size) or not (
            0 <= new_head[1] < self.config.grid_size
        )
        if hit_wall:
            return self._get_observation(), -1.0, True, False, {}

        ate_food = new_head == self.food
        body = self.snake if ate_food else self.snake[:-1]
        if new_head in body:
            return self._get_observation(), -1.0, True, False, {}

        self.snake.insert(0, new_head)
        if ate_food:
            self._spawn_food()
            reward = 1.0
        else:
            self.snake.pop()
            reward = -0.01 if self.config.reward_type == "dense" else 0.0

        return self._get_observation(), reward, False, False, {}

    def _spawn_food(self) -> None:
        empty_cells = [
            (x, y)
            for x in range(self.config.grid_size)
            for y in range(self.config.grid_size)
            if (x, y) not in self.snake
        ]
        if not empty_cells:
            self.food = None
            return
        self.food = empty_cells[self.np_random.integers(len(empty_cells))]

    def _get_observation(self) -> np.ndarray:
        grid = np.zeros(
            (self.config.grid_size, self.config.grid_size), dtype=np.float32
        )
        for segment in self.snake:
            grid[segment] = 0.5
        if self.food is not None:
            grid[self.food] = 1.0
        return grid

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        cell = 32
        height = self.config.grid_size * cell
        width = self.config.grid_size * cell
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255

        for x, y in self.snake:
            frame[
                x * cell : (x + 1) * cell, y * cell : (y + 1) * cell
            ] = np.array([50, 180, 50], dtype=np.uint8)
        if self.food is not None:
            fx, fy = self.food
            frame[
                fx * cell : (fx + 1) * cell, fy * cell : (fy + 1) * cell
            ] = np.array([220, 50, 50], dtype=np.uint8)
        return frame

    @staticmethod
    def _action_to_delta(action: int) -> Position:
        if action == 0:
            return (-1, 0)
        if action == 1:
            return (0, 1)
        if action == 2:
            return (1, 0)
        if action == 3:
            return (0, -1)
        raise ValueError("Invalid action")
