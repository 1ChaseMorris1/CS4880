from __future__ import annotations

from functools import lru_cache
import random

from .game import GameState, apply_move, legal_moves, terminal_value


@lru_cache(maxsize=None)
def minimax_value(state: GameState, perspective: str) -> float:
    terminal = terminal_value(state, perspective)
    if terminal is not None:
        return terminal

    child_values = [minimax_value(apply_move(state, move), perspective) for move in legal_moves(state)]
    if state.player == perspective:
        return max(child_values)
    return min(child_values)


def best_moves(state: GameState) -> list[tuple[int, float]]:
    perspective = state.player
    scored = []
    for move in legal_moves(state):
        value = minimax_value(apply_move(state, move), perspective)
        scored.append((move, value))
    if not scored:
        return []
    best_value = max(value for _, value in scored)
    return [(move, value) for move, value in scored if value == best_value]


def choose_perfect_move(state: GameState, rng: random.Random | None = None) -> tuple[int, float]:
    choices = best_moves(state)
    if not choices:
        raise ValueError("No legal moves available")
    picker = rng if rng is not None else random
    return picker.choice(choices)
