from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

PLAYER_X = "X"
PLAYER_O = "O"
EMPTY = "."
DRAW = "DRAW"

WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass(frozen=True)
class GameState:
    board: tuple[str, ...]
    player: str

    def __post_init__(self) -> None:
        if len(self.board) != 9:
            raise ValueError("Board must have exactly 9 cells")
        if self.player not in (PLAYER_X, PLAYER_O):
            raise ValueError("Player must be X or O")

    def render(self) -> str:
        rows = []
        for row in range(3):
            start = row * 3
            rows.append(" ".join(self.board[start : start + 3]))
        return "\n".join(rows)


def initial_state() -> GameState:
    return GameState(board=(EMPTY,) * 9, player=PLAYER_X)


def other_player(player: str) -> str:
    return PLAYER_O if player == PLAYER_X else PLAYER_X


def legal_moves(state: GameState) -> list[int]:
    return [idx for idx, cell in enumerate(state.board) if cell == EMPTY]


def apply_move(state: GameState, move: int) -> GameState:
    if move < 0 or move > 8:
        raise ValueError(f"Invalid move index: {move}")
    if state.board[move] != EMPTY:
        raise ValueError(f"Cell {move} is not empty")

    next_board = list(state.board)
    next_board[move] = state.player
    return GameState(board=tuple(next_board), player=other_player(state.player))


def winner(board: Iterable[str]) -> str | None:
    cells = tuple(board)
    for a, b, c in WIN_LINES:
        if cells[a] != EMPTY and cells[a] == cells[b] == cells[c]:
            return cells[a]
    return None


def terminal_outcome(state: GameState) -> str | None:
    who = winner(state.board)
    if who is not None:
        return who
    if EMPTY not in state.board:
        return DRAW
    return None


def terminal_value(state: GameState, perspective: str) -> float | None:
    outcome = terminal_outcome(state)
    if outcome is None:
        return None
    if outcome == DRAW:
        return 0.5
    return 1.0 if outcome == perspective else 0.0


def state_key(state: GameState) -> str:
    return "".join(state.board) + ":" + state.player


def parse_state(board_text: str, player: str) -> GameState:
    cleaned = [c for c in board_text if c in {PLAYER_X, PLAYER_O, EMPTY}]
    if len(cleaned) != 9:
        raise ValueError("board_text must contain exactly 9 symbols from {X, O, .}")
    return GameState(board=tuple(cleaned), player=player)
