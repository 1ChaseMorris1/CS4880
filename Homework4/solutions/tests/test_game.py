from __future__ import annotations

import unittest

from solutions.game import (
    DRAW,
    EMPTY,
    GameState,
    PLAYER_X,
    apply_move,
    initial_state,
    legal_moves,
    terminal_outcome,
    winner,
)


class GameTests(unittest.TestCase):
    def test_initial_state(self) -> None:
        state = initial_state()
        self.assertEqual(state.player, PLAYER_X)
        self.assertEqual(len(legal_moves(state)), 9)

    def test_apply_move_and_turn_switch(self) -> None:
        state = initial_state()
        next_state = apply_move(state, 4)
        self.assertEqual(next_state.board[4], PLAYER_X)
        self.assertNotEqual(next_state.player, state.player)

    def test_winner_and_draw(self) -> None:
        self.assertEqual(winner(("X", "X", "X", ".", ".", ".", ".", ".", ".")), "X")
        draw_state = GameState(
            board=("X", "O", "X", "X", "O", "O", "O", "X", "X"),
            player="X",
        )
        self.assertEqual(terminal_outcome(draw_state), DRAW)

    def test_legal_moves_count(self) -> None:
        state = GameState(
            board=("X", "O", "X", "O", EMPTY, "X", "O", "X", "O"),
            player="X",
        )
        self.assertEqual(legal_moves(state), [4])


if __name__ == "__main__":
    unittest.main()
