from __future__ import annotations

import random
import unittest

from solutions.game import GameState, initial_state
from solutions.minimax import choose_perfect_move, minimax_value


class MinimaxTests(unittest.TestCase):
    def test_initial_state_value_is_draw(self) -> None:
        value = minimax_value(initial_state(), perspective="X")
        self.assertEqual(value, 0.5)

    def test_choose_forced_win(self) -> None:
        state = GameState(
            board=("X", "X", ".", "O", "O", ".", ".", ".", "."),
            player="X",
        )
        move, value = choose_perfect_move(state, random.Random(0))
        self.assertEqual(move, 2)
        self.assertEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
