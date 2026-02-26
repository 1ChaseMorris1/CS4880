from __future__ import annotations

import unittest

from solutions.evaluators import LLMPositionEvaluator, RandomRolloutEvaluator
from solutions.game import EMPTY, GameState
from solutions.mcts import MCTSAgent


class MCTSTests(unittest.TestCase):
    def test_single_legal_move(self) -> None:
        state = GameState(
            board=("X", "O", "X", "X", "O", "O", "O", "X", EMPTY),
            player="X",
        )
        agent = MCTSAgent(evaluator=RandomRolloutEvaluator(seed=1), iterations=50, seed=1)
        move = agent.choose_move(state)
        self.assertEqual(move, 8)

    def test_llm_evaluator_cache_is_used(self) -> None:
        evaluator = LLMPositionEvaluator(provider="heuristic")
        state = GameState(
            board=("X", ".", ".", ".", "O", ".", ".", ".", "."),
            player="X",
        )
        v1 = evaluator.evaluate(state, root_player="X")
        v2 = evaluator.evaluate(state, root_player="X")
        self.assertEqual(v1, v2)
        self.assertGreaterEqual(evaluator.stats["cache_hits"], 1)

    def test_agent_blocks_forced_loss(self) -> None:
        # O must block at index 2, otherwise X wins immediately on the next turn.
        state = GameState(
            board=("X", "X", ".", "O", ".", ".", ".", "O", "."),
            player="O",
        )
        agent = MCTSAgent(evaluator=LLMPositionEvaluator(provider="heuristic"), iterations=300, seed=2)
        move = agent.choose_move(state)
        self.assertEqual(move, 2)


if __name__ == "__main__":
    unittest.main()
