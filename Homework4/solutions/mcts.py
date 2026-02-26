from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time

from .game import GameState, apply_move, legal_moves, terminal_outcome


@dataclass
class Node:
    state: GameState
    parent: "Node | None" = None
    move: int | None = None
    children: dict[int, "Node"] = field(default_factory=dict)
    untried_moves: list[int] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    def __post_init__(self) -> None:
        if not self.untried_moves:
            self.untried_moves = legal_moves(self.state)

    def is_terminal(self) -> bool:
        return terminal_outcome(self.state) is not None

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


@dataclass
class DecisionStats:
    move: int
    visits: int
    estimated_value: float
    elapsed_ms: float
    ms_per_iteration: float


class MCTSAgent:
    def __init__(
        self,
        evaluator,
        iterations: int = 100,
        exploration: float = math.sqrt(2.0),
        seed: int | None = None,
        name: str = "mcts",
    ) -> None:
        if iterations <= 0:
            raise ValueError("iterations must be > 0")
        self.evaluator = evaluator
        self.iterations = iterations
        self.exploration = exploration
        self.rng = random.Random(seed)
        self.name = name
        self.last_decision: DecisionStats | None = None

    def choose_move(self, root_state: GameState) -> int:
        root = Node(root_state)
        start = time.perf_counter()
        root_player = root_state.player

        for _ in range(self.iterations):
            node = root

            while not node.is_terminal() and node.is_fully_expanded() and node.children:
                node = self._select_child(node, root_player)

            if not node.is_terminal() and node.untried_moves:
                move = self.rng.choice(node.untried_moves)
                node.untried_moves.remove(move)
                child = Node(state=apply_move(node.state, move), parent=node, move=move)
                node.children[move] = child
                node = child

            value = self.evaluator.evaluate(node.state, root_player=root_state.player)

            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if not root.children:
            move = self.rng.choice(legal_moves(root_state))
            self.last_decision = DecisionStats(
                move=move,
                visits=0,
                estimated_value=0.5,
                elapsed_ms=elapsed_ms,
                ms_per_iteration=elapsed_ms / self.iterations,
            )
            return move

        # Standard move selection is highest visit count.
        ranked = sorted(
            root.children.values(),
            key=lambda c: (c.visits, c.mean_value()),
            reverse=True,
        )
        best = ranked[0]
        self.last_decision = DecisionStats(
            move=best.move if best.move is not None else -1,
            visits=best.visits,
            estimated_value=best.mean_value(),
            elapsed_ms=elapsed_ms,
            ms_per_iteration=elapsed_ms / self.iterations,
        )
        return best.move if best.move is not None else self.rng.choice(legal_moves(root_state))

    def _select_child(self, node: Node, root_player: str) -> Node:
        assert node.children
        log_parent = math.log(max(1, node.visits))
        node_player = node.state.player

        def ucb(child: Node) -> float:
            if child.visits == 0:
                return float("inf")
            root_value = child.value_sum / child.visits
            exploit = root_value if node_player == root_player else (1.0 - root_value)
            explore = self.exploration * math.sqrt(log_parent / child.visits)
            return exploit + explore

        best_score = float("-inf")
        best_children: list[Node] = []
        for child in node.children.values():
            score = ucb(child)
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
        return self.rng.choice(best_children)
