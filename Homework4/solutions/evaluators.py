from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import urllib.error
import urllib.request
from pathlib import Path

from .game import (
    DRAW,
    GameState,
    PLAYER_O,
    PLAYER_X,
    apply_move,
    legal_moves,
    other_player,
    state_key,
    terminal_outcome,
    terminal_value,
)


class RandomRolloutEvaluator:
    """Baseline evaluator that simulates random playouts to termination."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def evaluate(self, state: GameState, root_player: str) -> float:
        terminal = terminal_value(state, root_player)
        if terminal is not None:
            return terminal

        sim = state
        while True:
            outcome = terminal_outcome(sim)
            if outcome is not None:
                if outcome == DRAW:
                    return 0.5
                return 1.0 if outcome == root_player else 0.0
            move = self.rng.choice(legal_moves(sim))
            sim = apply_move(sim, move)


class LLMPositionEvaluator:
    """
    LLM-guided evaluator with local caching.

    provider:
      - heuristic (default): no API calls; deterministic static evaluator
      - openai: query OpenAI chat completions endpoint, fallback to heuristic on error
    """

    def __init__(
        self,
        provider: str = "heuristic",
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        cache_path: str | Path | None = None,
        timeout_seconds: int = 25,
        heuristic_weight: float = 0.7,
        rollout_samples: int = 4,
    ) -> None:
        self.provider = provider.lower().strip()
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.timeout_seconds = timeout_seconds
        self.heuristic_weight = _clamp(heuristic_weight)
        self.rollout_samples = max(0, rollout_samples)
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, float] = {}
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "fallback_to_heuristic": 0,
            "guided_rollouts": 0,
        }

        if self.cache_path and self.cache_path.exists():
            try:
                loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.cache = {k: float(v) for k, v in loaded.items()}
            except Exception:
                # Corrupt cache should not break experiments.
                self.cache = {}

    def evaluate(self, state: GameState, root_player: str) -> float:
        terminal = terminal_value(state, root_player)
        if terminal is not None:
            return terminal

        key = f"{self.provider}|{self.model}|{root_player}|{state_key(state)}"
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]

        self.stats["cache_misses"] += 1
        if self.provider == "openai" and self.api_key:
            try:
                value = self._evaluate_with_openai(state, root_player)
            except Exception:
                self.stats["fallback_to_heuristic"] += 1
                value = self._heuristic_value(state, root_player, key)
        else:
            if self.provider == "openai" and not self.api_key:
                self.stats["fallback_to_heuristic"] += 1
            value = self._heuristic_value(state, root_player, key)

        self.cache[key] = value
        return value

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.cache_path)

    @staticmethod
    def prompt_template() -> str:
        return (
            "You are given a Tic-Tac-Toe board and the root player perspective. "
            "Estimate the probability (0 to 1) that the root player eventually wins "
            "assuming reasonably strong play. Return JSON with keys: "
            "win_probability (number), reason (short string)."
        )

    def _evaluate_with_openai(self, state: GameState, root_player: str) -> float:
        self.stats["api_calls"] += 1
        prompt = _build_prompt(state, root_player)
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.prompt_template()},
                {"role": "user", "content": prompt},
            ],
        }

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url="https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {body[:250]}") from exc

        parsed = json.loads(raw)
        message = parsed["choices"][0]["message"]["content"]
        probability = _parse_probability_from_response(message)
        return _clamp(probability)

    def _heuristic_value(self, state: GameState, root_player: str, key: str) -> float:
        """
        Static evaluator calibrated with short tactical rollouts.
        """
        board = state.board
        opp = other_player(root_player)

        score = 0.0
        score += 0.25 if state.player == root_player else -0.25

        if board[4] == root_player:
            score += 0.65
        elif board[4] == opp:
            score -= 0.65

        corners = (0, 2, 6, 8)
        root_corners = sum(1 for idx in corners if board[idx] == root_player)
        opp_corners = sum(1 for idx in corners if board[idx] == opp)
        score += 0.20 * (root_corners - opp_corners)

        root_two = _count_open_lines(board, root_player, 2)
        opp_two = _count_open_lines(board, opp, 2)
        root_one = _count_open_lines(board, root_player, 1)
        opp_one = _count_open_lines(board, opp, 1)
        score += 0.95 * (root_two - opp_two)
        score += 0.18 * (root_one - opp_one)

        if state.player == root_player and root_two > 0:
            score += 0.75
        if state.player != root_player and opp_two > 0:
            score -= 0.75

        base_probability = 1.0 / (1.0 + math.exp(-score))
        rollout_probability = self._guided_rollout_average(state, root_player, key)
        blended = (self.heuristic_weight * base_probability) + (
            (1.0 - self.heuristic_weight) * rollout_probability
        )
        # Pull slightly toward 0.5 to avoid overconfident lock-in.
        calibrated = 0.5 + (blended - 0.5) * 0.85
        return _clamp(calibrated)

    def _guided_rollout_average(self, state: GameState, root_player: str, key: str) -> float:
        if self.rollout_samples <= 0:
            return 0.5
        seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        total = 0.0
        for _ in range(self.rollout_samples):
            self.stats["guided_rollouts"] += 1
            total += self._guided_rollout(state, root_player, rng)
        return total / self.rollout_samples

    def _guided_rollout(self, state: GameState, root_player: str, rng: random.Random) -> float:
        sim = state
        while True:
            outcome = terminal_outcome(sim)
            if outcome is not None:
                if outcome == DRAW:
                    return 0.5
                return 1.0 if outcome == root_player else 0.0
            move = self._rollout_policy_move(sim, rng)
            sim = apply_move(sim, move)

    def _rollout_policy_move(self, state: GameState, rng: random.Random) -> int:
        legal = legal_moves(state)
        winning = _winning_moves_for_current_player(state, legal)
        if winning:
            return rng.choice(winning)

        safe_moves: list[int] = []
        for move in legal:
            next_state = apply_move(state, move)
            if not _winning_moves_for_current_player(next_state, legal_moves(next_state)):
                safe_moves.append(move)
        candidate_moves = safe_moves if safe_moves else legal

        if 4 in candidate_moves:
            return 4
        corners = [m for m in candidate_moves if m in (0, 2, 6, 8)]
        if corners:
            return rng.choice(corners)
        return rng.choice(candidate_moves)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _count_open_lines(board: tuple[str, ...], player: str, player_marks: int) -> int:
    opponent = PLAYER_X if player != PLAYER_X else PLAYER_O
    lines = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    )
    count = 0
    for line in lines:
        marks = [board[idx] for idx in line]
        if marks.count(opponent) == 0 and marks.count(player) == player_marks:
            count += 1
    return count


def _winning_moves_for_current_player(state: GameState, legal: list[int] | None = None) -> list[int]:
    moves = legal if legal is not None else legal_moves(state)
    winners: list[int] = []
    for move in moves:
        next_state = apply_move(state, move)
        if terminal_outcome(next_state) == state.player:
            winners.append(move)
    return winners


def _build_prompt(state: GameState, root_player: str) -> str:
    board = state.board
    rows = [" ".join(board[idx : idx + 3]) for idx in (0, 3, 6)]
    return (
        f"Root player: {root_player}\n"
        f"Player to move: {state.player}\n"
        "Board:\n"
        f"{rows[0]}\n{rows[1]}\n{rows[2]}\n"
        "Return only JSON."
    )


def _parse_probability_from_response(raw_content: str) -> float:
    try:
        obj = json.loads(raw_content)
        return float(obj["win_probability"])
    except Exception:
        match = re.search(r"([01](?:\.\d+)?)", raw_content)
        if not match:
            raise ValueError(f"Could not parse probability from model output: {raw_content[:200]}")
        return float(match.group(1))
