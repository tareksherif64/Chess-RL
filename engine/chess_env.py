"""Gymnasium-style environment wrapping python-chess."""

from typing import Any, Optional

import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from engine.encoding import (
    ACTION_SPACE_SIZE,
    NUM_PLANES,
    action_mask,
    decode_move,
    encode_board,
    encode_move,
)


class ChessEnv(gym.Env):
    """A two-player chess environment, one ply per step().

    `player_color` only determines whose perspective terminal reward is
    reported from; both colors' moves are submitted through the same
    step() call, whichever side is currently to move (see
    `current_player`). This keeps the env a plain board simulator with no
    embedded opponent policy, so it can be driven by two agents (random,
    self-play, human, etc.) alternately. See docs/engine.md.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        player_color: chess.Color = chess.WHITE,
        claim_draw: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.player_color = player_color
        self.claim_draw = claim_draw
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8, 8, NUM_PLANES), dtype=np.float32
        )

        self.board = chess.Board()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        options = options or {}
        self.player_color = options.get("player_color", self.player_color)
        fen = options.get("fen")
        self.board = chess.Board(fen) if fen else chess.Board()

        obs = encode_board(self.board)
        info = self._info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.board.is_game_over(claim_draw=self.claim_draw):
            raise RuntimeError("step() called after game over; call reset()")

        move = decode_move(action, self.board)
        if move not in self.board.legal_moves:
            raise ValueError(
                f"Illegal action {action} decoded to {move}; use action_mask() "
                "to select from legal actions only."
            )

        self.board.push(move)

        terminated = self.board.is_game_over(claim_draw=self.claim_draw)
        truncated = False
        reward = 0.0
        if terminated:
            reward = self._terminal_reward()

        obs = encode_board(self.board)
        info = self._info()
        return obs, reward, terminated, truncated, info

    def legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    def action_mask(self) -> np.ndarray:
        return action_mask(self.board)

    def current_player(self) -> chess.Color:
        return self.board.turn

    def encode_move(self, move: chess.Move) -> int:
        return encode_move(move)

    def decode_move(self, action: int) -> chess.Move:
        return decode_move(action, self.board)

    def render(self):
        if self.render_mode == "ansi":
            return str(self.board)
        return None

    def _terminal_reward(self) -> float:
        outcome = self.board.outcome(claim_draw=self.claim_draw)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == self.player_color else -1.0

    def _info(self) -> dict[str, Any]:
        return {
            "turn": self.board.turn,
            "fullmove_number": self.board.fullmove_number,
            "is_check": self.board.is_check(),
            "outcome": self.board.outcome(claim_draw=self.claim_draw),
        }
