"""Headless human-vs-model game logic: no tkinter import anywhere in
this file. This is deliberate — it's what makes the feature directly
unit-testable (tests/test_human_vs_model.py drives a full game by
calling these methods with a scripted "human" move-picker, no GUI
event loop, no mouse clicks, no display needed) instead of only being
exercisable by a person clicking a window.

gui/human_vs_model_app.py (the Tkinter layer) is a thin wrapper around
this class: it turns mouse clicks into calls to `legal_moves_from()`/
`apply_human_move()`, runs `compute_agent_move()` on a background
thread so the UI doesn't freeze, and renders whatever this class's
state says. All chess/network logic lives here exactly once.
"""

from pathlib import Path

import chess
import torch

from engine.chess_env import ChessEnv
from training.checkpoint import load_checkpoint
from training.mcts import MCTS, select_action
from training.network import PolicyValueNet


class HumanVsModelController:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        num_simulations: int = 200,
        human_color: chess.Color = chess.WHITE,
        c_puct: float = 1.5,
        claim_draw: bool = True,
        max_plies: int = 600,
    ):
        self.device = device
        self.num_simulations = num_simulations
        self.human_color = human_color
        self.claim_draw = claim_draw
        self.max_plies = max_plies

        # Pure inference — no optimizer to restore, matching how
        # evaluation.py's checkpoint-vs-checkpoint games load networks.
        self.network = PolicyValueNet().to(device)
        load_checkpoint(checkpoint_path, self.network, optimizer=None, device=device)
        self.network.eval()

        self.mcts = MCTS(self.network, device=device, c_puct=c_puct, claim_draw=claim_draw)

        self.env = ChessEnv(claim_draw=claim_draw)
        self.ply = 0
        self.env.reset()

    @property
    def board(self) -> chess.Board:
        return self.env.board

    def new_game(self, human_color: chess.Color | None = None) -> None:
        """Starts a fresh game. `human_color` lets the player pick
        color again each game (requirement: choose color "at the start
        of each game", not just once at launch)."""
        if human_color is not None:
            self.human_color = human_color
        self.env.reset()
        self.ply = 0

    def is_human_turn(self) -> bool:
        return self.env.current_player() == self.human_color

    def is_game_over(self) -> bool:
        return self.env.board.is_game_over(claim_draw=self.claim_draw) or self.ply >= self.max_plies

    def legal_moves_from(self, square: chess.Square) -> list[chess.Move]:
        return [m for m in self.env.board.legal_moves if m.from_square == square]

    def apply_human_move(self, move: chess.Move) -> None:
        if not self.is_human_turn():
            raise RuntimeError("apply_human_move() called when it isn't the human's turn")
        if move not in self.env.board.legal_moves:
            raise ValueError(f"illegal move: {move}")
        self.env.step(self.env.encode_move(move))
        self.ply += 1

    def compute_agent_move(self) -> chess.Move:
        """The slow, blocking part (the whole point of running this on
        a background thread in the GUI layer) — deliberately does NOT
        mutate board state, so it's safe to call from a non-main
        thread while the main thread is otherwise idle; see
        `apply_agent_move()`.

        Greedy (temperature=0) and no Dirichlet noise: this is the
        agent playing its strongest move against a real opponent, not
        generating exploratory self-play data — same convention
        already used by training/evaluation.py's `mcts_player`."""
        root = self.mcts.run(self.env.board, self.num_simulations)
        action = select_action(root, temperature=0.0)
        return self.env.decode_move(action)

    def apply_agent_move(self, move: chess.Move) -> None:
        self.env.step(self.env.encode_move(move))
        self.ply += 1

    def get_result_text(self) -> str:
        outcome = self.env.board.outcome(claim_draw=self.claim_draw)
        if outcome is None:
            return "Game stopped: move limit reached without a decision."

        termination = outcome.termination.name.replace("_", " ").title()
        if outcome.winner is None:
            return f"Draw — {termination}."

        winner_is_human = outcome.winner == self.human_color
        winner_name = "White" if outcome.winner else "Black"
        who = "You win!" if winner_is_human else "The model wins."
        return f"{winner_name} wins by {termination}. {who}"
