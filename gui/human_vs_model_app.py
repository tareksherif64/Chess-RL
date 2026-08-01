"""Tkinter desktop app: a human plays against a loaded checkpoint by
clicking the board. All chess/network logic lives in
HumanVsModelController (gui/human_vs_model_controller.py, no tkinter
import) — this file only wires mouse clicks to it, runs its slow
MCTS call on a background thread so the window doesn't freeze, and
renders whatever state the controller reports. Same click-selects-
then-click-destination interaction most chess UIs use, drawn via
BoardCanvas's selected/legal-destination highlighting.
"""

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import chess
import torch

from gui.board_view import BoardCanvas
from gui.human_vs_model_controller import HumanVsModelController


class HumanVsModelApp(tk.Tk):
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        num_simulations: int = 200,
        c_puct: float = 1.5,
    ):
        super().__init__()
        self.title("Chess RL — Play vs Model")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.checkpoint_path = checkpoint_path
        self.controller = HumanVsModelController(
            checkpoint_path=checkpoint_path,
            device=device,
            num_simulations=num_simulations,
            c_puct=c_puct,
            human_color=chess.WHITE,
        )

        # Click-to-move state: the square the human has selected (if
        # any) and which squares are legal destinations from it — both
        # purely UI state, not game state, so they live here rather
        # than in the controller.
        self.selected_square: int | None = None
        self.legal_targets: set[int] = set()
        self.last_move: chess.Move | None = None

        # Background-thread bookkeeping for the agent's move (see
        # _start_agent_thinking below) — a Queue is the standard safe
        # hand-off from a worker thread back to Tkinter's main thread,
        # which is the only thread allowed to touch widgets.
        self.agent_thinking = False
        self._move_queue: "queue.Queue[chess.Move]" = queue.Queue()
        self._thinking_frame = 0

        self._build_widgets()
        self._start_new_game(chess.WHITE)

    def _build_widgets(self):
        main = ttk.Frame(self, padding=10)
        main.grid(row=0, column=0)

        self.board_canvas = BoardCanvas(main)
        self.board_canvas.grid(row=0, column=0)
        self.board_canvas.bind("<Button-1>", self._on_board_click)

        side = ttk.Frame(main, padding=(14, 0, 0, 0))
        side.grid(row=0, column=1, sticky="n")

        self.status_var = tk.StringVar()
        ttk.Label(side, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(anchor="w")

        self.thinking_var = tk.StringVar(value="")
        ttk.Label(side, textvariable=self.thinking_var, foreground="#2A6FDB").pack(anchor="w", pady=(2, 0))

        self.last_move_var = tk.StringVar(value="")
        ttk.Label(side, textvariable=self.last_move_var).pack(anchor="w", pady=(2, 0))

        self.result_var = tk.StringVar(value="")
        ttk.Label(
            side, textvariable=self.result_var, foreground="#c0392b",
            font=("Segoe UI", 10, "bold"), wraplength=190,
        ).pack(anchor="w", pady=(6, 14))

        ttk.Label(side, text="Play as:").pack(anchor="w", pady=(8, 0))
        self.color_choice = tk.StringVar(value="white")
        color_frame = ttk.Frame(side)
        color_frame.pack(anchor="w")
        ttk.Radiobutton(color_frame, text="White", value="white", variable=self.color_choice).pack(side="left")
        ttk.Radiobutton(color_frame, text="Black", value="black", variable=self.color_choice).pack(side="left", padx=(8, 0))

        ttk.Button(side, text="New Game", command=self._on_new_game_clicked).pack(fill="x", pady=(8, 2))

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=(16, 8))
        ttk.Label(side, text=f"Checkpoint:\n{Path(self.checkpoint_path).name}", wraplength=190).pack(anchor="w")
        ttk.Label(side, text=f"Simulations/move: {self.controller.num_simulations}").pack(anchor="w", pady=(4, 0))

    # --- new game / color choice -------------------------------------------------

    def _on_new_game_clicked(self):
        color = chess.WHITE if self.color_choice.get() == "white" else chess.BLACK
        self._start_new_game(color)

    def _start_new_game(self, human_color: chess.Color):
        self.controller.new_game(human_color)
        self.selected_square = None
        self.legal_targets = set()
        self.last_move = None
        self.result_var.set("")
        self.last_move_var.set("")
        self._redraw()
        self._advance_if_agents_turn()

    # --- click-to-move -------------------------------------------------

    def _on_board_click(self, event):
        if self.agent_thinking or self.controller.is_game_over() or not self.controller.is_human_turn():
            return

        square = self.board_canvas.square_at(event.x, event.y)
        if square is None:
            return

        board = self.controller.board

        if self.selected_square is not None and square in self.legal_targets:
            move = self._resolve_move(self.selected_square, square)
            self.selected_square = None
            self.legal_targets = set()
            self._make_human_move(move)
            return

        if square == self.selected_square:
            self.selected_square = None
            self.legal_targets = set()
            self._redraw()
            return

        piece = board.piece_at(square)
        if piece is not None and piece.color == self.controller.human_color:
            self.selected_square = square
            self.legal_targets = {m.to_square for m in self.controller.legal_moves_from(square)}
        else:
            self.selected_square = None
            self.legal_targets = set()
        self._redraw()

    def _resolve_move(self, from_square: int, to_square: int) -> chess.Move:
        """A promotion has 4 legal moves sharing the same (from, to) —
        one per promotion piece. Clicking the destination square can't
        distinguish which the human wants, so this always promotes to
        queen (the overwhelming majority choice in practice); see
        docs/human_vs_model.md for the underpromotion tradeoff this
        makes and how to work around it if you specifically need one."""
        candidates = [m for m in self.controller.legal_moves_from(from_square) if m.to_square == to_square]
        for move in candidates:
            if move.promotion in (None, chess.QUEEN):
                return move
        return candidates[0]

    def _make_human_move(self, move: chess.Move):
        san = self.controller.board.san(move)
        self.controller.apply_human_move(move)
        self.last_move = move
        self.last_move_var.set(f"You played: {san}")
        self._redraw()
        self._advance_if_agents_turn()

    # --- agent move (background thread + polling) -------------------------------------------------

    def _advance_if_agents_turn(self):
        if self.controller.is_game_over():
            self._show_result()
        elif not self.controller.is_human_turn():
            self._start_agent_thinking()

    def _start_agent_thinking(self):
        self.agent_thinking = True
        self._thinking_frame = 0
        self._animate_thinking()
        threading.Thread(target=self._agent_worker, daemon=True).start()
        self.after(100, self._poll_agent_move)

    def _agent_worker(self):
        # Runs off the main thread. Only *reads* board state (via MCTS
        # simulating on copies — see training/mcts.py) and does not
        # touch any tkinter widget; the move is applied back on the
        # main thread in _poll_agent_move, so all board mutation stays
        # single-threaded.
        move = self.controller.compute_agent_move()
        self._move_queue.put(move)

    def _animate_thinking(self):
        if not self.agent_thinking:
            self.thinking_var.set("")
            return
        dots = "." * (self._thinking_frame % 4)
        self.thinking_var.set(f"Model is thinking{dots}")
        self._thinking_frame += 1
        self.after(400, self._animate_thinking)

    def _poll_agent_move(self):
        if not self.winfo_exists():
            return
        try:
            move = self._move_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_agent_move)
            return

        self.agent_thinking = False
        san = self.controller.board.san(move)
        self.controller.apply_agent_move(move)
        self.last_move = move
        self.last_move_var.set(f"Model played: {san}")
        self._redraw()
        self._advance_if_agents_turn()

    # --- rendering -------------------------------------------------

    def _show_result(self):
        self.result_var.set(self.controller.get_result_text())
        self._redraw()

    def _redraw(self):
        board = self.controller.board
        turn_name = "White" if board.turn == chess.WHITE else "Black"
        whose_move = "Your move" if self.controller.is_human_turn() else "Model's move"
        self.status_var.set(f"{turn_name} to move — {whose_move}")
        self.board_canvas.draw(
            board, self.last_move,
            selected_square=self.selected_square,
            legal_destinations=self.legal_targets,
        )

    def _on_close(self):
        self.destroy()
