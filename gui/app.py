"""Tkinter desktop app that drives ChessEnv with two RandomAgents and
renders it live via BoardCanvas. Game logic stays in engine/agents —
this file only wires them to widgets and a tk.after()-based move loop.
"""

import tkinter as tk
from tkinter import ttk

import chess

from agents.random_agent import RandomAgent
from engine.chess_env import ChessEnv
from gui.board_view import BoardCanvas


class ChessWatcherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chess RL — Random vs Random")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.env = ChessEnv()
        self.white_agent = RandomAgent()
        self.black_agent = RandomAgent()

        self.last_move: chess.Move | None = None
        self.playing = False
        self.ply = 0
        self.games_played = 0
        self.delay_ms = tk.IntVar(value=500)
        self.auto_next_var = tk.BooleanVar(value=True)

        self._build_widgets()
        self._new_game()

    def _build_widgets(self):
        main = ttk.Frame(self, padding=10)
        main.grid(row=0, column=0)

        self.board_canvas = BoardCanvas(main)
        self.board_canvas.grid(row=0, column=0)

        side = ttk.Frame(main, padding=(14, 0, 0, 0))
        side.grid(row=0, column=1, sticky="n")

        self.status_var = tk.StringVar()
        ttk.Label(side, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(anchor="w")

        self.last_move_var = tk.StringVar(value="")
        ttk.Label(side, textvariable=self.last_move_var).pack(anchor="w", pady=(2, 0))

        self.result_var = tk.StringVar(value="")
        ttk.Label(
            side, textvariable=self.result_var, foreground="#c0392b",
            font=("Segoe UI", 10, "bold"), wraplength=180,
        ).pack(anchor="w", pady=(6, 14))

        self.play_button = ttk.Button(side, text="Pause", command=self._toggle_play)
        self.play_button.pack(fill="x", pady=2)
        ttk.Button(side, text="Step", command=self._do_ply).pack(fill="x", pady=2)
        ttk.Button(side, text="New Game", command=self._new_game).pack(fill="x", pady=2)

        ttk.Label(side, text="Speed (ms/move)").pack(anchor="w", pady=(14, 0))
        ttk.Scale(side, from_=2000, to=50, variable=self.delay_ms, orient="horizontal").pack(fill="x")

        ttk.Checkbutton(
            side, text="Auto-start next game", variable=self.auto_next_var
        ).pack(anchor="w", pady=(14, 0))

        self.games_played_var = tk.StringVar(value="Games played: 0")
        ttk.Label(side, textvariable=self.games_played_var).pack(anchor="w", pady=(20, 0))

    def _new_game(self):
        self.env.reset()
        self.last_move = None
        self.ply = 0
        self.playing = True
        self.play_button.config(text="Pause")
        self.result_var.set("")
        self.last_move_var.set("")
        self._redraw()
        self._schedule_tick()

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_button.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self._schedule_tick()

    def _schedule_tick(self):
        if self.playing:
            self.after(self.delay_ms.get(), self._tick)

    def _tick(self):
        if not self.playing:
            return
        terminated = self._do_ply()
        if not terminated:
            self._schedule_tick()

    def _do_ply(self) -> bool:
        if not self.winfo_exists():
            return True
        if self.env.board.is_game_over(claim_draw=self.env.claim_draw):
            return True

        white_to_move = self.env.current_player() == chess.WHITE
        agent = self.white_agent if white_to_move else self.black_agent
        action = agent.select_action(self.env.action_mask())
        move = self.env.decode_move(action)
        san = self.env.board.san(move)

        _, _reward, terminated, _truncated, info = self.env.step(action)
        self.last_move = move
        self.ply += 1
        self.last_move_var.set(f"Last move: {'White' if white_to_move else 'Black'} {san}")
        self._redraw()

        if terminated:
            self.playing = False
            self.play_button.config(text="Play")
            outcome = info["outcome"]
            result = outcome.result() if outcome else "*"
            termination = outcome.termination.name.replace("_", " ").title() if outcome else "Unknown"
            self.result_var.set(f"Game over: {result}\n({termination})")
            self.games_played += 1
            self.games_played_var.set(f"Games played: {self.games_played}")
            if self.auto_next_var.get():
                self.after(max(self.delay_ms.get(), 1500), self._new_game_if_alive)

        return terminated

    def _new_game_if_alive(self):
        if self.winfo_exists():
            self._new_game()

    def _redraw(self):
        mover = "White" if self.env.current_player() == chess.WHITE else "Black"
        self.status_var.set(f"{mover} to move   |   Ply {self.ply}")
        self.board_canvas.draw(self.env.board, self.last_move)

    def _on_close(self):
        self.playing = False
        self.destroy()
