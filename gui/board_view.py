"""A tkinter Canvas that draws a chess.Board — rendering only, no game
logic or event handling (that lives in gui/app.py and
gui/human_vs_model_app.py).

`square_at()` is the one exception to "rendering only": converting a
click's pixel coordinates into a board square is pure geometry (the
inverse of the same MARGIN/SQUARE math `draw()` already uses to place
squares), not game logic — it doesn't know or care what a legal move
is, that's still entirely the caller's job. Keeping it here means the
pixel<->square mapping is defined in exactly one place."""

import tkinter as tk
import tkinter.font as tkfont
from typing import Optional

import chess


class BoardCanvas(tk.Canvas):
    SQUARE = 64
    MARGIN = 24

    LIGHT_SQUARE = "#EEEED2"
    DARK_SQUARE = "#769656"
    LAST_MOVE_HIGHLIGHT = "#F6F669"
    CHECK_HIGHLIGHT = "#E8635A"
    SELECTED_HIGHLIGHT = "#FF8C00"
    LEGAL_MOVE_DOT = "#2A6FDB"
    BACKGROUND = "#2B2B2B"
    WHITE_PIECE_FILL = "#FAFAFA"
    WHITE_PIECE_OUTLINE = "#202020"
    BLACK_PIECE_FILL = "#202020"
    BLACK_PIECE_OUTLINE = "#FAFAFA"

    def __init__(self, master):
        size = self.MARGIN * 2 + self.SQUARE * 8
        super().__init__(master, width=size, height=size, highlightthickness=0, bg=self.BACKGROUND)
        self.piece_font = tkfont.Font(family="Segoe UI Symbol", size=34)
        self.label_font = tkfont.Font(family="Segoe UI", size=9)

    def square_at(self, x: float, y: float) -> Optional[int]:
        """Inverse of the square-placement math in `draw()`: pixel
        coordinates -> a chess square index, or None if the click
        landed outside the 8x8 board area (e.g. on the coordinate-label
        margin)."""
        file = int((x - self.MARGIN) // self.SQUARE)
        rank = 7 - int((y - self.MARGIN) // self.SQUARE)
        if 0 <= file < 8 and 0 <= rank < 8:
            return chess.square(file, rank)
        return None

    def draw(
        self,
        board: chess.Board,
        last_move: Optional[chess.Move] = None,
        selected_square: Optional[int] = None,
        legal_destinations: Optional[set] = None,
    ):
        """`selected_square`/`legal_destinations` are for human-vs-model
        click-to-move (gui/human_vs_model_app.py) — both default to
        None so the existing spectator app (gui/app.py), which never
        passes them, is completely unaffected."""
        self.delete("all")
        check_square = board.king(board.turn) if board.is_check() else None
        legal_destinations = legal_destinations or set()

        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                x0 = self.MARGIN + file * self.SQUARE
                y0 = self.MARGIN + (7 - rank) * self.SQUARE
                x1, y1 = x0 + self.SQUARE, y0 + self.SQUARE

                color = self.LIGHT_SQUARE if (file + rank) % 2 == 1 else self.DARK_SQUARE
                if last_move is not None and square in (last_move.from_square, last_move.to_square):
                    color = self.LAST_MOVE_HIGHLIGHT
                self.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

                if square == selected_square:
                    self.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2, outline=self.SELECTED_HIGHLIGHT, width=3)

                if square == check_square:
                    pad = 4
                    self.create_oval(
                        x0 + pad, y0 + pad, x1 - pad, y1 - pad,
                        outline=self.CHECK_HIGHLIGHT, width=3,
                    )

                piece = board.piece_at(square)
                if piece is not None:
                    self._draw_piece(x0 + self.SQUARE / 2, y0 + self.SQUARE / 2, piece)

                if square in legal_destinations:
                    self._draw_legal_marker(x0, y0, x1, y1, occupied=piece is not None)

        self._draw_coordinates()

    def _draw_legal_marker(self, x0: float, y0: float, x1: float, y1: float, occupied: bool):
        """Small dot for a legal move to an empty square; a ring around
        the square's edge for a legal capture, so the captured piece
        stays visible underneath (matches the convention most chess
        UIs use)."""
        if occupied:
            pad = 4
            self.create_oval(x0 + pad, y0 + pad, x1 - pad, y1 - pad, outline=self.LEGAL_MOVE_DOT, width=3)
        else:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = self.SQUARE * 0.14
            self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=self.LEGAL_MOVE_DOT, outline="")

    def _draw_piece(self, cx: float, cy: float, piece: chess.Piece):
        glyph = chess.UNICODE_PIECE_SYMBOLS[piece.symbol()]
        if piece.color == chess.WHITE:
            fill, outline = self.WHITE_PIECE_FILL, self.WHITE_PIECE_OUTLINE
        else:
            fill, outline = self.BLACK_PIECE_FILL, self.BLACK_PIECE_OUTLINE

        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            self.create_text(cx + dx, cy + dy, text=glyph, font=self.piece_font, fill=outline)
        self.create_text(cx, cy, text=glyph, font=self.piece_font, fill=fill)

    def _draw_coordinates(self):
        for file in range(8):
            x = self.MARGIN + file * self.SQUARE + self.SQUARE / 2
            self.create_text(
                x, self.MARGIN + self.SQUARE * 8 + self.MARGIN / 2,
                text=chess.FILE_NAMES[file], font=self.label_font, fill="#CCCCCC",
            )
        for rank in range(8):
            y = self.MARGIN + (7 - rank) * self.SQUARE + self.SQUARE / 2
            self.create_text(
                self.MARGIN / 2, y,
                text=chess.RANK_NAMES[rank], font=self.label_font, fill="#CCCCCC",
            )
