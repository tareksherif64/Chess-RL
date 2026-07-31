"""A tkinter Canvas that draws a chess.Board — rendering only, no game
logic or event handling (that lives in gui/app.py)."""

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

    def draw(self, board: chess.Board, last_move: Optional[chess.Move] = None):
        self.delete("all")
        check_square = board.king(board.turn) if board.is_check() else None

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

                if square == check_square:
                    pad = 4
                    self.create_oval(
                        x0 + pad, y0 + pad, x1 - pad, y1 - pad,
                        outline=self.CHECK_HIGHLIGHT, width=3,
                    )

                piece = board.piece_at(square)
                if piece is not None:
                    self._draw_piece(x0 + self.SQUARE / 2, y0 + self.SQUARE / 2, piece)

        self._draw_coordinates()

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
