"""action_mask() must agree exactly with python-chess's own legal move
generation (our ground truth) across a range of positions."""

import chess
import numpy as np

from engine.encoding import action_mask, encode_move

POSITIONS = [
    chess.Board(),  # start position
    chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
    # Promotion-rich middlegame-ish position.
    chess.Board("1r3k2/P1P5/8/8/8/8/8/4K2R w K - 0 1"),
    # Position with both sides able to castle both ways.
    chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"),
    # Endgame with few pieces (stresses long sliding moves, distance 7).
    chess.Board("8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1"),
]


def test_action_mask_matches_legal_moves_exactly():
    for board in POSITIONS:
        mask = action_mask(board)
        legal = list(board.legal_moves)

        assert mask.sum() == len(legal), board.fen()

        legal_actions = {encode_move(m) for m in legal}
        assert len(legal_actions) == len(legal), "encode_move collided on two legal moves"

        masked_indices = set(np.flatnonzero(mask).tolist())
        assert masked_indices == legal_actions


def test_action_mask_dtype_and_shape():
    mask = action_mask(chess.Board())
    assert mask.shape == (4672,)
    assert mask.dtype == np.float32
