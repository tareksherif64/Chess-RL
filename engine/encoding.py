"""
Board -> tensor encoding, and move <-> discrete action encoding.

Action space: AlphaZero-style 8x8x73 = 4672 scheme.
  - 64 "from" squares (a1..h8)
  - 73 move types per from-square:
      0-55  : 8 "queen" directions x 7 distances (covers all king/rook/
              bishop/queen moves, all pawn single/double pushes and
              captures, all castling moves, and queen-promotion moves)
      56-63 : 8 knight-move deltas
      64-72 : 9 underpromotions (3 directions [straight, capture-left,
              capture-right] x 3 piece types [knight, bishop, rook])

See docs/action_space.md for the full rationale.
"""

import chess
import numpy as np

# --- Board encoding -------------------------------------------------------

NUM_PIECE_PLANES = 12
NUM_EXTRA_PLANES = 6  # side-to-move, 4x castling rights, en-passant
NUM_PLANES = NUM_PIECE_PLANES + NUM_EXTRA_PLANES  # 18

_PIECE_PLANE_INDEX = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

SIDE_TO_MOVE_PLANE = 12
WHITE_KINGSIDE_PLANE = 13
WHITE_QUEENSIDE_PLANE = 14
BLACK_KINGSIDE_PLANE = 15
BLACK_QUEENSIDE_PLANE = 16
EN_PASSANT_PLANE = 17


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a python-chess Board as an (8, 8, 18) float32 tensor.

    Orientation is absolute (rank 0 = rank 1 / white's back rank, file 0 =
    file a), regardless of whose turn it is. Index as obs[rank, file, plane].
    """
    obs = np.zeros((8, 8, NUM_PLANES), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        plane = _PIECE_PLANE_INDEX[(piece.piece_type, piece.color)]
        obs[rank, file, plane] = 1.0

    if board.turn == chess.WHITE:
        obs[:, :, SIDE_TO_MOVE_PLANE] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        obs[:, :, WHITE_KINGSIDE_PLANE] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        obs[:, :, WHITE_QUEENSIDE_PLANE] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        obs[:, :, BLACK_KINGSIDE_PLANE] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        obs[:, :, BLACK_QUEENSIDE_PLANE] = 1.0

    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        obs[rank, file, EN_PASSANT_PLANE] = 1.0

    return obs


# --- Action encoding --------------------------------------------------------

ACTION_SPACE_SIZE = 64 * 73

# 8 queen-move directions as (delta_file, delta_rank), index order N, NE, E,
# SE, S, SW, W, NW.
_QUEEN_DIRECTIONS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]
_QUEEN_DIRECTION_INDEX = {d: i for i, d in enumerate(_QUEEN_DIRECTIONS)}

# 8 knight-move deltas as (delta_file, delta_rank).
_KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]
_KNIGHT_DELTA_INDEX = {d: i for i, d in enumerate(_KNIGHT_DELTAS)}

# Underpromotion categories by delta_file: straight, capture toward file-1,
# capture toward file+1.
_UNDERPROMO_CATEGORY_BY_DFILE = {0: 0, -1: 1, 1: 2}
_UNDERPROMO_DFILE_BY_CATEGORY = {0: 0, 1: -1, 2: 1}
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
_UNDERPROMO_PIECE_INDEX = {p: i for i, p in enumerate(_UNDERPROMO_PIECES)}

QUEEN_PLANES_START = 0
KNIGHT_PLANES_START = 56
UNDERPROMO_PLANES_START = 64
PLANES_PER_SQUARE = 73


def encode_move(move: chess.Move) -> int:
    """Encode a legal chess.Move as an integer in [0, 4672)."""
    from_square = move.from_square
    to_square = move.to_square
    from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
    to_file, to_rank = chess.square_file(to_square), chess.square_rank(to_square)
    delta_file = to_file - from_file
    delta_rank = to_rank - from_rank

    if move.promotion is not None and move.promotion != chess.QUEEN:
        category = _UNDERPROMO_CATEGORY_BY_DFILE[delta_file]
        piece_idx = _UNDERPROMO_PIECE_INDEX[move.promotion]
        plane = UNDERPROMO_PLANES_START + category * 3 + piece_idx
        return from_square * PLANES_PER_SQUARE + plane

    if (abs(delta_file), abs(delta_rank)) in ((1, 2), (2, 1)):
        knight_idx = _KNIGHT_DELTA_INDEX[(delta_file, delta_rank)]
        plane = KNIGHT_PLANES_START + knight_idx
        return from_square * PLANES_PER_SQUARE + plane

    distance = max(abs(delta_file), abs(delta_rank))
    unit = (_sign(delta_file), _sign(delta_rank))
    dir_idx = _QUEEN_DIRECTION_INDEX[unit]
    plane = QUEEN_PLANES_START + dir_idx * 7 + (distance - 1)
    return from_square * PLANES_PER_SQUARE + plane


def decode_move(action: int, board: chess.Board) -> chess.Move:
    """Decode an action index back into a chess.Move, using board context
    to determine automatic queen-promotion (pawn reaching the last rank).
    """
    from_square, plane = divmod(action, PLANES_PER_SQUARE)
    from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)

    if plane < KNIGHT_PLANES_START:
        dir_idx, dist_idx = divmod(plane, 7)
        delta_file, delta_rank = _QUEEN_DIRECTIONS[dir_idx]
        distance = dist_idx + 1
        to_file = from_file + delta_file * distance
        to_rank = from_rank + delta_rank * distance
        to_square = chess.square(to_file, to_rank)

        promotion = None
        piece = board.piece_at(from_square)
        if piece is not None and piece.piece_type == chess.PAWN and to_rank in (0, 7):
            promotion = chess.QUEEN
        return chess.Move(from_square, to_square, promotion=promotion)

    if plane < UNDERPROMO_PLANES_START:
        knight_idx = plane - KNIGHT_PLANES_START
        delta_file, delta_rank = _KNIGHT_DELTAS[knight_idx]
        to_square = chess.square(from_file + delta_file, from_rank + delta_rank)
        return chess.Move(from_square, to_square)

    offset = plane - UNDERPROMO_PLANES_START
    category, piece_idx = divmod(offset, 3)
    delta_file = _UNDERPROMO_DFILE_BY_CATEGORY[category]
    delta_rank = 1 if from_rank == 6 else -1  # rank 7 -> white, rank 2 -> black
    to_square = chess.square(from_file + delta_file, from_rank + delta_rank)
    promotion = _UNDERPROMO_PIECES[piece_idx]
    return chess.Move(from_square, to_square, promotion=promotion)


def action_mask(board: chess.Board) -> np.ndarray:
    """Boolean legality mask over the full action space, shape (4672,)."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        mask[encode_move(move)] = 1.0
    return mask


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)
