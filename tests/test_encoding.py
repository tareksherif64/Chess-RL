"""Board encoding shape/dtype checks, and move<->action round-trip
correctness across many game positions (including promotions,
castling, en passant)."""

import chess
import numpy as np

from engine.encoding import (
    ACTION_SPACE_SIZE,
    NUM_PLANES,
    decode_move,
    encode_board,
    encode_move,
)


def test_encode_board_shape_and_dtype():
    board = chess.Board()
    obs = encode_board(board)
    assert obs.shape == (8, 8, NUM_PLANES)
    assert obs.dtype == np.float32


def test_encode_board_initial_position_piece_planes():
    board = chess.Board()
    obs = encode_board(board)

    # White pawns on rank index 1 (rank 2), plane 0.
    assert np.all(obs[1, :, 0] == 1.0)
    # Black pawns on rank index 6 (rank 7), plane 6.
    assert np.all(obs[6, :, 6] == 1.0)
    # White king on e1 -> rank 0, file 4, plane 5.
    assert obs[0, 4, 5] == 1.0
    # Black king on e8 -> rank 7, file 4, plane 11.
    assert obs[7, 4, 11] == 1.0
    # No pieces on empty middle ranks.
    assert obs[2:6, :, :12].sum() == 0.0


def test_side_to_move_plane_toggles():
    board = chess.Board()
    obs = encode_board(board)
    assert np.all(obs[:, :, 12] == 1.0)  # white to move

    board.push_san("e4")
    obs = encode_board(board)
    assert np.all(obs[:, :, 12] == 0.0)  # black to move


def test_castling_rights_planes_initial_position():
    board = chess.Board()
    obs = encode_board(board)
    assert np.all(obs[:, :, 13] == 1.0)
    assert np.all(obs[:, :, 14] == 1.0)
    assert np.all(obs[:, :, 15] == 1.0)
    assert np.all(obs[:, :, 16] == 1.0)


def test_castling_rights_plane_clears_after_king_move():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Ke2")  # white king moves, forfeits both castling rights
    obs = encode_board(board)
    assert np.all(obs[:, :, 13] == 0.0)
    assert np.all(obs[:, :, 14] == 0.0)
    # Black rights untouched.
    assert np.all(obs[:, :, 15] == 1.0)
    assert np.all(obs[:, :, 16] == 1.0)


def test_en_passant_plane():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("a6")
    board.push_san("e5")
    board.push_san("d5")  # black pawn double push next to white pawn on e5
    obs = encode_board(board)
    assert board.ep_square == chess.D6
    assert obs[5, 3, 17] == 1.0  # d6 -> rank index 5, file index 3
    assert obs[:, :, 17].sum() == 1.0


def test_action_space_size_is_alphazero_scheme():
    assert ACTION_SPACE_SIZE == 64 * 73 == 4672


def test_move_round_trip_over_random_games_including_promotions():
    """Every legal move in every position of many random games must
    round-trip through encode_move -> decode_move back to an equal move."""
    rng = np.random.default_rng(0)
    positions_checked = 0
    promotions_checked = 0

    for game in range(15):
        board = chess.Board()
        for _ply in range(80):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            for move in legal:
                action = encode_move(move)
                assert 0 <= action < ACTION_SPACE_SIZE
                decoded = decode_move(action, board)
                assert decoded == move, f"{move} -> {action} -> {decoded}"
                if move.promotion is not None:
                    promotions_checked += 1
            positions_checked += 1
            move = legal[rng.integers(len(legal))]
            board.push(move)

    assert positions_checked > 100
    assert promotions_checked > 0, "expected at least one promotion move in random games"


def test_underpromotion_round_trip_explicit():
    # White pawn on a7, black rook on b8, king elsewhere: axb8=N/B/R all legal.
    board = chess.Board("1r3k2/P7/8/8/8/8/8/4K3 w - - 0 1")
    for promo in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        move = chess.Move(chess.A7, chess.B8, promotion=promo)
        assert move in board.legal_moves
        action = encode_move(move)
        decoded = decode_move(action, board)
        assert decoded == move


def test_castling_move_round_trip():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    kingside = chess.Move(chess.E1, chess.G1)
    assert kingside in board.legal_moves
    assert decode_move(encode_move(kingside), board) == kingside


def test_en_passant_capture_round_trip():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("a6")
    board.push_san("e5")
    board.push_san("d5")
    ep_move = chess.Move(chess.E5, chess.D6)
    assert ep_move in board.legal_moves
    assert board.is_en_passant(ep_move)
    assert decode_move(encode_move(ep_move), board) == ep_move
