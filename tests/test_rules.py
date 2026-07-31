"""Game-end detection: checkmate, stalemate, threefold repetition, and
the 50-move rule, verified through ChessEnv (not just python-chess
directly) so we're also testing that termination/reward wiring is
correct."""

import chess

from engine.chess_env import ChessEnv


def play_sans(env: ChessEnv, sans: list[str]):
    result = None
    for san in sans:
        move = env.board.parse_san(san)
        action = env.encode_move(move)
        result = env.step(action)
    return result


def test_checkmate_fools_mate_white_loses():
    env = ChessEnv(player_color=chess.WHITE)
    env.reset()
    obs, reward, terminated, truncated, info = play_sans(
        env, ["f3", "e5", "g4", "Qh4#"]
    )
    assert terminated
    assert not truncated
    assert reward == -1.0  # white (player_color) lost
    assert info["outcome"].termination == chess.Termination.CHECKMATE
    assert info["outcome"].winner == chess.BLACK


def test_checkmate_fools_mate_black_perspective_wins():
    env = ChessEnv(player_color=chess.BLACK)
    env.reset()
    obs, reward, terminated, truncated, info = play_sans(
        env, ["f3", "e5", "g4", "Qh4#"]
    )
    assert terminated
    assert reward == 1.0  # black (player_color) won


def test_stalemate_detection():
    env = ChessEnv()
    env.reset(options={"fen": "7k/8/6Q1/6K1/8/8/8/8 b - - 0 1"})
    assert env.board.is_stalemate()
    mask = env.action_mask()
    # No legal moves at all in a stalemate position.
    assert mask.sum() == 0
    assert env.board.is_game_over(claim_draw=True)


def test_threefold_repetition_terminates_with_draw_reward():
    """python-chess's claim_draw semantics let a player claim a draw
    *before* playing the move that would create the 3rd occurrence (FIDE
    rule: the claim right exists as soon as such a move is available), so
    with claim_draw=True the episode ends after 7 plies here, not 8 —
    verified directly against python-chess in isolation first."""
    env = ChessEnv(player_color=chess.WHITE)
    env.reset()
    sans = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1"]
    obs, reward, terminated, truncated, info = play_sans(env, sans)
    assert terminated
    assert reward == 0.0
    assert info["outcome"].termination == chess.Termination.THREEFOLD_REPETITION
    assert info["outcome"].winner is None


def test_fifty_move_rule_terminates_with_draw_reward():
    """Same early-claim semantics as threefold repetition: the episode
    ends as soon as a reversible move pushes the halfmove clock to 99
    (one move away from the automatic 100), since the mover could then
    claim the draw instead of moving again."""
    env = ChessEnv(player_color=chess.WHITE)
    env.reset(options={"fen": "8/8/8/4k3/8/8/4Q3/4K3 w - - 98 60"})
    move = env.board.parse_san("Kd2")
    action = env.encode_move(move)
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated
    assert reward == 0.0
    assert info["outcome"].termination == chess.Termination.FIFTY_MOVES
