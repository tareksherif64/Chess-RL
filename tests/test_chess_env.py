"""ChessEnv API-level tests: obs contract, illegal-action handling,
legal-move ground truth vs python-chess, and full-game smoke tests
using the random baseline agent for both colors."""

import chess
import numpy as np
import pytest

from agents.random_agent import RandomAgent
from engine.chess_env import ChessEnv
from engine.encoding import ACTION_SPACE_SIZE, NUM_PLANES


def test_reset_observation_contract():
    env = ChessEnv()
    obs, info = env.reset()
    assert obs.shape == (8, 8, NUM_PLANES)
    assert obs.dtype == np.float32
    assert env.action_space.n == ACTION_SPACE_SIZE
    assert env.observation_space.shape == (8, 8, NUM_PLANES)


def test_legal_moves_matches_python_chess_ground_truth():
    env = ChessEnv()
    env.reset(options={"fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"})
    assert set(env.legal_moves()) == set(env.board.legal_moves)


def test_step_rejects_illegal_action():
    env = ChessEnv()
    env.reset()
    mask = env.action_mask()
    illegal_action = int(np.flatnonzero(mask == 0)[0])
    with pytest.raises(ValueError):
        env.step(illegal_action)


def test_step_after_game_over_raises():
    env = ChessEnv()
    env.reset(options={"fen": "7k/8/6Q1/6K1/8/8/8/8 b - - 0 1"})
    assert env.board.is_game_over(claim_draw=True)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_intermediate_reward_is_sparse_zero():
    env = ChessEnv()
    env.reset()
    mask = env.action_mask()
    agent = RandomAgent(seed=1)
    action = agent.select_action(mask)
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward == 0.0
    assert not terminated


def test_full_random_vs_random_game_terminates_cleanly():
    env = ChessEnv(player_color=chess.WHITE)
    white = RandomAgent(seed=7)
    black = RandomAgent(seed=8)
    obs, info = env.reset()

    terminated = truncated = False
    plies = 0
    while not (terminated or truncated) and plies < 1000:
        agent = white if env.current_player() == chess.WHITE else black
        action = agent.select_action(env.action_mask())
        obs, reward, terminated, truncated, info = env.step(action)
        plies += 1

    assert terminated
    assert reward in (-1.0, 0.0, 1.0)
    assert plies < 1000  # sanity: didn't hit the safety cap


def test_player_color_affects_reward_sign_not_gameplay():
    """Same deterministic game, only player_color differs -> reward sign flips."""
    sans = ["f3", "e5", "g4", "Qh4#"]

    env_white = ChessEnv(player_color=chess.WHITE)
    env_white.reset()
    for san in sans:
        move = env_white.board.parse_san(san)
        obs, reward_white, terminated, truncated, info = env_white.step(
            env_white.encode_move(move)
        )

    env_black = ChessEnv(player_color=chess.BLACK)
    env_black.reset()
    for san in sans:
        move = env_black.board.parse_san(san)
        obs, reward_black, terminated, truncated, info = env_black.step(
            env_black.encode_move(move)
        )

    assert reward_white == -reward_black == -1.0
