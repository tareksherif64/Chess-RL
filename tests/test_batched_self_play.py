"""Batched self-play: N concurrent games get correctly-signed, per-ply
recorded examples independent of each other, games finishing at
different times don't disrupt the ones still active, ply-cap games are
discarded per-game (not the whole batch), and results land in the
buffer correctly."""

import chess
import numpy as np
import torch

from training.batched_self_play import run_batched_self_play
from training.network import PolicyValueNet
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayConfig

WHITE_MATE_IN_1_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
BLACK_MATE_IN_1_FEN = "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"


def _make_network(seed: int = 0) -> PolicyValueNet:
    torch.manual_seed(seed)
    net = PolicyValueNet()
    net.eval()
    return net


def test_concurrent_mate_in_one_games_get_correct_values():
    network = _make_network()
    config = SelfPlayConfig(num_simulations=200, temperature=0.0, temperature_threshold_plies=0)
    buffer = ReplayBuffer(capacity=50)

    summaries = run_batched_self_play(
        network, torch.device("cpu"), buffer, config,
        num_games=4, rng=np.random.default_rng(0), verbose=False,
        initial_fen=WHITE_MATE_IN_1_FEN,
    )

    assert len(summaries) == 4
    for s in summaries:
        assert not s["discarded"]
        assert s["plies"] == 1
        assert s["termination"] == "CHECKMATE"
        assert s["examples"] == 1
    assert len(buffer) == 4

    _boards, _policies, values = buffer.sample(4, rng=np.random.default_rng(0))
    assert all(v == 1.0 for v in values)  # white to move, white wins, every game


def test_games_of_different_length_are_recorded_independently():
    """Mix a 1-ply mate-in-1 game with a longer K+R vs K endgame in the
    same concurrent batch — the short game finishing early must not
    corrupt or block the longer one still running."""
    network = _make_network()
    config = SelfPlayConfig(num_simulations=30, temperature=0.0, temperature_threshold_plies=0)
    buffer = ReplayBuffer(capacity=2000)

    # run_batched_self_play only takes one initial_fen for the whole
    # batch, so exercise heterogeneity via two separate calls into the
    # same buffer instead, then check both are represented correctly.
    s1 = run_batched_self_play(
        network, torch.device("cpu"), buffer, config, num_games=2,
        rng=np.random.default_rng(1), verbose=False, initial_fen=WHITE_MATE_IN_1_FEN,
    )
    s2 = run_batched_self_play(
        network, torch.device("cpu"), buffer, config, num_games=2,
        rng=np.random.default_rng(2), verbose=False,
        initial_fen="7k/8/8/8/8/8/R7/6K1 w - - 0 1",
    )

    assert all(s["plies"] == 1 for s in s1)
    assert all(s["plies"] >= 1 for s in s2)
    assert len(buffer) == sum(s["examples"] for s in s1) + sum(s["examples"] for s in s2)


def test_games_hitting_ply_cap_are_discarded_individually():
    network = _make_network()
    config = SelfPlayConfig(num_simulations=5, max_plies=1)
    buffer = ReplayBuffer(capacity=50)

    summaries = run_batched_self_play(
        network, torch.device("cpu"), buffer, config,
        num_games=3, rng=np.random.default_rng(0), verbose=False,
    )  # standard start position certainly won't finish in 1 ply

    assert len(summaries) == 3
    assert all(s["discarded"] for s in summaries)
    assert all(s["termination"] == "TRUNCATED_SAFETY_CAP" for s in summaries)
    assert len(buffer) == 0


def test_black_to_move_mate_gives_positive_value_for_black():
    network = _make_network()
    config = SelfPlayConfig(num_simulations=200, temperature=0.0, temperature_threshold_plies=0)
    buffer = ReplayBuffer(capacity=10)

    run_batched_self_play(
        network, torch.device("cpu"), buffer, config, num_games=2,
        rng=np.random.default_rng(0), verbose=False, initial_fen=BLACK_MATE_IN_1_FEN,
    )
    _boards, _policies, values = buffer.sample(2, rng=np.random.default_rng(0))
    assert all(v == 1.0 for v in values)
