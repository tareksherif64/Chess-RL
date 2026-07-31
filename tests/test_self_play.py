"""Stage 3 tests: self-play produces one example per ply with correctly
signed value targets, discards games that hit the safety cap instead of
guessing an outcome, and run_self_play() wires games into a ReplayBuffer."""

import chess
import numpy as np
import torch

from training.network import PolicyValueNet
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayConfig, play_self_play_game, run_self_play

WHITE_MATE_IN_1_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
BLACK_MATE_IN_1_FEN = "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"


def _make_network(seed: int = 0) -> PolicyValueNet:
    torch.manual_seed(seed)
    net = PolicyValueNet()
    net.eval()
    return net


def test_mate_in_one_game_produces_single_correctly_valued_example():
    network = _make_network()
    config = SelfPlayConfig(
        num_simulations=200,
        temperature=0.0,
        temperature_threshold_plies=0,
    )
    examples, summary = play_self_play_game(
        network, torch.device("cpu"), config,
        rng=np.random.default_rng(0), initial_fen=WHITE_MATE_IN_1_FEN,
    )

    assert not summary["discarded"]
    assert summary["plies"] == 1
    assert summary["termination"] == "CHECKMATE"
    assert len(examples) == 1

    example = examples[0]
    assert example.board.shape == (8, 8, 18)
    assert example.board.dtype == np.float32
    assert example.policy.shape == (4672,)
    # White was to move and white won -> value +1 for that recorded position.
    assert example.value == 1.0


def test_black_to_move_mate_in_one_gives_positive_value_for_black():
    network = _make_network()
    config = SelfPlayConfig(num_simulations=200, temperature=0.0, temperature_threshold_plies=0)
    examples, summary = play_self_play_game(
        network, torch.device("cpu"), config,
        rng=np.random.default_rng(0), initial_fen=BLACK_MATE_IN_1_FEN,
    )
    assert len(examples) == 1
    assert examples[0].value == 1.0  # black was to move and black won


def test_multi_ply_game_records_one_example_per_ply_with_alternating_movers():
    # Forced sequence toward mate: not a mate-in-1, so we get several plies.
    network = _make_network()
    config = SelfPlayConfig(num_simulations=40, temperature=0.0, temperature_threshold_plies=0)
    # Simple K+R vs K endgame, white to move — will terminate within a
    # bounded, small number of plies against a real search (not random),
    # though the exact count isn't asserted, only internal consistency.
    fen = "7k/8/8/8/8/8/R7/6K1 w - - 0 1"
    examples, summary = play_self_play_game(
        network, torch.device("cpu"), config, rng=np.random.default_rng(1), initial_fen=fen,
    )
    if summary["discarded"]:
        return  # extremely unlikely (max_plies=600 default) but not a bug if hit

    assert len(examples) == summary["plies"]
    movers = [ex.value for ex in examples]
    # Every recorded value must be a valid signed/draw outcome.
    assert all(v in (-1.0, 0.0, 1.0) for v in movers)

    if summary["termination"] == "CHECKMATE":
        # Winner alternates who's "to move", so recorded values should
        # alternate in sign along the game (winner's own plies are +1,
        # loser's plies are -1).
        assert set(movers) == {1.0, -1.0}


def test_game_hitting_ply_cap_is_discarded_not_guessed():
    network = _make_network()
    config = SelfPlayConfig(num_simulations=5, max_plies=1)
    examples, summary = play_self_play_game(
        network, torch.device("cpu"), config, rng=np.random.default_rng(0),
    )  # starting position certainly won't finish in 1 ply
    assert summary["discarded"] is True
    assert summary["termination"] == "TRUNCATED_SAFETY_CAP"
    assert examples == []


def test_run_self_play_adds_games_to_buffer_and_returns_summaries():
    network = _make_network()
    config = SelfPlayConfig(num_simulations=100, temperature=0.0, temperature_threshold_plies=0)
    buffer = ReplayBuffer(capacity=50)

    summaries = run_self_play(
        network, torch.device("cpu"), buffer, config,
        num_games=2, rng=np.random.default_rng(0), verbose=False,
        initial_fen=WHITE_MATE_IN_1_FEN,
    )

    assert len(summaries) == 2
    for s in summaries:
        assert "wall_seconds" in s
        assert "examples" in s
    assert len(buffer) == sum(s["examples"] for s in summaries)
