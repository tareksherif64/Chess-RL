"""Batched evaluation correctness: the core check is that at minimal
concurrency (1 game, leaves_per_tree_per_round=1 — no two leaves ever
in flight in the same tree, so no virtual-loss approximation), batched
evaluation plays the *exact same game*, move for move, as serial
evaluation with the same two networks. Also covers independence across
concurrent games with different network assignments, vs-random, and
CUDA usage."""

import chess
import numpy as np
import pytest
import torch

from agents.random_agent import RandomAgent
from engine.encoding import decode_move
from training.batched_evaluation import (
    RANDOM,
    _GameSpec,
    _play_concurrent_games,
    run_batched_evaluate_against_random,
    run_batched_evaluate_checkpoints,
)
from training.device import resolve_device
from training.mcts import MCTS, select_action
from training.network import PolicyValueNet

WHITE_MATE_IN_1_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
BLACK_MATE_IN_1_FEN = "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"


def _make_network(seed: int) -> PolicyValueNet:
    torch.manual_seed(seed)
    net = PolicyValueNet()
    net.eval()
    return net


def test_batched_eval_matches_serial_move_sequence_at_minimal_concurrency():
    """With one game and leaves_per_tree_per_round=1, the batched path's
    per-ply "which network moves now" + BatchedMCTS search must produce
    literally the same move at every ply as manually driving serial
    MCTS with the same two networks — not just the same final result."""
    new_net = _make_network(seed=0)
    old_net = _make_network(seed=1)
    device = torch.device("cpu")
    sims = 20
    c_puct = 1.5

    # Serial reference: new plays white, old plays black.
    ref_board = chess.Board()
    ref_moves = []
    ply = 0
    while not ref_board.is_game_over(claim_draw=True) and ply < 12:
        network = new_net if ref_board.turn == chess.WHITE else old_net
        root = MCTS(network, device=device, c_puct=c_puct, claim_draw=True).run(ref_board, sims)
        action = select_action(root, temperature=0.0)
        move = decode_move(action, ref_board)
        ref_board.push(move)
        ref_moves.append(move)
        ply += 1

    specs = [_GameSpec(white_kind="new", black_kind="old")]
    _results, envs = _play_concurrent_games(
        specs, {"new": new_net, "old": old_net}, device, sims,
        c_puct=c_puct, claim_draw=True, max_plies=12, leaves_per_tree_per_round=1,
        random_agent=None, initial_fen=None,
    )

    assert list(envs[0].board.move_stack) == ref_moves
    assert envs[0].board.fen() == ref_board.fen()


def test_concurrent_checkpoint_eval_independent_of_each_other():
    """Different network pairings/outcomes across concurrent games in
    the same call must not leak into each other."""
    new_net = _make_network(seed=0)
    old_net = _make_network(seed=1)
    device = torch.device("cpu")

    result = run_batched_evaluate_checkpoints(
        new_net, old_net, device, num_games=4, num_simulations=30,
        leaves_per_tree_per_round=4, initial_fen=WHITE_MATE_IN_1_FEN,
    )
    # Whoever is to move in this position (white) wins in one ply,
    # regardless of which checkpoint that happens to be this game.
    assert result["games"] == 4
    assert result["new_wins"] + result["old_wins"] == 4
    assert result["draws"] == 0


def test_identical_networks_give_symmetric_batched_results():
    network = _make_network(seed=0)
    device = torch.device("cpu")

    result = run_batched_evaluate_checkpoints(
        network, network, device, num_games=4, num_simulations=20,
        leaves_per_tree_per_round=4, initial_fen=WHITE_MATE_IN_1_FEN,
    )
    assert result["new_wins"] == result["old_wins"]
    assert result["new_win_rate"] == result["old_win_rate"] == 0.5


def test_batched_evaluate_against_random_returns_well_formed_result():
    network = _make_network(seed=0)
    device = torch.device("cpu")

    result = run_batched_evaluate_against_random(
        network, device, num_games=4, num_simulations=20,
        rng=np.random.default_rng(0), leaves_per_tree_per_round=4,
    )
    assert result["games"] == 4
    assert result["network_wins"] + result["random_wins"] + result["draws"] == 4
    assert 0.0 <= result["network_win_rate"] <= 1.0


@pytest.mark.parametrize(
    "fen,network_to_move",
    [(WHITE_MATE_IN_1_FEN, chess.WHITE), (BLACK_MATE_IN_1_FEN, chess.BLACK)],
)
def test_batched_evaluate_against_random_finds_forced_mate(fen, network_to_move):
    """Uses _play_concurrent_games directly so the network is placed on
    whichever color is actually to move in the FEN (the public
    evaluate_against_random's alternating-color convention would put
    the *random* agent on move for one of these two FENs, testing
    nothing about the network's search — see the color-mismatch this
    caught during development)."""
    network = _make_network(seed=0)
    device = torch.device("cpu")
    random_agent = RandomAgent(seed=0)

    spec = _GameSpec(
        white_kind="network" if network_to_move == chess.WHITE else RANDOM,
        black_kind=RANDOM if network_to_move == chess.WHITE else "network",
    )
    results, _envs = _play_concurrent_games(
        [spec], {"network": network}, device, num_simulations=200,
        c_puct=1.5, claim_draw=True, max_plies=5, leaves_per_tree_per_round=2,
        random_agent=random_agent, initial_fen=fen,
    )
    expected = "white" if network_to_move == chess.WHITE else "black"
    assert results[0] == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")
def test_batched_evaluation_runs_end_to_end_on_cuda():
    device = resolve_device(require_cuda=True)
    new_net = _make_network(seed=0).to(device)
    old_net = _make_network(seed=1).to(device)

    result = run_batched_evaluate_checkpoints(
        new_net, old_net, device, num_games=4, num_simulations=10,
        leaves_per_tree_per_round=4,
    )
    assert result["games"] == 4
    assert next(new_net.parameters()).device.type == "cuda"
    assert next(old_net.parameters()).device.type == "cuda"
