"""Stage 2 tests: MCTS respects its simulation budget, backs up values
with correct alternating sign, and — the key correctness check — finds
the right move in known tactical positions (mate-in-1, for both sides
to move, and a single-legal-move forced position)."""

import chess
import numpy as np
import pytest
import torch

from engine.encoding import decode_move
from training.device import resolve_device
from training.mcts import MCTS, select_action, terminal_value, visit_count_policy
from training.network import PolicyValueNet

WHITE_MATE_IN_1_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
WHITE_MATE_IN_1_MOVE_UCI = "a1a8"

BLACK_MATE_IN_1_FEN = "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"
BLACK_MATE_IN_1_MOVE_UCI = "a8a1"


def _make_mcts(device=None) -> MCTS:
    torch.manual_seed(0)
    network = PolicyValueNet()
    network.eval()
    device = device or torch.device("cpu")
    return MCTS(network.to(device), device=device, c_puct=1.5)


def test_run_respects_configurable_simulation_count():
    mcts = _make_mcts()
    board = chess.Board()
    for num_sims in (5, 25, 60):
        root = mcts.run(board, num_simulations=num_sims)
        assert root.visit_count == num_sims
        assert sum(c.visit_count for c in root.children.values()) == num_sims


def test_root_expands_all_legal_moves_with_valid_priors():
    mcts = _make_mcts()
    board = chess.Board()
    root = mcts.run(board, num_simulations=10)

    assert len(root.children) == len(list(board.legal_moves))
    priors = np.array([c.prior for c in root.children.values()])
    assert np.all(priors >= 0.0)
    assert np.isclose(priors.sum(), 1.0, atol=1e-4)


def test_terminal_value_convention():
    checkmated_board = chess.Board(WHITE_MATE_IN_1_FEN)
    checkmated_board.push_uci(WHITE_MATE_IN_1_MOVE_UCI)
    assert checkmated_board.is_checkmate()
    # It's black's move on a board where black has just been checkmated.
    assert terminal_value(checkmated_board, claim_draw=True) == -1.0

    stalemate_board = chess.Board("7k/8/6Q1/6K1/8/8/8/8 b - - 0 1")
    assert terminal_value(stalemate_board, claim_draw=True) == 0.0


@pytest.mark.parametrize(
    "fen,mate_uci",
    [
        (WHITE_MATE_IN_1_FEN, WHITE_MATE_IN_1_MOVE_UCI),
        (BLACK_MATE_IN_1_FEN, BLACK_MATE_IN_1_MOVE_UCI),
    ],
)
def test_mcts_finds_mate_in_one(fen, mate_uci):
    mcts = _make_mcts()
    board = chess.Board(fen)

    root = mcts.run(board, num_simulations=200)
    chosen_action = select_action(root, temperature=0.0)
    chosen_move = decode_move(chosen_action, board)

    assert chosen_move == chess.Move.from_uci(mate_uci), (
        f"expected mating move {mate_uci}, MCTS chose {chosen_move.uci()} "
        f"(visit counts: { {decode_move(a, board).uci(): c.visit_count for a, c in root.children.items()} })"
    )


def test_mcts_handles_single_legal_move_forced_position():
    # White king boxed into a corner with exactly one legal move.
    board = chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1")
    legal = list(board.legal_moves)
    mcts = _make_mcts()
    root = mcts.run(board, num_simulations=15)

    assert len(root.children) == len(legal)
    action = select_action(root, temperature=0.0)
    assert decode_move(action, board) in legal


def test_visit_count_policy_matches_visits_and_sums_to_one():
    mcts = _make_mcts()
    board = chess.Board()
    root = mcts.run(board, num_simulations=40)

    policy = visit_count_policy(root, temperature=1.0)
    assert policy.shape == (4672,)
    assert np.isclose(policy.sum(), 1.0, atol=1e-4)

    for action, child in root.children.items():
        expected = child.visit_count / 40
        assert np.isclose(policy[action], expected, atol=1e-4)

    # Actions never expanded at the root carry zero probability.
    non_root_action = next(a for a in range(4672) if a not in root.children)
    assert policy[non_root_action] == 0.0


def test_visit_count_policy_temperature_zero_is_one_hot():
    mcts = _make_mcts()
    board = chess.Board(WHITE_MATE_IN_1_FEN)
    root = mcts.run(board, num_simulations=100)

    policy = visit_count_policy(root, temperature=0.0)
    assert np.isclose(policy.sum(), 1.0)
    assert (policy == 1.0).sum() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")
def test_mcts_runs_end_to_end_on_cuda():
    device = resolve_device(require_cuda=True)
    mcts = _make_mcts(device=device)
    board = chess.Board()
    root = mcts.run(board, num_simulations=10)
    assert root.visit_count == 10
    assert next(mcts.network.parameters()).device.type == "cuda"
