"""BatchedMCTS correctness: the virtual-loss apply/revert invariant in
isolation, bit-for-bit equivalence to serial MCTS in the reducible
(single tree, one leaf per round) case, independence across concurrent
trees, mate-in-1 correctness via the batched path, a low-branching
dedup safeguard, and CUDA usage."""

import chess
import numpy as np
import pytest
import torch

from engine.encoding import decode_move
from training.batched_mcts import (
    BatchedMCTS,
    apply_virtual_loss,
    revert_virtual_loss,
)
from training.device import resolve_device
from training.mcts import MCTS, MCTSNode, backup
from training.network import PolicyValueNet

WHITE_MATE_IN_1_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
WHITE_MATE_IN_1_MOVE_UCI = "a1a8"
BLACK_MATE_IN_1_FEN = "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"
BLACK_MATE_IN_1_MOVE_UCI = "a8a1"


def _make_network(seed: int = 0) -> PolicyValueNet:
    torch.manual_seed(seed)
    net = PolicyValueNet()
    net.eval()
    return net


def test_virtual_loss_revert_matches_serial_backup():
    """apply_virtual_loss then revert_virtual_loss then a real backup
    must leave every node's (visit_count, value_sum) identical to what
    a single plain backup(real_value) alone would have produced."""
    root = MCTSNode(parent=None, prior=1.0)
    child = MCTSNode(parent=root, prior=0.5)
    grandchild = MCTSNode(parent=child, prior=0.5)
    path = [root, child, grandchild]

    real_value = 0.37

    # Reference: plain serial backup on a fresh tree.
    ref_root = MCTSNode(parent=None, prior=1.0)
    ref_child = MCTSNode(parent=ref_root, prior=0.5)
    ref_grandchild = MCTSNode(parent=ref_child, prior=0.5)
    backup([ref_root, ref_child, ref_grandchild], real_value)

    # Virtual loss then revert then real backup, on the other tree.
    apply_virtual_loss(path)
    revert_virtual_loss(path)
    backup(path, real_value)

    for a, b in zip(path, [ref_root, ref_child, ref_grandchild]):
        assert a.visit_count == b.visit_count
        assert a.value_sum == pytest.approx(b.value_sum, abs=1e-12)


def test_virtual_loss_makes_leaf_less_attractive_to_immediate_parent():
    """The whole point of virtual loss: after applying it, the parent's
    PUCT selection should be less inclined to re-pick the same child
    (all else equal)."""
    from training.mcts import select_child

    root = MCTSNode(parent=None, prior=1.0)
    child_a = MCTSNode(parent=root, prior=0.5)
    child_b = MCTSNode(parent=root, prior=0.5)
    root.children = {1: child_a, 2: child_b}
    # Give both children one real visit each with equal value, so they
    # start tied.
    for c in (child_a, child_b):
        c.visit_count = 1
        c.value_sum = 0.0
    root.visit_count = 2

    action, chosen = select_child(root, c_puct=1.5)
    assert chosen in (child_a, child_b)  # tied, either is valid

    apply_virtual_loss([root, chosen])
    action2, chosen2 = select_child(root, c_puct=1.5)
    assert chosen2 is not chosen


def test_batched_run_reduces_to_serial_mcts_for_single_tree_one_leaf_per_round():
    """With N=1 game and leaves_per_tree_per_round=1, batching never
    actually overlaps two in-flight leaves — each round selects,
    evaluates (batch of 1), and backs up before the next round, which
    is exactly what serial MCTS.run() does. Final tree stats should
    match bit-for-bit."""
    board = chess.Board()
    num_sims = 30

    net_a = _make_network(seed=0)
    serial_root = MCTS(net_a, device=torch.device("cpu"), c_puct=1.5).run(board, num_sims)

    net_b = _make_network(seed=0)
    batched_roots = BatchedMCTS(
        net_b, device=torch.device("cpu"), c_puct=1.5, leaves_per_tree_per_round=1
    ).run_batch([board], num_sims)
    batched_root = batched_roots[0]

    assert serial_root.visit_count == batched_root.visit_count == num_sims
    assert serial_root.children.keys() == batched_root.children.keys()
    for action in serial_root.children:
        s, b = serial_root.children[action], batched_root.children[action]
        assert s.visit_count == b.visit_count, action
        assert s.value_sum == pytest.approx(b.value_sum, abs=1e-6), action
        assert s.prior == pytest.approx(b.prior, abs=1e-6), action


def test_run_batch_respects_simulation_count_per_tree():
    network = _make_network()
    mcts = BatchedMCTS(network, device=torch.device("cpu"), leaves_per_tree_per_round=4)
    boards = [chess.Board(), chess.Board(WHITE_MATE_IN_1_FEN)]

    roots = mcts.run_batch(boards, num_simulations=25)
    assert len(roots) == 2
    for root in roots:
        assert root.visit_count == 25
        assert sum(c.visit_count for c in root.children.values()) == 25


def test_concurrent_trees_are_independent():
    """Different starting positions in the same run_batch call must
    not leak state into each other."""
    network = _make_network()
    mcts = BatchedMCTS(network, device=torch.device("cpu"), leaves_per_tree_per_round=4)
    board_start = chess.Board()
    board_mate = chess.Board(WHITE_MATE_IN_1_FEN)

    roots = mcts.run_batch([board_start, board_mate], num_simulations=150)
    start_root, mate_root = roots

    assert len(start_root.children) == len(list(board_start.legal_moves))
    assert len(mate_root.children) == len(list(board_mate.legal_moves))

    from training.mcts import select_action

    mate_action = select_action(mate_root, temperature=0.0)
    assert decode_move(mate_action, board_mate) == chess.Move.from_uci(WHITE_MATE_IN_1_MOVE_UCI)


@pytest.mark.parametrize(
    "fen,mate_uci",
    [
        (WHITE_MATE_IN_1_FEN, WHITE_MATE_IN_1_MOVE_UCI),
        (BLACK_MATE_IN_1_FEN, BLACK_MATE_IN_1_MOVE_UCI),
    ],
)
def test_batched_mcts_finds_mate_in_one(fen, mate_uci):
    from training.mcts import select_action

    network = _make_network()
    mcts = BatchedMCTS(network, device=torch.device("cpu"), leaves_per_tree_per_round=4)
    board = chess.Board(fen)

    roots = mcts.run_batch([board], num_simulations=200)
    action = select_action(roots[0], temperature=0.0)
    chosen = decode_move(action, board)

    assert chosen == chess.Move.from_uci(mate_uci)


def test_low_branching_position_does_not_crash_or_miscount():
    # White king has exactly one legal move — forces the dedup
    # safeguard to kick in when leaves_per_tree_per_round > 1.
    board = chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1")
    network = _make_network()
    mcts = BatchedMCTS(network, device=torch.device("cpu"), leaves_per_tree_per_round=8)

    roots = mcts.run_batch([board], num_simulations=20)
    root = roots[0]
    assert root.visit_count == 20
    assert sum(c.visit_count for c in root.children.values()) == 20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")
def test_batched_mcts_runs_end_to_end_on_cuda():
    device = resolve_device(require_cuda=True)
    network = _make_network().to(device)
    mcts = BatchedMCTS(network, device=device, leaves_per_tree_per_round=4)
    boards = [chess.Board() for _ in range(4)]

    roots = mcts.run_batch(boards, num_simulations=10)
    assert all(r.visit_count == 10 for r in roots)
    assert next(mcts.network.parameters()).device.type == "cuda"
