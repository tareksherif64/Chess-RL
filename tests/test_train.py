"""Stage 4 tests: the core correctness check is whether train_steps can
overfit a tiny fixed batch (the standard way to catch a broken loss/
backward/optimizer wiring — if the network can't drive loss down on a
handful of repeated examples, nothing about real data will work
either). Also covers the full orchestration loop end-to-end at a tiny
scale, and that training doesn't silently run on CPU."""

import chess
import numpy as np
import pytest
import torch

from engine.encoding import ACTION_SPACE_SIZE, encode_board
from training.device import resolve_device
from training.network import PolicyValueNet
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayConfig
from training.train import TrainConfig, compute_loss, run_training_loop, train_steps


def _fixed_buffer(n: int = 8) -> ReplayBuffer:
    """A handful of real (non-degenerate) board encodings with
    arbitrary-but-valid policy/value targets, repeated across training
    steps — enough to check the network can fit *something* concrete."""
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity=n)
    board = chess.Board()
    for i in range(n):
        if board.is_game_over():
            board = chess.Board()
        legal = list(board.legal_moves)
        if legal:
            board.push(legal[i % len(legal)])

        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        policy[i % ACTION_SPACE_SIZE] = 1.0  # arbitrary one-hot target
        value = 1.0 if i % 2 == 0 else -1.0
        buffer.add(encode_board(board), policy, value)
    return buffer


def test_train_steps_can_overfit_a_tiny_fixed_batch():
    torch.manual_seed(0)
    network = PolicyValueNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    buffer = _fixed_buffer(n=8)
    device = torch.device("cpu")
    rng = np.random.default_rng(0)

    early = train_steps(network, optimizer, buffer, device, num_steps=5, batch_size=8, rng=rng)
    late = train_steps(network, optimizer, buffer, device, num_steps=500, batch_size=8, rng=rng)

    # A tiny 8-example, massively overparameterized (~493K param) problem
    # should be fit down substantially; a loose relative bound (rather
    # than a tight absolute one) avoids flakiness from Adam's normal
    # step-to-step noise on such a small/sharp loss surface — verified
    # empirically this run reliably lands around a ~10x reduction.
    assert late["total_loss"] < early["total_loss"] * 0.3
    assert late["policy_loss"] < early["policy_loss"] * 0.3
    assert late["total_loss"] < 2.0  # sanity floor: not diverged/NaN


def test_train_steps_leaves_network_in_eval_mode():
    network = PolicyValueNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    buffer = _fixed_buffer(n=4)
    train_steps(network, optimizer, buffer, torch.device("cpu"), num_steps=2, batch_size=4)
    assert not network.training


def test_compute_loss_shapes_and_finiteness():
    network = PolicyValueNet()
    boards = torch.randn(3, 18, 8, 8)
    policies = torch.zeros(3, ACTION_SPACE_SIZE)
    policies[:, 0] = 1.0
    values = torch.tensor([1.0, -1.0, 0.0])

    policy_loss, value_loss, total_loss = compute_loss(network, boards, policies, values)
    assert policy_loss.dim() == 0
    assert value_loss.dim() == 0
    assert torch.isfinite(total_loss)
    assert torch.isclose(total_loss, policy_loss + value_loss)


def test_run_training_loop_end_to_end_tiny_scale(tmp_path):
    torch.manual_seed(0)
    network = PolicyValueNet()
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=500)

    self_play_config = SelfPlayConfig(num_simulations=5, temperature=0.0, temperature_threshold_plies=0)
    train_config = TrainConfig(
        batch_size=4,
        train_steps_per_iteration=3,
        min_buffer_size=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_every_iterations=1,
        log_path=str(tmp_path / "logs" / "train_log.csv"),
    )

    history = run_training_loop(
        network, device, optimizer, buffer, self_play_config, train_config,
        num_iterations=2, games_per_iteration=1, rng=np.random.default_rng(0),
    )

    assert len(history) == 2
    for record in history:
        assert record["policy_loss"] is not None
        assert record["total_loss"] is not None

    checkpoints = list((tmp_path / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 2

    log_lines = (tmp_path / "logs" / "train_log.csv").read_text().strip().splitlines()
    assert len(log_lines) == 3  # header + 2 iterations


def test_run_training_loop_skips_training_below_min_buffer_size(tmp_path):
    network = PolicyValueNet()
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=500)

    self_play_config = SelfPlayConfig(num_simulations=5, temperature=0.0, temperature_threshold_plies=0)
    train_config = TrainConfig(
        batch_size=4,
        train_steps_per_iteration=3,
        min_buffer_size=10_000,  # unreachable in one tiny iteration
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_path=str(tmp_path / "logs" / "train_log.csv"),
    )

    history = run_training_loop(
        network, device, optimizer, buffer, self_play_config, train_config,
        num_iterations=1, games_per_iteration=1, rng=np.random.default_rng(0),
    )
    assert history[0]["policy_loss"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")
def test_train_steps_actually_uses_cuda():
    device = resolve_device(require_cuda=True)
    network = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    buffer = _fixed_buffer(n=4)

    train_steps(network, optimizer, buffer, device, num_steps=2, batch_size=4)
    assert next(network.parameters()).device.type == "cuda"
    for state in optimizer.state.values():
        if "exp_avg" in state:
            assert state["exp_avg"].device.type == "cuda"
