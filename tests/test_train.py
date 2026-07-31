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
        eval_vs_previous_games=4,  # wiring test, not an eval-thoroughness test
    )

    history = run_training_loop(
        network, device, optimizer, buffer, self_play_config, train_config,
        num_iterations=2, games_per_iteration=2, rng=np.random.default_rng(0),
    )

    assert len(history) == 2
    for record in history:
        assert record["policy_loss"] is not None
        assert record["total_loss"] is not None
        # min_buffer_size=1 -> training runs every iteration -> vs-previous
        # eval should run every iteration too (new weights each time).
        assert record["eval_vs_previous_new_win_rate"] is not None
        assert record["eval_vs_previous_games"] == train_config.eval_vs_previous_games

    checkpoints = list((tmp_path / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 2

    log_path = tmp_path / "logs" / "train_log.csv"
    log_lines = log_path.read_text().strip().splitlines()
    assert len(log_lines) == 3  # header + 2 iterations
    header = log_lines[0].split(",")
    for expected_field in (
        "eval_vs_previous_new_win_rate", "eval_vs_previous_draw_rate",
        "eval_vs_random_win_rate", "eval_vs_random_games",
    ):
        assert expected_field in header


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
    # No training happened -> "new" and "old" would be identical -> skip
    # the vs-previous eval rather than spending time on a meaningless comparison.
    assert history[0]["eval_vs_previous_new_win_rate"] is None


def test_eval_vs_random_only_runs_on_cadence_iterations(tmp_path):
    torch.manual_seed(0)
    network = PolicyValueNet()
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=500)

    self_play_config = SelfPlayConfig(num_simulations=5, temperature=0.0, temperature_threshold_plies=0)
    train_config = TrainConfig(
        batch_size=4,
        train_steps_per_iteration=2,
        min_buffer_size=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_path=str(tmp_path / "logs" / "train_log.csv"),
        eval_vs_previous_games=2,
        eval_vs_random_games=2,
        eval_vs_random_every_iterations=2,
    )

    history = run_training_loop(
        network, device, optimizer, buffer, self_play_config, train_config,
        num_iterations=2, games_per_iteration=2, rng=np.random.default_rng(0),
    )

    assert history[0]["eval_vs_random_win_rate"] is None  # iteration 0: not a multiple of 2
    assert history[1]["eval_vs_random_win_rate"] is not None  # iteration 1 -> (1+1)=2, matches cadence
    assert history[1]["eval_vs_random_games"] == 2


def test_run_training_loop_uses_batched_self_play(tmp_path, monkeypatch):
    """Regression guard: run_training_loop must call the batched self-
    play path, not the serial one, per the throughput fix being wired
    in as the default."""
    import training.train as train_module

    calls = []
    original = train_module.run_batched_self_play

    def spy(*args, **kwargs):
        calls.append(kwargs.get("leaves_per_tree_per_round"))
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "run_batched_self_play", spy)

    network = PolicyValueNet()
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=200)
    self_play_config = SelfPlayConfig(num_simulations=5, temperature=0.0, temperature_threshold_plies=0)
    train_config = TrainConfig(
        batch_size=4, train_steps_per_iteration=2, min_buffer_size=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_path=str(tmp_path / "logs" / "train_log.csv"),
        leaves_per_tree_per_round=3,
    )

    run_training_loop(
        network, device, optimizer, buffer, self_play_config, train_config,
        num_iterations=1, games_per_iteration=2, rng=np.random.default_rng(0),
    )

    assert len(calls) == 1
    assert calls[0] == 3  # leaves_per_tree_per_round threaded through correctly


def test_run_training_loop_uses_batched_evaluation(tmp_path, monkeypatch):
    """Regression guard: run_training_loop must call the batched
    evaluation path (training/batched_evaluation.py), not the serial
    one (training/evaluation.py), for both vs-previous and vs-random —
    per the throughput fix being wired in as the default."""
    import training.train as train_module

    checkpoint_calls = []
    random_calls = []
    original_checkpoints = train_module.run_batched_evaluate_checkpoints
    original_random = train_module.run_batched_evaluate_against_random

    def spy_checkpoints(*args, **kwargs):
        checkpoint_calls.append(kwargs.get("leaves_per_tree_per_round"))
        return original_checkpoints(*args, **kwargs)

    def spy_random(*args, **kwargs):
        random_calls.append(kwargs.get("leaves_per_tree_per_round"))
        return original_random(*args, **kwargs)

    monkeypatch.setattr(train_module, "run_batched_evaluate_checkpoints", spy_checkpoints)
    monkeypatch.setattr(train_module, "run_batched_evaluate_against_random", spy_random)

    network = PolicyValueNet()
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=200)
    self_play_config = SelfPlayConfig(num_simulations=5, temperature=0.0, temperature_threshold_plies=0)
    train_config = TrainConfig(
        batch_size=4, train_steps_per_iteration=2, min_buffer_size=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_path=str(tmp_path / "logs" / "train_log.csv"),
        leaves_per_tree_per_round=3,
        eval_vs_previous_games=2, eval_vs_random_games=2, eval_vs_random_every_iterations=1,
    )

    run_training_loop(
        network, device, optimizer, buffer, self_play_config, train_config,
        num_iterations=1, games_per_iteration=2, rng=np.random.default_rng(0),
    )

    assert len(checkpoint_calls) == 1 and checkpoint_calls[0] == 3
    assert len(random_calls) == 1 and random_calls[0] == 3


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
