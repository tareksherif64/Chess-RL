"""Tests for the overnight-run hardening added to run_training_loop:
multi-batch self-play reaching a target game count at a fixed tested
concurrency, and crash isolation (a failing self-play micro-batch,
training pass, evaluation call, or checkpoint save is logged and
skipped rather than taking the whole run down), plus the new
GPU-memory / total-iteration-time logging fields.
"""

import numpy as np
import pytest
import torch

from training.network import PolicyValueNet
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayConfig
from training.train import TrainConfig, run_training_loop


def _base_config(tmp_path, **overrides) -> TrainConfig:
    defaults = dict(
        batch_size=4,
        train_steps_per_iteration=2,
        min_buffer_size=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_path=str(tmp_path / "logs" / "train_log.csv"),
        error_log_path=str(tmp_path / "logs" / "errors.log"),
        eval_vs_previous_games=2,
        eval_vs_random_games=2,
        eval_vs_random_every_iterations=1,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _setup(tmp_path, **train_config_overrides):
    network = PolicyValueNet()
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=2000)
    self_play_config = SelfPlayConfig(num_simulations=5, temperature=0.0, temperature_threshold_plies=0)
    train_config = _base_config(tmp_path, **train_config_overrides)
    return network, device, optimizer, buffer, self_play_config, train_config


def test_multi_batch_self_play_reaches_target_game_count(tmp_path):
    """games_per_iteration=10 with self_play_batch_size=4 should run
    three micro-batches (4+4+2), not one batch of 10 (an untested
    concurrency level for the real run)."""
    import training.train as train_module

    calls = []
    original = train_module.run_batched_self_play

    def spy(*args, **kwargs):
        calls.append(kwargs["num_games"])
        return original(*args, **kwargs)

    network, device, optimizer, buffer, self_play_config, train_config = _setup(
        tmp_path, self_play_batch_size=4,
    )

    import unittest.mock
    with unittest.mock.patch.object(train_module, "run_batched_self_play", spy):
        history = run_training_loop(
            network, device, optimizer, buffer, self_play_config, train_config,
            num_iterations=1, games_per_iteration=10, rng=np.random.default_rng(0),
        )

    assert calls == [4, 4, 2]
    assert history[0]["games_played"] == 10


def test_self_play_batch_size_none_defaults_to_single_batch(tmp_path):
    """Backward compatibility: no self_play_batch_size -> one batch of
    games_per_iteration, matching pre-hardening behavior exactly."""
    import training.train as train_module

    calls = []
    original = train_module.run_batched_self_play

    def spy(*args, **kwargs):
        calls.append(kwargs["num_games"])
        return original(*args, **kwargs)

    network, device, optimizer, buffer, self_play_config, train_config = _setup(tmp_path)

    import unittest.mock
    with unittest.mock.patch.object(train_module, "run_batched_self_play", spy):
        run_training_loop(
            network, device, optimizer, buffer, self_play_config, train_config,
            num_iterations=1, games_per_iteration=3, rng=np.random.default_rng(0),
        )

    assert calls == [3]


def test_failing_self_play_micro_batch_is_logged_and_skipped(tmp_path):
    """One micro-batch raising must not stop the others from running,
    and must not crash the iteration — the failure is logged, that
    batch's games are simply not counted."""
    import training.train as train_module

    original = train_module.run_batched_self_play
    call_count = {"n": 0}

    def flaky(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated self-play crash")
        return original(*args, **kwargs)

    network, device, optimizer, buffer, self_play_config, train_config = _setup(
        tmp_path, self_play_batch_size=2,
    )

    import unittest.mock
    with unittest.mock.patch.object(train_module, "run_batched_self_play", flaky):
        history = run_training_loop(
            network, device, optimizer, buffer, self_play_config, train_config,
            num_iterations=1, games_per_iteration=6, rng=np.random.default_rng(0),
        )

    assert call_count["n"] == 3  # all 3 micro-batches attempted despite the 2nd failing
    assert history[0]["games_played"] == 4  # batch 1 (2 games) + batch 3 (2 games); batch 2 lost
    error_text = (tmp_path / "logs" / "errors.log").read_text()
    assert "simulated self-play crash" in error_text
    assert "self-play micro-batch" in error_text


def test_failing_train_steps_is_logged_and_treated_as_skipped(tmp_path):
    import training.train as train_module

    def broken_train_steps(*args, **kwargs):
        raise RuntimeError("simulated OOM")

    network, device, optimizer, buffer, self_play_config, train_config = _setup(tmp_path)

    import unittest.mock
    with unittest.mock.patch.object(train_module, "train_steps", broken_train_steps):
        history = run_training_loop(
            network, device, optimizer, buffer, self_play_config, train_config,
            num_iterations=1, games_per_iteration=2, rng=np.random.default_rng(0),
        )

    assert history[0]["policy_loss"] is None
    assert history[0]["eval_vs_previous_new_win_rate"] is None  # correctly treated as "no training happened"
    error_text = (tmp_path / "logs" / "errors.log").read_text()
    assert "simulated OOM" in error_text


def test_failing_evaluation_is_logged_and_run_continues(tmp_path):
    import training.train as train_module

    def broken_eval(*args, **kwargs):
        raise RuntimeError("simulated eval crash")

    network, device, optimizer, buffer, self_play_config, train_config = _setup(tmp_path)

    import unittest.mock
    with unittest.mock.patch.object(train_module, "run_batched_evaluate_checkpoints", broken_eval):
        history = run_training_loop(
            network, device, optimizer, buffer, self_play_config, train_config,
            num_iterations=2, games_per_iteration=2, rng=np.random.default_rng(0),
        )

    assert len(history) == 2  # run continued past the failed eval into iteration 2
    assert all(r["eval_vs_previous_new_win_rate"] is None for r in history)
    error_text = (tmp_path / "logs" / "errors.log").read_text()
    assert "simulated eval crash" in error_text


def test_iteration_level_backstop_catches_unexpected_failure_and_continues(tmp_path):
    """Something failing outside the individually-wrapped sub-steps
    (simulated here as a broken checkpoint dir causing save_checkpoint
    to raise past its own _safe_call — instead we directly break
    something unwrapped: the logger's log call) must not stop later
    iterations from running."""
    import training.train as train_module

    network, device, optimizer, buffer, self_play_config, train_config = _setup(tmp_path)

    original_log = train_module.TrainingLogger.log
    call_count = {"n": 0}

    def flaky_log(self, **fields):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated logging failure")
        return original_log(self, **fields)

    import unittest.mock
    with unittest.mock.patch.object(train_module.TrainingLogger, "log", flaky_log):
        history = run_training_loop(
            network, device, optimizer, buffer, self_play_config, train_config,
            num_iterations=2, games_per_iteration=2, rng=np.random.default_rng(0),
        )

    # Iteration 0's failure is swallowed by the iteration-level backstop
    # (no history entry for it), but iteration 1 still runs and is recorded.
    assert len(history) == 1
    assert history[0]["iteration"] == 1
    error_text = (tmp_path / "logs" / "errors.log").read_text()
    assert "simulated logging failure" in error_text
    assert "unhandled iteration-level failure" in error_text


def test_iteration_and_gpu_memory_fields_present(tmp_path):
    network, device, optimizer, buffer, self_play_config, train_config = _setup(tmp_path)

    history = run_training_loop(
        network, device, optimizer, buffer, self_play_config, train_config,
        num_iterations=1, games_per_iteration=2, rng=np.random.default_rng(0),
    )

    record = history[0]
    assert record["iteration_seconds"] > 0
    assert record["iteration_seconds"] >= record["self_play_seconds"]
    # CPU device -> None, not a crash or a fabricated number.
    assert record["gpu_memory_allocated_mb"] is None
    assert record["gpu_memory_reserved_mb"] is None

    header = (tmp_path / "logs" / "train_log.csv").read_text().splitlines()[0]
    assert "iteration_seconds" in header
    assert "gpu_memory_allocated_mb" in header
    assert "gpu_memory_reserved_mb" in header


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")
def test_gpu_memory_fields_populated_on_cuda(tmp_path):
    network = PolicyValueNet().to(torch.device("cuda"))
    device = torch.device("cuda")
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=2000)
    self_play_config = SelfPlayConfig(num_simulations=5, temperature=0.0, temperature_threshold_plies=0)
    train_config = _base_config(tmp_path)

    history = run_training_loop(
        network, device, optimizer, buffer, self_play_config, train_config,
        num_iterations=1, games_per_iteration=2, rng=np.random.default_rng(0),
    )

    assert history[0]["gpu_memory_allocated_mb"] is not None
    assert history[0]["gpu_memory_allocated_mb"] >= 0
    assert history[0]["gpu_memory_reserved_mb"] >= history[0]["gpu_memory_allocated_mb"]
