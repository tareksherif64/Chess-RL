"""Stage 4: train the policy/value network on self-play data, and the
full AlphaZero iteration loop (self-play -> train -> checkpoint -> repeat).

Hardened for unattended long runs (see docs/overnight_run.md): every
sub-step (a self-play micro-batch, a training pass, either evaluation,
a checkpoint save) is wrapped so a single failure is logged and
skipped rather than taking down the whole run, and an iteration-level
backstop catches anything that slips past those.
"""

import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.batched_evaluation import (
    run_batched_evaluate_against_random,
    run_batched_evaluate_checkpoints,
)
from training.batched_self_play import run_batched_self_play
from training.checkpoint import save_checkpoint
from training.logger import TrainingLogger
from training.network import PolicyValueNet
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayConfig
from training.tensors import obs_to_tensor


@dataclass
class TrainConfig:
    batch_size: int
    train_steps_per_iteration: int
    min_buffer_size: int
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_iterations: int = 1
    buffer_path: str | None = None
    log_path: str = "logs/train_log.csv"
    error_log_path: str | None = None
    # Batched execution — shared batching depth for both self-play
    # (training/batched_self_play.py) and evaluation
    # (training/batched_evaluation.py). games_per_iteration and
    # eval_*_games each act as N (concurrent games/trees) for their
    # respective batched call; this controls the other axis for both.
    leaves_per_tree_per_round: int = 4
    # Self-play concurrency per batched call — separate from
    # games_per_iteration (the *total* self-play games wanted per
    # iteration). If games_per_iteration exceeds this, self-play runs
    # as multiple back-to-back micro-batches of up to this size rather
    # than one huge batch, so a large per-iteration game target doesn't
    # require running at an untested concurrency level. None (default)
    # runs everything as a single batch (games_per_iteration == N),
    # matching the original behavior.
    self_play_batch_size: int | None = None
    # Strength evaluation (see training/batched_evaluation.py) — loss
    # going down doesn't prove the network plays better, these games
    # are the ground-truth signal. vs-previous runs every iteration
    # (skipped automatically if training itself was skipped that
    # iteration, since "new" and "old" would be identical); vs-random
    # runs every `eval_vs_random_every_iterations` iterations as a
    # sanity floor.
    eval_vs_previous_games: int = 10
    eval_vs_random_games: int = 10
    eval_vs_random_every_iterations: int = 5


def compute_loss(
    network: PolicyValueNet, boards: torch.Tensor, policies: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Policy loss is plain cross-entropy against the (already legal-
    move-masked, from MCTS visit counts) target distribution, computed
    against the network's *raw* (unmasked) logits — the network is
    expected to learn to assign near-zero probability to illegal moves
    from this signal alone, exactly as in the AlphaZero paper. No mask
    is applied here (masking is an MCTS/inference-time concern, see
    training/network.py::masked_softmax — using it during training
    would hide information the network needs to learn from).
    Value loss is plain MSE against the +1/-1/0 game-outcome target."""
    policy_logits, value_pred = network(boards)
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(policies * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(value_pred, values)
    return policy_loss, value_loss, policy_loss + value_loss


def train_steps(
    network: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    num_steps: int,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run `num_steps` minibatch gradient updates sampled from `buffer`.
    Returns average policy/value/total loss over the steps taken."""
    rng = rng or np.random.default_rng()
    network.train()

    totals = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
    for _ in range(num_steps):
        boards_np, policies_np, values_np = buffer.sample(batch_size, rng=rng)
        boards = obs_to_tensor(boards_np, device)
        policies = torch.from_numpy(policies_np).to(device)
        values = torch.from_numpy(values_np).to(device)

        policy_loss, value_loss, loss = compute_loss(network, boards, policies, values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totals["policy_loss"] += policy_loss.item()
        totals["value_loss"] += value_loss.item()
        totals["total_loss"] += loss.item()

    network.eval()
    return {k: v / num_steps for k, v in totals.items()}


class _ErrorLog:
    """Appends full tracebacks to a plain text file, kept separate from
    the structured CSV log. A no-op if no path is given."""

    def __init__(self, path: str | None):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a") if path else None

    def write(self, description: str, exc: Exception) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"[{timestamp}] {description}: {type(exc).__name__}: {exc}"
        print(f"ERROR: {header}")
        if self._file:
            self._file.write(header + "\n")
            self._file.write(traceback.format_exc() + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()


def _safe_call(description: str, error_log: _ErrorLog, fn, *args, **kwargs):
    """Run fn(*args, **kwargs); on any exception, log it and return
    None instead of propagating. Used around every sub-step of an
    iteration so one failure (a crashed game, a transient CUDA error,
    a full disk) can't take down an unattended overnight run — the
    iteration simply proceeds with that step's result treated as "did
    not happen" (same as the existing skip-if-buffer-too-small /
    skip-if-not-cadence conventions already use)."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — intentionally broad: this is the last line of defense
        error_log.write(description, exc)
        return None


def _gpu_memory_mb(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    reserved = torch.cuda.memory_reserved(device) / (1024**2)
    return allocated, reserved


def run_training_loop(
    network: PolicyValueNet,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    self_play_config: SelfPlayConfig,
    train_config: TrainConfig,
    num_iterations: int,
    games_per_iteration: int,
    start_iteration: int = 0,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """One full AlphaZero cycle per iteration: play `games_per_iteration`
    self-play games (as one or more concurrent micro-batches — see
    `TrainConfig.self_play_batch_size`), train on
    `train_steps_per_iteration` minibatches sampled from the
    (accumulated) buffer, evaluate the resulting checkpoint's strength,
    checkpoint, log. Training is skipped (but self-play still runs and
    still fills the buffer) on any iteration where the buffer hasn't
    reached `min_buffer_size` yet — avoids gradient-stepping on a
    handful of highly-correlated positions from one or two games."""
    rng = rng or np.random.default_rng()
    history = []
    error_log = _ErrorLog(train_config.error_log_path)
    batch_size = train_config.self_play_batch_size or games_per_iteration

    try:
        with TrainingLogger(train_config.log_path) as logger:
            for i in range(num_iterations):
                iteration = start_iteration + i
                iteration_start = time.perf_counter()
                try:
                    network.eval()

                    # Snapshot pre-training weights for the vs-previous
                    # eval below — cheap (small network), needed
                    # regardless of whether training runs this iteration.
                    previous_state = {k: v.clone() for k, v in network.state_dict().items()}

                    self_play_start = time.perf_counter()
                    summaries = []
                    games_remaining = games_per_iteration
                    while games_remaining > 0:
                        this_batch = min(batch_size, games_remaining)
                        batch_summaries = _safe_call(
                            f"iteration {iteration}: self-play micro-batch ({this_batch} games)",
                            error_log,
                            run_batched_self_play,
                            network, device, buffer, self_play_config,
                            num_games=this_batch,
                            leaves_per_tree_per_round=train_config.leaves_per_tree_per_round,
                            rng=rng, verbose=True,
                        )
                        summaries.extend(batch_summaries or [])
                        games_remaining -= this_batch
                    self_play_seconds = time.perf_counter() - self_play_start

                    loss_stats = None
                    train_seconds = 0.0
                    if len(buffer) >= train_config.min_buffer_size:
                        train_start = time.perf_counter()
                        loss_stats = _safe_call(
                            f"iteration {iteration}: train_steps",
                            error_log,
                            train_steps,
                            network, optimizer, buffer, device,
                            train_config.train_steps_per_iteration, train_config.batch_size, rng=rng,
                        )
                        train_seconds = time.perf_counter() - train_start
                        network.eval()  # train_steps() already restores eval mode; cheap insurance if it raised mid-way
                        if loss_stats is not None:
                            print(
                                f"iteration {iteration}: policy_loss={loss_stats['policy_loss']:.4f} "
                                f"value_loss={loss_stats['value_loss']:.4f} total_loss={loss_stats['total_loss']:.4f}"
                            )
                    else:
                        print(
                            f"iteration {iteration}: buffer has {len(buffer)}/{train_config.min_buffer_size} "
                            "examples, skipping training this iteration"
                        )

                    # vs-previous: only meaningful if training actually
                    # changed the weights this iteration (otherwise
                    # "new" == "old").
                    eval_vs_previous = None
                    eval_vs_previous_seconds = 0.0
                    if loss_stats is not None:
                        previous_network = PolicyValueNet().to(device)
                        previous_network.load_state_dict(previous_state)
                        previous_network.eval()

                        eval_start = time.perf_counter()
                        eval_vs_previous = _safe_call(
                            f"iteration {iteration}: vs-previous evaluation",
                            error_log,
                            run_batched_evaluate_checkpoints,
                            network, previous_network, device,
                            num_games=train_config.eval_vs_previous_games,
                            num_simulations=self_play_config.num_simulations,
                            c_puct=self_play_config.c_puct, claim_draw=self_play_config.claim_draw,
                            max_plies=self_play_config.max_plies,
                            leaves_per_tree_per_round=train_config.leaves_per_tree_per_round,
                        )
                        eval_vs_previous_seconds = time.perf_counter() - eval_start
                        if eval_vs_previous is not None:
                            print(
                                f"iteration {iteration}: vs-previous: "
                                f"new={eval_vs_previous['new_wins']} old={eval_vs_previous['old_wins']} "
                                f"draw={eval_vs_previous['draws']} (new_win_rate={eval_vs_previous['new_win_rate']:.2f})"
                            )

                    # vs-random: periodic sanity floor.
                    eval_vs_random = None
                    eval_vs_random_seconds = 0.0
                    if (iteration + 1) % train_config.eval_vs_random_every_iterations == 0:
                        eval_start = time.perf_counter()
                        eval_vs_random = _safe_call(
                            f"iteration {iteration}: vs-random evaluation",
                            error_log,
                            run_batched_evaluate_against_random,
                            network, device,
                            num_games=train_config.eval_vs_random_games,
                            num_simulations=self_play_config.num_simulations,
                            rng=rng,
                            c_puct=self_play_config.c_puct, claim_draw=self_play_config.claim_draw,
                            max_plies=self_play_config.max_plies,
                            leaves_per_tree_per_round=train_config.leaves_per_tree_per_round,
                        )
                        eval_vs_random_seconds = time.perf_counter() - eval_start
                        if eval_vs_random is not None:
                            print(
                                f"iteration {iteration}: vs-random: "
                                f"network_win_rate={eval_vs_random['network_win_rate']:.2f} "
                                f"({eval_vs_random['network_wins']}/{eval_vs_random['games']})"
                            )

                    if (iteration + 1) % train_config.checkpoint_every_iterations == 0:
                        ckpt_path = f"{train_config.checkpoint_dir}/iter_{iteration:06d}.pt"
                        _safe_call(
                            f"iteration {iteration}: checkpoint save",
                            error_log,
                            save_checkpoint,
                            ckpt_path, network, optimizer, iteration,
                        )
                        print(f"iteration {iteration}: saved checkpoint to {ckpt_path}")
                        if train_config.buffer_path:
                            _safe_call(
                                f"iteration {iteration}: buffer save",
                                error_log,
                                buffer.save,
                                train_config.buffer_path,
                            )

                    games_played = sum(1 for s in summaries if not s["discarded"])
                    gpu_allocated_mb, gpu_reserved_mb = _gpu_memory_mb(device)
                    iteration_seconds = time.perf_counter() - iteration_start
                    record = {
                        "iteration": iteration,
                        "buffer_size": len(buffer),
                        "games_played": games_played,
                        "games_discarded": len(summaries) - games_played,
                        "iteration_seconds": iteration_seconds,
                        "self_play_seconds": self_play_seconds,
                        "train_seconds": train_seconds,
                        "games_per_hour": (games_played / self_play_seconds * 3600) if self_play_seconds > 0 else 0.0,
                        "policy_loss": loss_stats["policy_loss"] if loss_stats else None,
                        "value_loss": loss_stats["value_loss"] if loss_stats else None,
                        "total_loss": loss_stats["total_loss"] if loss_stats else None,
                        "eval_vs_previous_new_win_rate": eval_vs_previous["new_win_rate"] if eval_vs_previous else None,
                        "eval_vs_previous_draw_rate": eval_vs_previous["draw_rate"] if eval_vs_previous else None,
                        "eval_vs_previous_games": eval_vs_previous["games"] if eval_vs_previous else None,
                        "eval_vs_previous_seconds": eval_vs_previous_seconds,
                        "eval_vs_random_win_rate": eval_vs_random["network_win_rate"] if eval_vs_random else None,
                        "eval_vs_random_games": eval_vs_random["games"] if eval_vs_random else None,
                        "eval_vs_random_seconds": eval_vs_random_seconds,
                        "gpu_memory_allocated_mb": gpu_allocated_mb,
                        "gpu_memory_reserved_mb": gpu_reserved_mb,
                    }
                    print(
                        f"iteration {iteration}: total={iteration_seconds:.1f}s"
                        + (f"  gpu_allocated={gpu_allocated_mb:.0f}MB gpu_reserved={gpu_reserved_mb:.0f}MB" if gpu_allocated_mb is not None else "")
                    )
                    logger.log(**record)
                    history.append(record)
                except Exception as exc:  # noqa: BLE001 — iteration-level backstop
                    error_log.write(f"iteration {iteration}: unhandled iteration-level failure", exc)
                    continue
    finally:
        error_log.close()

    return history
