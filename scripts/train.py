"""Run the full self-play + training loop.

This is the Stage 4 sanity-check entry point too: run a handful of
iterations at small scale (few games, few simulations, few train
steps) before ever pointing this at real multi-hour settings — that
scale-up is a deliberate later decision made together, not a default
this script reaches for on its own.

Usage:
    python scripts/train.py --iterations 2 --games-per-iteration 2 --simulations 20 --train-steps 20
    python scripts/train.py --resume checkpoints/iter_000001.pt --iterations 5 ...
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from training.checkpoint import load_checkpoint
from training.device import resolve_device
from training.network import PolicyValueNet
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayConfig
from training.train import TrainConfig, run_training_loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--games-per-iteration", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=20, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-plies", type=int, default=30)
    parser.add_argument("--max-plies", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=20, help="gradient steps per iteration")
    parser.add_argument("--min-buffer-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--buffer-capacity", type=int, default=20_000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--buffer-path", type=str, default="data/self_play_buffer.npz")
    parser.add_argument("--log-path", type=str, default="logs/train_log.csv")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    parser.add_argument("--load-buffer", type=str, default=None, help="existing replay buffer .npz to preload")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="allow CPU fallback (default requires CUDA)")
    args = parser.parse_args()

    device = resolve_device(require_cuda=not args.cpu)
    print(f"device: {device}")

    network = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(args.resume, network, optimizer, device=device) + 1
        print(f"resumed from {args.resume}, continuing at iteration {start_iteration}")

    if args.load_buffer and Path(args.load_buffer).exists():
        buffer = ReplayBuffer.load(args.load_buffer)
        print(f"loaded replay buffer with {len(buffer)} examples from {args.load_buffer}")
    else:
        buffer = ReplayBuffer(capacity=args.buffer_capacity)

    self_play_config = SelfPlayConfig(
        num_simulations=args.simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        temperature_threshold_plies=args.temperature_plies,
        max_plies=args.max_plies,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        train_steps_per_iteration=args.train_steps,
        min_buffer_size=args.min_buffer_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every_iterations=args.checkpoint_every,
        buffer_path=args.buffer_path,
        log_path=args.log_path,
    )

    print(
        f"running {args.iterations} iterations: {args.games_per_iteration} games/iter, "
        f"{args.simulations} simulations/move, {args.train_steps} train steps/iter\n"
    )
    history = run_training_loop(
        network, device, optimizer, buffer, self_play_config, train_config,
        num_iterations=args.iterations, games_per_iteration=args.games_per_iteration,
        start_iteration=start_iteration, rng=np.random.default_rng(args.seed),
    )

    print("\n--- run complete ---")
    for record in history:
        print(record)


if __name__ == "__main__":
    main()
