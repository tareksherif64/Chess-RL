"""Benchmark: serial (batch-size-1) self-play vs batched (N concurrent
games + virtual loss) self-play, at the *same* simulation count, to
measure the throughput fix in isolation.

Does not change simulation count, games-per-iteration, or any other
training hyperparameter — this only compares two ways of running self-
play at a fixed, held-constant 100 simulations/move, per the plan.

Usage:
    python scripts/benchmark_batched_self_play.py --games 8 --simulations 100
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from training.batched_self_play import run_batched_self_play
from training.device import resolve_device
from training.network import PolicyValueNet
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayConfig, run_self_play


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=8, help="games for both the serial and batched runs")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations/move — held fixed for a fair comparison")
    parser.add_argument("--leaves-per-tree", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="allow CPU fallback (default requires CUDA)")
    args = parser.parse_args()

    device = resolve_device(require_cuda=not args.cpu)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda device name: {torch.cuda.get_device_name(device)}")
    print(f"games={args.games}  simulations/move={args.simulations}  leaves_per_tree_per_round={args.leaves_per_tree}\n")

    torch.manual_seed(args.seed)
    network = PolicyValueNet().to(device)
    network.eval()
    print(f"network device check: {next(network.parameters()).device}\n")

    config = SelfPlayConfig(num_simulations=args.simulations)

    # --- old: serial, one game at a time, batch size 1 per network call ---
    print("=== serial (old) ===")
    serial_buffer = ReplayBuffer(capacity=100_000)
    serial_start = time.perf_counter()
    serial_summaries = run_self_play(
        network, device, serial_buffer, config,
        num_games=args.games, rng=np.random.default_rng(args.seed), verbose=True,
    )
    serial_wall = time.perf_counter() - serial_start
    serial_plies = sum(s["plies"] for s in serial_summaries)
    serial_games_per_hour = args.games / serial_wall * 3600
    serial_moves_per_sec = serial_plies / serial_wall

    # --- new: N games concurrently, batched leaf evaluation + virtual loss ---
    print("\n=== batched (new) ===")
    batched_buffer = ReplayBuffer(capacity=100_000)
    batched_start = time.perf_counter()
    batched_summaries = run_batched_self_play(
        network, device, batched_buffer, config,
        num_games=args.games, leaves_per_tree_per_round=args.leaves_per_tree,
        rng=np.random.default_rng(args.seed), verbose=True,
    )
    batched_wall = time.perf_counter() - batched_start
    batched_plies = sum(s["plies"] for s in batched_summaries)
    batched_games_per_hour = args.games / batched_wall * 3600
    batched_moves_per_sec = batched_plies / batched_wall

    print("\n=== results ===")
    print(f"{'':20s}{'serial (old)':>18s}{'batched (new)':>18s}")
    print(f"{'wall time (s)':20s}{serial_wall:>18.1f}{batched_wall:>18.1f}")
    print(f"{'total plies':20s}{serial_plies:>18d}{batched_plies:>18d}")
    print(f"{'moves/sec':20s}{serial_moves_per_sec:>18.3f}{batched_moves_per_sec:>18.3f}")
    print(f"{'games/hour':20s}{serial_games_per_hour:>18.1f}{batched_games_per_hour:>18.1f}")
    speedup = batched_moves_per_sec / serial_moves_per_sec if serial_moves_per_sec > 0 else float("nan")
    print(f"\nspeedup (moves/sec): {speedup:.2f}x")


if __name__ == "__main__":
    main()
