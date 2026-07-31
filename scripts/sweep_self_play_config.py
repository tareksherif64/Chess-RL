"""Sweep a few (N concurrent games, leaves_per_tree_per_round) configs
for batched self-play at a fixed simulation count, to check whether the
benchmarked N=8/leaves=4 config is actually the best use of the GPU or
just the first thing tried — an RTX 4060 may not be saturated yet at
batch size 32.

Runs num_games = N for each config (i.e. each config is tested exactly
as it would run in one real training iteration: N concurrent games in
one run_batched_self_play call), all at the same simulations/move for
a fair comparison, and reports moves/sec for each.

Usage:
    python scripts/sweep_self_play_config.py --simulations 100
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
from training.self_play import SelfPlayConfig


def run_config(network, device, config: SelfPlayConfig, num_games: int, leaves: int, seed: int):
    buffer = ReplayBuffer(capacity=100_000)
    start = time.perf_counter()
    summaries = run_batched_self_play(
        network, device, buffer, config,
        num_games=num_games, leaves_per_tree_per_round=leaves,
        rng=np.random.default_rng(seed), verbose=True,
    )
    wall = time.perf_counter() - start
    plies = sum(s["plies"] for s in summaries)
    return {
        "N": num_games,
        "leaves": leaves,
        "batch_size_cap": num_games * leaves,
        "wall_seconds": wall,
        "plies": plies,
        "moves_per_sec": plies / wall if wall > 0 else 0.0,
        "games_per_hour": num_games / wall * 3600 if wall > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = resolve_device(require_cuda=not args.cpu)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda device name: {torch.cuda.get_device_name(device)}")

    torch.manual_seed(args.seed)
    network = PolicyValueNet().to(device)
    network.eval()
    print(f"network device check: {next(network.parameters()).device}\n")

    config = SelfPlayConfig(num_simulations=args.simulations)
    configs = [(8, 4), (16, 4), (8, 8)]

    results = []
    for N, leaves in configs:
        print(f"=== N={N} leaves_per_tree_per_round={leaves} (batch cap {N * leaves}) ===")
        result = run_config(network, device, config, num_games=N, leaves=leaves, seed=args.seed)
        results.append(result)
        print(
            f"-> wall={result['wall_seconds']:.1f}s plies={result['plies']} "
            f"moves/sec={result['moves_per_sec']:.3f} games/hour={result['games_per_hour']:.1f}\n"
        )

    print("=== summary ===")
    print(f"{'N':>4s}{'leaves':>8s}{'batch cap':>11s}{'moves/sec':>12s}{'games/hour':>13s}")
    for r in results:
        print(
            f"{r['N']:>4d}{r['leaves']:>8d}{r['batch_size_cap']:>11d}"
            f"{r['moves_per_sec']:>12.3f}{r['games_per_hour']:>13.1f}"
        )

    winner = max(results, key=lambda r: r["moves_per_sec"])
    print(f"\nwinner: N={winner['N']} leaves={winner['leaves']} ({winner['moves_per_sec']:.3f} moves/sec)")


if __name__ == "__main__":
    main()
