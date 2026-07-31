"""Run a batch of self-play games and save the resulting replay buffer.

This is the Stage 3 sanity-check entry point: run a small number of
games, log each one, inspect the aggregate stats and saved data before
ever scaling up to a real multi-hour self-play run (that scale-up is a
deliberate later decision, not something this script defaults into).

Usage:
    python scripts/run_self_play.py --games 10 --simulations 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from training.device import resolve_device
from training.network import PolicyValueNet
from training.replay_buffer import ReplayBuffer
from training.self_play import SelfPlayConfig, run_self_play


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-plies", type=int, default=30)
    parser.add_argument("--max-plies", type=int, default=600)
    parser.add_argument("--buffer-capacity", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/self_play_buffer.npz")
    parser.add_argument("--cpu", action="store_true", help="allow CPU fallback (default requires CUDA)")
    args = parser.parse_args()

    device = resolve_device(require_cuda=not args.cpu)
    print(f"device: {device}")

    network = PolicyValueNet().to(device)
    network.eval()
    total_params = sum(p.numel() for p in network.parameters())
    print(f"network: {total_params:,} params (untrained — Stage 4 hasn't run yet)")

    config = SelfPlayConfig(
        num_simulations=args.simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        temperature_threshold_plies=args.temperature_plies,
        max_plies=args.max_plies,
    )
    buffer = ReplayBuffer(capacity=args.buffer_capacity)
    rng = np.random.default_rng(args.seed)

    print(f"playing {args.games} self-play games, {args.simulations} simulations/move...\n")
    summaries = run_self_play(network, device, buffer, config, num_games=args.games, rng=rng)

    discarded = sum(1 for s in summaries if s["discarded"])
    kept = [s for s in summaries if not s["discarded"]]
    total_examples = sum(s["examples"] for s in summaries)
    total_time = sum(s["wall_seconds"] for s in summaries)

    print("\n--- summary ---")
    print(f"games played:      {args.games}  (discarded: {discarded})")
    print(f"total examples:    {total_examples}")
    print(f"replay buffer size:{len(buffer)}")
    if kept:
        avg_plies = sum(s["plies"] for s in kept) / len(kept)
        print(f"avg plies/game:    {avg_plies:.1f}")
        from collections import Counter
        print(f"terminations:      {dict(Counter(s['termination'] for s in kept))}")
    print(f"total wall time:   {total_time:.1f}s ({total_time / max(args.games,1):.1f}s/game)")
    if total_time > 0:
        print(f"games/hour (est):  {args.games / total_time * 3600:.1f}")

    buffer.save(args.out)
    print(f"\nsaved replay buffer ({len(buffer)} examples) to {args.out}")


if __name__ == "__main__":
    main()
