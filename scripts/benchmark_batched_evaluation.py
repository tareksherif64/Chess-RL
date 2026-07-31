"""Benchmark: serial evaluation (training/evaluation.py, one game at a
time, batch size 1) vs batched evaluation (training/batched_evaluation.py,
N games concurrently, leaves split per network per round) at a fixed
simulation count, so the throughput fix is measured in isolation — same
as the self-play benchmark before it.

Reports moves/sec (total plies / wall time), not just raw wall-time for
a fixed game count — evaluation games run fully deterministically (zero
exploration noise, since evaluation measures strongest play, not
self-play exploration), and it turns out serial vs. batched search can
settle into *very* different length games for the exact same two
untrained networks (see docs/batched_evaluation.md) — that's the same
"games/hour conflates with total plies played" trap the self-play
benchmark already had to correct for, so this one measures the same way.

Usage:
    python scripts/benchmark_batched_evaluation.py --games 20 --simulations 100
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import torch

from engine.chess_env import ChessEnv
from training.batched_evaluation import run_batched_evaluate_checkpoints, _GameSpec, _play_concurrent_games
from training.device import resolve_device
from training.evaluation import mcts_player
from training.network import PolicyValueNet


def _play_and_count_plies(white_player, black_player, claim_draw=True, max_plies=600) -> int:
    """Mirrors training.evaluation.play_game_between's loop exactly,
    only additionally returning the ply count (that function's public
    return type — just a result string — is unchanged, since existing
    tests depend on it)."""
    env = ChessEnv(claim_draw=claim_draw)
    env.reset()
    terminated = truncated = False
    ply = 0
    while not (terminated or truncated) and ply < max_plies:
        player = white_player if env.current_player() == chess.WHITE else black_player
        action = player(env.board)
        _obs, _reward, terminated, truncated, _info = env.step(action)
        ply += 1
    return ply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--leaves-per-tree", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = resolve_device(require_cuda=not args.cpu)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda device name: {torch.cuda.get_device_name(device)}")

    torch.manual_seed(args.seed)
    new_network = PolicyValueNet().to(device)
    new_network.eval()
    torch.manual_seed(args.seed + 1)
    old_network = PolicyValueNet().to(device)
    old_network.eval()
    print(
        f"network device check: new={next(new_network.parameters()).device} "
        f"old={next(old_network.parameters()).device}\n"
    )
    print(f"games={args.games}  simulations/move={args.simulations}  leaves_per_tree_per_round={args.leaves_per_tree}\n")

    print("=== serial (old) ===")
    new_player = mcts_player(new_network, device, args.simulations)
    old_player = mcts_player(old_network, device, args.simulations)
    serial_start = time.perf_counter()
    serial_plies = 0
    for i in range(args.games):
        white, black = (new_player, old_player) if i % 2 == 0 else (old_player, new_player)
        serial_plies += _play_and_count_plies(white, black)
    serial_wall = time.perf_counter() - serial_start
    print(f"total plies={serial_plies}  wall time={serial_wall:.1f}s\n")

    print("=== batched (new) ===")
    specs = [
        _GameSpec(white_kind="new", black_kind="old") if i % 2 == 0
        else _GameSpec(white_kind="old", black_kind="new")
        for i in range(args.games)
    ]
    batched_start = time.perf_counter()
    _results, envs = _play_concurrent_games(
        specs, {"new": new_network, "old": old_network}, device, args.simulations,
        c_puct=1.5, claim_draw=True, max_plies=600, leaves_per_tree_per_round=args.leaves_per_tree,
        random_agent=None, initial_fen=None,
    )
    batched_wall = time.perf_counter() - batched_start
    batched_plies = sum(len(env.board.move_stack) for env in envs)
    print(f"total plies={batched_plies}  wall time={batched_wall:.1f}s\n")

    serial_moves_per_sec = serial_plies / serial_wall if serial_wall > 0 else 0.0
    batched_moves_per_sec = batched_plies / batched_wall if batched_wall > 0 else 0.0

    print("=== results ===")
    print(f"{'':20s}{'serial (old)':>18s}{'batched (new)':>18s}")
    print(f"{'wall time (s)':20s}{serial_wall:>18.1f}{batched_wall:>18.1f}")
    print(f"{'total plies':20s}{serial_plies:>18d}{batched_plies:>18d}")
    print(f"{'moves/sec':20s}{serial_moves_per_sec:>18.3f}{batched_moves_per_sec:>18.3f}")
    speedup = batched_moves_per_sec / serial_moves_per_sec if serial_moves_per_sec > 0 else float("nan")
    print(f"\nspeedup (moves/sec): {speedup:.2f}x")


if __name__ == "__main__":
    main()
