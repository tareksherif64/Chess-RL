"""Play N random-vs-random games through ChessEnv to validate the
environment before any RL code exists: no illegal moves are ever
attempted (every action comes from action_mask()), games always
terminate, and outcomes are tallied by cause (checkmate / stalemate /
insufficient material / repetition / 50-move rule).

Usage:
    python scripts/self_play_random.py --games 200
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.random_agent import RandomAgent
from engine.chess_env import ChessEnv


def play_one_game(env: ChessEnv, white: RandomAgent, black: RandomAgent) -> dict:
    obs, info = env.reset()
    terminated = truncated = False
    ply_count = 0

    while not (terminated or truncated):
        agent = white if env.current_player() else black
        mask = env.action_mask()
        action = agent.select_action(mask)
        obs, reward, terminated, truncated, info = env.step(action)
        ply_count += 1

    outcome = info["outcome"]
    return {
        "plies": ply_count,
        "result": outcome.result() if outcome else "*",
        "termination": outcome.termination.name if outcome else "UNKNOWN",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = ChessEnv()
    white = RandomAgent(seed=args.seed)
    black = RandomAgent(seed=args.seed + 1)

    results = Counter()
    terminations = Counter()
    total_plies = 0

    for game_idx in range(args.games):
        summary = play_one_game(env, white, black)
        results[summary["result"]] += 1
        terminations[summary["termination"]] += 1
        total_plies += summary["plies"]
        print(
            f"game {game_idx + 1}/{args.games}: "
            f"result={summary['result']} "
            f"termination={summary['termination']} "
            f"plies={summary['plies']}"
        )

    print("\n--- summary ---")
    print(f"games played:   {args.games}")
    print(f"avg plies/game: {total_plies / args.games:.1f}")
    print(f"results:        {dict(results)}")
    print(f"terminations:   {dict(terminations)}")


if __name__ == "__main__":
    main()
