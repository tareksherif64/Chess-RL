"""Play random-vs-random game(s) through ChessEnv with the board printed
to the terminal after every move, for visually watching/sanity-checking
behavior (as opposed to self_play_random.py's aggregate-stats mode).

Usage:
    python scripts/watch_random_game.py
    python scripts/watch_random_game.py --games 3 --delay 0.3
    python scripts/watch_random_game.py --ascii --no-clear   # plain terminals
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.random_agent import RandomAgent
from engine.chess_env import ChessEnv

CLEAR = "\033[2J\033[H"


def render(board, ascii_mode: bool) -> str:
    return str(board) if ascii_mode else board.unicode(borders=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.6, help="seconds between moves")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ascii", action="store_true", help="plain ASCII board instead of unicode pieces")
    parser.add_argument("--no-clear", action="store_true", help="don't clear the terminal between moves")
    args = parser.parse_args()

    env = ChessEnv()
    white = RandomAgent(seed=args.seed)
    black = RandomAgent(seed=None if args.seed is None else args.seed + 1)

    for game_idx in range(args.games):
        env.reset()
        terminated = truncated = False
        ply = 0

        try:
            while not (terminated or truncated):
                if not args.no_clear:
                    print(CLEAR, end="")
                mover = "White" if env.current_player() else "Black"
                print(f"Game {game_idx + 1}/{args.games}   Ply {ply}   {mover} to move\n")
                print(render(env.board, args.ascii))
                time.sleep(args.delay)

                agent = white if env.current_player() else black
                action = agent.select_action(env.action_mask())
                san = env.board.san(env.decode_move(action))
                _, reward, terminated, truncated, info = env.step(action)
                ply += 1
                print(f"\n{mover} played {san}")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return

        if not args.no_clear:
            print(CLEAR, end="")
        print(f"Game {game_idx + 1} finished after {ply} plies\n")
        print(render(env.board, args.ascii))
        outcome = info["outcome"]
        result = outcome.result() if outcome else "*"
        termination = outcome.termination.name if outcome else "UNKNOWN"
        print(f"\nResult: {result}   Termination: {termination}")
        time.sleep(max(args.delay, 1.5))


if __name__ == "__main__":
    main()
