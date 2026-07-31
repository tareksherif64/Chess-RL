"""Self-play driver for BatchedMCTS: run N games concurrently, one
`BatchedMCTS.run_batch` call per ply-round across every still-active
game, instead of one network call per leaf per game in isolation.

Per-game recording/labeling logic (record board+policy+mover each ply,
backfill +1/-1/0 once each game's own outcome is known) is identical
to training/self_play.py::play_self_play_game — this module changes
*how many games run concurrently and how leaf evaluations are batched*,
not what gets recorded or how a game's outcome is scored. See
docs/batched_self_play.md.
"""

import time

import numpy as np
import torch

from engine.chess_env import ChessEnv
from training.batched_mcts import BatchedMCTS
from training.mcts import visit_count_policy
from training.network import PolicyValueNet
from training.replay_buffer import ReplayBuffer, ReplayExample
from training.self_play import SelfPlayConfig, outcome_value, sample_action


def run_batched_self_play(
    network: PolicyValueNet,
    device: torch.device,
    buffer: ReplayBuffer,
    config: SelfPlayConfig,
    num_games: int,
    leaves_per_tree_per_round: int = 4,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
    initial_fen: str | None = None,
) -> list[dict]:
    """Play `num_games` self-play games concurrently. Returns one
    summary dict per game, in the same shape as
    `training.self_play.run_self_play` minus a per-game `wall_seconds`
    (concurrent games don't have a meaningful individual wall time —
    time the whole call instead if you need throughput)."""
    rng = rng or np.random.default_rng()
    mcts = BatchedMCTS(
        network, device=device, c_puct=config.c_puct, claim_draw=config.claim_draw,
        leaves_per_tree_per_round=leaves_per_tree_per_round,
    )

    reset_options = {"fen": initial_fen} if initial_fen else None
    envs = [ChessEnv(claim_draw=config.claim_draw) for _ in range(num_games)]
    obs = []
    for env in envs:
        game_obs, _info = env.reset(options=reset_options)
        obs.append(game_obs)

    pending: list[list[tuple[np.ndarray, np.ndarray, bool]]] = [[] for _ in range(num_games)]
    plies = [0] * num_games
    terminated = [False] * num_games
    infos: list[dict | None] = [None] * num_games
    active = list(range(num_games))

    start = time.perf_counter()
    while active:
        boards = [envs[i].board for i in active]
        movers = [envs[i].current_player() for i in active]
        temperatures = [
            config.temperature if plies[i] < config.temperature_threshold_plies else 0.0
            for i in active
        ]

        roots = mcts.run_batch(
            boards, config.num_simulations,
            add_dirichlet_noise=True,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            rng=rng,
        )

        still_active = []
        for slot, i in enumerate(active):
            temperature = temperatures[slot]
            policy_target = visit_count_policy(roots[slot], temperature=temperature)
            action = sample_action(policy_target, temperature, rng)

            pending[i].append((obs[i], policy_target, movers[slot]))
            game_obs, _reward, term, trunc, info = envs[i].step(action)
            obs[i] = game_obs
            plies[i] += 1
            infos[i] = info

            if term or trunc or plies[i] >= config.max_plies:
                terminated[i] = term
            else:
                still_active.append(i)
        active = still_active
    total_wall = time.perf_counter() - start

    summaries = []
    for i in range(num_games):
        if not terminated[i]:
            summaries.append(
                {"plies": plies[i], "result": "*", "termination": "TRUNCATED_SAFETY_CAP",
                 "discarded": True, "examples": 0}
            )
            continue

        outcome = infos[i]["outcome"]
        winner = outcome.winner
        examples = [
            ReplayExample(board=b, policy=p, value=outcome_value(winner, m))
            for b, p, m in pending[i]
        ]
        if examples:
            buffer.add_game(examples)
        summaries.append(
            {"plies": plies[i], "result": outcome.result(), "termination": outcome.termination.name,
             "discarded": False, "examples": len(examples)}
        )

    if verbose:
        for i, s in enumerate(summaries):
            print(
                f"game {i + 1}/{num_games}: result={s['result']} termination={s['termination']} "
                f"plies={s['plies']} examples={s['examples']}"
            )
        games_per_hour = num_games / total_wall * 3600 if total_wall > 0 else 0.0
        print(
            f"batched self-play: {num_games} games in {total_wall:.1f}s "
            f"({games_per_hour:.1f} games/hour)"
        )

    return summaries
