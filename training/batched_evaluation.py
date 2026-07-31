"""Batched checkpoint-vs-checkpoint / checkpoint-vs-random evaluation.

training/evaluation.py's serial evaluation is slow for the same reason
Stage 3's serial self-play was: one network call per leaf, batch size
1. Self-play fixed this with `BatchedMCTS` (training/batched_mcts.py),
which assumes one shared network drives every tree in a batch — true
for self-play (a game plays itself), but false for evaluation (a game
alternates between two *different* "players": new vs. old checkpoint,
or a checkpoint vs. the random baseline).

The fix here isn't a new search algorithm — it's splitting each round's
batch by which player actually needs to move. At any point across N
concurrent evaluation games, whichever color is to move in each game
determines which network must evaluate that position. Group the active
games by "which network is needed this round" (e.g. games where it's
the *new* checkpoint's turn vs. games where it's the *old* checkpoint's
turn) and run one `BatchedMCTS.run_batch` call per group — each group
gets the normal batching benefit among its own members, and the two
groups never need to share a forward pass (different weights, so
there's nothing to batch across them). A "random" resource skips
BatchedMCTS entirely for its games, since `RandomAgent` doesn't need
the network at all (essentially free compared to a search).

This reuses `BatchedMCTS` unchanged — same PUCT selection, same virtual
loss, same everything — just called once per network-group per round
instead of once per round.
"""

from dataclasses import dataclass

import chess
import numpy as np
import torch

from agents.random_agent import RandomAgent
from engine.chess_env import ChessEnv
from training.batched_mcts import BatchedMCTS
from training.mcts import select_action
from training.network import PolicyValueNet

RANDOM = "random"  # sentinel resource kind: moves are sampled directly, no MCTS/network involved


@dataclass
class _GameSpec:
    white_kind: str
    black_kind: str


def _play_concurrent_games(
    game_specs: list[_GameSpec],
    networks: dict[str, PolicyValueNet],
    device: torch.device,
    num_simulations: int,
    c_puct: float,
    claim_draw: bool,
    max_plies: int,
    leaves_per_tree_per_round: int,
    random_agent: RandomAgent | None,
    initial_fen: str | None,
) -> tuple[list[str], list[ChessEnv]]:
    """Play `len(game_specs)` games concurrently. Each ply-round,
    partitions the still-active games by which resource kind
    (a network, or RANDOM) needs to move in each of them, and runs one
    batched MCTS call per network group — no exploration noise/
    temperature (`add_dirichlet_noise` stays False, `select_action`
    uses temperature=0), since evaluation measures strength, not
    self-play exploration. Returns ("white"/"black"/"draw" per game,
    the finished ChessEnv per game) in `game_specs` order."""
    mcts_by_kind = {
        kind: BatchedMCTS(
            net, device=device, c_puct=c_puct, claim_draw=claim_draw,
            leaves_per_tree_per_round=leaves_per_tree_per_round,
        )
        for kind, net in networks.items()
    }

    envs = [ChessEnv(claim_draw=claim_draw) for _ in game_specs]
    reset_options = {"fen": initial_fen} if initial_fen else None
    for env in envs:
        env.reset(options=reset_options)

    plies = [0] * len(game_specs)
    terminated = [False] * len(game_specs)
    infos: list[dict | None] = [None] * len(game_specs)
    active = list(range(len(game_specs)))

    while active:
        groups: dict[str, list[int]] = {}
        for i in active:
            spec = game_specs[i]
            kind = spec.white_kind if envs[i].current_player() == chess.WHITE else spec.black_kind
            groups.setdefault(kind, []).append(i)

        actions: dict[int, int] = {}
        for kind, indices in groups.items():
            if kind == RANDOM:
                for i in indices:
                    actions[i] = random_agent.select_action(envs[i].action_mask())
            else:
                boards = [envs[i].board for i in indices]
                roots = mcts_by_kind[kind].run_batch(boards, num_simulations)
                for i, root in zip(indices, roots):
                    actions[i] = select_action(root, temperature=0.0)

        still_active = []
        for i in active:
            _obs, _reward, term, trunc, info = envs[i].step(actions[i])
            plies[i] += 1
            infos[i] = info
            if term or trunc or plies[i] >= max_plies:
                terminated[i] = term
            else:
                still_active.append(i)
        active = still_active

    results = []
    for i in range(len(game_specs)):
        if not terminated[i]:
            results.append("draw")  # safety cap hit — matches serial play_game_between's convention
            continue
        outcome = infos[i]["outcome"]
        if outcome.winner is None:
            results.append("draw")
        else:
            results.append("white" if outcome.winner else "black")

    return results, envs


def run_batched_evaluate_checkpoints(
    new_network: PolicyValueNet,
    old_network: PolicyValueNet,
    device: torch.device,
    num_games: int,
    num_simulations: int,
    c_puct: float = 1.5,
    claim_draw: bool = True,
    max_plies: int = 600,
    leaves_per_tree_per_round: int = 4,
    initial_fen: str | None = None,
) -> dict:
    """Batched drop-in replacement for evaluation.py::evaluate_checkpoints
    — same signature/semantics/return shape, `num_games` games run
    concurrently instead of one at a time. Colors alternate the same
    way: game 0 has new=white, game 1 has new=black, etc."""
    specs = [
        _GameSpec(white_kind="new", black_kind="old") if i % 2 == 0
        else _GameSpec(white_kind="old", black_kind="new")
        for i in range(num_games)
    ]

    results, _envs = _play_concurrent_games(
        specs, {"new": new_network, "old": old_network}, device, num_simulations,
        c_puct, claim_draw, max_plies, leaves_per_tree_per_round,
        random_agent=None, initial_fen=initial_fen,
    )

    new_wins = old_wins = draws = 0
    for i, result in enumerate(results):
        new_plays_white = i % 2 == 0
        if result == "draw":
            draws += 1
        elif (result == "white") == new_plays_white:
            new_wins += 1
        else:
            old_wins += 1

    return {
        "games": num_games,
        "new_wins": new_wins,
        "old_wins": old_wins,
        "draws": draws,
        "new_win_rate": new_wins / num_games,
        "old_win_rate": old_wins / num_games,
        "draw_rate": draws / num_games,
    }


def run_batched_evaluate_against_random(
    network: PolicyValueNet,
    device: torch.device,
    num_games: int,
    num_simulations: int,
    rng: np.random.Generator | None = None,
    c_puct: float = 1.5,
    claim_draw: bool = True,
    max_plies: int = 600,
    leaves_per_tree_per_round: int = 4,
    initial_fen: str | None = None,
) -> dict:
    """Batched drop-in replacement for evaluation.py::evaluate_against_random."""
    rng = rng or np.random.default_rng()
    random_agent = RandomAgent(seed=int(rng.integers(0, 2**31 - 1)))

    specs = [
        _GameSpec(white_kind="network", black_kind=RANDOM) if i % 2 == 0
        else _GameSpec(white_kind=RANDOM, black_kind="network")
        for i in range(num_games)
    ]

    results, _envs = _play_concurrent_games(
        specs, {"network": network}, device, num_simulations,
        c_puct, claim_draw, max_plies, leaves_per_tree_per_round,
        random_agent=random_agent, initial_fen=initial_fen,
    )

    network_wins = random_wins = draws = 0
    for i, result in enumerate(results):
        network_plays_white = i % 2 == 0
        if result == "draw":
            draws += 1
        elif (result == "white") == network_plays_white:
            network_wins += 1
        else:
            random_wins += 1

    return {
        "games": num_games,
        "network_wins": network_wins,
        "random_wins": random_wins,
        "draws": draws,
        "network_win_rate": network_wins / num_games,
    }
