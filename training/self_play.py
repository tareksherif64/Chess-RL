"""Self-play: network+MCTS plays a full game against itself, recording
(board, MCTS-improved policy, outcome) for every position, for
training to sample from later (Stage 4).
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

from engine.chess_env import ChessEnv
from training.mcts import MCTS, visit_count_policy
from training.network import PolicyValueNet
from training.replay_buffer import ReplayBuffer, ReplayExample


@dataclass
class SelfPlayConfig:
    num_simulations: int = 100
    c_puct: float = 1.5
    # Move-selection temperature: sampling proportional to
    # visit_count^(1/temperature) for the first `temperature_threshold_plies`
    # plies (exploration), then greedy argmax (temperature effectively 0)
    # for the rest of the game — same schedule AlphaZero uses, and the
    # *same* temperature-scaled distribution is stored as the training
    # target `pi` (not a separately-computed constant-temperature one),
    # matching the original paper.
    temperature: float = 1.0
    temperature_threshold_plies: int = 30
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    claim_draw: bool = True
    # Safety net only: real games terminate well before this (Phase 1
    # random-vs-random self-play topped out under 500 plies; see
    # docs/self_play.md). A game that somehow hits this cap has no
    # known outcome to label examples with, so it's discarded rather
    # than recorded with a guessed value.
    max_plies: int = 600


def play_self_play_game(
    network: PolicyValueNet,
    device: torch.device,
    config: SelfPlayConfig,
    rng: np.random.Generator | None = None,
    initial_fen: str | None = None,
) -> tuple[list[ReplayExample], dict]:
    """`initial_fen` overrides the standard starting position — mainly
    for tests that need a short, deterministic game rather than a full
    ~40-80 ply one."""
    rng = rng or np.random.default_rng()
    env = ChessEnv(claim_draw=config.claim_draw)
    mcts = MCTS(network, device=device, c_puct=config.c_puct, claim_draw=config.claim_draw)

    reset_options = {"fen": initial_fen} if initial_fen else None
    obs, info = env.reset(options=reset_options)
    pending: list[tuple[np.ndarray, np.ndarray, bool]] = []  # (board, policy, mover)
    terminated = truncated = False
    ply = 0

    while not (terminated or truncated) and ply < config.max_plies:
        temperature = (
            config.temperature if ply < config.temperature_threshold_plies else 0.0
        )
        mover = env.current_player()

        root = mcts.run(
            env.board,
            config.num_simulations,
            add_dirichlet_noise=True,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            rng=rng,
        )
        policy_target = visit_count_policy(root, temperature=temperature)
        action = sample_action(policy_target, temperature, rng)

        pending.append((obs, policy_target, mover))
        obs, _reward, terminated, truncated, info = env.step(action)
        ply += 1

    if not terminated:
        # Hit max_plies without a real chess-rules termination: no
        # ground-truth outcome exists, so don't fabricate one.
        return [], {"plies": ply, "result": "*", "termination": "TRUNCATED_SAFETY_CAP", "discarded": True}

    outcome = info["outcome"]
    winner = outcome.winner  # True=white, False=black, None=draw

    examples = [
        ReplayExample(board=board, policy=policy, value=outcome_value(winner, mover))
        for board, policy, mover in pending
    ]

    summary = {
        "plies": ply,
        "result": outcome.result(),
        "termination": outcome.termination.name,
        "discarded": False,
    }
    return examples, summary


def run_self_play(
    network: PolicyValueNet,
    device: torch.device,
    buffer: ReplayBuffer,
    config: SelfPlayConfig,
    num_games: int,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
    initial_fen: str | None = None,
) -> list[dict]:
    """Play `num_games` self-play games, add every non-discarded game's
    examples to `buffer`, and return per-game summaries (also used by
    scripts/run_self_play.py for logging). `initial_fen` is mainly for
    fast tests — real self-play always starts from the standard position."""
    rng = rng or np.random.default_rng()
    summaries = []

    for game_idx in range(num_games):
        start = time.perf_counter()
        examples, summary = play_self_play_game(
            network, device, config, rng=rng, initial_fen=initial_fen
        )
        summary["wall_seconds"] = time.perf_counter() - start
        summary["examples"] = len(examples)

        if examples:
            buffer.add_game(examples)

        summaries.append(summary)
        if verbose:
            print(
                f"game {game_idx + 1}/{num_games}: "
                f"result={summary['result']} termination={summary['termination']} "
                f"plies={summary['plies']} examples={summary['examples']} "
                f"time={summary['wall_seconds']:.1f}s"
            )

    return summaries


def sample_action(policy: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature == 0:
        return int(np.argmax(policy))
    return int(rng.choice(len(policy), p=policy))


def outcome_value(winner: bool | None, mover: bool) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == mover else -1.0
