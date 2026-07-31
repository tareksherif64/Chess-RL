"""Baseline agent that picks uniformly among legal actions.

Used to validate ChessEnv end-to-end (no illegal moves, correct
termination) before any learning code exists. Also a useful opponent
baseline once training starts.
"""

import numpy as np


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, action_mask: np.ndarray) -> int:
        """Sample uniformly among indices where action_mask is nonzero."""
        legal_indices = np.flatnonzero(action_mask)
        if legal_indices.size == 0:
            raise ValueError("No legal actions available (game should be over).")
        return int(self.rng.choice(legal_indices))
