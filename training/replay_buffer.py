"""Fixed-capacity ring buffer of self-play training examples.

Preallocated numpy arrays (not a deque of Python objects) so add() and
sample() are O(1)/O(batch_size) regardless of buffer size, and old
games are evicted automatically once the buffer fills — the same
structure Stage 4's training loop will sample minibatches from.

Capacity has no default on purpose: how much self-play history to keep
in memory is a real tuning knob (memory footprint vs. how "stale" the
oldest kept games are allowed to get), worth choosing deliberately per
call site rather than inheriting a guessed default. A small sanity-check
self-play run and a real multi-hour run should not accidentally share
the same buffer size.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np

from engine.encoding import ACTION_SPACE_SIZE


class ReplayExample(NamedTuple):
    board: np.ndarray  # (8, 8, 18) float32
    policy: np.ndarray  # (4672,) float32, MCTS visit-count distribution
    value: float  # scalar in [-1, 1], outcome from this board's mover's perspective


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        board_shape: tuple[int, int, int] = (8, 8, 18),
        num_actions: int = ACTION_SPACE_SIZE,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.board_shape = board_shape
        self.num_actions = num_actions

        self._boards = np.zeros((capacity, *board_shape), dtype=np.float32)
        self._policies = np.zeros((capacity, num_actions), dtype=np.float32)
        self._values = np.zeros((capacity,), dtype=np.float32)
        self._size = 0
        self._write_idx = 0

    def __len__(self) -> int:
        return self._size

    def add(self, board: np.ndarray, policy: np.ndarray, value: float) -> None:
        self._boards[self._write_idx] = board
        self._policies[self._write_idx] = policy
        self._values[self._write_idx] = value
        self._write_idx = (self._write_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_game(self, examples: list[ReplayExample]) -> None:
        for example in examples:
            self.add(example.board, example.policy, example.value)

    def sample(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self._size, size=batch_size)
        return self._boards[idx], self._policies[idx], self._values[idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            boards=self._boards[: self._size],
            policies=self._policies[: self._size],
            values=self._values[: self._size],
            capacity=np.array(self.capacity),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        data = np.load(path)
        boards, policies, values = data["boards"], data["policies"], data["values"]
        capacity = max(int(data["capacity"]), len(boards))
        buffer = cls(capacity=capacity, board_shape=boards.shape[1:], num_actions=policies.shape[1])
        buffer._boards[: len(boards)] = boards
        buffer._policies[: len(policies)] = policies
        buffer._values[: len(values)] = values
        buffer._size = len(boards)
        buffer._write_idx = len(boards) % capacity
        return buffer
