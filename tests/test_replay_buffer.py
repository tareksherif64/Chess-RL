"""ReplayBuffer: ring-buffer eviction, sampling shapes/dtypes, and
save/load round-trips."""

import numpy as np
import pytest

from training.replay_buffer import ReplayBuffer


def _fake_example(i: int):
    board = np.full((8, 8, 18), fill_value=i, dtype=np.float32)
    policy = np.zeros(4672, dtype=np.float32)
    policy[i % 4672] = 1.0
    value = float(i)
    return board, policy, value


def test_add_and_sample_shapes_and_dtypes():
    buf = ReplayBuffer(capacity=10)
    for i in range(4):
        buf.add(*_fake_example(i))

    assert len(buf) == 4
    boards, policies, values = buf.sample(batch_size=3, rng=np.random.default_rng(0))
    assert boards.shape == (3, 8, 8, 18)
    assert policies.shape == (3, 4672)
    assert values.shape == (3,)
    assert boards.dtype == np.float32
    assert policies.dtype == np.float32
    assert values.dtype == np.float32


def test_sample_from_empty_buffer_raises():
    buf = ReplayBuffer(capacity=5)
    with pytest.raises(ValueError):
        buf.sample(batch_size=1)


def test_ring_buffer_evicts_oldest_when_full():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):  # 0,1,2 fill it; 3,4 wrap around and evict 0,1
        buf.add(*_fake_example(i))

    assert len(buf) == 3
    # White-box check of internal storage (this test is specifically about
    # the ring-buffer wraparound invariant, not just sample()'s API).
    remaining = set(int(v) for v in buf._values[: len(buf)])
    assert remaining == {2, 3, 4}
    for b, v in zip(buf._boards[: len(buf)], buf._values[: len(buf)]):
        assert np.all(b == v)


def test_add_game_adds_every_example():
    from training.replay_buffer import ReplayExample

    buf = ReplayBuffer(capacity=10)
    examples = [ReplayExample(*_fake_example(i)) for i in range(3)]
    buf.add_game(examples)
    assert len(buf) == 3


def test_save_and_load_round_trip(tmp_path):
    buf = ReplayBuffer(capacity=10)
    for i in range(4):
        buf.add(*_fake_example(i))

    path = tmp_path / "buffer.npz"
    buf.save(path)
    assert path.exists()

    loaded = ReplayBuffer.load(path)
    assert len(loaded) == len(buf)
    boards, policies, values = loaded.sample(batch_size=4, rng=np.random.default_rng(0))
    orig_boards, orig_policies, orig_values = buf.sample(batch_size=4, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(boards, orig_boards)
    np.testing.assert_array_equal(policies, orig_policies)
    np.testing.assert_array_equal(values, orig_values)


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)
