"""Stage 1 tests: network output shapes/ranges, policy-head alignment
with the engine's action encoding (critical — a mismatch here would
silently corrupt every future training step), legal-move masking, and
that CUDA is actually used when required (no silent CPU fallback)."""

import chess
import numpy as np
import pytest
import torch

from engine.chess_env import ChessEnv
from engine.encoding import ACTION_SPACE_SIZE
from training.device import resolve_device
from training.network import PolicyValueNet, masked_softmax
from training.tensors import mask_to_tensor, obs_to_tensor


def _sample_boards(n: int) -> np.ndarray:
    """A handful of distinct board encodings from real games (not just
    the start position), stacked into an (n, 8, 8, 18) batch."""
    rng = np.random.default_rng(0)
    boards = []
    board = chess.Board()
    from engine.encoding import encode_board

    boards.append(encode_board(board))
    while len(boards) < n:
        if board.is_game_over():
            board = chess.Board()
            continue
        legal = list(board.legal_moves)
        board.push(legal[rng.integers(len(legal))])
        boards.append(encode_board(board))
    return np.stack(boards[:n]).astype(np.float32)


def test_forward_output_shapes_and_value_range():
    net = PolicyValueNet()
    net.eval()
    obs = _sample_boards(4)
    x = obs_to_tensor(obs, torch.device("cpu"))
    assert x.shape == (4, 18, 8, 8)

    with torch.no_grad():
        policy_logits, value = net(x)

    assert policy_logits.shape == (4, ACTION_SPACE_SIZE)
    assert value.shape == (4,)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)
    assert not torch.isnan(policy_logits).any()
    assert not torch.isnan(value).any()


def test_policy_head_matches_action_encoding_layout():
    """The policy head is a conv producing (N, 73, rank, file), permuted
    and flattened to line up with encode_move's from_square*73+plane
    indexing (see docs/action_space.md). Verify that alignment directly
    against the raw pre-flatten conv output, for random (square, plane)
    samples, rather than trusting the reshape math by inspection alone."""
    net = PolicyValueNet()
    net.eval()

    captured = {}

    def hook(_module, _input, output):
        captured["policy_conv_out"] = output.detach()

    net.policy_conv.register_forward_hook(hook)

    obs = _sample_boards(2)
    x = obs_to_tensor(obs, torch.device("cpu"))
    with torch.no_grad():
        policy_logits, _value = net(x)

    raw = captured["policy_conv_out"]  # (N, 73, rank, file)
    assert raw.shape == (2, 73, 8, 8)

    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(2))
        from_square = int(rng.integers(64))
        plane = int(rng.integers(73))
        rank, file = divmod(from_square, 8)
        action = from_square * 73 + plane

        expected = raw[n, plane, rank, file]
        actual = policy_logits[n, action]
        assert torch.equal(expected, actual), (n, from_square, plane, action)


def test_masked_softmax_zero_probability_on_illegal_actions():
    net = PolicyValueNet()
    net.eval()

    env = ChessEnv()
    obs, _info = env.reset()
    mask_np = env.action_mask()

    device = torch.device("cpu")
    x = obs_to_tensor(obs, device)
    mask = mask_to_tensor(mask_np, device)

    with torch.no_grad():
        policy_logits, _value = net(x)
    probs = masked_softmax(policy_logits, mask)

    assert probs.shape == (1, ACTION_SPACE_SIZE)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)

    illegal = mask == 0
    assert torch.all(probs[illegal] == 0.0)
    assert torch.all(probs[~illegal] >= 0.0)

    legal_count = int(mask_np.sum())
    assert legal_count == len(list(env.board.legal_moves))
    assert int((probs[0] > 0).sum()) <= legal_count


def test_network_is_reasonably_small():
    net = PolicyValueNet()
    total_params = sum(p.numel() for p in net.parameters())
    # ~493K params / ~2MB fp32 at the default channels=64, blocks=6 —
    # see docs/network.md for the size/depth tradeoff. Guard against
    # accidentally growing this by an order of magnitude unnoticed.
    assert total_params < 2_000_000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")
def test_network_and_tensors_actually_use_cuda():
    device = resolve_device(require_cuda=True)
    assert device.type == "cuda"

    net = PolicyValueNet().to(device)
    obs = _sample_boards(2)
    x = obs_to_tensor(obs, device)
    assert x.device.type == "cuda"

    policy_logits, value = net(x)
    assert policy_logits.device.type == "cuda"
    assert value.device.type == "cuda"
    assert next(net.parameters()).device.type == "cuda"


def test_resolve_device_raises_without_silently_falling_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device(require_cuda=True)
    # Explicit opt-in still works.
    assert resolve_device(require_cuda=False).type == "cpu"
