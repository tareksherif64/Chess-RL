"""AlphaZero-style policy/value network.

Input:  (N, 18, 8, 8) board tensor (see engine/encoding.py, training/tensors.py)
Output: policy_logits (N, 4672) raw logits over the action space,
        value (N,) scalar in [-1, 1] — estimated win probability for the
        side to move, from the same absolute-orientation perspective
        engine/encoding.py uses (the side-to-move plane in the input is
        how the network knows whose turn it is).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.encoding import ACTION_SPACE_SIZE, NUM_PLANES

POLICY_PLANES = 73  # must match engine/encoding.py's PLANES_PER_SQUARE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        num_blocks: int = 6,
        num_actions: int = ACTION_SPACE_SIZE,
    ):
        super().__init__()
        self.num_actions = num_actions

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # Policy head: 1x1 conv straight to 73 "move-type" planes, no
        # flatten-then-huge-FC-layer. This only works because engine's
        # action encoding is itself laid out as an (8, 8, 73) grid
        # (from_square = rank*8+file, action = from_square*73 + plane) —
        # see docs/action_space.md. Permuting (N,73,rank,file) to
        # (N,rank,file,73) and flattening reproduces that exact indexing,
        # verified in tests/test_network.py.
        self.policy_conv = nn.Conv2d(channels, POLICY_PLANES, kernel_size=1)

        # Value head: reduce to a few channels, then two small FC layers to a scalar.
        self.value_conv = nn.Conv2d(channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.res_blocks(x)

        policy = self.policy_conv(x)  # (N, 73, rank, file)
        policy = policy.permute(0, 2, 3, 1).contiguous()  # (N, rank, file, 73)
        policy_logits = policy.view(x.size(0), self.num_actions)  # (N, 4672)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.flatten(1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)  # (N,)

        return policy_logits, value


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Turn raw policy logits into a probability distribution with zero
    mass on illegal actions. `mask` is (num_actions,) or (N, num_actions)
    with 1.0 for legal actions. Assumes at least one legal action per row
    (true for any non-terminal position; the network is never queried on
    terminal states)."""
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand_as(logits)
    neg_fill = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(mask == 0, neg_fill)
    return F.softmax(masked_logits, dim=-1)
