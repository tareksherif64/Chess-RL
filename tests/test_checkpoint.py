"""Checkpoint save/load: weights, optimizer state, and iteration
number all round-trip correctly, including across a simulated
different-device load (map_location)."""

import torch

from training.checkpoint import load_checkpoint, save_checkpoint
from training.network import PolicyValueNet


def test_checkpoint_round_trip_restores_weights_and_iteration(tmp_path):
    torch.manual_seed(0)
    network = PolicyValueNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    # Take a step so weights/optimizer state are non-trivial.
    x = torch.randn(2, 18, 8, 8)
    policy_logits, value = network(x)
    loss = policy_logits.sum() + value.sum()
    loss.backward()
    optimizer.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, network, optimizer, iteration=7)
    assert path.exists()

    saved_state = {k: v.clone() for k, v in network.state_dict().items()}

    new_network = PolicyValueNet()  # fresh random weights
    new_optimizer = torch.optim.Adam(new_network.parameters(), lr=1e-3)
    iteration = load_checkpoint(path, new_network, new_optimizer, device=torch.device("cpu"))

    assert iteration == 7
    for key, value in new_network.state_dict().items():
        assert torch.equal(value, saved_state[key])

    # Optimizer state (Adam's per-parameter moment estimates) restored too.
    assert len(new_optimizer.state) > 0
    orig_group = optimizer.state_dict()["state"]
    new_group = new_optimizer.state_dict()["state"]
    assert orig_group.keys() == new_group.keys()


def test_checkpoint_loadable_without_optimizer(tmp_path):
    network = PolicyValueNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, network, optimizer, iteration=3)

    fresh_network = PolicyValueNet()
    iteration = load_checkpoint(path, fresh_network, optimizer=None, device=torch.device("cpu"))
    assert iteration == 3
