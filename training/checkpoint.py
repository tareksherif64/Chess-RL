"""Model/optimizer checkpointing, so a multi-hour self-play+training
run can be killed and resumed without losing learned weights or
optimizer momentum (resuming with a fresh optimizer state would cause
a visible hiccup in loss as Adam's moment estimates re-warm up)."""

from pathlib import Path

import torch

from training.network import PolicyValueNet


def save_checkpoint(
    path: str | Path,
    network: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    network: PolicyValueNet,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> int:
    """Loads weights (and optimizer state, if an optimizer is given)
    in place, mapped onto `device` regardless of what device the
    checkpoint was saved from. Returns the saved iteration number, so
    a resumed run continues numbering rather than restarting at 0."""
    # weights_only=True: this is always a locally self-produced
    # checkpoint (never a downloaded/untrusted file), but there's no
    # reason not to use torch's safer unpickling path anyway.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
