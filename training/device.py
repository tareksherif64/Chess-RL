"""Single source of truth for which torch device training code runs on.

Training is only worth doing on the GPU here (self-play + MCTS is CPU-bound
enough already; CPU tensor training on top would be painfully slow), so a
missing CUDA device is treated as a hard error, not something to quietly
paper over by falling back to CPU. Silent CPU fallback is exactly the kind
of bug that wastes hours of "training" that's actually running 10-50x
slower than intended without any visible symptom.
"""

import torch


def resolve_device(require_cuda: bool = True) -> torch.device:
    """Return the torch device training/inference code should use.

    require_cuda=True (default): raise RuntimeError if CUDA isn't
    available, rather than silently returning a CPU device.
    require_cuda=False: explicit opt-in to CPU, for shape/correctness
    unit tests that don't care about throughput.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError(
            "CUDA is not available in this torch install, but training code "
            "requires it (no silent CPU fallback). Check `torch.cuda.is_available()` "
            "and that the CUDA build of torch is installed — see docs/training.md."
        )
    return torch.device("cpu")
