---
module: training
depends_on: [engine, agents, torch (CUDA build required)]
depended_on_by: []
---

# training/

Phase 2 of the project — AlphaZero-style self-play RL on top of the
Phase 1 `ChessEnv`. Built in stages; see each stage's own doc note as
it lands:

- **Stage 1 (done):** policy/value network — see `docs/network.md`
  (`network.py`, `tensors.py`, `device.py`).
- **Stage 2:** MCTS guided by the network — not started.
- **Stage 3:** self-play data generation / replay buffer — not started.
- **Stage 4:** training loop + checkpointing + logging — not started.

## GPU status — resolved

Previously flagged: the installed `torch` was CPU-only (`2.9.1+cpu`).
This is now resolved — `torch 2.6.0+cu124` is installed and confirmed
working on the RTX 4060 (`torch.cuda.is_available() == True`,
`torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4060 Laptop GPU"`).
`training/device.py::resolve_device()` is the single place training
code asks for a device, and it raises rather than silently falling back
to CPU if that ever regresses — see `docs/network.md`.

## Expected dependencies

- `engine.chess_env.ChessEnv` for the environment.
- `engine.encoding` constants (`ACTION_SPACE_SIZE = 4672`, observation
  shape `(8, 8, 18)`) for network input/output layer sizing.
- `agents` baseline(s) as an opponent/sanity-check during early training.
- Observations are already `float32` numpy arrays convertible to torch
  tensors via `torch.from_numpy(obs)` with no dtype cast — this was a
  deliberate constraint on `engine/encoding.py` from Phase 1, and
  `training/tensors.py` builds directly on it.
