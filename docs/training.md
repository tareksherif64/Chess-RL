---
module: training
depends_on: [engine, agents, torch (CUDA build required)]
depended_on_by: []
---

# training/

Empty on purpose — this is the RL phase, not built yet.

## Known blocker for this phase

The installed `torch` in this environment is the **CPU-only build**
(`2.9.1+cpu`). The machine has an NVIDIA RTX 4060 (8GB) with a driver
supporting CUDA 13.2, so the GPU itself is not the problem — before any
training loop is written, `torch` needs to be reinstalled with a CUDA
build, e.g.:

```
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

(pick the `cuXXX` index matching a CUDA runtime <= the driver's 13.2
support window; check https://pytorch.org for the current recommended
index at install time).

## Expected dependencies once this phase starts

- `engine.chess_env.ChessEnv` for the environment.
- `engine.encoding` constants (`ACTION_SPACE_SIZE = 4672`,
  observation shape `(8, 8, 18)`) for network input/output layer sizing.
- `agents` baseline(s) as an opponent/sanity-check during early training.
- Observations are already `float32` numpy arrays convertible to torch
  tensors via `torch.from_numpy(obs)` with no dtype cast — this was a
  deliberate constraint on `engine/encoding.py` from the start, so this
  phase shouldn't hit data-format friction moving batches to CUDA.
