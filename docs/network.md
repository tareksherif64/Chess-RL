---
module: training (network.py, tensors.py, device.py)
depends_on: [engine (encoding contract), torch (CUDA build)]
depended_on_by: [training/mcts.py (future), training/self_play.py (future), training/train.py (future)]
---

# training/network.py, training/tensors.py, training/device.py

Responsibility: the policy/value network itself, plus the small amount
of glue needed to move data onto it correctly. This is Stage 1 of the
Phase 2 RL system (network only — no search, no self-play, no training
loop yet).

## Files

- **`network.py`** — `PolicyValueNet(nn.Module)` and `masked_softmax()`.
- **`tensors.py`** — `obs_to_tensor` / `mask_to_tensor`: numpy (channels-last,
  as `engine/encoding.py` produces) → torch (channels-first, as
  `nn.Conv2d` expects), on a given device, with no dtype cast (input is
  already `float32`).
- **`device.py`** — `resolve_device(require_cuda=True)`. Raises loudly if
  CUDA isn't available instead of returning a CPU device — see "no
  silent fallback" below.

## Architecture

```
input (N, 18, 8, 8)
  -> stem: 3x3 conv(18->64) + BN + ReLU
  -> 6x ResidualBlock(64)         [conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> +skip -> ReLU]
  -> policy head: 1x1 conv(64->73) -> permute -> flatten -> (N, 4672) logits
  -> value head:  1x1 conv(64->8) -> BN -> ReLU -> flatten -> FC(512->64) -> ReLU -> FC(64->1) -> tanh -> (N,)
```

~493K parameters, ~2MB in fp32.

### Policy head shape, and why it's not a giant FC layer

The action encoding (`docs/action_space.md`) is already laid out as an
implicit `(8, 8, 73)` grid: `action = from_square * 73 + plane`, where
`from_square = rank*8 + file`. That means the policy head doesn't need a
`(flattened_trunk) -> 4672` fully-connected layer (which at a
64-channel, 8x8 trunk flattened to 4096 features would be ~19M
parameters on its own — more than the rest of the network combined).
Instead, a 1x1 conv maps the 64-channel trunk straight to 73 output
channels per square (`64*73 = 4672` weights), and reshaping
`(N, 73, rank, file) -> permute -> (N, rank, file, 73) -> flatten`
reproduces the `from_square*73+plane` ordering exactly — verified in
`tests/test_network.py::test_policy_head_matches_action_encoding_layout`
by hooking the conv's raw output and comparing it element-by-element
against the flattened logits at random `(square, plane)` pairs. This
was worth a dedicated test: a silent transpose/reshape bug here
wouldn't crash anything, it would just train the network against the
wrong action indices forever.

### Size/depth tradeoff: 64 channels, 6 residual blocks

AlphaZero's original chess network used 256 channels and 19-40 residual
blocks — tuned for a distributed cluster running many thousands of
self-play games in parallel. We have one RTX 4060 running self-play and
training serially. At this stage the bottleneck isn't network capacity,
it's how many self-play games we can generate and iterate on per hour,
so a small, fast network that lets us validate the whole pipeline
(Stages 2-4) quickly matters more than squeezing out extra playing
strength from a bigger model we can barely afford to run.

64 channels / 6 blocks was picked as a "small but not toy" starting
point — deep enough to have real spatial reasoning capacity (each block
adds a 3x3 receptive field ring), small enough that a forward+backward
pass is cheap and MCTS (which calls the network once per simulated leaf
node, potentially hundreds of times per real move) doesn't become the
speed bottleneck. `PolicyValueNet(channels=..., num_blocks=...)` are
constructor args specifically so this can be scaled up later (e.g. to
128/10 or beyond) once we know the pipeline works and want more
strength, without changing any other code — `tests/test_network.py`
guards the *default* config's size (`< 2M` params) so that increase has
to be a deliberate choice, not an accident.

### Value head perspective

Output is a scalar in `[-1, 1]` (via `tanh`), representing the
estimated win probability for whichever side is to move, matching
`ChessEnv`'s own terminal-reward convention. Since the board encoding
is absolute (white-relative, not flipped per side to move — see
`docs/engine.md`), the network has to learn to use the side-to-move
plane to know whose perspective to evaluate from, rather than getting
that for free from a canonical/mirrored input. Same tradeoff noted in
Phase 1: simpler to implement and debug, at the cost of not handing the
network a free symmetry. Still open to revisiting if training turns out
to need it.

## `masked_softmax`

Raw policy logits are unmasked (the network doesn't know which moves
are legal in a given position — masking is applied at the point of
use). `masked_softmax(logits, mask)` fills illegal-action logits with
`torch.finfo(dtype).min` before softmax, giving exactly zero probability
mass on illegal actions rather than a small-but-nonzero epsilon.
Assumes at least one legal action per row, true for any non-terminal
position — the network is never queried on terminal states (MCTS/
self-play stop at terminal nodes and use `ChessEnv`'s own outcome
instead of a network value estimate there).

## GPU: no silent CPU fallback

`resolve_device(require_cuda=True)` (the default) raises `RuntimeError`
if `torch.cuda.is_available()` is `False`, rather than quietly handing
back a CPU device. This was an explicit requirement: a silent fallback
would mean a multi-hour training run could end up running 10-50x slower
than intended on CPU with no visible symptom other than "this is taking
a long time." `require_cuda=False` is available as an explicit opt-in,
used only by shape/correctness unit tests that don't care about
throughput (most of `tests/test_network.py` runs on CPU for speed; one
test — `test_network_and_tensors_actually_use_cuda` — specifically
proves the network, its parameters, and input tensors all end up on
`cuda:0`, skipped only if no CUDA GPU is present).

Confirmed on this machine: `torch 2.6.0+cu124`, RTX 4060 Laptop GPU,
`torch.cuda.is_available() == True`.
