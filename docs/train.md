---
module: training (train.py, checkpoint.py, logger.py)
depends_on: [engine, training/network.py, training/mcts.py, training/self_play.py, training/replay_buffer.py]
depended_on_by: []
---

# training/train.py, training/checkpoint.py, training/logger.py

Stage 4 of Phase 2, and the last stage of the planned build: sample
from the Stage 3 replay buffer, train the Stage 1 network on it, and
tie everything (self-play -> train -> checkpoint -> log) into one
repeatable iteration loop.

## Files

- **`train.py`** — `TrainConfig`, `compute_loss()`, `train_steps()`
  (a fixed number of minibatch gradient updates), `run_training_loop()`
  (the full per-iteration cycle: self-play, train, checkpoint, log).
- **`checkpoint.py`** — `save_checkpoint()` / `load_checkpoint()`:
  model weights + optimizer state + iteration number.
- **`logger.py`** — `TrainingLogger`: append-only CSV, one row per
  iteration.

Entry point: `scripts/train.py`.

## Loss

```
policy_loss = -(pi_target * log_softmax(raw_policy_logits)).sum(-1).mean()
value_loss  = mse_loss(predicted_value, outcome_target)
total_loss  = policy_loss + value_loss
```

The policy loss is computed against the network's **raw, unmasked**
logits — not `masked_softmax` from Stage 1. `pi_target` (the MCTS
visit-count distribution from Stage 3) already has exactly zero mass
on every action that wasn't a legal move at that position, so it's
already teaching the network to prefer legal, search-favored moves
without needing to additionally mask the network's own output during
training. Masking is purely an MCTS/inference-time concern (you can
only *play* a legal move); baking it into the loss as well would hide
information — the network is supposed to learn on its own, from this
signal, to assign low probability to illegal moves in the first place.
This matches the original AlphaZero paper's loss formulation.

No separate L2 term is added manually — `weight_decay` on the Adam
optimizer serves the same regularization purpose the paper's explicit
`c||theta||^2` term does, and is the standard PyTorch idiom rather than
computing it by hand.

## Optimizer: Adam, not AlphaZero's SGD+momentum+schedule

AlphaZero used plain SGD with momentum and a hand-tuned learning-rate
schedule (drops at specific training-step milestones), which needs
real tuning effort to get right and was designed for a much longer,
much larger-scale training run than anything happening here yet. Adam
(`lr=1e-3, weight_decay=1e-4` by default) adapts its own per-parameter
step size and is far more forgiving of an untuned learning rate — the
right choice while we're still validating that the pipeline works at
all. Worth revisiting (SGD+schedule can generalize better at scale)
once we're past small-scale validation and into real training —
exactly the kind of knob to set deliberately in the scale-up
conversation, not guess at now.

## Checkpointing

`save_checkpoint()` writes model weights, optimizer state, and the
iteration number to a single `.pt` file. Optimizer state matters, not
just weights: Adam tracks per-parameter moving averages of gradient
mean/variance, and resuming with a *fresh* optimizer would show up as
a visible loss spike while those estimates re-warm from zero — the
kind of thing that looks like a bug in a resumed multi-hour run if you
don't know to expect it. `load_checkpoint()` uses `map_location` so a
checkpoint saved on CUDA loads correctly regardless of what device
you're resuming on, and `weights_only=True` (torch's safer unpickling
path) since there's no reason not to use it even for a file we only
ever produce ourselves.

`run_training_loop`'s `checkpoint_every_iterations` also (optionally)
saves the replay buffer alongside the checkpoint (`TrainConfig.buffer_path`)
— resuming a long run needs *both* the learned weights and the
accumulated self-play history back, or a chunk of expensive-to-regenerate
game data is silently lost on every restart.

## Logging

Plain CSV via `TrainingLogger`, one row per iteration:
`iteration, buffer_size, games_played, games_discarded,
self_play_seconds, train_seconds, games_per_hour, policy_loss,
value_loss, total_loss`. No new dependency (no tensorboard/wandb) —
appropriate for a single-machine, early-stage project; the file loads
trivially into pandas/Excel/a plotting script once there's enough real
training history for a loss curve to be worth looking at. `policy_loss`/
`value_loss`/`total_loss` are `None` on any iteration where training
was skipped (buffer below `min_buffer_size`), so a monitoring script
can tell "not trained yet" apart from "trained but happened to have a
near-zero loss."

## Why training is skipped below `min_buffer_size`

Early in a run, the buffer might hold examples from only one or two
games — training on that is training on a handful of highly
correlated positions (same game, same network, same few lines),
which risks the network overfitting noise from a tiny, unrepresentative
sample rather than learning anything general. `run_training_loop`
still runs self-play and fills the buffer on a skipped iteration, it
just doesn't take gradient steps until there's enough accumulated
variety — worth tuning together at the scale-up conversation, same as
batch size and simulation count.

## Testing philosophy: can it overfit?

The core Stage 4 correctness test,
`tests/test_train.py::test_train_steps_can_overfit_a_tiny_fixed_batch`,
trains on 8 fixed, non-degenerate board positions for 500 steps and
checks loss drops by >70% (loose relative bound rather than a tight
absolute one, to avoid flakiness from Adam's normal step-to-step noise
on such a small/sharp loss surface — verified empirically this
consistently lands around a 10x reduction). This is the standard "can
the training code even overfit a trivial dataset" sanity check used to
catch a broken loss/backward/optimizer wiring — the same role the
mate-in-1 test played for Stage 2's MCTS and the mate-in-1 self-play
test played for Stage 3.

## Small-scale end-to-end confirmation run

Per the plan, ran the *full* loop (self-play -> train -> checkpoint)
before considering Stage 4 done, not just unit tests:

```
python scripts/train.py --iterations 2 --games-per-iteration 2 \
    --simulations 20 --train-steps 20 --batch-size 16 --min-buffer-size 16 \
    --buffer-path data/train_stage4_buffer.npz
```

```
game 1/2: result=1/2-1/2 termination=FIFTY_MOVES             plies=372 examples=372 time=60.7s
game 2/2: result=1/2-1/2 termination=INSUFFICIENT_MATERIAL   plies=414 examples=414 time=72.0s
iteration 0: policy_loss=7.7454 value_loss=0.0126 total_loss=7.7580
iteration 0: saved checkpoint to checkpoints/iter_000000.pt

game 1/2: result=1/2-1/2 termination=THREEFOLD_REPETITION    plies=58  examples=58  time=9.1s
game 2/2: result=1/2-1/2 termination=THREEFOLD_REPETITION    plies=246 examples=246 time=37.9s
iteration 1: policy_loss=5.9916 value_loss=0.0008 total_loss=5.9924
iteration 1: saved checkpoint to checkpoints/iter_000001.pt

buffer_size: 786 -> 1090   games_per_hour: 54.3 -> 153.0 (20 sims/move, much faster than Stage 3's 100-sim benchmark)
```

Verified independently after the run, not just from the printed log:
loaded `checkpoints/iter_000001.pt` fresh (new network/optimizer
objects) and confirmed it restores to iteration 1 with real (non-random-init)
weights; loaded `data/train_stage4_buffer.npz` and confirmed all 1090
examples have correct shapes, zero NaNs, and every policy row sums to
1.0. Both the training loop's own logging and an independent reload
agree.

`policy_loss` dropped 7.75 -> 5.99 and `value_loss` 0.0126 -> 0.0008
across the two iterations — not a meaningful trend from 2 data points
(and misleading to read too much into: iteration 1's training batch
draws from a *different*, now-larger buffer than iteration 0's), but
it confirms the mechanics — loss is finite, moving, and not NaN/diverging,
gradients are flowing, checkpoints capture real progress. The real
"does this improve with training" question needs many more iterations
than 2 to answer, which is exactly why this was kept small and why the
next conversation is about scaling up, not a conclusion drawn here.
