---
module: training (train.py hardening)
depends_on: [training/batched_self_play.py, training/batched_evaluation.py, training/checkpoint.py, training/logger.py]
depended_on_by: []
---

# Overnight-run hardening

Additions to `training/train.py::run_training_loop` specifically for
running unattended for hours with nobody able to intervene if
something goes wrong partway through. Nothing here changes correctness
of self-play, MCTS, or training — it's entirely about *what happens
when something fails*, and about giving accurate, complete numbers to
review the next morning.

## Multi-batch self-play: hit a game-count target without an untested concurrency jump

`games_per_iteration` (the total self-play games wanted per iteration,
e.g. ~65) and self-play *concurrency* (N — how many games run in one
batched call, e.g. 8, the benchmarked/swept config) are now separate
knobs. `TrainConfig.self_play_batch_size` sets N; if the target exceeds
it, `run_training_loop` runs back-to-back micro-batches of up to N
games until the target is reached (e.g. 65 games at N=8 → eight
batches of 8 plus one of 1), instead of launching a single N=65
concurrent batch — a concurrency level nobody has actually measured
(the config sweep only tested N=8 and N=16; see `docs/full_training_loop.md`).
`self_play_batch_size=None` (the default) runs everything as a single
batch, i.e. unchanged pre-hardening behavior — every existing test in
`tests/test_train.py` still passes with zero modifications, confirming
this is additive, not a behavior change for anyone not using the new
option.

## Crash isolation

An 8-hour run with no one watching can't afford to have one bad game,
one transient CUDA hiccup, or one full disk take down everything that
came before it. Every sub-step of an iteration is wrapped in
`_safe_call`, which runs the step, and on **any** exception, logs it
(full traceback to `TrainConfig.error_log_path`, a one-line summary to
stdout) and returns `None` instead of propagating:

- Each self-play micro-batch — a failure loses just that micro-batch's
  games (at most N, e.g. 8), not the whole iteration; the loop moves
  on to the next micro-batch.
- `train_steps` — a failure is treated exactly like the existing
  "buffer below `min_buffer_size`" case (`loss_stats = None`,
  vs-previous eval correctly skipped too, since there's no new
  checkpoint to compare).
- Both evaluation calls — a failure is treated exactly like the
  existing "not this iteration's cadence" case (`None` in that
  iteration's log row).
- Checkpoint and buffer saves.

On top of all of that, the **entire iteration body** is wrapped in a
final backstop `try/except` — anything that slips past the individual
`_safe_call` sites (state-dict cloning, bookkeeping, the logger itself)
is caught, logged, and the loop moves on to the next iteration rather
than dying. Verified directly, not just reasoned about:
`tests/test_train_hardening.py` injects failures at every one of these
points (a flaky self-play micro-batch, a broken `train_steps`, a
broken evaluation call, and — for the backstop specifically — a
broken logger call, chosen because it sits outside every individual
`_safe_call` site) and checks the run continues past each one with the
failure recorded in the error log.

**What this deliberately does *not* do**: isolate a single game's
failure *within* a concurrent batch (e.g. one of 8 games in a
micro-batch crashing while the other 7 continue). That would require
changes inside the already-tested, already-benchmarked
`BatchedMCTS`/`_play_concurrent_games` game loops themselves — real
surgery on code this run depends on being correct, undertaken right
before launching it unattended is exactly the wrong moment to take
that risk. Micro-batch-level isolation (lose at most N games, not the
whole night) was judged the right risk/robustness tradeoff instead —
worth revisiting if it turns out individual-game crashes are common in
practice, which nothing observed so far suggests.

## Resumability

Unaffected by any of the above, and still the right tool for a truly
unrecoverable failure (the whole process dying, not an exception the
hardening above already catches): `scripts/train.py --resume <checkpoint>`
loads weights + optimizer state + the iteration number
(`training/checkpoint.py`, Stage 4) and continues numbering from there.
Checkpointing every iteration (`checkpoint_every_iterations=1`, already
the default) means resuming after a crash loses at most one
iteration's progress.

## New logging

- **`iteration_seconds`** — direct wall-clock measurement around the
  *entire* iteration body (not a sum of the sub-step timings, which
  wouldn't capture bookkeeping/logging/checkpoint-save overhead) —
  what "how long did iteration N actually take" means without having
  to add up several CSV columns by hand.
- **`gpu_memory_allocated_mb`, `gpu_memory_reserved_mb`** — logged
  every iteration (`torch.cuda.memory_allocated`/`memory_reserved`),
  so a slow leak across many iterations shows up as a trend in the CSV
  rather than a mid-night OOM crash with no history to diagnose it
  from. `None` on CPU (not a fabricated 0 — genuinely not applicable).
- **`error_log_path`** (new `TrainConfig` field, defaults to
  `logs/train_errors.log` in `scripts/train.py`) — every caught
  failure's full traceback, kept separate from the structured CSV log
  so the CSV stays clean for plotting while nothing about a failure is
  lost.

Every existing CSV field (loss curves, games played, both evaluations'
win rates, buffer size) is unchanged — this is additive columns, not a
reworked log format.
