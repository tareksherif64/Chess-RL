---
module: training (batched_mcts.py, batched_self_play.py)
depends_on: [engine, training/mcts.py, training/network.py, training/self_play.py]
depended_on_by: [scripts/benchmark_batched_self_play.py]
---

# training/batched_mcts.py, training/batched_self_play.py

A throughput fix, not a new capability: Stage 3's self-play measured
~0.75s/move at 100 simulations/move on the RTX 4060 (`docs/self_play.md`),
and profiling that showed the cost was dominated by per-call GPU
overhead (kernel launch, host↔device transfer) at batch size 1, not
actual compute — an ~493K-parameter network barely uses the GPU at
all for one board at a time. This module batches many board positions
into a single network forward pass instead, so that fixed overhead is
paid once per *batch* rather than once per *position*.

## The two batching axes

1. **N concurrent games.** Self-play already wants to generate many
   games; running them concurrently rather than one-after-another
   means, at each "round," every active game's tree needs exactly one
   network evaluation for whatever leaf it's currently exploring — so
   collecting those N leaves into one forward pass is free extra batch
   size that was already implicit in wanting N games' worth of data.
2. **Multiple in-flight leaves within one tree, via virtual loss.**
   Even with a modest N (8-16), batch size = N alone may not be large
   enough to make the GPU worth its overhead. Pulling more than one
   leaf per tree per round pushes batch size past N — but a single
   tree's PUCT selection is deterministic given its current stats, so
   selecting "the best leaf" twice in a row *before either has been
   evaluated* would just return the identical leaf both times; nothing
   about the tree changed between the two calls to justify a different
   answer.

## Virtual loss

The fix for axis 2: the instant a leaf is selected for (batched, not
yet completed) evaluation, pretend it was immediately evaluated and
lost — `apply_virtual_loss` in `batched_mcts.py` calls the exact same
`backup()` used everywhere else in the search, with a placeholder
value, so every node along that path gets a temporary, pessimistic
adjustment. A second, concurrent selection in the same tree now sees
that edge as less attractive (its PUCT score's `Q` term worsens) and
is steered toward a different leaf instead — this is what "multiple
in-flight leaf selections per tree without collisions" (the throughput
task's own phrasing) means in practice. Once the real (batched)
network evaluation returns, `revert_virtual_loss` exactly undoes the
placeholder adjustment (decrementing what `backup()` only ever
increments, subtracting back the same sign-flipped value contributions)
before the real value is backed up in its place.

**Net-effect invariant, and why it matters for correctness:** revert
then a real backup leaves every node's `(visit_count, value_sum)`
bit-identical to what a single ordinary `backup(real_value)` alone
would have produced — proven directly in
`tests/test_batched_mcts.py::test_virtual_loss_revert_matches_serial_backup`.
Virtual loss only ever affects *intermediate* tree state while an
evaluation is in flight; it leaves no trace once resolved.

**This does not make batched search bit-identical to serial search in
general** — while several leaves are simultaneously in-flight (virtual
loss applied but not yet really evaluated), a second leaf selection
within the same round sees temporarily-adjusted stats that serial,
one-at-a-time MCTS never would have seen, so the *sequence* of leaves
explored can differ. This is the expected, accepted tradeoff virtual
loss makes for parallelism (standard in AlphaZero/AlphaGo Zero-style
engines), not a bug. There is one case where the two *are* provably
identical, and it's directly tested:
`test_batched_run_reduces_to_serial_mcts_for_single_tree_one_leaf_per_round`
runs `BatchedMCTS` with one game and `leaves_per_tree_per_round=1` (so
no two leaves are ever simultaneously in flight — each is resolved
before the next is selected, exactly like serial `MCTS.run()`) and
checks the resulting tree matches the serial `MCTS.run()` tree
exactly: same visit counts, same values, same priors, for every node.

## Same move-selection logic and policy/value usage — literally, not just similarly

`training/mcts.py`'s `select_child` (PUCT), `backup`, `expand_children`,
`apply_dirichlet_noise`, and `terminal_value` were extracted from
`MCTS` into free functions (a pure refactor — re-ran the full existing
test suite unchanged immediately after and confirmed nothing regressed,
before writing a single line of new batching code) and are imported
directly into `batched_mcts.py`. `BatchedMCTS` doesn't reimplement or
approximate any of this logic — it calls the exact same functions the
serial `MCTS` calls, just orchestrated across multiple trees with a
shared network-call step. The serial `MCTS` class itself is completely
unchanged in behavior (same file, same public API, same outputs) —
Stage 2's and Stage 4's existing tests
(`test_mcts_finds_mate_in_one`, `test_train_steps_can_overfit_a_tiny_fixed_batch`)
were re-run with zero modifications and still pass, confirming this
was purely additive.

## Low-branching safeguard

If `leaves_per_tree_per_round` exceeds how much genuine diversity a
position supports (e.g. a forced sequence with only one legal move),
virtual loss can't diversify a second selection away from an
already-selected, not-yet-expanded leaf — the same `MCTSNode` object
would otherwise get queued twice in the same batch and backed up
twice, silently double-counting one real simulation as two.
`run_batch` tracks already-selected leaves within a round (by object
identity) and stops collecting further leaves for that tree once a
repeat is hit, rather than double-counting; that tree just catches up
over more rounds. Covered by
`test_low_branching_position_does_not_crash_or_miscount` (a
one-legal-move king position with `leaves_per_tree_per_round=8`).

## Batched self-play orchestration

`batched_self_play.py::run_batched_self_play` drives N `ChessEnv`
instances concurrently: each round, it collects the current board from
every still-active game, runs one `BatchedMCTS.run_batch` call across
all of them, and steps each game forward with its own sampled move —
identical per-ply recording (`(board, MCTS policy, mover)`, backfilled
to `+1/-1/0` once each game's own outcome is known) to
`self_play.py::play_self_play_game`. Games that finish early (shorter
games) simply stop contributing to future rounds' batches while longer
games continue — the active batch size shrinks over the course of a
round of N games rather than staying fixed at N throughout. (A
worker-pool variant that immediately replaces a finished game with a
fresh one to keep the batch size constant throughout would push
utilization higher still — not implemented here, flagged as a natural
next optimization if the numbers below suggest it's worth it.)

## Benchmark

Per the plan, simulation count is held fixed at 100/move for a fair
comparison — nothing about MCTS parameters, self-play temperature
schedule, or training hyperparameters changed in this step, only how
self-play is executed.

```
python scripts/benchmark_batched_self_play.py --games 8 --simulations 100 --leaves-per-tree 4
```

Confirmed CUDA throughout — `device: cuda`, `cuda device name: NVIDIA GeForce RTX 4060 Laptop GPU`, and an explicit `network device check: cuda:0` printed before either run starts, so there's no ambiguity about a silent CPU fallback in either path:

```
                          serial (old)     batched (new)
wall time (s)                    571.4             221.3
total plies                        814              1250
moves/sec                        1.424             5.649
games/hour                        50.4             130.2

speedup (moves/sec): 3.97x
```

**moves/sec is the fair comparison metric** (games/hour is shown too,
but conflates the speedup with the fact the two runs happened to play
different total numbers of plies — self-play is stochastic, so an
8-game batch doesn't play the same total moves twice). At the same
100 simulations/move, batched self-play (N=8 concurrent games,
`leaves_per_tree_per_round=4`, batch size up to 32) processed **3.97x
more moves per second** than serial batch-size-1 self-play. Both runs
completed 8/8 games with 0 discarded (mostly decisive — 7-8 checkmates
each run, one insufficient-material draw in the batched run). This
benchmark script measures throughput only and doesn't persist its
buffers to disk; correctness of the recorded examples (shapes/dtypes,
correctly-signed values, no double-counted simulations) is established
separately by `tests/test_batched_mcts.py` and
`tests/test_batched_self_play.py`, run before this benchmark.

This is a real, meaningful win but not the full theoretical ceiling —
`leaves_per_tree_per_round=4` was left at its modest default per the
"don't change hyperparameters in this step" instruction; the
scale-up conversation is a natural place to also tune batch depth
(`leaves_per_tree_per_round`) and N (games concurrently) together,
now that there's a real baseline number to tune against instead of a
guess.

## Explicitly not done in this step

Per the plan (isolate the throughput fix, decide scale-up together
afterward): `run_training_loop` / `scripts/train.py` still call the
serial `training.self_play.run_self_play` — batched self-play is not
wired in as the default self-play path yet. That's a deliberate choice
to make together once the benchmark numbers below are reviewed, not a
default this step reaches for on its own.
