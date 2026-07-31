---
module: training (batched_evaluation.py)
depends_on: [engine, training/batched_mcts.py, training/mcts.py, training/evaluation.py, agents/random_agent.py]
depended_on_by: [training/train.py]
---

# training/batched_evaluation.py

Fixes the bottleneck the previous step's validation run surfaced:
evaluation was serial (batch size 1) for a structural reason —
`BatchedMCTS` assumes one shared network drives every tree in a batch,
true for self-play (a game plays itself) but false for evaluation
(a game alternates between two *different* players: new vs. old
checkpoint, or a checkpoint vs. the random baseline).

## The core design change: split by network, not just by game

Self-play's batching axis is simple: N concurrent games, all driven by
the same network, so every active game's current leaf goes into one
shared batch. Evaluation adds a second axis: at any point across N
concurrent evaluation games, *which* network needs to move depends on
whose turn it is in *that specific game* — some games need the "new"
checkpoint right now, others need "old" (or, for vs-random, need
nothing at all, since `RandomAgent` doesn't touch the network).

`_play_concurrent_games` (the shared driver behind both
`run_batched_evaluate_checkpoints` and `run_batched_evaluate_against_random`)
handles this by partitioning each round's active games by which
resource they need, and running one `BatchedMCTS.run_batch` call **per
group** instead of one call for the whole round:

```
round N: 20 active games, colors alternating who's "new"/"old"
  -> group "new":    8 games where it's the new checkpoint's move  -> one batched call
  -> group "old":   11 games where it's the old checkpoint's move  -> one batched call
  -> group "random": 1 game where it's the random agent's move     -> no network call at all
```

Each group still gets the full batching benefit among its own
members; the two network groups can't share a forward pass anyway
(different weights, nothing to batch across them). `BatchedMCTS`
itself is completely unchanged — same PUCT selection, same virtual
loss, same everything, just invoked once per group per round instead
of once per round.

## Correctness

Same bar as the self-play throughput fix: proven, not just tested for
plausibility.

- **`tests/test_batched_evaluation.py::test_batched_eval_matches_serial_move_sequence_at_minimal_concurrency`**
  — at N=1 game and `leaves_per_tree_per_round=1` (no two leaves ever
  in flight in the same tree — the same reducible case Stage 5 used
  for self-play), batched evaluation must produce the *exact same
  move at every ply*, not just the same final result, as manually
  driving serial `MCTS` with the same two networks. Verified via the
  actual final board FEN and move stack, not just the win/loss tally.
- **`tests/test_evaluation.py` (the serial module) re-run with zero
  modifications** — all 5 tests still pass, confirming the existing
  serial evaluation code and its tests were untouched.
- Additional batched-specific tests: concurrent games with different
  network assignments don't interfere with each other, identical
  networks produce symmetric (unbiased) win rates via the
  alternating-color bookkeeping, forced-mate positions are still
  found correctly through the batched path.

One test-writing lesson worth recording: an early version of the
forced-mate test asserted the network should win *both* games in a
2-game vs-random matchup — it didn't, because colors alternate between
games, so the second game had the *random* agent moving first in the
mate-in-1 position instead of the network. Not a bug — a wrong test
assumption, caught by the test itself. Fixed by explicitly placing the
network on the side actually to move in the FEN, rather than assuming
game-index-0's default color assignment matches.

## Benchmark — and why the first run's numbers were wrong

Ran the same style of comparison as the self-play benchmark: fixed
workload (20 games), fixed simulation count (100/move), serial vs.
batched, on CUDA (device explicitly logged before either run).

**First attempt** measured raw wall-time for a fixed game count and
got a nonsensical result: batched *4.4x slower* (660.7s vs 149.9s).
Investigating rather than trusting it: evaluation runs fully
deterministically (temperature=0, no Dirichlet noise — it's measuring
strongest play, not exploring), and it turns out that matters a lot
for *how long* an evaluation game between two untrained networks
runs. Serial search settled into a fast repetition trap almost
immediately — the same "shuffle a knight back and forth" degenerate
pattern seen in Phase 1's early random-vs-random testing — reaching a
claimable threefold repetition in as few as **11 plies**. Batched
search, using virtual loss (an *approximate* search — proven
non-identical to serial in general, only bit-exact in the reducible
N=1/leaves=1 case), found a *different* deterministic line for the
exact same two networks that avoided that trap and ran **136-402
plies** instead — actual decisive games, not repetition draws.

That means the first benchmark wasn't comparing "same workload,
different speed" — it was comparing 200 total plies (serial, mostly
quick repetition-trap draws) against 3940 total plies (batched, mostly
real decisive games) and reporting wall-clock as if the workloads
matched. Exactly the "games/hour conflates with total plies played"
trap already flagged and corrected for in the self-play benchmark —
this benchmark just hadn't been written with the same normalization
yet. Fixed by tracking total plies and reporting moves/sec, same as
self-play:

```
                          serial (old)     batched (new)
wall time (s)                    134.9             656.4
total plies                        200              3940
moves/sec                        1.483             6.003

speedup (moves/sec): 4.05x
```

**4.05x speedup in moves/sec** — consistent with (very slightly
better than) self-play's 3.97x, exactly as expected since it's the
same underlying batching mechanism.

## A real, separate finding worth carrying into the scale-up conversation

Independent of the throughput fix: **evaluation wall-clock time is
inherently unpredictable**, because it depends on which deterministic
line the zero-noise search happens to settle into for each specific
checkpoint pairing — anywhere from ~10 plies (quick repetition trap)
to 400+ plies (a long decisive or drawn-out game) for the *same two
networks*, with no randomness to average that out across repeated
runs the way self-play's Dirichlet noise does. This isn't something
this step was asked to fix, and it isn't a bug — just something to
budget for (wide error bars on "how long will evaluation take this
iteration") rather than assume away when planning a long run's total
wall-clock.

## Wired into the training loop

`training/train.py::run_training_loop` now calls
`run_batched_evaluate_checkpoints`/`run_batched_evaluate_against_random`
instead of the serial versions, reusing `TrainConfig.leaves_per_tree_per_round`
(the same field self-play batching uses — one shared batching-depth
knob rather than a separate one for evaluation with no strong reason
to differ). `eval_vs_previous_games`/`eval_vs_random_games` now each
act as N (concurrent games) for their respective batched call, the
same way `games_per_iteration` already did for self-play.

## Small-scale end-to-end validation, with corrected iteration math

Re-ran the exact same validation config as the previous step (3
iterations, 4 games/iteration, 100 sims/move, `eval_vs_random_every=2`)
— identical except evaluation is now batched:

```
python scripts/train.py --iterations 3 --games-per-iteration 4 --leaves-per-tree 4 \
    --simulations 100 --train-steps 20 --batch-size 16 --min-buffer-size 16 \
    --eval-vs-previous-games 4 --eval-vs-random-games 4 --eval-vs-random-every 2
```

All 12 self-play games + all 16 evaluation games (12 vs-previous + 4
vs-random) completed, 0 discarded, a checkpoint saved every iteration.
Independently reloaded afterward: the final checkpoint restores to
iteration 2 with real weights on a fresh network+optimizer; the buffer
has all 2122 examples, zero NaNs, policy rows summing to 1.

**Real total per-iteration wall-clock** (self-play + train + both
evals, all summed — this is the number the previous step's math was
missing entirely):

```
              self_play_s  train_s  eval_vs_prev_s  eval_vs_random_s  TOTAL
iteration 0       338.0      0.75       30.9              —           369.7s
iteration 1       123.9      0.42       45.4            230.5         400.2s
iteration 2        65.3      0.41       62.8              —           128.5s
```

vs. the previous (serial-eval) validation run's totals — 283.8s /
601.7s / 179.6s — this run isn't uniformly faster, because self-play's
own game-length variance (independent of anything changed this step)
dominates iteration-to-iteration: iteration 0's self-play alone took
338s here vs. 164s before, purely because these particular games
happened to run longer (412/507/279/50 plies vs. 326/154/311/30
before). **The isolated 20-game benchmark above is the trustworthy
apples-to-apples throughput number (4.05x); these validation-run
totals are real end-to-end measurements, not a controlled comparison**
— both self-play and evaluation are dominated by how long the
particular games that came up happened to run, not by the code change
itself.

### Redoing the games-per-iteration math, this time including evaluation

The previous table (`docs/full_training_loop.md`) computed
games/iteration from self-play throughput alone — accurate for
self-play, but silently assumed evaluation was free. It measurably
isn't:

- **Self-play**: ~130 games/hour at N=8/leaves=4 (unchanged from
  before, well-characterized across multiple measurements).
- **Evaluation**: ~6.0 moves/sec (batched, measured), but **per-game
  length is highly variable and doesn't average out the way self-play's
  does** — this validation run alone saw eval games ranging
  effectively 7.7-57.6 seconds/game already; the isolated benchmark's
  20-game average was ~32.8s/game (197 plies/game ÷ 6.003 moves/sec).
  Self-play's Dirichlet noise gives it a fairly consistent length
  distribution across many games; evaluation's zero-noise determinism
  does not (see above) — there's no way to shrink this uncertainty
  short of adding noise to evaluation too, which wasn't asked for here
  and would blur the "strongest play" comparison it's meant to make.

Using the ~32.8s/game *typical* estimate (with the explicit caveat
that any individual iteration can land well outside it):

| component | count | typical time |
|---|---|---|
| self-play | 65 games (30 min budget, unchanged from before) | 30.0 min |
| vs-previous eval (every iteration) | 10 games | ~5.5 min |
| vs-random eval (every 5th iteration, amortized) | 10 games ÷ 5 | ~1.1 min |
| **total, typical** | | **~36.6 min/iteration** |

Same budget as before (30 min self-play) now costs ~37 min/iteration
total once evaluation is counted — about 22% more wall-clock per
iteration than the earlier (incomplete) estimate assumed, *in the
typical case*, with real potential to run longer on iterations where
evaluation games happen to land on long lines. Smaller
`eval_vs_previous_games`/`eval_vs_random_games` trade evaluation signal
confidence for tighter, more predictable iteration timing — worth
deciding together now that the actual cost (and its variance) is
measured rather than assumed.
