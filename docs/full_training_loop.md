---
module: training (train.py wiring, evaluation.py)
depends_on: [engine, training/network.py, training/mcts.py, training/batched_mcts.py, training/batched_self_play.py, training/self_play.py, training/checkpoint.py, training/logger.py]
depended_on_by: []
---

# Wiring batched self-play + evaluation into the training loop

Three changes on top of the already-built, already-benchmarked pieces
(Stage 1-4, plus the batched-self-play throughput fix): make batched
self-play `run_training_loop`'s actual self-play step (it was built
standalone and benchmarked, but never wired in as the default), pick
real training-scale parameters instead of pipeline-validation-scale
placeholders, and add a real strength signal (games won/lost) since
loss going down doesn't prove the network plays better.

## 1. Batched self-play is now the default self-play step

`run_training_loop` calls `training.batched_self_play.run_batched_self_play`
instead of `training.self_play.run_self_play`. `games_per_iteration`
(an existing parameter) now doubles as **N**, the number of concurrent
games/trees; `TrainConfig.leaves_per_tree_per_round` (new, default 4)
controls the other batching axis. `training/self_play.py`'s serial
path is untouched and still used directly by `scripts/run_self_play.py`
and its own tests — nothing about Stage 3 was removed, just no longer
the loop's default.

### Config sweep: is N=8/leaves=4 actually the best use of the GPU?

Before locking in the already-benchmarked N=8/leaves=4 config, swept
two nearby configs at the same 100 simulations/move
(`scripts/sweep_self_play_config.py`, each run with `num_games = N`,
i.e. exactly as it would run in one real training iteration):

```
   N  leaves  batch cap   moves/sec   games/hour
   8       4         32       5.739        132.2
  16       4         64       5.752        105.7
   8       8         64       5.765        116.4
```

**moves/sec is statistically flat across all three** — the spread
(5.739 to 5.765) is well within run-to-run noise for stochastic
self-play, not a meaningful difference. This is a real finding, not a
non-result: it means **the RTX 4060 is already saturated somewhere
around batch size 32 with this ~493K-parameter network** — pushing
batch size to 64 (via more concurrent games, or more virtual-loss
depth per tree) buys nothing further. The bottleneck has most likely
shifted to CPU-side work (Python tree traversal, board copying,
`chess.Board.legal_moves` generation) rather than GPU compute or
call overhead at this point.

`games/hour` varies more across the three (132.2 / 105.7 / 116.4), but
that's a confound, not a real difference: self-play games have highly
variable length (9 to 515 plies observed across these runs), so total
wall time for a fixed N is mostly driven by which random game lengths
happened to come up, not by the batching config. moves/sec (normalized
by actual work done) is the metric that isolates the throughput
question the sweep asked, and it says these three configs are
equivalent.

**Chosen default: N=8, leaves_per_tree_per_round=4** — since the three
are tied on throughput, picked the smallest/simplest of them rather
than the nominal highest-numbered "winner" (leaves=8, ahead by 0.026
moves/sec, i.e. noise). Smaller `leaves_per_tree_per_round` means less
simultaneous virtual-loss approximation happening within any one tree
at a time (each in-flight leaf is a temporary distortion of that
tree's stats until its real evaluation returns — see
`docs/batched_self_play.md`), so there's no reason to carry more of
that approximation than necessary when it isn't buying throughput.

## 2. Training-scale parameters

- **Simulations/move: 100** (was 20 for pipeline validation). This is
  the move-quality knob, not a pipeline-correctness one — 20 was only
  ever meant to prove the wiring worked, not to search deeply enough
  to produce a meaningfully strong move.
- **Checkpoint every iteration** — already `TrainConfig`'s existing
  default (`checkpoint_every_iterations=1`), unchanged. Cheap
  insurance (~2MB per checkpoint) for an unsupervised multi-hour run:
  never lose more than one iteration's progress if something crashes.
- **Games per iteration — the math:**

  Using the chosen config's measured throughput (~130 games/hour,
  averaging the two N=8/leaves=4 measurements: 132.2 from the sweep,
  130.2 from the original benchmark — both runs of the identical
  workload, giving a stable estimate):

  | self-play budget/iteration | games/iteration (130 games/hour × budget) |
  |---|---|
  | 30 min | 65 |
  | 45 min | ~98 |
  | 60 min | 130 |

  **Caveat, stated plainly:** this is measured throughput at N=8
  extrapolated to a larger N (65-130 concurrent games), not a directly
  tested data point — the sweep only tested N=8 and N=16. Since
  moves/sec was flat (not declining) going from N=8 to N=16, it's a
  reasonable extrapolation that N=65+ continues to run at a similar
  moves/sec (if anything, larger batches should be at least as
  GPU-efficient, not less) — but it hasn't been measured, and this is
  exactly the kind of number worth confirming against the real
  achieved throughput once a long run is actually going, not assumed
  in advance. Proposed for confirmation, not yet chosen unilaterally.

## 3. Checkpoint-vs-previous and vs-random evaluation

`training/evaluation.py` (new): `evaluate_checkpoints()` plays games
between the current network and a snapshot of its pre-training-step
weights, alternating colors; `evaluate_against_random()` plays against
the Phase 1 `RandomAgent` baseline. Both use the serial `MCTS`, not
`BatchedMCTS` — deliberately, see the module docstring: an evaluation
game alternates control between two *different* networks within one
game, which doesn't fit `BatchedMCTS`'s one-network-per-batch design
built for self-play. This is real, non-negligible wall time (measured
below) and the first candidate for a future batching effort if
evaluation becomes the bottleneck instead of self-play.

**Wiring in `run_training_loop`:**
- Before self-play/training starts each iteration, the network's
  current weights are cloned (cheap — ~2MB). This is the "previous"
  snapshot.
- **vs-previous runs every iteration** — but only if training actually
  happened that iteration (`loss_stats is not None`). If the buffer
  was below `min_buffer_size` and training was skipped, "new" and
  "old" would be bit-identical, so the eval is skipped too rather than
  spending real time confirming a network hasn't changed.
- **vs-random runs every `eval_vs_random_every_iterations`** (default
  5) iterations, as a periodic sanity floor.
- Every iteration's logged row always contains the same set of CSV
  columns (`eval_vs_previous_*`, `eval_vs_random_*`), with `None` on
  iterations where an eval type didn't run — `csv.DictWriter` requires
  a fixed column set for the whole file, so "didn't run" has to be a
  value, not a missing column.

## Small-scale validation run

Per the plan, validated the full wired-together system (batched
self-play + training + both evaluations + checkpointing + logging) at
small scale before proposing any long run — 3 iterations, 4 games/
iteration, **100 simulations/move** (the real target, not the 20 used
for earlier pipeline validation), `eval_vs_random_every_iterations=2`
(lowered from the real default of 5 specifically so a 3-iteration
validation run actually exercises that code path once, confirming it
works rather than trusting it by inspection alone).

```
python scripts/train.py --iterations 3 --games-per-iteration 4 --leaves-per-tree 4 \
    --simulations 100 --train-steps 20 --batch-size 16 --min-buffer-size 16 \
    --eval-vs-previous-games 4 --eval-vs-random-games 4 --eval-vs-random-every 2
```

All 3 iterations completed cleanly: 12/12 self-play games finished (0
discarded), a checkpoint saved every iteration, vs-previous eval ran
every iteration, vs-random correctly fired *only* on iteration 1 (the
one matching `(iteration+1) % 2 == 0`, confirming the cadence logic
rather than trusting it by inspection). Independently reloaded
afterward: `checkpoints_v2/iter_000002.pt` restores to iteration 2 with
real weights on a fresh network+optimizer; `data/train_v2_buffer.npz`
has all 1473 examples, zero NaNs, policy rows summing to 1; the CSV log
has 4 rows (header + 3 iterations) with every eval column present on
every row (`None` where an eval type didn't run that iteration, never
a missing column).

```
iteration  self_play_s  train_s  eval_vs_prev_s  eval_vs_random_s  policy_loss  vs_prev (new/old/draw)  vs_random
    0          164.1      0.61       119.1            —              7.53          0 / 0 / 4                —
    1           51.6      0.34       271.5            278.3           5.79          0 / 0 / 4              1/4 (25%)
    2           53.3      0.35       125.9              —             5.16          0 / 0 / 4                —
```

**Every vs-previous matchup was a 4/4 draw.** Expected, not concerning,
at this stage: 20 gradient steps barely nudge a ~493K-parameter
network, both "new" and "old" are playing at temperature=0 with no
exploration noise, and MCTS search dominated by a still-mostly-random
value/policy head tends to converge on similar lines either way — real
separation between checkpoints needs more training per iteration than
this validation run's minimal 20 steps. `vs-random` at 25% (1/4) after
one iteration of real training isn't a strong signal either way from 4
games — it's there to confirm the mechanism runs and produces sane,
bounded output (it does), not to draw a conclusion from n=4.

**The real finding here: evaluation is currently the dominant cost,**
not self-play. Iteration 1 spent 51.6s on self-play but 271.5s +
278.3s = 549.8s on the two evaluations combined — over 10x the
self-play time. This is exactly the risk flagged when `evaluation.py`
was built (serial `MCTS`, not batched, since an eval game alternates
two different networks within one game): it's fine at n=4 validation
scale, but at real scale (10-20 vs-previous games every iteration,
plus periodic vs-random) this would make evaluation the long pole in
each iteration's wall-clock, not self-play. **Worth resolving — or at
least explicitly budgeting for — before committing to real
games-per-iteration numbers or a long run's total time estimate**, since
the games/hour and iteration-timing math elsewhere in this doc covers
self-play only.
