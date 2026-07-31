---
module: training (self_play.py, replay_buffer.py)
depends_on: [engine, training/mcts.py, training/network.py]
depended_on_by: [training/train.py (future)]
---

# training/self_play.py, training/replay_buffer.py

Stage 3 of Phase 2: the network+MCTS from Stages 1-2 play full games
against themselves, and every position played is recorded as a
training example — `(board tensor, MCTS-improved policy, game outcome)`
— for Stage 4 to train on.

## Files

- **`self_play.py`** — `SelfPlayConfig`, `play_self_play_game()` (one
  full game), `run_self_play()` (N games into a `ReplayBuffer`, with
  per-game logging).
- **`replay_buffer.py`** — `ReplayBuffer`, a fixed-capacity ring buffer
  of `(board, policy, value)` examples with `sample()`/`save()`/`load()`.

Entry point: `scripts/run_self_play.py`.

## How a game is recorded

At each ply: run MCTS from the current position, get back a
visit-count policy `pi` (`training/mcts.py::visit_count_policy`), and
record `(board_before_move, pi, mover_color)` — note `mover_color`, not
a value yet, since the outcome isn't known until the game ends. Once
the game terminates, every recorded position gets its `value` filled
in: `+1` if `mover_color` went on to win the game, `-1` if they lost,
`0` for a draw. This is the standard AlphaZero self-play labeling
scheme — the "supervision" comes entirely from real game outcomes plus
search-improved policies, no separate labeled data.

### Move selection / exploration

- **Dirichlet root noise** (deferred from Stage 2, applied here):
  `MCTS.run(..., add_dirichlet_noise=True)` mixes noise into the root's
  priors before searching. Without this, self-play would be nearly
  deterministic (same network → same search → same game, repeated)
  since the only other randomness is temperature-based sampling late
  in exploration — noise is what makes different self-play games
  actually explore different lines from move 1.
- **Temperature schedule**: `temperature=1.0` for the first
  `temperature_threshold_plies` (default 30) plies — sample the played
  move proportional to `visit_count^(1/temperature)` — then
  `temperature=0` (greedy argmax) for the rest of the game. Same
  schedule as the AlphaZero paper. Critically, **the same
  temperature-scaled distribution is stored as the training target**
  `pi`, not a separately-computed temperature=1 version — so early-game
  targets are intentionally soft (reflecting real search uncertainty)
  and endgame targets are sharp/near-one-hot (reflecting a search that's
  converged on one clearly-best move). Storing a mismatched target would
  train the network to imitate a distribution it never actually played
  from.

### Safety cap, not a guess

`max_plies` (default 600, well above anything observed in Phase 1's
random-vs-random testing, which topped out under 500 plies with much
weaker play) exists purely as a non-termination safety net. If a game
somehow hits it, `play_self_play_game` returns an **empty example
list** and `discarded: True` in its summary — there's no real game
outcome to label those positions with, so nothing is fabricated or
guessed. `tests/test_self_play.py::test_game_hitting_ply_cap_is_discarded_not_guessed`
covers this directly (via `max_plies=1`, not by waiting for a real
600-ply game).

## ReplayBuffer

Fixed-capacity ring buffer over preallocated numpy arrays (`sample()`
and `add()` are O(1)/O(batch), and old games get evicted automatically
once full — see the module docstring for why capacity has no default).
`save()`/`load()` round-trip through `.npz` so a self-play run's output
can be inspected or resumed independently of the process that
generated it (relevant for Stage 4's checkpoint/resume story).

Generated buffers and checkpoints are gitignored (`data/`,
`checkpoints/`, `*.npz`, `*.pt`) — they're regenerable multi-MB+
binary artifacts, not source.

## Sanity-check run

Per the plan, ran a small batch before considering Stage 3 done —
`python scripts/run_self_play.py --games 6 --simulations 100` on the
RTX 4060, **completely untrained network** (Stage 4 training doesn't
exist yet):

```
game 1/6: result=1/2-1/2 termination=THREEFOLD_REPETITION plies=381 examples=381 time=281.7s
game 2/6: result=1-0   termination=CHECKMATE            plies=53  examples=53  time=36.7s
game 3/6: result=0-1   termination=CHECKMATE            plies=48  examples=48  time=33.2s
game 4/6: result=1-0   termination=CHECKMATE            plies=73  examples=73  time=51.7s
game 5/6: result=0-1   termination=CHECKMATE            plies=12  examples=12  time=8.4s
game 6/6: result=1/2-1/2 termination=THREEFOLD_REPETITION plies=330 examples=330 time=249.7s

games played:      6  (discarded: 0)
total examples:    897
avg plies/game:    149.5
terminations:      {THREEFOLD_REPETITION: 2, CHECKMATE: 4}
total wall time:   661.4s (110.2s/game), ~32.7 games/hour at this simulation count
```

Loaded the saved `.npz` back and checked it directly: shapes
`(897,8,8,18)`/`(897,4672)`/`(897,)`, all `float32`, no NaNs, every
policy row sums to 1.0, and the value distribution
(`{-1.0: 89, 0.0: 714, 1.0: 94}`) lines up with the game log — 714 ≈
381+330 draw-labeled positions from the two threefold-repetition
games, and the decisive games split ~89/94 between losing/winning
positions. Data format and labeling logic both check out end-to-end.

Also notable, and unprompted: **4 of 6 games ended in actual
checkmate**, a much more decisive rate than Phase 1's pure-random
self-play (which was heavily draw-dominated — see `docs/scripts.md`,
15/100 checkmates there vs. 4/6 here). Consistent with what
`docs/mcts.md` predicted: MCTS's terminal-value backup finds forced
wins reliably even with a random, untrained network, because a
simulation that stumbles into checkmate gets an exact `+1`/`-1`
signal rather than a noisy guess — the same mechanism validated by the
mate-in-1 tests in Stage 2, now showing up in full games too.

Noted from this run: **per-move cost with an untrained network and
batch-size-1 MCTS leaf evaluation is dominated by GPU call overhead**,
not compute — roughly 0.75s/move at 100 simulations/move on the RTX
4060 (~110s/game average here, heavily skewed by the two 300+ ply
draws). This is the batched-leaf-evaluation gap flagged as a known
tradeoff in `docs/mcts.md`, now measured rather than theoretical. It
doesn't block Stage 3 (correctness and data-format validation, not
throughput, was the goal here) but is the first thing worth addressing
before any real multi-hour self-play run — a conversation for the
Stage 4 scale-up discussion, alongside simulation count and buffer
capacity.

## Design choices & tradeoffs

- **`play_self_play_game` always starts from the standard position**
  in real use; `initial_fen` exists only so tests can play short,
  deterministic games (e.g. from a mate-in-1 FEN) instead of waiting
  out full ~40-80 ply games for every assertion.
- **No batching across simulations or across games**, matching the
  same tradeoff already made in `MCTS` — one network call per leaf,
  one game played at a time. `run_self_play()`'s per-game loop is the
  natural place to eventually parallelize (multiple games' leaf
  evaluations batched together) if self-play throughput becomes the
  bottleneck for real training — not implemented now since Stage 3's
  job was proving the data-recording pipeline is correct end-to-end at
  small scale, not fast.
