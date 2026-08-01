---
module: gui (human_vs_model_controller.py, human_vs_model_app.py)
depends_on: [engine, training/network.py, training/mcts.py, training/checkpoint.py, gui/board_view.py]
depended_on_by: []
---

# Human vs. model

A standalone GUI mode: play a real game against a loaded checkpoint by
clicking the board. Deliberately isolated from everything training-
related — no file under `training/` was touched to build this, and
nothing here is used by the training loop. See "Design decisions"
below for why that separation mattered enough to shape the file
layout, not just the intent.

## Files

- **`gui/human_vs_model_controller.py`** — all the actual logic: loads
  a checkpoint into a `PolicyValueNet`, owns a `ChessEnv`, runs serial
  `MCTS` for the agent's moves. Contains **no `tkinter` import at
  all** — this is what makes it directly unit-testable (see
  `tests/test_human_vs_model.py`) without a display, a window, or a
  mouse.
- **`gui/human_vs_model_app.py`** — the Tkinter window. Turns mouse
  clicks into calls to the controller, runs the controller's slow MCTS
  call on a background thread so the window doesn't freeze, and
  redraws whatever the controller's state says.
- **`gui/board_view.py`** — extended (not replaced) with `square_at()`
  (pixel → board square) and highlight rendering for the selected
  square and legal destinations. The existing spectator app
  (`gui/app.py`, `ChessWatcherApp`) is untouched and unaffected — the
  new parameters default to `None`/`None`, so its existing
  two-positional-argument `draw(board, last_move)` calls behave
  exactly as before.
- **`scripts/play_human_vs_model.py`** — CLI entry point.

## Note on the existing move-input assumption

The task that started this feature assumed the GUI already had some
move-input mechanism to "reuse" for legal-move selection. Worth
recording plainly: it didn't. The existing GUI (`gui/app.py`) is a
pure spectator/autoplay app for watching two agents play each other —
no mouse binding, no square selection, nothing a human could click to
make a move. Click-to-move (square selection, legal-destination
highlighting, promotion resolution) was built from scratch as part of
this feature, not reused from anything pre-existing. Flagging this
rather than silently building on the mismatched assumption.

## How it works

### Loading the checkpoint

```python
self.network = PolicyValueNet().to(device)
load_checkpoint(checkpoint_path, self.network, optimizer=None, device=device)
self.network.eval()
```

Same `PolicyValueNet()` default architecture the training loop uses
(no constructor args — the checkpoint's saved `state_dict` shapes have
to match, and training never varies them), and the same
`load_checkpoint()` from `training/checkpoint.py` used everywhere else
in the project. `optimizer=None` because this is pure inference —
there's nothing to keep training, so there's no optimizer state to
restore (the same pattern `training/evaluation.py`'s checkpoint-vs-
checkpoint games already use). `network.eval()` matters here for the
same reason it matters during training's self-play/evaluation phases:
`BatchNorm` layers behave differently in train vs. eval mode, and a
loaded checkpoint should be evaluated deterministically, not with
batch statistics computed from whatever's currently in a batch of one.

### How MCTS is invoked for the agent's turn

```python
root = self.mcts.run(self.env.board, self.num_simulations)
action = select_action(root, temperature=0.0)
return self.env.decode_move(action)
```

This is the plain **serial** `MCTS` from `training/mcts.py` — not
`BatchedMCTS`. The throughput work earlier in this project
(`docs/batched_self_play.md`, `docs/batched_evaluation.md`) batches
leaf evaluation *across concurrent games* specifically because
self-play and evaluation run many games at once. Here there is
exactly one game and one network; there is nothing to batch across.
Requirement 4 asked for this explicitly, and it's the correct call for
the same reason `training/evaluation.py` also stays serial: batching
is a throughput optimization for many-games workloads, and forcing it
onto a single game would add real complexity (see
`training/batched_evaluation.py`'s network-splitting design) for zero
benefit.

Same convention as the checkpoint-vs-checkpoint evaluation games:
**`temperature=0`, no Dirichlet noise.** This isn't self-play data
generation (which wants exploration variety); it's the agent playing
its actual strongest move against a real opponent. Move quality is
controlled purely by `--simulations` (default 200 — higher than
training's 100, since interactive play only ever runs one search at a
time and isn't optimizing for throughput, and there's no reason not to
spend more compute per move when nothing else is competing for the
GPU).

### The "thinking" indicator, and why MCTS runs on a background thread

Tkinter is single-threaded: any code running on the main thread blocks
the entire UI, including the ability to repaint anything, while it
runs. At 200 simulations, `MCTS.run()` can take a few real seconds —
calling it directly from a click handler would freeze the window for
that whole time, which is indistinguishable from a hang to someone
who doesn't know better (exactly what requirement 5 was trying to
avoid).

```python
def _start_agent_thinking(self):
    self.agent_thinking = True
    self._animate_thinking()
    threading.Thread(target=self._agent_worker, daemon=True).start()
    self.after(100, self._poll_agent_move)

def _agent_worker(self):
    move = self.controller.compute_agent_move()
    self._move_queue.put(move)
```

The search runs on a background `threading.Thread`; the main thread
polls a `queue.Queue` every 100ms via `self.after()` (Tkinter's
non-blocking scheduler, already used throughout this project's GUI
work) until a result appears, and *only then* applies the move and
redraws. `queue.Queue` is the standard safe hand-off between a worker
thread and Tkinter's main thread — no widget is ever touched from the
background thread. While waiting, a status line animates
("Model is thinking...", "Model is thinking.", "..", "...", cycling
every 400ms) so the window visibly isn't frozen, and the board ignores
clicks during this window (`agent_thinking` gates `_on_board_click`)
so a human can't queue up a move while it isn't their turn.

**Why board mutation only ever happens on the main thread:**
`compute_agent_move()` (called from the background thread) only reads
`self.env.board` — `MCTS.run()` copies the board once per simulation
internally (`training/mcts.py`, established in Stage 2) and never
mutates the original. The move is applied back via
`apply_agent_move()`, called from `_poll_agent_move()` on the main
thread only, after the background thread's result has already arrived
through the queue. There's no point where both threads could be
writing to (or one reading while the other writes) the same board
state.

### Click-to-move

Standard click-source-then-click-destination interaction:

1. Click a square with one of your own pieces → it's highlighted
   (orange border), and every legal destination for it is marked (a
   small dot for a move to an empty square, a ring around the edge for
   a capture — so the captured piece stays visible, matching the
   convention most chess UIs use).
2. Click a highlighted destination → the move is made.
3. Click the already-selected square again → deselect.
4. Click a different one of your own pieces while one is already
   selected → reselect to the new piece.
5. Click anywhere else (empty, non-legal, opponent piece not a legal
   capture) → deselect.

All of this lives in `HumanVsModelApp._on_board_click` and is UI
*selection* state only (`selected_square`, `legal_targets`) — it isn't
game state, so it doesn't live in the controller. The controller only
ever sees a finished, validated `chess.Move` via `apply_human_move()`.

### Game-end display

`get_result_text()` on the controller distinguishes three cases and
returns a plain string the app puts in a result label:

- A real chess-rules ending — `"White wins by Checkmate. You win!"` /
  `"...The model wins."` / `"Draw — Threefold Repetition."` (the
  "you"/"the model" framing is resolved against whichever color the
  human picked that game, not hardcoded to white).
- The `max_plies` safety net (600, matching the rest of the project's
  convention — see `docs/self_play.md`) being hit without a real
  ending — `"Game stopped: move limit reached without a decision."`
  Included even though it should be rare in an interactive game (real
  chess-rules endings happen well before 600 plies in virtually all
  real games); consistent with the "always explain what happened,
  never just close or error" requirement.

## Design decisions & tradeoffs

- **`HumanVsModelController` has zero tkinter dependency, by design.**
  This was the single biggest design decision in this feature, made
  specifically because of requirement 7 ("test with a human-move stub,
  not just manual play"). Everything the controller does — loading a
  checkpoint, tracking turns, applying moves, running MCTS, reporting
  the result — is plain Python callable from a test with no window, no
  event loop, no mouse. `HumanVsModelApp` is a thin translation layer
  on top: mouse events in, controller calls out; controller state in,
  widget redraws out. If a bug ever shows up in "how a game plays
  out," it's almost certainly in the controller, and the controller is
  exactly what's covered by `tests/test_human_vs_model.py`.
- **Auto-queen promotion, no piece-choice dialog.** A promotion has 4
  legal moves sharing the same (from, to) square pair — clicking a
  destination square can't distinguish which one you meant.
  `_resolve_move()` always picks queen. This means **underpromotion
  (choosing knight/bishop/rook) isn't reachable through the GUI** —
  a deliberate scope cut, not an oversight: queen is correct in the
  overwhelming majority of real promotions, and a piece-choice popup
  is real additional UI work for a genuinely rare case. If you
  specifically need to test/play an underpromotion, it's not currently
  possible through this interface.
- **Isolated from the training loop, literally as well as in intent.**
  No file under `training/` was modified for this feature — only read
  from (`PolicyValueNet`, `MCTS`, `select_action`, `load_checkpoint`).
  `git diff --stat` against `training/` for this branch is empty by
  construction, not just by care.
- **Color choice is per-game, not per-launch.** `new_game(human_color=...)`
  takes the color fresh each call; the GUI's "Play as" radio buttons
  read at "New Game" click time, so switching sides between games
  doesn't need restarting the app.
- **Requires CUDA by default, `--cpu` to override** — the same
  convention every other entry point in this project uses
  (`resolve_device(require_cuda=True)`, no silent fallback). Serial
  MCTS at 200 sims/move is genuinely playable on CPU too (a human is
  thinking about their own move for longer than that anyway), so the
  override is a real, usable option here, not just a formality.

## Running it

```
python scripts/play_human_vs_model.py --checkpoint checkpoints_overnight/iter_000012.pt
python scripts/play_human_vs_model.py --checkpoint path/to/ckpt.pt --simulations 400
python scripts/play_human_vs_model.py --checkpoint path/to/ckpt.pt --cpu
```

`--checkpoint` is required and can point at any saved checkpoint, not
just the most recent — useful for comparing how the model plays at
different points in training, not just against its current best.

## Testing

`tests/test_human_vs_model.py` (9 tests) drives full games against a
real checkpoint (saved fresh via `training/checkpoint.py::save_checkpoint`
in a pytest tmp_path, so tests don't depend on any specific real
training run's output existing on disk) using a scripted "human" stub
— uniform-random legal move whenever it's the human's turn — for both
human colors. Turn alternation is enforced *structurally*: every
scripted human move goes through `apply_human_move()`, which raises if
called out of turn, so a passing test already proves alternation was
correct throughout, not merely that no exception happened to fire.
Also covers: a constructed forced-mate position (deterministic
real-outcome path for `get_result_text()`), the move-limit truncation
path (distinct message, via an artificially tiny `max_plies`), illegal
/ out-of-turn move rejection, `new_game()` color-switching, and
`legal_moves_from()` against ground-truth `python-chess` legality — the
same "compare against python-chess directly" pattern used throughout
this project's test suite (e.g. `tests/test_action_space.py`).

The Tkinter layer itself was verified by launching the real app
against a real trained checkpoint (`checkpoints_overnight/iter_000012.pt`
from the overnight run), confirming via a screenshot that it loads the
checkpoint and renders the correct starting position with the correct
window title, checkpoint name, and simulation count displayed, then
hand-tracing the click-handler logic (`_on_board_click`) against that
confirmed board geometry for every interaction case (select,
re-select, deselect, execute move, promotion resolution, and the
post-move handoff back to whichever side moves next) — each traced
case matches the already-tested controller behavior exactly. An
automated live-click simulation was attempted but abandoned after an
unrelated window-management mishap on the test machine (killing what
turned out to be the console host for the very process being tested,
not a bug in the app); continuing to script synthetic mouse input
across a busy multi-window desktop was judged more likely to cause
side effects than to add confidence beyond what the hand-traced
review and the headless controller tests already establish. Real
interactive play is still the final check — that's on you, per the
plan.
