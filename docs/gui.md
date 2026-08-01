---
module: gui
depends_on: [engine, agents, tkinter (Python stdlib)]
depended_on_by: [scripts]
---

# gui/

Responsibility: a desktop window for watching self-play live, as an
alternative to the terminal-only `scripts/watch_random_game.py`.

## Files

- **`board_view.py`** — `BoardCanvas(tk.Canvas)`, rendering only: draws
  squares, unicode chess glyphs (`chess.UNICODE_PIECE_SYMBOLS`), the
  last-move highlight, and a check indicator. No game logic, no event
  handling — it only knows how to draw a `chess.Board` it's handed.
- **`app.py`** — `ChessWatcherApp(tk.Tk)`, the controller: owns a
  `ChessEnv` + two `RandomAgent`s, drives the game loop via
  `tk.after()`-scheduled ticks (not a blocking `while` loop, so the
  window stays responsive to the Play/Pause/Step/speed-slider
  controls), and updates `BoardCanvas` + status labels each ply.
- **`human_vs_model_controller.py` / `human_vs_model_app.py`** — a
  second, separate mode: play against a loaded checkpoint by clicking
  the board, instead of watching two agents play each other. See
  `docs/human_vs_model.md` for the full writeup. `board_view.py` was
  extended (not replaced) to support this — `square_at()` for
  click-to-square geometry and selected/legal-destination highlighting
  — in a way that leaves `ChessWatcherApp` completely unaffected (the
  new `draw()` parameters default to `None`, matching its existing
  calls exactly).

Entry points:

```
python scripts/gui_random_vs_random.py           # watch two random agents play
python scripts/play_human_vs_model.py --checkpoint path/to/ckpt.pt   # play against a checkpoint
```

## Design choices & tradeoffs

- **tkinter over pygame/a web app.** It's stdlib — zero new
  dependencies, no extra `pip install`, and it's a real native window
  rather than something that needs a browser tab or an asset pipeline
  (piece images, etc.). Tradeoff: less polished visuals than a
  sprite-based renderer would give; acceptable since the goal here is
  "watch the environment behave correctly," not a production board UI.
- **`tk.after()` scheduling, not a loop with `time.sleep()`.** A sleep
  loop would freeze the Tk event loop and make Play/Pause/Step
  unresponsive while waiting. Scheduling the next tick via `after()`
  keeps the UI thread free between moves.
- **Rendering (`board_view.py`) is separated from control flow
  (`app.py`)** the same way `engine` is separated from `agents`: the
  canvas doesn't know what a `RandomAgent` is, and the app doesn't know
  how a piece glyph is drawn. This means swapping in a neural agent
  later, or reusing `BoardCanvas` in a different window layout, needs
  no changes to the other file.
- **Same `ChessEnv`/`RandomAgent` as the CLI scripts** — the GUI adds
  zero new game logic, it's purely a different driver + renderer on top
  of the already-tested `engine`/`agents` modules.
