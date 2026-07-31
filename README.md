# Chess RL

An RL-based chess bot, built in stages. This is stage 1: the game/
environment layer only — no RL agent yet.

## What's here

```
engine/     ChessEnv (Gymnasium-style) + board/move tensor encoding
agents/     Baseline agents (random for now; neural comes later)
training/   Empty — RL training loop comes later
tests/      pytest suite validating engine correctness against python-chess
scripts/    Runnable entry points (self-play stats, CLI watcher, GUI watcher)
gui/        Tkinter desktop app for watching self-play live
docs/       One note per module: responsibilities + dependencies
```

Each module's `docs/*.md` note has more detail; `docs/action_space.md`
specifically covers the action-space encoding design.

## Setup

```
pip install -r requirements.txt
```

torch is CPU-only via plain `pip install torch` — see `docs/training.md`
for the CUDA install command needed before the training phase.

## Run the tests

```
python -m pytest -v
```

## Validate the environment

```
python scripts/self_play_random.py --games 100
```

## Watch the bots play

```
python scripts/watch_random_game.py       # terminal
python scripts/gui_random_vs_random.py    # desktop GUI (tkinter)
```

## Status

Environment + encoding + random baseline + tests are complete and
passing. RL agent, MCTS, and neural network are **not** implemented yet
— next step, pending go-ahead.
