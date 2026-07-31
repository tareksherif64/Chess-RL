---
module: scripts
depends_on: [engine, agents]
depended_on_by: []
---

# scripts/

Responsibility: runnable entry points that exercise `engine` + `agents`
together, outside of the automated test suite — for eyeballing behavior,
not for CI/pytest.

## Files

- **`self_play_random.py`** — plays N random-vs-random games through
  `ChessEnv`, printing per-game result/termination-cause/ply-count and
  an aggregate summary. Every action comes from `env.action_mask()`, so
  a run completing without a raised `ValueError` is itself evidence the
  action mask never permits an illegal move. Used to validate the
  environment (100-game run surfaced checkmate, stalemate, threefold
  repetition, 50-move rule, and insufficient-material terminations, with
  zero illegal-move exceptions) before any RL/tests existed for those
  paths.

```
python scripts/self_play_random.py --games 100 --seed 42
```
