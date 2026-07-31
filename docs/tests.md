---
module: tests
depends_on: [engine, agents, pytest]
depended_on_by: []
---

# tests/

Responsibility: prove the environment layer is correct *before* any RL
code depends on it — bugs in board encoding or move encoding are much
harder to diagnose once a policy network is learning on top of them.

## Files

- **`test_encoding.py`** — board tensor shape/dtype/plane-correctness
  (piece planes, side-to-move, castling rights, en passant); action
  encode/decode round-trip across random games (including promotions),
  plus explicit underpromotion/castling/en-passant cases.
- **`test_action_space.py`** — `action_mask()` compared directly against
  `board.legal_moves` (the ground truth) across several hand-picked
  positions (opening, castling-rights-heavy, promotion-heavy, sparse
  endgame); also checks `encode_move` never collides two legal moves
  onto the same index in any tested position.
- **`test_rules.py`** — checkmate (fool's mate, both reward
  perspectives), stalemate, threefold repetition, 50-move rule — all
  verified through `ChessEnv.step()`, not just `python-chess` directly,
  so termination/reward wiring is covered too, not just python-chess's
  own correctness.
- **`test_chess_env.py`** — API contract (obs shape/dtype, action space
  size), illegal-action rejection, post-game-over rejection, sparse
  intermediate reward, a full random-vs-random game smoke test, and a
  `player_color` reward-sign check on an identical game.

## Running

```
python -m pytest -v
```

`pyproject.toml` sets `pythonpath = ["."]` so `engine`/`agents` import
cleanly regardless of what directory pytest is invoked from within the
repo.

## Notable non-obvious finding baked into the tests

`board.is_game_over(claim_draw=True)` can go `True` **one ply before**
the literal 3rd repeated position / 100th halfmove is on the board —
python-chess models the FIDE rule that a player may *claim* a draw as
soon as they have a legal move that would create the repetition/reach
the clock limit, without actually having to play it. `test_rules.py`'s
threefold/50-move tests account for this (see comments in that file) —
worth knowing before writing anything else that reasons about when a
game "should" end.
