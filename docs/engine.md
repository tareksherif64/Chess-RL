---
module: engine
depends_on: [python-chess, gymnasium, numpy]
depended_on_by: [agents, scripts, training (future)]
---

# engine/

Responsibility: everything about turning a chess position into RL-consumable
tensors, and back. This is the only module that imports `python-chess`
directly for rules/legality — no other module should hand-roll chess logic.

## Files

- **`encoding.py`** — pure functions, no environment state:
  - `encode_board(board) -> np.ndarray` shape `(8, 8, 18)`, `float32`.
  - `encode_move(move) -> int` / `decode_move(action, board) -> Move`,
    the AlphaZero-style 8x8x73 = 4672 action scheme. See
    [action_space.md](action_space.md) for the full design rationale.
  - `action_mask(board) -> np.ndarray` shape `(4672,)`, `float32`, 1.0 at
    indices corresponding to `board.legal_moves`.
- **`chess_env.py`** — `ChessEnv(gym.Env)`, the stateful wrapper:
  - `reset(seed=None, options=None)` — `options` may carry `fen` (start
    from a custom position) and `player_color`.
  - `step(action)` — decodes, validates against `board.legal_moves`
    (raises `ValueError` on illegal actions rather than silently
    penalizing — the action mask exists precisely so illegal actions are
    never sampled in the first place), pushes, returns
    `(obs, reward, terminated, truncated, info)`.
  - `action_mask()`, `legal_moves()`, `current_player()`.
  - Reward is sparse: `0.0` on every non-terminal step, `+1/-1/0` at game
    end relative to `player_color`. No intermediate shaping — that's
    deliberately deferred to the RL phase, since shaped rewards change
    the problem the agent is solving and we want a clean baseline first.

## Design choices & tradeoffs

- **Two-player, no embedded opponent.** `ChessEnv` does not decide who
  plays black vs white — it just advances whichever side `board.turn`
  says is next, whoever calls `step()`. `player_color` only sets the
  sign convention for terminal reward. This keeps the env a plain
  simulator that composes with self-play, random-vs-random, or
  human-vs-agent drivers without env code changes. The alternative
  (env owns a fixed opponent policy and auto-plays it between agent
  turns) was rejected because it would hard-code an opponent into what
  should be a reusable simulator, and would tie env code to whatever
  policy interface the RL phase ends up using.
- **Absolute (white-relative) board orientation**, not
  side-to-move-relative/mirrored. Simpler to implement and debug (no
  board-flipping, no move-mirroring), at the cost of the network not
  getting a canonical single-perspective input for free — a mirrored/
  canonical encoding (as AlphaZero uses) may be worth revisiting once
  we're training self-play, since it can help the network generalize
  across colors. Flagged here for later reconsideration, not a
  correctness issue now.
- **`claim_draw=True` by default** (constructor flag) — draws by
  threefold repetition / 50-move rule terminate the episode rather than
  requiring an explicit "claim" action, since RL agents have no notion
  of claiming a draw as a separate move. See `docs/action_space.md`
  and `tests/test_rules.py` for a subtlety this causes: python-chess
  allows claiming *before* playing the actual repeating/100th-halfmove
  move, so termination can land one ply earlier than naively expected.
- **Illegal actions raise, not penalize.** Per the requirements, chess's
  action space is too large and structured for "try it and get a
  penalty" — the mask is the intended interface, and a raised exception
  makes bugs in an agent's masking loud instead of silently corrupting
  training data.
