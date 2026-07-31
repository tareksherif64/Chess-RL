---
module: agents
depends_on: [engine (action_mask contract, numpy dtypes)]
depended_on_by: [scripts, training (future)]
---

# agents/

Responsibility: things that choose an action given a legal-move mask.
Nothing in here imports `chess` directly or touches board state — agents
only see `action_mask()` output (a `(4672,)` float32 array), keeping them
decoupled from `engine`'s internals so a future neural agent can swap in
without changing how it's driven by scripts/training code.

## Files

- **`random_agent.py`** — `RandomAgent.select_action(action_mask) -> int`,
  uniform sample over legal (nonzero-mask) indices. Exists purely to
  validate the environment (this milestone) and as a baseline opponent
  once training starts.

## Design choices & tradeoffs

- Agents take the **mask**, not the `chess.Board` or a list of
  `chess.Move`. This is the interface a neural policy will also use
  (mask logits, then sample/argmax), so the random agent exercises the
  exact same call shape the RL agent will later — no special-casing in
  `scripts/self_play_random.py` needed when we swap agents in.
- No `seed` state shared globally — each `RandomAgent` owns its own
  `numpy.random.Generator`, so white and black in a self-play script get
  independent, reproducible streams instead of fighting over global
  RNG state.
