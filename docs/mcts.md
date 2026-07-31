---
module: training (mcts.py)
depends_on: [engine, training/network.py, training/tensors.py]
depended_on_by: [training/self_play.py (future), training/train.py (future)]
---

# training/mcts.py

Stage 2 of Phase 2: Monte Carlo Tree Search guided by the Stage 1
network — the core algorithmic difference between AlphaZero-style
search and vanilla MCTS (which evaluates leaves via random rollouts to
the end of the game; here the network's value head replaces the
rollout, and its policy head biases which moves get explored at all).

## Algorithm

Standard AlphaZero PUCT loop, run once per real move to produce a
`num_simulations`-visit search tree:

1. **Select** — from the root, repeatedly pick the child maximizing
   `Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))` until reaching
   an unexpanded node. `Q(s,a) = -child.value`: each node's `value_sum`
   is accumulated from the perspective of *the side to move at that
   node*, so a child's value has to be negated to score it from its
   parent's perspective (a good position for whoever moves next at the
   child is a bad position for whoever just moved to get there).
2. **Expand + evaluate** — at the leaf, run the network once to get a
   policy (masked to legal moves via `masked_softmax`, from
   Stage 1) and a value. Attach one child per legal move with prior =
   network probability for that action. If the leaf is actually a
   terminal position (checkmate/stalemate/draw), skip the network
   entirely and use the ground-truth outcome instead (`terminal_value`)
   — the whole point of search is to find these forced results, so
   using a learned estimate where the true value is already known would
   throw away free, exact signal.
3. **Backup** — walk back up the visited path, incrementing
   `visit_count` and adding the (sign-flipping-every-ply) value to
   `value_sum` at each node.

After `num_simulations` runs, `visit_count_policy()` turns the root
children's visit counts into a probability distribution (the
"MCTS-improved policy" — this becomes the training target in Stage 3,
since visit counts after search are a strictly better move-quality
signal than the raw network priors that guided the search). `select_action()`
picks a move from that distribution: greedy (`temperature=0`) for
evaluation/play, sampled (`temperature>0`) for self-play exploration.

## Why the terminal-value test matters more than it looks

`tests/test_mcts.py::test_mcts_finds_mate_in_one` runs full search
against two mate-in-1 positions using a **completely untrained,
randomly-initialized** network — no learning has happened yet, so the
network's priors and value estimates are close to noise. The search
still finds the mate reliably (verified across 10 different random
network seeds, not just the one committed here) because of step 2's
terminal-value shortcut: the instant a simulation actually plays the
mating move, backup immediately assigns that action `value = 1/1 =
1.0` (a real value, not a network guess), which then dominates its
PUCT score in every later simulation and pulls visit count toward it.
This is the mechanism that will let the *trained* network refine its
priors over time without ever losing the ground-truth tiebreaker at
forced wins/losses — worth understanding now since it's the same
mechanism that makes self-play (Stage 3) work at all before the
network has learned anything.

## Design choices & tradeoffs

- **Board via copy-per-simulation, not push/pop-per-node or
  board-per-node.** `sim_board = board.copy()` once per simulation,
  then push moves while descending. Simpler and less bug-prone than
  interleaving push/pop with tree traversal (no risk of forgetting a
  pop on an early-exit path), and avoids storing a full board per
  node (would multiply memory by node count for no benefit — search
  trees here are shallow-but-wide, not something we need positions
  cached for after the fact). Tradeoff: a `Board.copy()` per
  simulation is redundant work vs. push/pop on one shared board;
  acceptable for now since network inference dominates simulation
  cost anyway. Worth revisiting only if self-play throughput (Stage 3)
  turns out to be copy-bound rather than inference-bound.
- **`c_puct = 1.5`** — a fixed constant (not AlphaZero's dynamic
  `c_puct(s)` formula that grows with visit count). Simpler, standard
  in most non-cluster-scale AlphaZero reimplementations, and easy to
  tune later as a single number if self-play games look too greedy
  (raise it) or too random (lower it).
- **No batched leaf evaluation.** Each simulation calls the network
  once, batch size 1. AlphaZero implementations at scale batch several
  in-flight simulations' leaf evaluations together (with "virtual
  loss" to keep them from colliding) to use the GPU efficiently: not
  implemented here since Stage 2's job is correctness on a single
  search, not throughput — flagged as the first thing to revisit if
  self-play generation (Stage 3) turns out to be network-call-bound.
- **No Dirichlet root noise.** AlphaZero adds Dirichlet noise to root
  priors during self-play specifically to force exploration away from
  the network's current favorite move. That's a self-play concern
  (Stage 3), not a search-correctness concern — `MCTS.run()` stays a
  plain, deterministic-given-network search here, easy to unit test
  exactly, with noise to be layered on at the self-play call site.
