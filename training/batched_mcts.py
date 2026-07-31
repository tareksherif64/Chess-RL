"""BatchedMCTS: the same PUCT search as `MCTS` (training/mcts.py), run
across N independent trees at once, with leaf evaluations from *all* of
them combined into a single network forward pass.

Why: at batch size 1, each simulation's network call is dominated by
GPU kernel-launch/transfer overhead, not compute (measured ~0.75s/move
at 100 simulations on an RTX 4060 — see docs/mcts.md, docs/self_play.md).
Batching many boards into one forward pass amortizes that fixed
overhead across many positions instead of paying it once per position.

Two batching axes, both configurable:
  1. N concurrent games/trees (`len(boards)` passed to `run_batch`) —
     the natural axis, since self-play already wants many games running
     anyway.
  2. Multiple in-flight leaves *within a single tree*, via virtual loss
     (`leaves_per_tree_per_round`) — lets the batch grow past N even
     with a modest number of concurrent games.

Virtual loss, and why it's needed at all: a single tree's PUCT
selection is deterministic given its current stats, so naively calling
"select a leaf" twice in a row *before* either has been evaluated would
just return the identical leaf both times — there's nothing to make the
second selection go anywhere else, since nothing about the tree has
changed. Virtual loss fixes this by pretending, the instant a leaf is
selected, that it was immediately evaluated and lost (a real network
call hasn't happened yet, but the tree stats are updated as if one had):
this makes that leaf's edge less attractive to a second, concurrent
selection in the same tree, encouraging it to diversify into a
different leaf instead. Once the real (batched) evaluation comes back,
the virtual loss is exactly reverted (see `revert_virtual_loss` below)
and the real value is backed up in its place — so the *final* tree
state after all of a leaf's virtual-then-real backup activity is
identical to what a single real backup would have produced. This is
the standard AlphaZero/AlphaGo Zero tree-parallelization technique.

This module deliberately reuses `select_child`, `backup`,
`expand_children`, `apply_dirichlet_noise`, `terminal_value`, and
`MCTSNode` from training/mcts.py unchanged — same move-selection logic,
same policy/value usage as the existing serial `MCTS`. This file only
adds the batching/virtual-loss orchestration around those primitives.
"""

from typing import Optional

import chess
import numpy as np
import torch

from engine.encoding import action_mask, decode_move, encode_board
from training.mcts import (
    MCTSNode,
    apply_dirichlet_noise,
    backup,
    expand_children,
    select_child,
    terminal_value,
)
from training.network import PolicyValueNet, masked_softmax
from training.tensors import mask_to_tensor, obs_to_tensor

VIRTUAL_LOSS_VALUE = 1.0  # see module docstring for why +1 (from the leaf's own perspective)


def apply_virtual_loss(search_path: list[MCTSNode]) -> None:
    """Pretend `search_path`'s leaf just lost, so a concurrent selection
    in the same tree (before the real evaluation returns) is discouraged
    from picking the identical leaf. Exactly `backup` with a placeholder
    value — reverted later by `revert_virtual_loss`."""
    backup(search_path, VIRTUAL_LOSS_VALUE)


def revert_virtual_loss(search_path: list[MCTSNode]) -> None:
    """Exactly undo `apply_virtual_loss` on the same path: decrement
    visit_count (backup only ever increments) and subtract back the
    same sign-flipped value contributions. After this, followed by a
    real `backup(search_path, real_value)`, the net effect on every
    node's (visit_count, value_sum) is identical to a single ordinary
    backup with `real_value` — see
    tests/test_batched_mcts.py::test_virtual_loss_revert_matches_serial_backup."""
    value = VIRTUAL_LOSS_VALUE
    for node in reversed(search_path):
        node.visit_count -= 1
        node.value_sum -= value
        value = -value


class BatchedMCTS:
    def __init__(
        self,
        network: PolicyValueNet,
        device: torch.device,
        c_puct: float = 1.5,
        claim_draw: bool = True,
        leaves_per_tree_per_round: int = 4,
    ):
        self.network = network
        self.device = device
        self.c_puct = c_puct
        self.claim_draw = claim_draw
        self.leaves_per_tree_per_round = leaves_per_tree_per_round

    def run_batch(
        self,
        boards: list[chess.Board],
        num_simulations: int,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        rng: Optional[np.random.Generator] = None,
    ) -> list[MCTSNode]:
        """Run `num_simulations` PUCT simulations on *each* of
        `boards` (independent positions/trees) concurrently, batching
        leaf evaluations across all of them. Returns one root MCTSNode
        per input board, same order."""
        rng = rng or np.random.default_rng()
        n = len(boards)
        roots = [MCTSNode(parent=None, prior=1.0) for _ in range(n)]

        # Initial expansion of every root, batched into one network call.
        self._batched_evaluate_and_expand(roots, boards)
        if add_dirichlet_noise:
            for root in roots:
                apply_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon, rng)

        remaining = [num_simulations] * n

        while any(r > 0 for r in remaining):
            pending: list[tuple[int, list[MCTSNode], chess.Board]] = []

            for i in range(n):
                if remaining[i] <= 0:
                    continue
                k = min(self.leaves_per_tree_per_round, remaining[i])
                seen_leaves: set[int] = set()

                for _ in range(k):
                    node = roots[i]
                    sim_board = boards[i].copy()
                    search_path = [node]

                    while node.expanded:
                        action, node = select_child(node, self.c_puct)
                        sim_board.push(decode_move(action, sim_board))
                        search_path.append(node)

                    if id(node) in seen_leaves:
                        # Virtual loss didn't diversify this round (e.g. a
                        # forced, low-branching sequence) — stop collecting
                        # for this tree this round rather than double-count
                        # the same not-yet-expanded leaf.
                        break
                    seen_leaves.add(id(node))

                    if sim_board.is_game_over(claim_draw=self.claim_draw):
                        value = terminal_value(sim_board, self.claim_draw)
                        backup(search_path, value)
                        remaining[i] -= 1
                    else:
                        apply_virtual_loss(search_path)
                        pending.append((i, search_path, sim_board))

            if pending:
                self._resolve_pending(pending)
                for i, _search_path, _sim_board in pending:
                    remaining[i] -= 1

        return roots

    def _resolve_pending(
        self, pending: list[tuple[int, list[MCTSNode], chess.Board]]
    ) -> None:
        """One batched network call for every leaf collected this round
        (from every tree), then distribute results back and finalize
        each leaf's backup (revert virtual loss, apply the real value)."""
        boards_np = np.stack([encode_board(b) for _, _, b in pending])
        masks_np = np.stack([action_mask(b) for _, _, b in pending])

        x = obs_to_tensor(boards_np, self.device)
        mask = mask_to_tensor(masks_np, self.device)

        with torch.no_grad():
            policy_logits, values = self.network(x)
            probs = masked_softmax(policy_logits, mask)

        probs = probs.cpu().numpy()
        values = values.cpu().numpy()

        for (_tree_idx, search_path, sim_board), prob_row, value in zip(pending, probs, values):
            leaf = search_path[-1]
            expand_children(leaf, sim_board, prob_row)
            revert_virtual_loss(search_path)
            backup(search_path, float(value))

    def _batched_evaluate_and_expand(
        self, nodes: list[MCTSNode], boards: list[chess.Board]
    ) -> None:
        boards_np = np.stack([encode_board(b) for b in boards])
        masks_np = np.stack([action_mask(b) for b in boards])

        x = obs_to_tensor(boards_np, self.device)
        mask = mask_to_tensor(masks_np, self.device)

        with torch.no_grad():
            policy_logits, _values = self.network(x)
            probs = masked_softmax(policy_logits, mask)

        probs = probs.cpu().numpy()
        for node, board, prob_row in zip(nodes, boards, probs):
            expand_children(node, board, prob_row)
