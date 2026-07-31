"""AlphaZero-style MCTS: PUCT selection guided by the network's policy
priors, leaf evaluation by the network's value head instead of random
rollouts, and backup that alternates sign every ply (chess is a
zero-sum, perfect-information, alternating-turn game).

A single `chess.Board` is copied once per simulation and pushed/popped
along the selection path — no per-node board storage, no per-node deep
copies, keeping memory flat as simulation count grows.
"""

import math
from typing import Optional

import chess
import numpy as np
import torch

from engine.encoding import ACTION_SPACE_SIZE, action_mask, decode_move, encode_board, encode_move
from training.network import PolicyValueNet, masked_softmax
from training.tensors import mask_to_tensor, obs_to_tensor


class MCTSNode:
    __slots__ = ("parent", "prior", "visit_count", "value_sum", "children")

    def __init__(self, parent: Optional["MCTSNode"], prior: float):
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "MCTSNode"] = {}

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0


def terminal_value(board: chess.Board, claim_draw: bool) -> float:
    """Value from the perspective of the side to move at a terminal
    board (the side with no move to make): -1 if they're checkmated,
    0 for any draw. Matches ChessEnv's own terminal reward convention."""
    outcome = board.outcome(claim_draw=claim_draw)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


class MCTS:
    def __init__(
        self,
        network: PolicyValueNet,
        device: torch.device,
        c_puct: float = 1.5,
        claim_draw: bool = True,
    ):
        self.network = network
        self.device = device
        self.c_puct = c_puct
        self.claim_draw = claim_draw

    def run(self, board: chess.Board, num_simulations: int) -> MCTSNode:
        """Run `num_simulations` PUCT simulations from `board`'s current
        position and return the root node (its children's visit_count
        is the search-improved policy signal)."""
        root = MCTSNode(parent=None, prior=1.0)
        self._evaluate_and_expand(root, board)

        for _ in range(num_simulations):
            node = root
            sim_board = board.copy()
            search_path = [node]

            while node.expanded:
                action, node = self._select_child(node)
                sim_board.push(decode_move(action, sim_board))
                search_path.append(node)

            if sim_board.is_game_over(claim_draw=self.claim_draw):
                value = terminal_value(sim_board, self.claim_draw)
            else:
                value = self._evaluate_and_expand(node, sim_board)

            self._backup(search_path, value)

        return root

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        parent_visits_sqrt = math.sqrt(max(node.visit_count, 1))
        best_score = -float("inf")
        best_action, best_child = None, None

        for action, child in node.children.items():
            q = -child.value
            u = self.c_puct * child.prior * parent_visits_sqrt / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score, best_action, best_child = score, action, child

        return best_action, best_child

    def _evaluate_and_expand(self, node: MCTSNode, board: chess.Board) -> float:
        """Run the network on `board`, attach one child per legal move
        with its network-policy prior, and return the position's value
        (from the perspective of `board.turn`)."""
        obs = encode_board(board)
        mask_np = action_mask(board)

        x = obs_to_tensor(obs, self.device)
        mask = mask_to_tensor(mask_np, self.device)

        with torch.no_grad():
            policy_logits, value = self.network(x)
            probs = masked_softmax(policy_logits, mask)

        probs = probs[0].cpu().numpy()

        for move in board.legal_moves:
            action = encode_move(move)
            node.children[action] = MCTSNode(parent=node, prior=float(probs[action]))

        return float(value.item())

    @staticmethod
    def _backup(search_path: list[MCTSNode], value: float) -> None:
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value


def visit_count_policy(
    root: MCTSNode, num_actions: int = ACTION_SPACE_SIZE, temperature: float = 1.0
) -> np.ndarray:
    """Search-improved policy target: root children's visit counts,
    normalized. temperature < 1 sharpens toward the most-visited move,
    temperature > 1 flattens. Used as the training target `pi` in
    Stage 3 self-play."""
    policy = np.zeros(num_actions, dtype=np.float32)
    if not root.children:
        return policy

    actions = list(root.children.keys())
    visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)

    if temperature == 0:
        best = actions[int(np.argmax(visits))]
        policy[best] = 1.0
        return policy

    weighted = visits ** (1.0 / temperature)
    weighted /= weighted.sum()
    for a, p in zip(actions, weighted):
        policy[a] = p
    return policy


def select_action(root: MCTSNode, temperature: float = 0.0) -> int:
    """Pick an action from the searched root: temperature=0 is greedy
    argmax over visit counts (deterministic — used for evaluation/play);
    temperature>0 samples proportional to visit_count^(1/temperature)
    (used during self-play for exploration)."""
    actions = list(root.children.keys())
    visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)

    if temperature == 0:
        return actions[int(np.argmax(visits))]

    weighted = visits ** (1.0 / temperature)
    weighted /= weighted.sum()
    return int(np.random.choice(actions, p=weighted))
