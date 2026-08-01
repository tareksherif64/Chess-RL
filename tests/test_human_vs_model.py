"""Human-vs-model controller tests. Per the requirement, this drives a
full game end-to-end with a scripted "human" move-picker (uniform
random among legal moves) instead of real mouse clicks — the whole
point of keeping HumanVsModelController tkinter-free is that this is
possible at all. Confirms: no crashes, correct turn alternation
(enforced structurally — see below), and correct game-end detection
(both a real chess-rules ending and the move-limit safety net)."""

from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from gui.human_vs_model_controller import HumanVsModelController
from training.checkpoint import save_checkpoint
from training.network import PolicyValueNet


def _make_test_checkpoint(tmp_path: Path, seed: int = 0) -> Path:
    torch.manual_seed(seed)
    network = PolicyValueNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    path = tmp_path / "test_checkpoint.pt"
    save_checkpoint(path, network, optimizer, iteration=0)
    return path


def _play_scripted_game(controller: HumanVsModelController, rng: np.random.Generator, max_iterations: int = 300):
    """Scripted "human" stub: pick a uniformly random legal move
    whenever it's the human's turn. Returns (human_turns, agent_turns)."""
    human_turns = agent_turns = 0
    for _ in range(max_iterations):
        if controller.is_game_over():
            break

        if controller.is_human_turn():
            legal = list(controller.board.legal_moves)
            move = legal[rng.integers(len(legal))]
            controller.apply_human_move(move)
            human_turns += 1
        else:
            move = controller.compute_agent_move()
            # Correctness of the move itself: must be legal in the
            # *current* position (guards against any stale-board bug).
            assert move in controller.board.legal_moves
            controller.apply_agent_move(move)
            agent_turns += 1

    return human_turns, agent_turns


@pytest.mark.parametrize("human_color", [chess.WHITE, chess.BLACK])
def test_full_scripted_game_no_crash_and_correct_turn_alternation(tmp_path, human_color):
    checkpoint = _make_test_checkpoint(tmp_path)
    controller = HumanVsModelController(
        checkpoint, device=torch.device("cpu"), num_simulations=10,
        human_color=human_color, max_plies=150,
    )

    # Turn alternation is enforced structurally, not just observed:
    # apply_human_move() raises if called when it isn't the human's
    # turn (tested separately below), and _play_scripted_game only
    # ever calls it behind an is_human_turn() check — so a passing run
    # here already proves alternation was correct throughout, not just
    # that no exception happened to occur.
    rng = np.random.default_rng(0)
    human_turns, agent_turns = _play_scripted_game(controller, rng)

    assert controller.is_game_over()
    assert human_turns > 0
    assert agent_turns > 0
    # Plies split between the two sides within 1 of each other, since
    # colors strictly alternate.
    assert abs(human_turns - agent_turns) <= 1

    result_text = controller.get_result_text()
    assert isinstance(result_text, str) and len(result_text) > 0


def test_game_end_detection_reflects_real_chess_outcome(tmp_path):
    """Play into a constructed near-mate position rather than relying
    on random play to stumble into a decisive ending — deterministic,
    fast, and directly exercises get_result_text()'s real-outcome path
    (as opposed to the move-limit path, covered separately below)."""
    checkpoint = _make_test_checkpoint(tmp_path)
    controller = HumanVsModelController(
        checkpoint, device=torch.device("cpu"), num_simulations=10,
        human_color=chess.WHITE, max_plies=600,
    )
    # White (human) delivers Fool's-mate-style checkmate immediately.
    controller.env.reset(options={"fen": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"})
    controller.ply = 0

    mate_move = chess.Move.from_uci("a1a8")
    assert mate_move in controller.board.legal_moves
    controller.apply_human_move(mate_move)

    assert controller.is_game_over()
    assert controller.board.is_checkmate()
    result_text = controller.get_result_text()
    assert "Checkmate" in result_text
    assert "You win" in result_text  # human (white) delivered it


def test_move_limit_truncation_is_reported_distinctly(tmp_path):
    checkpoint = _make_test_checkpoint(tmp_path)
    controller = HumanVsModelController(
        checkpoint, device=torch.device("cpu"), num_simulations=5,
        human_color=chess.WHITE, max_plies=2,
    )
    rng = np.random.default_rng(0)
    _play_scripted_game(controller, rng, max_iterations=10)

    assert controller.is_game_over()
    assert controller.ply >= 2
    assert not controller.board.is_game_over(claim_draw=True)  # truncated, not a real ending
    assert "move limit" in controller.get_result_text().lower()


def test_apply_human_move_rejects_wrong_turn(tmp_path):
    checkpoint = _make_test_checkpoint(tmp_path)
    controller = HumanVsModelController(
        checkpoint, device=torch.device("cpu"), num_simulations=5,
        human_color=chess.BLACK,  # white (agent) moves first
    )
    any_move = next(iter(controller.board.legal_moves))
    with pytest.raises(RuntimeError):
        controller.apply_human_move(any_move)


def test_apply_human_move_rejects_illegal_move(tmp_path):
    checkpoint = _make_test_checkpoint(tmp_path)
    controller = HumanVsModelController(
        checkpoint, device=torch.device("cpu"), num_simulations=5,
        human_color=chess.WHITE,
    )
    illegal_move = chess.Move.from_uci("a1a5")  # rook can't jump the pawn on a2
    with pytest.raises(ValueError):
        controller.apply_human_move(illegal_move)


def test_new_game_resets_board_and_allows_switching_color(tmp_path):
    checkpoint = _make_test_checkpoint(tmp_path)
    controller = HumanVsModelController(
        checkpoint, device=torch.device("cpu"), num_simulations=5,
        human_color=chess.WHITE,
    )
    move = next(iter(controller.board.legal_moves))
    controller.apply_human_move(move)
    assert controller.ply == 1

    controller.new_game(human_color=chess.BLACK)
    assert controller.ply == 0
    assert controller.board == chess.Board()
    assert controller.human_color == chess.BLACK
    assert not controller.is_human_turn()  # white (agent) moves first now


def test_legal_moves_from_matches_python_chess(tmp_path):
    checkpoint = _make_test_checkpoint(tmp_path)
    controller = HumanVsModelController(
        checkpoint, device=torch.device("cpu"), num_simulations=5,
        human_color=chess.WHITE,
    )
    e2 = chess.E2
    expected = {m for m in controller.board.legal_moves if m.from_square == e2}
    assert set(controller.legal_moves_from(e2)) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA-capable GPU")
def test_controller_loads_checkpoint_onto_cuda(tmp_path):
    checkpoint = _make_test_checkpoint(tmp_path)
    controller = HumanVsModelController(
        checkpoint, device=torch.device("cuda"), num_simulations=10,
        human_color=chess.WHITE,
    )
    assert next(controller.network.parameters()).device.type == "cuda"

    move = controller.compute_agent_move() if not controller.is_human_turn() else None
    if move is None:
        # human moves first as White; step once so it's the agent's turn
        controller.apply_human_move(next(iter(controller.board.legal_moves)))
        move = controller.compute_agent_move()
    assert move in controller.board.legal_moves
