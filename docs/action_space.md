---
module: engine (encoding.py action-space section)
depends_on: [python-chess]
depended_on_by: [engine/chess_env.py, agents, tests]
---

# Action space encoding

Chess doesn't have a small fixed action vocabulary the way e.g. Atari does
(18 buttons). A move is (from-square, to-square, optional promotion piece),
and the naive "all (from, to) pairs" encoding is 64x64 = 4096, which
*undercounts* (doesn't represent underpromotion choices) while also
*overcounting* (includes geometrically impossible pairs like a1->h8 via a
knight-shaped jump). We need a scheme that's a fixed size (required for a
`Discrete` action space / a policy network's output layer), covers every
legal move exactly once, and is decodable back to a `chess.Move` without
ambiguity.

## Chosen scheme: AlphaZero's 8x8x73 = 4672

For each of the 64 **from-squares**, 73 **move-type planes**:

| Planes | Count | Covers |
|---|---|---|
| 0–55 | 56 | 8 directions (N, NE, E, SE, S, SW, W, NW) x 7 distances (1–7) |
| 56–63 | 8 | 8 knight-move deltas |
| 64–72 | 9 | 3 pawn-underpromotion directions x 3 promotion pieces (N/B/R) |

`action = from_square * 73 + plane`, giving `64 * 73 = 4672` total actions.

**Why the 56 "queen move" planes cover far more than queens:** a king move,
a rook slide, a bishop slide, a pawn push, a pawn capture, and a castling
move (the king's two-square slide) are all geometrically just "move N
squares in one of 8 straight-line directions" — they only differ in which
distances are *legal* for that piece, and legality is python-chess's job,
not the encoding's. The encoding only needs to be able to *represent* the
move; a fixed-size scheme necessarily has unreachable combinations for any
given board (e.g. plane "slide 7 squares" from a square with a pawn on it),
and that's fine — `action_mask()` zeroes those out per-position.

**Why underpromotion needs its own 9 planes:** queen promotion is just a
1-square forward/diagonal pawn move, already covered by the 56 queen-move
planes (decode attaches `promotion=QUEEN` automatically when a pawn move's
target is a pawn move landing on the back rank — see `decode_move`).
Under-promoting to knight/bishop/rook is a real, sometimes necessary
choice (e.g. underpromoting to a knight to deliver check or avoid
stalemate) that the queen-move planes can't distinguish, since they encode
geometry only, not the promotion piece.

**Castling, en passant:** no special-casing needed. Castling is encoded as
the king's own 2-square horizontal slide (plane within 0–55); the rook's
jump is handled by `board.push()`, not by the action encoding. En passant
is encoded as an ordinary diagonal pawn move (from/to only); python-chess
recognizes it as en passant from board context when the move is pushed.

## Encode / decode contract

- `encode_move(move: chess.Move) -> int` — pure function of the move
  object itself (uses `move.promotion` for the underpromotion branch).
  No board needed.
- `decode_move(action: int, board: chess.Board) -> chess.Move` — needs
  the board only to disambiguate one case: whether a queen-direction,
  distance-1, forward move by a pawn onto the back rank should carry
  `promotion=QUEEN`. Every other branch is pure geometry.

Both directions are covered by `tests/test_encoding.py`, including a
round-trip check across ~100+ positions sampled from 15 random games
(so real games, not just synthetic FENs, exercise the encoding) and
explicit checks for underpromotion, castling, and en passant.

## Alternatives considered

- **64x64 (+ promotion piece) = 4096 x up-to-4 promotion variants.**
  Simpler to reason about (from, to) directly, but doesn't map cleanly to
  a single `Discrete` action space without a secondary promotion-piece
  head, and still has ~1900+ geometrically-impossible entries wasted per
  from-square in the 4096 base — not meaningfully more compact than
  4672, just less standard and less proven for this exact problem.
- **Move-list-index encoding** (rank moves 0..len(legal_moves)-1 per
  position). Rejected: the action space would change shape every
  position, which breaks a fixed-size policy network output and makes
  masking/exploration bookkeeping in most RL algorithms awkward.

4672 was chosen because it's the scheme AlphaZero/Leela-style chess
engines use and is well-understood, and because it decouples the action
space from any particular position (needed for a fixed network output
layer later).
