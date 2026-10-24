import CompressionLean.Lib.Cube
import Mathlib.Data.Nat.Size

/-!
Owns the port of the carry walk `diagonal`
(`crates/neo-spartan/src/shrink/layout.rs:230-279`).

The walk reads the bits of the block index `b`, low bit first. A state is a
carry mask, one bit per track, and whether the low bits of `b` are below
those of `count`.

`Shrink/Diagonal.lean` proves its value.
-/

namespace CompressionLean.Shrink

open CompressionLean.Lib

variable {K : Type} [CommRing K]

/-- A track of the walk: a point and an offset (`layout.rs:234`). -/
abbrev Track (K : Type) := List K × ℕ

/-- The bits the walk reads (`layout.rs:235-240`): the longest point, and
the bit lengths of the offsets and of `count`. -/
def walkBits (tracks : List (Track K)) (count : ℕ) : ℕ :=
  (tracks.map fun track => max track.1.length (Nat.size track.2)).foldr max (Nat.size count)

/-- One track's factor at bit `t` when its value has bit `bit` there
(`layout.rs:265-270`): `x_t` or `1 - x_t` inside the point. Past the point a
zero bit gives `1` and a one bit gives `0`; Rust drops that branch
(`continue 'bit`), which adds nothing. -/
def trackFactor (point : List K) (t bit : ℕ) : K :=
  if t < point.length then (if bit = 1 then point.getD t 0 else 1 - point.getD t 0)
  else if bit = 1 then 0 else 1

/-- Every track at bit `t` for the block bit `b` and the carry mask `carry`
(`layout.rs:259-271`): the product of the track factors and the next carry
mask. Track `i` owns bit `i` of the mask. -/
def tracksStep (t b : ℕ) : List (Track K) → ℕ → K × ℕ
  | [], _ => (1, 0)
  | (point, offset) :: rest, carry =>
      (trackFactor point t ((offset / 2 ^ t % 2 + b + carry % 2) % 2) * (tracksStep t b rest (carry / 2)).1,
        (offset / 2 ^ t % 2 + b + carry % 2) / 2 + 2 * (tracksStep t b rest (carry / 2)).2)

/-- `next_below` (`layout.rs:254-258`). -/
def nextBelow (countBit b below : ℕ) : ℕ :=
  if b < countBit then 1 else if countBit < b then 0 else below

/-- One bit of the walk (`layout.rs:246-276`): each state `(carry, below)`
and each block bit `b` add `value·factor` to the state they lead to. Rust
skips zero values (`layout.rs:250-252`), which add nothing. -/
def walkStep (tracks : List (Track K)) (count t : ℕ) (states : ℕ → ℕ → K) (carry' below' : ℕ) : K :=
  ∑ carry ∈ Finset.range (2 ^ tracks.length), ∑ below ∈ Finset.range 2, ∑ b ∈ Finset.range 2,
    if (tracksStep t b tracks carry).2 = carry' ∧ nextBelow (count / 2 ^ t % 2) b below = below' then
      states carry below * (tracksStep t b tracks carry).1
    else 0

/-- The states after the low `t` bits, from `states[0][0] = 1`
(`layout.rs:243-245`). -/
def walk (tracks : List (Track K)) (count : ℕ) : ℕ → ℕ → ℕ → K
  | 0 => fun carry below => if carry = 0 ∧ below = 0 then 1 else 0
  | t + 1 => walkStep tracks count t (walk tracks count t)

/-- `diagonal` (`layout.rs:230-279`): the state with no carry and `b` below
`count`, after every bit. -/
def diagonal (tracks : List (Track K)) (count : ℕ) : K :=
  walk tracks count (walkBits tracks count) 0 1

end CompressionLean.Shrink
