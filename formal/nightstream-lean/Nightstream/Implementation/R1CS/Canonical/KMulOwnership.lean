import Nightstream.Implementation.R1CS.Canonical.KMulHonest

/-!
Contract: exact row ownership and conservation for one `K` multiplication.

Owns: the receipt list for the three emitted rows, the proof that the program
is that list's image, and the classification of every column any emitted row
can touch.

Does not own: soundness (`KMul`), completeness (`KMulHonest`), or the
allocator (`KFrames`).

## Ownership is positional

The program *is* the receipt list's image, so position `i` is emitted by
receipt `i` and by no other. That is stronger than proving the three row values
are pairwise distinct, and it is the shape the Poseidon2 track settled on after
`POSEIDON2-ROW-OWNERSHIP-UNIQUENESS` showed that pairwise distinctness of row
*values* does not establish exactly-one-owner.

## Conservation names the operands explicitly

A `K` multiplication reads four operand combinations and writes three frame
columns. `rows_conservation` says every column of every emitted row is one of
those — nothing else is reachable. The `cross` row is the case that needs care:
its operands are `sumComb`, concatenations, so a column it mentions comes from
either coordinate of that side.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KMulOwnership

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul

/-- One receipt per emitted row. -/
inductive RowOwner where
  | lowLow
  | highHigh
  | cross
deriving DecidableEq, Repr

def allOwners : List RowOwner := [.lowLow, .highHigh, .cross]

theorem allOwners_length : allOwners.length = 3 := rfl

theorem allOwners_nodup : allOwners.Nodup := by decide

/-- The row each receipt emits. -/
def ownedRow (left right : Carried) (frame : Frame) : RowOwner → Row
  | .lowLow => productRow left.low right.low frame.lowLow
  | .highHigh => productRow left.high right.high frame.highHigh
  | .cross => productRow (sumComb left) (sumComb right) frame.cross

/-- **The emitted program is the receipt list's image.**  With
`allOwners_nodup` and `allOwners_length`, position `i` is emitted by receipt `i`
and by no other. -/
theorem rows_eq_map_owners (left right : Carried) (frame : Frame) :
    rows left right frame = allOwners.map (ownedRow left right frame) := rfl

/-- Ownership is by position, not by row value. -/
theorem ownership_is_positional
    (left right : Carried) (frame : Frame) (index : Nat)
    (inRange : index < 3) :
    (rows left right frame)[index]?
      = (allOwners.map (ownedRow left right frame))[index]? := by
  rw [rows_eq_map_owners]

/-! ## Conservation -/

/-- A column one of the four operand combinations mentions. -/
def Operand (left right : Carried) (column : Nat) : Prop :=
  Mentions left.low column ∨ Mentions left.high column
    ∨ Mentions right.low column ∨ Mentions right.high column

/-- A column this multiplication allocates. -/
def FrameColumn (frame : Frame) (column : Nat) : Prop :=
  column = frame.lowLow ∨ column = frame.highHigh ∨ column = frame.cross

theorem mentions_sumComb
    (value : Carried) (column : Nat)
    (mentioned : Mentions (sumComb value) column) :
    Mentions value.low column ∨ Mentions value.high column := by
  simp only [Mentions, sumComb, List.map_append, List.mem_append] at mentioned
  exact mentioned

theorem mentions_target (target column : Nat)
    (mentioned : Mentions [(target, 1)] column) : column = target := by
  simpa only [Mentions, List.map_cons, List.map_nil,
    List.mem_singleton] using mentioned

/-- **Every column of every emitted row is an operand or this frame.**  No row
reaches outside the four combinations it reads and the three columns it
writes. -/
theorem rows_conservation
    (left right : Carried) (frame : Frame) (row : Row)
    (member : row ∈ rows left right frame) (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    Operand left right column ∨ FrameColumn frame column := by
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl <;>
    simp only [productRow] at mentioned
  · rcases mentioned with a | b | c
    · exact Or.inl (Or.inl a)
    · exact Or.inl (Or.inr (Or.inr (Or.inl b)))
    · exact Or.inr (Or.inl (mentions_target _ _ c))
  · rcases mentioned with a | b | c
    · exact Or.inl (Or.inr (Or.inl a))
    · exact Or.inl (Or.inr (Or.inr (Or.inr b)))
    · exact Or.inr (Or.inr (Or.inl (mentions_target _ _ c)))
  · rcases mentioned with a | b | c
    · rcases mentions_sumComb left column a with low | high
      · exact Or.inl (Or.inl low)
      · exact Or.inl (Or.inr (Or.inl high))
    · rcases mentions_sumComb right column b with low | high
      · exact Or.inl (Or.inr (Or.inr (Or.inl low)))
      · exact Or.inl (Or.inr (Or.inr (Or.inr high)))
    · exact Or.inr (Or.inr (Or.inr (mentions_target _ _ c)))

/-- **Each receipt writes its own column and no other's.**  This is what makes
the three frame columns exactly-owned rather than merely allocated. -/
theorem ownedRow_target
    (left right : Carried) (frame : Frame) (owner : RowOwner) (column : Nat)
    (mentioned : Mentions (ownedRow left right frame owner).c column) :
    column = (match owner with
      | .lowLow => frame.lowLow
      | .highHigh => frame.highHigh
      | .cross => frame.cross) := by
  cases owner <;>
    exact mentions_target _ _ mentioned

end Nightstream.Implementation.R1CS.Canonical.KMulOwnership
