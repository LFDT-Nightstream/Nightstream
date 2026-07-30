import Nightstream.Implementation.R1CS.Canonical.KMulOwnership

/-!
Contract: equality of two `K` values as emitted rows.

Owns: the row program asserting two carried extension values are equal, its
derived count, soundness, honest completeness, ownership and conservation.

Does not own: the projection identity, which composes two Horner evaluations
with this; the PiRLC batch; or any NIFS structure.

## Two rows, not one

A `K`-valued equality is **two** Goldilocks coordinate equalities. Describing
the projection check as "two evaluations plus one equality row" — which earlier
cycles of this project repeatedly did — undercounts by one row per identity, and
the PiRLC batch checks many.

`rows_length` derives the two from the emitted list rather than asserting it.

## Zero auxiliary columns

Each coordinate is emitted as `left · 1 = right`, reading the constant-one wire
in the `B` operand. No intermediate value is allocated, so an equality costs
rows but no columns. That asymmetry matters for the cost tuple: a
`Typed.Cost` for the projection block cannot assume rows and columns move
together.

The constant-one wire is column `0`, and `z 0 = 1` is a hypothesis rather than
an assumption about the layout — every consumer in this project already
establishes it.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KEquality

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul

/-- One coordinate equality: `left · 1 = right`. -/
def equalityRow (left right : LinComb) : Row where
  a := left
  b := [(0, 1)]
  c := right

/-- **The two rows.**  One per extension coordinate. -/
def rows (left right : Carried) : List Row :=
  [ equalityRow left.low right.low, equalityRow left.high right.high ]

/-- **The derived row count.**  Two, from the emitted list — not one. -/
theorem rows_length (left right : Carried) : (rows left right).length = 2 := rfl

/-- **No auxiliary column is allocated.**  An equality costs rows only. -/
theorem allocates_nothing (left right : Carried) :
    ∀ row ∈ rows left right, row.c = left.low ∨ row.c = left.high
      ∨ row.c = right.low ∨ row.c = right.high := by
  intro row member
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact Or.inr (Or.inr (Or.inl rfl))
  · exact Or.inr (Or.inr (Or.inr rfl))

/-! ## Soundness -/

theorem one_wire (z : Nat → Nat) (constantWire : z 0 = 1) :
    lcEval z [(0, 1)] = 1 := by
  simp [lcEval, constantWire]
  decide

theorem equalityRow_iff
    (z : Nat → Nat) (left right : LinComb) (constantWire : z 0 = 1) :
    RowHolds z (equalityRow left right) ↔ lcEval z left = lcEval z right := by
  unfold RowHolds equalityRow
  rw [one_wire z constantWire, Nat.mul_one, lcEval_eq_rawSum, Nat.mod_mod]

/-- **Satisfaction forces coordinatewise equality.** -/
theorem rows_sound
    (z : Nat → Nat) (left right : Carried) (constantWire : z 0 = 1)
    (satisfied : Satisfies (rows left right) z) :
    lcEval z left.low = lcEval z right.low
      ∧ lcEval z left.high = lcEval z right.high := by
  constructor
  · exact (equalityRow_iff z _ _ constantWire).1
      (satisfied _ (by simp [rows]))
  · exact (equalityRow_iff z _ _ constantWire).1
      (satisfied _ (by simp [rows]))

/-- **Honest completeness.**  Equal values satisfy the rows, and no witness
extension is needed because nothing is allocated. -/
theorem rows_complete
    (z : Nat → Nat) (left right : Carried) (constantWire : z 0 = 1)
    (lowEqual : lcEval z left.low = lcEval z right.low)
    (highEqual : lcEval z left.high = lcEval z right.high) :
    Satisfies (rows left right) z := by
  intro row member
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact (equalityRow_iff z _ _ constantWire).2 lowEqual
  · exact (equalityRow_iff z _ _ constantWire).2 highEqual

/-! ## Ownership and conservation -/

inductive RowOwner where
  | low
  | high
deriving DecidableEq, Repr

def allOwners : List RowOwner := [.low, .high]

theorem allOwners_length : allOwners.length = 2 := rfl

theorem allOwners_nodup : allOwners.Nodup := by decide

def ownedRow (left right : Carried) : RowOwner → Row
  | .low => equalityRow left.low right.low
  | .high => equalityRow left.high right.high

/-- **The emitted program is the receipt list's image.** -/
theorem rows_eq_map_owners (left right : Carried) :
    rows left right = allOwners.map (ownedRow left right) := rfl

/-- **Every column of every emitted row is an operand or the constant wire.**
Nothing else is reachable, and in particular nothing is allocated. -/
theorem rows_conservation
    (left right : Carried) (row : Row) (member : row ∈ rows left right)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column = 0 ∨ Mentions left.low column ∨ Mentions left.high column
      ∨ Mentions right.low column ∨ Mentions right.high column := by
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;> simp only [equalityRow] at mentioned
  · rcases mentioned with a | b | c
    · exact Or.inr (Or.inl a)
    · exact Or.inl (by simpa only [Mentions, List.map_cons, List.map_nil,
        List.mem_singleton] using b)
    · exact Or.inr (Or.inr (Or.inr (Or.inl c)))
  · rcases mentioned with a | b | c
    · exact Or.inr (Or.inr (Or.inl a))
    · exact Or.inl (by simpa only [Mentions, List.map_cons, List.map_nil,
        List.mem_singleton] using b)
    · exact Or.inr (Or.inr (Or.inr (Or.inr c)))

end Nightstream.Implementation.R1CS.Canonical.KEquality
