import Nightstream.Implementation.R1CS.Canonical.KEquality
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: the emitted row program for Π_DEC's pairwise consistency checks.

Owns: the concatenated equality program over a list of value pairs, its folded
row count, soundness, honest completeness, and cost.

## One generic program

Each PiDEC consistency obligation reduces to "these two carried values agree".
One folded equality program serves all such pairs. Which pairs to check is
decoder work and belongs to the claim-shape owner; this module owns only the
rows.

## Two rows per pair

`KEquality` already records the fact that matters: an equality in `K` is **two**
physical rows, one per coordinate, and it allocates nothing. So the receipt is
`2 · pairs.length` rows and an empty allocation, and the column count stays zero
however many pairs there are.

## The cost is a fold

The pair count comes from a claim's shape, not from a protocol constant, so the
row count is stated as a fold over per-pair receipts first and evaluated
second. A closed formula alone would be a subtotal presented as a total.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KConsistency

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner

/-- **The emitted consistency program.**  One `K`-equality per pair. -/
def consistencyRows (pairs : List (Carried × Carried)) : List Row :=
  pairs.flatMap (fun pair => KEquality.rows pair.1 pair.2)

/-- **The derived row count, as a fold over per-pair receipts.** -/
theorem consistencyRows_length (pairs : List (Carried × Carried)) :
    (consistencyRows pairs).length = (pairs.map (fun _ => 2)).sum := by
  unfold consistencyRows
  rw [List.length_flatMap]
  exact congrArg List.sum
    (List.map_congr_left (fun pair _ => KEquality.rows_length pair.1 pair.2))

/-- Two rows per pair, once the fold is evaluated. -/
theorem consistencyRows_length_eq (pairs : List (Carried × Carried)) :
    (consistencyRows pairs).length = 2 * pairs.length := by
  rw [consistencyRows_length]
  induction pairs with
  | nil => rfl
  | cons pair rest inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis]
      omega

/-- **Satisfaction forces every pair to agree.** -/
theorem consistencyRows_sound
    (z : Nat → Nat) (pairs : List (Carried × Carried)) (constantWire : z 0 = 1)
    (satisfied : Satisfies (consistencyRows pairs) z)
    (pair : Carried × Carried) (member : pair ∈ pairs) :
    carriedValue z pair.1 = carriedValue z pair.2 := by
  have rowsSat : Satisfies (KEquality.rows pair.1 pair.2) z :=
    fun row rowMember =>
      satisfied row (List.mem_flatMap.2 ⟨pair, member, rowMember⟩)
  rcases KEquality.rows_sound z pair.1 pair.2 constantWire rowsSat with
    ⟨lowEqual, highEqual⟩
  unfold carriedValue
  simp only [Pair.mk.injEq]
  exact ⟨lowEqual, highEqual⟩

/-- **Agreeing pairs satisfy the check**, under the caller's own assignment.
Nothing is allocated, so no witness extension is needed. -/
theorem consistencyRows_honest
    (z : Nat → Nat) (pairs : List (Carried × Carried)) (constantWire : z 0 = 1)
    (agree : ∀ pair ∈ pairs, carriedValue z pair.1 = carriedValue z pair.2) :
    Satisfies (consistencyRows pairs) z := by
  intro row member
  rcases List.mem_flatMap.1 member with ⟨pair, pairMember, rowMember⟩
  have equal := agree pair pairMember
  unfold carriedValue at equal
  simp only [Pair.mk.injEq] at equal
  exact KEquality.rows_complete z pair.1 pair.2 constantWire equal.1 equal.2
    row rowMember

/-- **Every column belongs to some compared pair**, or is the constant wire. -/
theorem consistencyRows_conservation
    (pairs : List (Carried × Carried)) (row : Row)
    (member : row ∈ consistencyRows pairs) (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    ∃ pair ∈ pairs,
      column = 0 ∨ Mentions pair.1.low column ∨ Mentions pair.1.high column
        ∨ Mentions pair.2.low column ∨ Mentions pair.2.high column := by
  unfold consistencyRows at member
  rcases List.mem_flatMap.1 member with ⟨pair, pairMember, rowMember⟩
  exact ⟨pair, pairMember,
    KEquality.rows_conservation pair.1 pair.2 row rowMember column mentioned⟩

/-- **The check's cost**, folded over pairs.  `KEquality` allocates nothing, so
the auxiliary component stays zero at every arity. -/
def consistencyCost (pairs : List (Carried × Carried)) : Lowering.Typed.Cost where
  recurringRows := 2 * pairs.length
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem consistencyCost_rows (pairs : List (Carried × Carried)) :
    (consistencyRows pairs).length = (consistencyCost pairs).recurringRows :=
  consistencyRows_length_eq pairs

end Nightstream.Implementation.R1CS.Canonical.KConsistency
