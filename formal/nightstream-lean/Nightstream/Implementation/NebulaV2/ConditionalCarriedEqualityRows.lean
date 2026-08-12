import Nightstream.Implementation.R1CS.Canonical.KMulChain
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.LinearEquality

/-!
Contract: equality of two carried extension-field values enabled exactly when
a Boolean phase wire is zero.

Assurance tier: implementation model.

Owns two gated subtraction rows, sound coordinate equality in phase zero,
and honest completeness in phase zero and one.

Does not own the Boolean phase row, the carried computations, or phase
semantics.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ConditionalCarriedEqualityRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Program

private theorem pair_ext
    {left right : Pair} (low : left.low = right.low)
    (high : left.high = right.high) : left = right := by
  cases left
  cases right
  simp_all

def coordinateRow (phaseColumn : Nat) (left right : LinComb) : Row :=
  ⟨left ++ negateTerms right,
    [(0, 1), (phaseColumn, goldilocksP - 1)], []⟩

def rows (phaseColumn : Nat) (left right : Carried) : List Row :=
  [coordinateRow phaseColumn left.low right.low,
    coordinateRow phaseColumn left.high right.high]

theorem rows_length (phaseColumn : Nat) (left right : Carried) :
    (rows phaseColumn left right).length = 2 := rfl

private theorem coordinate_ungated
    {phaseColumn : Nat} {left right : LinComb}
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (phaseClosed : assignment phaseColumn = 0)
    (holds : RowHolds assignment
      (coordinateRow phaseColumn left right)) :
    RowHolds assignment ⟨left ++ negateTerms right, [(0, 1)], []⟩ := by
  simpa [coordinateRow, RowHolds, lcEval, one, phaseClosed,
    goldilocksP] using holds

private theorem coordinate_sound
    {phaseColumn : Nat} {left right : LinComb}
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (phaseClosed : assignment phaseColumn = 0)
    (rightCanonical : CanonicalTerms right)
    (holds : RowHolds assignment
      (coordinateRow phaseColumn left right)) :
    lcEval assignment left = lcEval assignment right :=
  Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.LinearEquality.sound
    one left right rightCanonical
      (coordinate_ungated one phaseClosed holds)

theorem rows_sound_closed
    {phaseColumn : Nat} {left right : Carried}
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (phaseClosed : assignment phaseColumn = 0)
    (rightCanonicalLow : CanonicalTerms right.low)
    (rightCanonicalHigh : CanonicalTerms right.high)
    (holds : Satisfies (rows phaseColumn left right) assignment) :
    carriedValue assignment left = carriedValue assignment right := by
  apply pair_ext
  · exact coordinate_sound one phaseClosed rightCanonicalLow
      (holds _ (by simp [rows]))
  · exact coordinate_sound one phaseClosed rightCanonicalHigh
      (holds _ (by simp [rows]))

private theorem coordinate_complete_closed
    {phaseColumn : Nat} {left right : LinComb}
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (phaseClosed : assignment phaseColumn = 0)
    (rightCanonical : CanonicalTerms right)
    (equal : lcEval assignment left = lcEval assignment right) :
    RowHolds assignment (coordinateRow phaseColumn left right) := by
  have cancel := lcEval_append_negateTerms_eq_zero assignment right
    rightCanonical
  rw [lcEval_append] at cancel
  have difference :
      lcEval assignment (left ++ negateTerms right) = 0 := by
    rw [lcEval_append, equal]
    exact cancel
  simpa [coordinateRow, RowHolds, lcEval, one, phaseClosed,
    goldilocksP] using difference

theorem rows_complete_closed
    {phaseColumn : Nat} {left right : Carried}
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (phaseClosed : assignment phaseColumn = 0)
    (rightCanonicalLow : CanonicalTerms right.low)
    (rightCanonicalHigh : CanonicalTerms right.high)
    (equal : carriedValue assignment left = carriedValue assignment right) :
    Satisfies (rows phaseColumn left right) assignment := by
  intro row member
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact coordinate_complete_closed one phaseClosed rightCanonicalLow
      (congrArg Pair.low equal)
  · exact coordinate_complete_closed one phaseClosed rightCanonicalHigh
      (congrArg Pair.high equal)

theorem rows_complete_active
    {phaseColumn : Nat} {left right : Carried}
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (phaseActive : assignment phaseColumn = 1) :
    Satisfies (rows phaseColumn left right) assignment := by
  intro row member
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;>
    simp [coordinateRow, RowHolds, lcEval, one, phaseActive, goldilocksP]

end Nightstream.Implementation.NebulaV2.ConditionalCarriedEqualityRows
