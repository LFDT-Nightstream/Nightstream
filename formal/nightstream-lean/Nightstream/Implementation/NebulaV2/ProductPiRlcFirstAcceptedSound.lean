import Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedRows
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: row-derived soundness of the V2 first-accepted selector.

Given three Boolean accept inputs and three residues in `0..4`, satisfaction
of the exact nine rows proves that:

* at least one attempt is accepted;
* the output is the residue of the first accepted attempt; and
* the output is also in `0..4`.

The fail-closed conclusion comes from the success row. Availability is not an
assumption.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedRows

def AcceptBits (assignment : Nat -> Nat) (layout : Layout) : Prop :=
  forall attempt, assignment (layout.accept attempt) ≤ 1

def ResiduesInRange (assignment : Nat -> Nat) (layout : Layout) : Prop :=
  forall attempt, assignment (layout.residue attempt) < 5

private theorem singleton_eval
    (assignment : Nat -> Nat) (column : Nat)
    (canonical : assignment column < goldilocksP) :
    lcEval assignment [(column, 1)] = assignment column := by
  simp [lcEval, Nat.mod_eq_of_lt canonical]

private theorem one_eval
    (assignment : Nat -> Nat) (one : assignment 0 = 1) :
    lcEval assignment [(0, 1)] = 1 := by
  simp [lcEval, one, goldilocksP]

private theorem oneMinus_eval
    {assignment : Nat -> Nat} (one : assignment 0 = 1)
    {column : Nat} (bit : assignment column ≤ 1) :
    lcEval assignment (oneMinus column) = 1 - assignment column := by
  have cases : assignment column = 0 ∨ assignment column = 1 := by omega
  rcases cases with zero | oneValue
  · simp [oneMinus, lcEval, one, zero, goldilocksP]
  · simp [oneMinus, lcEval, one, oneValue, goldilocksP]

private theorem mul_bits_le_one {left right : Nat}
    (leftLe : left ≤ 1) (rightLe : right ≤ 1) :
    left * right ≤ 1 := by
  exact Nat.le_trans (Nat.mul_le_mul leftLe rightLe) (by norm_num)

private theorem mul_bit_residue_le_four {bit residue : Nat}
    (bitLe : bit ≤ 1) (residueLt : residue < 5) :
    bit * residue ≤ 4 := by
  have residueLe : residue ≤ 4 := by omega
  exact Nat.le_trans (Nat.mul_le_mul bitLe residueLe) (by norm_num)

private theorem selection_sum_eval
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (firstLe : assignment (selectFirstColumn layout) ≤ 1)
    (secondLe : assignment (selectSecondColumn layout) ≤ 1)
    (thirdLe : assignment (selectThirdColumn layout) ≤ 1) :
    lcEval assignment
        [(selectFirstColumn layout, 1), (selectSecondColumn layout, 1),
          (selectThirdColumn layout, 1)] =
      assignment (selectFirstColumn layout) +
        assignment (selectSecondColumn layout) +
        assignment (selectThirdColumn layout) := by
  unfold lcEval
  simp only [List.foldl, Nat.one_mul, Nat.zero_add]
  rw [Nat.mod_eq_of_lt]
  have small : 3 < goldilocksP := by decide
  omega

private theorem product_sum_eval
    {assignment : Nat -> Nat} {layout : Layout}
    (firstLe : assignment (productColumn layout 0) ≤ 4)
    (secondLe : assignment (productColumn layout 1) ≤ 4)
    (thirdLe : assignment (productColumn layout 2) ≤ 4) :
    lcEval assignment
        [(productColumn layout 0, 1), (productColumn layout 1, 1),
          (productColumn layout 2, 1)] =
      assignment (productColumn layout 0) +
        assignment (productColumn layout 1) +
        assignment (productColumn layout 2) := by
  unfold lcEval
  simp only [List.foldl, Nat.one_mul, Nat.zero_add]
  rw [Nat.mod_eq_of_lt]
  have small : 12 < goldilocksP := by decide
  omega

theorem select_first_eq
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (selectFirstColumn layout) =
      assignment (layout.accept first) := by
  have holds := satisfied (selectFirstRow layout) (by simp [rows])
  rw [RowHolds, selectFirstRow,
    singleton_eval assignment _ (canonical _), one_eval assignment one,
    singleton_eval assignment _ (canonical _)] at holds
  simpa [Nat.mod_eq_of_lt (canonical _)] using holds

theorem select_second_eq
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bits : AcceptBits assignment layout)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (selectSecondColumn layout) =
      (1 - assignment (layout.accept first)) *
        assignment (layout.accept second) := by
  have holds := satisfied (selectSecondRow layout) (by simp [rows])
  rw [RowHolds, selectSecondRow,
    oneMinus_eval one (bits first),
    singleton_eval assignment _ (canonical _),
    singleton_eval assignment _ (canonical _)] at holds
  have productLt :
      (1 - assignment (layout.accept first)) *
          assignment (layout.accept second) < goldilocksP := by
    have firstLe : 1 - assignment (layout.accept first) ≤ 1 := by omega
    have productLe := mul_bits_le_one firstLe (bits second)
    exact Nat.lt_of_le_of_lt productLe (by decide : 1 < goldilocksP)
  rw [Nat.mod_eq_of_lt productLt] at holds
  exact holds.symm

theorem reject_first_two_eq
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bits : AcceptBits assignment layout)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (rejectFirstTwoColumn layout) =
      (1 - assignment (layout.accept first)) *
        (1 - assignment (layout.accept second)) := by
  have holds := satisfied (rejectFirstTwoRow layout) (by simp [rows])
  rw [RowHolds, rejectFirstTwoRow,
    oneMinus_eval one (bits first), oneMinus_eval one (bits second),
    singleton_eval assignment _ (canonical _)] at holds
  have productLt :
      (1 - assignment (layout.accept first)) *
          (1 - assignment (layout.accept second)) < goldilocksP := by
    have firstLe : 1 - assignment (layout.accept first) ≤ 1 := by omega
    have secondLe : 1 - assignment (layout.accept second) ≤ 1 := by omega
    have productLe := mul_bits_le_one firstLe secondLe
    exact Nat.lt_of_le_of_lt productLe (by decide : 1 < goldilocksP)
  rw [Nat.mod_eq_of_lt productLt] at holds
  exact holds.symm

theorem select_third_eq
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bits : AcceptBits assignment layout)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (selectThirdColumn layout) =
      (1 - assignment (layout.accept first)) *
        (1 - assignment (layout.accept second)) *
        assignment (layout.accept third) := by
  have rejectEq := reject_first_two_eq canonical one bits satisfied
  have holds := satisfied (selectThirdRow layout) (by simp [rows])
  rw [RowHolds, selectThirdRow,
    singleton_eval assignment _ (canonical _),
    singleton_eval assignment _ (canonical _),
    singleton_eval assignment _ (canonical _)] at holds
  have rejectLe : assignment (rejectFirstTwoColumn layout) ≤ 1 := by
    rw [rejectEq]
    exact mul_bits_le_one (by omega) (by omega)
  have productLt :
      assignment (rejectFirstTwoColumn layout) *
          assignment (layout.accept third) < goldilocksP := by
    have productLe := mul_bits_le_one rejectLe (bits third)
    exact Nat.lt_of_le_of_lt productLe (by decide : 1 < goldilocksP)
  rw [Nat.mod_eq_of_lt productLt] at holds
  rw [← holds, rejectEq]

private theorem selection_bounds
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bits : AcceptBits assignment layout)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (selectFirstColumn layout) ≤ 1 ∧
      assignment (selectSecondColumn layout) ≤ 1 ∧
      assignment (selectThirdColumn layout) ≤ 1 := by
  constructor
  · rw [select_first_eq canonical one satisfied]
    exact bits first
  · constructor
    · rw [select_second_eq canonical one bits satisfied]
      exact mul_bits_le_one (by omega) (bits second)
    · rw [select_third_eq canonical one bits satisfied]
      exact mul_bits_le_one
        (mul_bits_le_one (by omega) (by omega)) (bits third)

theorem success_eq
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bits : AcceptBits assignment layout)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (selectFirstColumn layout) +
        assignment (selectSecondColumn layout) +
        assignment (selectThirdColumn layout) = 1 := by
  have bounds := selection_bounds canonical one bits satisfied
  have holds := satisfied (successRow layout) (by simp [rows])
  rw [RowHolds, successRow,
    selection_sum_eval canonical bounds.1 bounds.2.1 bounds.2.2,
    one_eval assignment one] at holds
  have sumLt :
      assignment (selectFirstColumn layout) +
          assignment (selectSecondColumn layout) +
          assignment (selectThirdColumn layout) < goldilocksP := by
    have small : 4 < goldilocksP := by decide
    omega
  simp only [Nat.mul_one] at holds
  change
    (assignment (selectFirstColumn layout) +
      assignment (selectSecondColumn layout) +
      assignment (selectThirdColumn layout)) % goldilocksP = 1 at holds
  rw [Nat.mod_eq_of_lt sumLt] at holds
  exact holds

/-- The success row makes three rejections impossible. -/
theorem available
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bits : AcceptBits assignment layout)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (layout.accept first) = 1 ∨
      assignment (layout.accept second) = 1 ∨
      assignment (layout.accept third) = 1 := by
  by_cases firstAccepted : assignment (layout.accept first) = 1
  · exact Or.inl firstAccepted
  · by_cases secondAccepted : assignment (layout.accept second) = 1
    · exact Or.inr (Or.inl secondAccepted)
    · by_cases thirdAccepted : assignment (layout.accept third) = 1
      · exact Or.inr (Or.inr thirdAccepted)
      · have firstZero : assignment (layout.accept first) = 0 := by
          have := bits first
          omega
        have secondZero : assignment (layout.accept second) = 0 := by
          have := bits second
          omega
        have thirdZero : assignment (layout.accept third) = 0 := by
          have := bits third
          omega
        have success := success_eq canonical one bits satisfied
        rw [select_first_eq canonical one satisfied,
          select_second_eq canonical one bits satisfied,
          select_third_eq canonical one bits satisfied] at success
        simp [firstZero, secondZero, thirdZero] at success

private theorem product_eq
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (satisfied : Satisfies (rows layout) assignment)
    (attempt : Fin attemptCount)
    (selectedLe : assignment (selectedColumns layout attempt) ≤ 1)
    (residueLt : assignment (layout.residue attempt) < 5) :
    assignment (productColumn layout attempt.val) =
      assignment (selectedColumns layout attempt) *
        assignment (layout.residue attempt) := by
  have holds := satisfied (productRow layout attempt) (by
    simp [rows, productRows])
  rw [RowHolds, productRow,
    singleton_eval assignment _ (canonical _),
    singleton_eval assignment _ (canonical _),
    singleton_eval assignment _ (canonical _)] at holds
  have productLt :
      assignment (selectedColumns layout attempt) *
          assignment (layout.residue attempt) < goldilocksP := by
    have productLe := mul_bit_residue_le_four selectedLe residueLt
    exact Nat.lt_of_le_of_lt productLe (by decide : 4 < goldilocksP)
  rw [Nat.mod_eq_of_lt productLt] at holds
  exact holds.symm

private theorem output_sum_eq
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (productsLe : forall index, index < 3 ->
      assignment (productColumn layout index) ≤ 4)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (outputColumn layout) =
      assignment (productColumn layout 0) +
        assignment (productColumn layout 1) +
        assignment (productColumn layout 2) := by
  have holds := satisfied (outputRow layout) (by simp [rows])
  rw [RowHolds, outputRow,
    singleton_eval assignment _ (canonical _), one_eval assignment one,
    product_sum_eval (productsLe 0 (by decide))
      (productsLe 1 (by decide)) (productsLe 2 (by decide))] at holds
  simpa [Nat.mod_eq_of_lt (canonical _)] using holds

/-- The output is the residue of the first accepted attempt. -/
theorem output_first_accepted
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bits : AcceptBits assignment layout)
    (residues : ResiduesInRange assignment layout)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (outputColumn layout) =
      if assignment (layout.accept first) = 1 then
        assignment (layout.residue first)
      else if assignment (layout.accept second) = 1 then
        assignment (layout.residue second)
      else assignment (layout.residue third) := by
  have s0 := select_first_eq canonical one satisfied
  have s1 := select_second_eq canonical one bits satisfied
  have s2 := select_third_eq canonical one bits satisfied
  have selectionLe := selection_bounds canonical one bits satisfied
  have p0 : assignment (productColumn layout 0) =
      assignment (selectFirstColumn layout) *
        assignment (layout.residue first) := by
    simpa [selectedColumns, first] using
      product_eq canonical satisfied first
        (by simpa [selectedColumns, first] using selectionLe.1)
        (residues first)
  have p1 : assignment (productColumn layout 1) =
      assignment (selectSecondColumn layout) *
        assignment (layout.residue second) := by
    simpa [selectedColumns, second] using
      product_eq canonical satisfied second
        (by simpa [selectedColumns, second] using selectionLe.2.1)
        (residues second)
  have p2 : assignment (productColumn layout 2) =
      assignment (selectThirdColumn layout) *
        assignment (layout.residue third) := by
    simpa [selectedColumns, third] using
      product_eq canonical satisfied third
        (by simpa [selectedColumns, third] using selectionLe.2.2)
        (residues third)
  have productsLe : forall index, index < 3 ->
      assignment (productColumn layout index) ≤ 4 := by
    intro index bounded
    interval_cases index
    · rw [p0]
      exact mul_bit_residue_le_four selectionLe.1 (residues first)
    · rw [p1]
      exact mul_bit_residue_le_four selectionLe.2.1 (residues second)
    · rw [p2]
      exact mul_bit_residue_le_four selectionLe.2.2 (residues third)
  have out := output_sum_eq canonical one productsLe satisfied
  have hasAccepted := available canonical one bits satisfied
  by_cases firstAccepted : assignment (layout.accept first) = 1
  · rw [if_pos firstAccepted]
    have s0Value : assignment (selectFirstColumn layout) = 1 :=
      s0.trans firstAccepted
    have s1Value : assignment (selectSecondColumn layout) = 0 := by
      rw [s1, firstAccepted]
      simp
    have s2Value : assignment (selectThirdColumn layout) = 0 := by
      rw [s2, firstAccepted]
      simp
    rw [out, p0, p1, p2, s0Value, s1Value, s2Value]
    simp
  · rw [if_neg firstAccepted]
    have firstZero : assignment (layout.accept first) = 0 := by
      have := bits first
      omega
    by_cases secondAccepted : assignment (layout.accept second) = 1
    · rw [if_pos secondAccepted]
      have s0Value : assignment (selectFirstColumn layout) = 0 :=
        s0.trans firstZero
      have s1Value : assignment (selectSecondColumn layout) = 1 := by
        rw [s1, firstZero, secondAccepted]
      have s2Value : assignment (selectThirdColumn layout) = 0 := by
        rw [s2, firstZero, secondAccepted]
        simp
      rw [out, p0, p1, p2, s0Value, s1Value, s2Value]
      simp
    · rw [if_neg secondAccepted]
      have secondZero : assignment (layout.accept second) = 0 := by
        have := bits second
        omega
      have thirdAccepted : assignment (layout.accept third) = 1 := by
        rcases hasAccepted with firstCase | secondCase | thirdCase
        · exact False.elim (firstAccepted firstCase)
        · exact False.elim (secondAccepted secondCase)
        · exact thirdCase
      have s0Value : assignment (selectFirstColumn layout) = 0 :=
        s0.trans firstZero
      have s1Value : assignment (selectSecondColumn layout) = 0 := by
        rw [s1, firstZero, secondZero]
      have s2Value : assignment (selectThirdColumn layout) = 1 := by
        rw [s2, firstZero, secondZero, thirdAccepted]
      rw [out, p0, p1, p2, s0Value, s1Value, s2Value]
      simp

theorem output_in_range
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bits : AcceptBits assignment layout)
    (residues : ResiduesInRange assignment layout)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (outputColumn layout) < 5 := by
  rw [output_first_accepted canonical one bits residues satisfied]
  split
  · exact residues first
  · split
    · exact residues second
    · exact residues third

structure Refines (assignment : Nat -> Nat) (layout : Layout) : Prop where
  available : assignment (layout.accept first) = 1 ∨
    assignment (layout.accept second) = 1 ∨
    assignment (layout.accept third) = 1
  output : assignment (outputColumn layout) =
    if assignment (layout.accept first) = 1 then
      assignment (layout.residue first)
    else if assignment (layout.accept second) = 1 then
      assignment (layout.residue second)
    else assignment (layout.residue third)
  outputInRange : assignment (outputColumn layout) < 5

theorem sound
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bits : AcceptBits assignment layout)
    (residues : ResiduesInRange assignment layout)
    (satisfied : Satisfies (rows layout) assignment) :
    Refines assignment layout where
  available := available canonical one bits satisfied
  output := output_first_accepted canonical one bits residues satisfied
  outputInRange := output_in_range canonical one bits residues satisfied

end Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedSound
