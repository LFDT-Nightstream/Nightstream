import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictCanonicalX

/-!
Model-level necessity witnesses for the canonical production `PiDEC` public-X
gadget.

Owns: two ordinary kernel-checked assignments showing that neither the radix
recomposition row nor the common-sign canonicality block follows from the
other; and a parameterized witness showing that every digit-selector row is
independent of the remaining digit-selector rows.

Does not own: production column placement, generated artifacts, Rust
refinement, whole-program acceptance, or permission to remove rows.

The witnesses use canonical Goldilocks representatives and the exact
seventeen-row compiler vocabulary from `PiDecStrictCanonicalX`.  The first
sets parent residue one and every digit to zero, so all canonicality rows hold
while recomposition fails.  The second uses the signed alias
`[-1, 1, 0, ...]` for parent residue one, so recomposition holds while the
common-sign digit selectors reject it.

Assurance tier: model-level.  All finite calculations below use kernel
`decide`; this file contains no generated artifact and no `native_decide`.
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictCanonicalX.Necessity

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.PiDecStrictCanonicalX

/-- A collision-free concrete layout for the finite countermodels. -/
def counterexampleLayout : Layout where
  parentColumn := 1
  signColumn := 2
  signOutputColumn := 3
  digitColumns := fun index => 4 + index.val

def firstChild : ChildIndex := ⟨0, by decide⟩
def secondChild : ChildIndex := ⟨1, by decide⟩

/-- Parent one with zero sign and zero digits.  It satisfies canonicality but
not radix recomposition. -/
def canonicalityOnlyAssignment (column : Nat) : Nat :=
  if column = 0 then 1
  else if column = counterexampleLayout.parentColumn then 1
  else 0

/-- Parent one with signed radix alias `[-1, 1, 0, ...]`.  It recomposes but
cannot use a single common sign when the explicit sign is zero. -/
def recompositionOnlyAssignment (column : Nat) : Nat :=
  if column = 0 then 1
  else if column = counterexampleLayout.parentColumn then 1
  else if column = counterexampleLayout.digitColumns firstChild then
    goldilocksP - 1
  else if column = counterexampleLayout.digitColumns secondChild then 1
  else 0

theorem canonicalityOnlyAssignment_canonical (column : Nat) :
    canonicalityOnlyAssignment column < goldilocksP := by
  by_cases zero : column = 0
  · subst column
    decide
  by_cases parent : column = counterexampleLayout.parentColumn
  · subst column
    decide
  · simp [canonicalityOnlyAssignment, zero, parent, goldilocksP]

theorem recompositionOnlyAssignment_canonical (column : Nat) :
    recompositionOnlyAssignment column < goldilocksP := by
  by_cases zero : column = 0
  · subst column
    decide
  by_cases parent : column = counterexampleLayout.parentColumn
  · subst column
    decide
  by_cases first : column = counterexampleLayout.digitColumns firstChild
  · subst column
    decide
  by_cases second : column = counterexampleLayout.digitColumns secondChild
  · subst column
    decide
  · simp [recompositionOnlyAssignment, zero, parent, first, second,
      goldilocksP]

theorem canonicalityOnly_one : canonicalityOnlyAssignment 0 = 1 := by
  decide

theorem recompositionOnly_one : recompositionOnlyAssignment 0 = 1 := by
  decide

private theorem counterexample_childColumns :
    childColumns counterexampleLayout =
      [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] := by
  decide

private theorem counterexample_powers :
    powers =
      [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] := by
  decide

private theorem canonicalityOnly_lcEval :
    lcEval canonicalityOnlyAssignment
      ((childColumns counterexampleLayout).zip powers) = 0 := by
  rw [counterexample_childColumns, counterexample_powers]
  change 0 = 0
  rfl

private theorem recompositionOnly_lcEval :
    lcEval recompositionOnlyAssignment
      ((childColumns counterexampleLayout).zip powers) = 1 := by
  rw [counterexample_childColumns, counterexample_powers]
  change (goldilocksP - 1 + 2) % goldilocksP = 1
  have sum : goldilocksP - 1 + 2 = goldilocksP + 1 := by
    simp [goldilocksP]
  rw [sum]
  simp [Nat.add_mod, goldilocksP]

private theorem canonicalityOnly_digit_value (index : ChildIndex) :
    canonicalityOnlyAssignment
      (counterexampleLayout.digitColumns index) = 0 := by
  have nonzero : counterexampleLayout.digitColumns index ≠ 0 := by
    change 4 + index.val ≠ 0
    omega
  have notParent : counterexampleLayout.digitColumns index ≠
      counterexampleLayout.parentColumn := by
    change 4 + index.val ≠ 1
    omega
  simp [canonicalityOnlyAssignment, nonzero, notParent]

private theorem canonicalityOnly_satisfies_centered :
    Satisfies
      (CheckedProgram.rows (centeredUnitInstructions
        counterexampleLayout.signColumn
        counterexampleLayout.signOutputColumn))
      canonicalityOnlyAssignment := by
  intro row member
  simp only [centeredUnitInstructions, CheckedProgram.rows, List.map_cons,
    List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;>
    simp [Instruction.row, Definition.builderRow, builderLinearRow,
      RowHolds, lcEval, canonicalityOnlyAssignment, counterexampleLayout,
      goldilocksP]

private theorem canonicalityOnly_satisfies_digits :
    Satisfies
      (CheckedProgram.rows (digitInstructions counterexampleLayout))
      canonicalityOnlyAssignment := by
  intro row member
  rcases List.mem_map.mp member with
    ⟨instruction, instructionMember, rfl⟩
  rcases List.mem_ofFn.mp instructionMember with ⟨index, rfl⟩
  simp [digitInstruction, Instruction.row, RowHolds, lcEval,
    canonicalityOnly_digit_value, goldilocksP]

theorem canonicalityOnly_satisfies_canonicality :
    Satisfies
      (CheckedProgram.rows (canonicalityInstructions counterexampleLayout))
      canonicalityOnlyAssignment := by
  simpa [canonicalityInstructions, CheckedProgram.rows] using
    Nightstream.Implementation.R1CS.PiDecStrictSound.satisfies_append
      canonicalityOnly_satisfies_centered
      canonicalityOnly_satisfies_digits

theorem canonicalityOnly_fails_recomposition :
    ¬ RowHolds canonicalityOnlyAssignment
      (recompositionInstruction counterexampleLayout).row := by
  intro holds
  have recomposes :=
    Nightstream.Implementation.R1CS.PiDecStrictSound.recompositionCheck_sound
      canonicalityOnlyAssignment_canonical canonicalityOnly_one
      powers_canonical holds
  unfold Recomposes at recomposes
  rw [canonicalityOnly_lcEval] at recomposes
  have parentValue :
      canonicalityOnlyAssignment counterexampleLayout.parentColumn = 1 := by
    simp [canonicalityOnlyAssignment, counterexampleLayout]
  rw [parentValue] at recomposes
  omega

private theorem canonicalityOnly_decodedParent :
    decodedParent counterexampleLayout canonicalityOnlyAssignment =
      fieldOfNat 1 := by
  simp [decodedParent, canonicalityOnlyAssignment, counterexampleLayout]

private theorem recompositionOnly_decodedParent :
    decodedParent counterexampleLayout recompositionOnlyAssignment =
      fieldOfNat 1 := by
  simp [decodedParent, recompositionOnlyAssignment, counterexampleLayout]

private theorem fieldOne_centeredMagnitude :
    centeredMagnitude (fieldOfNat 1) = 1 := by
  simp [centeredMagnitude, fieldOfNat, goldilocksModulus]

private theorem fieldOne_bounded :
    centeredMagnitude (fieldOfNat 1) < combinedBound := by
  rw [fieldOne_centeredMagnitude, production_parameters.2.2]
  omega

private theorem fieldOne_nonnegative :
    isNonnegative (fieldOfNat 1) := by
  unfold isNonnegative
  change 1 ≤
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Centered.halfModulus
  unfold Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Centered.halfModulus
  simp [goldilocksModulus]

private theorem splitScalar_one_first :
    splitScalar (fieldOfNat 1) firstChild = (1 : F) := by
  unfold splitScalar
  rw [if_pos fieldOne_bounded]
  unfold boundedDigit magnitudeDigit
  rw [if_pos fieldOne_nonnegative, fieldOne_centeredMagnitude]
  simp [natBit, firstChild]

theorem canonicalityOnly_digits_ne_splitScalar :
    decodedDigits counterexampleLayout canonicalityOnlyAssignment ≠
      splitScalar
        (decodedParent counterexampleLayout canonicalityOnlyAssignment) := by
  intro equal
  have atZero := congrFun equal firstChild
  have leftValue :
      decodedDigits counterexampleLayout canonicalityOnlyAssignment
        firstChild = 0 := by
    simp [decodedDigits, canonicalityOnlyAssignment, counterexampleLayout,
      firstChild]
  have rightValue :
      splitScalar
        (decodedParent counterexampleLayout canonicalityOnlyAssignment)
        firstChild = 1 := by
    rw [canonicalityOnly_decodedParent]
    exact splitScalar_one_first
  have zeroEqOne : (0 : F) = 1 := by
    rw [← leftValue, atZero, rightValue]
  have zeroNeOne : (0 : F) ≠ 1 := by decide
  exact zeroNeOne zeroEqOne

theorem canonicalityOnly_semantically_invalid :
    ¬ Accepted
      (decodedParent counterexampleLayout canonicalityOnlyAssignment)
      (decodedSign counterexampleLayout canonicalityOnlyAssignment)
      (decodedDigits counterexampleLayout canonicalityOnlyAssignment) := by
  intro accepted
  apply canonicalityOnly_digits_ne_splitScalar
  apply accepted_digits_eq_splitScalar
  · rw [canonicalityOnly_decodedParent]
    exact fieldOne_bounded
  · exact accepted

/-- Canonicality alone admits an invalid parent/digit transition.  In
particular, the retained recomposition row is necessary relative to the fixed
compiler vocabulary. -/
theorem canonicality_without_recomposition_counterexample :
    (∀ column,
      canonicalityOnlyAssignment column < goldilocksP) ∧
    canonicalityOnlyAssignment 0 = 1 ∧
    Satisfies
      (CheckedProgram.rows (canonicalityInstructions counterexampleLayout))
      canonicalityOnlyAssignment ∧
    ¬ RowHolds canonicalityOnlyAssignment
      (recompositionInstruction counterexampleLayout).row ∧
    ¬ Accepted
      (decodedParent counterexampleLayout canonicalityOnlyAssignment)
      (decodedSign counterexampleLayout canonicalityOnlyAssignment)
      (decodedDigits counterexampleLayout canonicalityOnlyAssignment) ∧
    decodedDigits counterexampleLayout canonicalityOnlyAssignment ≠
      splitScalar
        (decodedParent counterexampleLayout canonicalityOnlyAssignment) :=
  ⟨canonicalityOnlyAssignment_canonical, canonicalityOnly_one,
    canonicalityOnly_satisfies_canonicality,
    canonicalityOnly_fails_recomposition,
    canonicalityOnly_semantically_invalid,
    canonicalityOnly_digits_ne_splitScalar⟩

theorem recompositionOnly_satisfies_recomposition :
    RowHolds recompositionOnlyAssignment
      (recompositionInstruction counterexampleLayout).row := by
  apply Nightstream.Implementation.R1CS.PiDecStrictSound.recompositionCheck_complete
    recompositionOnly_one powers_canonical
  unfold Recomposes
  rw [recompositionOnly_lcEval]
  simp [recompositionOnlyAssignment, counterexampleLayout]

theorem recompositionOnly_fails_first_digit_selector :
    ¬ RowHolds recompositionOnlyAssignment
      (digitInstruction counterexampleLayout firstChild).row := by
  simp [digitInstruction, Instruction.row, RowHolds, lcEval,
    recompositionOnlyAssignment, counterexampleLayout, firstChild,
    secondChild, goldilocksP]

theorem recompositionOnly_fails_canonicality :
    ¬ Satisfies
      (CheckedProgram.rows (canonicalityInstructions counterexampleLayout))
      recompositionOnlyAssignment := by
  intro satisfies
  apply recompositionOnly_fails_first_digit_selector
  apply satisfies
  simp [canonicalityInstructions, CheckedProgram.rows, digitInstructions]

theorem recompositionOnly_digits_ne_splitScalar :
    decodedDigits counterexampleLayout recompositionOnlyAssignment ≠
      splitScalar
        (decodedParent counterexampleLayout recompositionOnlyAssignment) := by
  intro equal
  have atZero := congrFun equal firstChild
  have leftValue :
      decodedDigits counterexampleLayout recompositionOnlyAssignment
        firstChild = fieldOfNat (goldilocksP - 1) := by
    simp [decodedDigits, recompositionOnlyAssignment, counterexampleLayout,
      firstChild, secondChild]
  have rightValue :
      splitScalar
        (decodedParent counterexampleLayout recompositionOnlyAssignment)
        firstChild = 1 := by
    rw [recompositionOnly_decodedParent]
    exact splitScalar_one_first
  have aliasEqOne : fieldOfNat (goldilocksP - 1) = (1 : F) := by
    rw [← leftValue, atZero, rightValue]
  have aliasNeOne : fieldOfNat (goldilocksP - 1) ≠ (1 : F) := by decide
  exact aliasNeOne aliasEqOne

theorem recompositionOnly_semantically_invalid :
    ¬ Accepted
      (decodedParent counterexampleLayout recompositionOnlyAssignment)
      (decodedSign counterexampleLayout recompositionOnlyAssignment)
      (decodedDigits counterexampleLayout recompositionOnlyAssignment) := by
  intro accepted
  apply recompositionOnly_digits_ne_splitScalar
  apply accepted_digits_eq_splitScalar
  · rw [recompositionOnly_decodedParent]
    exact fieldOne_bounded
  · exact accepted

/-- Recomposition alone admits a non-canonical signed alias for parent residue
one.  The common-sign canonicality block is therefore necessary relative to
the fixed compiler vocabulary. -/
theorem recomposition_without_canonicality_counterexample :
    (∀ column,
      recompositionOnlyAssignment column < goldilocksP) ∧
    recompositionOnlyAssignment 0 = 1 ∧
    RowHolds recompositionOnlyAssignment
      (recompositionInstruction counterexampleLayout).row ∧
    ¬ Satisfies
      (CheckedProgram.rows (canonicalityInstructions counterexampleLayout))
      recompositionOnlyAssignment ∧
    ¬ Accepted
      (decodedParent counterexampleLayout recompositionOnlyAssignment)
      (decodedSign counterexampleLayout recompositionOnlyAssignment)
      (decodedDigits counterexampleLayout recompositionOnlyAssignment) ∧
    decodedDigits counterexampleLayout recompositionOnlyAssignment ≠
      splitScalar
        (decodedParent counterexampleLayout recompositionOnlyAssignment) :=
  ⟨recompositionOnlyAssignment_canonical, recompositionOnly_one,
    recompositionOnly_satisfies_recomposition,
    recompositionOnly_fails_canonicality,
    recompositionOnly_semantically_invalid,
    recompositionOnly_digits_ne_splitScalar⟩

/-! ## Per-digit selector independence -/

/-- Set exactly one selected digit to one while keeping the explicit common
sign at zero.  This parameterizes the local selector witness without
enumerating fourteen assignments. -/
def omittedDigitAssignment (omitted : ChildIndex) (column : Nat) : Nat :=
  if column = 0 then 1
  else if column = counterexampleLayout.digitColumns omitted then 1
  else 0

private theorem omittedDigitAssignment_self (omitted : ChildIndex) :
    omittedDigitAssignment omitted
      (counterexampleLayout.digitColumns omitted) = 1 := by
  have nonzero : counterexampleLayout.digitColumns omitted ≠ 0 := by
    change 4 + omitted.val ≠ 0
    omega
  simp [omittedDigitAssignment, nonzero]

private theorem omittedDigitAssignment_other
    (omitted index : ChildIndex) (different : index ≠ omitted) :
    omittedDigitAssignment omitted
      (counterexampleLayout.digitColumns index) = 0 := by
  have nonzero : counterexampleLayout.digitColumns index ≠ 0 := by
    change 4 + index.val ≠ 0
    omega
  have columnDifferent :
      counterexampleLayout.digitColumns index ≠
        counterexampleLayout.digitColumns omitted := by
    change 4 + index.val ≠ 4 + omitted.val
    intro equal
    apply different
    apply Fin.ext
    omega
  simp [omittedDigitAssignment, nonzero, columnDifferent]

private theorem omittedDigitAssignment_sign (omitted : ChildIndex) :
    omittedDigitAssignment omitted counterexampleLayout.signColumn = 0 := by
  have nonzero : counterexampleLayout.signColumn ≠ 0 := by
    change 2 ≠ 0
    omega
  have notDigit : counterexampleLayout.signColumn ≠
      counterexampleLayout.digitColumns omitted := by
    change 2 ≠ 4 + omitted.val
    omega
  unfold omittedDigitAssignment
  rw [if_neg nonzero, if_neg notDigit]

theorem omitted_digit_selector_fails (omitted : ChildIndex) :
    ¬ RowHolds (omittedDigitAssignment omitted)
      (digitInstruction counterexampleLayout omitted).row := by
  simp [digitInstruction, Instruction.row, RowHolds, lcEval,
    omittedDigitAssignment_self, omittedDigitAssignment_sign, goldilocksP]

theorem other_digit_selectors_hold
    (omitted index : ChildIndex) (different : index ≠ omitted) :
    RowHolds (omittedDigitAssignment omitted)
      (digitInstruction counterexampleLayout index).row := by
  simp [digitInstruction, Instruction.row, RowHolds, lcEval,
    omittedDigitAssignment_other omitted index different, goldilocksP]

/-- Every one of the fourteen selector equations excludes a witness admitted
by all thirteen peer selector equations. -/
theorem each_digit_selector_is_independent (omitted : ChildIndex) :
    (¬ RowHolds (omittedDigitAssignment omitted)
      (digitInstruction counterexampleLayout omitted).row) ∧
    (∀ index, index ≠ omitted →
      RowHolds (omittedDigitAssignment omitted)
        (digitInstruction counterexampleLayout index).row) :=
  ⟨omitted_digit_selector_fails omitted,
    fun index different => other_digit_selectors_hold omitted index different⟩

end Nightstream.Implementation.R1CS.PiDecStrictCanonicalX.Necessity
