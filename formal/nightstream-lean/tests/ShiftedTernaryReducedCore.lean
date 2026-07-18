import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.ReducedCore

/-! Narrow compile-time checks for the model-level 123-gate opening core. -/

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.ShiftedTernaryComplete
open Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore

example : gates.length = 123 := gates_length

example (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : Accepts assignment)
    {index : Nat} (indexLt : index < digitCount) :
    Digit
      (assignment (ShiftedTernary.digitCols.getD index 0))
      (assignment (ShiftedTernary.negativeCols.getD index 0)) := by
  exact digit_of_centeredUnit_and_definition prime canonical one
    (accepted.centeredUnit index indexLt)
    (accepted.negativeDefinition index indexLt)

example (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : Accepts assignment)
    {index : Nat} (indexLt : index < digitCount - 1) :
    RowHolds assignment
      (bitRow (ShiftedTernary.borrowCols.getD index 0)) := by
  exact accepted.borrow_bitness_follows prime canonical one indexLt

example {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (alias : SharedFieldDigitAlias assignment) :
    RowHolds assignment reconstructionRow := by
  exact reconstructionRow_holds_of_shared_alias one alias

example (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    (Accepts assignment ∧ SharedFieldDigitAlias assignment) ↔
      Satisfies canonicalRows assignment := by
  exact reduced_iff_canonicalRows prime canonical one

example (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : Accepts assignment)
    (alias : SharedFieldDigitAlias assignment) :
    CanonicalOpening assignment := by
  exact canonicalOpening_of_reduced prime canonical one accepted alias

example {assignment : Nat → Nat}
    (witness : CanonicalWitness assignment) :
    Accepts assignment ∧ SharedFieldDigitAlias assignment := by
  exact
    Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore.CanonicalWitness.reducedCore_complete
      witness
