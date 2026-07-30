import Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

/-!
Contract: kernel-check independence of the two setup-owned fields in the raw
fixed-one footprint record.

HyperNova Construction 2 supplies the application step and NIFS verifier in
setup.  Correspondingly, `Vocabulary.Footprints` has independent `step` and
`nifsVerify` fields.  The fixed non-application call footprints do not
determine either field.

This is a record-field independence result. The witnesses below do not carry
certified recipes and therefore do not prove that two valid deployments exist
or that any selected proof-carrying deployment has an undetermined cost.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DeploymentSelectionBoundary

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

private def rowsOnly (rows : Nat) : CallFootprint where
  recurringRows := rows
  temporaries := []

/-- A minimal witness family that changes only the two setup-owned slots. -/
def footprints (stepRows nifsRows : Nat) : Footprints where
  iterationZero := rowsOnly 0
  stateEqual := rowsOnly 0
  step := rowsOnly stepRows
  hash := rowsOnly 0
  freshPublic := rowsOnly 0
  encodeInstance := rowsOnly 0
  encodedEqual := rowsOnly 0
  nifsVerify := rowsOnly nifsRows
  runningCheck := rowsOnly 0
  freshCheck := rowsOnly 0

theorem fixed_call_footprints_equal
    (leftStep leftNifs rightStep rightNifs : Nat) :
    (footprints leftStep leftNifs).iterationZero =
        (footprints rightStep rightNifs).iterationZero ∧
      (footprints leftStep leftNifs).stateEqual =
        (footprints rightStep rightNifs).stateEqual ∧
      (footprints leftStep leftNifs).hash =
        (footprints rightStep rightNifs).hash ∧
      (footprints leftStep leftNifs).freshPublic =
        (footprints rightStep rightNifs).freshPublic ∧
      (footprints leftStep leftNifs).encodeInstance =
        (footprints rightStep rightNifs).encodeInstance ∧
      (footprints leftStep leftNifs).encodedEqual =
        (footprints rightStep rightNifs).encodedEqual ∧
      (footprints leftStep leftNifs).runningCheck =
        (footprints rightStep rightNifs).runningCheck ∧
      (footprints leftStep leftNifs).freshCheck =
        (footprints rightStep rightNifs).freshCheck := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

@[simp] theorem step_rows
    (stepRows nifsRows : Nat) :
    (footprints stepRows nifsRows).step.recurringRows = stepRows :=
  rfl

@[simp] theorem nifs_rows
    (stepRows nifsRows : Nat) :
    (footprints stepRows nifsRows).nifsVerify.recurringRows = nifsRows :=
  rfl

/-- **The eight fixed raw fields do not determine the other two raw fields.**

The two witnesses agree definitionally at every fixed call slot yet disagree
at both setup-owned row counts. This theorem is deliberately about the
unvalidated `Footprints` record; proof-carrying deployment selection is owned
by `CompleteApplicationCertification`. -/
theorem footprint_fields_do_not_determine_step_or_nifs_rows :
    ∃ left right : Footprints,
      left.iterationZero = right.iterationZero ∧
      left.stateEqual = right.stateEqual ∧
      left.hash = right.hash ∧
      left.freshPublic = right.freshPublic ∧
      left.encodeInstance = right.encodeInstance ∧
      left.encodedEqual = right.encodedEqual ∧
      left.runningCheck = right.runningCheck ∧
      left.freshCheck = right.freshCheck ∧
      left.step.recurringRows ≠ right.step.recurringRows ∧
      left.nifsVerify.recurringRows ≠ right.nifsVerify.recurringRows := by
  refine ⟨footprints 0 0, footprints 1 1, ?_⟩
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl,
    Nat.zero_ne_add_one 0, Nat.zero_ne_add_one 0⟩

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DeploymentSelectionBoundary
