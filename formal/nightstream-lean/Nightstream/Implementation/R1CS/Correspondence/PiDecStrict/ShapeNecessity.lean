import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictCompiler

/-!
Exact countermodels for the strict-PiDEC host-shape conditions.

The first layout has radix three and no emitted rows. The second layout has a
parent without the historical sidecar and one child with a sidecar. Its emitted
rows accept the all-zero data assignment, because sidecar rows are omitted when
the parent sidecar is absent. Neither layout satisfies the independent
`Accepted` predicate.

These examples justify `ShapeValid.radixTwo` and `ShapeValid.advPresence`.
They do not claim that a layout with a valid verifier-owned shape is unsound.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.PiDecStrictShapeNecessity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiDecStrictCompiler

def emptyCommitment : CommitmentLayout where
  dCol := 1
  kappaCol := 2
  dataCols := []

def emptyAdv : AdvLayout where
  ops := emptyCommitment
  is := emptyCommitment
  fs := emptyCommitment

def emptyClaim (adv : Option AdvLayout) : ClaimLayout where
  commitment := emptyCommitment
  adv := adv
  xActiveCols := []
  xRows := 0
  xWidth := 0
  xRowsCol := 3
  xWidthCol := 4
  mIn := 0
  mInCol := 5
  yRingCols := []
  ctCols := []
  rCols := []
  foldDigestCols := []

def radixThreeLayout : Layout where
  radix := 3
  ringDimension := 54
  extensionLimbs := 2
  firstAllocatedColumn := 20
  parent := emptyClaim none
  children := []

def mismatchedPresenceLayout : Layout where
  radix := 2
  ringDimension := 54
  extensionLimbs := 2
  firstAllocatedColumn := 20
  parent := emptyClaim none
  children := [emptyClaim (some emptyAdv)]

def assignment (column : Nat) : Nat :=
  if column = 0 then 1 else 0

theorem assignment_canonical (column : Nat) :
    assignment column < goldilocksP := by
  by_cases zero : column = 0
  · simp [assignment, zero, goldilocksP]
  · simp [assignment, zero, goldilocksP]

theorem assignment_one : assignment 0 = 1 := by
  simp [assignment]

theorem radixThree_rows_satisfied :
    Satisfies (rows radixThreeLayout) assignment := by
  intro row member
  simp [rows, CheckedProgram.rows, instructions, groups, radixThreeLayout, emptyClaim,
    emptyCommitment, advInstructions, dataRecomposition,
    xRecompositionInstructions, yRecompositionInstructions,
    shapeInstructions, pairEqualityInstructions, alphabetInstructions,
    alphabetFrom, ctInstructions, paddingInstructions,
    foldDigestInstructions] at member

theorem radixThree_not_accepted :
    ¬ Accepted radixThreeLayout assignment := by
  intro accepted
  have radix := accepted.radixTwo
  simp [radixThreeLayout] at radix

theorem mismatchedPresence_rows_satisfied :
    Satisfies (rows mismatchedPresenceLayout) assignment := by
  decide

theorem mismatchedPresence_not_accepted :
    ¬ Accepted mismatchedPresenceLayout assignment := by
  intro accepted
  have adv := accepted.adv
  simp [AdvAccepted, mismatchedPresenceLayout, emptyClaim, emptyAdv,
    emptyCommitment] at adv

/-- Exact necessity statement for both added host-shape fields. -/
theorem rows_alone_do_not_imply_strict_acceptance :
    (Satisfies (rows radixThreeLayout) assignment ∧
      ¬ Accepted radixThreeLayout assignment) ∧
    (Satisfies (rows mismatchedPresenceLayout) assignment ∧
      ¬ Accepted mismatchedPresenceLayout assignment) :=
  ⟨⟨radixThree_rows_satisfied, radixThree_not_accepted⟩,
    ⟨mismatchedPresence_rows_satisfied,
      mismatchedPresence_not_accepted⟩⟩

end Nightstream.Implementation.R1CS.PiDecStrictShapeNecessity
