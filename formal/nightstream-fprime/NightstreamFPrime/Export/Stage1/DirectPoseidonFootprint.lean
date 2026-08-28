import NightstreamFPrime.Export.Stage1.PiCCSInvocations
import NightstreamFPrime.Export.Stage1.PiRLCSamplerInvocations
import NightstreamFPrime.Layout.PilotValues
import NightstreamFPrime.Layout.ProductionRelation.PoseidonTemplatePlan

/-!
Owns the constant-time footprint guard for the current all-direct Poseidon2
low-norm plan. Existing Lean counts imply that opening only its retained S-box
outputs with the canonical 41-trit field encoding already exceeds the fixed
`2^26` carrier domain.

This module does not select or authorize a replacement binding schedule.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPoseidonFootprint

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- The two pilot chains each contain 11,484 absorption permutations and one
final padding permutation. -/
def pilotPermutationCount : Nat :=
  2 * (PilotValues.absorbCount + 1)

@[simp] theorem pilotPermutationCount_eq : pilotPermutationCount = 22970 := by
  rfl

/-- PiCCS and PiRLC sampler permutations in the current Stage 1 package. -/
def laterPermutationCount : Nat := 7526 + 153

@[simp] theorem laterPermutationCount_eq : laterPermutationCount = 7679 := by
  rfl

/-- The two phase schedule owners have exactly the fixed later count at every
valid production-width instantiation. -/
theorem phaseInvocations_length (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (PiCCSInvocations.invocations logicalWidth publicFits ++
      PiRLCSamplerInvocations.invocations
        (logicalWidth := logicalWidth) (publicFits := publicFits)).length =
      laterPermutationCount := by
  rw [List.length_append,
    PiCCSInvocations.invocations_length logicalWidth publicFits,
    PiRLCSamplerInvocations.invocations_length]
  rfl

def totalPermutationCount : Nat :=
  pilotPermutationCount + laterPermutationCount

@[simp] theorem totalPermutationCount_eq : totalPermutationCount = 30649 := by
  simp [totalPermutationCount]

/-- One authoritative general-field slot for every retained S-box output in
the current direct template. -/
def directSboxFieldCount : Nat :=
  totalPermutationCount *
    (PoseidonTemplatePlan.plan.map fun step => step.sboxes.length).sum

@[simp] theorem directSboxFieldCount_eq : directSboxFieldCount = 2635814 := by
  rw [directSboxFieldCount, totalPermutationCount_eq,
    PoseidonTemplatePlan.sboxRowCount_eq]

/-- Canonical low-norm coordinates needed by only those S-box output slots.
This excludes final outputs and every non-Poseidon source value. -/
def directSboxCoordinateCount : Nat :=
  directSboxFieldCount * BalancedTernary.width

@[simp] theorem directSboxCoordinateCount_eq :
    directSboxCoordinateCount = 108068374 := by
  rw [directSboxCoordinateCount, directSboxFieldCount_eq]
  rfl

/-- The current all-direct plan cannot satisfy the fixed Stage 1 carrier
domain, even before any other assignment coordinate is added. -/
theorem directSboxCoordinates_exceed_cube :
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables <
      directSboxCoordinateCount := by
  rw [directSboxCoordinateCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

end NightstreamFPrime.Export.Stage1.DirectPoseidonFootprint
