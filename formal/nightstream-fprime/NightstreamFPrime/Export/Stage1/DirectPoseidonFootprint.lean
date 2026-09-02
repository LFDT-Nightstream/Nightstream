import NightstreamFPrime.Export.Stage1.PiCCSInvocations
import NightstreamFPrime.Export.Stage1.PiRLCSamplerInvocations
import NightstreamFPrime.Layout.PilotValues
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedSlots

/-!
Owns the constant-time footprint guard for the current all-direct Poseidon2
low-norm plan. Existing Lean counts imply that opening only its retained S-box
outputs with the canonical 41-trit field encoding fit below the fixed `2^28`
carrier domain. This is not a complete Stage 1 fit theorem.

This module does not select or authorize a replacement binding schedule.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPoseidonFootprint

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- The two pilot chains each contain 12,349 absorption permutations and one
final padding permutation. -/
def pilotPermutationCount : Nat :=
  2 * (PilotValues.absorbCount + 1)

@[simp] theorem pilotPermutationCount_eq : pilotPermutationCount = 24700 := by
  rfl

/-- PiCCS and PiRLC sampler permutations in the current Stage 1 package. -/
def laterPermutationCount : Nat := 7604 + 153

@[simp] theorem laterPermutationCount_eq : laterPermutationCount = 7757 := by
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

@[simp] theorem totalPermutationCount_eq : totalPermutationCount = 32457 := by
  simp [totalPermutationCount]

/-- One authoritative general-field slot for every retained S-box output in
the current direct template. -/
def directSboxFieldCount : Nat :=
  totalPermutationCount *
    PoseidonRetainedSlots.slots.length

@[simp] theorem directSboxFieldCount_eq : directSboxFieldCount = 2791302 := by
  rw [directSboxFieldCount, totalPermutationCount_eq,
    PoseidonRetainedSlots.slots_length]

/-- Canonical low-norm coordinates of the actual retained S-box slot plan.
This excludes final outputs and every non-Poseidon source value. -/
def directSboxCoordinateCount : Nat :=
  totalPermutationCount *
    LowNormAssignment.logicalWidth PoseidonRetainedSlots.slots

@[simp] theorem directSboxCoordinateCount_eq :
    directSboxCoordinateCount = 114443382 := by
  rw [directSboxCoordinateCount, totalPermutationCount_eq,
    PoseidonRetainedSlots.slots_logicalWidth]

/-- The retained S-box coordinates in the current direct plan fit below the
Stage 1 carrier domain. Other assignment coordinates still require the final
joint-domain proof. -/
theorem directSboxCoordinates_le_cube :
    directSboxCoordinateCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [directSboxCoordinateCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

end NightstreamFPrime.Export.Stage1.DirectPoseidonFootprint
