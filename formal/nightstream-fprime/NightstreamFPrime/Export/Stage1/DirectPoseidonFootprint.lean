import NightstreamFPrime.Export.Stage1.PiCCSInvocations
import NightstreamFPrime.Export.Stage1.PiRLCSamplerInvocations
import NightstreamFPrime.Layout.PilotValues
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedSlots

/-!
Owns the constant-time footprint guard for the current all-direct Poseidon2
low-norm plan. This is not a complete Stage 1 fit theorem.

This module does not select or authorize a replacement binding schedule.
-/

namespace NightstreamFPrime.Export.Stage1.DirectPoseidonFootprint

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def pilotPermutationCount : Nat :=
  2 * (PilotValues.absorbCount + 1)

@[simp] theorem pilotPermutationCount_eq : pilotPermutationCount = 4640 := by
  rfl

def laterPermutationCount : Nat := 948 + 34

@[simp] theorem laterPermutationCount_eq : laterPermutationCount = 982 := by
  rfl

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

@[simp] theorem totalPermutationCount_eq : totalPermutationCount = 5622 := by
  simp [totalPermutationCount]

def directSboxFieldCount : Nat :=
  totalPermutationCount *
    PoseidonRetainedSlots.slots.length

@[simp] theorem directSboxFieldCount_eq : directSboxFieldCount = 843300 := by
  rw [directSboxFieldCount, totalPermutationCount_eq,
    PoseidonRetainedSlots.slots_length]

def directSboxCoordinateCount : Nat :=
  totalPermutationCount *
    LowNormAssignment.logicalWidth PoseidonRetainedSlots.slots

@[simp] theorem directSboxCoordinateCount_eq :
    directSboxCoordinateCount = 34575300 := by
  rw [directSboxCoordinateCount, totalPermutationCount_eq,
    PoseidonRetainedSlots.slots_logicalWidth]

theorem directSboxCoordinates_le_cube :
    directSboxCoordinateCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [directSboxCoordinateCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

end NightstreamFPrime.Export.Stage1.DirectPoseidonFootprint
