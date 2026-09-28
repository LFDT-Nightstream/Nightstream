import NightstreamFPrime.Export.Stage1.PiDECInputCheck
import NightstreamFPrime.Lifecycle.PaperExtractionAlgebra
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingPowerInverse

/-!
Selected value refinement of the four PiRLC extraction primitives. The
production scalar/module operations stay unchanged. Inversion executes the
existing stored 54-lane inverse and erases its array result.

Caller clocks remain explicit declared values. The inverse retains its own
returned work and adds the caller's storage/adapter clock. This module proves
no clock bound, runtime refinement, EPT claim, or hardness assumption.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCExtractionPrimitives

open _root_.NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Lifecycle
open _root_.NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkAlgebra (UnitWitness)
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Primitives Correct)

abbrev Assignment := PaperAlgebra.Assignment
  (logicalWidth := PiDECInputCheck.logicalWidth) (publicFits := PiDECInputCheck.publicFits)

private theorem inverse_of_erasure (stored : StoredRingArithmetic.StoredRing)
    (value : RingF) (same : stored.get = value)
    (unit : UnitWitness Phi81Relation.PiRLCAlgebra.ForkStrongSet.ring value) :
    (StoredRingPowerInverse.inverse stored).value.get = unit.inverse := by
  cases same
  exact StoredRingPowerInverse.inverse_eq_unitInverse stored unit

private theorem ofFn_view {Value : Type} {count : Nat}
    (value : Fin count → Value) : (Vector.ofFn value).get = value := by
  funext lane
  simp [Vector.get]

/-- Reuse the actual production operations. The inverse adapter clock covers
function-to-array storage and return overhead; it does not replace the work
returned by the executed inverse call. No caller clock is assumed bounded. -/
def program
    (scalarSubClock : RingF → RingF → Nat)
    (inverseAdapterClock : RingF → Nat)
    (assignmentSubClock : Assignment → Assignment → Nat)
    (scalarActionClock : RingF → Assignment → Nat) : Primitives RingF Assignment where
  scalarSub := fun left right =>
    ⟨Phi81StrongSet.ringFSub left right, scalarSubClock left right⟩
  unitInverse := fun value =>
    let stored : StoredRingArithmetic.StoredRing := Vector.ofFn value
    let inverse := StoredRingPowerInverse.inverse stored
    ⟨inverse.value.get, inverse.work + inverseAdapterClock value⟩
  assignmentSub := fun left right =>
    ⟨(PaperExtractionAlgebra.extractionAlgebra
      Poseidon2HashChainV1Setup.productionAjtaiKey).assignmentModule.sub left right,
      assignmentSubClock left right⟩
  scalarAction := fun scalar assignment =>
    ⟨CarrierAction.act (logicalWidth := PiDECInputCheck.logicalWidth) scalar assignment,
      scalarActionClock scalar assignment⟩

/-- The exact Correct parameter used by the existing extraction and
Fiat--Shamir consumers. Unit evidence is used only in the inverse proof;
the executed primitive receives only its scalar value. -/
theorem program_correct
    (scalarSubClock : RingF → RingF → Nat)
    (inverseAdapterClock : RingF → Nat)
    (assignmentSubClock : Assignment → Assignment → Nat)
    (scalarActionClock : RingF → Assignment → Nat) :
    Correct (PaperExtractionAlgebra.extractionAlgebra
      Poseidon2HashChainV1Setup.productionAjtaiKey).ring
      (PaperExtractionAlgebra.extractionAlgebra
        Poseidon2HashChainV1Setup.productionAjtaiKey).assignmentModule
      (program scalarSubClock inverseAdapterClock assignmentSubClock scalarActionClock) := by
  constructor
  · intro left right
    exact (Phi81Relation.PiRLCAlgebra.ForkStrongSet.ring_sub_eq left right).symm
  · intro value unit
    change (StoredRingPowerInverse.inverse (Vector.ofFn value)).value.get = unit.inverse
    exact inverse_of_erasure (Vector.ofFn value) value (ofFn_view value) unit
  · intro left right
    rfl
  · intro scalar assignment
    rfl

end NightstreamFPrime.Export.Stage1.PiRLCExtractionPrimitives
