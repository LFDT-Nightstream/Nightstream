import NightstreamFPrime.Export.Stage1.PiCCSAssignmentSoundness
import NightstreamFPrime.Export.Stage1.PilotPoseidonPreservation

/-!
Owns the arbitrary-assignment hash equations for the actual pilot preimages.
Both chains read the same retained input forms as the PiCCS decoder. Digest
binding and the remaining pilot ordinary rows are separate obligations.
-/

namespace NightstreamFPrime.Export.Stage1.PilotDecodedHashes

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}

def inputEnv (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Env :=
  fun column => PiCCSAssignmentSoundness.decodedEnv geometry assignment
    (Spartan.liftPilotColumn column)

private theorem prior_input_lift (index : Fin Data.priorChain.inputLength) :
    Spartan.sourceToSpartan (PilotProduction.priorPreimageStart + index.val) =
      Spartan.liftPilotColumn (PilotData.priorChain.inputStart + index.val) := by
  have bound : index.val < 49393 := index.isLt
  unfold Spartan.sourceToSpartan
  rw [if_pos (by
    norm_num [PilotProduction.priorPreimageStart, Spartan.pilotSourceColumnCount]
    omega)]
  apply congrArg Spartan.liftPilotColumn
  unfold PilotSpartan.sourceToSpartan
  rw [if_pos (by
    norm_num [PilotProduction.priorPreimageStart, PilotSpartan.priorPublicStart,
      PilotProduction.stateHashWords_eq]
    exact bound)]
  rfl

private theorem output_input_lift (index : Fin Data.outputChain.inputLength) :
    Spartan.sourceToSpartan (PilotProduction.outputPreimageStart + index.val) =
      Spartan.liftPilotColumn (PilotData.outputChain.inputStart + index.val) := by
  have bound : index.val < 49393 := index.isLt
  unfold Spartan.sourceToSpartan
  rw [if_pos (by
    change 49663 + index.val < 14722512
    omega)]
  apply congrArg Spartan.liftPilotColumn
  unfold PilotSpartan.sourceToSpartan
  rw [if_neg (by change ¬49663 + index.val < 49393; omega)]
  rw [if_neg (by change ¬49663 + index.val < 49663; omega)]
  rw [if_pos (by change 49663 + index.val < 99056; omega)]
  change 49393 + ((49663 + index.val) - 49663) = 49393 + index.val
  omega

/-- Every prior-hash input is the actual PiCCS decoded preimage word. -/
theorem priorInputForm_eval
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (index : Fin Data.priorChain.inputLength) :
    ((PiRLCPoseidonGeometry.priorInputBlock program).form
      (PiRLCPoseidonGeometry.priorInputStart program)
      (PiRLCPoseidonGeometry.priorInputFits
        (PiCCSOrdinaryRetainedGeometry.pilotGeometry geometry)) index).eval assignment =
      inputEnv geometry assignment (PilotData.priorChain.inputStart + index.val) := by
  have mapped := PiCCSAssignmentSoundness.decodedEnv_location geometry assignment
    (.priorInput index)
  rw [PiCCSOrdinaryDirectPlan.Location.priorInput_form_eq_pilot] at mapped
  rw [PiCCSOrdinaryDirectPlan.Location.sourceColumn, prior_input_lift] at mapped
  exact mapped.symm

/-- Every output-hash input is the actual PiCCS decoded next-preimage word. -/
theorem outputInputForm_eval
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (index : Fin Data.outputChain.inputLength) :
    ((PiRLCPoseidonGeometry.outputInputBlock program).form
      (PiRLCPoseidonGeometry.outputInputStart program)
      (PiRLCPoseidonGeometry.outputInputFits
        (PiCCSOrdinaryRetainedGeometry.pilotGeometry geometry)) index).eval assignment =
      inputEnv geometry assignment (PilotData.outputChain.inputStart + index.val) := by
  have mapped := PiCCSAssignmentSoundness.decodedEnv_location geometry assignment
    (.outputInput index)
  rw [PiCCSOrdinaryDirectPlan.Location.outputInput_form_eq_pilot] at mapped
  rw [PiCCSOrdinaryDirectPlan.Location.sourceColumn, output_input_lift] at mapped
  exact mapped.symm

/-- Arbitrary accepted pilot Poseidon rows compute both hashes of the exact
preimages read by PiCCS. No canonical coordinate encoding is assumed. -/
theorem rowsZero_implies_hashFacts
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PilotPoseidonPlan.plan
      (PiCCSOrdinaryRetainedGeometry.pilotGeometry geometry)).RowsZero assignment) :
    PilotPoseidonPreservation.HashFacts
      (PiCCSOrdinaryRetainedGeometry.pilotGeometry geometry) assignment
      (inputEnv geometry assignment) := by
  have sameOne : PiRLCPoseidonGeometry.oneColumn
      (PiCCSOrdinaryRetainedGeometry.pilotGeometry geometry) =
        PiCCSOrdinaryRetainedGeometry.oneColumn geometry := by
    apply Fin.ext
    rfl
  have poseidonOne : assignment (PiRLCPoseidonGeometry.oneColumn
      (PiCCSOrdinaryRetainedGeometry.pilotGeometry geometry)) = 1 := by
    rw [sameOne]
    exact one
  exact PilotPoseidonPreservation.semantics_imply_hashFacts
    (PiCCSOrdinaryRetainedGeometry.pilotGeometry geometry) assignment
    (inputEnv geometry assignment) poseidonOne
    (priorInputForm_eval geometry assignment) (outputInputForm_eval geometry assignment)
    (PilotPoseidonPlan.rowsZero_implies_semantics
      (PiCCSOrdinaryRetainedGeometry.pilotGeometry geometry) assignment poseidonOne rows)

end NightstreamFPrime.Export.Stage1.PilotDecodedHashes
