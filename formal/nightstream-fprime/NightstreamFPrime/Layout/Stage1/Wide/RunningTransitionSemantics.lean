import NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionData
import NightstreamFPrime.Layout.Stage1.Wide.PiDECInputBounds
import NightstreamFPrime.Layout.Stage1.PiCCSRepresentation

/-! The wide transition reads the exact sixteen-child PiDEC result. -/

namespace NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionInputs

open NightstreamFPrime.Circuit NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint
open Lifecycle.Stage1 Lifecycle.PiCCS.v1_1

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

def piDecRunningOutput (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (env : Env) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  let piDec := piDecInterface logicalWidth publicFits
  let outputs := PiDEC.v1_1.Semantics.output relation piDec PiDECInputs.phaseOffset env
  { point := StatementAbsorption.evalPoint (piDec.point PiDECInputs.phaseOffset) env
    commitments := fun source => (outputs (childOfRunning source)).commitment
    publicInputs := fun source => (outputs (childOfRunning source)).publicInput
    evaluations := fun source => (outputs (childOfRunning source)).evaluations.getD 0 PaperAlgebra.evaluationZero }

theorem eval_recursiveRunningExpr_eq_piDecRunningOutput
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (env : Env) :
    StatementAbsorption.evalRunning (recursiveRunningExpr logicalWidth publicFits) env =
      piDecRunningOutput relation env := rfl

theorem spec_typed_base {env : Env}
    (specification : RunningTransition.SpecHolds (interface logicalWidth publicFits) phaseOffset env)
    (zero : RunningTransition.iterationValue (interface logicalWidth publicFits) phaseOffset env = 0) :
    StatementAbsorption.evalRunning (outputRunningExpr logicalWidth publicFits) env =
      defaultRunning (logicalWidth := logicalWidth) (publicFits := publicFits) := by
  apply PiCCSRepresentation.serializeRunning_injective
  exact RunningTransition.spec_serialized_base specification zero

theorem spec_typed_recursive (relation : ProductionKey.LogicalRelation logicalWidth publicFits) {env : Env}
    (specification : RunningTransition.SpecHolds (interface logicalWidth publicFits) phaseOffset env)
    (nonzero : RunningTransition.iterationValue (interface logicalWidth publicFits) phaseOffset env ≠ 0) :
    StatementAbsorption.evalRunning (outputRunningExpr logicalWidth publicFits) env = piDecRunningOutput relation env := by
  rw [← eval_recursiveRunningExpr_eq_piDecRunningOutput]
  apply PiCCSRepresentation.serializeRunning_injective
  exact RunningTransition.spec_serialized_recursive specification nonzero

/-- The exact PiDEC running result is preserved when every input source
below the PiDEC phase offset is preserved. No child opening is asserted. -/
theorem piDecOutput_eq_of_agree
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (before after : Env)
    (agrees : ∀ index, index < PiDECInputs.phaseOffset → before index = after index) :
    RunningTransitionInputs.piDecRunningOutput relation before =
      RunningTransitionInputs.piDecRunningOutput relation after := by
  have outputs := PiDEC.v1_1.Semantics.output_eq_of_agree relation
    (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset before after
    (PiDECInputs.assumptions relation before) agrees
  unfold RunningTransitionInputs.piDecRunningOutput
  dsimp only
  congr 1
  · exact congrArg (fun family => (family ⟨0, by decide⟩).point) outputs
  · funext source
    exact congrArg (fun family => (family (RunningTransitionInputs.childOfRunning source)).commitment) outputs
  · funext source
    exact congrArg (fun family => (family (RunningTransitionInputs.childOfRunning source)).publicInput) outputs
  · funext source
    exact congrArg (fun family => (family (RunningTransitionInputs.childOfRunning source)).evaluations.getD
      0 PaperAlgebra.evaluationZero) outputs

end NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionInputs
