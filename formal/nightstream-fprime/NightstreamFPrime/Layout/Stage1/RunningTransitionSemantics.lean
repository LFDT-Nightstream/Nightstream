import NightstreamFPrime.Layout.Stage1.RunningTransitionBounds
import NightstreamFPrime.Layout.Stage1.PiCCSRepresentation

/-! Owns typed base and recursive semantics for the running transition. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Serialized branch equality lifts to exact typed base-state equality. -/
theorem spec_typed_base
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {env : Env}
    (specification : RunningTransition.SpecHolds
      (interface logicalWidth publicFits) phaseOffset env)
    (iterationZero : RunningTransition.iterationValue
      (interface logicalWidth publicFits) phaseOffset env = 0) :
    StatementAbsorption.evalRunning
        (outputRunningExpr logicalWidth publicFits) env =
      defaultRunning (logicalWidth := logicalWidth)
        (publicFits := publicFits) := by
  apply PiCCSRepresentation.serializeRunning_injective
  exact RunningTransition.spec_serialized_base specification iterationZero

/-- Serialized branch equality lifts to exact typed PiDEC-output equality. -/
theorem spec_typed_recursive
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {env : Env}
    (specification : RunningTransition.SpecHolds
      (interface logicalWidth publicFits) phaseOffset env)
    (iterationNonzero : RunningTransition.iterationValue
      (interface logicalWidth publicFits) phaseOffset env ≠ 0) :
    StatementAbsorption.evalRunning
        (outputRunningExpr logicalWidth publicFits) env =
      StatementAbsorption.evalRunning
        (recursiveRunningExpr logicalWidth publicFits) env := by
  apply PiCCSRepresentation.serializeRunning_injective
  exact RunningTransition.spec_serialized_recursive specification
    iterationNonzero

/-- The complete 16-slot running value selected from the exact PiDEC child
outputs. Child order is the proved `runningCount_eq_childCount` cast. -/
def piDecRunningOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Running K
      PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      productionShape :=
  let piDec := piDecInterface logicalWidth publicFits
  let outputs := NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.output
    relation piDec PiDECInputs.phaseOffset env
  { point := StatementAbsorption.evalPoint
      (piDec.point PiDECInputs.phaseOffset) env
    commitments := fun source =>
      (outputs (childOfRunning source)).commitment
    publicInputs := fun source =>
      (outputs (childOfRunning source)).publicInput
    evaluations := fun source =>
      (outputs (childOfRunning source)).evaluations.getD 0
        PaperAlgebra.evaluationZero }

/-- The recursive transition input is literally the complete PiDEC result,
not a digest or a separately caller-selected running value. -/
theorem eval_recursiveRunningExpr_eq_piDecRunningOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    StatementAbsorption.evalRunning
        (recursiveRunningExpr logicalWidth publicFits) env =
      piDecRunningOutput relation env := by
  rfl

/-- On a recursive step, the typed pilot output running value is the exact
16-child PiDEC result. -/
theorem spec_typed_recursive_eq_piDecOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {env : Env}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (specification : RunningTransition.SpecHolds
      (interface logicalWidth publicFits) phaseOffset env)
    (iterationNonzero : RunningTransition.iterationValue
      (interface logicalWidth publicFits) phaseOffset env ≠ 0) :
    StatementAbsorption.evalRunning
        (outputRunningExpr logicalWidth publicFits) env =
      piDecRunningOutput relation env := by
  rw [spec_typed_recursive specification iterationNonzero,
    eval_recursiveRunningExpr_eq_piDecRunningOutput relation env]

end NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
