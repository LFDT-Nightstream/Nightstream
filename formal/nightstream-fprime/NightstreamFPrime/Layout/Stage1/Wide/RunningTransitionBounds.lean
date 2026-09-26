import NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionSemantics
import NightstreamFPrime.Layout.Stage1.RunningTransitionOutputBounds

/-! Derive the transition's source bounds from the wide PiDEC inputs and
the unchanged output-state encoding. No row traversal is used. -/

namespace NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionInputs

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

theorem recursiveRunningBelow (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    RunningTransition.RunningBelow (recursiveRunningExpr logicalWidth publicFits) phaseOffset := by
  have inputs := PiDECInputs.inputsBelow relation
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro coordinate
    exact Quadratic.KExpr.varsBelow_mono _ (inputs.point coordinate) piDecPhaseOffset_le
  · intro source row lane
    exact Expr.VarsBelow.mono _ (inputs.messageCommitment (childOfRunning source) row lane) piDecPhaseOffset_le
  · intro source column
    exact Expr.VarsBelow.mono _ (inputs.digit (childOfRunning source) column) piDecPhaseOffset_le
  · intro source coefficient
    exact Quadratic.KExpr.varsBelow_mono _ (inputs.messageEval_K (childOfRunning source) coefficient) piDecPhaseOffset_le
  · intro source matrix coefficient
    exact Quadratic.KExpr.varsBelow_mono _ (inputs.messageEval_A (childOfRunning source) matrix coefficient) piDecPhaseOffset_le

theorem outputRunningBelow :
    RunningTransition.RunningBelow (outputRunningExpr logicalWidth publicFits) phaseOffset :=
  (Stage1.RunningTransitionInputs.outputRunningBelowOutputDigestStart logicalWidth publicFits).mono (by decide)

theorem assumptions (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (env : Env) :
    RunningTransition.Assumptions (interface logicalWidth publicFits) phaseOffset env := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · change 28 < 27563400
    decide
  · intro index
    have bound : index.val < 4 := index.isLt
    change 30 + index.val < 27563400
    omega
  · intro index
    have bound : index.val < 4 := index.isLt
    change 35 + index.val < 27563400
    omega
  · intro index
    exact RunningTransition.runningWord_varsBelow _ _ (recursiveRunningBelow relation) index
  · intro index
    exact RunningTransition.runningWord_varsBelow _ _ outputRunningBelow index

end NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionInputs
