import NightstreamFPrime.Layout.Stage1.RunningTransitionOutputBounds
import NightstreamFPrime.Layout.Stage1.RunningTransitionRecursiveBounds
import NightstreamFPrime.Layout.Stage1.RunningTransitionPointBoundsDirect

/-! Assembles the proved source bounds for the running transition. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def assumptions
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    RunningTransition.Assumptions
      (interface logicalWidth publicFits) phaseOffset env := by
  have phaseLower := piDecPhaseOffset_le
  rw [PiDECInputs.phaseOffset, PiDECInputs.proofInputColumnCount_eq] at phaseLower
  refine {
    iteration := ?_
    initialState := ?_
    currentState := ?_
    recursive := ?_
    output := ?_ }
  · simp [interface, iterationExpr, Expr.VarsBelow,
      iterationWordIndex, PilotProduction.priorPreimageStart]
    omega
  · intro index
    simp [interface, initialStateExpr, Expr.VarsBelow,
      initialStateWordStart, PilotProduction.priorPreimageStart]
    have bound := index.isLt
    norm_num [RunningTransition.stateWordCount] at bound ⊢
    omega
  · intro index
    simp [interface, currentStateExpr, Expr.VarsBelow,
      currentStateWordStart, PilotProduction.priorPreimageStart]
    have bound := index.isLt
    norm_num [RunningTransition.stateWordCount] at bound ⊢
    omega
  · intro index
    exact RunningTransition.runningWord_varsBelow _ phaseOffset
      (recursiveRunningBelow logicalWidth publicFits
        (recursivePointBelow relation)) index
  · intro index
    exact RunningTransition.runningWord_varsBelow _ phaseOffset
      (outputRunningBelow logicalWidth publicFits) index

end NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
