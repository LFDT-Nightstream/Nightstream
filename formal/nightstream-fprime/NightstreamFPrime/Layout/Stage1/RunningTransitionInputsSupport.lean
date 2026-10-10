import NightstreamFPrime.Layout.Stage1.RunningTransitionCost
import NightstreamFPrime.Layout.Stage1.RunningTransitionOutputSupport
import NightstreamFPrime.Layout.Stage1.RunningTransitionRecursiveSupport

/-! Owns compact support for the complete running-transition interface. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open RunningTransitionInputs

theorem inputsSupported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    RunningTransition.InputsSupported
      (interface logicalWidth publicFits) phaseOffset Logical := by
  refine {
    iteration := ?_
    inverse := ?_
    flag := ?_
    initialState := ?_
    currentState := ?_
    recursive := recursiveSupported logicalWidth publicFits
    output := outputSupported }
  · simp only [interface, iterationExpr, Expr.VarsSatisfy]
    apply logical_state
    rw [stateStart_eq]
    norm_num [InRange, stateCount, iterationExpr,
      PilotProduction.priorPreimageStart, iterationWordIndex]
  · exact Or.inr (Or.inl rfl)
  · exact Or.inr (Or.inr rfl)
  · intro index
    simp only [interface, initialStateExpr, Expr.VarsSatisfy]
    apply logical_state
    have bound := index.isLt
    rw [stateStart_eq]
    norm_num [InRange, stateCount, initialStateExpr,
      PilotProduction.priorPreimageStart, initialStateWordStart,
      RunningTransition.stateWordCount] at bound ⊢
    omega
  · intro index
    simp only [interface, currentStateExpr, Expr.VarsSatisfy]
    apply logical_state
    have bound := index.isLt
    rw [stateStart_eq]
    norm_num [InRange, stateCount, currentStateExpr,
      PilotProduction.priorPreimageStart, currentStateWordStart,
      RunningTransition.stateWordCount] at bound ⊢
    omega

theorem logicalConstraints_varsSatisfy
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    ∀ expression ∈ RunningTransitionLayout.logicalConstraints
        logicalWidth publicFits,
      expression.VarsSatisfy Logical := by
  exact RunningTransition.flatConstraints_varsSatisfy
    (interface logicalWidth publicFits) phaseOffset Logical
      (inputsSupported logicalWidth publicFits)

end NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport
