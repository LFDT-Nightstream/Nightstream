import NightstreamFPrime.Layout.Stage1.RunningTransitionOutputBounds
import NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupportData
import NightstreamFPrime.Lifecycle.Stage1.RunningTransitionSupport

/-! Owns compact source support for the output running vector. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open RunningTransitionInputs

theorem outputSupported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    RunningTransition.RunningSupported
      (outputRunningExpr logicalWidth publicFits) Logical := by
  let below := outputRunningBelowOutputDigestStart logicalWidth publicFits
  refine {
    point := ?_
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_ }
  · intro coordinate
    have upper := below.point coordinate
    simp only [outputRunningExpr, outputPoint, outputPairAt, KExpr.VarsBelow,
      Expr.VarsBelow] at upper
    simp only [outputRunningExpr, outputPoint, outputPairAt,
      Expr.VarsSatisfy]
    constructor
    · apply logical_output
      exact ⟨by unfold outputStart outputBase; omega, by
        simpa [InRange, outputStart, outputCount,
          PilotProduction.outputDigestStart] using upper.1⟩
    · apply logical_output
      exact ⟨by unfold outputStart outputBase; omega, by
        simpa [InRange, outputStart, outputCount,
          PilotProduction.outputDigestStart] using upper.2⟩
  · intro source row coefficient
    have upper := below.commitment source row coefficient
    simp only [outputRunningExpr, outputCommitment, Expr.VarsBelow] at upper
    simp only [outputRunningExpr, outputCommitment, Expr.VarsSatisfy]
    apply logical_output
    exact ⟨by unfold outputStart outputBase; omega, by
      simpa [InRange, outputStart, outputCount,
        PilotProduction.outputDigestStart] using upper⟩
  · intro source column
    have upper := below.publicInput source column
    simp only [outputRunningExpr, outputPublicInput, Expr.VarsBelow] at upper
    simp only [outputRunningExpr, outputPublicInput, Expr.VarsSatisfy]
    apply logical_output
    exact ⟨by unfold outputStart outputBase; omega, by
      simpa [InRange, outputStart, outputCount,
        PilotProduction.outputDigestStart] using upper⟩
  · intro source coefficient
    have upper := below.eval_K source coefficient
    simp only [outputRunningExpr, outputEval_K, outputPairAt, KExpr.VarsBelow,
      Expr.VarsBelow] at upper
    simp only [outputRunningExpr, outputEval_K, outputPairAt,
      Expr.VarsSatisfy]
    constructor
    · apply logical_output
      exact ⟨by unfold outputStart outputBase; omega, by
        simpa [InRange, outputStart, outputCount,
          PilotProduction.outputDigestStart] using upper.1⟩
    · apply logical_output
      exact ⟨by unfold outputStart outputBase; omega, by
        simpa [InRange, outputStart, outputCount,
          PilotProduction.outputDigestStart] using upper.2⟩
  · intro source matrix coefficient
    have upper := below.eval_A source matrix coefficient
    simp only [outputRunningExpr, outputEval_A, outputPairAt, KExpr.VarsBelow,
      Expr.VarsBelow] at upper
    simp only [outputRunningExpr, outputEval_A, outputPairAt,
      Expr.VarsSatisfy]
    constructor
    · apply logical_output
      exact ⟨by unfold outputStart outputBase; omega, by
        simpa [InRange, outputStart, outputCount,
          PilotProduction.outputDigestStart] using upper.1⟩
    · apply logical_output
      exact ⟨by unfold outputStart outputBase; omega, by
        simpa [InRange, outputStart, outputCount,
          PilotProduction.outputDigestStart] using upper.2⟩

end NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport
