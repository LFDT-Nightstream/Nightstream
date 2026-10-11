import NightstreamFPrime.Layout.Stage1.RunningTransitionOutputBounds
import NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupportData
import NightstreamFPrime.Lifecycle.Stage1.RunningTransitionSupport

/-! Owns compact source support for the output running words. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Stage1
open RunningTransitionInputs

theorem outputSupported (index : RunningTransition.WordIndex) :
    (outputWord index).VarsSatisfy Logical := by
  have upper := outputWordBelowOutputDigestStart index
  simp only [outputWord, Expr.VarsBelow] at upper
  simp only [outputWord, Expr.VarsSatisfy]
  apply logical_output
  exact ⟨by unfold outputStart outputBase; omega, by
    simpa [InRange, outputStart, outputCount,
      PilotProduction.outputDigestStart] using upper⟩

end NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport
