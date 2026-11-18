import NightstreamFPrime.Export.BoundedDepth.IvcSecurity
import NightstreamFPrime.Export.BoundedDepth.RunCompleteness

/-! Owns the result of the terminal verifier, Rust `Verifier::verify` (SPEC.md Algorithm 6).
`IvcSecurity.probability_bound` bounds an acceptance of the terminal relation. The verifier also
holds each slot to the bound of its depth (`RunCompleteness.Verifies`), so this file states the
bound for the whole accept condition of the verifier. -/

namespace NightstreamFPrime.Export.BoundedDepth.VerifierSecurity

open scoped ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.ReverseStages
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Export.BoundedDepth.TerminalBridge
open NightstreamFPrime.Export.BoundedDepth.IvcWalk
open NightstreamFPrime.Export.BoundedDepth.IvcSecurity
open NightstreamFPrime.Export.BoundedDepth.RunCompleteness

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  {step : AppState → AppWitness → AppState}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
  (exports : Exports relation ajtai circuitKeys step cubeFits)

/-- The verifier of SPEC.md Algorithm 6 accepts, and no `iteration` valid steps of the
application lead from `z0` to `zi`. -/
def FalseVerification (packageWidths : Fin 2 → Nat) (control : AppState → AppWitness → Fin 2)
    (cap : Nat) (output : Output logicalWidth publicFits) : Prop :=
  Verifies ajtai circuitKeys step packageWidths cap control output.1 output.2 ∧
    ¬ Reaches exports.program output.1

variable {Admitted : NifsAdversary (WalkAttempt logicalWidth publicFits) → Prop}
  {StageAdmitted : Stage (Visit logicalWidth publicFits) (Answer logicalWidth publicFits) → Prop}
  {Efficient : (adversary : NifsAdversary (WalkAttempt logicalWidth publicFits)) →
    NifsExtractor (Answer logicalWidth publicFits) adversary → Prop}
  {error : NifsAdversary (WalkAttempt logicalWidth publicFits) → ℝ}
  (assumption : Assumption1 (WalkSucceeds ajtai) (WalkExtracted ajtai) Admitted Efficient error)
  (closed : Closed (attemptOf ajtai circuitKeys exports) (back ajtai circuitKeys exports) Admitted
    StageAdmitted Efficient)
  (adversary : IvcAdversary (Output logicalWidth publicFits))
  (admitted : StageAdmitted (Stage.start (walkAdversary adversary)))

/-- The probability that the verifier of SPEC.md Algorithm 6 accepts the adversary's output and no
`iteration` valid steps lead from `z0` to `zi` is at most `walkBound`, the bound of
`probability_bound`. -/
theorem soundness (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (packageWidths : Fin 2 → Nat) (control : AppState → AppWitness → Fin 2) (cap depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    (adversary.tape.toOuterMeasure
        {tape | FalseVerification ajtai circuitKeys exports packageWidths control cap
          (adversary.output tape)}).toReal ≤
      walkBound ajtai circuitKeys exports assumption closed adversary admitted depth :=
  le_trans
    (ENNReal.toReal_mono (event_ne_top _ _)
      (MeasureTheory.OuterMeasure.mono _ fun tape
          (accepted : FalseVerification ajtai circuitKeys exports packageWidths control cap
            (adversary.output tape)) =>
        (⟨holds_of_verifies accepted.1, accepted.2⟩ :
          FalseAcceptance ajtai circuitKeys exports control cap (adversary.output tape))))
    (probability_bound ajtai circuitKeys exports assumption closed adversary admitted keysShaped
      control cap depth depthBound)

end NightstreamFPrime.Export.BoundedDepth.VerifierSecurity
