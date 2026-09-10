import NightstreamFPrime.Export.Stage1.PerApplicationSecurity
import NightstreamFPrime.Lifecycle.Nifs.TotalizedComparison

/-!
Owns the deterministic sampler-comparison edge for a verifier-selected
application step and its bound raw assignment. The base branch requires no
NIFS acceptance. The recursive branch preserves the complete running output
and successful sampler batch. No sampling law, work, or security premise is
added, and the production verifier is unchanged.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PerApplicationSamplerComparison

open NightstreamFPrime.Export.Stage1.PerApplicationSecurity
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold

/-- The exact augmented step separates the base branch from recursive NIFS
acceptance. On the recursive branch, scalarwise totalization preserves the
complete selected running output and actual returned sampler batch. -/
theorem stepHoldsFor_implies_base_or_comparison {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program)
    (fallback : Scalar)
    (input : StepInput program fits) (output : StepOutput program)
    (step : Lifecycle.StepHoldsFor
      (PerApplicationFixedPoint.relation program fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (verifierContextDigest fits commitmentSetup) program input output) :
    let initial := ((canonicalKey fits commitmentSetup).piCcsExecution
      (selectedRunning input) input.fresh input.nifsProof).outgoingState
    Lifecycle.StepHoldsFor
        (PerApplicationFixedPoint.relation program fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (verifierContextDigest fits commitmentSetup) program input output ∧
      (input.iteration = 0 ∨
        (0 < input.iteration ∧
          Lifecycle.Nifs.TotalizedComparison.verify fallback
              (PerApplicationFixedPoint.relation program fits)
              (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
              (selectedRunning input) input.fresh input.nifsProof =
            some (output.runningNext functionIndex) ∧
          Transcript.PiRlcSampler.sampleBatch initial
              Spec.Folding.Nifs.PaperProfile.arity.total =
            some (Lifecycle.Nifs.TotalizedComparison.totalizedBatch fallback
              initial Spec.Folding.Nifs.PaperProfile.arity.total))) := by
  refine ⟨step, ?_⟩
  change FixedAugmentedTransition
    (Lifecycle.setup (PerApplicationFixedPoint.relation program fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (verifierContextDigest fits commitmentSetup))
    (Lifecycle.machineFor (PerApplicationFixedPoint.publicFits program) program)
    functionIndex input output at step
  rcases step.2.2.2 with base | recursive
  · exact Or.inl base.1
  · rcases recursive with
      ⟨priorPcValid, iterationPositive, _priorPublic, selectedNifs, _unchanged⟩
    have selectedEq : selectedIndex priorPcValid = functionIndex := by
      apply Fin.ext
      have bound := (selectedIndex priorPcValid).isLt
      change (selectedIndex priorPcValid).val < 1 at bound
      change (selectedIndex priorPcValid).val = 0
      omega
    rw [selectedEq] at selectedNifs
    have accepted :
        Spec.Folding.Nifs.PaperNonInteractive.verify
            (canonicalKey fits commitmentSetup) (selectedRunning input)
            input.fresh input.nifsProof =
          some (output.runningNext functionIndex) := by
      simpa [Accepts, Lifecycle.setup, Lifecycle.nifsVerifier,
        canonicalKey, selectedRunning] using selectedNifs
    exact Or.inr ⟨iterationPositive,
      Lifecycle.Nifs.TotalizedComparison.accepted_actual_implies_comparison
        fallback (PerApplicationFixedPoint.relation program fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (selectedRunning input) input.fresh input.nifsProof
        (output.runningNext functionIndex) accepted⟩

/-- The same verifier-bound raw assignment determines the input, proof, and
output under the canonical key. Accepted rows imply the exact augmented step and
the base-or-recursive comparison result, with no semantic premise supplied
by the caller. -/
theorem verifierBoundRowsZero_implies_base_or_comparison
    {program : Program} (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program)
    (fallback : Scalar)
    (raw : PerApplicationCanonicalAssignment.RawValues program)
    (accepted : (PerApplicationFixedPoint.structuralPlan program fits).RowsZero
      (PerApplicationVerifierBoundAssignment.bind fits commitmentSetup raw).assignment) :
    let bound := PerApplicationVerifierBoundAssignment.bind fits commitmentSetup raw
    let input := PerApplicationDecodedIO.input program fits bound
    let output := PerApplicationDecodedIO.output program bound
    let initial := ((canonicalKey fits commitmentSetup).piCcsExecution
      (selectedRunning input) input.fresh input.nifsProof).outgoingState
    Lifecycle.StepHoldsFor
        (PerApplicationFixedPoint.relation program fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (verifierContextDigest fits commitmentSetup) program input output ∧
      (input.iteration = 0 ∨
        (0 < input.iteration ∧
          Lifecycle.Nifs.TotalizedComparison.verify fallback
              (PerApplicationFixedPoint.relation program fits)
              (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
              (selectedRunning input) input.fresh input.nifsProof =
            some (output.runningNext functionIndex) ∧
          Transcript.PiRlcSampler.sampleBatch initial
              Spec.Folding.Nifs.PaperProfile.arity.total =
            some (Lifecycle.Nifs.TotalizedComparison.totalizedBatch fallback
              initial Spec.Folding.Nifs.PaperProfile.arity.total))) := by
  dsimp only
  apply stepHoldsFor_implies_base_or_comparison fits commitmentSetup fallback _ _
  exact PerApplicationFixedPointSoundness.verifierBoundRowsZero_implies_stepHoldsFor
    program fits commitmentSetup raw accepted

end NightstreamFPrime.Export.Stage1.PerApplicationSamplerComparison
