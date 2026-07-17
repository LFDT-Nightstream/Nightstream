import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.Evaluator
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Evaluator.SemanticBoundary

/-!
Semantic boundary for the payload-minimal fixed-one F-prime evaluator.

Owns: the explicit assumptions that close physical execution to independent
F-prime semantics, conditional soundness, and honest completeness with exact
sampler shortfall.

Does not own: physical checking, output construction, derivation of source or
output authority, bad-event probability bounds, Rust/R1CS refinement, costs,
or row removal.

Emits constraints: no.

| Stage path | Mathematical obligation | Assurance status | Lean owner |
|---|---|---|---|
| `fprime.fixed_one.semantic.closure` | source/output authority and bad-event exclusion | explicit semantic/security premises | `SoundnessClosure` |
| `fprime.fixed_one.semantic.sound` | successful execution implies independent F-prime semantics | conditional model theorem | `run_sound_of_closure` |
| `fprime.fixed_one.semantic.complete` | honest premises execute or expose bounded sampler shortfall | exhaustive model theorem | `exists_run_and_holds_or_samplerShortfall` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

universe uOuterKey uAppState uWitness uDigest uTranscriptState

section

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {domain : FlatNcDomain}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Semantic/security premises that are deliberately not inferred from
physical acceptance. -/
structure SoundnessClosure
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input) where
  data : Data shape
  nifs : FixedActive.Evaluator.SoundnessClosure
    (nifsContext setup input).materialize data certificate

/-- With explicit source/output authority and bad-event exclusion, successful
physical execution implies the independent fixed-one F-prime relation. -/
theorem run_sound_of_closure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows 1)
    (closure : SoundnessClosure setup input certificate)
    (executed : run machine setup input certificate = some output) :
    ActiveSemantics.FixedOneCanonical.Holds setup machine
      (input.toSemantic setup) output := by
  rcases
      (run_eq_some_iff_physicalChecks machine setup input certificate output).1
        executed with
    ⟨result, physical⟩
  have nifsExecuted :
      FixedActive.Evaluator.run
          (FixedActive.Canonical.Checker.evaluatorChecker
            (nifsContext setup input)) certificate = some result :=
    (FixedActive.Evaluator.run_eq_some_iff_accepted
      (FixedActive.Canonical.Checker.evaluatorChecker
        (nifsContext setup input)) certificate result).2
      ⟨physical.nifsAccepted, physical.resultExact⟩
  have transition :
      FixedActive.ResultTransition
        (nifsContext setup input).materialize result :=
    FixedActive.Evaluator.run_sound_of_closure
      noZeroDivisors closure.nifs nifsExecuted
  refine ⟨result, ?_, physical.outputExact⟩
  exact {
    iterationPositive := physical.outer.iterationPositive
    priorPublicInput := physical.outer.priorPublicInput
    selectedNifs := by
      simpa [nifsContext_materialize] using transition
  }

/-- Independent honest inputs either execute to a canonical output satisfying
the fixed-one relation or expose one exact bounded-sampler shortfall. -/
theorem exists_run_and_holds_or_samplerShortfall
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (outer : OuterChecks machine setup input)
    (honest : ActiveSemantics.HonestNifs.SemanticPremises setup
      (input.toActive setup) ActiveSemantics.FixedOneCanonical.selected) :
    (exists certificate : Certificate setup input,
      exists output :
          Output Digest AppState shape publicRingColumns publicFits
            verifierRows 1,
        run machine setup input certificate = some output /\
          ActiveSemantics.FixedOneCanonical.Holds setup machine
            (input.toSemantic setup) output) \/
      ConcretePhi81.HonestSamplerShortfall
        (nifsContext setup input).materialize honest.data := by
  have semanticInput : ConcretePhi81.SemanticInput
      (nifsContext setup input).materialize honest.data := by
    simpa [nifsContext_materialize] using honest.semanticInput
  have running : ConcretePhi81.RunningAuthority.Accepted
      (nifsContext setup input).materialize := by
    simpa [nifsContext_materialize] using honest.running
  rcases
      FixedActive.Evaluator.run_complete_or_samplerShortfall
        (nifsContext setup input).materialize
        (FixedActive.Canonical.Checker.evaluatorChecker
          (nifsContext setup input))
        honest.data honest.paper semanticInput running with
    completed | shortfall
  · rcases completed with
      ⟨certificate, result, nifsExecuted, transition⟩
    have outerChecked : outerCheck machine setup input = true :=
      (outerCheck_eq_true_iff machine setup input).2 outer
    have executed :
        run machine setup input certificate =
          some (ActiveSemantics.outputOf machine (input.toActive setup)
            ActiveSemantics.FixedOneCanonical.selected result) := by
      simp [run, outerChecked, nifsExecuted]
    apply Or.inl
    refine ⟨certificate,
      ActiveSemantics.outputOf machine (input.toActive setup)
        ActiveSemantics.FixedOneCanonical.selected result,
      executed, result, ?_, rfl⟩
    exact {
      iterationPositive := outer.iterationPositive
      priorPublicInput := outer.priorPublicInput
      selectedNifs := by
        simpa [nifsContext_materialize] using transition
    }
  · exact Or.inr shortfall

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical
