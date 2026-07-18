import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.Evaluator
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2
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
| `fprime.fixed_one.semantic.construction2` | successful execution projects to the public selected-NIFS recursive edge | conditional model theorem | `run_refinesConstruction2_of_closure` |
| `fprime.fixed_one.semantic.security_partition` | execution reaches Construction 2 or one named NIFS binding/security failure | exhaustive theorem | `run_refinesConstruction2_or_securityFailure` |
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
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Semantic/security premises that are deliberately not inferred from
physical acceptance. -/
structure SoundnessClosure
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
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
      Setup OuterKey AppState Witness TranscriptState shape
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

/-- The complete conditional executable-to-semantics chain projected to the
paper-faithful Construction-2 recursive branch. The closure remains explicit;
this theorem does not claim a bad-event bound or Rust/R1CS conformance. -/
theorem run_refinesConstruction2_of_closure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows 1)
    (closure : SoundnessClosure setup input certificate)
    (executed : run machine setup input certificate = some output) :
    Paper.Construction2.RecursiveHolds
      (SelectedNifsSemantics.family
        (ActiveSemantics.Construction2.selectedNifsSetup setup))
      machine functionIndex (input.toActive setup).toPaper output.toPaper := by
  have canonical :
      ActiveSemantics.FixedOneCanonical.Holds setup machine
        (input.toSemantic setup) output :=
    run_sound_of_closure noZeroDivisors machine setup input certificate output
      closure executed
  have active :
      ActiveSemantics.Holds setup machine functionIndex
        (input.toActive setup) output :=
    (ActiveSemantics.FixedOneCanonical.holds_iff_active setup machine
      functionIndex (input.toSemantic setup) output).1 canonical
  exact ActiveSemantics.Construction2.sound_selectedNifs active

/-- Successful fixed-one execution reaches the paper-faithful recursive edge
or exposes exactly one unresolved NIFS binding/security family. No closure
premise is hidden in this theorem; probability bounds remain open. -/
theorem run_refinesConstruction2_or_securityFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows 1)
    (executed : run machine setup input certificate = some output) :
    Paper.Construction2.RecursiveHolds
        (SelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex (input.toActive setup).toPaper output.toPaper ∨
      FixedActive.Evaluator.SecurityFailure
        (nifsContext setup input).materialize certificate := by
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
  rcases FixedActive.Evaluator.run_sound_or_securityFailure noZeroDivisors
      nifsExecuted with transition | failure
  · have canonical :
        ActiveSemantics.FixedOneCanonical.Holds setup machine
          (input.toSemantic setup) output := by
      refine ⟨result, ?_, physical.outputExact⟩
      exact {
        iterationPositive := physical.outer.iterationPositive
        priorPublicInput := physical.outer.priorPublicInput
        selectedNifs := by
          simpa [nifsContext_materialize] using transition
      }
    have active :
        ActiveSemantics.Holds setup machine functionIndex
          (input.toActive setup) output :=
      (ActiveSemantics.FixedOneCanonical.holds_iff_active setup machine
        functionIndex (input.toSemantic setup) output).1 canonical
    exact Or.inl
      (ActiveSemantics.Construction2.sound_selectedNifs active)
  · exact Or.inr failure

/-- Independent honest inputs either execute to a canonical output satisfying
the fixed-one relation or expose one exact bounded-sampler shortfall. -/
theorem exists_run_and_holds_or_samplerShortfall
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
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
