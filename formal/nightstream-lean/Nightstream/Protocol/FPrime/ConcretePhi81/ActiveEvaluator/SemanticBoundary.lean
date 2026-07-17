import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Evaluator.SemanticBoundary

/-!
Semantic soundness and honest completeness for the fixed-active outer
F-prime evaluator.

Owns: the exact per-certificate semantic/security premises needed to close
physical execution to the independent outer relation, exhaustive honest
construction of an accepted canonical output or exact sampler shortfall, and
a stronger compatibility theorem for callers that already prove sampler
success.

Does not own: derivation of source authority from public inputs, output
binding, bad-event probability bounds, production checker refinement, base
steps, Rust, R1CS, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `SoundnessClosure` is not verifier input and is never
hashed as authority. It packages the semantic source interpretation, complete
PiCCS output binding, and bad-event exclusion required by the paper's
soundness reduction. Honest completeness uses `SemanticPremises` and preserves
bounded rejection-sampler shortfall as an explicit outcome; it does not claim
that every arbitrary `ResultTransition` is executable.

| Stage path | Mathematical obligation | Assurance status | Lean owner |
|---|---|---|---|
| `fprime.active.soundness.nifs_closure` | the selected physical NIFS execution has source/output authority and no named bad event | explicit semantic/security premises | `SoundnessClosure.nifs` |
| `fprime.active.soundness.closed` | successful outer execution yields the independent `Holds` relation | conditional model theorem | `run_sound_of_closure` |
| `fprime.active.completeness.outcome` | independent paper/source premises plus the five outer equations produce canonical acceptance or exact sampler shortfall | exhaustive model theorem | `exists_run_and_holds_or_samplerShortfall` |
| `fprime.active.completeness.nifs` | successful-sampler premises construct one accepted selected result | compatibility theorem | `run_complete_of_outer_and_honestNifs` |
| `fprime.active.completeness.outer` | the five outer equations plus successful sampling construct canonical executable acceptance | compatibility theorem | `exists_run_and_holds_of_outer_and_honestNifs` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
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
variable {Digest : Type uDigest}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {domain : FlatNcDomain}
variable {publicRingColumns verifierRows slotCount : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Per-certificate semantic/security closure for the selected NIFS
invocation. The five outer equations remain physically checked by `run`. -/
structure SoundnessClosure
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (certificate : Certificate setup input) where
  data : Data shape
  nifs :
    FixedActive.Evaluator.SoundnessClosure
      (contextAt setup input certificate.selected) data certificate.nifs

/-- Once the selected NIFS execution has the explicit semantic/security
closure, successful physical outer execution implies the independent active
F-prime relation. -/
theorem run_sound_of_closure
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (certificate : Certificate setup input)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount)
    (closure : SoundnessClosure setup input certificate)
    (executed : run checkers input certificate = some output) :
    ActiveSemantics.Holds setup machine functionIndex input output := by
  rcases
      (run_eq_some_iff_physicalChecks checkers input certificate output).1
        executed with
    ⟨selectedNext, physical⟩
  have nifsExecuted :
      FixedActive.Evaluator.run
          (checkers.nifs input certificate.selected) certificate.nifs =
        some selectedNext :=
    (FixedActive.Evaluator.run_eq_some_iff_accepted
      (checkers.nifs input certificate.selected) certificate.nifs
      selectedNext).2
      ⟨physical.nifsAccepted, physical.resultExact⟩
  have transition :
      FixedActive.ResultTransition
        (contextAt setup input certificate.selected) selectedNext :=
    FixedActive.Evaluator.run_sound_of_closure
      noZeroDivisors closure.nifs nifsExecuted
  exact ⟨certificate.selected, selectedNext, {
    iterationPositive := physical.outer.iterationPositive
    priorSlot := physical.outer.priorSlot
    priorPublicInput := physical.outer.priorPublicInput
    expectedStructure := physical.outer.expectedStructure
    selectedNifs := transition
    dispatch := physical.outer.dispatch
  }, physical.outputExact⟩

/-- Independent honest completeness for the active evaluator. The exact outer
equations and semantic NIFS premises produce one executable canonical output,
unless the fixed bounded sampler names a concrete shortfall coordinate. -/
theorem exists_run_and_holds_or_samplerShortfall
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (outer : OuterChecks setup machine functionIndex input selected)
    (honest :
      ActiveSemantics.HonestNifs.SemanticPremises setup input selected) :
    (∃ certificate : Certificate setup input,
      ∃ output :
          Output Digest AppState shape publicRingColumns publicFits
            verifierRows slotCount,
        run checkers input certificate = some output ∧
          ActiveSemantics.Holds setup machine functionIndex input output) ∨
      ConcretePhi81.HonestSamplerShortfall
        (contextAt setup input selected) honest.data := by
  rcases
      FixedActive.Evaluator.run_complete_or_samplerShortfall
        (contextAt setup input selected)
        (checkers.nifs input selected)
        honest.data honest.paper honest.semanticInput honest.running with
    completed | shortfall
  · rcases completed with
      ⟨nifsCertificate, result, nifsExecuted, transition⟩
    let certificate : Certificate setup input := {
      selected := selected
      nifs := nifsCertificate
    }
    have outerChecked : outerCheck checkers input selected = true :=
      (outerCheck_eq_true_iff checkers input selected).2 outer
    have executed :
        run checkers input certificate =
          some (outputOf machine input selected result) := by
      simp [run, certificate, outerChecked, nifsExecuted]
    let output := outputOf machine input selected result
    apply Or.inl
    refine ⟨certificate, output, executed, ?_⟩
    exact ⟨selected, result, {
      iterationPositive := outer.iterationPositive
      priorSlot := outer.priorSlot
      priorPublicInput := outer.priorPublicInput
      expectedStructure := outer.expectedStructure
      selectedNifs := transition
      dispatch := outer.dispatch
    }, rfl⟩
  · exact Or.inr shortfall

/-- Honest NIFS construction plus the exact outer checks produces one raw
certificate and its canonical semantic result. -/
theorem run_complete_of_outer_and_honestNifs
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (outer : OuterChecks setup machine functionIndex input selected)
    (honest :
      ActiveSemantics.HonestNifs.Premises setup input selected) :
    exists certificate : Certificate setup input,
      exists result : Slot shape publicRingColumns publicFits verifierRows,
        run checkers input certificate =
            some (outputOf machine input selected result) /\
          FixedActive.ResultTransition
            (contextAt setup input selected) result := by
  rcases
      FixedActive.Evaluator.run_complete
        (contextAt setup input selected)
        (checkers.nifs input selected)
        honest.data honest.paper honest.semanticInput honest.running
        honest.challenges honest.samplerAvailable with
    ⟨nifsCertificate, result, nifsExecuted, transition⟩
  let certificate : Certificate setup input := {
    selected := selected
    nifs := nifsCertificate
  }
  have outerChecked : outerCheck checkers input selected = true :=
    (outerCheck_eq_true_iff checkers input selected).2 outer
  have executed :
      run checkers input certificate =
        some (outputOf machine input selected result) := by
    simp [run, certificate, outerChecked, nifsExecuted]
  exact ⟨certificate, result, executed, transition⟩

/-- Honest completeness of the active evaluator: independent paper/source
premises and the exact outer equations yield an executable output satisfying
the independent active relation. -/
theorem exists_run_and_holds_of_outer_and_honestNifs
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (outer : OuterChecks setup machine functionIndex input selected)
    (honest :
      ActiveSemantics.HonestNifs.Premises setup input selected) :
    exists certificate : Certificate setup input,
      exists output :
          Output Digest AppState shape publicRingColumns publicFits
            verifierRows slotCount,
        run checkers input certificate = some output /\
          ActiveSemantics.Holds setup machine functionIndex input output := by
  rcases
      run_complete_of_outer_and_honestNifs
        checkers input selected outer honest with
    ⟨certificate, result, executed, transition⟩
  let output := outputOf machine input selected result
  refine ⟨certificate, output, executed, ?_⟩
  exact ⟨selected, result, {
    iterationPositive := outer.iterationPositive
    priorSlot := outer.priorSlot
    priorPublicInput := outer.priorPublicInput
    expectedStructure := outer.expectedStructure
    selectedNifs := transition
    dispatch := outer.dispatch
  }, rfl⟩

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
