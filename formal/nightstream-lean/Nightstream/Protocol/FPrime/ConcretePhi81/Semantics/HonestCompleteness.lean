import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs
import Nightstream.Protocol.FPrime.ConcretePhi81.Semantics

/-!
Branch-complete honest construction for the ConcretePhi81 F-prime relation.

Assurance tier: model-level.

Owns: the five non-NIFS recursive equations, a branch-tagged honest premise
language, and exhaustive construction of a canonical accepted output or an
exact bounded-sampler shortfall.

Does not own: a reverse refinement from the public Construction-2 relation,
sampler probability, transcript security, executable checking, Rust, R1CS,
costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: recursive premises contain independent paper/source
truth and checked incoming-parent authority through
`HonestNifs.SemanticPremises`; they contain neither a completed NIFS result
nor a sampler-success bit. The accepted output is computed. Sampler failure
is preserved as a typed outcome rather than excluded by assumption.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.completeness.base` | the three general base equations construct the canonical base output | semantic premises | `Premises.base`, `complete` |
| `fprime.completeness.recursive.outer` | positivity, slot, prior link, structure, and dispatch | semantic premises | `RecursivePremises` |
| `fprime.completeness.recursive.nifs` | independent honest source family and incoming authority | semantic premises | `RecursivePremises.nifs` |
| `fprime.completeness.recursive.output` | one semantic NIFS result determines the full rich output | computed | `complete` |
| `fprime.completeness.sampler_shortfall` | bounded rejection sampling may name one failed coordinate | explicit exhaustive outcome | `Outcome`, `complete` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Semantics.HonestCompleteness

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Protocol.FPrime.ConcretePhi81.Outer

universe uOuterKey uAppState uWitness uDigest uTranscriptState

/-- Exact outer equations plus independent source truth for one honest
recursive branch. -/
structure RecursivePremises
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Semantics.Setup OuterKey Digest AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount) where
  iterationPositive : 0 < input.iteration
  priorSlot : input.priorPc = selected.val + 1
  priorPublicInput :
    input.fresh.publicInput =
      setup.machine.encodeInstance
        (setup.machine.hash (Paper.priorHashPreimage input.toPaper))
  expectedStructure :
    input.fresh.constraintSystem =
      setup.active.expectedStructure input.verifierKey selected
  dispatch :
    setup.machine.control input.zi input.witness =
      Paper.ProgramCounter.ofIndex functionIndex
  nifs : ActiveSemantics.HonestNifs.SemanticPremises
    setup.active input selected

/-- Honest branch language. No output or completed recursive result is
caller-supplied. -/
inductive Premises
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Semantics.Setup OuterKey Digest AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) where
  | base (obligations :
      BaseSemantics.Obligations setup.machine functionIndex input) :
      Premises setup functionIndex input
  | recursive
      (selected : Fin slotCount)
      (premises : RecursivePremises setup functionIndex input selected) :
      Premises setup functionIndex input

/-- Exhaustive honest result: canonical relation acceptance or one exact
bounded-sampler shortfall. -/
def Outcome
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup :
      Semantics.Setup OuterKey Digest AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) : Prop :=
  (exists output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount,
    Semantics.Holds setup functionIndex input output) \/
  exists selected : Fin slotCount,
    exists data : PiCCS.SplitNc.Sources.Data shape,
      ConcretePhi81.HonestSamplerShortfall
        (ActiveSemantics.contextAt setup.active input selected) data

/-- Honest branch completeness without assuming sampler success. Base inputs
always produce the canonical base output. Recursive inputs either produce the
canonical semantic NIFS/F-prime output or expose the exact sampler shortfall
already named by the finite production sampler. -/
theorem complete
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup :
      Semantics.Setup OuterKey Digest AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount}
    (premises : Premises setup functionIndex input) :
    Outcome setup functionIndex input := by
  cases premises with
  | base obligations =>
      apply Or.inl
      refine ⟨BaseSemantics.outputOf setup.machine setup.default input, ?_⟩
      exact .base (BaseSemantics.complete setup.machine setup.default
        functionIndex input obligations)
  | recursive selected recursive =>
      rcases
          recursive.nifs.exists_resultTransition_or_samplerShortfall
            setup.active input selected with
        completed | shortfall
      · rcases completed with ⟨certificate, _accepted, transition⟩
        let selectedNext :=
          FixedActive.resultOf
            (ActiveSemantics.contextAt setup.active input selected) certificate
        apply Or.inl
        refine ⟨ActiveSemantics.outputOf setup.machine input selected
          selectedNext, ?_⟩
        apply Semantics.Holds.recursive
        exact ⟨selected, selectedNext, {
          iterationPositive := recursive.iterationPositive
          priorSlot := recursive.priorSlot
          priorPublicInput := recursive.priorPublicInput
          expectedStructure := recursive.expectedStructure
          selectedNifs := transition
          dispatch := recursive.dispatch
        }, rfl⟩
      · exact Or.inr ⟨selected, recursive.nifs.data, shortfall⟩

end Nightstream.Protocol.FPrime.ConcretePhi81.Semantics.HonestCompleteness
