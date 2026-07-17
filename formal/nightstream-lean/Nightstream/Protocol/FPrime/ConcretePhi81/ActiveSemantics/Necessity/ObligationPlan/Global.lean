import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan

/-!
Global six-family check plan for the independent fixed-active F-prime
obligations.

Owns: the actual verifier-language carrier containing the outer input,
selected slot, and complete fold result; exactness of the six named families
over that carrier; lifting of per-input removal witnesses; and the final
inclusion-minimality closure interface.

Does not own: construction of any family witness, physical certificate
acceptance, production fixtures, Rust, R1CS, costs, global gate-count
minimality, or row removal.

Emits constraints: no.

Authority boundary: unlike the local `ObligationPlan.Candidate`, `Case`
contains the outer input itself. This is necessary for an honest
inclusion-minimality claim: counterexamples for iteration, link, structure,
NIFS, and dispatch need not share one fixed input. `Witnesses` requires a real
typed removal witness for every family before `inclusionMinimalSound` can be
derived.

| Stage path | Mathematical obligation | Assurance status | Lean owner |
|---|---|---|---|
| `fprime.active.obligation_plan.global.case` | one outer input, selected slot, and complete fold result | actual typed carrier | `Case` |
| `fprime.active.obligation_plan.global.exact` | the six leaves accept exactly the independent obligations across all inputs | exact model theorem | `exact` |
| `fprime.active.obligation_plan.global.lift` | a local actual-type counterexample embeds into the global verifier language | exact model theorem | `lift_local_necessary` |
| `fprime.active.obligation_plan.global.witnesses` | one global removal witness exists for every retained family | explicit closure boundary | `Witnesses` |
| `fprime.active.obligation_plan.global.minimal` | exactness plus all six witnesses gives inclusion-minimal soundness | conditional model theorem | `inclusionMinimalSound` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uOuterKey uAppState uWitness uDigest uTranscriptState

/-- One actual verifier-language case. Different removal witnesses may carry
different outer inputs while sharing the same verifier-owned setup and
machine. -/
structure Case
    (OuterKey : Type uOuterKey)
    (AppState : Type uAppState)
    (Witness : Type uWitness)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows slotCount : Nat) where
  input :
    Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount
  selected : Fin slotCount
  selectedNext :
    Slot shape publicRingColumns publicFits verifierRows

namespace Case

/-- Project one global case to the local selected-slot/result carrier. -/
def toLocal
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (case :
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) :
    Candidate shape publicRingColumns publicFits verifierRows slotCount := {
  selected := case.selected
  selectedNext := case.selectedNext
}

end Case

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

/-- Interpret each family over the complete global case. -/
def semantics
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount) :
    Family ->
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount ->
        Prop :=
  fun family case =>
    ObligationPlan.semantics setup machine functionIndex case.input
      family case.toLocal

/-- Independent active target over the complete global case. -/
def target
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (case :
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) :
    Prop :=
  ObligationPlan.target setup machine functionIndex case.input case.toLocal

/-- Global plan acceptance is exactly the independent six-field obligations
for every actual outer input and selected result. -/
theorem accepts_iff_obligations
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (case :
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) :
    CheckPlan.Accepts
        (semantics setup machine functionIndex) checks case ↔
      target setup machine functionIndex case := by
  exact
    ObligationPlan.accepts_iff_obligations
      setup machine functionIndex case.input case.toLocal

/-- The six-family plan is exact across the complete actual input language. -/
theorem exact
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount) :
    CheckPlan.Exact
      (semantics setup machine functionIndex)
      (target setup machine functionIndex)
      checks := by
  intro case
  exact accepts_iff_obligations setup machine functionIndex case

/-- Embed any actual per-input removal witness into the common global
verifier language. -/
theorem lift_local_necessary
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (family : Family)
    (necessary :
      CheckPlan.NecessaryForSoundness
        (ObligationPlan.semantics setup machine functionIndex input)
        (ObligationPlan.target setup machine functionIndex input)
        checks family) :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex)
      (target setup machine functionIndex)
      checks family := by
  rcases necessary with ⟨candidate, weakened, rejected⟩
  let case :
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount := {
    input := input
    selected := candidate.selected
    selectedNext := candidate.selectedNext
  }
  refine ⟨case, ?_, ?_⟩
  · simpa [semantics, case, Case.toLocal] using weakened
  · simpa [target, case, Case.toLocal] using rejected

/-- Exact closure evidence required for a global inclusion-minimality claim. -/
structure Witnesses
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount) where
  iterationPositive :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex)
      (target setup machine functionIndex) checks .iterationPositive
  priorSlot :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex)
      (target setup machine functionIndex) checks .priorSlot
  priorPublicInput :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex)
      (target setup machine functionIndex) checks .priorPublicInput
  expectedStructure :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex)
      (target setup machine functionIndex) checks .expectedStructure
  selectedNifs :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex)
      (target setup machine functionIndex) checks .selectedNifs
  dispatch :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex)
      (target setup machine functionIndex) checks .dispatch

/-- Exact global plan soundness plus all six actual-type removal witnesses
establish inclusion-minimality relative to the selected protocol
obligations. -/
theorem inclusionMinimalSound
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (witnesses : Witnesses setup machine functionIndex) :
    CheckPlan.InclusionMinimalSound
      (semantics setup machine functionIndex)
      (target setup machine functionIndex)
      checks := by
  apply CheckPlan.inclusionMinimalSound_of_witnesses
  · intro case accepted
    exact
      (accepts_iff_obligations setup machine functionIndex case).1 accepted
  · intro family _member
    cases family with
    | iterationPositive =>
        exact witnesses.iterationPositive
    | priorSlot =>
        exact witnesses.priorSlot
    | priorPublicInput =>
        exact witnesses.priorPublicInput
    | expectedStructure =>
        exact witnesses.expectedStructure
    | selectedNifs =>
        exact witnesses.selectedNifs
    | dispatch =>
        exact witnesses.dispatch

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global
