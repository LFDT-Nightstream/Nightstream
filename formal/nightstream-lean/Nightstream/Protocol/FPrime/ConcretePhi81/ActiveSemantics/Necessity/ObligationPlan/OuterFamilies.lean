import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.SideAnchor

/-!
Actual-type lifting of the three outer F-prime removal witnesses into the
complete six-family obligation plan.

Owns: the exact mapping from the outer iteration/prior-link/dispatch families
to the six-family plan; construction of the real selected-slot/result
candidate from an actual `ConcreteRealization`; preservation of every retained
family; rejection by the independent active obligations; and independent
honest-anchor construction with explicit sampler shortfall.

Does not own: construction of a production bad input, an honest side anchor,
the prior-slot, structure, or selected-NIFS removal witnesses, executable
checking, Rust, R1CS, costs, inclusion-minimality, or row removal.

Emits constraints: no.

Authority boundary: `ConcreteRealization` contains a real ConcretePhi81
`SideAnchor` plus an actual outer view. This module only transports that
evidence into the exact six-family plan. It cannot manufacture the bad view or
promote the generic Boolean countermodels to production evidence.

| Stage path | Mathematical obligation | Assurance status | Lean owner |
|---|---|---|---|
| `fprime.active.necessity.outer.map` | map the three outer families into the exact six-family vocabulary | computed | `mappedFamily` |
| `fprime.active.necessity.outer.candidate` | retain the actual selected slot and complete fold result | direct dataflow | `candidate` |
| `fprime.active.necessity.outer.preservation` | the other five actual obligations survive one outer-family removal | conditional actual-type theorem | `weakened` |
| `fprime.active.necessity.outer.rejection` | the resulting candidate violates the independent active target | conditional actual-type theorem | `rejected` |
| `fprime.active.necessity.outer.necessary` | each supplied actual realization is a six-family removal witness | conditional actual-type necessity | `necessary` |
| `fprime.active.necessity.outer.outcome` | independent honest NIFS authority yields each actual outer witness or one exact sampler shortfall | exhaustive model outcome | `*_necessary_or_samplerShortfall_of_semanticPremises` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity

universe uOuterKey uAppState uWitness uDigest uTranscriptState

/-- Exact embedding of the generic outer-family vocabulary into the complete
active-obligation vocabulary. -/
def mappedFamily : OuterPlan.Family -> Family
  | .activeIteration => .iterationPositive
  | .priorPublicLink => .priorPublicInput
  | .dispatch => .dispatch

section

variable {OuterKey : Type uOuterKey}
variable {Digest : Type uDigest}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows slotCount : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Project an actual outer realization to the exact candidate carrier shared
by all six active families. -/
def candidate
    {removed : OuterPlan.Family}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount}
    (realization :
      ConcreteRealization removed setup machine functionIndex input) :
    Candidate shape publicRingColumns publicFits verifierRows slotCount := {
  selected := realization.anchor.selected
  selectedNext := realization.anchor.selectedNext
}

/-- The exact six-family plan without the mapped outer family accepts the
actual candidate carried by the realization. -/
theorem weakened
    {removed : OuterPlan.Family}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount}
    (realization :
      ConcreteRealization removed setup machine functionIndex input) :
    CheckPlan.Accepts
      (semantics setup machine functionIndex input)
      (CheckPlan.without checks (mappedFamily removed))
      (candidate realization) := by
  intro family member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  cases removed with
  | activeIteration =>
      cases family with
      | iterationPositive =>
          exact (retained rfl).elim
      | priorSlot =>
          exact realization.anchor.side.priorSlot
      | priorPublicInput =>
          exact realization.weakenedOuter .priorPublicLink (by
            simp [OuterPlan.checks, CheckPlan.without])
      | expectedStructure =>
          exact realization.anchor.side.expectedStructure
      | selectedNifs =>
          exact realization.anchor.side.selectedNifs
      | dispatch =>
          exact realization.weakenedOuter .dispatch (by
            simp [OuterPlan.checks, CheckPlan.without])
  | priorPublicLink =>
      cases family with
      | iterationPositive =>
          exact realization.weakenedOuter .activeIteration (by
            simp [OuterPlan.checks, CheckPlan.without])
      | priorSlot =>
          exact realization.anchor.side.priorSlot
      | priorPublicInput =>
          exact (retained rfl).elim
      | expectedStructure =>
          exact realization.anchor.side.expectedStructure
      | selectedNifs =>
          exact realization.anchor.side.selectedNifs
      | dispatch =>
          exact realization.weakenedOuter .dispatch (by
            simp [OuterPlan.checks, CheckPlan.without])
  | dispatch =>
      cases family with
      | iterationPositive =>
          exact realization.weakenedOuter .activeIteration (by
            simp [OuterPlan.checks, CheckPlan.without])
      | priorSlot =>
          exact realization.anchor.side.priorSlot
      | priorPublicInput =>
          exact realization.weakenedOuter .priorPublicLink (by
            simp [OuterPlan.checks, CheckPlan.without])
      | expectedStructure =>
          exact realization.anchor.side.expectedStructure
      | selectedNifs =>
          exact realization.anchor.side.selectedNifs
      | dispatch =>
          exact (retained rfl).elim

/-- The actual candidate is outside the six-obligation target whenever its
outer realization is outside the three-equation outer boundary. -/
theorem rejected
    {removed : OuterPlan.Family}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount}
    (realization :
      ConcreteRealization removed setup machine functionIndex input) :
    ¬ target setup machine functionIndex input (candidate realization) := by
  intro targetHolds
  have obligations :
      Obligations setup machine functionIndex input
        realization.anchor.selected realization.anchor.selectedNext := by
    simpa [target, candidate] using targetHolds
  exact realization.rejectedOuter
    ((obligations_iff_side_and_outer
      setup machine functionIndex input realization.anchor.selected
        realization.anchor.selectedNext).1 obligations).2

/-- Any supplied actual outer realization is a removal witness in the exact
six-family plan. This remains conditional until the corresponding production
bad view and side anchor are constructed. -/
theorem necessary
    {removed : OuterPlan.Family}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount}
    (realization :
      ConcreteRealization removed setup machine functionIndex input) :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex input)
      (target setup machine functionIndex input)
      checks (mappedFamily removed) :=
  ⟨candidate realization, weakened realization, rejected realization⟩

/-- Independent semantic NIFS premises and an actual iteration-zero input
produce the iteration-removal witness, unless the fixed bounded sampler names
one exact shortfall coordinate. -/
theorem activeIteration_necessary_or_samplerShortfall_of_semanticPremises
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (premises : HonestNifs.SemanticPremises setup input selected)
    (notPositive : ¬ 0 < input.iteration)
    (priorSlot : input.priorPc = selected.val + 1)
    (priorPublicLink :
      input.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)))
    (expectedStructure :
      input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey selected)
    (dispatch :
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex) :
    CheckPlan.NecessaryForSoundness
        (semantics setup machine functionIndex input)
        (target setup machine functionIndex input)
        checks .iterationPositive ∨
      ConcretePhi81.HonestSamplerShortfall
        (contextAt setup input selected) premises.data := by
  rcases
      exists_sideAnchor_or_samplerShortfall_of_semanticPremises
        setup input selected premises priorSlot expectedStructure with
    baseline | shortfall
  · rcases baseline with ⟨anchor, _selectedEq⟩
    exact Or.inl <| necessary <|
      ConcreteRealization.activeIteration
        setup machine functionIndex input anchor notPositive priorPublicLink
          dispatch
  · exact Or.inr shortfall

/-- Independent semantic NIFS premises and an actual broken prior-public-link
input produce the link-removal witness, unless the fixed bounded sampler names
one exact shortfall coordinate. -/
theorem priorPublicLink_necessary_or_samplerShortfall_of_semanticPremises
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (premises : HonestNifs.SemanticPremises setup input selected)
    (iterationPositive : 0 < input.iteration)
    (priorSlot : input.priorPc = selected.val + 1)
    (priorPublicLinkFails :
      input.fresh.publicInput ≠
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)))
    (expectedStructure :
      input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey selected)
    (dispatch :
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex) :
    CheckPlan.NecessaryForSoundness
        (semantics setup machine functionIndex input)
        (target setup machine functionIndex input)
        checks .priorPublicInput ∨
      ConcretePhi81.HonestSamplerShortfall
        (contextAt setup input selected) premises.data := by
  rcases
      exists_sideAnchor_or_samplerShortfall_of_semanticPremises
        setup input selected premises priorSlot expectedStructure with
    baseline | shortfall
  · rcases baseline with ⟨anchor, _selectedEq⟩
    exact Or.inl <| necessary <|
      ConcreteRealization.priorPublicLink
        setup machine functionIndex input anchor iterationPositive
          priorPublicLinkFails dispatch
  · exact Or.inr shortfall

/-- Independent semantic NIFS premises and an actual broken-dispatch input
produce the dispatch-removal witness, unless the fixed bounded sampler names
one exact shortfall coordinate. -/
theorem dispatch_necessary_or_samplerShortfall_of_semanticPremises
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (premises : HonestNifs.SemanticPremises setup input selected)
    (iterationPositive : 0 < input.iteration)
    (priorSlot : input.priorPc = selected.val + 1)
    (priorPublicLink :
      input.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)))
    (expectedStructure :
      input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey selected)
    (dispatchFails :
      machine.control input.zi input.witness ≠
        Paper.ProgramCounter.ofIndex functionIndex) :
    CheckPlan.NecessaryForSoundness
        (semantics setup machine functionIndex input)
        (target setup machine functionIndex input)
        checks .dispatch ∨
      ConcretePhi81.HonestSamplerShortfall
        (contextAt setup input selected) premises.data := by
  rcases
      exists_sideAnchor_or_samplerShortfall_of_semanticPremises
        setup input selected premises priorSlot expectedStructure with
    baseline | shortfall
  · rcases baseline with ⟨anchor, _selectedEq⟩
    exact Or.inl <| necessary <|
      ConcreteRealization.dispatch
        setup machine functionIndex input anchor iterationPositive
          priorPublicLink dispatchFails
  · exact Or.inr shortfall

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies
