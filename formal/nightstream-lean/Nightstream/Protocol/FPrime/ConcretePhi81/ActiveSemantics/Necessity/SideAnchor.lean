import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity

/-!
Honest construction and concrete realization of fixed-active necessity side
anchors.

Owns: the non-circular conversion from independent honest NIFS premises plus
the two side equalities into an actual `SideAnchor` or exact sampler
shortfall; context-safe anchor transport; and an actual-type realization
interface for the iteration, prior-link, and dispatch removal countermodels.

Does not own: a closed production fixture, construction of a bad outer input,
proof that setup functions are invariant under outer-field mutation,
executable checking, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: `HonestNifs.SemanticPremises` contains neither an outer
F-prime equation nor a sampler-success assumption. The side-anchor constructor
therefore receives `priorSlot` and `expectedStructure` separately and returns
sampler shortfall explicitly. Neither equality follows from NIFS completeness
or from the current unconstrained `Setup` functions.
`ConcreteRealization` still requires an actual outer view with the removed
equation false and the other two equations true; it does not manufacture a
production counterexample.

| Stage path | Mathematical obligation | Assurance status | Lean owner |
|---|---|---|---|
| `fprime.active.necessity.anchor.outcome` | honest ConcretePhi81 sources construct the selected semantic NIFS result or expose exact sampler shortfall | exhaustive model outcome | `exists_sideAnchor_or_samplerShortfall_of_semanticPremises` |
| `fprime.active.necessity.anchor.honest_nifs` | successful-sampler premises construct the selected semantic NIFS result | compatibility construction | `exists_sideAnchor_of_honestNifs` |
| `fprime.active.necessity.anchor.transport` | reuse a side anchor only across an explicitly equal verifier-built NIFS context | exact model theorem | `StableSideMutation.transport` |
| `fprime.active.necessity.realize` | one actual side anchor plus an actual bad outer view lifts to the weakened active relation | exact model theorem | `ConcreteRealization.lift` |
| `fprime.active.necessity.realize.iteration` | retain link and dispatch while iteration positivity fails | conditional concrete constructor | `ConcreteRealization.activeIteration` |
| `fprime.active.necessity.realize.prior_link` | retain iteration and dispatch while the prior link fails | conditional concrete constructor | `ConcreteRealization.priorPublicLink` |
| `fprime.active.necessity.realize.dispatch` | retain iteration and prior link while dispatch fails | conditional concrete constructor | `ConcreteRealization.dispatch` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity

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
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs

universe uOuterKey uAppState uWitness uDigest uTranscriptState

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

/-- Independent honest NIFS completeness supplies an actual side anchor, or
exposes the exact bounded-sampler coordinate that prevented construction. The
two outer side equalities remain separate premises. -/
theorem exists_sideAnchor_or_samplerShortfall_of_semanticPremises
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (premises : HonestNifs.SemanticPremises setup input selected)
    (priorSlot : input.priorPc = selected.val + 1)
    (expectedStructure :
      input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey selected) :
    (∃ anchor : SideAnchor setup input, anchor.selected = selected) ∨
      ConcretePhi81.HonestSamplerShortfall
        (contextAt setup input selected) premises.data := by
  rcases premises.exists_resultTransition_or_samplerShortfall
      setup input selected with completed | shortfall
  · rcases completed with ⟨certificate, _accepted, transition⟩
    let selectedNext :=
      FixedActive.resultOf (contextAt setup input selected) certificate
    apply Or.inl
    exact ⟨{
      selected := selected
      selectedNext := selectedNext
      side := {
        priorSlot := priorSlot
        expectedStructure := expectedStructure
        selectedNifs := transition
      }
    }, rfl⟩
  · exact Or.inr shortfall

/-- Strong successful-sampler premises supply an actual side anchor without
assuming any of the three outer equations whose necessity is under study.

The equality records that the existential anchor uses the caller's selected
slot, rather than merely some unrelated slot. -/
theorem exists_sideAnchor_of_honestNifs
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (premises : HonestNifs.Premises setup input selected)
    (priorSlot : input.priorPc = selected.val + 1)
    (expectedStructure :
      input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey selected) :
    ∃ anchor : SideAnchor setup input, anchor.selected = selected := by
  rcases premises.exists_resultTransition setup input selected with
    ⟨certificate, _accepted, transition⟩
  let selectedNext :=
    FixedActive.resultOf (contextAt setup input selected) certificate
  refine ⟨{
    selected := selected
    selectedNext := selectedNext
    side := {
      priorSlot := priorSlot
      expectedStructure := expectedStructure
      selectedNifs := transition
    }
  }, rfl⟩

/-- Exact premises for transporting an existing side anchor to a modified
outer input.

The context equality is intentionally strong. Because the current `Setup`
callbacks may inspect the whole input, equality of fresh/running fields alone
does not justify reusing the semantic NIFS transition. -/
structure StableSideMutation
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (source target :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (anchor : SideAnchor setup source) where
  priorSlot : target.priorPc = anchor.selected.val + 1
  expectedStructure :
    target.fresh.constraintSystem =
      setup.expectedStructure target.verifierKey anchor.selected
  contextStable :
    contextAt setup target anchor.selected =
      contextAt setup source anchor.selected

namespace StableSideMutation

/-- Transport keeps the selected slot and semantic NIFS result unchanged, but
only after the verifier-built contexts are proved equal. -/
def transport
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (source target :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (anchor : SideAnchor setup source)
    (stable : StableSideMutation setup source target anchor) :
    SideAnchor setup target where
  selected := anchor.selected
  selectedNext := anchor.selectedNext
  side := {
    priorSlot := stable.priorSlot
    expectedStructure := stable.expectedStructure
    selectedNifs := by
      rw [stable.contextStable]
      exact anchor.side.selectedNifs
  }

end StableSideMutation

/-- One actual ConcretePhi81 outer countermodel realization.

Unlike `OuterPlan.Countermodel`, this carrier uses the real public-input and
program-counter types and contains the actual semantic NIFS side anchor. -/
structure ConcreteRealization
    (removed : OuterPlan.Family)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) where
  anchor : SideAnchor setup input
  weakenedOuter :
    CheckPlan.Accepts OuterPlan.semantics
      (CheckPlan.without OuterPlan.checks removed)
      (viewOf machine functionIndex input)
  rejectedOuter :
    ¬ OuterPlan.Boundary (viewOf machine functionIndex input)

namespace ConcreteRealization

/-- Lift one actual realization into the weakened ConcretePhi81 relation and
rejection by both witness-level obligations and the independent `Holds`
relation. -/
theorem lift
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
    (removed : OuterPlan.Family)
    (realization :
      ConcreteRealization removed setup machine functionIndex input) :
    WeakAccepts removed setup machine functionIndex input
        realization.anchor.selected realization.anchor.selectedNext ∧
      ¬ Obligations setup machine functionIndex input
        realization.anchor.selected realization.anchor.selectedNext ∧
      ¬ Holds setup machine functionIndex input
        (outputOf machine input realization.anchor.selected
          realization.anchor.selectedNext) := by
  let countermodel :
      OuterPlan.Countermodel
        (RelationPublicInput shape publicRingColumns publicFits)
        (Paper.ProgramCounter slotCount) removed := {
    view := viewOf machine functionIndex input
    weakened := realization.weakenedOuter
    rejected := realization.rejectedOuter
  }
  exact
    SideAnchor.liftCountermodel setup machine functionIndex input
      realization.anchor removed countermodel rfl

/-- Actual iteration-removal realization. It retains the prior-link and
dispatch equations and rejects only because positivity fails. -/
def activeIteration
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
    (anchor : SideAnchor setup input)
    (notPositive : ¬ 0 < input.iteration)
    (priorPublicLink :
      input.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)))
    (dispatch :
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex) :
    ConcreteRealization .activeIteration setup machine functionIndex input where
  anchor := anchor
  weakenedOuter := by
    intro family member
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | activeIteration =>
        exact (retained rfl).elim
    | priorPublicLink =>
        exact priorPublicLink
    | dispatch =>
        exact dispatch
  rejectedOuter := by
    intro boundary
    exact notPositive boundary.iterationPositive

/-- Actual prior-link-removal realization. It retains positivity and dispatch
and rejects only because the observed and required public inputs differ. -/
def priorPublicLink
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
    (anchor : SideAnchor setup input)
    (iterationPositive : 0 < input.iteration)
    (priorPublicLinkFails :
      input.fresh.publicInput ≠
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)))
    (dispatch :
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex) :
    ConcreteRealization .priorPublicLink setup machine functionIndex input where
  anchor := anchor
  weakenedOuter := by
    intro family member
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | activeIteration =>
        exact iterationPositive
    | priorPublicLink =>
        exact (retained rfl).elim
    | dispatch =>
        exact dispatch
  rejectedOuter := by
    intro boundary
    exact priorPublicLinkFails boundary.priorPublicLink

/-- Actual dispatch-removal realization. It retains positivity and the exact
prior link and rejects only because control chooses a different counter. -/
def dispatch
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
    (anchor : SideAnchor setup input)
    (iterationPositive : 0 < input.iteration)
    (priorPublicLink :
      input.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)))
    (dispatchFails :
      machine.control input.zi input.witness ≠
        Paper.ProgramCounter.ofIndex functionIndex) :
    ConcreteRealization .dispatch setup machine functionIndex input where
  anchor := anchor
  weakenedOuter := by
    intro family member
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | activeIteration =>
        exact iterationPositive
    | priorPublicLink =>
        exact priorPublicLink
    | dispatch =>
        exact (retained rfl).elim
  rejectedOuter := by
    intro boundary
    exact dispatchFails boundary.dispatch

end ConcreteRealization

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity
