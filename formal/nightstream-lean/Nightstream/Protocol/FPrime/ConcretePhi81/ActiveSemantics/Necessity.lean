import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
import Nightstream.SuperNeo.CheckPlan

/-!
Generic outer-check countermodels and their explicit lift into the independent
fixed-active ConcretePhi81 semantics.

Owns: a factorization of the six active obligations into the three
ConcretePhi81 side conditions and three generic outer equations; an exact
three-family check plan for those outer equations; closed model-level
countermodels for iteration, prior-link, and dispatch; and a lifting theorem
that requires an explicit actual `SideAnchor`.

Does not own: construction of an honest ConcretePhi81 NIFS transition, an
actual production `SideAnchor`, executable checking, Rust, R1CS, costs, row
removal, or a claim that the production relation is already
inclusion-minimal.

Emits constraints: no.

Authority boundary: the closed Boolean countermodels prove only generic,
model-level necessity of the three outer equations. They do not instantiate
the ConcretePhi81 relation. A concrete necessity result must separately
provide both an actual `SideAnchor` and a realization equality connecting the
active outer view to one generic countermodel. The lift theorem deliberately
cannot manufacture either premise.

| Stage path | Mathematical obligation | Assurance status | Lean owner |
|---|---|---|---|
| `fprime.active.necessity.factor` | split actual obligations into ConcretePhi81 side conditions and generic outer equations | exact model factorization | `obligations_iff_side_and_outer` |
| `fprime.active.necessity.relation` | expose the same split below the independent existential `Holds` target | exact model factorization | `holds_iff_exists_side_and_outer` |
| `fprime.active.necessity.outer_plan` | three named checks accept exactly the generic outer boundary | generic/model-level | `OuterPlan.exact` |
| `fprime.active.necessity.iteration` | omitting `i > 0` admits an iteration-zero outer view | generic/model-level only | `OuterPlan.activeIteration_countermodel` |
| `fprime.active.necessity.prior_link` | omitting the exact prior link admits unequal public inputs | generic/model-level only | `OuterPlan.priorPublicLink_countermodel` |
| `fprime.active.necessity.dispatch` | omitting fixed-function dispatch admits unequal counters | generic/model-level only | `OuterPlan.dispatch_countermodel` |
| `fprime.active.necessity.lift` | lift a generic countermodel only through an explicit actual side anchor and realization | conditional concrete interface | `SideAnchor.liftCountermodel` |
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
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

namespace OuterPlan

universe uPublicInput uCounter

/-- The three outer values and their independently computed requirements.

The verifier-specific computations remain outside this generic carrier. An
actual active invocation enters through `viewOf`. -/
structure View
    (PublicInput : Type uPublicInput)
    (Counter : Type uCounter) where
  iteration : Nat
  observedPriorPublic : PublicInput
  requiredPriorPublic : PublicInput
  observedControl : Counter
  requiredControl : Counter

/-- Independent outer boundary projected from HyperNova Construction 2. -/
structure Boundary
    {PublicInput : Type uPublicInput}
    {Counter : Type uCounter}
    (view : View PublicInput Counter) : Prop where
  iterationPositive : 0 < view.iteration
  priorPublicLink : view.observedPriorPublic = view.requiredPriorPublic
  dispatch : view.observedControl = view.requiredControl

/-- The first three outer families selected for necessity analysis. -/
inductive Family where
  | activeIteration
  | priorPublicLink
  | dispatch
  deriving DecidableEq

/-- Stable review order for the outer plan. -/
def checks : List Family :=
  [.activeIteration, .priorPublicLink, .dispatch]

/-- One family owns exactly one independent outer equation. -/
def semantics
    {PublicInput : Type uPublicInput}
    {Counter : Type uCounter} :
    Family -> View PublicInput Counter -> Prop
  | .activeIteration, view => 0 < view.iteration
  | .priorPublicLink, view =>
      view.observedPriorPublic = view.requiredPriorPublic
  | .dispatch, view => view.observedControl = view.requiredControl

/-- The three-family plan is exact for the generic outer boundary. -/
theorem accepts_iff_boundary
    {PublicInput : Type uPublicInput}
    {Counter : Type uCounter}
    (view : View PublicInput Counter) :
    CheckPlan.Accepts semantics checks view ↔ Boundary view := by
  constructor
  · intro accepted
    exact {
      iterationPositive :=
        accepted .activeIteration (by simp [checks])
      priorPublicLink :=
        accepted .priorPublicLink (by simp [checks])
      dispatch :=
        accepted .dispatch (by simp [checks])
    }
  · intro boundary family _member
    cases family with
    | activeIteration =>
        exact boundary.iterationPositive
    | priorPublicLink =>
        exact boundary.priorPublicLink
    | dispatch =>
        exact boundary.dispatch

/-- Exactness is relative to the independently named generic boundary, not to
the production ConcretePhi81 relation. -/
theorem exact
    {PublicInput : Type uPublicInput}
    {Counter : Type uCounter} :
    CheckPlan.Exact
      (semantics :
        Family -> View PublicInput Counter -> Prop)
      Boundary checks := by
  intro view
  exact accepts_iff_boundary view

/-- One closed generic removal countermodel. -/
structure Countermodel
    (PublicInput : Type uPublicInput)
    (Counter : Type uCounter)
    (removed : Family) where
  view : View PublicInput Counter
  weakened :
    CheckPlan.Accepts semantics (CheckPlan.without checks removed) view
  rejected : ¬ Boundary view

/-- Iteration-zero view with both retained equations satisfied. -/
def activeIteration_countermodel : Countermodel Bool Bool .activeIteration where
  view := {
    iteration := 0
    observedPriorPublic := false
    requiredPriorPublic := false
    observedControl := false
    requiredControl := false
  }
  weakened := by
    intro family member
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | activeIteration =>
        exact (retained rfl).elim
    | priorPublicLink =>
        rfl
    | dispatch =>
        rfl
  rejected := by
    intro boundary
    simpa using boundary.iterationPositive

/-- Positive-iteration view whose observed and required prior public inputs
differ while dispatch remains correct. -/
def priorPublicLink_countermodel :
    Countermodel Bool Bool .priorPublicLink where
  view := {
    iteration := 1
    observedPriorPublic := false
    requiredPriorPublic := true
    observedControl := false
    requiredControl := false
  }
  weakened := by
    intro family member
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | activeIteration =>
        exact Nat.zero_lt_succ 0
    | priorPublicLink =>
        exact (retained rfl).elim
    | dispatch =>
        rfl
  rejected := by
    intro boundary
    cases boundary.priorPublicLink

/-- Positive, correctly linked view whose control result differs from the
fixed-function counter. -/
def dispatch_countermodel : Countermodel Bool Bool .dispatch where
  view := {
    iteration := 1
    observedPriorPublic := false
    requiredPriorPublic := false
    observedControl := false
    requiredControl := true
  }
  weakened := by
    intro family member
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | activeIteration =>
        exact Nat.zero_lt_succ 0
    | priorPublicLink =>
        rfl
    | dispatch =>
        exact (retained rfl).elim
  rejected := by
    intro boundary
    cases boundary.dispatch

/-- Generic inclusion-necessity for the iteration family. -/
theorem activeIteration_necessary :
    CheckPlan.NecessaryForSoundness
      (semantics : Family -> View Bool Bool -> Prop)
      Boundary checks .activeIteration :=
  ⟨activeIteration_countermodel.view,
    activeIteration_countermodel.weakened,
    activeIteration_countermodel.rejected⟩

/-- Generic inclusion-necessity for the prior-public-link family. -/
theorem priorPublicLink_necessary :
    CheckPlan.NecessaryForSoundness
      (semantics : Family -> View Bool Bool -> Prop)
      Boundary checks .priorPublicLink :=
  ⟨priorPublicLink_countermodel.view,
    priorPublicLink_countermodel.weakened,
    priorPublicLink_countermodel.rejected⟩

/-- Generic inclusion-necessity for the dispatch family. -/
theorem dispatch_necessary :
    CheckPlan.NecessaryForSoundness
      (semantics : Family -> View Bool Bool -> Prop)
      Boundary checks .dispatch :=
  ⟨dispatch_countermodel.view,
    dispatch_countermodel.weakened,
    dispatch_countermodel.rejected⟩

end OuterPlan

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

/-- The three ConcretePhi81 conditions deliberately absent from the generic
outer countermodels. -/
structure SideConditions
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows) : Prop where
  priorSlot : input.priorPc = selected.val + 1
  expectedStructure :
    input.fresh.constraintSystem =
      setup.expectedStructure input.verifierKey selected
  selectedNifs :
    FixedActive.ResultTransition
      (contextAt setup input selected) selectedNext

/-- Compute the generic outer view from one actual active invocation. -/
def viewOf
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) :
    OuterPlan.View
      (RelationPublicInput shape publicRingColumns publicFits)
      (Paper.ProgramCounter slotCount) where
  iteration := input.iteration
  observedPriorPublic := input.fresh.publicInput
  requiredPriorPublic :=
    machine.encodeInstance
      (machine.hash (Paper.priorHashPreimage input.toPaper))
  observedControl := machine.control input.zi input.witness
  requiredControl := Paper.ProgramCounter.ofIndex functionIndex

/-- Exact factorization of the independent six-field active obligations. -/
theorem obligations_iff_side_and_outer
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
    (selected : Fin slotCount)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows) :
    Obligations setup machine functionIndex input selected selectedNext ↔
      SideConditions setup input selected selectedNext ∧
        OuterPlan.Boundary (viewOf machine functionIndex input) := by
  constructor
  · intro obligations
    exact ⟨{
      priorSlot := obligations.priorSlot
      expectedStructure := obligations.expectedStructure
      selectedNifs := obligations.selectedNifs
    }, {
      iterationPositive := obligations.iterationPositive
      priorPublicLink := obligations.priorPublicInput
      dispatch := obligations.dispatch
    }⟩
  · rintro ⟨side, outer⟩
    exact {
      iterationPositive := outer.iterationPositive
      priorSlot := side.priorSlot
      priorPublicInput := outer.priorPublicLink
      expectedStructure := side.expectedStructure
      selectedNifs := side.selectedNifs
      dispatch := outer.dispatch
    }

/-- The same exact factorization below the independent existential `Holds`
relation and its canonical output equation. -/
theorem holds_iff_exists_side_and_outer
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
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount) :
    Holds setup machine functionIndex input output ↔
      ∃ selected : Fin slotCount,
        ∃ selectedNext :
            Slot shape publicRingColumns publicFits verifierRows,
          SideConditions setup input selected selectedNext ∧
            OuterPlan.Boundary (viewOf machine functionIndex input) ∧
              output = outputOf machine input selected selectedNext := by
  constructor
  · rintro ⟨selected, selectedNext, obligations, outputEq⟩
    rcases
        (obligations_iff_side_and_outer
          setup machine functionIndex input selected selectedNext).1
          obligations with
      ⟨side, outer⟩
    exact ⟨selected, selectedNext, side, outer, outputEq⟩
  · rintro ⟨selected, selectedNext, side, outer, outputEq⟩
    exact ⟨selected, selectedNext,
      (obligations_iff_side_and_outer
        setup machine functionIndex input selected selectedNext).2
        ⟨side, outer⟩,
      outputEq⟩

/-- Actual weakened acceptance for one of the three generic outer families,
while retaining all ConcretePhi81 side conditions. -/
def WeakAccepts
    (removed : OuterPlan.Family)
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
    (selected : Fin slotCount)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows) : Prop :=
  SideConditions setup input selected selectedNext ∧
    CheckPlan.Accepts OuterPlan.semantics
      (CheckPlan.without OuterPlan.checks removed)
      (viewOf machine functionIndex input)

/-- Explicit actual side witness required before a generic outer countermodel
may say anything about ConcretePhi81 semantics. -/
structure SideAnchor
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) where
  selected : Fin slotCount
  selectedNext :
    Slot shape publicRingColumns publicFits verifierRows
  side : SideConditions setup input selected selectedNext

namespace SideAnchor

/-- Conditional lifting only: an actual side anchor and an exact realization
of the generic bad view yield weakened acceptance and rejection by both the
actual witness-level obligations and the independent `Holds` relation.

This theorem does not construct an anchor or a production countermodel. -/
theorem liftCountermodel
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
    (anchor : SideAnchor setup input)
    (removed : OuterPlan.Family)
    (countermodel :
      OuterPlan.Countermodel
        (RelationPublicInput shape publicRingColumns publicFits)
        (Paper.ProgramCounter slotCount) removed)
    (realizes :
      viewOf machine functionIndex input = countermodel.view) :
    WeakAccepts removed setup machine functionIndex input
        anchor.selected anchor.selectedNext ∧
      ¬ Obligations setup machine functionIndex input
        anchor.selected anchor.selectedNext ∧
      ¬ Holds setup machine functionIndex input
        (outputOf machine input anchor.selected anchor.selectedNext) := by
  have weakened :
      CheckPlan.Accepts OuterPlan.semantics
        (CheckPlan.without OuterPlan.checks removed)
        (viewOf machine functionIndex input) := by
    simpa only [realizes] using countermodel.weakened
  have rejected :
      ¬ OuterPlan.Boundary (viewOf machine functionIndex input) := by
    intro boundary
    apply countermodel.rejected
    simpa only [realizes] using boundary
  refine ⟨⟨anchor.side, weakened⟩, ?_, ?_⟩
  · intro obligations
    exact rejected
      ((obligations_iff_side_and_outer
        setup machine functionIndex input anchor.selected
          anchor.selectedNext).1 obligations).2
  · intro holds
    rcases
        (holds_iff_exists_side_and_outer
          setup machine functionIndex input
            (outputOf machine input anchor.selected anchor.selectedNext)).1
          holds with
      ⟨selected, selectedNext, side, outer, outputEq⟩
    exact rejected outer

end SideAnchor

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity
