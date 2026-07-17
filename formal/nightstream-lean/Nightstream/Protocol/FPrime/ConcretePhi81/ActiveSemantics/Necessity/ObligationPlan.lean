import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity

/-!
Exact six-family check plan for the independent fixed-active F-prime
obligations.

Owns: the stable protocol-level family names, their review order, one actual
selected-slot/result candidate carrier, the exact proposition owned by each
family, and equivalence of complete plan acceptance with
`ActiveSemantics.Obligations`.

Does not own: Boolean checking, certificates, generic stand-in values,
per-family removal witnesses, inclusion-minimality, Rust, R1CS, costs, or row
removal.

Emits constraints: no.

Authority boundary: `Candidate` contains the real typed slot and complete
ConcretePhi81 `FoldResult`. `semantics` states the six independent
mathematical obligations directly over the verifier-owned setup, machine, and
input. Exactness therefore targets the semantic `Obligations` structure, not
an executable checker or a duplicate model.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.active.iteration` | recursive execution has positive iteration | checked | `Family.iterationPositive` |
| `fprime.active.prior_slot` | raw prior counter selects the typed slot | checked | `Family.priorSlot` |
| `fprime.active.prior_link` | fresh public input equals the exact prior-state image | checked | `Family.priorPublicInput` |
| `fprime.active.structure` | fresh claim uses the verifier-selected structure | checked | `Family.expectedStructure` |
| `fprime.active.nifs` | selected sources yield the complete semantic fold result; the child relation opens into the exact nine-leaf NIFS plan | checked parent | `Family.selectedNifs`, `Result.resultTransition_iff_exists_obligationPlan` |
| `fprime.active.dispatch` | application control selects this fixed function | checked | `Family.dispatch` |
| `fprime.active.obligation_plan.exact` | all six leaves are exactly `Obligations` | exact model theorem | `exact` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan

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

/-- The complete retained active-obligation family. -/
inductive Family where
  | iterationPositive
  | priorSlot
  | priorPublicInput
  | expectedStructure
  | selectedNifs
  | dispatch
  deriving DecidableEq

/-- Stable protocol review order. This is not a physical row order. -/
def checks : List Family :=
  [.iterationPositive, .priorSlot, .priorPublicInput, .expectedStructure,
    .selectedNifs, .dispatch]

/-- The actual semantic witness surface shared by all six families. -/
structure Candidate
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows slotCount : Nat) where
  selected : Fin slotCount
  selectedNext :
    Slot shape publicRingColumns publicFits verifierRows

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

/-- One family owns exactly one field of the independent semantic
`Obligations` structure. -/
def semantics
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) :
    Family ->
      Candidate shape publicRingColumns publicFits verifierRows slotCount ->
        Prop
  | .iterationPositive, _ =>
      0 < input.iteration
  | .priorSlot, candidate =>
      input.priorPc = candidate.selected.val + 1
  | .priorPublicInput, _ =>
      input.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper))
  | .expectedStructure, candidate =>
      input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey candidate.selected
  | .selectedNifs, candidate =>
      FixedActive.ResultTransition
        (contextAt setup input candidate.selected) candidate.selectedNext
  | .dispatch, _ =>
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex

/-- The existing independent semantic target, expressed over one plan
candidate. -/
def target
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
    (candidate :
      Candidate shape publicRingColumns publicFits verifierRows slotCount) :
    Prop :=
  Obligations setup machine functionIndex input candidate.selected
    candidate.selectedNext

/-- Complete plan acceptance is definitionally aligned, field by field, with
the actual six-field semantic obligations. -/
theorem accepts_iff_obligations
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
    (candidate :
      Candidate shape publicRingColumns publicFits verifierRows slotCount) :
    CheckPlan.Accepts
        (semantics setup machine functionIndex input) checks candidate ↔
      target setup machine functionIndex input candidate := by
  constructor
  · intro accepted
    exact {
      iterationPositive :=
        accepted .iterationPositive (by simp [checks])
      priorSlot :=
        accepted .priorSlot (by simp [checks])
      priorPublicInput :=
        accepted .priorPublicInput (by simp [checks])
      expectedStructure :=
        accepted .expectedStructure (by simp [checks])
      selectedNifs :=
        accepted .selectedNifs (by simp [checks])
      dispatch :=
        accepted .dispatch (by simp [checks])
    }
  · intro obligations family _member
    cases family with
    | iterationPositive =>
        exact obligations.iterationPositive
    | priorSlot =>
        exact obligations.priorSlot
    | priorPublicInput =>
        exact obligations.priorPublicInput
    | expectedStructure =>
        exact obligations.expectedStructure
    | selectedNifs =>
        exact obligations.selectedNifs
    | dispatch =>
        exact obligations.dispatch

/-- The six-family plan is exact for the actual independent active
obligations. This theorem makes no necessity claim for any individual leaf. -/
theorem exact
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) :
    CheckPlan.Exact
      (semantics setup machine functionIndex input)
      (target setup machine functionIndex input)
      checks := by
  intro candidate
  exact accepts_iff_obligations setup machine functionIndex input candidate

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan
