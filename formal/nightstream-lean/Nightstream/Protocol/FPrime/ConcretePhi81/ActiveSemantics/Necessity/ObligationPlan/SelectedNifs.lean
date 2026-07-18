import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan

/-!
Actual-type removal witness for the selected-NIFS obligation.

Owns: one deterministic adversarial mutation of an otherwise valid
ConcretePhi81 fold result, preservation of the other five active obligations,
rejection of the mutated result by semantic NIFS authority, and the resulting
inclusion-necessity theorem for `Family.selectedNifs`.

Does not own: a production fixture, physical certificate acceptance,
construction of the five outer obligations, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: `Realization.accepted` is the independently stated valid
baseline. `forgedNext` changes only the derived parent norm stage and preserves
the checked child accumulator. The forged cache is adversarial data, never
authority. `ResultTransition.parent_eq_of_children_eq` proves that no semantic
transition can accept that mutation. The semantic-premise theorem obtains the
baseline through the honest prover or preserves one exact sampler shortfall;
it never assumes a total rejection sampler. The stronger compatibility
theorem remains for callers that already prove sampler success.

| Stage path | Mathematical obligation | Assurance status | Lean owner |
|---|---|---|---|
| `fprime.active.necessity.selected_nifs.mutation` | change only the derived parent norm stage while preserving all children | exact actual-type construction | `Realization.forgedNext` |
| `fprime.active.necessity.selected_nifs.rejection` | unchanged checked children determine the parent, so the changed parent cannot be a transition | exact model theorem | `Realization.forged_not_transition` |
| `fprime.active.necessity.selected_nifs.preservation` | all five non-NIFS active obligations survive the mutation | exact model theorem | `Realization.weakened` |
| `fprime.active.necessity.selected_nifs.necessary` | removing `selectedNifs` admits the mutated non-target candidate | conditional actual-type necessity | `Realization.necessary` |
| `fprime.active.necessity.selected_nifs.outcome` | independent honest NIFS construction supplies the valid baseline or one exact sampler shortfall | exhaustive model outcome | `necessary_or_samplerShortfall_of_semanticPremises` |
| `fprime.active.necessity.selected_nifs.honest` | successful-sampler premises supply the valid baseline used by the mutation | compatibility construction | `necessary_of_honestNifs` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs

universe uOuterKey uAppState uWitness uDigest uTranscriptState

/-- Pick a visibly different verifier-owned norm stage. -/
def differentStage : NormStage -> NormStage
  | .fresh => .combined
  | .combined => .fresh
  | .ambient => .fresh

theorem differentStage_ne (stage : NormStage) :
    differentStage stage ≠ stage := by
  cases stage <;> simp [differentStage]

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

/-- One valid actual active candidate from which the selected-NIFS-only
mutation is constructed. -/
structure Realization
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
  selected : Fin slotCount
  acceptedNext :
    Slot shape publicRingColumns publicFits verifierRows
  accepted :
    Obligations setup machine functionIndex input selected acceptedNext

namespace Realization

variable {setup :
  Setup OuterKey AppState Witness TranscriptState shape
    publicRingColumns publicFits verifierRows slotCount}
variable {machine :
  Machine OuterKey Digest AppState Witness shape publicRingColumns
    publicFits verifierRows slotCount}
variable {functionIndex : Fin slotCount}
variable {input :
  Input OuterKey AppState Witness shape publicRingColumns publicFits
    verifierRows slotCount}

/-- The adversarial parent differs in one explicit semantic field. -/
def forgedParent
    (realization : Realization setup machine functionIndex input) :=
  { realization.acceptedNext.parent with
    stage := differentStage realization.acceptedNext.parent.stage }

/-- Mutate only the derived parent cache; preserve the complete checked child
accumulator definitionally. -/
def forgedNext
    (realization : Realization setup machine functionIndex input) :
    Slot shape publicRingColumns publicFits verifierRows := {
  parent := realization.forgedParent
  children := realization.acceptedNext.children
}

/-- The actual six-family candidate containing the forged result. -/
def candidate
    (realization : Realization setup machine functionIndex input) :
    Candidate shape publicRingColumns publicFits verifierRows slotCount := {
  selected := realization.selected
  selectedNext := realization.forgedNext
}

theorem forgedParent_ne
    (realization : Realization setup machine functionIndex input) :
    realization.forgedParent ≠ realization.acceptedNext.parent := by
  intro equal
  have stageEqual :=
    congrArg (fun parent => parent.stage) equal
  exact differentStage_ne realization.acceptedNext.parent.stage (by
    simpa [forgedParent] using stageEqual)

/-- Parent uniqueness from the unchanged checked children rejects the forged
result independently of any executable certificate checker. -/
theorem forged_not_transition
    (realization : Realization setup machine functionIndex input) :
    ¬ FixedActive.ResultTransition
        (contextAt setup input realization.selected)
        realization.forgedNext := by
  intro forgedAccepted
  have parentEqual :=
    realization.accepted.selectedNifs.parent_eq_of_children_eq
      forgedAccepted (by rfl)
  exact realization.forgedParent_ne (by
    simpa [forgedNext] using parentEqual.symm)

/-- The weakened plan accepts: the mutation changes neither the selected slot
nor any input-owned field, and the sole failed family has been removed. -/
theorem weakened
    (realization : Realization setup machine functionIndex input) :
    CheckPlan.Accepts
      (semantics setup machine functionIndex input)
      (CheckPlan.without checks .selectedNifs)
      realization.candidate := by
  intro family member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  cases family with
  | iterationPositive =>
      exact realization.accepted.iterationPositive
  | priorSlot =>
      exact realization.accepted.priorSlot
  | priorPublicInput =>
      exact realization.accepted.priorPublicInput
  | expectedStructure =>
      exact realization.accepted.expectedStructure
  | selectedNifs =>
      exact (retained rfl).elim
  | dispatch =>
      exact realization.accepted.dispatch

/-- The forged candidate is outside the independent six-obligation target
specifically because its selected NIFS result is not semantic. -/
theorem rejected
    (realization : Realization setup machine functionIndex input) :
    ¬ target setup machine functionIndex input realization.candidate := by
  intro obligations
  exact realization.forged_not_transition obligations.selectedNifs

/-- Actual-type inclusion-necessity of `selectedNifs`, relative to the exact
six-family active-obligation plan and one valid baseline transition. -/
theorem necessary
    (realization : Realization setup machine functionIndex input) :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex input)
      (target setup machine functionIndex input)
      checks .selectedNifs :=
  ⟨realization.candidate, realization.weakened, realization.rejected⟩

end Realization

/-- Independent honest NIFS premises plus the five non-NIFS outer equations
either supply the valid actual-type baseline required by `Realization`, or
name one exact bounded-sampler shortfall coordinate. -/
theorem exists_or_samplerShortfall_of_semanticPremises
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
    (priorPublicInput :
      input.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)))
    (expectedStructure :
      input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey selected)
    (dispatch :
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex) :
    (∃ realization : Realization setup machine functionIndex input,
      realization.selected = selected) ∨
      ConcretePhi81.HonestSamplerShortfall
        (contextAt setup input selected) premises.data := by
  rcases
      premises.exists_resultTransition_or_samplerShortfall
        setup input selected with completed | shortfall
  · rcases completed with
      ⟨certificate, _physicalAccepted, transition⟩
    exact Or.inl ⟨{
      selected := selected
      acceptedNext :=
        FixedActive.resultOf (contextAt setup input selected) certificate
      accepted := {
        iterationPositive := iterationPositive
        priorSlot := priorSlot
        priorPublicInput := priorPublicInput
        expectedStructure := expectedStructure
        selectedNifs := transition
        dispatch := dispatch
      }
    }, rfl⟩
  · exact Or.inr shortfall

/-- Inclusion-necessity of the selected-NIFS obligation from independent
semantic premises, with bounded-sampler shortfall retained as the only other
model-level outcome. -/
theorem necessary_or_samplerShortfall_of_semanticPremises
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
    (priorPublicInput :
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
        checks .selectedNifs ∨
      ConcretePhi81.HonestSamplerShortfall
        (contextAt setup input selected) premises.data := by
  rcases
      exists_or_samplerShortfall_of_semanticPremises
        setup machine functionIndex input selected premises iterationPositive
          priorSlot priorPublicInput expectedStructure dispatch with
    baseline | shortfall
  · rcases baseline with ⟨realization, _selectedEq⟩
    exact Or.inl realization.necessary
  · exact Or.inr shortfall

/-- Strong successful-sampler premises plus the five non-NIFS outer equations
supply the valid baseline required by `Realization`. -/
theorem exists_of_honestNifs
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
    (premises : HonestNifs.Premises setup input selected)
    (iterationPositive : 0 < input.iteration)
    (priorSlot : input.priorPc = selected.val + 1)
    (priorPublicInput :
      input.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)))
    (expectedStructure :
      input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey selected)
    (dispatch :
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex) :
    ∃ realization : Realization setup machine functionIndex input,
      realization.selected = selected := by
  rcases premises.exists_resultTransition setup input selected with
    ⟨certificate, _physicalAccepted, transition⟩
  exact ⟨{
    selected := selected
    acceptedNext :=
      FixedActive.resultOf (contextAt setup input selected) certificate
    accepted := {
      iterationPositive := iterationPositive
      priorSlot := priorSlot
      priorPublicInput := priorPublicInput
      expectedStructure := expectedStructure
      selectedNifs := transition
      dispatch := dispatch
    }
  }, rfl⟩

/-- Compatibility selected-NIFS necessity result: from strong
successful-sampler premises and the other five active equations, removing
`selectedNifs` is unsound over the actual typed candidate carrier. -/
theorem necessary_of_honestNifs
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
    (premises : HonestNifs.Premises setup input selected)
    (iterationPositive : 0 < input.iteration)
    (priorSlot : input.priorPc = selected.val + 1)
    (priorPublicInput :
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
      checks .selectedNifs := by
  rcases
      exists_of_honestNifs setup machine functionIndex input selected premises
        iterationPositive priorSlot priorPublicInput expectedStructure dispatch with
    ⟨realization, _selectedEq⟩
  exact realization.necessary

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs
