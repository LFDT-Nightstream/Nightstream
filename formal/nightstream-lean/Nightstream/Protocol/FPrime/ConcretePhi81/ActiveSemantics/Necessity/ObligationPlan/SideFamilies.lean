import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan

/-!
Actual-type conditional removal witnesses for the prior-slot and selected
structure families.

Owns: lifting one complete ConcretePhi81 NIFS result plus the five retained
active obligations into `NecessaryForSoundness` for `priorSlot` or
`expectedStructure`; exhaustive semantic-premise outcome constructors; and
strong successful-sampler compatibility constructors.

Does not own: a production input where either side equality fails while all
retained obligations hold, mutation stability for verifier callbacks, an
executable checker, Rust, R1CS, costs, inclusion-minimality, or row removal.

Emits constraints: no.

Authority boundary: the candidate is the real selected `Fin` slot and complete
ConcretePhi81 `FoldResult`. `HonestNifs.SemanticPremises` constructs the NIFS
result or returns one exact sampler shortfall, without assuming an outer
F-prime equation. The failing side equality and all retained outer equations
remain explicit model premises, so these theorems do not claim a closed
production fixture.

| Stage path | Mathematical obligation | Assurance status | Lean owner |
|---|---|---|---|
| `fprime.active.necessity.prior_slot.transition` | omitting raw-counter/slot equality admits an actual typed candidate when the other five obligations hold | conditional model theorem | `priorSlot_necessary_of_transition` |
| `fprime.active.necessity.prior_slot.outcome` | independent honest NIFS constructs the complete result or one exact sampler shortfall | exhaustive model outcome | `priorSlot_necessary_or_samplerShortfall_of_semanticPremises` |
| `fprime.active.necessity.prior_slot.honest_nifs` | successful-sampler premises construct the complete result needed by the prior-slot removal witness | compatibility theorem | `priorSlot_necessary_of_honestNifs` |
| `fprime.active.necessity.structure.transition` | omitting selected-structure equality admits an actual typed candidate when the other five obligations hold | conditional model theorem | `expectedStructure_necessary_of_transition` |
| `fprime.active.necessity.structure.outcome` | independent honest NIFS constructs the complete result or one exact sampler shortfall | exhaustive model outcome | `expectedStructure_necessary_or_samplerShortfall_of_semanticPremises` |
| `fprime.active.necessity.structure.honest_nifs` | successful-sampler premises construct the complete result needed by the structure removal witness | compatibility theorem | `expectedStructure_necessary_of_honestNifs` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies

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
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan

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

/-- Conditional actual-type necessity of the raw prior-counter/slot equality.

The theorem consumes a real complete NIFS result and states every retained
family directly. It does not construct an input with the requested mismatch. -/
theorem priorSlot_necessary_of_transition
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
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows)
    (iterationPositive : 0 < input.iteration)
    (priorSlotFails : input.priorPc ≠ selected.val + 1)
    (priorPublicInput :
      input.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)))
    (expectedStructure :
      input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey selected)
    (selectedNifs :
      FixedActive.ResultTransition
        (contextAt setup input selected) selectedNext)
    (dispatch :
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex) :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex input)
      (target setup machine functionIndex input)
      checks .priorSlot := by
  let candidate : Candidate shape publicRingColumns publicFits verifierRows slotCount := {
    selected := selected
    selectedNext := selectedNext
  }
  refine ⟨candidate, ?_, ?_⟩
  · intro family member
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | iterationPositive =>
        exact iterationPositive
    | priorSlot =>
        exact (retained rfl).elim
    | priorPublicInput =>
        exact priorPublicInput
    | expectedStructure =>
        exact expectedStructure
    | selectedNifs =>
        exact selectedNifs
    | dispatch =>
        exact dispatch
  · intro obligations
    exact priorSlotFails obligations.priorSlot

/-- Independent semantic NIFS premises either supply the complete fold result
needed by the prior-slot removal witness, or name one exact bounded-sampler
shortfall coordinate. -/
theorem priorSlot_necessary_or_samplerShortfall_of_semanticPremises
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
    (priorSlotFails : input.priorPc ≠ selected.val + 1)
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
        checks .priorSlot ∨
      ConcretePhi81.HonestSamplerShortfall
        (contextAt setup input selected) premises.data := by
  rcases
      premises.exists_resultTransition_or_samplerShortfall
        setup input selected with completed | shortfall
  · rcases completed with ⟨certificate, _accepted, transition⟩
    apply Or.inl
    exact
      priorSlot_necessary_of_transition
        setup machine functionIndex input selected
        (FixedActive.resultOf (contextAt setup input selected) certificate)
        iterationPositive priorSlotFails priorPublicInput expectedStructure
        transition dispatch
  · exact Or.inr shortfall

/-- Strong successful-sampler premises supply the complete typed fold result
for the conditional prior-slot removal witness.

The wrong prior slot and all retained outer obligations remain caller
premises; therefore this is not a closed production counterexample. -/
theorem priorSlot_necessary_of_honestNifs
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
    (priorSlotFails : input.priorPc ≠ selected.val + 1)
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
      checks .priorSlot := by
  rcases premises.exists_resultTransition setup input selected with
    ⟨certificate, _accepted, transition⟩
  exact
    priorSlot_necessary_of_transition
      setup machine functionIndex input selected
      (FixedActive.resultOf (contextAt setup input selected) certificate)
      iterationPositive priorSlotFails priorPublicInput expectedStructure
      transition dispatch

/-- Conditional actual-type necessity of the verifier-selected fresh
structure equality.

The theorem consumes a real complete NIFS result and states every retained
family directly. It does not construct an input with the requested mismatch. -/
theorem expectedStructure_necessary_of_transition
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
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows)
    (iterationPositive : 0 < input.iteration)
    (priorSlot : input.priorPc = selected.val + 1)
    (priorPublicInput :
      input.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)))
    (expectedStructureFails :
      input.fresh.constraintSystem ≠
        setup.expectedStructure input.verifierKey selected)
    (selectedNifs :
      FixedActive.ResultTransition
        (contextAt setup input selected) selectedNext)
    (dispatch :
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex) :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex input)
      (target setup machine functionIndex input)
      checks .expectedStructure := by
  let candidate : Candidate shape publicRingColumns publicFits verifierRows slotCount := {
    selected := selected
    selectedNext := selectedNext
  }
  refine ⟨candidate, ?_, ?_⟩
  · intro family member
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | iterationPositive =>
        exact iterationPositive
    | priorSlot =>
        exact priorSlot
    | priorPublicInput =>
        exact priorPublicInput
    | expectedStructure =>
        exact (retained rfl).elim
    | selectedNifs =>
        exact selectedNifs
    | dispatch =>
        exact dispatch
  · intro obligations
    exact expectedStructureFails obligations.expectedStructure

/-- Independent semantic NIFS premises either supply the complete fold result
needed by the selected-structure removal witness, or name one exact bounded-
sampler shortfall coordinate. -/
theorem expectedStructure_necessary_or_samplerShortfall_of_semanticPremises
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
    (expectedStructureFails :
      input.fresh.constraintSystem ≠
        setup.expectedStructure input.verifierKey selected)
    (dispatch :
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex) :
    CheckPlan.NecessaryForSoundness
        (semantics setup machine functionIndex input)
        (target setup machine functionIndex input)
        checks .expectedStructure ∨
      ConcretePhi81.HonestSamplerShortfall
        (contextAt setup input selected) premises.data := by
  rcases
      premises.exists_resultTransition_or_samplerShortfall
        setup input selected with completed | shortfall
  · rcases completed with ⟨certificate, _accepted, transition⟩
    apply Or.inl
    exact
      expectedStructure_necessary_of_transition
        setup machine functionIndex input selected
        (FixedActive.resultOf (contextAt setup input selected) certificate)
        iterationPositive priorSlot priorPublicInput expectedStructureFails
        transition dispatch
  · exact Or.inr shortfall

/-- Strong successful-sampler premises supply the complete typed fold result
for the conditional selected-structure removal witness.

The wrong structure and all retained outer obligations remain caller premises;
therefore this is not a closed production counterexample. -/
theorem expectedStructure_necessary_of_honestNifs
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
    (expectedStructureFails :
      input.fresh.constraintSystem ≠
        setup.expectedStructure input.verifierKey selected)
    (dispatch :
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex) :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex input)
      (target setup machine functionIndex input)
      checks .expectedStructure := by
  rcases premises.exists_resultTransition setup input selected with
    ⟨certificate, _accepted, transition⟩
  exact
    expectedStructure_necessary_of_transition
      setup machine functionIndex input selected
      (FixedActive.resultOf (contextAt setup input selected) certificate)
      iterationPositive priorSlot priorPublicInput expectedStructureFails
      transition dispatch

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies
