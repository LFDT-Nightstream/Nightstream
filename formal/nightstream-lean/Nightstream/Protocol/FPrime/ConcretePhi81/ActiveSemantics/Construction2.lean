import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
import Nightstream.Protocol.FPrime.ConcretePhi81.SelectedNifsSemantics
import Nightstream.Protocol.FPrime.Paper.Construction2

/-!
Outer recursive refinement from ConcretePhi81 to HyperNova Construction 2.

Assurance tier: model-level.

Owns: the exact compatibility contract between verifier-owned ConcretePhi81
setup and one abstract paper NIFS family, projection to the canonical public
ConcretePhi81 edge, and projection of every accepted active step to the paper
recursive branch.

Does not own: physical transcript replay, a reverse/extraction bridge, the old
candidate `PaperNifsTransition`, base semantics, Rust, R1CS, costs, or row
removal.

Emits constraints: no.

Authority boundary: `Refinement.transition` is an explicit theorem premise,
not a definitional alias. The canonical instance discharges that premise by
projecting the independent `FixedActive.ResultTransition` into
`SelectedNifsSemantics.Transition`; it does not ask the active outer relation
to define paper acceptance. The expected relation structure is bound
independently.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.refinement.nifs.structure` | concrete key/slot setup selects the abstract family structure | refinement premise | `Refinement.expectedStructure` |
| `fprime.refinement.nifs.transition` | independent complete result projects to the selected public NIFS edge | refinement premise | `Refinement.transition` |
| `fprime.refinement.nifs.canonical` | active setup projects to the independent public ConcretePhi81 edge | derived | `selectedNifsRefinement` |
| `fprime.refinement.recursive` | all remaining outer checks and computed fields match Construction 2 | model-level theorem | `sound` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.Outer

universe uOuterKey uAppState uWitness uDigest uTranscriptState

/-- Exact two-field contract still required to instantiate the abstract
Construction-2 selected NIFS family with ConcretePhi81. -/
structure Refinement
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup : Setup OuterKey AppState Witness TranscriptState shape
      publicRingColumns publicFits verifierRows slotCount)
    (family : Paper.Construction2.Family OuterKey
      (RelationStructure shape publicRingColumns publicFits)
      (RelationPublicInput shape publicRingColumns publicFits)
      (RelationPoint shape publicRingColumns publicFits)
      Phi81Relation.Evaluation (CommitmentValue verifierRows)
      productionGlobalParams slotCount) : Prop where
  expectedStructure : forall key slot,
    setup.expectedStructure key slot = family.expectedStructure key slot
  transition : forall
      (input : Input OuterKey AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
      (slot : Fin slotCount)
      (result : Slot shape publicRingColumns publicFits verifierRows),
    FixedActive.ResultTransition (contextAt setup input slot) result ->
      family.transition input.verifierKey slot
        (Paper.Construction2.selectedInput input.toPaper slot) result.children

/-- Drop the two outer-input callbacks from the static public-edge setup.
Their values remain internal witnesses of each selected transition. -/
def selectedNifsSetup
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup : Setup OuterKey AppState Witness TranscriptState shape
      publicRingColumns publicFits verifierRows slotCount) :
    SelectedNifsSemantics.Setup OuterKey TranscriptState shape
      publicRingColumns publicFits verifierRows slotCount where
  template := setup.template
  expectedStructure := setup.expectedStructure

/-- The active relation refines the canonical public ConcretePhi81 NIFS edge
without an additional semantic premise. -/
theorem selectedNifsRefinement
    {OuterKey : Type uOuterKey}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (setup : Setup OuterKey AppState Witness TranscriptState shape
      publicRingColumns publicFits verifierRows slotCount) :
    Refinement setup
      (SelectedNifsSemantics.family (selectedNifsSetup setup)) := by
  refine {
    expectedStructure := ?_
    transition := ?_
  }
  · intro key slot
    rfl
  · intro input slot result accepted
    refine ⟨(input.running slot).parent, setup.piCcsInput input slot,
      setup.priorTranscriptState input slot, result.parent, ?_⟩
    simpa [SelectedNifsSemantics.contextOf, selectedNifsSetup, contextAt,
      invocationAt]
      using accepted

/-- Every accepted active ConcretePhi81 execution satisfies the abstract
Construction-2 recursive branch, provided only the explicit selected-NIFS
refinement contract above. -/
theorem sound
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup : Setup OuterKey AppState Witness TranscriptState shape
      publicRingColumns publicFits verifierRows slotCount}
    {family : Paper.Construction2.Family OuterKey
      (RelationStructure shape publicRingColumns publicFits)
      (RelationPublicInput shape publicRingColumns publicFits)
      (RelationPoint shape publicRingColumns publicFits)
      Phi81Relation.Evaluation (CommitmentValue verifierRows)
      productionGlobalParams slotCount}
    {machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount}
    {output : Output Digest AppState shape publicRingColumns publicFits
      verifierRows slotCount}
    (refinement : Refinement setup family)
    (accepted : Holds setup machine functionIndex input output) :
    Paper.Construction2.RecursiveHolds family machine functionIndex
      input.toPaper output.toPaper := by
  rcases accepted with
    ⟨selected, selectedNext, obligations, rfl⟩
  let priorPcValid := obligations.priorPcValid
  have selectedEq : Paper.selectedIndex priorPcValid = selected :=
    obligations.selectedIndex_eq
  refine {
    iterationPositive := obligations.iterationPositive
    priorPcValid := priorPcValid
    priorPublicInput := obligations.priorPublicInput
    application := ?_
    selectedStructures := ?_
    selectedNifs := ?_
    unchanged := ?_
    outputHash := ?_
  }
  · have derived := Paper.derivedOutput_application machine input.toPaper
      (updatedRunning input selected selectedNext).toPaper
    have controlEq :
        machine.control input.toPaper.zi input.toPaper.witness =
          Paper.ProgramCounter.ofIndex functionIndex :=
      obligations.dispatch
    have indexEq :
        (machine.control input.toPaper.zi input.toPaper.witness).index =
          functionIndex := by
      rw [controlEq]
      exact Paper.ProgramCounter.index_ofIndex functionIndex
    rw [indexEq] at derived
    simpa using derived
  · constructor
    · change input.fresh.constraintSystem =
        family.expectedStructure input.verifierKey
          (Paper.selectedIndex priorPcValid)
      rw [selectedEq]
      exact obligations.expectedStructure.trans
        (refinement.expectedStructure input.verifierKey selected)
    · intro child
      have runningStructure :=
        obligations.selectedStructures_eq_expected.1 child
      change ((input.running (Paper.selectedIndex priorPcValid)).children
          child).constraintSystem =
        family.expectedStructure input.verifierKey
          (Paper.selectedIndex priorPcValid)
      rw [selectedEq]
      exact runningStructure.trans
        (refinement.expectedStructure input.verifierKey selected)
  · change family.transition input.verifierKey
      (Paper.selectedIndex priorPcValid)
      (Paper.Construction2.selectedInput input.toPaper
        (Paper.selectedIndex priorPcValid))
      ((outputOf machine input selected selectedNext).toPaper.runningNext
        (Paper.selectedIndex priorPcValid))
    rw [selectedEq]
    have outputSelected :
        (outputOf machine input selected selectedNext).toPaper.runningNext
            selected = selectedNext.children := by
      funext child
      exact congrArg (fun result => result.children child)
        (outputOf_runningNext_selected machine input selected selectedNext)
    rw [outputSelected]
    exact refinement.transition input selected selectedNext
      obligations.selectedNifs
  · intro slot notSelected
    have notConcrete : slot ≠ selected := by
      intro slotEq
      apply notSelected
      exact slotEq.trans selectedEq.symm
    have richEq := outputOf_runningNext_other machine input selected
      selectedNext slot notConcrete
    simpa [Outer.Output.toPaper, Outer.Running.toPaper,
      Outer.Input.toPaper] using
        congrArg (fun result => result.children) richEq
  · exact Paper.derivedOutput_outputHolds machine input.toPaper
      (updatedRunning input selected selectedNext).toPaper

/-- Canonical recursive soundness with the selected public ConcretePhi81 edge
installed directly. Transcript/security refinement remains separate. -/
theorem sound_selectedNifs
    {OuterKey : Type uOuterKey}
    {Digest : Type uDigest}
    {AppState : Type uAppState}
    {Witness : Type uWitness}
    {TranscriptState : Type uTranscriptState}
    {shape : SemanticShape}
    {publicRingColumns verifierRows slotCount : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {setup : Setup OuterKey AppState Witness TranscriptState shape
      publicRingColumns publicFits verifierRows slotCount}
    {machine : Machine OuterKey Digest AppState Witness shape
      publicRingColumns publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    {input : Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount}
    {output : Output Digest AppState shape publicRingColumns publicFits
      verifierRows slotCount}
    (accepted : Holds setup machine functionIndex input output) :
    Paper.Construction2.RecursiveHolds
      (SelectedNifsSemantics.family (selectedNifsSetup setup)) machine
      functionIndex input.toPaper output.toPaper :=
  sound (selectedNifsRefinement setup) accepted

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2
