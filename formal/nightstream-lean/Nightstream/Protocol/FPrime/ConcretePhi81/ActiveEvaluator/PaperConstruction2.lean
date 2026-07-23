import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2
import Nightstream.Protocol.FPrime.ConcretePhi81.PaperSelectedNifsSemantics

/-!
Physical outer execution composed with a paper-exact selected NIFS transition.

Assurance tier: model-level.

Owns: the direct projection from successful active outer execution plus one
exact fixed-active paper transition to HyperNova Construction 2.

Does not own: derivation of the paper transition, child openings,
deterministic child equality, delayed packed authority, security-event
bounds, Rust, R1CS, costs, or row removal.

Authority boundary: `paperTransition` is the complete selected-NIFS semantic
premise. The executable outer relation contributes only its checked counter,
prior link, structure, dispatch, and canonical output equalities. No stronger
`FixedActive.ResultTransition` is reconstructed.

Emits constraints: no.
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
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows slotCount : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Successful outer execution plus the exact paper transition for the
certificate-derived child family satisfies the recursive Construction-2
branch. The selected parent cache is used only to interpret the certificate
and is erased from the conclusion. -/
theorem run_refinesConstruction2_of_paperTransition
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (certificate : Certificate setup input)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount)
    (paperTransition :
      FixedActive.PaperProfile.Transition
        (FixedActive.paperProfileOf
          (contextAt setup input certificate.selected))
        (contextAt setup input certificate.selected).input
        (outputChildren (contextAt setup input certificate.selected)
          certificate.nifs))
    (executed : run checkers input certificate = some output) :
    Paper.Construction2.RecursiveHolds
      (PaperSelectedNifsSemantics.family
        (ActiveSemantics.Construction2.selectedNifsSetup setup))
      machine functionIndex input.toPaper output.toPaper := by
  rcases
      (run_eq_some_iff_physicalChecks checkers input certificate output).1
        executed with
    ⟨selectedNext, physical⟩
  let selected := certificate.selected
  have selectedPaperTransition :
      PaperSelectedNifsSemantics.Transition
        (ActiveSemantics.Construction2.selectedNifsSetup setup)
        input.verifierKey selected
        (Paper.Construction2.selectedInput input.toPaper selected)
        (outputChildren (contextAt setup input selected) certificate.nifs) := by
    apply PaperSelectedNifsSemantics.transition_of_paper
      (incomingParent := (input.running selected).parent)
      (polynomial := setup.piCcsInput input selected)
      (priorState := setup.priorTranscriptState input selected)
    simpa [selected, PaperSelectedNifsSemantics.contextOf,
      SelectedNifsSemantics.contextOf,
      ActiveSemantics.Construction2.selectedNifsSetup, contextAt, invocationAt]
      using paperTransition
  rcases paperTransition with ⟨data, witness, realized⟩
  have selectedChildrenEq :
      selectedNext.children =
        outputChildren (contextAt setup input selected) certificate.nifs := by
    have resultChildren := congrArg
      (fun result => result.children) physical.resultExact
    exact (by simpa [selected] using resultChildren.symm)
  have selectedOutputEq :
      (outputOf machine input selected selectedNext).toPaper.runningNext
          selected =
        outputChildren (contextAt setup input selected) certificate.nifs := by
    have selectedSlot :=
      outputOf_runningNext_selected machine input selected selectedNext
    exact (congrArg (fun result => result.children) selectedSlot).trans
      selectedChildrenEq
  have priorPcValid : Paper.InRange slotCount input.priorPc := by
    rw [physical.outer.priorSlot]
    exact (Paper.ProgramCounter.ofIndex selected).valid
  have selectedEq : Paper.selectedIndex priorPcValid = selected := by
    apply Fin.ext
    simp [selected, Paper.selectedIndex, Paper.ProgramCounter.index,
      physical.outer.priorSlot]
  rw [physical.outputExact]
  refine {
    iterationPositive := physical.outer.iterationPositive
    priorPcValid := priorPcValid
    priorPublicInput := physical.outer.priorPublicInput
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
      physical.outer.dispatch
    have indexEq :
        (machine.control input.toPaper.zi input.toPaper.witness).index =
          functionIndex := by
      rw [controlEq]
      exact Paper.ProgramCounter.index_ofIndex functionIndex
    rw [indexEq] at derived
    simpa using derived
  · constructor
    · change input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey
          (Paper.selectedIndex priorPcValid)
      rw [selectedEq]
      exact physical.outer.expectedStructure
    · intro child
      have runningStructure :
          ((input.running selected).children child).constraintSystem =
            Phi81Relation.Structure.ofSourceData publicRingColumns publicFits
              data := by
        simpa [selected, contextAt, invocationAt,
          Nightstream.Protocol.FPrime.ConcretePhi81.Context.Template.build,
          Nightstream.Protocol.FPrime.ConcretePhi81.Context.Invocation.sourceProduct]
          using (realized.input.running child).constraintSystem
      have freshStructure :
          input.fresh.constraintSystem =
            Phi81Relation.Structure.ofSourceData publicRingColumns publicFits
              data := by
        simpa [selected, contextAt, invocationAt,
          Nightstream.Protocol.FPrime.ConcretePhi81.Context.Template.build,
          Nightstream.Protocol.FPrime.ConcretePhi81.Context.Invocation.sourceProduct]
          using
            (realized.input.fresh
              ⟨0, FixedActive.arity.freshPositive⟩).constraintSystem
      change ((input.running (Paper.selectedIndex priorPcValid)).children
          child).constraintSystem =
        setup.expectedStructure input.verifierKey
          (Paper.selectedIndex priorPcValid)
      rw [selectedEq]
      exact runningStructure.trans
        (freshStructure.symm.trans physical.outer.expectedStructure)
  · change PaperSelectedNifsSemantics.Transition
      (ActiveSemantics.Construction2.selectedNifsSetup setup)
      input.verifierKey (Paper.selectedIndex priorPcValid)
      (Paper.Construction2.selectedInput input.toPaper
        (Paper.selectedIndex priorPcValid))
      ((outputOf machine input selected selectedNext).toPaper.runningNext
        (Paper.selectedIndex priorPcValid))
    rw [selectedEq, selectedOutputEq]
    exact selectedPaperTransition
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

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
