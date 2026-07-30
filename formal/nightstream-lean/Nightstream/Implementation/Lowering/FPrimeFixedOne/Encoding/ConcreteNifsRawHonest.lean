import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerHonest
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionBridge
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionHonest
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawSemantics

/-!
Contract: honest completion of the complete ungated selected-NIFS row program.

Starting from authoritative encoded inputs and the deterministic selected call
result, this module constructs one assignment for all six raw slices.  The
operational and sampler prefix is completed first; the Phi81 action products
are completed second.  Every other slice is allocation-free.

This module owns neither activation nor the final `CallRecipe`.  It accepts no
checker conclusion, output equation, sampler result, or source-authority
proposition as a premise.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawHonest

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

private abbrev TranscriptState := Poseidon2Duplex.State

section SelectedFrame

variable {shape : SemanticShape}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {keys : Fin 1 →
  SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows}
variable {defaultRunning :
  SelectedRunning shape publicRingColumns publicFits verifierRows}
variable {machine :
  Machine
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    RunningWitness
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  ConcreteNifsParameters.selected keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

private abbrev FamilyFor
    (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

private abbrev FrameFor
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)} :=
  CallFrame (signature := signature Selected)
    (FamilyFor application) Call.nifsVerify
    (Refs.cons runningRef
      (Refs.cons freshRef (Refs.cons proofRef .nil)))

private theorem outputs_subset_visible
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ∀ column, column ∈ frame.outputs.ids → column ∈ frame.visibleIds := by
  intro column member
  simp [CallFrame.visibleIds, member]

private theorem operands_subset_visible
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ∀ column, column ∈ frame.operands.ids → column ∈ frame.visibleIds := by
  intro column member
  have contextMember : column ∈ frame.contextBundles.ids :=
    RefBundles.fromSchema_ids_subset _ _ column member
  simp [CallFrame.visibleIds, contextMember]

private theorem action_changes_only_temporaries
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (before : ColumnId → Field) :
    ChangesOnly frame.temporaries.ids before
      (ConcreteNifsPiRlcActionHonest.honestAssignment
        application profile frame before) := by
  intro column notTemporary
  apply ConcreteNifsPiRlcActionHonest.honestAssignment_changesOnly
  intro actionMember
  exact notTemporary
    (ConcreteNifsPiRlcActionHonest.column_mem_temporaries
      application profile frame fits column actionMember)

private theorem prefix_satisfied_after_action
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (before : ColumnId → Field)
    (satisfied :
      Satisfies
        (ConcreteNifsOperationalSampler.rows application profile frame)
        (numericAssignment (columnMap frame) before)) :
    Satisfies
      (ConcreteNifsOperationalSampler.rows application profile frame)
      (numericAssignment (columnMap frame)
        (ConcreteNifsPiRlcActionHonest.honestAssignment
          application profile frame before)) := by
  apply KHornerSupport.satisfies_extend _
    (numericAssignment (columnMap frame) before)
    (numericAssignment (columnMap frame)
      (ConcreteNifsPiRlcActionHonest.honestAssignment
        application profile frame before))
  · intro row rowMember column mentioned
    have below :=
      ConcreteNifsOperationalSamplerConservation.rows_below_actionBase
        application profile frame row rowMember column mentioned
    change
      (before (columnMap frame column)).val =
        (ConcreteNifsPiRlcActionHonest.honestAssignment
          application profile frame before (columnMap frame column)).val
    exact congrArg Fin.val
      (ConcreteNifsPiRlcActionHonest.honestAssignment_preserves_before_actionBase
        application profile frame fits before column below).symm
  · exact satisfied

/-- **Headline raw honest completeness.** A successful selected `nifsVerify`
call has one exact completion that changes only declared temporaries and
satisfies every Lean-owned raw row. -/
theorem rows_honest
    (prime : EuclidPrime goldilocksP)
    (field : PiRlcCanonicalCandidateHonest.FieldInverse)
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (constantWire : initial frame.one = 1)
    (encodedInputs :
      frame.operands.Encodes (FamilyFor application) initial
        (.cons running (.cons fresh (.cons proof .nil))))
    (encodedOutput :
      frame.outputs.Encodes (FamilyFor application) initial
        (.cons output .nil))
    (resultExact :
      callEval Selected Call.nifsVerify
          (.cons running (.cons fresh (.cons proof .nil))) =
        some (.cons output .nil)) :
    ∃ completed : ColumnId → Field,
      AgreesOn frame.visibleIds initial completed ∧
        ChangesOnly frame.temporaries.ids initial completed ∧
        RawSatisfies
          (ConcreteNifsRawProgram.rawRows application profile frame)
          completed := by
  rcases
      (ConcreteNifsParameters.callEval_nifsVerify_eq_some_iff
        keys defaultRunning machine terminalRelations terminalChecks
        widths footprints running fresh proof output).mp resultExact with
    ⟨accepted, resultEq⟩
  subst output
  rcases
      ConcreteNifsOperationalSamplerHonest.rows_honest
        prime field application profile frame initial running fresh proof
        fits encodedInputs constantWire accepted with
    ⟨middle, middleAgrees, middleChanges, middleEncodes, middleWire,
      middleSatisfied⟩
  let completed :=
    ConcreteNifsPiRlcActionHonest.honestAssignment
      application profile frame middle
  have actionVisible :
      AgreesOn frame.visibleIds middle completed := by
    exact
      ConcreteNifsPiRlcActionHonest.honestAssignment_agreesOn_visible
        application profile frame fits middle
  have completedAgrees :
      AgreesOn frame.visibleIds initial completed :=
    agreesOn_trans middleAgrees actionVisible
  have actionChanges :
      ChangesOnly frame.temporaries.ids middle completed := by
    exact action_changes_only_temporaries
      application profile frame fits middle
  have completedChanges :
      ChangesOnly frame.temporaries.ids initial completed := by
    intro column notTemporary
    rw [actionChanges column notTemporary,
      middleChanges column notTemporary]
  have middleOutputEncodes :
      frame.outputs.Encodes (FamilyFor application) middle
        (.cons
          (SelectedRunning.ofResult
            (FixedActive.resultOf
              (ConcreteNifsParameters.context
                (keys
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                running fresh proof).materialize
              proof.certificate))
          .nil) := by
    apply frame.outputs.encodes_of_agrees
      (FamilyFor application) initial middle
    · exact agreesOn_of_subset
        (outputs_subset_visible application frame) middleAgrees
    · exact encodedOutput
  have completedInputsEncodes :
      frame.operands.Encodes (FamilyFor application) completed
        (.cons running (.cons fresh (.cons proof .nil))) := by
    apply frame.operands.encodes_of_agrees
      (FamilyFor application) middle completed
    · exact agreesOn_of_subset
        (operands_subset_visible application frame)
        actionVisible
    · exact middleEncodes
  have completedOutputEncodes :
      frame.outputs.Encodes (FamilyFor application) completed
        (.cons
          (SelectedRunning.ofResult
            (FixedActive.resultOf
              (ConcreteNifsParameters.context
                (keys
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                running fresh proof).materialize
              proof.certificate))
          .nil) := by
    apply frame.outputs.encodes_of_agrees
      (FamilyFor application) middle completed
    · exact agreesOn_of_subset
        (outputs_subset_visible application frame) actionVisible
    · exact middleOutputEncodes
  have middleDecodedInputs :=
    frame.operands.decodes_of_encodes
      (FamilyFor application) middle
      (.cons running (.cons fresh (.cons proof .nil)))
      middleEncodes
  have middleDecodedOutput :=
    frame.outputs.decodes_of_encodes
      (FamilyFor application) middle
      (.cons
        (SelectedRunning.ofResult
          (FixedActive.resultOf
            (ConcreteNifsParameters.context
              (keys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              running fresh proof).materialize
            proof.certificate))
        .nil)
      middleOutputEncodes
  have completedDecodedInputs :=
    frame.operands.decodes_of_encodes
      (FamilyFor application) completed
      (.cons running (.cons fresh (.cons proof .nil)))
      completedInputsEncodes
  have completedDecodedOutput :=
    frame.outputs.decodes_of_encodes
      (FamilyFor application) completed
      (.cons
        (SelectedRunning.ofResult
          (FixedActive.resultOf
            (ConcreteNifsParameters.context
              (keys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              running fresh proof).materialize
            proof.certificate))
        .nil)
      completedOutputEncodes
  have completedWire : completed frame.one = 1 :=
    ConcreteNifsPiRlcActionHonest.honestAssignment_constantWire
      application profile frame fits middle middleWire
  have operationalSamplerSatisfied :
      Satisfies
        (ConcreteNifsOperationalSampler.rows application profile frame)
        (numericAssignment (columnMap frame) completed) :=
    prefix_satisfied_after_action application profile frame fits middle
      middleSatisfied
  have operationalSatisfied :=
    ConcreteNifsRawSemantics.operationalOccurrence_satisfied
      application profile frame completed operationalSamplerSatisfied
  have runningEquations :=
    FixedActive.Canonical.RunningAuthority.equations_of_accepted
      accepted.running
  have actionEquations :=
    ConcreteNifsPiRlcActionBridge.equations_of_result
      (keys := keys) running fresh proof
  have piDecEquations :
      ConcreteNifsPiDecSemantics.OutputEquations
        (SelectedRunning.ofResult
          (FixedActive.resultOf
            (ConcreteNifsParameters.context
              (keys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              running fresh proof).materialize
            proof.certificate))
        proof := by
    refine {
      commitment := ?_
      publicInput := ?_
      evaluations := ?_
    }
    · simpa [SelectedRunning.ofResult, FixedActive.resultOf, Result.resultOf,
        DerivedPiDec.RecompositionEquations, ConcretePhi81.outputChildren,
        Execution.piDecChildren, PiDecChildPayload.materialize] using
          accepted.tail.piDecRecomposition.commitment
    · simpa [SelectedRunning.ofResult, FixedActive.resultOf, Result.resultOf,
        DerivedPiDec.RecompositionEquations, ConcretePhi81.outputChildren,
        Execution.piDecChildren, PiDecChildPayload.materialize] using
          accepted.tail.piDecRecomposition.publicInput
    · simpa [SelectedRunning.ofResult, FixedActive.resultOf, Result.resultOf,
        DerivedPiDec.RecompositionEquations, ConcretePhi81.outputChildren,
        Execution.piDecChildren, PiDecChildPayload.materialize] using
          accepted.tail.piDecRecomposition.evaluations
  have childEquations :
      ConcreteNifsOutputSemantics.ChildEquations
        (SelectedRunning.ofResult
          (FixedActive.resultOf
            (ConcreteNifsParameters.context
              (keys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              running fresh proof).materialize
            proof.certificate))
        proof := by
    refine {
      commitment := ?_
      publicInput := ?_
      point := ?_
      evaluations := ?_
    }
    · intro child
      rfl
    · intro child
      rfl
    · intro child
      rfl
    · intro child
      rfl
  have canonicalitySatisfied :=
    ConcreteNifsProofCanonicalityRows.rows_honest
      application profile frame completed running fresh proof completedWire
      completedDecodedInputs
  have runningSatisfied :=
    ConcreteNifsRunningAuthoritySemantics.rows_honest
      application profile frame completed running fresh proof completedWire
      completedDecodedInputs runningEquations
  have pointSatisfied :=
    ConcreteNifsPiRlcPointSemantics.rows_honest
      application profile frame completed running fresh proof
      (SelectedRunning.ofResult
        (FixedActive.resultOf
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate))
      completedWire completedDecodedInputs completedDecodedOutput
      operationalSatisfied (by rfl)
  have actionSatisfied :=
    ConcreteNifsPiRlcActionHonest.rows_honest
      application profile frame fits middle running fresh proof
      (SelectedRunning.ofResult
        (FixedActive.resultOf
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate))
      middleWire middleDecodedInputs middleDecodedOutput
      actionEquations
  have piDecSatisfied :=
    ConcreteNifsPiDecSemantics.rows_honest
      application profile frame completed running fresh proof
      (SelectedRunning.ofResult
        (FixedActive.resultOf
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate))
      completedWire completedDecodedInputs completedDecodedOutput
      piDecEquations
  have outputSatisfied :=
    ConcreteNifsOutputSemantics.rows_honest
      application profile frame completed running fresh proof
      (SelectedRunning.ofResult
        (FixedActive.resultOf
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate))
      completedWire completedDecodedInputs completedDecodedOutput
      childEquations
  have runningRaw :=
    (ConcreteNifsRawSemantics.translate_satisfies_iff application frame
      (ConcreteNifsRunningAuthorityRows.rows application profile frame)
      completed).2 runningSatisfied
  have operationalRaw :=
    (ConcreteNifsRawSemantics.translate_satisfies_iff application frame
      (ConcreteNifsOperationalSampler.rows application profile frame)
      completed).2 operationalSamplerSatisfied
  have pointRaw :=
    (ConcreteNifsRawSemantics.translate_satisfies_iff application frame
      (ConcreteNifsPiRlcPointRows.rows application profile frame)
      completed).2 pointSatisfied
  have piDecRaw :=
    (ConcreteNifsRawSemantics.translate_satisfies_iff application frame
      (ConcreteNifsPiDecRows.rows application profile frame)
      completed).2 piDecSatisfied
  have outputRaw :=
    (ConcreteNifsRawSemantics.translate_satisfies_iff application frame
      (ConcreteNifsOutputRows.rows application profile frame)
      completed).2 outputSatisfied
  refine ⟨completed, completedAgrees, completedChanges, ?_⟩
  unfold ConcreteNifsRawProgram.rawRows
  exact
    (rawSatisfies_append_iff _ _ completed).2
      ⟨
        (rawSatisfies_append_iff _ _ completed).2
          ⟨
            (rawSatisfies_append_iff _ _ completed).2
              ⟨
                (rawSatisfies_append_iff _ _ completed).2
                  ⟨
                    (rawSatisfies_append_iff _ _ completed).2
                      ⟨
                        (rawSatisfies_append_iff _ _ completed).2
                          ⟨canonicalitySatisfied, runningRaw⟩,
                        operationalRaw⟩,
                    pointRaw⟩,
                actionSatisfied⟩,
            piDecRaw⟩,
        outputRaw⟩

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawHonest
