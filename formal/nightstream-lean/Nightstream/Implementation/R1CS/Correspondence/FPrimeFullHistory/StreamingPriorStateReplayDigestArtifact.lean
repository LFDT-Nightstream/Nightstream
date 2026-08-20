import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeStateDigest
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPriorStateReplayArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPriorStateReplayTransitionArtifact
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Operations

/-!
Contract: exact local-state digest meaning for prior-state replay arms.

Owns the structural equality between each accepted ten-field state-digest
trace and the protocol-bound native Poseidon2 digest of the same replay-state
assignment fields. It exposes the exact full/final before/after source lanes.

Does not own replay transitions, target authority, phase selection, collision
resistance, or the complete lifecycle relation.

Assurance tier: artifact-checked.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayDigestArtifact

open Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeDigestDomain
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayDigestExecutionCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayTransitionArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

def stateFrame : List Nat := [2, 5, 435744240755, 10]

def stateColumns (start : Nat) : List Nat :=
  List.range' start 10

def digestOperations (start : Nat) : List ColumnReplay.Operation :=
  stateFrame.map ColumnReplay.Operation.pinned ++
    (stateColumns start).map ColumnReplay.Operation.external ++
      [ColumnReplay.Operation.digest]

private theorem range'_add (start left right : Nat) :
    List.range' start (left + right) =
      List.range' start left ++ List.range' (start + left) right := by
  apply (List.range'_eq_append_iff).2
  refine ⟨left, by omega, rfl, ?_⟩
  simp

theorem digestOperations_eq_stateOperations (start : Nat) :
    digestOperations start = stateOperations start := by
  unfold digestOperations stateColumns stateOperations stateFrame
    statePinOperations stateExternalOperations0 stateExternalOperations1
    stateExternalOperations2 stateDigestOperations
  rw [range'_add start 4 6, range'_add (start + 4) 4 2]
  simp only [List.map_cons, List.map_nil, List.map_append, List.append_assoc]

private theorem fieldAt_eq_wordField
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (column : Nat) :
    CallRefinement.fieldAt assignment canonical column =
      wordField (assignment column) := by
  apply Fin.ext
  have belowU64 : assignment column < 2 ^ 64 :=
    lt_trans (canonical column) (by norm_num [goldilocksP])
  simp only [CallRefinement.fieldAt_val]
  change assignment column =
    assignment column % u64Modulus % goldilocksP
  have wordExact : assignment column % u64Modulus = assignment column :=
    Nat.mod_eq_of_lt (by simpa [u64Modulus] using belowU64)
  have fieldExact : assignment column % goldilocksP = assignment column :=
    Nat.mod_eq_of_lt (canonical column)
  rw [wordExact, fieldExact]

private theorem semanticExecute_pinned
    (assignment : Nat -> Nat)
    (canonical : ColumnReplay.CanonicalAssignment assignment)
    (run : ColumnReplay.SemanticRun) (values : List Nat) :
    ColumnReplay.semanticExecute assignment canonical run
        (values.map ColumnReplay.Operation.pinned) =
      { run with state := absorbWords run.state values } := by
  induction values generalizing run with
  | nil => rfl
  | cons value values inductionHypothesis =>
      change ColumnReplay.semanticExecute assignment canonical
          { run with state := absorbElem run.state (wordField value) }
          (values.map ColumnReplay.Operation.pinned) = _
      rw [inductionHypothesis]
      rfl

private theorem semanticExecute_external
    (assignment : Nat -> Nat)
    (canonical : ColumnReplay.CanonicalAssignment assignment)
    (run : ColumnReplay.SemanticRun) (columns : List Nat) :
    ColumnReplay.semanticExecute assignment canonical run
        (columns.map ColumnReplay.Operation.external) =
      { run with state := absorbWords run.state (columns.map assignment) } := by
  induction columns generalizing run with
  | nil => rfl
  | cons column columns inductionHypothesis =>
      change ColumnReplay.semanticExecute assignment canonical
          { run with state := (absorbElem run.state
              (CallRefinement.fieldAt assignment canonical column)) }
          (columns.map ColumnReplay.Operation.external) = _
      rw [fieldAt_eq_wordField, inductionHypothesis]
      rfl

def initialSemanticRun : ColumnReplay.SemanticRun where
  state := collapsedInitialState
  digests := []

def semanticDigestRun
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (start : Nat) : ColumnReplay.SemanticRun :=
  ColumnReplay.semanticExecute assignment canonical initialSemanticRun
    (digestOperations start)

def zeroDigest : Fin 4 -> Field := fun _ => wordField 0

def stateDigest
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (start : Nat) : Fin 4 -> Field :=
  (semanticDigestRun assignment canonical start).digests.getD 0 zeroDigest

private theorem semanticExecute_first_digest
    (assignment : Nat -> Nat)
    (canonical : ColumnReplay.CanonicalAssignment assignment)
    (state :
      Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.State) :
    (ColumnReplay.semanticExecute assignment canonical
        { state := state, digests := [] }
        [ColumnReplay.Operation.digest]).digests.getD 0 zeroDigest =
      (Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.digest
        state).2 := by
  rfl

private theorem state_fields_label_exact_local :
    packedBytesWithLen stateFieldsLabel = [5, 435744240755] := by
  native_decide

private theorem stateFieldsWords_exact
    (assignment : Nat -> Nat) (start : Nat) :
    FPrimeFullHistoryStreamingPreludeStateDigest.stateFieldsWords
        ((stateColumns start).map assignment) =
      stateFrame ++ (stateColumns start).map assignment := by
  simp [FPrimeFullHistoryStreamingPreludeStateDigest.stateFieldsWords,
    stateColumns, stateFrame, state_fields_label_exact_local]

/-- The independent semantic execution is the named native Poseidon2 digest
of the same ten assignment fields. -/
theorem stateDigest_eq_native
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (start : Nat) :
    stateDigest assignment canonical start =
      FPrimeFullHistoryStreamingPreludeStateDigest.stateDigest
        ((stateColumns start).map assignment) := by
  have inputExact :
      absorbWords (absorbWords collapsedInitialState stateFrame)
          ((stateColumns start).map assignment) =
        absorbWords collapsedInitialState
          (FPrimeFullHistoryStreamingPreludeStateDigest.stateFieldsWords
            ((stateColumns start).map assignment)) := by
    rw [stateFieldsWords_exact, absorbWords_append]
  unfold stateDigest semanticDigestRun digestOperations
  rw [Operations.semanticExecute_append, Operations.semanticExecute_append,
    semanticExecute_pinned, semanticExecute_external]
  change
    (ColumnReplay.semanticExecute assignment canonical
      { state := absorbWords (absorbWords collapsedInitialState stateFrame)
          ((stateColumns start).map assignment),
        digests := [] }
      [ColumnReplay.Operation.digest]).digests.getD 0 zeroDigest = _
  rw [inputExact, semanticExecute_first_digest]
  rfl

private theorem getD_mem_of_lt {alpha : Type} [Inhabited alpha]
    {entries : List alpha} {index : Nat} (bounded : index < entries.length) :
    entries.getD index default ∈ entries := by
  have member := List.getElem_mem (l := entries) bounded
  rwa [List.getElem_eq_getD default] at member

private theorem initial_pin_shape
    (initialStart frameStart padColumn queryStart : Nat) :
    forall lane : Fin 8,
      (statePins initialStart frameStart padColumn queryStart).getD lane.val
          default =
        (initialStart + lane.val,
          collapsedInitialValues.getD lane.val 0) := by
  intro lane
  fin_cases lane <;> rfl

private theorem state_ext
    {left right :
      Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

private theorem semantic_run_ext
    {left right : ColumnReplay.SemanticRun}
    (state : left.state = right.state)
    (digests : left.digests = right.digests) : left = right := by
  cases left
  cases right
  simp_all

private theorem decoded_start_eq_collapsed
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (initialStart frameStart padColumn queryStart : Nat)
    (pins : ∀ pin ∈ statePins initialStart frameStart padColumn queryStart,
      assignment pin.1 = pin.2) :
    ColumnReplay.decodeRun assignment canonical
        (checkpointRun (fun lane => initialStart + lane.val) ⟨0, by decide⟩
          8 0) =
      initialSemanticRun := by
  apply semantic_run_ext
  · apply state_ext
    · funext lane
      apply Fin.ext
      have bounded : lane.val <
          (statePins initialStart frameStart padColumn queryStart).length := by
        simp [statePins]
        exact Nat.lt_trans lane.isLt (by decide)
      have pinEqual := pins
        ((statePins initialStart frameStart padColumn queryStart).getD lane.val
          default)
        (getD_mem_of_lt bounded)
      have shape := initial_pin_shape initialStart frameStart padColumn
        queryStart lane
      change assignment (initialStart + lane.val) =
        (collapsedInitialState.lanes lane).val
      rw [shape] at pinEqual
      exact pinEqual.trans (collapsed_initial_state_exact.1 lane).symm
    · apply Fin.ext
      exact collapsed_initial_state_exact.2.symm
  · rfl

private theorem outputDigest_of_refinement
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (start : Nat) (result : ColumnReplay.Run) (columns : Fin 4 -> Nat)
    (resultShape : result.digests = [columns])
    (refined : semanticDigestRun assignment canonical start =
      ColumnReplay.decodeRun assignment canonical result) :
    stateDigest assignment canonical start =
      ColumnReplay.decodeDigest assignment canonical columns := by
  have digestsEqual := congrArg ColumnReplay.SemanticRun.digests refined
  funext lane
  have selected := congrArg
    (fun digests => (digests.getD 0 zeroDigest) lane) digestsEqual
  simpa [stateDigest, semanticDigestRun, ColumnReplay.decodeRun,
    resultShape] using selected

private theorem full_before_refinement
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    semanticDigestRun assignment canonical 1 =
      ColumnReplay.decodeRun assignment canonical fullBeforeResult := by
  have accepted := full_before_trace_accepted assignment canonical one satisfied
  have refined := full_before_state_refines assignment canonical one satisfied
  change ColumnReplay.semanticExecute assignment canonical
      (ColumnReplay.decodeRun assignment canonical
        (checkpointRun (fun lane => 154779 + lane.val) ⟨0, by decide⟩ 8 0))
      (stateOperations 1) = _ at refined
  rw [decoded_start_eq_collapsed assignment canonical 154779 154787 156591
    157192 accepted.1] at refined
  unfold semanticDigestRun
  rw [digestOperations_eq_stateOperations]
  exact refined

private theorem full_after_refinement
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    semanticDigestRun assignment canonical 11 =
      ColumnReplay.decodeRun assignment canonical fullAfterResult := by
  have accepted := full_after_trace_accepted assignment canonical one satisfied
  have refined := full_after_state_refines assignment canonical one satisfied
  change ColumnReplay.semanticExecute assignment canonical
      (ColumnReplay.decodeRun assignment canonical
        (checkpointRun (fun lane => 157194 + lane.val) ⟨0, by decide⟩ 8 0))
      (stateOperations 11) = _ at refined
  rw [decoded_start_eq_collapsed assignment canonical 157194 157202 159006
    159607 accepted.1] at refined
  unfold semanticDigestRun
  rw [digestOperations_eq_stateOperations]
  exact refined

private theorem final_before_refinement
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    semanticDigestRun assignment canonical 1 =
      ColumnReplay.decodeRun assignment canonical finalBeforeResult := by
  have accepted := final_before_trace_accepted assignment canonical one satisfied
  have refined := final_before_state_refines assignment canonical one satisfied
  change ColumnReplay.semanticExecute assignment canonical
      (ColumnReplay.decodeRun assignment canonical
        (checkpointRun (fun lane => 79786 + lane.val) ⟨0, by decide⟩ 8 0))
      (stateOperations 1) = _ at refined
  rw [decoded_start_eq_collapsed assignment canonical 79786 79794 81598
    82199 accepted.1] at refined
  unfold semanticDigestRun
  rw [digestOperations_eq_stateOperations]
  exact refined

private theorem final_after_refinement
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    semanticDigestRun assignment canonical 11 =
      ColumnReplay.decodeRun assignment canonical finalAfterResult := by
  have accepted := final_after_trace_accepted assignment canonical one satisfied
  have refined := final_after_state_refines assignment canonical one satisfied
  change ColumnReplay.semanticExecute assignment canonical
      (ColumnReplay.decodeRun assignment canonical
        (checkpointRun (fun lane => 82201 + lane.val) ⟨0, by decide⟩ 8 0))
      (stateOperations 11) = _ at refined
  rw [decoded_start_eq_collapsed assignment canonical 82201 82209 84013
    84614 accepted.1] at refined
  unfold semanticDigestRun
  rw [digestOperations_eq_stateOperations]
  exact refined

private theorem persistentFields_replayStateAt
    (assignment : Nat -> Nat) (start : Nat) :
    Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming.persistentFields
        (replayStateAt assignment start) =
      (stateColumns start).map assignment := by
  apply List.ext_get
  · simp [Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming.persistentFields,
      Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.duplexStateFields,
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.width,
      stateColumns]
  · intro index leftBound rightBound
    have indexBound : index < 10 := by
      simpa [stateColumns] using rightBound
    interval_cases index <;>
      simp [Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming.persistentFields,
        Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.duplexStateFields,
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.width,
        replayStateAt, stateColumns]

/-- Full-arm rows bind the before-state digest source lanes to the exact ten
before-state assignment fields. -/
theorem full_before_digest_exact
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) (lane : Fin 4) :
    assignment (157184 + lane.val) =
      (FPrimeFullHistoryStreamingPreludeStateDigest.stateDigest
        (persistentFields (replayStateAt assignment 1)) lane).val := by
  have output := outputDigest_of_refinement assignment canonical 1
    fullBeforeResult (fun lane => 157184 + lane.val) rfl
    (full_before_refinement assignment canonical one satisfied)
  have native :=
    (stateDigest_eq_native assignment canonical 1).symm.trans output
  rw [persistentFields_replayStateAt]
  exact (congrArg (fun digest => (digest lane).val) native).symm

/-- Full-arm rows bind the after-state digest source lanes to the exact ten
after-state assignment fields. -/
theorem full_after_digest_exact
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) (lane : Fin 4) :
    assignment (159599 + lane.val) =
      (FPrimeFullHistoryStreamingPreludeStateDigest.stateDigest
        (persistentFields (replayStateAt assignment 11)) lane).val := by
  have output := outputDigest_of_refinement assignment canonical 11
    fullAfterResult (fun lane => 159599 + lane.val) rfl
    (full_after_refinement assignment canonical one satisfied)
  have native :=
    (stateDigest_eq_native assignment canonical 11).symm.trans output
  rw [persistentFields_replayStateAt]
  exact (congrArg (fun digest => (digest lane).val) native).symm

/-- Final-arm rows bind the before-state digest source lanes to the exact ten
before-state assignment fields. -/
theorem final_before_digest_exact
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) (lane : Fin 4) :
    assignment (82191 + lane.val) =
      (FPrimeFullHistoryStreamingPreludeStateDigest.stateDigest
        (persistentFields (replayStateAt assignment 1)) lane).val := by
  have output := outputDigest_of_refinement assignment canonical 1
    finalBeforeResult (fun lane => 82191 + lane.val) rfl
    (final_before_refinement assignment canonical one satisfied)
  have native :=
    (stateDigest_eq_native assignment canonical 1).symm.trans output
  rw [persistentFields_replayStateAt]
  exact (congrArg (fun digest => (digest lane).val) native).symm

/-- Final-arm rows bind the after-state digest source lanes to the exact ten
after-state assignment fields. -/
theorem final_after_digest_exact
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) (lane : Fin 4) :
    assignment (84606 + lane.val) =
      (FPrimeFullHistoryStreamingPreludeStateDigest.stateDigest
        (persistentFields (replayStateAt assignment 11)) lane).val := by
  have output := outputDigest_of_refinement assignment canonical 11
    finalAfterResult (fun lane => 84606 + lane.val) rfl
    (final_after_refinement assignment canonical one satisfied)
  have native :=
    (stateDigest_eq_native assignment canonical 11).symm.trans output
  rw [persistentFields_replayStateAt]
  exact (congrArg (fun digest => (digest lane).val) native).symm

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayDigestArtifact
