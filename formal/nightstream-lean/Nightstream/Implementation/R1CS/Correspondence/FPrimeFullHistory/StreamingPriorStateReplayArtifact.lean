import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPriorStateReplayRelation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPriorStateReplayExecutionCertificate

/-!
Exact source-row boundary for prior-state replay executions.

Owns the exact residual slices that bind state and target digest pins,
acceptance of each local Poseidon2 trace from source-row satisfaction, and
refinement of each certified physical execution to the independent transcript
machine. It owns no lifecycle target authority or complete F-prime relation.

Assurance tier: artifact-checked. Rust-conformant status also requires the
separate current Rust-to-Lean drift test for the selected source artifact.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayTransitionExecutionCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayDigestExecutionCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

def fullStatePinRows : List IndexedRow :=
  (fullResidualRows0Part0.drop 13).take 30

def finalTargetPinRows : List IndexedRow :=
  (finalResidualRows0Part2.drop 6).take 3

def finalStatePinRows : List IndexedRow :=
  (finalResidualRows0Part2.drop 13).take 30

private theorem full_state_pin_rows_subset :
    ∀ indexed ∈ fullStatePinRows, indexed ∈ fullArtifact.residualRows := by
  intro indexed member
  apply fullResidualRows0Part0_subset indexed
  apply List.mem_of_mem_drop
  exact List.mem_of_mem_take member

private theorem final_target_pin_rows_subset :
    ∀ indexed ∈ finalTargetPinRows, indexed ∈ finalArtifact.residualRows := by
  intro indexed member
  apply finalResidualRows0Part2_subset indexed
  apply List.mem_of_mem_drop
  exact List.mem_of_mem_take member

private theorem final_state_pin_rows_subset :
    ∀ indexed ∈ finalStatePinRows, indexed ∈ finalArtifact.residualRows := by
  intro indexed member
  apply finalResidualRows0Part2_subset indexed
  apply List.mem_of_mem_drop
  exact List.mem_of_mem_take member

theorem full_state_pin_rows_exact :
    fullStatePinRows.map (fun indexed => indexed.row) =
      Poseidon2Normalized.normalizeProgram
        (ConstantPins.rows fullBeforeTrace.pins ++
          ConstantPins.rows fullAfterTrace.pins) := by
  rfl

theorem final_target_pin_rows_exact :
    finalTargetPinRows.map (fun indexed => indexed.row) =
      Poseidon2Normalized.normalizeProgram
        (ConstantPins.rows finalTargetTrace.pins) := by
  rfl

theorem final_state_pin_rows_exact :
    finalStatePinRows.map (fun indexed => indexed.row) =
      Poseidon2Normalized.normalizeProgram
        (ConstantPins.rows finalBeforeTrace.pins ++
          ConstantPins.rows finalAfterTrace.pins) := by
  rfl

theorem full_before_pins_canonical :
    ConstantPins.ValuesCanonical fullBeforeTrace.pins := by
  decide

theorem full_after_pins_canonical :
    ConstantPins.ValuesCanonical fullAfterTrace.pins := by
  decide

theorem final_before_pins_canonical :
    ConstantPins.ValuesCanonical finalBeforeTrace.pins := by
  decide

theorem final_after_pins_canonical :
    ConstantPins.ValuesCanonical finalAfterTrace.pins := by
  decide

theorem final_target_pins_canonical :
    ConstantPins.ValuesCanonical finalTargetTrace.pins := by
  decide

private theorem rowsIncluded_self (rows : List Row) :
    rowsIncluded rows rows = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true member

private theorem indexed_rows_satisfy
    (arm : RawArm) (rows : List IndexedRow) (assignment : Nat → Nat)
    (subset : ∀ indexed ∈ rows, indexed ∈ arm.residualRows)
    (satisfied : arm.Satisfied assignment) :
    Satisfies (rows.map (fun indexed => indexed.row)) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact satisfied.2.2 indexed (subset indexed indexedMember)

private theorem full_pin_rows_satisfy
    (assignment : Nat → Nat) (satisfied : fullArtifact.Satisfied assignment) :
    Satisfies
      (ConstantPins.rows fullBeforeTrace.pins ++
        ConstantPins.rows fullAfterTrace.pins) assignment := by
  apply (Poseidon2Normalized.satisfies_normalizeProgram _ assignment).mp
  rw [← full_state_pin_rows_exact]
  exact indexed_rows_satisfy fullArtifact fullStatePinRows assignment
    full_state_pin_rows_subset satisfied

private theorem final_target_pin_rows_satisfy
    (assignment : Nat → Nat) (satisfied : finalArtifact.Satisfied assignment) :
    Satisfies (ConstantPins.rows finalTargetTrace.pins) assignment := by
  apply (Poseidon2Normalized.satisfies_normalizeProgram _ assignment).mp
  rw [← final_target_pin_rows_exact]
  exact indexed_rows_satisfy finalArtifact finalTargetPinRows assignment
    final_target_pin_rows_subset satisfied

private theorem final_pin_rows_satisfy
    (assignment : Nat → Nat) (satisfied : finalArtifact.Satisfied assignment) :
    Satisfies
      (ConstantPins.rows finalBeforeTrace.pins ++
        ConstantPins.rows finalAfterTrace.pins) assignment := by
  apply (Poseidon2Normalized.satisfies_normalizeProgram _ assignment).mp
  rw [← final_state_pin_rows_exact]
  exact indexed_rows_satisfy finalArtifact finalStatePinRows assignment
    final_state_pin_rows_subset satisfied

private theorem full_before_pin_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    ∀ pin ∈ fullBeforeTrace.pins, assignment pin.1 = pin.2 := by
  have combined := full_pin_rows_satisfy assignment satisfied
  have beforeSatisfies :
      Satisfies (ConstantPins.rows fullBeforeTrace.pins) assignment := by
    intro row member
    exact combined row (List.mem_append_left _ member)
  exact ConstantPins.sound full_before_pins_canonical
    (rowsIncluded_self _) canonical one beforeSatisfies

private theorem full_after_pin_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    ∀ pin ∈ fullAfterTrace.pins, assignment pin.1 = pin.2 := by
  have combined := full_pin_rows_satisfy assignment satisfied
  have afterSatisfies :
      Satisfies (ConstantPins.rows fullAfterTrace.pins) assignment := by
    intro row member
    exact combined row (List.mem_append_right _ member)
  exact ConstantPins.sound full_after_pins_canonical
    (rowsIncluded_self _) canonical one afterSatisfies

private theorem final_before_pin_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ∀ pin ∈ finalBeforeTrace.pins, assignment pin.1 = pin.2 := by
  have combined := final_pin_rows_satisfy assignment satisfied
  have beforeSatisfies :
      Satisfies (ConstantPins.rows finalBeforeTrace.pins) assignment := by
    intro row member
    exact combined row (List.mem_append_left _ member)
  exact ConstantPins.sound final_before_pins_canonical
    (rowsIncluded_self _) canonical one beforeSatisfies

private theorem final_after_pin_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ∀ pin ∈ finalAfterTrace.pins, assignment pin.1 = pin.2 := by
  have combined := final_pin_rows_satisfy assignment satisfied
  have afterSatisfies :
      Satisfies (ConstantPins.rows finalAfterTrace.pins) assignment := by
    intro row member
    exact combined row (List.mem_append_right _ member)
  exact ConstantPins.sound final_after_pins_canonical
    (rowsIncluded_self _) canonical one afterSatisfies

private theorem final_target_pin_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ∀ pin ∈ finalTargetTrace.pins, assignment pin.1 = pin.2 := by
  exact ConstantPins.sound final_target_pins_canonical
    (rowsIncluded_self _) canonical one
    (final_target_pin_rows_satisfy assignment satisfied)

private theorem source_call_accepted
    (arm : RawArm) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) (satisfied : arm.Satisfied assignment)
    (call : Poseidon2Call.Call) (member : call ∈ arm.poseidon2Calls) :
    TranscriptCertificate.CallAccepted call assignment := by
  apply Poseidon2PermutationSound.poseidon2Permutation_renamed_sound
    call.columnMap call.columnMap_zero canonical one
  exact satisfied.2.1 call member

theorem full_slice0_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    fullSlice0Trace.Accepted assignment := by
  constructor
  · simp [fullSlice0Trace]
  · intro call member
    apply source_call_accepted fullArtifact assignment canonical one satisfied
    exact fullCallsPart0_subset call (by simpa [fullSlice0Trace] using member)

theorem full_slice1_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    fullSlice1Trace.Accepted assignment := by
  constructor
  · simp [fullSlice1Trace]
  · intro call member
    apply source_call_accepted fullArtifact assignment canonical one satisfied
    exact fullCallsPart1_subset call (by simpa [fullSlice1Trace] using member)

theorem full_slice2_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    fullSlice2Trace.Accepted assignment := by
  constructor
  · simp [fullSlice2Trace]
  · intro call member
    apply source_call_accepted fullArtifact assignment canonical one satisfied
    exact fullCallsPart2_subset call (by simpa [fullSlice2Trace] using member)

theorem full_slice3_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    fullSlice3Trace.Accepted assignment := by
  constructor
  · simp [fullSlice3Trace]
  · intro call member
    apply source_call_accepted fullArtifact assignment canonical one satisfied
    exact fullCallsPart3_subset call (by simpa [fullSlice3Trace] using member)

theorem final_slice0_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    finalSlice0Trace.Accepted assignment := by
  constructor
  · simp [finalSlice0Trace]
  · intro call member
    apply source_call_accepted finalArtifact assignment canonical one satisfied
    exact finalCallsPart0_subset call (by simpa [finalSlice0Trace] using member)

theorem final_slice1_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    finalSlice1Trace.Accepted assignment := by
  constructor
  · simp [finalSlice1Trace]
  · intro call member
    apply source_call_accepted finalArtifact assignment canonical one satisfied
    exact finalCallsPart1_subset call (by simpa [finalSlice1Trace] using member)

theorem final_tail_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    finalTailTrace.Accepted assignment := by
  constructor
  · simp [finalTailTrace]
  · intro call member
    apply source_call_accepted finalArtifact assignment canonical one satisfied
    apply finalCallsPart2_subset call
    apply List.mem_of_mem_take
    simpa [finalTailTrace] using member

theorem full_before_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    fullBeforeTrace.Accepted assignment := by
  constructor
  · exact full_before_pin_facts assignment canonical one satisfied
  · intro call member
    apply source_call_accepted fullArtifact assignment canonical one satisfied
    apply fullCallsPart4_subset call
    apply List.mem_of_mem_take
    simpa [fullBeforeTrace] using member

theorem full_after_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    fullAfterTrace.Accepted assignment := by
  constructor
  · exact full_after_pin_facts assignment canonical one satisfied
  · intro call member
    apply source_call_accepted fullArtifact assignment canonical one satisfied
    apply fullCallsPart4_subset call
    apply List.mem_of_mem_drop
    apply List.mem_of_mem_take
    simpa [fullAfterTrace] using member

theorem final_before_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    finalBeforeTrace.Accepted assignment := by
  constructor
  · exact final_before_pin_facts assignment canonical one satisfied
  · intro call member
    apply source_call_accepted finalArtifact assignment canonical one satisfied
    apply finalCallsPart2_subset call
    apply List.mem_of_mem_drop
    apply List.mem_of_mem_take
    simpa [finalBeforeTrace] using member

theorem final_after_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    finalAfterTrace.Accepted assignment := by
  constructor
  · exact final_after_pin_facts assignment canonical one satisfied
  · intro call member
    apply source_call_accepted finalArtifact assignment canonical one satisfied
    apply finalCallsPart2_subset call
    apply List.mem_of_mem_drop
    apply List.mem_of_mem_take
    simpa [finalAfterTrace] using member

theorem final_target_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    finalTargetTrace.Accepted assignment := by
  constructor
  · exact final_target_pin_facts assignment canonical one satisfied
  · intro call member
    apply source_call_accepted finalArtifact assignment canonical one satisfied
    apply finalCallsPart2_subset call
    apply List.mem_of_mem_drop
    apply List.mem_of_mem_take
    simpa [finalTargetTrace] using member

theorem full_slice0_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical fullSlice0Start)
        fullSlice0Operations =
      ColumnReplay.decodeRun assignment canonical fullSlice0Result := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [fullSlice0Trace, ConstantPins.ValuesCanonical]) one
    (full_slice0_trace_accepted assignment canonical one satisfied)
  exact full_slice0_execution

theorem full_slice1_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical fullSlice1Start)
        fullSlice1Operations =
      ColumnReplay.decodeRun assignment canonical fullSlice1Result := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [fullSlice1Trace, ConstantPins.ValuesCanonical]) one
    (full_slice1_trace_accepted assignment canonical one satisfied)
  exact full_slice1_execution

theorem full_slice2_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical fullSlice2Start)
        fullSlice2Operations =
      ColumnReplay.decodeRun assignment canonical fullSlice2Result := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [fullSlice2Trace, ConstantPins.ValuesCanonical]) one
    (full_slice2_trace_accepted assignment canonical one satisfied)
  exact full_slice2_execution

theorem full_slice3_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical fullSlice3Start)
        fullSlice3Operations =
      ColumnReplay.decodeRun assignment canonical fullSlice3Result := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [fullSlice3Trace, ConstantPins.ValuesCanonical]) one
    (full_slice3_trace_accepted assignment canonical one satisfied)
  exact full_slice3_execution

theorem final_slice0_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical finalSlice0Start)
        finalSlice0Operations =
      ColumnReplay.decodeRun assignment canonical finalSlice0Result := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [finalSlice0Trace, ConstantPins.ValuesCanonical]) one
    (final_slice0_trace_accepted assignment canonical one satisfied)
  exact final_slice0_execution

theorem final_slice1_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical finalSlice1Start)
        finalSlice1Operations =
      ColumnReplay.decodeRun assignment canonical finalSlice1Result := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [finalSlice1Trace, ConstantPins.ValuesCanonical]) one
    (final_slice1_trace_accepted assignment canonical one satisfied)
  exact final_slice1_execution

theorem final_tail_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical finalTailStart)
        finalTailOperations =
      ColumnReplay.decodeRun assignment canonical finalTailResult := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [finalTailTrace, ConstantPins.ValuesCanonical]) one
    (final_tail_trace_accepted assignment canonical one satisfied)
  exact final_tail_execution

theorem full_before_state_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    ColumnReplay.semanticExecute assignment canonical
        (ColumnReplay.decodeRun assignment canonical fullBeforeStart)
        (stateOperations 1) =
      ColumnReplay.decodeRun assignment canonical fullBeforeResult := by
  apply ColumnReplay.execute_sound canonical full_before_pins_canonical one
    (full_before_trace_accepted assignment canonical one satisfied)
  exact full_before_state_execution

theorem full_after_state_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    ColumnReplay.semanticExecute assignment canonical
        (ColumnReplay.decodeRun assignment canonical fullAfterStart)
        (stateOperations 11) =
      ColumnReplay.decodeRun assignment canonical fullAfterResult := by
  apply ColumnReplay.execute_sound canonical full_after_pins_canonical one
    (full_after_trace_accepted assignment canonical one satisfied)
  exact full_after_state_execution

theorem final_before_state_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ColumnReplay.semanticExecute assignment canonical
        (ColumnReplay.decodeRun assignment canonical finalBeforeStart)
        (stateOperations 1) =
      ColumnReplay.decodeRun assignment canonical finalBeforeResult := by
  apply ColumnReplay.execute_sound canonical final_before_pins_canonical one
    (final_before_trace_accepted assignment canonical one satisfied)
  exact final_before_state_execution

theorem final_after_state_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ColumnReplay.semanticExecute assignment canonical
        (ColumnReplay.decodeRun assignment canonical finalAfterStart)
        (stateOperations 11) =
      ColumnReplay.decodeRun assignment canonical finalAfterResult := by
  apply ColumnReplay.execute_sound canonical final_after_pins_canonical one
    (final_after_trace_accepted assignment canonical one satisfied)
  exact final_after_state_execution

theorem final_target_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ColumnReplay.semanticExecute assignment canonical
        (ColumnReplay.decodeRun assignment canonical finalTargetStart)
        stateDigestOperations =
      ColumnReplay.decodeRun assignment canonical finalTargetResult := by
  apply ColumnReplay.execute_sound canonical final_target_pins_canonical one
    (final_target_trace_accepted assignment canonical one satisfied)
  exact final_target_execution

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayArtifact
