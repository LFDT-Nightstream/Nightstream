import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPriorStateReplaySliceComposition

/-!
Exact retained-row proof of the prior-state replay transcript transition.

Owns the two absorbed-cursor rows, replay and program-cursor rows, and eight
output-lane rows for each source arm. These rows connect the public ten-field
replay-state layout to the exact physical Poseidon2 replay. It does not own
schedule cursor authority, final padding, or final target-digest authority.

Assurance tier: artifact-checked.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayTransitionArtifact

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFullClaimStreaming
open Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySliceComposition
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayTransitionExecutionCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachineDuplex
open Nightstream.Implementation.R1CS.Program

def artifactLinearRow (output : Nat) (terms : List (Nat × Nat)) : Row :=
  ⟨negateTerms terms ++ [(output, 1)], [(0, 1)], []⟩

def fullScalarRows : List Row :=
  [artifactLinearRow 9 [], artifactLinearRow 19 [],
    artifactLinearRow 20 [(0, 1024), (10, 1)]]

def finalScalarRows : List Row :=
  [artifactLinearRow 9 [], artifactLinearRow 19 [(0, 2)],
    artifactLinearRow 20 [(0, 522), (10, 1)]]

def lanePairs (declaredStart : Nat) (result : ColumnReplay.Run) :
    List (Nat × Nat) :=
  List.ofFn fun lane : Fin 8 =>
    (declaredStart + lane.val, result.cursor.lanes lane)

def fullLanePairs : List (Nat × Nat) :=
  lanePairs 11 fullSlice3Result

def finalLanePairs : List (Nat × Nat) :=
  lanePairs 11 finalTailResult

def fullSemanticTransitionRows : List Row :=
  fullScalarRows ++ EqualityPins.rows fullLanePairs

def finalSemanticTransitionRows : List Row :=
  finalScalarRows ++ EqualityPins.rows finalLanePairs

/-- Select only the retained rows that own the typed transition. The omitted
rows own program-cursor alignment and later digest stages. -/
def fullTransitionIndexedRows : List IndexedRow :=
  fullResidualRows0Part0.take 2 ++
    (fullResidualRows0Part0.drop 3).take 1 ++
      (fullResidualRows0Part0.drop 5).take 8

def finalTransitionIndexedRows : List IndexedRow :=
  finalResidualRows0Part0.take 2 ++
    (finalResidualRows0Part0.drop 3).take 1 ++
      (finalResidualRows0Part0.drop 5).take 8

/-- Exact emitted row for `frameCursor = 1024 * (programCursor - 1)`. -/
def cursorAlignmentRow : Row :=
  ⟨[(0, 1024), (10, 1), (21, 18446744069414583297)], [(0, 1)], []⟩

/-- Exact emitted row for `afterProgramCursor = beforeProgramCursor + 1`. -/
def programCursorAdvanceRow : Row :=
  ⟨[(0, 18446744069414584320), (21, 18446744069414584320), (88, 1)],
    [(0, 1)], []⟩

def cursorRows : List Row :=
  [cursorAlignmentRow, programCursorAdvanceRow]

def fullCursorIndexedRows : List IndexedRow :=
  (fullResidualRows0Part0.drop 2).take 1 ++
    (fullResidualRows0Part0.drop 4).take 1

def finalCursorIndexedRows : List IndexedRow :=
  (finalResidualRows0Part0.drop 2).take 1 ++
    (finalResidualRows0Part0.drop 4).take 1

/-- Exact Rust row identity for the eleven full-arm transition rows. -/
theorem full_transition_rows_exact :
    fullTransitionIndexedRows.map IndexedRow.row =
      fullSemanticTransitionRows := by
  rfl

/-- Exact Rust row identity for the eleven final-arm transition rows. -/
theorem final_transition_rows_exact :
    finalTransitionIndexedRows.map IndexedRow.row =
      finalSemanticTransitionRows := by
  rfl

/-- Exact Rust row identity for the two full-arm cursor rows. -/
theorem full_cursor_rows_exact :
    fullCursorIndexedRows.map IndexedRow.row = cursorRows := by
  rfl

/-- Exact Rust row identity for the two final-arm cursor rows. -/
theorem final_cursor_rows_exact :
    finalCursorIndexedRows.map IndexedRow.row = cursorRows := by
  rfl

private theorem full_transition_rows_subset :
    ∀ indexed ∈ fullTransitionIndexedRows,
      indexed ∈ fullArtifact.residualRows := by
  intro indexed member
  simp only [fullTransitionIndexedRows, List.mem_append] at member
  rcases member with (first | middle) | last
  · exact fullResidualRows0Part0_subset indexed
      (List.mem_of_mem_take first)
  · exact fullResidualRows0Part0_subset indexed
      (List.mem_of_mem_drop (List.mem_of_mem_take middle))
  · exact fullResidualRows0Part0_subset indexed
      (List.mem_of_mem_drop (List.mem_of_mem_take last))

private theorem final_transition_rows_subset :
    ∀ indexed ∈ finalTransitionIndexedRows,
      indexed ∈ finalArtifact.residualRows := by
  intro indexed member
  simp only [finalTransitionIndexedRows, List.mem_append] at member
  rcases member with (first | middle) | last
  · exact finalResidualRows0Part0_subset indexed
      (List.mem_of_mem_take first)
  · exact finalResidualRows0Part0_subset indexed
      (List.mem_of_mem_drop (List.mem_of_mem_take middle))
  · exact finalResidualRows0Part0_subset indexed
      (List.mem_of_mem_drop (List.mem_of_mem_take last))

private theorem full_cursor_rows_subset :
    ∀ indexed ∈ fullCursorIndexedRows,
      indexed ∈ fullArtifact.residualRows := by
  intro indexed member
  simp only [fullCursorIndexedRows, List.mem_append] at member
  rcases member with first | second
  · exact fullResidualRows0Part0_subset indexed
      (List.mem_of_mem_drop (List.mem_of_mem_take first))
  · exact fullResidualRows0Part0_subset indexed
      (List.mem_of_mem_drop (List.mem_of_mem_take second))

private theorem final_cursor_rows_subset :
    ∀ indexed ∈ finalCursorIndexedRows,
      indexed ∈ finalArtifact.residualRows := by
  intro indexed member
  simp only [finalCursorIndexedRows, List.mem_append] at member
  rcases member with first | second
  · exact finalResidualRows0Part0_subset indexed
      (List.mem_of_mem_drop (List.mem_of_mem_take first))
  · exact finalResidualRows0Part0_subset indexed
      (List.mem_of_mem_drop (List.mem_of_mem_take second))

private theorem full_semantic_transition_satisfies
    (assignment : Nat → Nat)
    (satisfied : fullArtifact.Satisfied assignment) :
    Satisfies fullSemanticTransitionRows assignment := by
  intro row member
  rw [← full_transition_rows_exact] at member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact satisfied.2.2 indexed
    (full_transition_rows_subset indexed indexedMember)

private theorem final_semantic_transition_satisfies
    (assignment : Nat → Nat)
    (satisfied : finalArtifact.Satisfied assignment) :
    Satisfies finalSemanticTransitionRows assignment := by
  intro row member
  rw [← final_transition_rows_exact] at member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact satisfied.2.2 indexed
    (final_transition_rows_subset indexed indexedMember)

private theorem full_cursor_rows_satisfy
    (assignment : Nat → Nat)
    (satisfied : fullArtifact.Satisfied assignment) :
    Satisfies cursorRows assignment := by
  intro row member
  rw [← full_cursor_rows_exact] at member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact satisfied.2.2 indexed
    (full_cursor_rows_subset indexed indexedMember)

private theorem final_cursor_rows_satisfy
    (assignment : Nat → Nat)
    (satisfied : finalArtifact.Satisfied assignment) :
    Satisfies cursorRows assignment := by
  intro row member
  rw [← final_cursor_rows_exact] at member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact satisfied.2.2 indexed
    (final_cursor_rows_subset indexed indexedMember)

private theorem rowHolds_of_operand_perms
    (assignment : Nat → Nat) {source target : Row}
    (a : source.a.Perm target.a)
    (b : source.b.Perm target.b)
    (c : source.c.Perm target.c)
    (holds : RowHolds assignment source) :
    RowHolds assignment target := by
  unfold RowHolds at holds ⊢
  calc
    lcEval assignment target.a * lcEval assignment target.b % goldilocksP =
        lcEval assignment source.a * lcEval assignment source.b %
          goldilocksP := by
      rw [Program.lcEval_eq_of_perm assignment a,
        Program.lcEval_eq_of_perm assignment b]
    _ = lcEval assignment source.c := holds
    _ = lcEval assignment target.c :=
      Program.lcEval_eq_of_perm assignment c

private theorem artifact_linear_row_a_perm
    (output : Nat) (terms : List (Nat × Nat)) :
    (artifactLinearRow output terms).a.Perm
      (builderLinearRow output terms).a := by
  simpa [artifactLinearRow, builderLinearRow] using
    (List.perm_append_comm : List.Perm
      (negateTerms terms ++ [(output, 1)])
      ([(output, 1)] ++ negateTerms terms))

theorem artifact_linear_row_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (output : Nat) (terms : List (Nat × Nat))
    (termsCanonical : CanonicalTerms terms)
    (holds : RowHolds assignment (artifactLinearRow output terms)) :
    assignment output = lcEval assignment terms := by
  have builderHolds := rowHolds_of_operand_perms assignment
    (artifact_linear_row_a_perm output terms) (List.Perm.refl _)
    (List.Perm.refl _) holds
  exact builderLinearRow_sound canonical one output terms termsCanonical
    builderHolds

private theorem cursor_alignment_row_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : RowHolds assignment cursorAlignmentRow) :
    assignment 10 = lcEval assignment
      [(21, 1024), (0, 18446744069414583297)] := by
  have artifactHolds : RowHolds assignment
      (artifactLinearRow 10
        [(21, 1024), (0, 18446744069414583297)]) :=
    rowHolds_of_operand_perms assignment
      (source := cursorAlignmentRow)
      (target := artifactLinearRow 10
        [(21, 1024), (0, 18446744069414583297)])
      (by decide) (List.Perm.refl _) (List.Perm.refl _) holds
  exact artifact_linear_row_sound assignment canonical one 10
    [(21, 1024), (0, 18446744069414583297)]
    (by simp [CanonicalTerms, goldilocksP]) artifactHolds

private theorem program_cursor_advance_row_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : RowHolds assignment programCursorAdvanceRow) :
    assignment 88 = lcEval assignment [(0, 1), (21, 1)] := by
  have artifactHolds : RowHolds assignment
      (artifactLinearRow 88 [(0, 1), (21, 1)]) := by
    simpa [programCursorAdvanceRow, artifactLinearRow, negateTerms,
      goldilocksP] using holds
  exact artifact_linear_row_sound assignment canonical one 88
    [(0, 1), (21, 1)] (by simp [CanonicalTerms]; decide) artifactHolds

/-- Full-arm rows derive both source cursor equations. -/
theorem full_cursor_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    assignment 10 = lcEval assignment
        [(21, 1024), (0, 18446744069414583297)] ∧
      assignment 88 = lcEval assignment [(0, 1), (21, 1)] := by
  have rows := full_cursor_rows_satisfy assignment satisfied
  exact ⟨cursor_alignment_row_sound assignment canonical one
      (rows cursorAlignmentRow (by simp [cursorRows])),
    program_cursor_advance_row_sound assignment canonical one
      (rows programCursorAdvanceRow (by simp [cursorRows]))⟩

/-- Final-arm rows derive both source cursor equations. -/
theorem final_cursor_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    assignment 10 = lcEval assignment
        [(21, 1024), (0, 18446744069414583297)] ∧
      assignment 88 = lcEval assignment [(0, 1), (21, 1)] := by
  have rows := final_cursor_rows_satisfy assignment satisfied
  exact ⟨cursor_alignment_row_sound assignment canonical one
      (rows cursorAlignmentRow (by simp [cursorRows])),
    program_cursor_advance_row_sound assignment canonical one
      (rows programCursorAdvanceRow (by simp [cursorRows]))⟩

private theorem full_scalar_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    assignment 9 = 0 ∧ assignment 19 = 0 ∧
      assignment 20 = lcEval assignment [(0, 1024), (10, 1)] := by
  have rows := full_semantic_transition_satisfies assignment satisfied
  have first : RowHolds assignment (artifactLinearRow 9 []) :=
    rows _ (by simp [fullSemanticTransitionRows, fullScalarRows])
  have second : RowHolds assignment (artifactLinearRow 19 []) :=
    rows _ (by simp [fullSemanticTransitionRows, fullScalarRows])
  have third : RowHolds assignment
      (artifactLinearRow 20 [(0, 1024), (10, 1)]) :=
    rows _ (by simp [fullSemanticTransitionRows, fullScalarRows])
  refine ⟨?_, ?_, ?_⟩
  · simpa [lcEval] using artifact_linear_row_sound assignment canonical
      one 9 [] (by simp [CanonicalTerms]) first
  · simpa [lcEval] using artifact_linear_row_sound assignment canonical
      one 19 [] (by simp [CanonicalTerms]) second
  · exact artifact_linear_row_sound assignment canonical one 20
      [(0, 1024), (10, 1)] (by simp [CanonicalTerms]; decide) third

private theorem final_scalar_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    assignment 9 = 0 ∧ assignment 19 = 2 ∧
      assignment 20 = lcEval assignment [(0, 522), (10, 1)] := by
  have rows := final_semantic_transition_satisfies assignment satisfied
  have first : RowHolds assignment (artifactLinearRow 9 []) :=
    rows _ (by simp [finalSemanticTransitionRows, finalScalarRows])
  have second : RowHolds assignment (artifactLinearRow 19 [(0, 2)]) :=
    rows _ (by simp [finalSemanticTransitionRows, finalScalarRows])
  have third : RowHolds assignment
      (artifactLinearRow 20 [(0, 522), (10, 1)]) :=
    rows _ (by simp [finalSemanticTransitionRows, finalScalarRows])
  refine ⟨?_, ?_, ?_⟩
  · simpa [lcEval] using artifact_linear_row_sound assignment canonical
      one 9 [] (by simp [CanonicalTerms]) first
  · simpa [lcEval, one] using artifact_linear_row_sound assignment canonical
      one 19 [(0, 2)] (by simp [CanonicalTerms]; decide) second
  · exact artifact_linear_row_sound assignment canonical one 20
      [(0, 522), (10, 1)] (by simp [CanonicalTerms]; decide) third

private theorem full_lane_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    ∀ lane : Fin 8,
      assignment (11 + lane.val) =
        assignment (fullSlice3Result.cursor.lanes lane) := by
  have rows := full_semantic_transition_satisfies assignment satisfied
  have laneRows : Satisfies (EqualityPins.rows fullLanePairs) assignment := by
    intro row member
    exact rows row (List.mem_append_right _ member)
  have facts := EqualityPins.rows_sound canonical one laneRows
  intro lane
  exact facts _ (List.mem_ofFn.mpr ⟨lane, rfl⟩)

private theorem final_lane_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ∀ lane : Fin 8,
      assignment (11 + lane.val) =
        assignment (finalTailResult.cursor.lanes lane) := by
  have rows := final_semantic_transition_satisfies assignment satisfied
  have laneRows : Satisfies (EqualityPins.rows finalLanePairs) assignment := by
    intro row member
    exact rows row (List.mem_append_right _ member)
  have facts := EqualityPins.rows_sound canonical one laneRows
  intro lane
  exact facts _ (List.mem_ofFn.mpr ⟨lane, rfl⟩)

/-- Source-coordinate interpretation of one ten-field replay state. -/
def replayStateAt (assignment : Nat → Nat) (start : Nat) :
    ProductionSuccessorStateStreaming.ReplayState where
  transcript := {
    lanes := fun lane => assignment (start + lane.val)
    absorbed := assignment (start + 8)
  }
  cursor := assignment (start + 9)

/-- Full-arm rows fix the pre-replay absorb cursor. -/
theorem full_before_absorbed
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    (replayStateAt assignment 1).transcript.absorbed = 0 :=
  (full_scalar_facts assignment canonical one satisfied).1

/-- Final-arm rows fix the pre-replay absorb cursor. -/
theorem final_before_absorbed
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    (replayStateAt assignment 1).transcript.absorbed = 0 :=
  (final_scalar_facts assignment canonical one satisfied).1

def fullSlices (assignment : Nat → Nat) : List (List Nat) :=
  [fullSlice0Columns.map assignment, fullSlice1Columns.map assignment,
    fullSlice2Columns.map assignment, fullSlice3Columns.map assignment]

def finalSlices (assignment : Nat → Nat) : List (List Nat) :=
  [finalSlice0Columns.map assignment, finalSlice1Columns.map assignment,
    finalTailColumns.map assignment]

@[simp] theorem fullSlices_flatten_length (assignment : Nat → Nat) :
    (fullSlices assignment).flatten.length = 1024 := by
  simp [fullSlices, fullSlice0Columns, fullSlice1Columns,
    fullSlice2Columns, fullSlice3Columns]

@[simp] theorem finalSlices_flatten_length (assignment : Nat → Nat) :
    (finalSlices assignment).flatten.length = 522 := by
  simp [finalSlices, finalSlice0Columns, finalSlice1Columns,
    finalTailColumns]

/-- The four full-arm source slices are one contiguous 1,024-field chunk. -/
theorem fullSlices_flatten_eq_chunk (assignment : Nat → Nat) :
    (fullSlices assignment).flatten =
      (List.range' 155 1024).map assignment := by
  unfold fullSlices fullSlice0Columns fullSlice1Columns fullSlice2Columns
    fullSlice3Columns
  simp only [List.flatten_cons, List.flatten_nil, List.append_nil]
  repeat rw [← List.map_append]
  repeat rw [List.range'_append]

/-- The final-arm source slices are the contiguous 522 active fields. -/
theorem finalSlices_flatten_eq_activeChunk (assignment : Nat → Nat) :
    (finalSlices assignment).flatten =
      (List.range' 159 522).map assignment := by
  unfold finalSlices finalSlice0Columns finalSlice1Columns finalTailColumns
  simp only [List.flatten_cons, List.flatten_nil, List.append_nil]
  repeat rw [← List.map_append]
  repeat rw [List.range'_append]

private theorem duplexStateExt
    {left right : Poseidon2Duplex.State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

private theorem full_start_state_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (absorbed : assignment 9 = 0) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state =
      (replayStateAt assignment 1).transcript := by
  apply duplexStateExt
  · funext lane
    rfl
  · exact absorbed.symm

private theorem final_start_state_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (absorbed : assignment 9 = 0) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical finalSlice0Start).state =
      (replayStateAt assignment 1).transcript := by
  apply duplexStateExt
  · funext lane
    rfl
  · exact absorbed.symm

private theorem full_result_state_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (absorbed : assignment 19 = 0)
    (lanes : ∀ lane : Fin 8,
      assignment (11 + lane.val) =
        assignment (fullSlice3Result.cursor.lanes lane)) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice3Result).state =
      (replayStateAt assignment 11).transcript := by
  apply duplexStateExt
  · funext lane
    exact (lanes lane).symm
  · exact absorbed.symm

private theorem final_result_state_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (absorbed : assignment 19 = 2)
    (lanes : ∀ lane : Fin 8,
      assignment (11 + lane.val) =
        assignment (finalTailResult.cursor.lanes lane)) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical finalTailResult).state =
      (replayStateAt assignment 11).transcript := by
  apply duplexStateExt
  · funext lane
    exact (lanes lane).symm
  · exact absorbed.symm

/-- Exact full-arm rows imply the typed transcript transition on the exact
1,024 assignment values. -/
theorem full_transcript_transition
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    (replayStateAt assignment 11).transcript =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (fullSlices assignment).flatten
        (replayStateAt assignment 1).transcript := by
  have scalars := full_scalar_facts assignment canonical one satisfied
  have lanes := full_lane_facts assignment canonical one satisfied
  have startExact := full_start_state_exact assignment canonical scalars.1
  have resultExact := full_result_state_exact assignment canonical
    scalars.2.1 lanes
  have composed := full_eq_absorbSlices assignment canonical one satisfied
  have asChunks :
      toDuplex
          (ColumnReplay.decodeRun assignment canonical fullSlice3Result).state =
        absorbChunks (fullSlices assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              fullSlice0Start).state) := by
    simpa only [fullSlices, absorbChunks] using composed
  have flattened := absorbChunks_eq_absorbSlice_flatten
    (fullSlices assignment)
    (toDuplex
      (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state)
    (by
      change 0 < Poseidon2Sponge.rate
      decide)
  calc
    (replayStateAt assignment 11).transcript =
        toDuplex
          (ColumnReplay.decodeRun assignment canonical
            fullSlice3Result).state := resultExact.symm
    _ = absorbChunks (fullSlices assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              fullSlice0Start).state) := asChunks
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlices assignment).flatten
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              fullSlice0Start).state) := flattened
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlices assignment).flatten
          (replayStateAt assignment 1).transcript := by
      rw [startExact]

/-- Exact final-arm rows imply the typed transcript transition on the exact
522 active assignment values. -/
theorem final_transcript_transition
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    (replayStateAt assignment 11).transcript =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (finalSlices assignment).flatten
        (replayStateAt assignment 1).transcript := by
  have scalars := final_scalar_facts assignment canonical one satisfied
  have lanes := final_lane_facts assignment canonical one satisfied
  have startExact := final_start_state_exact assignment canonical scalars.1
  have resultExact := final_result_state_exact assignment canonical
    scalars.2.1 lanes
  have composed := final_eq_absorbSlices assignment canonical one satisfied
  have asChunks :
      toDuplex
          (ColumnReplay.decodeRun assignment canonical finalTailResult).state =
        absorbChunks (finalSlices assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              finalSlice0Start).state) := by
    simpa only [finalSlices, absorbChunks] using composed
  have flattened := absorbChunks_eq_absorbSlice_flatten
    (finalSlices assignment)
    (toDuplex
      (ColumnReplay.decodeRun assignment canonical finalSlice0Start).state)
    (by
      change 0 < Poseidon2Sponge.rate
      decide)
  calc
    (replayStateAt assignment 11).transcript =
        toDuplex
          (ColumnReplay.decodeRun assignment canonical
            finalTailResult).state := resultExact.symm
    _ = absorbChunks (finalSlices assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              finalSlice0Start).state) := asChunks
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalSlices assignment).flatten
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              finalSlice0Start).state) := flattened
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalSlices assignment).flatten
          (replayStateAt assignment 1).transcript := by
      rw [startExact]

theorem full_cursor_transition
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment)
    (noWrap : assignment 10 + 1024 < goldilocksP) :
    (replayStateAt assignment 11).cursor =
      (replayStateAt assignment 1).cursor +
        (fullSlices assignment).flatten.length := by
  change assignment 20 = assignment 10 +
    (fullSlices assignment).flatten.length
  rw [fullSlices_flatten_length]
  rw [(full_scalar_facts assignment canonical one satisfied).2.2]
  simp only [lcEval, List.foldl, one, Nat.mul_one, Nat.one_mul,
    Nat.zero_add]
  rw [Nat.add_comm 1024 (assignment 10)]
  exact Nat.mod_eq_of_lt noWrap

theorem final_cursor_transition
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment)
    (noWrap : assignment 10 + 522 < goldilocksP) :
    (replayStateAt assignment 11).cursor =
      (replayStateAt assignment 1).cursor +
        (finalSlices assignment).flatten.length := by
  change assignment 20 = assignment 10 +
    (finalSlices assignment).flatten.length
  rw [finalSlices_flatten_length]
  rw [(final_scalar_facts assignment canonical one satisfied).2.2]
  simp only [lcEval, List.foldl, one, Nat.mul_one, Nat.one_mul,
    Nat.zero_add]
  rw [Nat.add_comm 522 (assignment 10)]
  exact Nat.mod_eq_of_lt noWrap

/-- Exact full-arm rows advance the typed replay state on the exact 1,024
active fields. The no-wrap premise remains owned by the verifier schedule. -/
theorem full_replay_state_transition
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment)
    (noWrap : assignment 10 + 1024 < goldilocksP) :
    replayStateAt assignment 11 =
      (replayStateAt assignment 1).advance
        (fullSlices assignment).flatten :=
  ProductionSuccessorStateStreaming.ReplayState.eq_advance_of_transcript_cursor
    (replayStateAt assignment 1) (replayStateAt assignment 11)
    (fullSlices assignment).flatten
    (full_transcript_transition assignment canonical one satisfied)
    (full_cursor_transition assignment canonical one satisfied noWrap)

/-- Exact final-arm rows advance the typed replay state on the exact 522
active fields. The 502 padding fields are outside this theorem. -/
theorem final_replay_state_transition
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment)
    (noWrap : assignment 10 + 522 < goldilocksP) :
    replayStateAt assignment 11 =
      (replayStateAt assignment 1).advance
        (finalSlices assignment).flatten :=
  ProductionSuccessorStateStreaming.ReplayState.eq_advance_of_transcript_cursor
    (replayStateAt assignment 1) (replayStateAt assignment 11)
    (finalSlices assignment).flatten
    (final_transcript_transition assignment canonical one satisfied)
    (final_cursor_transition assignment canonical one satisfied noWrap)

/-- Exact final-arm rows fix the post-replay absorb cursor used by the target
digest call. -/
theorem final_after_absorbed
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    (replayStateAt assignment 11).transcript.absorbed = 2 :=
  (final_scalar_facts assignment canonical one satisfied).2.1

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayTransitionArtifact
