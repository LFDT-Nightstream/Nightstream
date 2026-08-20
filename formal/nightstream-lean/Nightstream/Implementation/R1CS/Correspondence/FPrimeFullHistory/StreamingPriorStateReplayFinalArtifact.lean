import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPriorStateReplayTransitionArtifact

/-!
Exact retained-row proof of the prior-state replay final checks.

Owns the 502 zero-padding rows, the final replay and program cursors, and the
four target-digest equalities. It proves `FinalChecks .final` for the explicit
target columns. It does not make those columns verifier-owned; lifecycle
composition must bind them to the running-instance prior-state digest.

Assurance tier: artifact-checked.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayFinalArtifact

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayDigestExecutionCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayTransitionArtifact
open Nightstream.Implementation.R1CS.PiRlcChallenge
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachineDuplex
open Nightstream.Implementation.R1CS.Program

def zeroPins (start count : Nat) : List (Nat × Nat) :=
  (List.range' start count).map fun column => (column, 0)

def padding0 : List IndexedRow :=
  (finalResidualRows0Part0.drop 16).take 64

def padding1 : List IndexedRow :=
  (finalResidualRows0Part0.drop 80).take 64

def padding2 : List IndexedRow :=
  (finalResidualRows0Part0.drop 144).take 64

def padding3 : List IndexedRow :=
  (finalResidualRows0Part0.drop 208).take 48

def padding4 : List IndexedRow :=
  finalResidualRows0Part1.take 64

def padding5 : List IndexedRow :=
  (finalResidualRows0Part1.drop 64).take 64

def padding6 : List IndexedRow :=
  (finalResidualRows0Part1.drop 128).take 64

def padding7 : List IndexedRow :=
  (finalResidualRows0Part1.drop 192).take 64

def padding8 : List IndexedRow :=
  finalResidualRows0Part2.take 6

theorem padding0_exact :
    padding0.map IndexedRow.row = ConstantPins.rows (zeroPins 681 64) := by
  rfl

theorem padding1_exact :
    padding1.map IndexedRow.row = ConstantPins.rows (zeroPins 745 64) := by
  rfl

theorem padding2_exact :
    padding2.map IndexedRow.row = ConstantPins.rows (zeroPins 809 64) := by
  rfl

theorem padding3_exact :
    padding3.map IndexedRow.row = ConstantPins.rows (zeroPins 873 48) := by
  rfl

theorem padding4_exact :
    padding4.map IndexedRow.row = ConstantPins.rows (zeroPins 921 64) := by
  rfl

theorem padding5_exact :
    padding5.map IndexedRow.row = ConstantPins.rows (zeroPins 985 64) := by
  rfl

theorem padding6_exact :
    padding6.map IndexedRow.row = ConstantPins.rows (zeroPins 1049 64) := by
  rfl

theorem padding7_exact :
    padding7.map IndexedRow.row = ConstantPins.rows (zeroPins 1113 64) := by
  rfl

theorem padding8_exact :
    padding8.map IndexedRow.row = ConstantPins.rows (zeroPins 1177 6) := by
  rfl

def paddingColumns : List Nat :=
  ((((((((List.range' 681 64 ++ List.range' 745 64) ++
      List.range' 809 64) ++ List.range' 873 48) ++
      List.range' 921 64) ++ List.range' 985 64) ++
      List.range' 1049 64) ++ List.range' 1113 64) ++
      List.range' 1177 6)

def paddingPins : List (Nat × Nat) :=
  paddingColumns.map fun column => (column, 0)

def paddingRows : List Row :=
  ((((((((ConstantPins.rows (zeroPins 681 64) ++
      ConstantPins.rows (zeroPins 745 64)) ++
      ConstantPins.rows (zeroPins 809 64)) ++
      ConstantPins.rows (zeroPins 873 48)) ++
      ConstantPins.rows (zeroPins 921 64)) ++
      ConstantPins.rows (zeroPins 985 64)) ++
      ConstantPins.rows (zeroPins 1049 64)) ++
      ConstantPins.rows (zeroPins 1113 64)) ++
      ConstantPins.rows (zeroPins 1177 6))

theorem paddingColumns_exact :
    paddingColumns = List.range' 681 502 := by
  unfold paddingColumns
  rw [List.range'_append_1, List.range'_append_1,
    List.range'_append_1, List.range'_append_1,
    List.range'_append_1, List.range'_append_1,
    List.range'_append_1, List.range'_append_1]

theorem paddingRows_eq_pins :
    paddingRows = ConstantPins.rows paddingPins := by
  simp only [paddingRows, paddingPins, paddingColumns, zeroPins,
    ConstantPins.rows, List.map_append, List.map_map]

private theorem part0_slice_satisfies
    (assignment : Nat → Nat)
    (satisfied : finalArtifact.Satisfied assignment)
    (drop count : Nat) (semanticRows : List Row)
    (exactRows :
      ((finalResidualRows0Part0.drop drop).take count).map IndexedRow.row =
        semanticRows) :
    Satisfies semanticRows assignment := by
  intro row member
  rw [← exactRows] at member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact satisfied.2.2 indexed
    (finalResidualRows0Part0_subset indexed
      (List.mem_of_mem_drop (List.mem_of_mem_take indexedMember)))

private theorem part1_slice_satisfies
    (assignment : Nat → Nat)
    (satisfied : finalArtifact.Satisfied assignment)
    (drop count : Nat) (semanticRows : List Row)
    (exactRows :
      ((finalResidualRows0Part1.drop drop).take count).map IndexedRow.row =
        semanticRows) :
    Satisfies semanticRows assignment := by
  intro row member
  rw [← exactRows] at member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact satisfied.2.2 indexed
    (finalResidualRows0Part1_subset indexed
      (List.mem_of_mem_drop (List.mem_of_mem_take indexedMember)))

private theorem part2_slice_satisfies
    (assignment : Nat → Nat)
    (satisfied : finalArtifact.Satisfied assignment)
    (drop count : Nat) (semanticRows : List Row)
    (exactRows :
      ((finalResidualRows0Part2.drop drop).take count).map IndexedRow.row =
        semanticRows) :
    Satisfies semanticRows assignment := by
  intro row member
  rw [← exactRows] at member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact satisfied.2.2 indexed
    (finalResidualRows0Part2_subset indexed
      (List.mem_of_mem_drop (List.mem_of_mem_take indexedMember)))

private theorem satisfies_append
    {assignment : Nat → Nat} {left right : List Row}
    (leftSatisfied : Satisfies left assignment)
    (rightSatisfied : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  rcases List.mem_append.mp member with member | member
  · exact leftSatisfied row member
  · exact rightSatisfied row member

private theorem padding_rows_satisfy
    (assignment : Nat → Nat)
    (satisfied : finalArtifact.Satisfied assignment) :
    Satisfies (ConstantPins.rows paddingPins) assignment := by
  have s0 := part0_slice_satisfies assignment satisfied 16 64 _ padding0_exact
  have s1 := part0_slice_satisfies assignment satisfied 80 64 _ padding1_exact
  have s2 := part0_slice_satisfies assignment satisfied 144 64 _ padding2_exact
  have s3 := part0_slice_satisfies assignment satisfied 208 48 _ padding3_exact
  have s4 := part1_slice_satisfies assignment satisfied 0 64 _ padding4_exact
  have s5 := part1_slice_satisfies assignment satisfied 64 64 _ padding5_exact
  have s6 := part1_slice_satisfies assignment satisfied 128 64 _ padding6_exact
  have s7 := part1_slice_satisfies assignment satisfied 192 64 _ padding7_exact
  have s8 := part2_slice_satisfies assignment satisfied 0 6 _ padding8_exact
  rw [← paddingRows_eq_pins]
  exact satisfies_append
    (satisfies_append
      (satisfies_append
        (satisfies_append
          (satisfies_append
            (satisfies_append
              (satisfies_append
                (satisfies_append s0 s1) s2) s3) s4) s5) s6) s7) s8

private theorem rowsIncluded_self (rows : List Row) :
    rowsIncluded rows rows = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true member

theorem paddingPins_canonical : ConstantPins.ValuesCanonical paddingPins := by
  intro pin member
  rcases List.mem_map.mp member with ⟨column, _, rfl⟩
  change 0 < goldilocksP
  decide

private theorem map_eq_replicate_of_pointwise
    {Alpha Beta : Type} (items : List Alpha) (value : Alpha → Beta)
    (constant : Beta)
    (pointwise : ∀ item ∈ items, value item = constant) :
    items.map value = List.replicate items.length constant := by
  induction items with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.length_cons, List.replicate_succ,
        List.cons.injEq]
      exact ⟨pointwise head (by simp),
        inductionHypothesis (fun item member =>
          pointwise item (by simp [member]))⟩

/-- Exact Rust padding rows force all 502 inactive final-chunk fields to
zero. Coverage is composed from nine leaves of at most 64 rows. -/
theorem final_padding_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ((List.range' 159 1024).map assignment).drop finalFields =
      List.replicate (chunkWidth - finalFields) 0 := by
  have pinFacts := ConstantPins.sound paddingPins_canonical
    (rowsIncluded_self _) canonical one
    (padding_rows_satisfy assignment satisfied)
  have allZero :
      ∀ column ∈ List.range' 681 502, assignment column = 0 := by
    intro column member
    apply pinFacts (column, 0)
    rw [← paddingColumns_exact] at member
    exact List.mem_map.mpr ⟨column, member, rfl⟩
  rw [← List.map_drop, List.drop_range']
  change (List.range' 681 502).map assignment = List.replicate 502 0
  exact map_eq_replicate_of_pointwise _ assignment 0 allZero

def boundaryRows : List Row :=
  [artifactLinearRow 10 [(0, 95232)],
    artifactLinearRow 21 [(0, 94)],
    artifactLinearRow 20 [(0, 95754)]]

theorem boundary_rows_exact :
    ((finalResidualRows0Part0.drop 13).take 3).map IndexedRow.row =
      boundaryRows := by
  rfl

theorem boundary_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    assignment 10 = 95232 ∧ assignment 21 = 94 ∧
      assignment 20 = 95754 := by
  have rows := part0_slice_satisfies assignment satisfied 13 3 _
    boundary_rows_exact
  have first : RowHolds assignment (artifactLinearRow 10 [(0, 95232)]) :=
    rows _ (by simp [boundaryRows])
  have second : RowHolds assignment (artifactLinearRow 21 [(0, 94)]) :=
    rows _ (by simp [boundaryRows])
  have third : RowHolds assignment (artifactLinearRow 20 [(0, 95754)]) :=
    rows _ (by simp [boundaryRows])
  refine ⟨?_, ?_, ?_⟩
  · simpa [lcEval, one] using artifact_linear_row_sound assignment canonical
      one 10 [(0, 95232)] (by simp [CanonicalTerms]; decide) first
  · simpa [lcEval, one] using artifact_linear_row_sound assignment canonical
      one 21 [(0, 94)] (by simp [CanonicalTerms]; decide) second
  · simpa [lcEval, one] using artifact_linear_row_sound assignment canonical
      one 20 [(0, 95754)] (by simp [CanonicalTerms]; decide) third

def targetPairs : List (Nat × Nat) :=
  List.ofFn fun lane : Fin 4 => (155 + lane.val, 79776 + lane.val)

theorem target_rows_exact :
    ((finalResidualRows0Part2.drop 9).take 4).map IndexedRow.row =
      EqualityPins.rows targetPairs := by
  rfl

theorem target_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ∀ lane : Fin 4,
      assignment (155 + lane.val) = assignment (79776 + lane.val) := by
  have rows := part2_slice_satisfies assignment satisfied 9 4 _
    target_rows_exact
  have facts := EqualityPins.rows_sound canonical one rows
  intro lane
  exact facts _ (List.mem_ofFn.mpr ⟨lane, rfl⟩)

private theorem duplexStateExt
    {left right : Poseidon2Duplex.State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

private theorem final_target_start_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (absorbed :
      (replayStateAt assignment 11).transcript.absorbed = 2) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical finalTargetStart).state =
      (replayStateAt assignment 11).transcript := by
  apply duplexStateExt
  · funext lane
    rfl
  · exact absorbed.symm

theorem final_target_output_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    ∀ lane : Fin 4,
      assignment (79776 + lane.val) =
        outputDigest
          (Poseidon2Duplex.gate ProductPoseidon2.constants
            (replayStateAt assignment 11).transcript) lane := by
  intro lane
  let zeroDigest : Fin 4 → TranscriptMachine.Field := fun _ => 0
  have refined := final_target_refines assignment canonical one satisfied
  have decodedLane := congrArg
    (fun run => ((run.digests.getD 0 zeroDigest) lane).val) refined
  have machineLane :
      ((TranscriptMachine.digest
        (ColumnReplay.decodeRun assignment canonical finalTargetStart).state
        ).2 lane).val = assignment (79776 + lane.val) := by
    simpa [stateDigestOperations, ColumnReplay.semanticExecute,
      ColumnReplay.semanticStep, ColumnReplay.decodeRun, finalTargetStart,
      finalTargetResult, checkpointRun, ColumnReplay.decodeDigest,
      zeroDigest] using decodedLane
  have gateLane := digest_output_toDuplex
    (ColumnReplay.decodeRun assignment canonical finalTargetStart).state lane
  have startExact := final_target_start_exact assignment canonical
    (final_after_absorbed assignment canonical one satisfied)
  calc
    assignment (79776 + lane.val) =
        ((TranscriptMachine.digest
          (ColumnReplay.decodeRun assignment canonical finalTargetStart).state
          ).2 lane).val := machineLane.symm
    _ = (Poseidon2Duplex.gate Poseidon2CanonicalConstants.selected
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical finalTargetStart
              ).state)).lanes ⟨lane.val, by
            have laneLt := lane.isLt
            change lane.val < 8
            omega⟩ := gateLane
    _ = (Poseidon2Duplex.gate ProductPoseidon2.constants
          (replayStateAt assignment 11).transcript).lanes
          ⟨lane.val, by
            have laneLt := lane.isLt
            change lane.val < 8
            omega⟩ := by
      rw [startExact]
      rfl
    _ = outputDigest
          (Poseidon2Duplex.gate ProductPoseidon2.constants
            (replayStateAt assignment 11).transcript) lane := by
      rfl

def finalChunk (assignment : Nat → Nat) : List Nat :=
  (List.range' 159 1024).map assignment

def targetDigestAt (assignment : Nat → Nat) : Digest :=
  fun lane => assignment (155 + lane.val)

/-- The exact retained final-arm rows imply the complete typed final checks.
The target remains an explicit source value until the lifecycle bridge binds
it to verifier-owned running-instance data. -/
theorem final_rows_imply_finalChecks
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    FinalChecks .final (replayStateAt assignment 11)
      (finalChunk assignment) (targetDigestAt assignment) := by
  simp only [FinalChecks]
  refine ⟨?_, ?_, ?_⟩
  · exact final_padding_exact assignment canonical one satisfied
  · change assignment 20 = frameFields
    simpa [frameFields, fullChunks, chunkWidth, finalFields] using
      (boundary_facts assignment canonical one satisfied).2.2
  · funext lane
    change outputDigest
        (Poseidon2Duplex.gate ProductPoseidon2.constants
          (replayStateAt assignment 11).transcript) lane =
      assignment (155 + lane.val)
    have targetToOutput :
        assignment (155 + lane.val) =
          outputDigest
            (Poseidon2Duplex.gate ProductPoseidon2.constants
              (replayStateAt assignment 11).transcript) lane :=
      Eq.trans (target_facts assignment canonical one satisfied lane)
        (final_target_output_exact assignment canonical one satisfied lane)
    exact targetToOutput.symm

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayFinalArtifact
