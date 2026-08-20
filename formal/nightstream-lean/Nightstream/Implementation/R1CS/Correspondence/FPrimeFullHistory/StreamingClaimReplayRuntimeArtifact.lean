import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayTransitionRowCertificate
import Nightstream.Implementation.Nebula.NIFS.Core.Poseidon2
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayStateCursorArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayTransitionExecutionCertificate
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.TranscriptMachineDuplex

/-!
Contract: exact generated-row refinement for the runtime Poseidon2 replay and
final-readiness fields of one production claim-replay phase.

Assurance tier: artifact-checked for the current Goldilocks `b = 2`,
`k_rho = 16` full and final arms. Rust-conformant status also requires the
separate current Rust-to-Lean drift test.

The verifier-owned chunk values remain an explicit premise. Generated rows
cannot select or authorize that frame. This module does not own coordinate
accumulation, cursor selection, collision reduction, or lifecycle selection.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayRuntimeArtifact

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayState
open Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayTransition
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExpectedCarryArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayTransitionExecutionCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayTransitionRowCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachineDuplex
open Nightstream.Implementation.R1CS.SeededPhi81RingRefinement
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

private theorem duplexState_ext
    {left right : Poseidon2Duplex.State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

@[simp] theorem transitionColumn_before_runtimeLane
    (kind : ArmKind) (lane : Fin spongeWidth) :
    transitionColumn kind
        (transitionIndex .before (runtimeLaneIndex lane)) =
      10 + lane.val := by
  rw [transitionColumn_eq_structural]
  fin_cases lane <;> rfl

@[simp] theorem transitionColumn_after_runtimeLane
    (kind : ArmKind) (lane : Fin spongeWidth) :
    transitionColumn kind
        (transitionIndex .after (runtimeLaneIndex lane)) =
      420 + lane.val := by
  rw [transitionColumn_eq_structural]
  fin_cases lane <;> rfl

@[simp] theorem transitionColumn_after_expectedLane
    (kind : ArmKind) (lane : Fin spongeWidth) :
    transitionColumn kind
        (transitionIndex .after (expectedLaneIndex lane)) =
      411 + lane.val := by
  let index : Fin 9 := ⟨lane.val, by
    have bound := lane.isLt
    unfold spongeWidth at bound
    omega⟩
  have wordIndex : expectedLaneIndex lane = expectedWordIndex index := by
    apply Fin.ext
    simp [expectedLaneIndex, expectedOffset, expectedWordIndex, index]
  rw [wordIndex, transitionColumn_after_expected]

/-! ## Exact output and readiness row semantics -/

private theorem full_output_rows_satisfy
    (assignment : Nat → Nat)
    (satisfied : fullArm.Satisfied assignment) :
    Satisfies fullOutputRows assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds fullArm assignment satisfied indexed
    (fullOutputIndexed_member indexedMember)

private theorem final_output_rows_satisfy
    (assignment : Nat → Nat)
    (satisfied : finalArm.Satisfied assignment) :
    Satisfies finalOutputRows assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds finalArm assignment satisfied indexed
    (finalOutputIndexed_member indexedMember)

private theorem final_readiness_rows_satisfy
    (assignment : Nat → Nat)
    (satisfied : finalArm.Satisfied assignment) :
    Satisfies finalReadinessRows assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds finalArm assignment satisfied indexed
    (finalReadinessIndexed_member indexedMember)

theorem full_output_lane
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment)
    (lane : Fin 8) :
    assignment (420 + lane.val) = assignment (155437 + lane.val) := by
  have facts := EqualityPins.rows_sound canonical one (by
    rw [← fullOutputRows_exact]
    exact full_output_rows_satisfy assignment satisfied)
  apply facts (420 + lane.val, 155437 + lane.val)
  exact List.mem_map.mpr
    ⟨lane.val, List.mem_range.mpr lane.isLt, rfl⟩

theorem final_output_lane
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment)
    (lane : Fin 8) :
    assignment (420 + lane.val) =
      assignment (if lane.val < 3 then 1393 + lane.val
        else 87637 + lane.val) := by
  have facts := EqualityPins.rows_sound canonical one (by
    rw [← finalOutputRows_exact]
    exact final_output_rows_satisfy assignment satisfied)
  apply facts
    (420 + lane.val,
      if lane.val < 3 then 1393 + lane.val else 87637 + lane.val)
  exact List.mem_map.mpr
    ⟨lane.val, List.mem_range.mpr lane.isLt, rfl⟩

theorem final_readiness_lane
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment)
    (lane : Fin 8) :
    assignment (420 + lane.val) = assignment (411 + lane.val) := by
  have facts := permutedEqualityRows_sound canonical one (by
    rw [← finalReadinessRows_exact]
    exact final_readiness_rows_satisfy assignment satisfied)
  apply facts (420 + lane.val, 411 + lane.val)
  exact List.mem_map.mpr
    ⟨lane.val, List.mem_range.mpr lane.isLt, rfl⟩

/-! ## Current replay trace acceptance -/

private theorem source_call_accepted
    (arm : RawArm) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) (satisfied : arm.Satisfied assignment)
    (call : Poseidon2Call.Call) (member : call ∈ arm.poseidon2Calls) :
    TranscriptCertificate.CallAccepted call assignment := by
  exact poseidon2_call_refines arm assignment canonical one satisfied call member

private theorem fullChunk0_subset
    {call : Poseidon2Call.Call} (member : call ∈ fullChunk0) :
    call ∈ fullArm.poseidon2Calls := by
  exact List.mem_of_mem_take member

private theorem fullChunk1_subset
    {call : Poseidon2Call.Call} (member : call ∈ fullChunk1) :
    call ∈ fullArm.poseidon2Calls := by
  exact List.mem_of_mem_drop (List.mem_of_mem_take member)

private theorem fullChunk2_subset
    {call : Poseidon2Call.Call} (member : call ∈ fullChunk2) :
    call ∈ fullArm.poseidon2Calls := by
  exact List.mem_of_mem_drop
    (List.mem_of_mem_drop (List.mem_of_mem_take member))

private theorem fullChunk3_subset
    {call : Poseidon2Call.Call} (member : call ∈ fullChunk3) :
    call ∈ fullArm.poseidon2Calls := by
  exact List.mem_of_mem_drop
    (List.mem_of_mem_drop
      (List.mem_of_mem_drop (List.mem_of_mem_take member)))

private theorem finalChunk0_subset
    {call : Poseidon2Call.Call} (member : call ∈ finalChunk0) :
    call ∈ finalArm.poseidon2Calls := by
  exact List.mem_of_mem_take member

private theorem finalChunk1_subset
    {call : Poseidon2Call.Call} (member : call ∈ finalChunk1) :
    call ∈ finalArm.poseidon2Calls := by
  exact List.mem_of_mem_drop (List.mem_of_mem_take member)

private theorem finalChunk2_subset
    {call : Poseidon2Call.Call} (member : call ∈ finalChunk2) :
    call ∈ finalArm.poseidon2Calls := by
  exact List.mem_of_mem_drop
    (List.mem_of_mem_drop (List.mem_of_mem_take member))

private theorem full_slice0_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    fullSlice0Trace.Accepted assignment := by
  constructor
  · simp [fullSlice0Trace]
  · intro call member
    exact source_call_accepted fullArm assignment canonical one satisfied call
      (fullChunk0_subset (by simpa [fullSlice0Trace] using member))

private theorem full_slice1_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    fullSlice1Trace.Accepted assignment := by
  constructor
  · simp [fullSlice1Trace]
  · intro call member
    exact source_call_accepted fullArm assignment canonical one satisfied call
      (fullChunk1_subset (by simpa [fullSlice1Trace] using member))

private theorem full_slice2_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    fullSlice2Trace.Accepted assignment := by
  constructor
  · simp [fullSlice2Trace]
  · intro call member
    exact source_call_accepted fullArm assignment canonical one satisfied call
      (fullChunk2_subset (by simpa [fullSlice2Trace] using member))

private theorem full_slice3_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    fullSlice3Trace.Accepted assignment := by
  constructor
  · simp [fullSlice3Trace]
  · intro call member
    exact source_call_accepted fullArm assignment canonical one satisfied call
      (fullChunk3_subset (by simpa [fullSlice3Trace] using member))

private theorem final_slice0_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    finalSlice0Trace.Accepted assignment := by
  constructor
  · simp [finalSlice0Trace]
  · intro call member
    exact source_call_accepted finalArm assignment canonical one satisfied call
      (finalChunk0_subset (by simpa [finalSlice0Trace] using member))

private theorem final_slice1_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    finalSlice1Trace.Accepted assignment := by
  constructor
  · simp [finalSlice1Trace]
  · intro call member
    exact source_call_accepted finalArm assignment canonical one satisfied call
      (finalChunk1_subset (by simpa [finalSlice1Trace] using member))

private theorem final_tail_trace_accepted
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    finalTailTrace.Accepted assignment := by
  constructor
  · simp [finalTailTrace]
  · intro call member
    exact source_call_accepted finalArm assignment canonical one satisfied call
      (finalChunk2_subset (by simpa [finalTailTrace] using member))

/-! ## Physical execution to independent duplex replay -/

private theorem replay_eq_absorbSlice
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (start result : ColumnReplay.Run)
    (columns : List Nat)
    (operations : List ColumnReplay.Operation)
    (operationsExact :
      operations = columns.map ColumnReplay.Operation.external)
    (refines :
      ColumnReplay.semanticExecuteSlice assignment canonical
          (ColumnReplay.decodeRun assignment canonical start) operations =
        ColumnReplay.decodeRun assignment canonical result) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical result).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (columns.map assignment)
        (toDuplex
          (ColumnReplay.decodeRun assignment canonical start).state) := by
  calc
    toDuplex
        (ColumnReplay.decodeRun assignment canonical result).state =
        toDuplex
          (ColumnReplay.semanticExecuteSlice assignment canonical
            (ColumnReplay.decodeRun assignment canonical start)
            operations).state :=
      congrArg (fun run => toDuplex run.state) refines.symm
    _ = Poseidon2Duplex.absorbSlice
          Poseidon2CanonicalConstants.selected (columns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical start).state) := by
      rw [operationsExact]
      exact semanticExecuteSlice_external_toDuplex assignment canonical _ _
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (columns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical start).state) := by
      rfl

def fullSlice0Columns : List Nat :=
  List.range' 821 64 ++ (List.range' 885 64 ++
    (List.range' 949 64 ++ List.range' 1013 64))

def fullSlice1Columns : List Nat :=
  List.range' 1077 64 ++ (List.range' 1141 64 ++
    (List.range' 1205 64 ++ List.range' 1269 64))

def fullSlice2Columns : List Nat :=
  List.range' 1333 64 ++ (List.range' 1397 64 ++
    (List.range' 1461 64 ++ List.range' 1525 64))

def fullSlice3Columns : List Nat :=
  List.range' 1589 64 ++ (List.range' 1653 64 ++
    (List.range' 1717 64 ++ List.range' 1781 64))

def finalTailColumns : List Nat := List.range' 1333 63

def fullReplayColumns : List Nat :=
  fullSlice0Columns ++ (fullSlice1Columns ++
    (fullSlice2Columns ++ fullSlice3Columns))

def finalReplayColumns : List Nat :=
  fullSlice0Columns ++ (fullSlice1Columns ++ finalTailColumns)

def replayColumns : ArmKind → List Nat
  | .full => fullReplayColumns
  | .final => finalReplayColumns

private theorem fullSlice0Operations_exact :
    fullSlice0Operations =
      fullSlice0Columns.map ColumnReplay.Operation.external := by
  simp [fullSlice0Operations, fullSlice0Operations0,
    fullSlice0Operations1, fullSlice0Operations2,
    fullSlice0Operations3, fullSlice0Columns]

private theorem fullSlice1Operations_exact :
    fullSlice1Operations =
      fullSlice1Columns.map ColumnReplay.Operation.external := by
  simp [fullSlice1Operations, fullSlice1Operations0,
    fullSlice1Operations1, fullSlice1Operations2,
    fullSlice1Operations3, fullSlice1Columns]

private theorem fullSlice2Operations_exact :
    fullSlice2Operations =
      fullSlice2Columns.map ColumnReplay.Operation.external := by
  simp [fullSlice2Operations, fullSlice2Operations0,
    fullSlice2Operations1, fullSlice2Operations2,
    fullSlice2Operations3, fullSlice2Columns]

private theorem fullSlice3Operations_exact :
    fullSlice3Operations =
      fullSlice3Columns.map ColumnReplay.Operation.external := by
  simp [fullSlice3Operations, fullSlice3Operations0,
    fullSlice3Operations1, fullSlice3Operations2,
    fullSlice3Operations3, fullSlice3Columns]

private theorem full_slice0_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical fullSlice0Start)
        fullSlice0Operations =
      ColumnReplay.decodeRun assignment canonical fullSlice0Result := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [fullSlice0Trace, ConstantPins.ValuesCanonical]) one
    (full_slice0_trace_accepted assignment canonical one satisfied)
  exact full_slice0_execution

private theorem full_slice1_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical fullSlice1Start)
        fullSlice1Operations =
      ColumnReplay.decodeRun assignment canonical fullSlice1Result := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [fullSlice1Trace, ConstantPins.ValuesCanonical]) one
    (full_slice1_trace_accepted assignment canonical one satisfied)
  exact full_slice1_execution

private theorem full_slice2_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical fullSlice2Start)
        fullSlice2Operations =
      ColumnReplay.decodeRun assignment canonical fullSlice2Result := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [fullSlice2Trace, ConstantPins.ValuesCanonical]) one
    (full_slice2_trace_accepted assignment canonical one satisfied)
  exact full_slice2_execution

private theorem full_slice3_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical fullSlice3Start)
        fullSlice3Operations =
      ColumnReplay.decodeRun assignment canonical fullSlice3Result := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [fullSlice3Trace, ConstantPins.ValuesCanonical]) one
    (full_slice3_trace_accepted assignment canonical one satisfied)
  exact full_slice3_execution

private theorem final_slice0_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical fullSlice0Start)
        fullSlice0Operations =
      ColumnReplay.decodeRun assignment canonical fullSlice0Result := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [finalSlice0Trace, ConstantPins.ValuesCanonical]) one
    (final_slice0_trace_accepted assignment canonical one satisfied)
  exact final_slice0_execution

private theorem final_slice1_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical fullSlice1Start)
        fullSlice1Operations =
      ColumnReplay.decodeRun assignment canonical fullSlice1Result := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [finalSlice1Trace, ConstantPins.ValuesCanonical]) one
    (final_slice1_trace_accepted assignment canonical one satisfied)
  exact final_slice1_execution

private theorem final_tail_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical finalTailStart)
        finalTailOperations =
      ColumnReplay.decodeRun assignment canonical finalTailResult := by
  apply ColumnReplay.executeSlice_sound canonical
    (by simp [finalTailTrace, ConstantPins.ValuesCanonical]) one
    (final_tail_trace_accepted assignment canonical one satisfied)
  exact final_tail_execution

private theorem full_slice0_eq_absorbSlice
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice0Result).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (fullSlice0Columns.map assignment)
        (toDuplex
          (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state) :=
  replay_eq_absorbSlice assignment canonical _ _ _ _
    fullSlice0Operations_exact
    (full_slice0_refines assignment canonical one satisfied)

private theorem full_slice1_eq_absorbSlice
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice1Result).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (fullSlice1Columns.map assignment)
        (toDuplex
          (ColumnReplay.decodeRun assignment canonical fullSlice1Start).state) :=
  replay_eq_absorbSlice assignment canonical _ _ _ _
    fullSlice1Operations_exact
    (full_slice1_refines assignment canonical one satisfied)

private theorem full_slice2_eq_absorbSlice
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice2Result).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (fullSlice2Columns.map assignment)
        (toDuplex
          (ColumnReplay.decodeRun assignment canonical fullSlice2Start).state) :=
  replay_eq_absorbSlice assignment canonical _ _ _ _
    fullSlice2Operations_exact
    (full_slice2_refines assignment canonical one satisfied)

private theorem full_slice3_eq_absorbSlice
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice3Result).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (fullSlice3Columns.map assignment)
        (toDuplex
          (ColumnReplay.decodeRun assignment canonical fullSlice3Start).state) :=
  replay_eq_absorbSlice assignment canonical _ _ _ _
    fullSlice3Operations_exact
    (full_slice3_refines assignment canonical one satisfied)

private theorem final_slice0_eq_absorbSlice
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice0Result).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (fullSlice0Columns.map assignment)
        (toDuplex
          (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state) :=
  replay_eq_absorbSlice assignment canonical _ _ _ _
    fullSlice0Operations_exact
    (final_slice0_refines assignment canonical one satisfied)

private theorem final_slice1_eq_absorbSlice
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice1Result).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (fullSlice1Columns.map assignment)
        (toDuplex
          (ColumnReplay.decodeRun assignment canonical fullSlice1Start).state) :=
  replay_eq_absorbSlice assignment canonical _ _ _ _
    fullSlice1Operations_exact
    (final_slice1_refines assignment canonical one satisfied)

private theorem final_tail_eq_absorbSlice
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical finalTailResult).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (finalTailColumns.map assignment)
        (toDuplex
          (ColumnReplay.decodeRun assignment canonical finalTailStart).state) :=
  replay_eq_absorbSlice assignment canonical _ _ _ _ rfl
    (final_tail_refines assignment canonical one satisfied)

/-! ## Slice composition and transition-state binding -/

private theorem replay_start_eq_before
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state =
      spongeToDuplex (decodedTransition kind assignment).before.runtime := by
  apply duplexState_ext
  · funext lane
    change assignment (10 + lane.val) =
      (residueNat
        (assignment (transitionColumn kind
          (transitionIndex .before
            (runtimeLaneIndex (duplexLaneIndex lane)))))).val
    rw [transitionColumn_before_runtimeLane]
    simp only [duplexLaneIndex]
    rw [residueNat_val,
      Nat.mod_eq_of_lt (canonical (10 + lane.val))]
  · have constants := generated_rows_imply_state_constants kind assignment
      canonical one satisfied
    change 0 = (decodedTransition kind assignment).before.runtime.absorbed.val
    rw [constants.2.2.1]
    rfl

private theorem full_result_eq_after
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice3Result).state =
      spongeToDuplex (decodedTransition .full assignment).after.runtime := by
  apply duplexState_ext
  · funext lane
    change assignment (155437 + lane.val) =
      (residueNat
        (assignment (transitionColumn .full
          (transitionIndex .after
            (runtimeLaneIndex (duplexLaneIndex lane)))))).val
    rw [transitionColumn_after_runtimeLane]
    simp only [duplexLaneIndex]
    rw [residueNat_val,
      Nat.mod_eq_of_lt (canonical (420 + lane.val))]
    exact (full_output_lane assignment canonical one satisfied lane).symm
  · have constants := generated_rows_imply_state_constants .full assignment
      canonical one satisfied
    change 0 = (decodedTransition .full assignment).after.runtime.absorbed.val
    rw [constants.2.2.2]
    rfl

private theorem final_result_eq_after
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical finalTailResult).state =
      spongeToDuplex (decodedTransition .final assignment).after.runtime := by
  apply duplexState_ext
  · funext lane
    change assignment
        (if lane.val < 3 then 1393 + lane.val else 87637 + lane.val) =
      (residueNat
        (assignment (transitionColumn .final
          (transitionIndex .after
            (runtimeLaneIndex (duplexLaneIndex lane)))))).val
    rw [transitionColumn_after_runtimeLane]
    simp only [duplexLaneIndex]
    rw [residueNat_val,
      Nat.mod_eq_of_lt (canonical (420 + lane.val))]
    exact (final_output_lane assignment canonical one satisfied lane).symm
  · have constants := generated_rows_imply_state_constants .final assignment
      canonical one satisfied
    change 3 = (decodedTransition .final assignment).after.runtime.absorbed.val
    rw [constants.2.2.2]
    rfl

private theorem full_slices_compose
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice3Result).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (fullReplayColumns.map assignment)
        (toDuplex
          (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state) := by
  calc
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice3Result).state =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice3Columns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice3Start).state) :=
      full_slice3_eq_absorbSlice assignment canonical one satisfied
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice3Columns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice2Result).state) := by
      rfl
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice3Columns.map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (fullSlice2Columns.map assignment)
            (toDuplex
              (ColumnReplay.decodeRun assignment canonical fullSlice2Start).state)) := by
      rw [full_slice2_eq_absorbSlice assignment canonical one satisfied]
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((fullSlice2Columns ++ fullSlice3Columns).map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice2Start).state) := by
      rw [List.map_append]
      exact (Poseidon2Duplex.absorbSlice_append ProductPoseidon2.constants
        _ _ _).symm
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((fullSlice2Columns ++ fullSlice3Columns).map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice1Result).state) := by
      rfl
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((fullSlice2Columns ++ fullSlice3Columns).map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (fullSlice1Columns.map assignment)
            (toDuplex
              (ColumnReplay.decodeRun assignment canonical fullSlice1Start).state)) := by
      rw [full_slice1_eq_absorbSlice assignment canonical one satisfied]
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((fullSlice1Columns ++
            (fullSlice2Columns ++ fullSlice3Columns)).map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice1Start).state) := by
      rw [List.map_append]
      exact (Poseidon2Duplex.absorbSlice_append ProductPoseidon2.constants
        _ _ _).symm
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((fullSlice1Columns ++
            (fullSlice2Columns ++ fullSlice3Columns)).map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice0Result).state) := by
      rfl
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((fullSlice1Columns ++
            (fullSlice2Columns ++ fullSlice3Columns)).map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (fullSlice0Columns.map assignment)
            (toDuplex
              (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state)) := by
      rw [full_slice0_eq_absorbSlice assignment canonical one satisfied]
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullReplayColumns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state) := by
      unfold fullReplayColumns
      rw [List.map_append]
      exact (Poseidon2Duplex.absorbSlice_append ProductPoseidon2.constants
        _ _ _).symm

private theorem final_slices_compose
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical finalTailResult).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (finalReplayColumns.map assignment)
        (toDuplex
          (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state) := by
  calc
    toDuplex
        (ColumnReplay.decodeRun assignment canonical finalTailResult).state =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalTailColumns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical finalTailStart).state) :=
      final_tail_eq_absorbSlice assignment canonical one satisfied
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalTailColumns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice1Result).state) := by
      rfl
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalTailColumns.map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (fullSlice1Columns.map assignment)
            (toDuplex
              (ColumnReplay.decodeRun assignment canonical fullSlice1Start).state)) := by
      rw [final_slice1_eq_absorbSlice assignment canonical one satisfied]
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((fullSlice1Columns ++ finalTailColumns).map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice1Start).state) := by
      rw [List.map_append]
      exact (Poseidon2Duplex.absorbSlice_append ProductPoseidon2.constants
        _ _ _).symm
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((fullSlice1Columns ++ finalTailColumns).map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice0Result).state) := by
      rfl
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((fullSlice1Columns ++ finalTailColumns).map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (fullSlice0Columns.map assignment)
            (toDuplex
              (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state)) := by
      rw [final_slice0_eq_absorbSlice assignment canonical one satisfied]
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalReplayColumns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state) := by
      unfold finalReplayColumns
      rw [List.map_append]
      exact (Poseidon2Duplex.absorbSlice_append ProductPoseidon2.constants
        _ _ _).symm

private theorem full_rows_imply_replay_columns
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    spongeToDuplex (decodedTransition .full assignment).after.runtime =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (fullReplayColumns.map assignment)
        (spongeToDuplex
          (decodedTransition .full assignment).before.runtime) := by
  calc
    spongeToDuplex (decodedTransition .full assignment).after.runtime =
        toDuplex
          (ColumnReplay.decodeRun assignment canonical fullSlice3Result).state :=
      (full_result_eq_after assignment canonical one satisfied).symm
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullReplayColumns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state) :=
      full_slices_compose assignment canonical one satisfied
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullReplayColumns.map assignment)
          (spongeToDuplex
            (decodedTransition .full assignment).before.runtime) := by
      rw [replay_start_eq_before .full assignment canonical one satisfied]

private theorem final_rows_imply_replay_columns
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    spongeToDuplex (decodedTransition .final assignment).after.runtime =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (finalReplayColumns.map assignment)
        (spongeToDuplex
          (decodedTransition .final assignment).before.runtime) := by
  calc
    spongeToDuplex (decodedTransition .final assignment).after.runtime =
        toDuplex
          (ColumnReplay.decodeRun assignment canonical finalTailResult).state :=
      (final_result_eq_after assignment canonical one satisfied).symm
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalReplayColumns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical fullSlice0Start).state) :=
      final_slices_compose assignment canonical one satisfied
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalReplayColumns.map assignment)
          (spongeToDuplex
            (decodedTransition .final assignment).before.runtime) := by
      rw [replay_start_eq_before .final assignment canonical one satisfied]

/-- Exact current generated rows imply the authoritative replay field once
the verifier-owned chunk values are linked to the emitted chunk columns. -/
theorem generated_rows_imply_runtimeReplay
    (kind : ArmKind) (frame : ClaimFrame) (chunk : Fin claimChunkCount)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (chunkLinked :
      (replayColumns kind).map assignment = chunkValues frame chunk) :
    spongeToDuplex (decodedTransition kind assignment).after.runtime =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (chunkValues frame chunk)
        (spongeToDuplex
          (decodedTransition kind assignment).before.runtime) := by
  cases kind with
  | full =>
      rw [← chunkLinked]
      exact full_rows_imply_replay_columns assignment canonical one satisfied
  | final =>
      rw [← chunkLinked]
      exact final_rows_imply_replay_columns assignment canonical one satisfied

/-- The exact final-readiness rows identify every runtime lane with the
expected lane. The generated state pins identify both absorbed counters. -/
theorem generated_rows_imply_final
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    spongeToDuplex (decodedTransition .final assignment).after.runtime =
      spongeToDuplex (decodedTransition .final assignment).after.expected := by
  apply duplexState_ext
  · funext lane
    change
      (residueNat
        (assignment (transitionColumn .final
          (transitionIndex .after
            (runtimeLaneIndex (duplexLaneIndex lane)))))).val =
      (residueNat
        (assignment (transitionColumn .final
          (transitionIndex .after
            (expectedLaneIndex (duplexLaneIndex lane)))))).val
    rw [transitionColumn_after_runtimeLane,
      transitionColumn_after_expectedLane]
    simp only [duplexLaneIndex]
    rw [residueNat_val, residueNat_val,
      Nat.mod_eq_of_lt (canonical (420 + lane.val)),
      Nat.mod_eq_of_lt (canonical (411 + lane.val))]
    exact final_readiness_lane assignment canonical one satisfied lane
  · have constants := generated_rows_imply_state_constants .final assignment
      canonical one satisfied
    change
      (decodedTransition .final assignment).after.runtime.absorbed.val =
        (decodedTransition .final assignment).after.expected.absorbed.val
    rw [constants.2.2.2, constants.2.1]
    rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayRuntimeArtifact
