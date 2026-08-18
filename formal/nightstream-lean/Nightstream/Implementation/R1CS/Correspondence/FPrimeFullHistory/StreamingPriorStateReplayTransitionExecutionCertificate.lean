import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPriorStateReplaySource
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplayExecution

/-!
Bounded structural execution certificates for prior-state transition slices.

Owns one exact Rust `absorb_slice` boundary at a time. A 256-field slice uses
one exact 64-call source shard, four 64-field execution leaves, and one final
normalization call. The final 10-field tail uses its exact two-call prefix. It
owns no row satisfaction, semantic refinement, lifecycle selection, or
final-target authority.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayTransitionExecutionCertificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

def checkpointRun
    (lanes : Fin width → Nat) (absorbed : Fin (rate + 1))
    (nextCall : Nat) : ColumnReplay.Run where
  cursor := { lanes, absorbed, nextPin := 0, nextCall }
  digests := []

def fullSlice0Trace : TranscriptCertificate.Trace where
  pins := []
  calls := fullCallsPart0

def fullSlice0Operations0 : List ColumnReplay.Operation :=
  (List.range' 155 64).map ColumnReplay.Operation.external

def fullSlice0Operations1 : List ColumnReplay.Operation :=
  (List.range' 219 64).map ColumnReplay.Operation.external

def fullSlice0Operations2 : List ColumnReplay.Operation :=
  (List.range' 283 64).map ColumnReplay.Operation.external

def fullSlice0Operations3 : List ColumnReplay.Operation :=
  (List.range' 347 64).map ColumnReplay.Operation.external

def fullSlice0Operations : List ColumnReplay.Operation :=
  fullSlice0Operations0 ++ (fullSlice0Operations1 ++
    (fullSlice0Operations2 ++ fullSlice0Operations3))

def fullSlice0Start : ColumnReplay.Run :=
  checkpointRun (fun lane => 1 + lane.val) ⟨0, by decide⟩ 0

def fullSlice0Run1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 215 + lane.val else 10171 + lane.val)
    ⟨4, by decide⟩ 15

def fullSlice0Run2 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 279 + lane.val else 19771 + lane.val)
    ⟨4, by decide⟩ 31

def fullSlice0Run3 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 343 + lane.val else 29371 + lane.val)
    ⟨4, by decide⟩ 47

def fullSlice0BeforeNormalize : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 407 + lane.val else 38971 + lane.val)
    ⟨4, by decide⟩ 63

def fullSlice0Result : ColumnReplay.Run :=
  checkpointRun (fun lane => 39571 + lane.val) ⟨0, by decide⟩ 64

def fullSlice0LastCall : Poseidon2Call.Call := {
  rowStart := 37943
  rowEnd := 38543
  inputColumns := [407, 408, 409, 410, 38975, 38976, 38977, 38978]
  firstAllocatedColumn := 38979
}

private theorem full_slice0_trace_length : fullSlice0Trace.calls.length = 64 := by
  rfl

private theorem full_slice0_last_call_bounded :
    63 < fullSlice0Trace.calls.length := by
  rw [full_slice0_trace_length]
  decide

private theorem full_slice0_last_call_exact :
    fullSlice0Trace.calls.get ⟨63, full_slice0_last_call_bounded⟩ =
      fullSlice0LastCall := by
  rfl

private theorem full_slice0_result_cursor_exact :
    ({
      lanes := ColumnReplay.callOutputColumns fullSlice0LastCall
      absorbed := ⟨0, by decide⟩
      nextPin := fullSlice0BeforeNormalize.cursor.nextPin
      nextCall := fullSlice0BeforeNormalize.cursor.nextCall + 1
    } : ColumnReplay.Cursor) = fullSlice0Result.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem full_slice0_permute :
    ColumnReplay.permute fullSlice0Trace fullSlice0BeforeNormalize.cursor =
      some fullSlice0Result.cursor := by
  rw [ColumnReplayExecution.permute_of_call fullSlice0Trace
    fullSlice0BeforeNormalize.cursor
    fullSlice0LastCall full_slice0_last_call_bounded
    full_slice0_last_call_exact (by rfl)]
  exact congrArg some full_slice0_result_cursor_exact

private theorem full_slice0_leaf0 :
    ColumnReplay.execute fullSlice0Trace fullSlice0Start
        fullSlice0Operations0 = some fullSlice0Run1 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice0_leaf1 :
    ColumnReplay.execute fullSlice0Trace fullSlice0Run1
        fullSlice0Operations1 = some fullSlice0Run2 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice0_leaf2 :
    ColumnReplay.execute fullSlice0Trace fullSlice0Run2
        fullSlice0Operations2 = some fullSlice0Run3 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice0_leaf3 :
    ColumnReplay.execute fullSlice0Trace fullSlice0Run3
        fullSlice0Operations3 = some fullSlice0BeforeNormalize := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice0_raw_execution :
    ColumnReplay.execute fullSlice0Trace fullSlice0Start
        fullSlice0Operations = some fullSlice0BeforeNormalize := by
  unfold fullSlice0Operations
  exact ColumnReplay.execute_append full_slice0_leaf0
    (ColumnReplay.execute_append full_slice0_leaf1
      (ColumnReplay.execute_append full_slice0_leaf2 full_slice0_leaf3))

private theorem full_slice0_normalization :
    ColumnReplay.normalizeSlice fullSlice0Trace fullSlice0BeforeNormalize =
      some fullSlice0Result := by
  unfold ColumnReplay.normalizeSlice
  split
  · rw [full_slice0_permute]
    rfl
  · rename_i notFull
    have full :
        rate ≤ fullSlice0BeforeNormalize.cursor.absorbed.val := by
      change 4 ≤ 4
      omega
    exact (notFull full).elim

theorem full_slice0_execution :
    ColumnReplay.executeSlice fullSlice0Trace fullSlice0Start
        fullSlice0Operations = some fullSlice0Result := by
  simp only [ColumnReplay.executeSlice, full_slice0_raw_execution]
  exact full_slice0_normalization

def fullSlice1Trace : TranscriptCertificate.Trace where
  pins := []
  calls := fullCallsPart1

def fullSlice1Operations0 : List ColumnReplay.Operation :=
  (List.range' 411 64).map ColumnReplay.Operation.external

def fullSlice1Operations1 : List ColumnReplay.Operation :=
  (List.range' 475 64).map ColumnReplay.Operation.external

def fullSlice1Operations2 : List ColumnReplay.Operation :=
  (List.range' 539 64).map ColumnReplay.Operation.external

def fullSlice1Operations3 : List ColumnReplay.Operation :=
  (List.range' 603 64).map ColumnReplay.Operation.external

def fullSlice1Operations : List ColumnReplay.Operation :=
  fullSlice1Operations0 ++ (fullSlice1Operations1 ++
    (fullSlice1Operations2 ++ fullSlice1Operations3))

def fullSlice1Start : ColumnReplay.Run :=
  checkpointRun (fun lane => 39571 + lane.val) ⟨0, by decide⟩ 0

def fullSlice1Run1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 471 + lane.val else 48571 + lane.val)
    ⟨4, by decide⟩ 15

def fullSlice1Run2 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 535 + lane.val else 58171 + lane.val)
    ⟨4, by decide⟩ 31

def fullSlice1Run3 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 599 + lane.val else 67771 + lane.val)
    ⟨4, by decide⟩ 47

def fullSlice1BeforeNormalize : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 663 + lane.val else 77371 + lane.val)
    ⟨4, by decide⟩ 63

def fullSlice1Result : ColumnReplay.Run :=
  checkpointRun (fun lane => 77971 + lane.val) ⟨0, by decide⟩ 64

def fullSlice1LastCall : Poseidon2Call.Call := {
  rowStart := 76343
  rowEnd := 76943
  inputColumns := [663, 664, 665, 666, 77375, 77376, 77377, 77378]
  firstAllocatedColumn := 77379
}

private theorem full_slice1_trace_length : fullSlice1Trace.calls.length = 64 := by
  rfl

private theorem full_slice1_last_call_bounded :
    63 < fullSlice1Trace.calls.length := by
  rw [full_slice1_trace_length]
  decide

private theorem full_slice1_last_call_exact :
    fullSlice1Trace.calls.get ⟨63, full_slice1_last_call_bounded⟩ =
      fullSlice1LastCall := by
  rfl

private theorem full_slice1_result_cursor_exact :
    ({
      lanes := ColumnReplay.callOutputColumns fullSlice1LastCall
      absorbed := ⟨0, by decide⟩
      nextPin := fullSlice1BeforeNormalize.cursor.nextPin
      nextCall := fullSlice1BeforeNormalize.cursor.nextCall + 1
    } : ColumnReplay.Cursor) = fullSlice1Result.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem full_slice1_permute :
    ColumnReplay.permute fullSlice1Trace fullSlice1BeforeNormalize.cursor =
      some fullSlice1Result.cursor := by
  rw [ColumnReplayExecution.permute_of_call fullSlice1Trace
    fullSlice1BeforeNormalize.cursor
    fullSlice1LastCall full_slice1_last_call_bounded
    full_slice1_last_call_exact (by rfl)]
  exact congrArg some full_slice1_result_cursor_exact

private theorem full_slice1_leaf0 :
    ColumnReplay.execute fullSlice1Trace fullSlice1Start
        fullSlice1Operations0 = some fullSlice1Run1 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice1_leaf1 :
    ColumnReplay.execute fullSlice1Trace fullSlice1Run1
        fullSlice1Operations1 = some fullSlice1Run2 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice1_leaf2 :
    ColumnReplay.execute fullSlice1Trace fullSlice1Run2
        fullSlice1Operations2 = some fullSlice1Run3 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice1_leaf3 :
    ColumnReplay.execute fullSlice1Trace fullSlice1Run3
        fullSlice1Operations3 = some fullSlice1BeforeNormalize := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice1_raw_execution :
    ColumnReplay.execute fullSlice1Trace fullSlice1Start
        fullSlice1Operations = some fullSlice1BeforeNormalize := by
  unfold fullSlice1Operations
  exact ColumnReplay.execute_append full_slice1_leaf0
    (ColumnReplay.execute_append full_slice1_leaf1
      (ColumnReplay.execute_append full_slice1_leaf2 full_slice1_leaf3))

private theorem full_slice1_normalization :
    ColumnReplay.normalizeSlice fullSlice1Trace fullSlice1BeforeNormalize =
      some fullSlice1Result := by
  unfold ColumnReplay.normalizeSlice
  split
  · rw [full_slice1_permute]
    rfl
  · rename_i notFull
    have full :
        rate ≤ fullSlice1BeforeNormalize.cursor.absorbed.val := by
      change 4 ≤ 4
      omega
    exact (notFull full).elim

theorem full_slice1_execution :
    ColumnReplay.executeSlice fullSlice1Trace fullSlice1Start
        fullSlice1Operations = some fullSlice1Result := by
  simp only [ColumnReplay.executeSlice, full_slice1_raw_execution]
  exact full_slice1_normalization

def fullSlice2Trace : TranscriptCertificate.Trace where
  pins := []
  calls := fullCallsPart2

def fullSlice2Operations0 : List ColumnReplay.Operation :=
  (List.range' 667 64).map ColumnReplay.Operation.external

def fullSlice2Operations1 : List ColumnReplay.Operation :=
  (List.range' 731 64).map ColumnReplay.Operation.external

def fullSlice2Operations2 : List ColumnReplay.Operation :=
  (List.range' 795 64).map ColumnReplay.Operation.external

def fullSlice2Operations3 : List ColumnReplay.Operation :=
  (List.range' 859 64).map ColumnReplay.Operation.external

def fullSlice2Operations : List ColumnReplay.Operation :=
  fullSlice2Operations0 ++ (fullSlice2Operations1 ++
    (fullSlice2Operations2 ++ fullSlice2Operations3))

def fullSlice2Start : ColumnReplay.Run :=
  checkpointRun (fun lane => 77971 + lane.val) ⟨0, by decide⟩ 0

def fullSlice2Run1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 727 + lane.val else 86971 + lane.val)
    ⟨4, by decide⟩ 15

def fullSlice2Run2 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 791 + lane.val else 96571 + lane.val)
    ⟨4, by decide⟩ 31

def fullSlice2Run3 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 855 + lane.val else 106171 + lane.val)
    ⟨4, by decide⟩ 47

def fullSlice2BeforeNormalize : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 919 + lane.val else 115771 + lane.val)
    ⟨4, by decide⟩ 63

def fullSlice2Result : ColumnReplay.Run :=
  checkpointRun (fun lane => 116371 + lane.val) ⟨0, by decide⟩ 64

def fullSlice2LastCall : Poseidon2Call.Call := {
  rowStart := 114743
  rowEnd := 115343
  inputColumns := [919, 920, 921, 922, 115775, 115776, 115777, 115778]
  firstAllocatedColumn := 115779
}

private theorem full_slice2_trace_length : fullSlice2Trace.calls.length = 64 := by
  rfl

private theorem full_slice2_last_call_bounded :
    63 < fullSlice2Trace.calls.length := by
  rw [full_slice2_trace_length]
  decide

private theorem full_slice2_last_call_exact :
    fullSlice2Trace.calls.get ⟨63, full_slice2_last_call_bounded⟩ =
      fullSlice2LastCall := by
  rfl

private theorem full_slice2_result_cursor_exact :
    ({
      lanes := ColumnReplay.callOutputColumns fullSlice2LastCall
      absorbed := ⟨0, by decide⟩
      nextPin := fullSlice2BeforeNormalize.cursor.nextPin
      nextCall := fullSlice2BeforeNormalize.cursor.nextCall + 1
    } : ColumnReplay.Cursor) = fullSlice2Result.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem full_slice2_permute :
    ColumnReplay.permute fullSlice2Trace fullSlice2BeforeNormalize.cursor =
      some fullSlice2Result.cursor := by
  rw [ColumnReplayExecution.permute_of_call fullSlice2Trace
    fullSlice2BeforeNormalize.cursor
    fullSlice2LastCall full_slice2_last_call_bounded
    full_slice2_last_call_exact (by rfl)]
  exact congrArg some full_slice2_result_cursor_exact

private theorem full_slice2_leaf0 :
    ColumnReplay.execute fullSlice2Trace fullSlice2Start
        fullSlice2Operations0 = some fullSlice2Run1 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice2_leaf1 :
    ColumnReplay.execute fullSlice2Trace fullSlice2Run1
        fullSlice2Operations1 = some fullSlice2Run2 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice2_leaf2 :
    ColumnReplay.execute fullSlice2Trace fullSlice2Run2
        fullSlice2Operations2 = some fullSlice2Run3 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice2_leaf3 :
    ColumnReplay.execute fullSlice2Trace fullSlice2Run3
        fullSlice2Operations3 = some fullSlice2BeforeNormalize := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice2_raw_execution :
    ColumnReplay.execute fullSlice2Trace fullSlice2Start
        fullSlice2Operations = some fullSlice2BeforeNormalize := by
  unfold fullSlice2Operations
  exact ColumnReplay.execute_append full_slice2_leaf0
    (ColumnReplay.execute_append full_slice2_leaf1
      (ColumnReplay.execute_append full_slice2_leaf2 full_slice2_leaf3))

private theorem full_slice2_normalization :
    ColumnReplay.normalizeSlice fullSlice2Trace fullSlice2BeforeNormalize =
      some fullSlice2Result := by
  unfold ColumnReplay.normalizeSlice
  split
  · rw [full_slice2_permute]
    rfl
  · rename_i notFull
    have full :
        rate ≤ fullSlice2BeforeNormalize.cursor.absorbed.val := by
      change 4 ≤ 4
      omega
    exact (notFull full).elim

theorem full_slice2_execution :
    ColumnReplay.executeSlice fullSlice2Trace fullSlice2Start
        fullSlice2Operations = some fullSlice2Result := by
  simp only [ColumnReplay.executeSlice, full_slice2_raw_execution]
  exact full_slice2_normalization

def fullSlice3Trace : TranscriptCertificate.Trace where
  pins := []
  calls := fullCallsPart3

def fullSlice3Operations0 : List ColumnReplay.Operation :=
  (List.range' 923 64).map ColumnReplay.Operation.external

def fullSlice3Operations1 : List ColumnReplay.Operation :=
  (List.range' 987 64).map ColumnReplay.Operation.external

def fullSlice3Operations2 : List ColumnReplay.Operation :=
  (List.range' 1051 64).map ColumnReplay.Operation.external

def fullSlice3Operations3 : List ColumnReplay.Operation :=
  (List.range' 1115 64).map ColumnReplay.Operation.external

def fullSlice3Operations : List ColumnReplay.Operation :=
  fullSlice3Operations0 ++ (fullSlice3Operations1 ++
    (fullSlice3Operations2 ++ fullSlice3Operations3))

def fullSlice3Start : ColumnReplay.Run :=
  checkpointRun (fun lane => 116371 + lane.val) ⟨0, by decide⟩ 0

def fullSlice3Run1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 983 + lane.val else 125371 + lane.val)
    ⟨4, by decide⟩ 15

def fullSlice3Run2 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1047 + lane.val else 134971 + lane.val)
    ⟨4, by decide⟩ 31

def fullSlice3Run3 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1111 + lane.val else 144571 + lane.val)
    ⟨4, by decide⟩ 47

def fullSlice3BeforeNormalize : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1175 + lane.val else 154171 + lane.val)
    ⟨4, by decide⟩ 63

def fullSlice3Result : ColumnReplay.Run :=
  checkpointRun (fun lane => 154771 + lane.val) ⟨0, by decide⟩ 64

def fullSlice3LastCall : Poseidon2Call.Call := {
  rowStart := 153143
  rowEnd := 153743
  inputColumns := [1175, 1176, 1177, 1178, 154175, 154176, 154177, 154178]
  firstAllocatedColumn := 154179
}

private theorem full_slice3_trace_length : fullSlice3Trace.calls.length = 64 := by
  rfl

private theorem full_slice3_last_call_bounded :
    63 < fullSlice3Trace.calls.length := by
  rw [full_slice3_trace_length]
  decide

private theorem full_slice3_last_call_exact :
    fullSlice3Trace.calls.get ⟨63, full_slice3_last_call_bounded⟩ =
      fullSlice3LastCall := by
  rfl

private theorem full_slice3_result_cursor_exact :
    ({
      lanes := ColumnReplay.callOutputColumns fullSlice3LastCall
      absorbed := ⟨0, by decide⟩
      nextPin := fullSlice3BeforeNormalize.cursor.nextPin
      nextCall := fullSlice3BeforeNormalize.cursor.nextCall + 1
    } : ColumnReplay.Cursor) = fullSlice3Result.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem full_slice3_permute :
    ColumnReplay.permute fullSlice3Trace fullSlice3BeforeNormalize.cursor =
      some fullSlice3Result.cursor := by
  rw [ColumnReplayExecution.permute_of_call fullSlice3Trace
    fullSlice3BeforeNormalize.cursor
    fullSlice3LastCall full_slice3_last_call_bounded
    full_slice3_last_call_exact (by rfl)]
  exact congrArg some full_slice3_result_cursor_exact

private theorem full_slice3_leaf0 :
    ColumnReplay.execute fullSlice3Trace fullSlice3Start
        fullSlice3Operations0 = some fullSlice3Run1 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice3_leaf1 :
    ColumnReplay.execute fullSlice3Trace fullSlice3Run1
        fullSlice3Operations1 = some fullSlice3Run2 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice3_leaf2 :
    ColumnReplay.execute fullSlice3Trace fullSlice3Run2
        fullSlice3Operations2 = some fullSlice3Run3 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice3_leaf3 :
    ColumnReplay.execute fullSlice3Trace fullSlice3Run3
        fullSlice3Operations3 = some fullSlice3BeforeNormalize := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_slice3_raw_execution :
    ColumnReplay.execute fullSlice3Trace fullSlice3Start
        fullSlice3Operations = some fullSlice3BeforeNormalize := by
  unfold fullSlice3Operations
  exact ColumnReplay.execute_append full_slice3_leaf0
    (ColumnReplay.execute_append full_slice3_leaf1
      (ColumnReplay.execute_append full_slice3_leaf2 full_slice3_leaf3))

private theorem full_slice3_normalization :
    ColumnReplay.normalizeSlice fullSlice3Trace fullSlice3BeforeNormalize =
      some fullSlice3Result := by
  unfold ColumnReplay.normalizeSlice
  split
  · rw [full_slice3_permute]
    rfl
  · rename_i notFull
    have full :
        rate ≤ fullSlice3BeforeNormalize.cursor.absorbed.val := by
      change 4 ≤ 4
      omega
    exact (notFull full).elim

theorem full_slice3_execution :
    ColumnReplay.executeSlice fullSlice3Trace fullSlice3Start
        fullSlice3Operations = some fullSlice3Result := by
  simp only [ColumnReplay.executeSlice, full_slice3_raw_execution]
  exact full_slice3_normalization

def finalSlice0Trace : TranscriptCertificate.Trace where
  pins := []
  calls := finalCallsPart0

def finalSlice0Operations0 : List ColumnReplay.Operation :=
  (List.range' 159 64).map ColumnReplay.Operation.external

def finalSlice0Operations1 : List ColumnReplay.Operation :=
  (List.range' 223 64).map ColumnReplay.Operation.external

def finalSlice0Operations2 : List ColumnReplay.Operation :=
  (List.range' 287 64).map ColumnReplay.Operation.external

def finalSlice0Operations3 : List ColumnReplay.Operation :=
  (List.range' 351 64).map ColumnReplay.Operation.external

def finalSlice0Operations : List ColumnReplay.Operation :=
  finalSlice0Operations0 ++ (finalSlice0Operations1 ++
    (finalSlice0Operations2 ++ finalSlice0Operations3))

def finalSlice0Start : ColumnReplay.Run :=
  checkpointRun (fun lane => 1 + lane.val) ⟨0, by decide⟩ 0

def finalSlice0Run1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 219 + lane.val else 10175 + lane.val)
    ⟨4, by decide⟩ 15

def finalSlice0Run2 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 283 + lane.val else 19775 + lane.val)
    ⟨4, by decide⟩ 31

def finalSlice0Run3 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 347 + lane.val else 29375 + lane.val)
    ⟨4, by decide⟩ 47

def finalSlice0BeforeNormalize : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 411 + lane.val else 38975 + lane.val)
    ⟨4, by decide⟩ 63

def finalSlice0Result : ColumnReplay.Run :=
  checkpointRun (fun lane => 39575 + lane.val) ⟨0, by decide⟩ 64

def finalSlice0LastCall : Poseidon2Call.Call := {
  rowStart := 37943
  rowEnd := 38543
  inputColumns := [411, 412, 413, 414, 38979, 38980, 38981, 38982]
  firstAllocatedColumn := 38983
}

private theorem final_slice0_trace_length : finalSlice0Trace.calls.length = 64 := by
  rfl

private theorem final_slice0_last_call_bounded :
    63 < finalSlice0Trace.calls.length := by
  rw [final_slice0_trace_length]
  decide

private theorem final_slice0_last_call_exact :
    finalSlice0Trace.calls.get ⟨63, final_slice0_last_call_bounded⟩ =
      finalSlice0LastCall := by
  rfl

private theorem final_slice0_result_cursor_exact :
    ({
      lanes := ColumnReplay.callOutputColumns finalSlice0LastCall
      absorbed := ⟨0, by decide⟩
      nextPin := finalSlice0BeforeNormalize.cursor.nextPin
      nextCall := finalSlice0BeforeNormalize.cursor.nextCall + 1
    } : ColumnReplay.Cursor) = finalSlice0Result.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem final_slice0_permute :
    ColumnReplay.permute finalSlice0Trace finalSlice0BeforeNormalize.cursor =
      some finalSlice0Result.cursor := by
  rw [ColumnReplayExecution.permute_of_call finalSlice0Trace
    finalSlice0BeforeNormalize.cursor
    finalSlice0LastCall final_slice0_last_call_bounded
    final_slice0_last_call_exact (by rfl)]
  exact congrArg some final_slice0_result_cursor_exact

private theorem final_slice0_leaf0 :
    ColumnReplay.execute finalSlice0Trace finalSlice0Start
        finalSlice0Operations0 = some finalSlice0Run1 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice0_leaf1 :
    ColumnReplay.execute finalSlice0Trace finalSlice0Run1
        finalSlice0Operations1 = some finalSlice0Run2 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice0_leaf2 :
    ColumnReplay.execute finalSlice0Trace finalSlice0Run2
        finalSlice0Operations2 = some finalSlice0Run3 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice0_leaf3 :
    ColumnReplay.execute finalSlice0Trace finalSlice0Run3
        finalSlice0Operations3 = some finalSlice0BeforeNormalize := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice0_raw_execution :
    ColumnReplay.execute finalSlice0Trace finalSlice0Start
        finalSlice0Operations = some finalSlice0BeforeNormalize := by
  unfold finalSlice0Operations
  exact ColumnReplay.execute_append final_slice0_leaf0
    (ColumnReplay.execute_append final_slice0_leaf1
      (ColumnReplay.execute_append final_slice0_leaf2 final_slice0_leaf3))

private theorem final_slice0_normalization :
    ColumnReplay.normalizeSlice finalSlice0Trace finalSlice0BeforeNormalize =
      some finalSlice0Result := by
  unfold ColumnReplay.normalizeSlice
  split
  · rw [final_slice0_permute]
    rfl
  · rename_i notFull
    have full :
        rate ≤ finalSlice0BeforeNormalize.cursor.absorbed.val := by
      change 4 ≤ 4
      omega
    exact (notFull full).elim

theorem final_slice0_execution :
    ColumnReplay.executeSlice finalSlice0Trace finalSlice0Start
        finalSlice0Operations = some finalSlice0Result := by
  simp only [ColumnReplay.executeSlice, final_slice0_raw_execution]
  exact final_slice0_normalization

def finalSlice1Trace : TranscriptCertificate.Trace where
  pins := []
  calls := finalCallsPart1

def finalSlice1Operations0 : List ColumnReplay.Operation :=
  (List.range' 415 64).map ColumnReplay.Operation.external

def finalSlice1Operations1 : List ColumnReplay.Operation :=
  (List.range' 479 64).map ColumnReplay.Operation.external

def finalSlice1Operations2 : List ColumnReplay.Operation :=
  (List.range' 543 64).map ColumnReplay.Operation.external

def finalSlice1Operations3 : List ColumnReplay.Operation :=
  (List.range' 607 64).map ColumnReplay.Operation.external

def finalSlice1Operations : List ColumnReplay.Operation :=
  finalSlice1Operations0 ++ (finalSlice1Operations1 ++
    (finalSlice1Operations2 ++ finalSlice1Operations3))

def finalSlice1Start : ColumnReplay.Run :=
  checkpointRun (fun lane => 39575 + lane.val) ⟨0, by decide⟩ 0

def finalSlice1Run1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 475 + lane.val else 48575 + lane.val)
    ⟨4, by decide⟩ 15

def finalSlice1Run2 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 539 + lane.val else 58175 + lane.val)
    ⟨4, by decide⟩ 31

def finalSlice1Run3 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 603 + lane.val else 67775 + lane.val)
    ⟨4, by decide⟩ 47

def finalSlice1BeforeNormalize : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 667 + lane.val else 77375 + lane.val)
    ⟨4, by decide⟩ 63

def finalSlice1Result : ColumnReplay.Run :=
  checkpointRun (fun lane => 77975 + lane.val) ⟨0, by decide⟩ 64

def finalSlice1LastCall : Poseidon2Call.Call := {
  rowStart := 76343
  rowEnd := 76943
  inputColumns := [667, 668, 669, 670, 77379, 77380, 77381, 77382]
  firstAllocatedColumn := 77383
}

private theorem final_slice1_trace_length : finalSlice1Trace.calls.length = 64 := by
  rfl

private theorem final_slice1_last_call_bounded :
    63 < finalSlice1Trace.calls.length := by
  rw [final_slice1_trace_length]
  decide

private theorem final_slice1_last_call_exact :
    finalSlice1Trace.calls.get ⟨63, final_slice1_last_call_bounded⟩ =
      finalSlice1LastCall := by
  rfl

private theorem final_slice1_result_cursor_exact :
    ({
      lanes := ColumnReplay.callOutputColumns finalSlice1LastCall
      absorbed := ⟨0, by decide⟩
      nextPin := finalSlice1BeforeNormalize.cursor.nextPin
      nextCall := finalSlice1BeforeNormalize.cursor.nextCall + 1
    } : ColumnReplay.Cursor) = finalSlice1Result.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem final_slice1_permute :
    ColumnReplay.permute finalSlice1Trace finalSlice1BeforeNormalize.cursor =
      some finalSlice1Result.cursor := by
  rw [ColumnReplayExecution.permute_of_call finalSlice1Trace
    finalSlice1BeforeNormalize.cursor
    finalSlice1LastCall final_slice1_last_call_bounded
    final_slice1_last_call_exact (by rfl)]
  exact congrArg some final_slice1_result_cursor_exact

private theorem final_slice1_leaf0 :
    ColumnReplay.execute finalSlice1Trace finalSlice1Start
        finalSlice1Operations0 = some finalSlice1Run1 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice1_leaf1 :
    ColumnReplay.execute finalSlice1Trace finalSlice1Run1
        finalSlice1Operations1 = some finalSlice1Run2 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice1_leaf2 :
    ColumnReplay.execute finalSlice1Trace finalSlice1Run2
        finalSlice1Operations2 = some finalSlice1Run3 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice1_leaf3 :
    ColumnReplay.execute finalSlice1Trace finalSlice1Run3
        finalSlice1Operations3 = some finalSlice1BeforeNormalize := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice1_raw_execution :
    ColumnReplay.execute finalSlice1Trace finalSlice1Start
        finalSlice1Operations = some finalSlice1BeforeNormalize := by
  unfold finalSlice1Operations
  exact ColumnReplay.execute_append final_slice1_leaf0
    (ColumnReplay.execute_append final_slice1_leaf1
      (ColumnReplay.execute_append final_slice1_leaf2 final_slice1_leaf3))

private theorem final_slice1_normalization :
    ColumnReplay.normalizeSlice finalSlice1Trace finalSlice1BeforeNormalize =
      some finalSlice1Result := by
  unfold ColumnReplay.normalizeSlice
  split
  · rw [final_slice1_permute]
    rfl
  · rename_i notFull
    have full :
        rate ≤ finalSlice1BeforeNormalize.cursor.absorbed.val := by
      change 4 ≤ 4
      omega
    exact (notFull full).elim

theorem final_slice1_execution :
    ColumnReplay.executeSlice finalSlice1Trace finalSlice1Start
        finalSlice1Operations = some finalSlice1Result := by
  simp only [ColumnReplay.executeSlice, final_slice1_raw_execution]
  exact final_slice1_normalization

def finalTailTrace : TranscriptCertificate.Trace where
  pins := []
  calls := finalCallsPart2.take 2

def finalTailOperations : List ColumnReplay.Operation :=
  (List.range' 671 10).map ColumnReplay.Operation.external

def finalTailStart : ColumnReplay.Run :=
  checkpointRun (fun lane => 77975 + lane.val) ⟨0, by decide⟩ 0

def finalTailResult : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 2 then 679 + lane.val else 79175 + lane.val)
    ⟨2, by decide⟩ 2

theorem final_tail_execution :
    ColumnReplay.executeSlice finalTailTrace finalTailStart
        finalTailOperations = some finalTailResult := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayTransitionExecutionCertificate
