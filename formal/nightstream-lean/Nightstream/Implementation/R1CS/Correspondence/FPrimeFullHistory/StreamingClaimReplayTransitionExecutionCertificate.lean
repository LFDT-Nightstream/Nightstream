import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplayExecution

/-!
Contract: bounded physical Poseidon2 execution certificates for the current
streaming claim-replay transition.

Assurance tier: artifact-checked for the exact Rust-emitted full and final
replay-call prefixes in the Goldilocks `b = 2`, `k_rho = 16` profile.

Owns four exact 256-field slices for the full arm, two exact 256-field slices
and one exact 63-field tail for the final arm. Every closed execution leaf has
at most 64 external operations. It does not own row satisfaction, semantic
Poseidon2 refinement, output glue, frame authority, or lifecycle composition.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayTransitionExecutionCertificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

/-! The current physical artifact contains replay calls first and state-digest
calls second. These bounded source shards expose only the replay prefix. -/

def fullTail0 := fullArm.poseidon2Calls
def fullChunk0 := fullTail0.take 64
def fullTail1 := fullTail0.drop 64
def fullChunk1 := fullTail1.take 64
def fullTail2 := fullTail1.drop 64
def fullChunk2 := fullTail2.take 64
def fullTail3 := fullTail2.drop 64
def fullChunk3 := fullTail3.take 64

theorem fullChunk0_length : fullChunk0.length = 64 := by rfl
theorem fullChunk1_length : fullChunk1.length = 64 := by rfl
theorem fullChunk2_length : fullChunk2.length = 64 := by rfl
theorem fullChunk3_length : fullChunk3.length = 64 := by rfl

def finalTail0 := finalArm.poseidon2Calls
def finalChunk0 := finalTail0.take 64
def finalTail1 := finalTail0.drop 64
def finalChunk1 := finalTail1.take 64
def finalTail2 := finalTail1.drop 64
def finalChunk2 := finalTail2.take 15

theorem finalChunk0_length : finalChunk0.length = 64 := by rfl
theorem finalChunk1_length : finalChunk1.length = 64 := by rfl
theorem finalChunk2_length : finalChunk2.length = 15 := by rfl

def checkpointRun
    (lanes : Fin width → Nat) (absorbed : Fin (rate + 1))
    (nextCall : Nat) : ColumnReplay.Run where
  cursor := { lanes, absorbed, nextPin := 0, nextCall }
  digests := []

private theorem execute_four_leaves
    {trace : TranscriptCertificate.Trace}
    {start run1 run2 run3 beforeNormalize : ColumnReplay.Run}
    {operations0 operations1 operations2 operations3 :
      List ColumnReplay.Operation}
    (leaf0 :
      ColumnReplay.execute trace start operations0 = some run1)
    (leaf1 :
      ColumnReplay.execute trace run1 operations1 = some run2)
    (leaf2 :
      ColumnReplay.execute trace run2 operations2 = some run3)
    (leaf3 :
      ColumnReplay.execute trace run3 operations3 = some beforeNormalize) :
    ColumnReplay.execute trace start
        (operations0 ++ (operations1 ++ (operations2 ++ operations3))) =
      some beforeNormalize :=
  ColumnReplay.execute_append leaf0
    (ColumnReplay.execute_append leaf1
      (ColumnReplay.execute_append leaf2 leaf3))

/-! ## Full arm: fields 0 through 255 -/

def fullSlice0Trace : TranscriptCertificate.Trace where
  pins := []
  calls := fullChunk0

def fullSlice0Operations0 : List ColumnReplay.Operation :=
  (List.range' 821 64).map ColumnReplay.Operation.external

def fullSlice0Operations1 : List ColumnReplay.Operation :=
  (List.range' 885 64).map ColumnReplay.Operation.external

def fullSlice0Operations2 : List ColumnReplay.Operation :=
  (List.range' 949 64).map ColumnReplay.Operation.external

def fullSlice0Operations3 : List ColumnReplay.Operation :=
  (List.range' 1013 64).map ColumnReplay.Operation.external

def fullSlice0Operations : List ColumnReplay.Operation :=
  fullSlice0Operations0 ++ (fullSlice0Operations1 ++
    (fullSlice0Operations2 ++ fullSlice0Operations3))

def fullSlice0Start : ColumnReplay.Run :=
  checkpointRun (fun lane => 10 + lane.val) ⟨0, by decide⟩ 0

def fullSlice0Run1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 881 + lane.val else 10837 + lane.val)
    ⟨4, by decide⟩ 15

def fullSlice0Run2 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 945 + lane.val else 20437 + lane.val)
    ⟨4, by decide⟩ 31

def fullSlice0Run3 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1009 + lane.val else 30037 + lane.val)
    ⟨4, by decide⟩ 47

def fullSlice0BeforeNormalize : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1073 + lane.val else 39637 + lane.val)
    ⟨4, by decide⟩ 63

def fullSlice0Result : ColumnReplay.Run :=
  checkpointRun (fun lane => 40237 + lane.val) ⟨0, by decide⟩ 64

def fullSlice0LastCall : Poseidon2Call.Call := {
  rowStart := 38278
  rowEnd := 38878
  inputColumns := [1073, 1074, 1075, 1076, 39641, 39642, 39643, 39644]
  firstAllocatedColumn := 39645
}

private theorem full_slice0_trace_length :
    fullSlice0Trace.calls.length = 64 :=
  fullChunk0_length

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
    fullSlice0BeforeNormalize.cursor fullSlice0LastCall
    full_slice0_last_call_bounded full_slice0_last_call_exact (by rfl)]
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
  unfold ColumnReplay.executeSlice fullSlice0Operations
  rw [execute_four_leaves full_slice0_leaf0 full_slice0_leaf1
    full_slice0_leaf2 full_slice0_leaf3]
  exact full_slice0_normalization

/-! ## Full arm: fields 256 through 511 -/

def fullSlice1Trace : TranscriptCertificate.Trace where
  pins := []
  calls := fullChunk1

def fullSlice1Operations0 : List ColumnReplay.Operation :=
  (List.range' 1077 64).map ColumnReplay.Operation.external

def fullSlice1Operations1 : List ColumnReplay.Operation :=
  (List.range' 1141 64).map ColumnReplay.Operation.external

def fullSlice1Operations2 : List ColumnReplay.Operation :=
  (List.range' 1205 64).map ColumnReplay.Operation.external

def fullSlice1Operations3 : List ColumnReplay.Operation :=
  (List.range' 1269 64).map ColumnReplay.Operation.external

def fullSlice1Operations : List ColumnReplay.Operation :=
  fullSlice1Operations0 ++ (fullSlice1Operations1 ++
    (fullSlice1Operations2 ++ fullSlice1Operations3))

def fullSlice1Start : ColumnReplay.Run :=
  checkpointRun (fun lane => 40237 + lane.val) ⟨0, by decide⟩ 0

def fullSlice1Run1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1137 + lane.val else 49237 + lane.val)
    ⟨4, by decide⟩ 15

def fullSlice1Run2 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1201 + lane.val else 58837 + lane.val)
    ⟨4, by decide⟩ 31

def fullSlice1Run3 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1265 + lane.val else 68437 + lane.val)
    ⟨4, by decide⟩ 47

def fullSlice1BeforeNormalize : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1329 + lane.val else 78037 + lane.val)
    ⟨4, by decide⟩ 63

def fullSlice1Result : ColumnReplay.Run :=
  checkpointRun (fun lane => 78637 + lane.val) ⟨0, by decide⟩ 64

def fullSlice1LastCall : Poseidon2Call.Call := {
  rowStart := 76678
  rowEnd := 77278
  inputColumns := [1329, 1330, 1331, 1332, 78041, 78042, 78043, 78044]
  firstAllocatedColumn := 78045
}

private theorem full_slice1_trace_length :
    fullSlice1Trace.calls.length = 64 :=
  fullChunk1_length

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
    fullSlice1BeforeNormalize.cursor fullSlice1LastCall
    full_slice1_last_call_bounded full_slice1_last_call_exact (by rfl)]
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
  unfold ColumnReplay.executeSlice fullSlice1Operations
  rw [execute_four_leaves full_slice1_leaf0 full_slice1_leaf1
    full_slice1_leaf2 full_slice1_leaf3]
  exact full_slice1_normalization

/-! ## Full arm: fields 512 through 767 -/

def fullSlice2Trace : TranscriptCertificate.Trace where
  pins := []
  calls := fullChunk2

def fullSlice2Operations0 : List ColumnReplay.Operation :=
  (List.range' 1333 64).map ColumnReplay.Operation.external

def fullSlice2Operations1 : List ColumnReplay.Operation :=
  (List.range' 1397 64).map ColumnReplay.Operation.external

def fullSlice2Operations2 : List ColumnReplay.Operation :=
  (List.range' 1461 64).map ColumnReplay.Operation.external

def fullSlice2Operations3 : List ColumnReplay.Operation :=
  (List.range' 1525 64).map ColumnReplay.Operation.external

def fullSlice2Operations : List ColumnReplay.Operation :=
  fullSlice2Operations0 ++ (fullSlice2Operations1 ++
    (fullSlice2Operations2 ++ fullSlice2Operations3))

def fullSlice2Start : ColumnReplay.Run :=
  checkpointRun (fun lane => 78637 + lane.val) ⟨0, by decide⟩ 0

def fullSlice2Run1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1393 + lane.val else 87637 + lane.val)
    ⟨4, by decide⟩ 15

def fullSlice2Run2 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1457 + lane.val else 97237 + lane.val)
    ⟨4, by decide⟩ 31

def fullSlice2Run3 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1521 + lane.val else 106837 + lane.val)
    ⟨4, by decide⟩ 47

def fullSlice2BeforeNormalize : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1585 + lane.val else 116437 + lane.val)
    ⟨4, by decide⟩ 63

def fullSlice2Result : ColumnReplay.Run :=
  checkpointRun (fun lane => 117037 + lane.val) ⟨0, by decide⟩ 64

def fullSlice2LastCall : Poseidon2Call.Call := {
  rowStart := 115078
  rowEnd := 115678
  inputColumns := [1585, 1586, 1587, 1588, 116441, 116442, 116443, 116444]
  firstAllocatedColumn := 116445
}

private theorem full_slice2_trace_length :
    fullSlice2Trace.calls.length = 64 :=
  fullChunk2_length

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
    fullSlice2BeforeNormalize.cursor fullSlice2LastCall
    full_slice2_last_call_bounded full_slice2_last_call_exact (by rfl)]
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
  unfold ColumnReplay.executeSlice fullSlice2Operations
  rw [execute_four_leaves full_slice2_leaf0 full_slice2_leaf1
    full_slice2_leaf2 full_slice2_leaf3]
  exact full_slice2_normalization

/-! ## Full arm: fields 768 through 1023 -/

def fullSlice3Trace : TranscriptCertificate.Trace where
  pins := []
  calls := fullChunk3

def fullSlice3Operations0 : List ColumnReplay.Operation :=
  (List.range' 1589 64).map ColumnReplay.Operation.external

def fullSlice3Operations1 : List ColumnReplay.Operation :=
  (List.range' 1653 64).map ColumnReplay.Operation.external

def fullSlice3Operations2 : List ColumnReplay.Operation :=
  (List.range' 1717 64).map ColumnReplay.Operation.external

def fullSlice3Operations3 : List ColumnReplay.Operation :=
  (List.range' 1781 64).map ColumnReplay.Operation.external

def fullSlice3Operations : List ColumnReplay.Operation :=
  fullSlice3Operations0 ++ (fullSlice3Operations1 ++
    (fullSlice3Operations2 ++ fullSlice3Operations3))

def fullSlice3Start : ColumnReplay.Run :=
  checkpointRun (fun lane => 117037 + lane.val) ⟨0, by decide⟩ 0

def fullSlice3Run1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1649 + lane.val else 126037 + lane.val)
    ⟨4, by decide⟩ 15

def fullSlice3Run2 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1713 + lane.val else 135637 + lane.val)
    ⟨4, by decide⟩ 31

def fullSlice3Run3 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1777 + lane.val else 145237 + lane.val)
    ⟨4, by decide⟩ 47

def fullSlice3BeforeNormalize : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1841 + lane.val else 154837 + lane.val)
    ⟨4, by decide⟩ 63

def fullSlice3Result : ColumnReplay.Run :=
  checkpointRun (fun lane => 155437 + lane.val) ⟨0, by decide⟩ 64

def fullSlice3LastCall : Poseidon2Call.Call := {
  rowStart := 153478
  rowEnd := 154078
  inputColumns := [1841, 1842, 1843, 1844, 154841, 154842, 154843, 154844]
  firstAllocatedColumn := 154845
}

private theorem full_slice3_trace_length :
    fullSlice3Trace.calls.length = 64 :=
  fullChunk3_length

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
    fullSlice3BeforeNormalize.cursor fullSlice3LastCall
    full_slice3_last_call_bounded full_slice3_last_call_exact (by rfl)]
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
  unfold ColumnReplay.executeSlice fullSlice3Operations
  rw [execute_four_leaves full_slice3_leaf0 full_slice3_leaf1
    full_slice3_leaf2 full_slice3_leaf3]
  exact full_slice3_normalization

/-! ## Final arm: fields 0 through 255 -/

def finalSlice0Trace : TranscriptCertificate.Trace where
  pins := []
  calls := finalChunk0

def finalSlice0LastCall : Poseidon2Call.Call := {
  rowStart := 38405
  rowEnd := 39005
  inputColumns := [1073, 1074, 1075, 1076, 39641, 39642, 39643, 39644]
  firstAllocatedColumn := 39645
}

private theorem final_slice0_trace_length :
    finalSlice0Trace.calls.length = 64 :=
  finalChunk0_length

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
      nextPin := fullSlice0BeforeNormalize.cursor.nextPin
      nextCall := fullSlice0BeforeNormalize.cursor.nextCall + 1
    } : ColumnReplay.Cursor) = fullSlice0Result.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem final_slice0_permute :
    ColumnReplay.permute finalSlice0Trace fullSlice0BeforeNormalize.cursor =
      some fullSlice0Result.cursor := by
  rw [ColumnReplayExecution.permute_of_call finalSlice0Trace
    fullSlice0BeforeNormalize.cursor finalSlice0LastCall
    final_slice0_last_call_bounded final_slice0_last_call_exact (by rfl)]
  exact congrArg some final_slice0_result_cursor_exact

private theorem final_slice0_leaf0 :
    ColumnReplay.execute finalSlice0Trace fullSlice0Start
        fullSlice0Operations0 = some fullSlice0Run1 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice0_leaf1 :
    ColumnReplay.execute finalSlice0Trace fullSlice0Run1
        fullSlice0Operations1 = some fullSlice0Run2 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice0_leaf2 :
    ColumnReplay.execute finalSlice0Trace fullSlice0Run2
        fullSlice0Operations2 = some fullSlice0Run3 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice0_leaf3 :
    ColumnReplay.execute finalSlice0Trace fullSlice0Run3
        fullSlice0Operations3 = some fullSlice0BeforeNormalize := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice0_normalization :
    ColumnReplay.normalizeSlice finalSlice0Trace fullSlice0BeforeNormalize =
      some fullSlice0Result := by
  unfold ColumnReplay.normalizeSlice
  split
  · rw [final_slice0_permute]
    rfl
  · rename_i notFull
    have full :
        rate ≤ fullSlice0BeforeNormalize.cursor.absorbed.val := by
      change 4 ≤ 4
      omega
    exact (notFull full).elim

theorem final_slice0_execution :
    ColumnReplay.executeSlice finalSlice0Trace fullSlice0Start
        fullSlice0Operations = some fullSlice0Result := by
  unfold ColumnReplay.executeSlice fullSlice0Operations
  rw [execute_four_leaves final_slice0_leaf0 final_slice0_leaf1
    final_slice0_leaf2 final_slice0_leaf3]
  exact final_slice0_normalization

/-! ## Final arm: fields 256 through 511 -/

def finalSlice1Trace : TranscriptCertificate.Trace where
  pins := []
  calls := finalChunk1

def finalSlice1LastCall : Poseidon2Call.Call := {
  rowStart := 76805
  rowEnd := 77405
  inputColumns := [1329, 1330, 1331, 1332, 78041, 78042, 78043, 78044]
  firstAllocatedColumn := 78045
}

private theorem final_slice1_trace_length :
    finalSlice1Trace.calls.length = 64 :=
  finalChunk1_length

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
      nextPin := fullSlice1BeforeNormalize.cursor.nextPin
      nextCall := fullSlice1BeforeNormalize.cursor.nextCall + 1
    } : ColumnReplay.Cursor) = fullSlice1Result.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem final_slice1_permute :
    ColumnReplay.permute finalSlice1Trace fullSlice1BeforeNormalize.cursor =
      some fullSlice1Result.cursor := by
  rw [ColumnReplayExecution.permute_of_call finalSlice1Trace
    fullSlice1BeforeNormalize.cursor finalSlice1LastCall
    final_slice1_last_call_bounded final_slice1_last_call_exact (by rfl)]
  exact congrArg some final_slice1_result_cursor_exact

private theorem final_slice1_leaf0 :
    ColumnReplay.execute finalSlice1Trace fullSlice1Start
        fullSlice1Operations0 = some fullSlice1Run1 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice1_leaf1 :
    ColumnReplay.execute finalSlice1Trace fullSlice1Run1
        fullSlice1Operations1 = some fullSlice1Run2 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice1_leaf2 :
    ColumnReplay.execute finalSlice1Trace fullSlice1Run2
        fullSlice1Operations2 = some fullSlice1Run3 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice1_leaf3 :
    ColumnReplay.execute finalSlice1Trace fullSlice1Run3
        fullSlice1Operations3 = some fullSlice1BeforeNormalize := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_slice1_normalization :
    ColumnReplay.normalizeSlice finalSlice1Trace fullSlice1BeforeNormalize =
      some fullSlice1Result := by
  unfold ColumnReplay.normalizeSlice
  split
  · rw [final_slice1_permute]
    rfl
  · rename_i notFull
    have full :
        rate ≤ fullSlice1BeforeNormalize.cursor.absorbed.val := by
      change 4 ≤ 4
      omega
    exact (notFull full).elim

theorem final_slice1_execution :
    ColumnReplay.executeSlice finalSlice1Trace fullSlice1Start
        fullSlice1Operations = some fullSlice1Result := by
  unfold ColumnReplay.executeSlice fullSlice1Operations
  rw [execute_four_leaves final_slice1_leaf0 final_slice1_leaf1
    final_slice1_leaf2 final_slice1_leaf3]
  exact final_slice1_normalization

/-! ## Final arm: fields 512 through 574 -/

def finalTailTrace : TranscriptCertificate.Trace where
  pins := []
  calls := finalChunk2

def finalTailOperations : List ColumnReplay.Operation :=
  (List.range' 1333 63).map ColumnReplay.Operation.external

def finalTailStart : ColumnReplay.Run :=
  checkpointRun (fun lane => 78637 + lane.val) ⟨0, by decide⟩ 0

def finalTailResult : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 3 then 1393 + lane.val else 87637 + lane.val)
    ⟨3, by decide⟩ 15

theorem final_tail_execution :
    ColumnReplay.executeSlice finalTailTrace finalTailStart
        finalTailOperations = some finalTailResult := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayTransitionExecutionCertificate
