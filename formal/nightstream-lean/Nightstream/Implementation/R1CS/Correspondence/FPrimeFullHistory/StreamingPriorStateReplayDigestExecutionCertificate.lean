import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPriorStateReplaySource
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplayExecution

/-!
Bounded structural execution certificates for prior-state digest traces.

Owns the local verifier-pinned state headers, ten authoritative state columns,
and the exact final digest call. It owns no transition-slice execution, row
satisfaction, semantic refinement, lifecycle selection, or digest authority.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayDigestExecutionCertificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

def checkpointRun
    (lanes : Fin width → Nat) (absorbed : Fin (rate + 1))
    (nextPin nextCall : Nat)
    (digests : List (Fin 4 → Nat) := []) : ColumnReplay.Run where
  cursor := { lanes, absorbed, nextPin, nextCall }
  digests := digests

def statePins
    (initialStart frameStart padColumn queryStart : Nat) :
    List (Nat × Nat) :=
  [(initialStart, 2644702416324735075),
   (initialStart + 1, 8852586734026622474),
   (initialStart + 2, 10961611613478088853),
   (initialStart + 3, 3758899379070171657),
   (initialStart + 4, 9085448732628946009),
   (initialStart + 5, 13680608928383082747),
   (initialStart + 6, 1991093790229263654),
   (initialStart + 7, 6906233132260090641),
   (frameStart, 2), (frameStart + 1, 5),
   (frameStart + 2, 435744240755), (frameStart + 3, 10),
   (padColumn, 1), (queryStart, 260), (queryStart + 1, 32)]

def statePinOperations : List ColumnReplay.Operation :=
  [.pinned 2, .pinned 5, .pinned 435744240755, .pinned 10]

def stateExternalOperations0 (stateStart : Nat) : List ColumnReplay.Operation :=
  (List.range' stateStart 4).map ColumnReplay.Operation.external

def stateExternalOperations1 (stateStart : Nat) : List ColumnReplay.Operation :=
  (List.range' (stateStart + 4) 4).map ColumnReplay.Operation.external

def stateExternalOperations2 (stateStart : Nat) : List ColumnReplay.Operation :=
  (List.range' (stateStart + 8) 2).map ColumnReplay.Operation.external

def stateDigestOperations : List ColumnReplay.Operation := [.digest]

def stateOperations (stateStart : Nat) : List ColumnReplay.Operation :=
  statePinOperations ++ (stateExternalOperations0 stateStart ++
    (stateExternalOperations1 stateStart ++
      (stateExternalOperations2 stateStart ++ stateDigestOperations)))

def fullBeforeTrace : TranscriptCertificate.Trace where
  pins := statePins 154779 154787 156591 157192
  calls := fullCallsPart4.take 4

def fullBeforeStart : ColumnReplay.Run :=
  checkpointRun (fun lane => 154779 + lane.val) ⟨0, by decide⟩ 8 0

def fullBeforeAfterPins : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 154787 + lane.val else 154779 + lane.val)
    ⟨4, by decide⟩ 12 0

def fullBeforeAfterExternal0 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1 + lane.val else 155383 + lane.val)
    ⟨4, by decide⟩ 12 1

def fullBeforeAfterExternal1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 5 + lane.val else 155983 + lane.val)
    ⟨4, by decide⟩ 12 2

def fullBeforeBeforeDigest : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 2 then 9 + lane.val else 156583 + lane.val)
    ⟨2, by decide⟩ 12 3

def fullBeforeBeforePermutation : ColumnReplay.Cursor where
  lanes := fun lane =>
    if lane.val = 2 then 156591 else fullBeforeBeforeDigest.cursor.lanes lane
  absorbed := ⟨3, by decide⟩
  nextPin := 13
  nextCall := 3

def fullBeforeResult : ColumnReplay.Run :=
  checkpointRun (fun lane => 157184 + lane.val) ⟨0, by decide⟩ 13 4
    [fun lane => 157184 + lane.val]

def fullBeforeLastCall : Poseidon2Call.Call := {
  rowStart := 155564
  rowEnd := 156164
  inputColumns := [9, 10, 156591, 156586, 156587, 156588, 156589, 156590]
  firstAllocatedColumn := 156592
}

private theorem full_before_trace_length : fullBeforeTrace.calls.length = 4 := by
  rfl

private theorem full_before_last_call_bounded :
    3 < fullBeforeTrace.calls.length := by
  rw [full_before_trace_length]
  decide

private theorem full_before_last_call_exact :
    fullBeforeTrace.calls.get ⟨3, full_before_last_call_bounded⟩ =
      fullBeforeLastCall := by
  rfl

private theorem full_before_absorb_digest_pin :
    ColumnReplay.absorbPinned fullBeforeTrace fullBeforeBeforeDigest.cursor 1 =
      some fullBeforeBeforePermutation := by
  rfl

private theorem full_before_result_cursor_exact :
    ({
      lanes := ColumnReplay.callOutputColumns fullBeforeLastCall
      absorbed := ⟨0, by decide⟩
      nextPin := fullBeforeBeforePermutation.nextPin
      nextCall := fullBeforeBeforePermutation.nextCall + 1
    } : ColumnReplay.Cursor) = fullBeforeResult.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem full_before_permute :
    ColumnReplay.permute fullBeforeTrace fullBeforeBeforePermutation =
      some fullBeforeResult.cursor := by
  rw [ColumnReplayExecution.permute_of_call fullBeforeTrace
    fullBeforeBeforePermutation fullBeforeLastCall
    full_before_last_call_bounded full_before_last_call_exact (by rfl)]
  exact congrArg some full_before_result_cursor_exact

private theorem full_before_digest :
    ColumnReplay.digest fullBeforeTrace fullBeforeBeforeDigest.cursor =
      some (fullBeforeResult.cursor, fun lane => 157184 + lane.val) := by
  unfold ColumnReplay.digest
  rw [full_before_absorb_digest_pin]
  change (ColumnReplay.permute fullBeforeTrace fullBeforeBeforePermutation).bind
      (fun afterPermutation => some
        (afterPermutation, ColumnReplay.firstFourColumns afterPermutation)) =
    some (fullBeforeResult.cursor, fun lane => 157184 + lane.val)
  rw [full_before_permute]
  rfl

private theorem full_before_pin_leaf :
    ColumnReplay.execute fullBeforeTrace fullBeforeStart statePinOperations =
      some fullBeforeAfterPins := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_before_external0_leaf :
    ColumnReplay.execute fullBeforeTrace fullBeforeAfterPins
        (stateExternalOperations0 1) = some fullBeforeAfterExternal0 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_before_external1_leaf :
    ColumnReplay.execute fullBeforeTrace fullBeforeAfterExternal0
        (stateExternalOperations1 1) = some fullBeforeAfterExternal1 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_before_external2_leaf :
    ColumnReplay.execute fullBeforeTrace fullBeforeAfterExternal1
        (stateExternalOperations2 1) = some fullBeforeBeforeDigest := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_before_digest_step :
    ColumnReplay.step fullBeforeTrace fullBeforeBeforeDigest .digest =
      some fullBeforeResult := by
  unfold ColumnReplay.step
  rw [full_before_digest]
  apply congrArg some
  apply ColumnReplayExecution.runView_injective
  rfl

private theorem full_before_digest_leaf :
    ColumnReplay.execute fullBeforeTrace fullBeforeBeforeDigest
        stateDigestOperations = some fullBeforeResult := by
  unfold stateDigestOperations ColumnReplay.execute
  rw [full_before_digest_step]
  rfl

theorem full_before_state_execution :
    ColumnReplay.execute fullBeforeTrace fullBeforeStart (stateOperations 1) =
      some fullBeforeResult := by
  unfold stateOperations
  exact ColumnReplay.execute_append full_before_pin_leaf
    (ColumnReplay.execute_append full_before_external0_leaf
      (ColumnReplay.execute_append full_before_external1_leaf
        (ColumnReplay.execute_append full_before_external2_leaf
          full_before_digest_leaf)))

def fullAfterTrace : TranscriptCertificate.Trace where
  pins := statePins 157194 157202 159006 159607
  calls := (fullCallsPart4.drop 4).take 4

def fullAfterStart : ColumnReplay.Run :=
  checkpointRun (fun lane => 157194 + lane.val) ⟨0, by decide⟩ 8 0

def fullAfterAfterPins : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 157202 + lane.val else 157194 + lane.val)
    ⟨4, by decide⟩ 12 0

def fullAfterAfterExternal0 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 11 + lane.val else 157798 + lane.val)
    ⟨4, by decide⟩ 12 1

def fullAfterAfterExternal1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 15 + lane.val else 158398 + lane.val)
    ⟨4, by decide⟩ 12 2

def fullAfterBeforeDigest : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 2 then 19 + lane.val else 158998 + lane.val)
    ⟨2, by decide⟩ 12 3

def fullAfterBeforePermutation : ColumnReplay.Cursor where
  lanes := fun lane =>
    if lane.val = 2 then 159006 else fullAfterBeforeDigest.cursor.lanes lane
  absorbed := ⟨3, by decide⟩
  nextPin := 13
  nextCall := 3

def fullAfterResult : ColumnReplay.Run :=
  checkpointRun (fun lane => 159599 + lane.val) ⟨0, by decide⟩ 13 4
    [fun lane => 159599 + lane.val]

def fullAfterLastCall : Poseidon2Call.Call := {
  rowStart := 157979
  rowEnd := 158579
  inputColumns := [19, 20, 159006, 159001, 159002, 159003, 159004, 159005]
  firstAllocatedColumn := 159007
}

private theorem full_after_trace_length : fullAfterTrace.calls.length = 4 := by
  rfl

private theorem full_after_last_call_bounded :
    3 < fullAfterTrace.calls.length := by
  rw [full_after_trace_length]
  decide

private theorem full_after_last_call_exact :
    fullAfterTrace.calls.get ⟨3, full_after_last_call_bounded⟩ =
      fullAfterLastCall := by
  rfl

private theorem full_after_absorb_digest_pin :
    ColumnReplay.absorbPinned fullAfterTrace fullAfterBeforeDigest.cursor 1 =
      some fullAfterBeforePermutation := by
  rfl

private theorem full_after_result_cursor_exact :
    ({
      lanes := ColumnReplay.callOutputColumns fullAfterLastCall
      absorbed := ⟨0, by decide⟩
      nextPin := fullAfterBeforePermutation.nextPin
      nextCall := fullAfterBeforePermutation.nextCall + 1
    } : ColumnReplay.Cursor) = fullAfterResult.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem full_after_permute :
    ColumnReplay.permute fullAfterTrace fullAfterBeforePermutation =
      some fullAfterResult.cursor := by
  rw [ColumnReplayExecution.permute_of_call fullAfterTrace
    fullAfterBeforePermutation fullAfterLastCall
    full_after_last_call_bounded full_after_last_call_exact (by rfl)]
  exact congrArg some full_after_result_cursor_exact

private theorem full_after_digest :
    ColumnReplay.digest fullAfterTrace fullAfterBeforeDigest.cursor =
      some (fullAfterResult.cursor, fun lane => 159599 + lane.val) := by
  unfold ColumnReplay.digest
  rw [full_after_absorb_digest_pin]
  change (ColumnReplay.permute fullAfterTrace fullAfterBeforePermutation).bind
      (fun afterPermutation => some
        (afterPermutation, ColumnReplay.firstFourColumns afterPermutation)) =
    some (fullAfterResult.cursor, fun lane => 159599 + lane.val)
  rw [full_after_permute]
  rfl

private theorem full_after_pin_leaf :
    ColumnReplay.execute fullAfterTrace fullAfterStart statePinOperations =
      some fullAfterAfterPins := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_after_external0_leaf :
    ColumnReplay.execute fullAfterTrace fullAfterAfterPins
        (stateExternalOperations0 11) = some fullAfterAfterExternal0 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_after_external1_leaf :
    ColumnReplay.execute fullAfterTrace fullAfterAfterExternal0
        (stateExternalOperations1 11) = some fullAfterAfterExternal1 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_after_external2_leaf :
    ColumnReplay.execute fullAfterTrace fullAfterAfterExternal1
        (stateExternalOperations2 11) = some fullAfterBeforeDigest := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem full_after_digest_step :
    ColumnReplay.step fullAfterTrace fullAfterBeforeDigest .digest =
      some fullAfterResult := by
  unfold ColumnReplay.step
  rw [full_after_digest]
  apply congrArg some
  apply ColumnReplayExecution.runView_injective
  rfl

private theorem full_after_digest_leaf :
    ColumnReplay.execute fullAfterTrace fullAfterBeforeDigest
        stateDigestOperations = some fullAfterResult := by
  unfold stateDigestOperations ColumnReplay.execute
  rw [full_after_digest_step]
  rfl

theorem full_after_state_execution :
    ColumnReplay.execute fullAfterTrace fullAfterStart (stateOperations 11) =
      some fullAfterResult := by
  unfold stateOperations
  exact ColumnReplay.execute_append full_after_pin_leaf
    (ColumnReplay.execute_append full_after_external0_leaf
      (ColumnReplay.execute_append full_after_external1_leaf
        (ColumnReplay.execute_append full_after_external2_leaf
          full_after_digest_leaf)))

def finalBeforeTrace : TranscriptCertificate.Trace where
  pins := statePins 79786 79794 81598 82199
  calls := (finalCallsPart2.drop 3).take 4

def finalBeforeStart : ColumnReplay.Run :=
  checkpointRun (fun lane => 79786 + lane.val) ⟨0, by decide⟩ 8 0

def finalBeforeAfterPins : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 79794 + lane.val else 79786 + lane.val)
    ⟨4, by decide⟩ 12 0

def finalBeforeAfterExternal0 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 1 + lane.val else 80390 + lane.val)
    ⟨4, by decide⟩ 12 1

def finalBeforeAfterExternal1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 5 + lane.val else 80990 + lane.val)
    ⟨4, by decide⟩ 12 2

def finalBeforeBeforeDigest : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 2 then 9 + lane.val else 81590 + lane.val)
    ⟨2, by decide⟩ 12 3

def finalBeforeBeforePermutation : ColumnReplay.Cursor where
  lanes := fun lane =>
    if lane.val = 2 then 81598 else finalBeforeBeforeDigest.cursor.lanes lane
  absorbed := ⟨3, by decide⟩
  nextPin := 13
  nextCall := 3

def finalBeforeResult : ColumnReplay.Run :=
  checkpointRun (fun lane => 82191 + lane.val) ⟨0, by decide⟩ 13 4
    [fun lane => 82191 + lane.val]

def finalBeforeLastCall : Poseidon2Call.Call := {
  rowStart := 81076
  rowEnd := 81676
  inputColumns := [9, 10, 81598, 81593, 81594, 81595, 81596, 81597]
  firstAllocatedColumn := 81599
}

private theorem final_before_trace_length : finalBeforeTrace.calls.length = 4 := by
  rfl

private theorem final_before_last_call_bounded :
    3 < finalBeforeTrace.calls.length := by
  rw [final_before_trace_length]
  decide

private theorem final_before_last_call_exact :
    finalBeforeTrace.calls.get ⟨3, final_before_last_call_bounded⟩ =
      finalBeforeLastCall := by
  rfl

private theorem final_before_absorb_digest_pin :
    ColumnReplay.absorbPinned finalBeforeTrace finalBeforeBeforeDigest.cursor 1 =
      some finalBeforeBeforePermutation := by
  rfl

private theorem final_before_result_cursor_exact :
    ({
      lanes := ColumnReplay.callOutputColumns finalBeforeLastCall
      absorbed := ⟨0, by decide⟩
      nextPin := finalBeforeBeforePermutation.nextPin
      nextCall := finalBeforeBeforePermutation.nextCall + 1
    } : ColumnReplay.Cursor) = finalBeforeResult.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem final_before_permute :
    ColumnReplay.permute finalBeforeTrace finalBeforeBeforePermutation =
      some finalBeforeResult.cursor := by
  rw [ColumnReplayExecution.permute_of_call finalBeforeTrace
    finalBeforeBeforePermutation finalBeforeLastCall
    final_before_last_call_bounded final_before_last_call_exact (by rfl)]
  exact congrArg some final_before_result_cursor_exact

private theorem final_before_digest :
    ColumnReplay.digest finalBeforeTrace finalBeforeBeforeDigest.cursor =
      some (finalBeforeResult.cursor, fun lane => 82191 + lane.val) := by
  unfold ColumnReplay.digest
  rw [final_before_absorb_digest_pin]
  change (ColumnReplay.permute finalBeforeTrace finalBeforeBeforePermutation).bind
      (fun afterPermutation => some
        (afterPermutation, ColumnReplay.firstFourColumns afterPermutation)) =
    some (finalBeforeResult.cursor, fun lane => 82191 + lane.val)
  rw [final_before_permute]
  rfl

private theorem final_before_pin_leaf :
    ColumnReplay.execute finalBeforeTrace finalBeforeStart statePinOperations =
      some finalBeforeAfterPins := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_before_external0_leaf :
    ColumnReplay.execute finalBeforeTrace finalBeforeAfterPins
        (stateExternalOperations0 1) = some finalBeforeAfterExternal0 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_before_external1_leaf :
    ColumnReplay.execute finalBeforeTrace finalBeforeAfterExternal0
        (stateExternalOperations1 1) = some finalBeforeAfterExternal1 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_before_external2_leaf :
    ColumnReplay.execute finalBeforeTrace finalBeforeAfterExternal1
        (stateExternalOperations2 1) = some finalBeforeBeforeDigest := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_before_digest_step :
    ColumnReplay.step finalBeforeTrace finalBeforeBeforeDigest .digest =
      some finalBeforeResult := by
  unfold ColumnReplay.step
  rw [final_before_digest]
  apply congrArg some
  apply ColumnReplayExecution.runView_injective
  rfl

private theorem final_before_digest_leaf :
    ColumnReplay.execute finalBeforeTrace finalBeforeBeforeDigest
        stateDigestOperations = some finalBeforeResult := by
  unfold stateDigestOperations ColumnReplay.execute
  rw [final_before_digest_step]
  rfl

theorem final_before_state_execution :
    ColumnReplay.execute finalBeforeTrace finalBeforeStart (stateOperations 1) =
      some finalBeforeResult := by
  unfold stateOperations
  exact ColumnReplay.execute_append final_before_pin_leaf
    (ColumnReplay.execute_append final_before_external0_leaf
      (ColumnReplay.execute_append final_before_external1_leaf
        (ColumnReplay.execute_append final_before_external2_leaf
          final_before_digest_leaf)))

def finalAfterTrace : TranscriptCertificate.Trace where
  pins := statePins 82201 82209 84013 84614
  calls := (finalCallsPart2.drop 7).take 4

def finalAfterStart : ColumnReplay.Run :=
  checkpointRun (fun lane => 82201 + lane.val) ⟨0, by decide⟩ 8 0

def finalAfterAfterPins : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 82209 + lane.val else 82201 + lane.val)
    ⟨4, by decide⟩ 12 0

def finalAfterAfterExternal0 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 11 + lane.val else 82805 + lane.val)
    ⟨4, by decide⟩ 12 1

def finalAfterAfterExternal1 : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 4 then 15 + lane.val else 83405 + lane.val)
    ⟨4, by decide⟩ 12 2

def finalAfterBeforeDigest : ColumnReplay.Run :=
  checkpointRun
    (fun lane => if lane.val < 2 then 19 + lane.val else 84005 + lane.val)
    ⟨2, by decide⟩ 12 3

def finalAfterBeforePermutation : ColumnReplay.Cursor where
  lanes := fun lane =>
    if lane.val = 2 then 84013 else finalAfterBeforeDigest.cursor.lanes lane
  absorbed := ⟨3, by decide⟩
  nextPin := 13
  nextCall := 3

def finalAfterResult : ColumnReplay.Run :=
  checkpointRun (fun lane => 84606 + lane.val) ⟨0, by decide⟩ 13 4
    [fun lane => 84606 + lane.val]

def finalAfterLastCall : Poseidon2Call.Call := {
  rowStart := 83491
  rowEnd := 84091
  inputColumns := [19, 20, 84013, 84008, 84009, 84010, 84011, 84012]
  firstAllocatedColumn := 84014
}

private theorem final_after_trace_length : finalAfterTrace.calls.length = 4 := by
  rfl

private theorem final_after_last_call_bounded :
    3 < finalAfterTrace.calls.length := by
  rw [final_after_trace_length]
  decide

private theorem final_after_last_call_exact :
    finalAfterTrace.calls.get ⟨3, final_after_last_call_bounded⟩ =
      finalAfterLastCall := by
  rfl

private theorem final_after_absorb_digest_pin :
    ColumnReplay.absorbPinned finalAfterTrace finalAfterBeforeDigest.cursor 1 =
      some finalAfterBeforePermutation := by
  rfl

private theorem final_after_result_cursor_exact :
    ({
      lanes := ColumnReplay.callOutputColumns finalAfterLastCall
      absorbed := ⟨0, by decide⟩
      nextPin := finalAfterBeforePermutation.nextPin
      nextCall := finalAfterBeforePermutation.nextCall + 1
    } : ColumnReplay.Cursor) = finalAfterResult.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem final_after_permute :
    ColumnReplay.permute finalAfterTrace finalAfterBeforePermutation =
      some finalAfterResult.cursor := by
  rw [ColumnReplayExecution.permute_of_call finalAfterTrace
    finalAfterBeforePermutation finalAfterLastCall
    final_after_last_call_bounded final_after_last_call_exact (by rfl)]
  exact congrArg some final_after_result_cursor_exact

private theorem final_after_digest :
    ColumnReplay.digest finalAfterTrace finalAfterBeforeDigest.cursor =
      some (finalAfterResult.cursor, fun lane => 84606 + lane.val) := by
  unfold ColumnReplay.digest
  rw [final_after_absorb_digest_pin]
  change (ColumnReplay.permute finalAfterTrace finalAfterBeforePermutation).bind
      (fun afterPermutation => some
        (afterPermutation, ColumnReplay.firstFourColumns afterPermutation)) =
    some (finalAfterResult.cursor, fun lane => 84606 + lane.val)
  rw [final_after_permute]
  rfl

private theorem final_after_pin_leaf :
    ColumnReplay.execute finalAfterTrace finalAfterStart statePinOperations =
      some finalAfterAfterPins := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_after_external0_leaf :
    ColumnReplay.execute finalAfterTrace finalAfterAfterPins
        (stateExternalOperations0 11) = some finalAfterAfterExternal0 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_after_external1_leaf :
    ColumnReplay.execute finalAfterTrace finalAfterAfterExternal0
        (stateExternalOperations1 11) = some finalAfterAfterExternal1 := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_after_external2_leaf :
    ColumnReplay.execute finalAfterTrace finalAfterAfterExternal1
        (stateExternalOperations2 11) = some finalAfterBeforeDigest := by
  apply ColumnReplayExecution.executionMatches_sound
  rfl

private theorem final_after_digest_step :
    ColumnReplay.step finalAfterTrace finalAfterBeforeDigest .digest =
      some finalAfterResult := by
  unfold ColumnReplay.step
  rw [final_after_digest]
  apply congrArg some
  apply ColumnReplayExecution.runView_injective
  rfl

private theorem final_after_digest_leaf :
    ColumnReplay.execute finalAfterTrace finalAfterBeforeDigest
        stateDigestOperations = some finalAfterResult := by
  unfold stateDigestOperations ColumnReplay.execute
  rw [final_after_digest_step]
  rfl

theorem final_after_state_execution :
    ColumnReplay.execute finalAfterTrace finalAfterStart (stateOperations 11) =
      some finalAfterResult := by
  unfold stateOperations
  exact ColumnReplay.execute_append final_after_pin_leaf
    (ColumnReplay.execute_append final_after_external0_leaf
      (ColumnReplay.execute_append final_after_external1_leaf
        (ColumnReplay.execute_append final_after_external2_leaf
          final_after_digest_leaf)))

def finalTargetTrace : TranscriptCertificate.Trace where
  pins := [(79183, 1), (79784, 260), (79785, 32)]
  calls := (finalCallsPart2.drop 2).take 1

def finalTargetStart : ColumnReplay.Run :=
  checkpointRun (fun lane => 11 + lane.val) ⟨2, by decide⟩ 0 0

def finalTargetBeforePermutation : ColumnReplay.Cursor where
  lanes := fun lane =>
    if lane.val = 2 then 79183 else finalTargetStart.cursor.lanes lane
  absorbed := ⟨3, by decide⟩
  nextPin := 1
  nextCall := 0

def finalTargetResult : ColumnReplay.Run :=
  checkpointRun (fun lane => 79776 + lane.val) ⟨0, by decide⟩ 1 1
    [fun lane => 79776 + lane.val]

def finalTargetCall : Poseidon2Call.Call := {
  rowStart := 78657
  rowEnd := 79257
  inputColumns := [11, 12, 79183, 14, 15, 16, 17, 18]
  firstAllocatedColumn := 79184
}

private theorem final_target_trace_length : finalTargetTrace.calls.length = 1 := by
  rfl

private theorem final_target_call_bounded :
    0 < finalTargetTrace.calls.length := by
  rw [final_target_trace_length]
  decide

private theorem final_target_call_exact :
    finalTargetTrace.calls.get ⟨0, final_target_call_bounded⟩ =
      finalTargetCall := by
  rfl

private theorem final_target_absorb_digest_pin :
    ColumnReplay.absorbPinned finalTargetTrace finalTargetStart.cursor 1 =
      some finalTargetBeforePermutation := by
  rfl

private theorem final_target_result_cursor_exact :
    ({
      lanes := ColumnReplay.callOutputColumns finalTargetCall
      absorbed := ⟨0, by decide⟩
      nextPin := finalTargetBeforePermutation.nextPin
      nextCall := finalTargetBeforePermutation.nextCall + 1
    } : ColumnReplay.Cursor) = finalTargetResult.cursor := by
  apply ColumnReplayExecution.cursorView_injective
  rfl

private theorem final_target_permute :
    ColumnReplay.permute finalTargetTrace finalTargetBeforePermutation =
      some finalTargetResult.cursor := by
  rw [ColumnReplayExecution.permute_of_call finalTargetTrace
    finalTargetBeforePermutation finalTargetCall final_target_call_bounded
    final_target_call_exact (by rfl)]
  exact congrArg some final_target_result_cursor_exact

private theorem final_target_digest :
    ColumnReplay.digest finalTargetTrace finalTargetStart.cursor =
      some (finalTargetResult.cursor, fun lane => 79776 + lane.val) := by
  unfold ColumnReplay.digest
  rw [final_target_absorb_digest_pin]
  change (ColumnReplay.permute finalTargetTrace finalTargetBeforePermutation).bind
      (fun afterPermutation => some
        (afterPermutation, ColumnReplay.firstFourColumns afterPermutation)) =
    some (finalTargetResult.cursor, fun lane => 79776 + lane.val)
  rw [final_target_permute]
  rfl

private theorem final_target_digest_step :
    ColumnReplay.step finalTargetTrace finalTargetStart .digest =
      some finalTargetResult := by
  unfold ColumnReplay.step
  rw [final_target_digest]
  apply congrArg some
  apply ColumnReplayExecution.runView_injective
  rfl

theorem final_target_execution :
    ColumnReplay.execute finalTargetTrace finalTargetStart
        stateDigestOperations = some finalTargetResult := by
  unfold stateDigestOperations ColumnReplay.execute
  rw [final_target_digest_step]
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayDigestExecutionCertificate
