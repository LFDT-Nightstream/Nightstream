import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayArtifact
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplay
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: exact physical Poseidon2 replay schedule for one streaming claim
chunk.

Owns the interpretation of the Rust artifact's call columns as a bounded
slice execution. It proves that all 1,024 full-chunk fields consume 256 calls
and that all 983 final-chunk fields consume 245 calls with three rate lanes
left pending.

Does not own glue-row decoding, equality with the public successor state, the
generic extracted-permutation-to-reference bridge, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplay
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript

/-- Compact call trace extracted from one exact Rust arm. -/
def traceFor (arm : RawArm) : TranscriptCertificate.Trace where
  pins := []
  calls := arm.poseidon2Calls

/-- Field column of one canonical public transition word. -/
def publicWordColumn (arm : RawArm) (index : Nat) : Nat :=
  (arm.canonicalCalls.getD index default).fieldColumn

/-- The chunk starts after the last canonical-word decomposition allocation. -/
def chunkBase (arm : RawArm) : Nat :=
  (arm.canonicalCalls.getD 39 default).inverseColumn + 1

def chunkColumn (arm : RawArm) (index : Nat) : Nat :=
  chunkBase arm + index

def chunkOperations (arm : RawArm) (count : Nat) :
    List ColumnReplay.Operation :=
  (List.range count).map fun index =>
    .external (chunkColumn arm index)

def afterRuntimeColumn (arm : RawArm) (lane : Fin 8) : Nat :=
  publicWordColumn arm (29 + lane.val)

/-- Runtime state columns in the public before-state block. -/
def startRun (arm : RawArm) : ColumnReplay.Run where
  cursor := {
    lanes := fun lane => publicWordColumn arm (9 + lane.val)
    absorbed := ⟨0, by decide⟩
    nextPin := 0
    nextCall := 0 }
  digests := []

def fullLastCall : Poseidon2Call.Call :=
  fullArm.poseidon2Calls.getD 255 default

/-- A full slice consumes the pending 256th call and returns at cursor zero. -/
def fullResult : ColumnReplay.Run where
  cursor := {
    lanes := ColumnReplay.callOutputColumns fullLastCall
    absorbed := ⟨0, by decide⟩
    nextPin := 0
    nextCall := 256 }
  digests := []

def finalLastCall : Poseidon2Call.Call :=
  finalArm.poseidon2Calls.getD 244 default

/-- The final slice consumes 245 calls. Its last three fields overwrite the
first three output lanes of the last call. -/
def finalResultLanes : Fin 8 → Nat := fun lane =>
  if lane.val < 3 then
    chunkColumn finalArm (980 + lane.val)
  else
    ColumnReplay.callOutputColumns finalLastCall lane

def finalResult : ColumnReplay.Run where
  cursor := {
    lanes := finalResultLanes
    absorbed := ⟨3, by decide⟩
    nextPin := 0
    nextCall := 245 }
  digests := []

private structure CursorView where
  lanes : List Nat
  absorbed : Nat
  nextPin : Nat
  nextCall : Nat
deriving DecidableEq

private def cursorView (cursor : ColumnReplay.Cursor) : CursorView where
  lanes := List.ofFn cursor.lanes
  absorbed := cursor.absorbed.val
  nextPin := cursor.nextPin
  nextCall := cursor.nextCall

private theorem cursorView_injective : Function.Injective cursorView := by
  intro left right equal
  cases left with
  | mk leftLanes leftAbsorbed leftPin leftCall =>
      cases right with
      | mk rightLanes rightAbsorbed rightPin rightCall =>
          have lanesEqual : leftLanes = rightLanes :=
            List.ofFn_injective (congrArg CursorView.lanes equal)
          have absorbedEqual : leftAbsorbed = rightAbsorbed :=
            Fin.ext (congrArg CursorView.absorbed equal)
          have pinEqual : leftPin = rightPin :=
            congrArg CursorView.nextPin equal
          have callEqual : leftCall = rightCall :=
            congrArg CursorView.nextCall equal
          subst rightLanes
          subst rightAbsorbed
          subst rightPin
          subst rightCall
          rfl

private structure RunView where
  cursor : CursorView
  digests : List (List Nat)
deriving DecidableEq

private def runView (run : ColumnReplay.Run) : RunView where
  cursor := cursorView run.cursor
  digests := run.digests.map List.ofFn

private theorem runView_injective : Function.Injective runView := by
  intro left right equal
  cases left with
  | mk leftCursor leftDigests =>
      cases right with
      | mk rightCursor rightDigests =>
          have cursorEqual : leftCursor = rightCursor :=
            cursorView_injective (congrArg RunView.cursor equal)
          have digestEqual : leftDigests = rightDigests := by
            apply (List.map_injective_iff.mpr fun first second valuesEqual =>
              List.ofFn_injective valuesEqual)
            exact congrArg RunView.digests equal
          subst rightCursor
          subst rightDigests
          rfl

private def executionMatches
    (result : Option ColumnReplay.Run) (expected : ColumnReplay.Run) : Bool :=
  match result with
  | none => false
  | some actual => decide (runView actual = runView expected)

private theorem executionMatches_sound
    {result : Option ColumnReplay.Run} {expected : ColumnReplay.Run}
    (checked : executionMatches result expected = true) :
    result = some expected := by
  cases result with
  | none => simp [executionMatches] at checked
  | some actual =>
      have viewsEqual : runView actual = runView expected := by
        exact of_decide_eq_true (by simpa [executionMatches] using checked)
      rw [runView_injective viewsEqual]

private theorem full_execution_checked :
    executionMatches
      (ColumnReplay.executeSlice (traceFor fullArm) (startRun fullArm)
        (chunkOperations fullArm 1024))
      fullResult = true := by
  native_decide

private theorem final_execution_checked :
    executionMatches
      (ColumnReplay.executeSlice (traceFor finalArm) (startRun finalArm)
        (chunkOperations finalArm 983))
      finalResult = true := by
  native_decide

/-- Exact compact execution certificate for a complete 1,024-field arm. -/
theorem full_execution :
    ColumnReplay.executeSlice (traceFor fullArm) (startRun fullArm)
        (chunkOperations fullArm 1024) =
      some fullResult := by
  exact executionMatches_sound full_execution_checked

/-- Exact compact execution certificate for the 983-field final arm. -/
theorem final_execution :
    ColumnReplay.executeSlice (traceFor finalArm) (startRun finalArm)
        (chunkOperations finalArm 983) =
      some finalResult := by
  exact executionMatches_sound final_execution_checked

theorem trace_pins_canonical (arm : RawArm) :
    ConstantPins.ValuesCanonical (traceFor arm).pins := by
  simp [traceFor, ConstantPins.ValuesCanonical]

/-- Satisfaction of every Rust-emitted call reconstructs acceptance of the
compact physical trace. -/
theorem trace_accepted
    (arm : RawArm) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : arm.Satisfied assignment) :
    (traceFor arm).Accepted assignment := by
  constructor
  · simp [traceFor]
  · intro call member
    exact poseidon2_call_refines arm assignment canonical one satisfied call
      (by simpa [traceFor] using member)

/-- Accepted full-arm rows refine the exact extracted bulk permutation
execution. This result still stops at the generated SSA permutation semantics. -/
theorem full_execution_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical (startRun fullArm))
        (chunkOperations fullArm 1024) =
      ColumnReplay.decodeRun assignment canonical fullResult := by
  apply ColumnReplay.executeSlice_sound canonical (trace_pins_canonical fullArm)
    one (trace_accepted fullArm assignment canonical one satisfied)
  exact full_execution

/-- Accepted final-arm rows refine the exact extracted bulk permutation
execution and preserve the three-field final cursor. -/
theorem final_execution_refines
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical (startRun finalArm))
        (chunkOperations finalArm 983) =
      ColumnReplay.decodeRun assignment canonical finalResult := by
  apply ColumnReplay.executeSlice_sound canonical (trace_pins_canonical finalArm)
    one (trace_accepted finalArm assignment canonical one satisfied)
  exact final_execution

private def fullOutputGlue (lane : Fin 8) : IndexedRow :=
  fullArm.glueRows.get ⟨16 + lane.val, by
    have lengthExact : fullArm.glueRows.length = 24 := by native_decide
    rw [lengthExact]
    omega⟩

private def finalOutputGlue (lane : Fin 8) : IndexedRow :=
  finalArm.glueRows.get ⟨59 + lane.val, by
    have lengthExact : finalArm.glueRows.length = 77 := by native_decide
    rw [lengthExact]
    omega⟩

private theorem fullOutputGlue_rows :
    ∀ lane : Fin 8,
      (fullOutputGlue lane).row = EqualityPins.equalityRow
        (afterRuntimeColumn fullArm lane, fullResult.cursor.lanes lane) := by
  native_decide

private theorem finalOutputGlue_rows :
    ∀ lane : Fin 8,
      (finalOutputGlue lane).row = EqualityPins.equalityRow
        (afterRuntimeColumn finalArm lane, finalResult.cursor.lanes lane) := by
  native_decide

private theorem equalityRow_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) {left right : Nat}
    (holds : RowHolds assignment (EqualityPins.equalityRow (left, right))) :
    assignment left = assignment right := by
  have singleton : Satisfies (EqualityPins.rows [(left, right)]) assignment := by
    intro row member
    simp only [EqualityPins.rows, List.map_cons, List.map_nil,
      List.mem_cons, List.not_mem_nil, or_false] at member
    subst row
    exact holds
  exact EqualityPins.rows_sound canonical one singleton (left, right)
    (by simp)

/-- The full-arm output glue identifies every declared runtime lane with the
last bulk-permutation output lane. -/
theorem full_output_lanes
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ∀ lane : Fin 8,
      assignment (afterRuntimeColumn fullArm lane) =
        assignment (fullResult.cursor.lanes lane) := by
  intro lane
  have holds := glue_row_holds fullArm assignment satisfied
    (fullOutputGlue lane) (List.get_mem _ _)
  rw [fullOutputGlue_rows lane] at holds
  exact equalityRow_sound canonical one holds

/-- The final-arm output glue includes the final three direct overwrites and
the five unchanged lanes from the last permutation. -/
theorem final_output_lanes
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    ∀ lane : Fin 8,
      assignment (afterRuntimeColumn finalArm lane) =
        assignment (finalResult.cursor.lanes lane) := by
  intro lane
  have holds := glue_row_holds finalArm assignment satisfied
    (finalOutputGlue lane) (List.get_mem _ _)
  rw [finalOutputGlue_rows lane] at holds
  exact equalityRow_sound canonical one holds

def declaredRuntimeState
    (arm : RawArm) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (absorbed : Fin 5) :
    PiRlcChallenge.TranscriptMachine.State where
  lanes := fun lane =>
    PiRlcChallenge.Transcript.CallRefinement.fieldAt assignment canonical
      (afterRuntimeColumn arm lane)
  absorbed := absorbed

private theorem transcriptStateExt
    {left right : PiRlcChallenge.TranscriptMachine.State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

/-- The full arm's declared public successor is the state reconstructed from
the complete accepted Rust call trace. -/
theorem full_declared_runtime_eq_result
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    declaredRuntimeState fullArm assignment canonical ⟨0, by decide⟩ =
      (ColumnReplay.decodeRun assignment canonical fullResult).state := by
  apply transcriptStateExt
  · funext lane
    apply Fin.ext
    exact full_output_lanes assignment canonical one satisfied lane
  · rfl

/-- The final arm's declared public successor is the state reconstructed from
the complete accepted Rust call trace. -/
theorem final_declared_runtime_eq_result
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    declaredRuntimeState finalArm assignment canonical ⟨3, by decide⟩ =
      (ColumnReplay.decodeRun assignment canonical finalResult).state := by
  apply transcriptStateExt
  · funext lane
    apply Fin.ext
    exact final_output_lanes assignment canonical one satisfied lane
  · rfl

/-- Same-assignment conformance for the complete full-chunk Poseidon2 path,
from public before-state columns and private chunk columns to public
successor-state columns. -/
theorem full_rows_refine_declared_runtime
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    (ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical (startRun fullArm))
        (chunkOperations fullArm 1024)).state =
      declaredRuntimeState fullArm assignment canonical ⟨0, by decide⟩ := by
  exact (congrArg ColumnReplay.SemanticRun.state
    (full_execution_refines assignment canonical one satisfied)).trans
    (full_declared_runtime_eq_result assignment canonical one satisfied).symm

/-- Same-assignment conformance for the complete final-chunk Poseidon2 path. -/
theorem final_rows_refine_declared_runtime
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArm.Satisfied assignment) :
    (ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical (startRun finalArm))
        (chunkOperations finalArm 983)).state =
      declaredRuntimeState finalArm assignment canonical ⟨3, by decide⟩ := by
  exact (congrArg ColumnReplay.SemanticRun.state
    (final_execution_refines assignment canonical one satisfied)).trans
    (final_declared_runtime_eq_result assignment canonical one satisfied).symm

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution
