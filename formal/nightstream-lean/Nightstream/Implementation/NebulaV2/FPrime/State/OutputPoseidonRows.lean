import Nightstream.Implementation.NebulaV2.FPrime.State.OutputFrameRows
import Nightstream.Implementation.NebulaV2.Memory.Carry.PoseidonRows

/-!
Contract: exact outer Poseidon2 state-output relation for the fixed 32-field
V2 stateful-with-Nebula frame.

Assurance tier: implementation model and cryptographic primitive semantics.

Owns eight full absorb permutations, one terminal-padding permutation,
structural trace linkage, row soundness to a pure column-independent digest,
and honest row completeness from independent permutation executions.

Does not own the carry-digest computation, outer collision resistance,
placement of non-memory recursive-state fields, absolute generated columns,
or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.StateOutputPoseidonRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonRows
open Nightstream.Implementation.NebulaV2.StateOutputFrameRows

def expectedSchedule : List ValueSchedule :=
  List.replicate 8 (.absorb 4) ++ [.pad]

theorem expectedSchedule_exact :
    expectedSchedule.length = 9 ∧
      (expectedSchedule.filter (· = .absorb 4)).length = 8 ∧
      (expectedSchedule.filter (· = .pad)).length = 1 := by
  decide

def representativeRounds : List Round :=
  expectedSchedule.map MemoryCarryPoseidonRows.representativeRound

theorem representativeRounds_schedule :
    valueSchedules representativeRounds = expectedSchedule := by
  rw [representativeRounds, valueSchedules, List.map_map]
  change
    expectedSchedule.map
        (fun schedule =>
          (MemoryCarryPoseidonRows.representativeRound schedule).valueSchedule) =
      expectedSchedule
  generalize expectedSchedule = schedules
  induction schedules with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.cons.injEq]
      exact ⟨MemoryCarryPoseidonRows.representativeRound_schedule head,
        inductionHypothesis⟩

def pureDigest (values : List Nat) (lane : Nat) : Nat :=
  runValueRounds representativeRounds values (fun _ => 0) lane

structure Layout where
  frame : StateOutputFrameRows.Layout
  trace : Trace

def rows (layout : Layout) : List Row :=
  StateOutputFrameRows.rows layout.frame ++ layout.trace.rows

structure Layout.Valid (layout : Layout) : Prop where
  exactInputColumns :
    layout.trace.inputColumns = StateOutputFrameRows.inputColumns layout.frame
  exactSchedule : valueSchedules layout.trace.rounds = expectedSchedule
  traceValid : layout.trace.Valid (rows layout)

theorem Layout.Valid.round_count_exact
    {layout : Layout} (valid : layout.Valid) :
    layout.trace.rounds.length = 9 := by
  have lengths := congrArg List.length valid.exactSchedule
  simpa [valueSchedules, expectedSchedule_exact.1] using lengths

private theorem frame_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (StateOutputFrameRows.rows layout.frame) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

/-- The exact outer rows hash the fixed stateful-with-Nebula source frame.
The carry-output premise is a wire-value statement discharged by the
composed carry-digest relation. -/
theorem output_columns_eq_pureDigest
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (carryDigest : Fin 4 → Nat)
    (carryOutputs : ∀ lane,
      assignment (layout.frame.carryDigestOutputColumn lane) =
        carryDigest lane) :
    ∀ lane : Fin 4,
      assignment (layout.trace.outputColumns.getD lane.val 0) =
        pureDigest
          (StateOutputFrameRows.sourceFrame layout.frame assignment carryDigest)
          lane.val := by
  have traceSound := trace_values_sound valid.traceValid canonical one holds
  have frameValues := StateOutputFrameRows.input_column_values canonical one
    (frame_rows_hold holds) carryDigest carryOutputs
  have inputValues :
      layout.trace.inputColumns.map assignment =
        StateOutputFrameRows.sourceFrame layout.frame assignment carryDigest := by
    rw [valid.exactInputColumns]
    exact frameValues
  have schedules :
      valueSchedules layout.trace.rounds =
        valueSchedules representativeRounds :=
    valid.exactSchedule.trans representativeRounds_schedule.symm
  have runEqual := runValueRounds_eq_of_schedules schedules
    (StateOutputFrameRows.sourceFrame layout.frame assignment carryDigest)
    (fun _ => 0)
  intro lane
  calc
    assignment (layout.trace.outputColumns.getD lane.val 0) =
        runValueRounds layout.trace.rounds
          (layout.trace.inputColumns.map assignment) (fun _ => 0) lane.val :=
      traceSound lane.val lane.isLt
    _ = runValueRounds layout.trace.rounds
          (StateOutputFrameRows.sourceFrame layout.frame assignment carryDigest)
          (fun _ => 0) lane.val := by rw [inputValues]
    _ = runValueRounds representativeRounds
          (StateOutputFrameRows.sourceFrame layout.frame assignment carryDigest)
          (fun _ => 0) lane.val := congrFun runEqual lane.val
    _ = pureDigest
          (StateOutputFrameRows.sourceFrame layout.frame assignment carryDigest)
          lane.val := rfl

structure Honest (layout : Layout) (assignment : Nat → Nat) : Prop where
  frame : StateOutputFrameRows.Honest layout.frame assignment
  trace : layout.trace.ExecutionWitness assignment

theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest layout assignment) :
    Satisfies (rows layout) assignment := by
  intro row member
  rw [rows, List.mem_append] at member
  rcases member with frameMember | traceMember
  · exact StateOutputFrameRows.rows_complete canonical one honest.frame
      row frameMember
  · exact Trace.execution_complete canonical one honest.trace row traceMember

end Nightstream.Implementation.NebulaV2.StateOutputPoseidonRows
