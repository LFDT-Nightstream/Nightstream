import Nightstream.Implementation.NebulaV2.MemoryClaimHashFrameRows
import Nightstream.Implementation.NebulaV2.StateOutputRowCensus

/-!
Contract: exact Poseidon2 sponge relation for the complete 91-field V2
memory-claim frame.

Assurance tier: implementation model and cryptographic primitive semantics.

Owns the exact 22 full absorbs, one three-field absorb, one terminal padding
round, structural trace linkage, reduction from satisfying generated rows to
the fixed memory-claim digest, honest local completeness, and the exact row
census.

Does not own Poseidon2 collision resistance, absolute generated columns,
the enclosing paper NIFS relation, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryClaimPoseidonRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.NebulaV2.MemoryClaimPoseidonBinding

structure Layout where
  frame : MemoryClaimHashFrameRows.Layout
  trace : Trace

def rows (layout : Layout) : List Row :=
  MemoryClaimHashFrameRows.rows layout.frame ++ layout.trace.rows

/-- All validity fields are structural row, schedule, and column facts. -/
structure Layout.Valid (layout : Layout) : Prop where
  exactInputColumns :
    layout.trace.inputColumns = MemoryClaimHashFrameRows.inputColumns layout.frame
  exactSchedule : valueSchedules layout.trace.rounds = expectedSchedule
  traceValid : layout.trace.Valid (rows layout)

theorem Layout.Valid.round_count_exact
    {layout : Layout} (valid : layout.Valid) :
    layout.trace.rounds.length = 24 := by
  have lengths := congrArg List.length valid.exactSchedule
  simpa [valueSchedules, expectedSchedule_exact.1] using lengths

private theorem frame_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryClaimHashFrameRows.rows layout.frame) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem exact_input_values
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {claim : MemoryClaimCodec.Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.frame.claim assignment claim)
    (holds : Satisfies (rows layout) assignment) :
    layout.trace.inputColumns.map assignment = frame claim := by
  rw [valid.exactInputColumns]
  exact MemoryClaimHashFrameRows.input_column_values canonical one parsed
    (frame_rows_hold holds)

/-- Satisfying the exact generated trace computes the fixed V2 digest from
the same typed memory suffix selected by the full-claim parser. -/
theorem output_columns_eq_digest
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {claim : MemoryClaimCodec.Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      layout.frame.claim assignment claim)
    (holds : Satisfies (rows layout) assignment) :
    ∀ lane : Fin 4,
      assignment (layout.trace.outputColumns.getD lane.val 0) =
        digest claim lane := by
  have traceSound := trace_values_sound valid.traceValid canonical one holds
  have inputValues := exact_input_values valid canonical one parsed holds
  have schedules :
      valueSchedules layout.trace.rounds =
        valueSchedules representativeRounds :=
    valid.exactSchedule.trans representativeRounds_schedule.symm
  have runEqual := runValueRounds_eq_of_schedules schedules
    (frame claim) (fun _ => 0)
  intro lane
  calc
    assignment (layout.trace.outputColumns.getD lane.val 0) =
        runValueRounds layout.trace.rounds
          (layout.trace.inputColumns.map assignment) (fun _ => 0) lane.val :=
      traceSound lane.val lane.isLt
    _ = runValueRounds layout.trace.rounds
          (frame claim) (fun _ => 0) lane.val := by rw [inputValues]
    _ = runValueRounds representativeRounds
          (frame claim) (fun _ => 0) lane.val := congrFun runEqual lane.val
    _ = digest claim lane := rfl

theorem trace_rows_length_exact
    {layout : Layout} (valid : layout.Valid) :
    layout.trace.rows.length = 14493 := by
  rw [StateOutputRowCensus.trace_rows_length]
  have costs := congrArg
    (fun schedules =>
      (schedules.map StateOutputRowCensus.scheduleRowCount).sum)
    valid.exactSchedule
  simp only [valueSchedules, Function.comp_def, List.map_map] at costs
  change 1 +
    (layout.trace.rounds.map fun round =>
      StateOutputRowCensus.scheduleRowCount round.valueSchedule).sum = 14493
  rw [costs]
  norm_num [expectedSchedule, StateOutputRowCensus.scheduleRowCount,
    Poseidon2Permutation.rowCount]

theorem rows_length_exact
    {layout : Layout} (valid : layout.Valid) :
    (rows layout).length = 14501 := by
  rw [rows, List.length_append,
    MemoryClaimHashFrameRows.rows_length_exact,
    trace_rows_length_exact valid]

structure Honest (layout : Layout) (assignment : Nat → Nat) : Prop where
  frame : MemoryClaimHashFrameRows.Honest layout.frame assignment
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
  · exact MemoryClaimHashFrameRows.rows_complete one honest.frame
      row frameMember
  · exact Trace.execution_complete canonical one honest.trace row traceMember

end Nightstream.Implementation.NebulaV2.MemoryClaimPoseidonRows
