import Nightstream.Implementation.NebulaV2.FPrime.State.MemoryCarryOutputRows

/-!
Contract: exact row census for the two-stage V2 memory-carry and recursive
state-output Poseidon2 block.

Assurance tier: implementation model.

Owns the row-cost interpretation of each certified sponge schedule, the exact
carry and outer trace costs, and the exact composed local row count.

Does not own absolute generated-row placement, the other recursive-relation
rows, cryptographic security, or Rust conformance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.StateOutputRowCensus

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.NebulaV2

/-- One absorb round has one definition row for each absorbed value. A pad
round has one definition row. Every round then has the exact 600-row
Poseidon2 permutation artifact. -/
def scheduleRowCount : ValueSchedule → Nat
  | .absorb count => count + Poseidon2Permutation.rowCount
  | .pad => 1 + Poseidon2Permutation.rowCount

private theorem call_rows_length (call : Poseidon2Call.Call) :
    call.rows.length = Poseidon2Permutation.rowCount := by
  simp [Poseidon2Call.Call.rows, Poseidon2Permutation.rows_length]

theorem round_rows_length (round : Round) :
    round.rows.length = scheduleRowCount round.valueSchedule := by
  cases kind : round.kind with
  | absorb chunkColumns =>
      simp [Round.rows, Round.expectedDefinitionRows, Round.valueSchedule,
        scheduleRowCount, kind, call_rows_length]
  | pad =>
      simp [Round.rows, Round.expectedDefinitionRows, Round.valueSchedule,
        scheduleRowCount, kind, call_rows_length, Nat.add_comm]

theorem trace_rows_length (trace : Trace) :
    trace.rows.length =
      1 + (trace.rounds.map (scheduleRowCount ∘ Round.valueSchedule)).sum := by
  simp [Trace.rows, Trace.zeroDefinitionRows, List.length_flatMap,
    round_rows_length, Function.comp_def, Nat.add_comm]

private theorem trace_rows_length_of_schedule
    {trace : Trace} {schedule : List ValueSchedule}
    (exactSchedule : valueSchedules trace.rounds = schedule) :
    trace.rows.length = 1 + (schedule.map scheduleRowCount).sum := by
  rw [trace_rows_length]
  have costs := congrArg (fun entries => (entries.map scheduleRowCount).sum)
    exactSchedule
  simpa [valueSchedules, Function.comp_def, List.map_map] using costs

theorem carry_trace_rows_length
    {layout : MemoryCarryPoseidonRows.Layout}
    (valid : layout.Valid) :
    layout.trace.rows.length = 18719 := by
  rw [trace_rows_length_of_schedule valid.exactSchedule]
  norm_num [MemoryCarryPoseidonRows.expectedSchedule, scheduleRowCount,
    Poseidon2Permutation.rowCount]

theorem carry_rows_length
    {layout : MemoryCarryPoseidonRows.Layout}
    (valid : layout.Valid) :
    (MemoryCarryPoseidonRows.rows layout).length = 18859 := by
  rw [MemoryCarryPoseidonRows.rows, List.length_append,
    MemoryCarryHashFrameRows.rows_length_exact, carry_trace_rows_length valid]

theorem outer_trace_rows_length
    {layout : StateOutputPoseidonRows.Layout}
    (valid : layout.Valid) :
    layout.trace.rows.length = 5434 := by
  rw [trace_rows_length_of_schedule valid.exactSchedule]
  norm_num [StateOutputPoseidonRows.expectedSchedule, scheduleRowCount,
    Poseidon2Permutation.rowCount]

theorem outer_rows_length
    {layout : StateOutputPoseidonRows.Layout}
    (valid : layout.Valid) :
    (StateOutputPoseidonRows.rows layout).length = 5440 := by
  rw [StateOutputPoseidonRows.rows, List.length_append,
    StateOutputFrameRows.rows_length_exact, outer_trace_rows_length valid]

/-- Exact local row cost of the complete two-stage hash path. -/
theorem composed_rows_length
    {layout : MemoryCarryStateOutputRows.Layout}
    (valid : layout.Valid) :
    (MemoryCarryStateOutputRows.rows layout).length = 24299 := by
  rw [MemoryCarryStateOutputRows.rows, List.length_append,
    carry_rows_length valid.carryValid,
    outer_rows_length valid.stateOutputValid]

end Nightstream.Implementation.NebulaV2.StateOutputRowCensus
