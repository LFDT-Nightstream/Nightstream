import Nightstream.Implementation.NebulaV2.Production.Memory.BatchHashFrameRows
import Nightstream.Implementation.NebulaV2.FPrime.State.OutputRowCensus

/-!
Contract: exact Poseidon2 row relation for one field-native memory batch.

The trace consumes the exact candidate frame selected by the checked-memory
decoder. Satisfying rows compute the fixed batch digest. The result is not a
prover-supplied digest premise.

Does not own CCS public placement, state-digest semantics, absolute generated
columns, Poseidon2 collision security, Rust refinement, candidate selection,
or a verifier key.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonBinding
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

structure Layout (candidate : Id) where
  frame : ProductionMemoryBatchHashFrameRows.Layout candidate
  trace : Trace

def rows {candidate : Id} (layout : Layout candidate) : List Row :=
  ProductionMemoryBatchHashFrameRows.rows layout.frame ++ layout.trace.rows

/-- Structural certificate only. It contains no assignment or digest result. -/
structure Layout.Valid {candidate : Id} (layout : Layout candidate) : Prop where
  exactInputColumns :
    layout.trace.inputColumns =
      ProductionMemoryBatchHashFrameRows.inputColumns layout.frame
  exactSchedule : valueSchedules layout.trace.rounds = expectedSchedule candidate
  traceValid : layout.trace.Valid (rows layout)

private theorem frame_rows_hold
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (ProductionMemoryBatchHashFrameRows.rows layout.frame)
      assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

private theorem exact_input_values
    {candidate : Id} {layout : Layout candidate}
    (valid : layout.Valid)
    {assignment : Nat -> Nat} {headers : FPrime.ChainHeaders Digest.Value}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (memory : ProductionMemoryCheckedBatchRows.Result
      layout.frame.memory assignment headers)
    (satisfied : Satisfies (rows layout) assignment) :
    layout.trace.inputColumns.map assignment = frame memory.suffixBatch := by
  rw [valid.exactInputColumns]
  exact ProductionMemoryBatchHashFrameRows.input_column_values
    canonical one memory (frame_rows_hold satisfied)

/-- Satisfying the exact generated trace computes the fixed candidate digest
from the same ordered batch derived by the memory relation. -/
theorem output_columns_eq_digest
    {candidate : Id} {layout : Layout candidate}
    (valid : layout.Valid)
    {assignment : Nat -> Nat} {headers : FPrime.ChainHeaders Digest.Value}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (memory : ProductionMemoryCheckedBatchRows.Result
      layout.frame.memory assignment headers)
    (satisfied : Satisfies (rows layout) assignment) :
    forall lane : Fin 4,
      assignment (layout.trace.outputColumns.getD lane.val 0) =
        digest memory.suffixBatch lane := by
  have traceSound := trace_values_sound valid.traceValid canonical one satisfied
  have inputValues := exact_input_values valid canonical one memory satisfied
  have schedules :
      valueSchedules layout.trace.rounds =
        valueSchedules (representativeRounds candidate) :=
    valid.exactSchedule.trans
      (representativeRounds_schedule candidate).symm
  have runEqual := runValueRounds_eq_of_schedules schedules
    (frame memory.suffixBatch) (fun _ => 0)
  intro lane
  calc
    assignment (layout.trace.outputColumns.getD lane.val 0) =
        runValueRounds layout.trace.rounds
          (layout.trace.inputColumns.map assignment) (fun _ => 0) lane.val :=
      traceSound lane.val lane.isLt
    _ = runValueRounds layout.trace.rounds
          (frame memory.suffixBatch) (fun _ => 0) lane.val := by
      rw [inputValues]
    _ = runValueRounds (representativeRounds candidate)
          (frame memory.suffixBatch) (fun _ => 0) lane.val :=
      congrFun runEqual lane.val
    _ = digest memory.suffixBatch lane := rfl

def traceRowCount (candidate : Id) : Nat :=
  1 + ((expectedSchedule candidate).map
    StateOutputRowCensus.scheduleRowCount).sum

def rowCount (candidate : Id) : Nat := 8 + traceRowCount candidate

theorem trace_rows_length_exact
    {candidate : Id} {layout : Layout candidate} (valid : layout.Valid) :
    layout.trace.rows.length = traceRowCount candidate := by
  rw [StateOutputRowCensus.trace_rows_length]
  have costs := congrArg
    (fun schedules =>
      (schedules.map StateOutputRowCensus.scheduleRowCount).sum)
    valid.exactSchedule
  simp only [valueSchedules, Function.comp_def, List.map_map] at costs
  simpa [traceRowCount, Function.comp_def] using costs

theorem rows_length_exact
    {candidate : Id} {layout : Layout candidate} (valid : layout.Valid) :
    (rows layout).length = rowCount candidate := by
  rw [rows, List.length_append,
    ProductionMemoryBatchHashFrameRows.rows_length_exact,
    trace_rows_length_exact valid]
  rfl

theorem candidate_row_count_table :
    rowCount .e1 = 14501 /\
      rowCount .e4 = 51950 /\
      rowCount .e8 = 102082 /\
      rowCount .e16 = 202346 := by
  decide

end Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonRows
