import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-!
Contract: nonoptional physical emission receipts for compact Poseidon2 sponge
traces.

Owns:
- the source-computed rate-four sponge row/column cost;
- the exact ordered row indices named by a compact trace;
- the exact ordered fresh columns allocated by its zero row, wrapper rows,
  and isolated permutation calls;
- a receipt requiring the named program slice and both physical intervals to
  equal the trace emission exactly.

Does not own: any generated trace, semantic preimage encoding, call-site
placement, typed-lowering call recipe, whole-program ownership partition,
native Poseidon2 parity, or collision resistance.

Emits constraints: no. A receipt describes rows already emitted by the trace.
-/

namespace Nightstream.Implementation.R1CS.Poseidon2Sponge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Call

/-- Production Poseidon2 sponge rate. -/
def receiptRate : Nat := 4

/-- Number of absorb permutations required for a field vector. -/
def absorbPermutationCount (inputFields : Nat) : Nat :=
  (inputFields + receiptRate - 1) / receiptRate

/-- Absorb permutations followed by the mandatory padding permutation. -/
def permutationCount (inputFields : Nat) : Nat :=
  absorbPermutationCount inputFields + 1

/-- One zero row, one wrapper row per input field, one padding row, and the
exact isolated permutation program for every sponge round. -/
def emissionCost (inputFields : Nat) : Nat :=
  inputFields + 2 +
    Nightstream.Implementation.R1CS.Poseidon2Permutation.rowCount *
      permutationCount inputFields

theorem permutationRowCount_eq :
    Nightstream.Implementation.R1CS.Poseidon2Permutation.rowCount = 600 := by
  rfl

/-- Wrapper outputs allocated before one permutation call. Unchanged state
lanes are reused and therefore do not occur in this list. -/
def Round.wrapperOutputColumns (round : Round) : List Nat :=
  match round.kind with
  | .absorb chunkColumns =>
      round.permutationInputColumns.take chunkColumns.length
  | .pad =>
      round.permutationInputColumns.take 1

/-- Fresh columns allocated by the exact isolated permutation program. -/
def Round.permutationAllocatedColumns (round : Round) : List Nat :=
  List.range'
    round.call.firstAllocatedColumn
    Nightstream.Implementation.R1CS.Poseidon2Permutation.rowCount

/-- All fresh columns owned by one sponge round, in emission order. -/
def Round.allocatedColumns (round : Round) : List Nat :=
  round.wrapperOutputColumns ++ round.permutationAllocatedColumns

/-- Physical row indices named by one sponge round, in emission order. -/
def Round.rowIndices (round : Round) : List Nat :=
  round.definingRows ++
    List.range' round.call.rowStart
      (round.call.rowEnd - round.call.rowStart)

/-- All fresh columns owned by a sponge trace. The zero-state column is the
first allocation. -/
def Trace.allocatedColumns (trace : Trace) : List Nat :=
  trace.zeroColumn :: trace.rounds.flatMap Round.allocatedColumns

/-- All physical row indices owned by a sponge trace. The zero-state row is
the first emitted row. -/
def Trace.rowIndices (trace : Trace) : List Nat :=
  trace.zeroRow :: trace.rounds.flatMap Round.rowIndices

/-- Exact, nonoptional emission receipt for one sponge trace.

`rowsExact` checks actual row values in the named program slice. The two
interval equalities retain order and multiplicity, so they simultaneously
exclude gaps, duplicates, and hidden trace-local emissions. -/
structure EmissionReceipt
    (trace : Trace)
    (programRows : List Row)
    (inputFields rowStart firstAllocatedColumn : Nat) : Prop where
  inputFieldsExact :
    trace.inputColumns.length = inputFields
  programRangeAvailable :
    rowStart + emissionCost inputFields ≤ programRows.length
  rowsExact :
    (programRows.drop rowStart).take (emissionCost inputFields) =
      trace.rows
  rowIndicesExact :
    trace.rowIndices =
      List.range' rowStart (emissionCost inputFields)
  allocatedColumnsExact :
    trace.allocatedColumns =
      List.range' firstAllocatedColumn (emissionCost inputFields)

namespace EmissionReceipt

theorem traceRows_length
    {trace : Trace}
    {programRows : List Row}
    {inputFields rowStart firstAllocatedColumn : Nat}
    (receipt :
      EmissionReceipt trace programRows inputFields rowStart
        firstAllocatedColumn) :
    trace.rows.length = emissionCost inputFields := by
  rw [← receipt.rowsExact, List.length_take]
  have enough :
      emissionCost inputFields ≤ (programRows.drop rowStart).length := by
    simp only [List.length_drop]
    have available := receipt.programRangeAvailable
    omega
  exact Nat.min_eq_left enough

theorem rowIndices_nodup
    {trace : Trace}
    {programRows : List Row}
    {inputFields rowStart firstAllocatedColumn : Nat}
    (receipt :
      EmissionReceipt trace programRows inputFields rowStart
        firstAllocatedColumn) :
    trace.rowIndices.Nodup := by
  rw [receipt.rowIndicesExact]
  exact List.nodup_range'

theorem allocatedColumns_nodup
    {trace : Trace}
    {programRows : List Row}
    {inputFields rowStart firstAllocatedColumn : Nat}
    (receipt :
      EmissionReceipt trace programRows inputFields rowStart
        firstAllocatedColumn) :
    trace.allocatedColumns.Nodup := by
  rw [receipt.allocatedColumnsExact]
  exact List.nodup_range'

theorem row_column_conservation
    {trace : Trace}
    {programRows : List Row}
    {inputFields rowStart firstAllocatedColumn : Nat}
    (receipt :
      EmissionReceipt trace programRows inputFields rowStart
        firstAllocatedColumn) :
    trace.rowIndices.length = trace.allocatedColumns.length := by
  rw [receipt.rowIndicesExact, receipt.allocatedColumnsExact]
  simp

end EmissionReceipt

end Nightstream.Implementation.R1CS.Poseidon2Sponge
