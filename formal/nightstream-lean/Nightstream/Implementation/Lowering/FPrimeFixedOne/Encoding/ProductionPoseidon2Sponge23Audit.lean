import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Recipe
import Nightstream.Implementation.Lowering.Goldilocks.NormalForm

/-!
Contract: conservation and finite-class minimum certificate for the selected
fused 23-field production Poseidon2 sponge occurrence.

Assurance tier: artifact-checked.

Owns:
- complete row-support classification through the explicit source map;
- exact receipt allocation count, ownership, uniqueness, and conservation;
- the finite rewrite class that independently retains or eliminates the
  redundant eight-lane output gates after each of seven internal calls;
- minimum selection under the fixed order `(recurring rows, committed
  columns, public columns, auxiliary columns)`.

Does not own: serialization, optional-digest framing, generated placement,
native Poseidon2 parity, collision resistance, or global arithmetization
minimality.

Emits constraints: no. It audits the recipe's already emitted rows.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Audit

set_option maxRecDepth 131072
set_option maxHeartbeats 8000000

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

namespace Recipe

open ProductionPoseidon2Sponge23Recipe

private def sourceRowsBoundCheck : Bool :=
  NumericSponge.trace.rows.all fun row =>
    (row.a ++ row.b ++ row.c).all fun term =>
      decide (term.1 < 4249)

private theorem sourceRowsBoundCheck_true :
    sourceRowsBoundCheck = true := by
  decide

private theorem source_row_column_lt
    (row : Numeric.Row)
    (rowMember : row ∈ NumericSponge.trace.rows)
    (term : Nat × Nat)
    (termMember : term ∈ row.a ++ row.b ++ row.c) :
    term.1 < 4249 := by
  have rowAccepted :=
    (List.all_eq_true.mp sourceRowsBoundCheck_true) row rowMember
  have termAccepted :=
    (List.all_eq_true.mp rowAccepted) term termMember
  exact of_decide_eq_true termAccepted

private theorem inputColumn_mem
    (frame : Frame)
    (index : Nat)
    (indexLt : index < inputWidth) :
    inputColumn frame index ∈ frame.input.ids := by
  have idsLt : index < frame.input.ids.length := by
    rw [Frame.input_ids_length]
    exact indexLt
  unfold inputColumn
  rw [← List.getElem_eq_getD
    (l := frame.input.ids) (i := index) (h := idsLt) frame.one]
  exact List.getElem_mem idsLt

private theorem outputColumn_mem
    (frame : Frame)
    (lane : Nat)
    (laneLt : lane < outputWidth) :
    outputColumn frame lane ∈ frame.output.ids := by
  have idsLt : lane < frame.output.ids.length := by
    rw [Frame.output_ids_length]
    exact laneLt
  unfold outputColumn
  rw [← List.getElem_eq_getD
    (l := frame.output.ids) (i := lane) (h := idsLt) frame.one]
  exact List.getElem_mem idsLt

private theorem temporaryColumn_mem
    (frame : Frame)
    (index : Nat)
    (indexLt : index < temporaryWidth) :
    temporaryColumn frame index ∈ frame.temporaries.ids := by
  have idsLt : index < frame.temporaries.ids.length := by
    rw [Frame.temporary_ids_length]
    exact indexLt
  unfold temporaryColumn
  rw [← List.getElem_eq_getD
    (l := frame.temporaries.ids) (i := index) (h := idsLt) frame.one]
  exact List.getElem_mem idsLt

private theorem mapped_source_supported
    (frame : Frame)
    (source : Nat)
    (sourceLt : source < 4249) :
    columnMap frame source = frame.one ∨
      columnMap frame source ∈ frame.input.ids ∨
      columnMap frame source ∈ frame.temporaries.ids := by
  by_cases sourceZero : source = 0
  · subst source
    exact Or.inl (columnMap_zero frame)
  · by_cases sourceKnown : source < 24
    · right
      left
      have indexLt : source - 1 < inputWidth := by
        unfold inputWidth
        omega
      have sourceExact : source - 1 + 1 = source := by
        omega
      rw [← sourceExact,
        columnMap_input frame (source - 1) indexLt]
      exact inputColumn_mem frame (source - 1) indexLt
    · right
      right
      have sourceGe : 24 ≤ source := Nat.le_of_not_gt sourceKnown
      have indexLt : source - 24 < temporaryWidth := by
        unfold temporaryWidth
        omega
      unfold columnMap
      rw [if_neg sourceZero, if_neg (by omega)]
      exact temporaryColumn_mem frame (source - 24) indexLt

private theorem core_rows_supported
    (frame : Frame)
    (owned : OwnedRow)
    (member : owned ∈ coreRows frame)
    (column : ColumnId)
    (columnMember : column ∈ owned.columnIds) :
    column = frame.one ∨
      column ∈ frame.input.ids ∨
      column ∈ frame.temporaries.ids := by
  have typedRowMember :
      owned.row ∈ (coreRows frame).map (fun item => item.row) :=
    List.mem_map.mpr ⟨owned, member, rfl⟩
  rw [coreRows,
    ownedRowsFrom_rows frame.owner frame.firstOrdinal
      (columnMap frame) NumericSponge.trace.rows] at typedRowMember
  rcases List.mem_map.mp typedRowMember with
    ⟨sourceRow, sourceRowMember, rowExact⟩
  change column ∈ owned.row.columnIds at columnMember
  rw [← rowExact,
    row_columnIds (columnMap frame) sourceRow] at columnMember
  rcases List.mem_map.mp columnMember with
    ⟨sourceTerm, sourceTermMember, columnExact⟩
  have sourceLt :=
    source_row_column_lt sourceRow sourceRowMember
      sourceTerm sourceTermMember
  have supported :=
    mapped_source_supported frame sourceTerm.1 sourceLt
  simpa [columnExact] using supported

/-- Every dependency of every physical row is classified as constant one,
activation, ordered input, visible output, or receipt-owned temporary. -/
theorem rows_supported
    (frame : Frame)
    (owned : OwnedRow)
    (member : owned ∈ rows frame)
    (column : ColumnId)
    (columnMember : column ∈ owned.columnIds) :
    column = frame.one ∨
      column = frame.active ∨
      column ∈ frame.input.ids ∨
      column ∈ frame.output.ids ∨
      column ∈ frame.temporaries.ids := by
  rcases List.mem_append.mp member with coreMember | gateMember
  · rcases core_rows_supported frame owned coreMember column columnMember with
      one | input | temporary
    · exact Or.inl one
    · exact Or.inr (Or.inr (Or.inl input))
    · exact Or.inr (Or.inr (Or.inr (Or.inr temporary)))
  · rcases List.mem_map.mp gateMember with
      ⟨lane, laneMember, equal⟩
    subst owned
    have laneLt : lane < outputWidth := by
      have gateLt := List.mem_range.mp laneMember
      simpa [gateRowCount, outputWidth] using gateLt
    have support :
        column = frame.active ∨
          column = internalOutputColumn frame lane ∨
          column = outputColumn frame lane := by
      simpa [OwnedRow.columnIds, Row.columnIds, gateRow,
        Goldilocks.singleton, Goldilocks.difference] using columnMember
    rcases support with active | internal | output
    · exact Or.inr (Or.inl active)
    · right
      right
      right
      right
      rw [internal]
      unfold internalOutputColumn
      apply temporaryColumn_mem
      unfold temporaryWidth outputWidth at *
      omega
    · right
      right
      right
      left
      rw [output]
      exact outputColumn_mem frame lane laneLt

theorem receipt_allocations_exact (frame : Frame) :
    (receipt frame).allocations = frame.allocations := by
  simp [receipt, CallReceipt.allocations, Frame.allocations]

theorem receipt_allocation_count (frame : Frame) :
    (receipt frame).allocations.length = recurringRows := by
  rw [receipt_allocations_exact]
  simp only [Frame.allocations, List.length_append]
  rw [frame.output.length_eq, frame.temporaries.length_eq]
  rfl

theorem receipt_allocation_ids_nodup (frame : Frame) :
    ((receipt frame).allocations.map fun column => column.id).Nodup := by
  simpa [receipt_allocations_exact, Frame.allocations,
    ColumnBundle.ids, List.map_append] using frame.allocationsNodup

theorem receipt_allocations_owned
    (frame : Frame)
    (column : OwnedColumn)
    (member : column ∈ (receipt frame).allocations) :
    column.id.owner = frame.owner := by
  apply frame.allocationsOwned
  simpa [receipt_allocations_exact, Frame.allocations] using member

/-- The typed receipt has one classified allocation for every emitted row. -/
theorem receipt_row_column_conservation (frame : Frame) :
    (receipt frame).rows.length =
      (receipt frame).allocations.length := by
  change (rows frame).length =
    (receipt frame).allocations.length
  rw [rows_length, receipt_allocation_count]

/-- The normalized fused source trace also has exact row/column conservation. -/
theorem normalized_row_column_conservation :
    NumericSponge.trace.rowIndices.length =
      NumericSponge.trace.allocatedColumns.length :=
  Nightstream.Implementation.R1CS.Poseidon2Sponge.EmissionReceipt.row_column_conservation
    NumericSponge.emissionReceipt

end Recipe

/-! ## Finite internal-gate rewrite class -/

namespace RewriteClass

open ProductionPoseidon2Sponge23Recipe

/-- A seven-bit mask, hence exactly 128 candidates. Bit `i` retains the
redundant eight output-copy gates after internal permutation call `i`. -/
abbrev Candidate := Fin 128

def retainsGateBlock (candidate : Candidate) (call : Fin 7) : Bool :=
  candidate.val.testBit call.val

def internalGateBlocks (candidate : Candidate) : Nat :=
  (List.finRange 7).countP (retainsGateBlock candidate)

/-- All candidates retain the same fused core, four terminal activation
gates, and 4,229 auxiliary allocations. Only redundant internal gates vary. -/
def cost (candidate : Candidate) : Cost :=
  ⟨recurringRows + 8 * internalGateBlocks candidate,
    0, 0, outputWidth + temporaryWidth⟩

def selected : Candidate :=
  ⟨0, by decide⟩

def isolatedCalls : Candidate :=
  ⟨127, by decide⟩

theorem selected_cost :
    cost selected = ⟨4229, 0, 0, 4229⟩ := by
  decide

theorem isolatedCalls_cost :
    cost isolatedCalls = ⟨4285, 0, 0, 4229⟩ := by
  decide

/-- The fused choice is minimum over all 128 internal-gate retention choices
under the project's fixed four-way lexicographic optimization order. -/
theorem selected_minimum (candidate : Candidate) :
    Cost.LexLe (cost selected) (cost candidate) := by
  rw [selected_cost]
  change
    4229 < 4229 + 8 * internalGateBlocks candidate ∨
      (4229 = 4229 + 8 * internalGateBlocks candidate ∧
        (0 < 0 ∨
          (0 = 0 ∧
            (0 < 0 ∨
              (0 = 0 ∧ 4229 ≤ 4229)))))
  omega

end RewriteClass

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Audit
