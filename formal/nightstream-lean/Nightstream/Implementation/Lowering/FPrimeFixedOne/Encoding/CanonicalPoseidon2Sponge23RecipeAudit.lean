import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23RecipeSemantics

/-!
Contract: complete ownership, support, receipt, and cost audit for the typed
fixed-23 canonical Poseidon2 sponge occurrence.

Owns: classification of every physical row dependency, exact allocation
conservation, and exact typed coefficient accounting.

Does not own: hash-call serialization, optional-digest semantics, or Rust
placement.
-/

set_option autoImplicit false
set_option maxRecDepth 32768

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge

private theorem inputColumn_mem
    (frame : Frame) (index : Nat) (indexLt : index < inputWidth) :
    inputColumn frame index ∈ frame.input.ids := by
  have idsLt : index < frame.input.ids.length := by
    rw [Frame.input_ids_length]
    exact indexLt
  unfold inputColumn
  rw [← List.getElem_eq_getD
    (l := frame.input.ids) (i := index) (h := idsLt) frame.one]
  exact List.getElem_mem idsLt

private theorem outputColumn_mem
    (frame : Frame) (lane : Nat) (laneLt : lane < outputWidth) :
    outputColumn frame lane ∈ frame.output.ids := by
  have idsLt : lane < frame.output.ids.length := by
    rw [Frame.output_ids_length]
    exact laneLt
  unfold outputColumn
  rw [← List.getElem_eq_getD
    (l := frame.output.ids) (i := lane) (h := idsLt) frame.one]
  exact List.getElem_mem idsLt

private theorem temporaryColumn_mem
    (frame : Frame) (index : Nat) (indexLt : index < temporaryWidth) :
    temporaryColumn frame index ∈ frame.temporaries.ids := by
  have idsLt : index < frame.temporaries.ids.length := by
    rw [Frame.temporary_ids_length]
    exact indexLt
  unfold temporaryColumn
  rw [← List.getElem_eq_getD
    (l := frame.temporaries.ids) (i := index) (h := idsLt) frame.one]
  exact List.getElem_mem idsLt

private theorem mapped_allocated_supported
    (frame : Frame) (source : Nat)
    (allocated :
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.Allocated
        source) :
    columnMap frame source = frame.one ∨
      columnMap frame source ∈ frame.input.ids ∨
      columnMap frame source ∈ frame.temporaries.ids := by
  rcases allocated with rfl | inputMember | temporaryMember
  · exact Or.inl (columnMap_zero frame)
  · rcases List.mem_ofFn.mp inputMember with ⟨index, rfl⟩
    right
    left
    have indexLt : index.val < inputWidth := by
      have := index.isLt
      change index.val < 23 at this
      exact this
    rw [show
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputColumn
            index.val = 2527 + index.val by
          simp only [
            Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputColumn,
            Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23.inputBase_eq],
      columnMap_input frame index.val indexLt]
    exact inputColumn_mem frame index.val indexLt
  · rcases List.mem_ofFn.mp temporaryMember with ⟨position, rfl⟩
    right
    right
    rw [columnMap_sourceTemporary frame position]
    exact temporaryColumn_mem frame position.val
      (by simpa [temporaryWidth,
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.temporaries]
        using position.isLt)

private theorem core_rows_supported
    (frame : Frame) (owned : OwnedRow) (member : owned ∈ coreRows frame)
    (column : ColumnId) (columnMember : column ∈ owned.columnIds) :
    column = frame.one ∨
      column ∈ frame.input.ids ∨
      column ∈ frame.temporaries.ids := by
  have typedRowMember :
      owned.row ∈ (coreRows frame).map (fun item => item.row) :=
    List.mem_map.mpr ⟨owned, member, rfl⟩
  rw [coreRows,
    ownedRowsFrom_rows frame.owner frame.firstOrdinal
      (columnMap frame) Canonical.rows] at typedRowMember
  rcases List.mem_map.mp typedRowMember with
    ⟨sourceRow, sourceRowMember, rowExact⟩
  change column ∈ owned.row.columnIds at columnMember
  rw [← rowExact, row_columnIds (columnMap frame) sourceRow] at columnMember
  rcases List.mem_map.mp columnMember with
    ⟨sourceTerm, sourceTermMember, columnExact⟩
  have allocated :=
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.program_conservation
      Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
      sourceRow sourceRowMember sourceTerm.1
      (by
        rcases List.mem_append.mp sourceTermMember with inAB | inC
        · rcases List.mem_append.mp inAB with inA | inB
          · exact Or.inl (List.mem_map.mpr ⟨sourceTerm, inA, rfl⟩)
          · exact Or.inr
              (Or.inl (List.mem_map.mpr ⟨sourceTerm, inB, rfl⟩))
        · exact Or.inr
            (Or.inr (List.mem_map.mpr ⟨sourceTerm, inC, rfl⟩)))
  simpa [columnExact] using
    mapped_allocated_supported frame sourceTerm.1 allocated

/-- Every emitted dependency is visible or belongs to the mandatory
temporary receipt. -/
theorem rows_supported
    (frame : Frame) (owned : OwnedRow) (member : owned ∈ rows frame)
    (column : ColumnId) (columnMember : column ∈ owned.columnIds) :
    column ∈ frame.visibleIds ++ frame.temporaries.ids := by
  rcases List.mem_append.mp member with coreMember | gateMember
  · rcases core_rows_supported frame owned coreMember column columnMember with
      one | input | temporary
    · simp [Frame.visibleIds, one]
    · simp [Frame.visibleIds, input]
    · exact List.mem_append_right _ temporary
  · rcases List.mem_map.mp gateMember with ⟨lane, laneMember, rfl⟩
    have laneLt : lane < outputWidth := by
      have := List.mem_range.mp laneMember
      simpa [gateRowCount, outputWidth] using this
    have support :
        column = frame.active ∨
          column = internalOutputColumn frame lane ∨
          column = outputColumn frame lane := by
      simpa [OwnedRow.columnIds, Row.columnIds, gateRow,
        Goldilocks.singleton, Goldilocks.difference] using columnMember
    rcases support with active | internal | output
    · simp [Frame.visibleIds, active]
    · have allocated :=
        Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Ownership.output_allocated
          6 (by decide)
          ⟨lane % Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.width,
            Nat.mod_lt _ (by decide)⟩
      rcases mapped_allocated_supported frame _ allocated with
        one | input | temporary
      · simp [Frame.visibleIds, internal, internalOutputColumn, one]
      · simp [Frame.visibleIds, internal, internalOutputColumn, input]
      · exact List.mem_append_right _ (by
          simpa [internal, internalOutputColumn] using temporary)
    · simp [Frame.visibleIds, output,
        outputColumn_mem frame lane laneLt]

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
    (frame : Frame) (column : OwnedColumn)
    (member : column ∈ (receipt frame).allocations) :
    column.id.owner = frame.owner := by
  apply frame.allocationsOwned
  simpa [receipt_allocations_exact, Frame.allocations] using member

theorem receipt_row_column_conservation (frame : Frame) :
    (receipt frame).rows.length = (receipt frame).allocations.length := by
  change (rows frame).length = (receipt frame).allocations.length
  rw [rows_length, receipt_allocation_count]

/-! ## Definitional resource cost -/

/-- The seven normalized permutation calls before the four visible digest
copies.  Every one of these rows owns exactly one auxiliary result column. -/
def intrinsicCost : Cost :=
  { recurringRows := coreRowCount
    committedColumns := 0
    publicColumns := 0
    auxiliaryColumns := temporaryWidth }

/-- Activation-safe publication of the four digest lanes. -/
def outputCost : Cost :=
  { recurringRows := gateRowCount
    committedColumns := 0
    publicColumns := 0
    auxiliaryColumns := outputWidth }

/-- Exact standalone cost of this nonoptional auxiliary-output core.

This is not the cost of either frozen hash call: the optional presence and
alignment wrapper remains outside this core. -/
def standaloneCost : Cost :=
  intrinsicCost + outputCost

@[simp] theorem intrinsicCost_exact :
    intrinsicCost = (⟨2464, 0, 0, 2464⟩ : Cost) :=
  rfl

@[simp] theorem outputCost_exact :
    outputCost = (⟨4, 0, 0, 4⟩ : Cost) :=
  rfl

@[simp] theorem standaloneCost_exact :
    standaloneCost = (⟨2468, 0, 0, 2468⟩ : Cost) :=
  rfl

theorem standaloneCost_matches_receipt (frame : Frame) :
    standaloneCost.recurringRows = (receipt frame).rows.length /\
      standaloneCost.auxiliaryColumns =
        (receipt frame).allocations.length := by
  constructor
  · change 2468 = (rows frame).length
    exact (rows_length frame).symm
  · change 2468 = (receipt frame).allocations.length
    exact (receipt_allocation_count frame).symm

def typedRowTermCount (row : Row) : Nat :=
  row.a.length + row.b.length + row.c.length

def typedProgramTermCount (source : List OwnedRow) : Nat :=
  (source.map (fun owned => typedRowTermCount owned.row)).sum

private theorem translated_row_termCount
    (frame : Frame) (row : Nightstream.Implementation.R1CS.Row) :
    typedRowTermCount
        (NumericRowBridge.row (columnMap frame) row) =
      Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.rawTermCount
        row := by
  simp [typedRowTermCount,
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized.rawTermCount,
    NumericRowBridge.row, NumericRowBridge.terms]

private theorem typedProgramTermCount_ownedRowsFrom
    (frame : Frame) (ordinal : Nat)
    (source : List Nightstream.Implementation.R1CS.Row) :
    typedProgramTermCount
        (ownedRowsFrom frame.owner ordinal (columnMap frame) source) =
      (source.map fun row =>
        typedRowTermCount
          (NumericRowBridge.row (columnMap frame) row)).sum := by
  induction source generalizing ordinal with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      have tailEqual := inductionHypothesis (ordinal + 1)
      unfold typedProgramTermCount at tailEqual ⊢
      simp only [ownedRowsFrom, List.map_cons, List.sum_cons]
      rw [tailEqual]

theorem core_nonzero_coefficient_count (frame : Frame) :
    typedProgramTermCount (coreRows frame) = 31139 := by
  rw [coreRows, typedProgramTermCount_ownedRowsFrom]
  rw [List.map_congr_left (fun row _ => translated_row_termCount frame row)]
  exact
    Nightstream.Implementation.R1CS.Canonical.Poseidon2ExactCoefficients.program_nonzero_coefficient_count

theorem gate_nonzero_coefficient_count (frame : Frame) :
    typedProgramTermCount (gateRows frame) = 12 := by
  rfl

theorem exact_nonzero_coefficient_count (frame : Frame) :
    typedProgramTermCount (rows frame) = 31151 := by
  unfold typedProgramTermCount rows
  rw [List.map_append, List.sum_append]
  change
    typedProgramTermCount (coreRows frame) +
      typedProgramTermCount (gateRows frame) = 31151
  rw [core_nonzero_coefficient_count, gate_nonzero_coefficient_count]

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe
