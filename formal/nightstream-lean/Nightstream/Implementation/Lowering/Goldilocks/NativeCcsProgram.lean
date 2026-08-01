import Nightstream.Implementation.Lowering.Goldilocks.Compiler
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector

/-!
Contract: receipt-conserving native selected-CCS programs.

Assurance tier: model-level.

Owns:
- one selector column for every physical instruction receipt;
- exact flattening to native CCS rows;
- exact reuse of the source receipt allocation and row streams;
- receipt-folded cost with no selector witness allocation.

Does not own: selector choice, protocol semantics, Rust emission, or a
proof-free manifest.

Emits constraints: exactly one CCS row for each source receipt row and no
additional columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector

private theorem columnCost_recurringRows
    (columns : List OwnedColumn) :
    (columnCost columns).recurringRows = 0 := by
  induction columns with
  | nil =>
      rfl
  | cons column rest inductionHypothesis =>
      unfold columnCost at inductionHypothesis ⊢
      simp only [List.map_cons, Cost.sum, Cost.add_recurringRows]
      rw [inductionHypothesis]
      cases column.ownership <;> rfl

private theorem rowCost_recurringRows
    (rows : List OwnedRow) :
    (rowCost rows).recurringRows = rows.length := by
  induction rows with
  | nil =>
      rfl
  | cons row rest inductionHypothesis =>
      unfold rowCost at inductionHypothesis ⊢
      simp only [List.map_cons, Cost.sum, Cost.add_recurringRows,
        List.length_cons]
      rw [inductionHypothesis]
      simp [Cost.oneRow, Nat.add_comm]

private def columnTotal (cost : Cost) : Nat :=
  cost.committedColumns + cost.publicColumns + cost.auxiliaryColumns

private theorem columnTotal_add (left right : Cost) :
    columnTotal (left + right) =
      columnTotal left + columnTotal right := by
  unfold columnTotal
  simp only [Cost.add_committedColumns, Cost.add_publicColumns,
    Cost.add_auxiliaryColumns]
  omega

private theorem columnCost_columnTotal
    (columns : List OwnedColumn) :
    columnTotal (columnCost columns) = columns.length := by
  induction columns with
  | nil =>
      rfl
  | cons column rest inductionHypothesis =>
      unfold columnCost
      simp only [List.map_cons, Cost.sum]
      rw [columnTotal_add]
      change
        columnTotal (Cost.oneColumn column.ownership) +
            columnTotal (columnCost rest) =
          (column :: rest).length
      rw [inductionHypothesis]
      cases column.ownership <;>
        simp [columnTotal, Cost.oneColumn, Nat.add_comm]

private theorem rowCost_columnTotal
    (rows : List OwnedRow) :
    columnTotal (rowCost rows) = 0 := by
  induction rows with
  | nil =>
      rfl
  | cons _ rest inductionHypothesis =>
      unfold rowCost
      simp only [List.map_cons, Cost.sum]
      rw [columnTotal_add]
      change
        columnTotal Cost.oneRow + columnTotal (rowCost rest) = 0
      rw [inductionHypothesis]
      rfl

private theorem physicalCost_columnTotal
    (columns : List OwnedColumn)
    (rows : List OwnedRow) :
    columnTotal (physicalCost columns rows) = columns.length := by
  unfold physicalCost
  rw [columnTotal_add, columnCost_columnTotal, rowCost_columnTotal]
  omega

/-- One physical instruction receipt and the existing column selected into the
fourth CCS matrix for all rows in that receipt. -/
structure SelectedReceipt where
  receipt : InstructionReceipt
  selector : ColumnId

namespace SelectedReceipt

def rows (receipt : SelectedReceipt) : List SelectedRow :=
  select receipt.selector receipt.receipt.rows

def allocations (receipt : SelectedReceipt) : List OwnedColumn :=
  receipt.receipt.allocations

def cost (receipt : SelectedReceipt) : Cost :=
  receipt.receipt.cost

theorem rows_length (receipt : SelectedReceipt) :
    receipt.rows.length = receipt.receipt.rows.length :=
  select_length _ _

theorem row_ids (receipt : SelectedReceipt) :
    receipt.rows.map (fun row => row.source.id) =
      receipt.receipt.rows.map (fun row => row.id) :=
  select_row_ids _ _

theorem row_ids_nodup (receipt : SelectedReceipt) :
    (receipt.receipt.rows.map fun row => row.id).Nodup →
      (receipt.rows.map fun row => row.source.id).Nodup :=
  select_row_ids_nodup _ _

theorem rows_owned
    (receipt : SelectedReceipt)
    (row : SelectedRow)
    (member : row ∈ receipt.rows) :
    row.source.id.owner = receipt.receipt.owner :=
  select_rows_owned receipt.receipt.owner receipt.selector
    receipt.receipt.rows receipt.receipt.rowsOwned row member

@[simp] theorem cost_recurringRows (receipt : SelectedReceipt) :
    receipt.cost.recurringRows = receipt.rows.length := by
  rw [rows_length]
  simp [cost, InstructionReceipt.cost, physicalCost,
    columnCost_recurringRows, rowCost_recurringRows]

end SelectedReceipt

/-- A native selected-CCS image.  Receipt order remains the exact physical
execution and allocation order. -/
structure Program where
  one : ColumnId
  receipts : List SelectedReceipt

namespace Program

def rows (program : Program) : List SelectedRow :=
  program.receipts.flatMap SelectedReceipt.rows

def allocations (program : Program) : List OwnedColumn :=
  program.receipts.flatMap SelectedReceipt.allocations

def columnIds (program : Program) : List ColumnId :=
  program.allocations.map fun column => column.id

def rowIds (program : Program) : List RowId :=
  program.rows.map fun row => row.source.id

def cost (program : Program) : Cost :=
  Cost.sum (program.receipts.map SelectedReceipt.cost)

def Satisfies
    (program : Program)
    (assignment : ColumnId → Nightstream.SuperNeo.Concrete.F) : Prop :=
  assignment program.one = 1 ∧
    NativeCcsSelector.Satisfies program.rows assignment

theorem satisfies_flattened_receipts_iff
    (receipts : List SelectedReceipt)
    (assignment : ColumnId → Nightstream.SuperNeo.Concrete.F) :
    NativeCcsSelector.Satisfies
        (receipts.flatMap SelectedReceipt.rows) assignment ↔
      ∀ receipt, receipt ∈ receipts →
        NativeCcsSelector.Satisfies receipt.rows assignment := by
  induction receipts with
  | nil =>
      simp [NativeCcsSelector.Satisfies]
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons,
        NativeCcsSelector.satisfies_append_iff,
        inductionHypothesis]
      constructor
      · rintro ⟨headHolds, tailHolds⟩ receipt member
        rcases List.mem_cons.1 member with rfl | tailMember
        · exact headHolds
        · exact tailHolds receipt tailMember
      · intro all
        exact ⟨all head List.mem_cons_self,
          fun receipt member =>
            all receipt (List.mem_cons_of_mem head member)⟩

theorem receipt_satisfies
    (program : Program)
    (assignment : ColumnId → Nightstream.SuperNeo.Concrete.F)
    (satisfied : program.Satisfies assignment)
    (receipt : SelectedReceipt)
    (member : receipt ∈ program.receipts) :
    NativeCcsSelector.Satisfies receipt.rows assignment := by
  exact
    (satisfies_flattened_receipts_iff program.receipts assignment).1
      (by simpa only [rows] using satisfied.2)
      receipt member

theorem source_satisfies_of_selector_one
    (program : Program)
    (assignment : ColumnId → Nightstream.SuperNeo.Concrete.F)
    (satisfied : program.Satisfies assignment)
    (receipt : SelectedReceipt)
    (member : receipt ∈ program.receipts)
    (selectorOne : assignment receipt.selector = 1) :
    Goldilocks.Satisfies receipt.receipt.rows assignment :=
  NativeCcsSelector.active_sound receipt.selector receipt.receipt.rows
    assignment selectorOne
    (program.receipt_satisfies assignment satisfied receipt member)

theorem rows_conserved (program : Program) :
    program.rows =
      program.receipts.flatMap SelectedReceipt.rows :=
  rfl

theorem allocations_conserved (program : Program) :
    program.allocations =
      program.receipts.flatMap SelectedReceipt.allocations :=
  rfl

/-- The physical column identifier stream is exactly the ordered flattening
of the source receipt allocations. -/
theorem columnIds_conserved (program : Program) :
    program.columnIds =
      program.receipts.flatMap
        (fun receipt => receipt.receipt.columnIds) := by
  simp only [columnIds, allocations, List.map_flatMap,
    SelectedReceipt.allocations, InstructionReceipt.columnIds]

private theorem rows_length_sum
    (receipts : List SelectedReceipt) :
    (receipts.flatMap SelectedReceipt.rows).length =
      (Cost.sum (receipts.map SelectedReceipt.cost)).recurringRows := by
  induction receipts with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, List.length_append, List.map_cons,
        Cost.sum, Cost.add_recurringRows]
      rw [← head.cost_recurringRows, inductionHypothesis]

theorem rows_length (program : Program) :
    program.rows.length = program.cost.recurringRows := by
  exact rows_length_sum program.receipts

private theorem allocations_length_sum
    (receipts : List SelectedReceipt) :
    (receipts.flatMap SelectedReceipt.allocations).length =
      columnTotal (Cost.sum (receipts.map SelectedReceipt.cost)) := by
  induction receipts with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, List.length_append, List.map_cons,
        Cost.sum]
      rw [columnTotal_add]
      change
        head.receipt.allocations.length +
            (List.flatMap SelectedReceipt.allocations tail).length =
          columnTotal head.receipt.cost +
            columnTotal (Cost.sum (tail.map SelectedReceipt.cost))
      rw [inductionHypothesis]
      unfold SelectedReceipt.cost InstructionReceipt.cost
      rw [physicalCost_columnTotal]

/-- Every physical column occurrence is counted once in exactly one ownership
component of the receipt-derived cost. -/
theorem columnIds_length_eq_cost_columns (program : Program) :
    program.columnIds.length =
      program.cost.committedColumns +
        program.cost.publicColumns +
        program.cost.auxiliaryColumns := by
  rw [show program.columnIds.length = program.allocations.length by
    simp [columnIds]]
  change
    (program.receipts.flatMap SelectedReceipt.allocations).length =
      columnTotal (Cost.sum (program.receipts.map SelectedReceipt.cost))
  exact allocations_length_sum program.receipts

private theorem flatMap_ids_nodup
    {Receipt Owner Id : Type}
    (ownerOf : Receipt → Owner)
    (idOwner : Id → Owner)
    (ids : Receipt → List Id)
    (receipts : List Receipt)
    (ownersNodup : (receipts.map ownerOf).Nodup)
    (localNodup :
      ∀ receipt, receipt ∈ receipts → (ids receipt).Nodup)
    (idsOwned :
      ∀ receipt id, id ∈ ids receipt →
        idOwner id = ownerOf receipt) :
    (receipts.flatMap ids).Nodup := by
  induction receipts with
  | nil =>
      exact List.nodup_nil
  | cons head tail inductionHypothesis =>
      have ownerSplit :
          ownerOf head ∉ tail.map ownerOf ∧
            (tail.map ownerOf).Nodup := by
        simpa only [List.map_cons, List.nodup_cons] using ownersNodup
      rw [List.flatMap_cons, List.nodup_append]
      refine ⟨
        localNodup head List.mem_cons_self,
        inductionHypothesis ownerSplit.2
          (fun receipt member =>
            localNodup receipt (List.mem_cons_of_mem head member)),
        ?_
      ⟩
      intro headId headMember tailId tailMember idsEqual
      rcases List.mem_flatMap.mp tailMember with
        ⟨tailReceipt, tailReceiptMember, tailIdMember⟩
      have ownersEqual : ownerOf head = ownerOf tailReceipt := by
        calc
          ownerOf head = idOwner headId :=
            (idsOwned head headId headMember).symm
          _ = idOwner tailId := congrArg idOwner idsEqual
          _ = ownerOf tailReceipt :=
            idsOwned tailReceipt tailId tailIdMember
      exact False.elim (ownerSplit.1
        (List.mem_map.mpr
          ⟨tailReceipt, tailReceiptMember, ownersEqual.symm⟩))

theorem columnIds_nodup
    (program : Program)
    (ownersNodup :
      (program.receipts.map fun receipt => receipt.receipt.owner).Nodup)
    (localNodup :
      ∀ receipt, receipt ∈ program.receipts →
        receipt.receipt.columnIds.Nodup) :
    program.columnIds.Nodup := by
  simp only [columnIds, allocations, List.map_flatMap,
    SelectedReceipt.allocations, InstructionReceipt.columnIds]
  exact flatMap_ids_nodup
    (fun receipt : SelectedReceipt => receipt.receipt.owner)
    (fun id : ColumnId => id.owner)
    (fun receipt : SelectedReceipt => receipt.receipt.columnIds)
    program.receipts ownersNodup localNodup
    (by
      intro receipt id member
      rcases List.mem_map.1 member with ⟨column, columnMember, rfl⟩
      exact receipt.receipt.allocationsOwned column columnMember)

theorem rowIds_nodup
    (program : Program)
    (ownersNodup :
      (program.receipts.map fun receipt => receipt.receipt.owner).Nodup)
    (localNodup :
      ∀ receipt, receipt ∈ program.receipts →
        receipt.receipt.rowIds.Nodup) :
    program.rowIds.Nodup := by
  simp only [rowIds, rows, List.map_flatMap, SelectedReceipt.rows,
    select_row_ids, InstructionReceipt.rowIds]
  exact flatMap_ids_nodup
    (fun receipt : SelectedReceipt => receipt.receipt.owner)
    (fun id : RowId => id.owner)
    (fun receipt : SelectedReceipt => receipt.receipt.rowIds)
    program.receipts ownersNodup localNodup
    (by
      intro receipt id member
      rcases List.mem_map.1 member with ⟨row, rowMember, rfl⟩
      exact receipt.receipt.rowsOwned row rowMember)

end Program

end Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
