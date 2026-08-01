import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsManifest

/-!
Contract: exact native-program column counts derived from physical allocation
lists.

Assurance tier: model-level.

Owns: role-specific allocation counts and their equality to the native
program and manifest cost fields.

Does not own: protocol semantics, program construction, Rust, or measured
deployment sizes.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization.NativeCounts

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram

/-- Number of physical allocation occurrences with one ownership role. -/
def ownershipCount
    (ownership : Ownership)
    (columns : List OwnedColumn) : Nat :=
  (columns.filter fun column =>
    decide (column.ownership = ownership)).length

/-- Total physical allocation count represented by a four-way cost. -/
def allocatedColumns (cost : Cost) : Nat :=
  cost.committedColumns + cost.publicColumns + cost.auxiliaryColumns

/-- Select the cost field for one physical ownership role. -/
def roleCost (ownership : Ownership) (cost : Cost) : Nat :=
  match ownership with
  | .committedColumn => cost.committedColumns
  | .publicColumn => cost.publicColumns
  | .auxiliaryColumn => cost.auxiliaryColumns

@[simp] theorem ownershipCount_nil (ownership : Ownership) :
    ownershipCount ownership [] = 0 :=
  rfl

theorem ownershipCount_append
    (ownership : Ownership)
    (left right : List OwnedColumn) :
    ownershipCount ownership (left ++ right) =
      ownershipCount ownership left + ownershipCount ownership right := by
  simp [ownershipCount, List.filter_append]

private theorem columnCost_role
    (ownership : Ownership)
    (columns : List OwnedColumn) :
    roleCost ownership (columnCost columns) =
      ownershipCount ownership columns := by
  induction columns with
  | nil =>
      cases ownership <;> rfl
  | cons column rest inductionHypothesis =>
      cases column with
      | mk id actualOwnership =>
          unfold columnCost
          simp only [List.map_cons, Cost.sum]
          change
            roleCost ownership
                (Cost.oneColumn actualOwnership + columnCost rest) =
              ownershipCount ownership
                ({ id := id, ownership := actualOwnership } :: rest)
          rw [show
            roleCost ownership
                (Cost.oneColumn actualOwnership + columnCost rest) =
              roleCost ownership (Cost.oneColumn actualOwnership) +
                roleCost ownership (columnCost rest) by
              cases ownership <;> rfl]
          rw [inductionHypothesis]
          cases ownership <;> cases actualOwnership <;>
            simp [roleCost, Cost.oneColumn, ownershipCount, Nat.add_comm]

theorem columnCost_committedColumns
    (columns : List OwnedColumn) :
    (columnCost columns).committedColumns =
      ownershipCount .committedColumn columns :=
  columnCost_role .committedColumn columns

theorem columnCost_publicColumns
    (columns : List OwnedColumn) :
    (columnCost columns).publicColumns =
      ownershipCount .publicColumn columns :=
  columnCost_role .publicColumn columns

theorem columnCost_auxiliaryColumns
    (columns : List OwnedColumn) :
    (columnCost columns).auxiliaryColumns =
      ownershipCount .auxiliaryColumn columns :=
  columnCost_role .auxiliaryColumn columns

private theorem rowCost_role_zero
    (ownership : Ownership)
    (rows : List OwnedRow) :
    roleCost ownership (rowCost rows) = 0 := by
  induction rows with
  | nil =>
      cases ownership <;> rfl
  | cons row rest inductionHypothesis =>
      unfold rowCost
      simp only [List.map_cons, Cost.sum]
      change
        roleCost ownership (Cost.oneRow + rowCost rest) = 0
      rw [show
        roleCost ownership (Cost.oneRow + rowCost rest) =
          roleCost ownership Cost.oneRow +
            roleCost ownership (rowCost rest) by
          cases ownership <;> rfl]
      rw [inductionHypothesis]
      cases ownership <;> rfl

theorem physicalCost_committedColumns
    (columns : List OwnedColumn)
    (rows : List OwnedRow) :
    (physicalCost columns rows).committedColumns =
      ownershipCount .committedColumn columns := by
  change
    roleCost .committedColumn (physicalCost columns rows) =
      ownershipCount .committedColumn columns
  unfold physicalCost
  change
    roleCost .committedColumn (columnCost columns) +
        roleCost .committedColumn (rowCost rows) =
      ownershipCount .committedColumn columns
  rw [columnCost_role, rowCost_role_zero, Nat.add_zero]

theorem physicalCost_publicColumns
    (columns : List OwnedColumn)
    (rows : List OwnedRow) :
    (physicalCost columns rows).publicColumns =
      ownershipCount .publicColumn columns := by
  change
    roleCost .publicColumn (physicalCost columns rows) =
      ownershipCount .publicColumn columns
  unfold physicalCost
  change
    roleCost .publicColumn (columnCost columns) +
        roleCost .publicColumn (rowCost rows) =
      ownershipCount .publicColumn columns
  rw [columnCost_role, rowCost_role_zero, Nat.add_zero]

theorem physicalCost_auxiliaryColumns
    (columns : List OwnedColumn)
    (rows : List OwnedRow) :
    (physicalCost columns rows).auxiliaryColumns =
      ownershipCount .auxiliaryColumn columns := by
  change
    roleCost .auxiliaryColumn (physicalCost columns rows) =
      ownershipCount .auxiliaryColumn columns
  unfold physicalCost
  change
    roleCost .auxiliaryColumn (columnCost columns) +
        roleCost .auxiliaryColumn (rowCost rows) =
      ownershipCount .auxiliaryColumn columns
  rw [columnCost_role, rowCost_role_zero, Nat.add_zero]

private theorem costSum_role
    (ownership : Ownership)
    (receipts : List SelectedReceipt) :
    roleCost ownership
        (Cost.sum (receipts.map SelectedReceipt.cost)) =
      ownershipCount ownership
        (receipts.flatMap SelectedReceipt.allocations) := by
  induction receipts with
  | nil =>
      cases ownership <;> rfl
  | cons receipt rest inductionHypothesis =>
      simp only [List.map_cons, Cost.sum, List.flatMap_cons,
        ownershipCount_append]
      rw [show
        roleCost ownership
            (receipt.cost +
              Cost.sum (rest.map SelectedReceipt.cost)) =
          roleCost ownership receipt.cost +
            roleCost ownership
              (Cost.sum (rest.map SelectedReceipt.cost)) by
          cases ownership <;> rfl]
      rw [inductionHypothesis]
      unfold SelectedReceipt.cost InstructionReceipt.cost
        SelectedReceipt.allocations
      cases ownership
      · exact congrArg
          (fun count =>
            count + ownershipCount .committedColumn
              (rest.flatMap SelectedReceipt.allocations))
          (physicalCost_committedColumns
            receipt.receipt.allocations receipt.receipt.rows)
      · exact congrArg
          (fun count =>
            count + ownershipCount .publicColumn
              (rest.flatMap SelectedReceipt.allocations))
          (physicalCost_publicColumns
            receipt.receipt.allocations receipt.receipt.rows)
      · exact congrArg
          (fun count =>
            count + ownershipCount .auxiliaryColumn
              (rest.flatMap SelectedReceipt.allocations))
          (physicalCost_auxiliaryColumns
            receipt.receipt.allocations receipt.receipt.rows)

theorem program_committedColumns
    (program : NativeCcsProgram.Program) :
    program.cost.committedColumns =
      ownershipCount .committedColumn program.allocations :=
  costSum_role .committedColumn program.receipts

theorem program_publicColumns
    (program : NativeCcsProgram.Program) :
    program.cost.publicColumns =
      ownershipCount .publicColumn program.allocations :=
  costSum_role .publicColumn program.receipts

theorem program_auxiliaryColumns
    (program : NativeCcsProgram.Program) :
    program.cost.auxiliaryColumns =
      ownershipCount .auxiliaryColumn program.allocations :=
  costSum_role .auxiliaryColumn program.receipts

theorem ownershipCounts_sum
    (columns : List OwnedColumn) :
    ownershipCount .committedColumn columns +
        ownershipCount .publicColumn columns +
        ownershipCount .auxiliaryColumn columns =
      columns.length := by
  rw [← columnCost_committedColumns, ← columnCost_publicColumns,
    ← columnCost_auxiliaryColumns]
  induction columns with
  | nil =>
      rfl
  | cons column rest inductionHypothesis =>
      cases column with
      | mk id actualOwnership =>
          unfold columnCost
          simp only [List.map_cons, Cost.sum]
          change
            (Cost.oneColumn actualOwnership).committedColumns +
                  (columnCost rest).committedColumns +
                ((Cost.oneColumn actualOwnership).publicColumns +
                  (columnCost rest).publicColumns) +
              ((Cost.oneColumn actualOwnership).auxiliaryColumns +
                (columnCost rest).auxiliaryColumns) =
              rest.length + 1
          cases actualOwnership <;>
            simp only [Cost.oneColumn] <;>
            omega

theorem program_allocations_length
    (program : NativeCcsProgram.Program) :
    program.allocations.length = allocatedColumns program.cost := by
  unfold allocatedColumns
  rw [program_committedColumns, program_publicColumns,
    program_auxiliaryColumns]
  exact (ownershipCounts_sum program.allocations).symm

theorem manifest_committedColumns
    (program : NativeCcsProgram.Program) :
    (NativeCcsManifest.Program.ofProgram program).cost.committedColumns =
      ownershipCount .committedColumn
        (NativeCcsManifest.Program.ofProgram program).columns := by
  rw [NativeCcsManifest.Program.cost_ofProgram,
    NativeCcsManifest.Program.columns_ofProgram]
  exact program_committedColumns program

theorem manifest_publicColumns
    (program : NativeCcsProgram.Program) :
    (NativeCcsManifest.Program.ofProgram program).cost.publicColumns =
      ownershipCount .publicColumn
        (NativeCcsManifest.Program.ofProgram program).columns := by
  rw [NativeCcsManifest.Program.cost_ofProgram,
    NativeCcsManifest.Program.columns_ofProgram]
  exact program_publicColumns program

theorem manifest_auxiliaryColumns
    (program : NativeCcsProgram.Program) :
    (NativeCcsManifest.Program.ofProgram program).cost.auxiliaryColumns =
      ownershipCount .auxiliaryColumn
        (NativeCcsManifest.Program.ofProgram program).columns := by
  rw [NativeCcsManifest.Program.cost_ofProgram,
    NativeCcsManifest.Program.columns_ofProgram]
  exact program_auxiliaryColumns program

theorem manifest_columns_length
    (program : NativeCcsProgram.Program) :
    (NativeCcsManifest.Program.ofProgram program).columns.length =
      allocatedColumns
        (NativeCcsManifest.Program.ofProgram program).cost := by
  rw [NativeCcsManifest.Program.columns_ofProgram,
    NativeCcsManifest.Program.cost_ofProgram]
  exact program_allocations_length program

end Nightstream.Implementation.Lowering.Goldilocks.Optimization.NativeCounts
