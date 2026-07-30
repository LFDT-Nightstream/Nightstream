import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.Ownership

/-!
Contract: connect the exact physical `Cost` fields to the lengths of an
emitted encoding's row and column lists.

Assurance tier: model-level.

Owns: the generic count theorem needed by current fixed-point dimensions.

Does not own: one application, closed numeric costs, Rust, or protocol
semantics.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.EncodingCostCount

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed

universe u

private theorem columnCost_total (columns : List OwnedColumn) :
    columns.length =
      (columnCost columns).committedColumns +
        (columnCost columns).publicColumns +
        (columnCost columns).auxiliaryColumns := by
  induction columns with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      change
        tail.length + 1 =
          (Cost.oneColumn head.ownership +
              columnCost tail).committedColumns +
            (Cost.oneColumn head.ownership +
              columnCost tail).publicColumns +
            (Cost.oneColumn head.ownership +
              columnCost tail).auxiliaryColumns
      simp only [Cost.add_committedColumns, Cost.add_publicColumns,
        Cost.add_auxiliaryColumns]
      cases head.ownership <;>
        simp only [Cost.oneColumn] <;>
        omega

private theorem rowCost_has_no_columns (rows : List OwnedRow) :
    (rowCost rows).committedColumns = 0 ∧
      (rowCost rows).publicColumns = 0 ∧
      (rowCost rows).auxiliaryColumns = 0 := by
  induction rows with
  | nil =>
      exact ⟨rfl, rfl, rfl⟩
  | cons _ tail inductionHypothesis =>
      change
        (Cost.oneRow + rowCost tail).committedColumns = 0 ∧
          (Cost.oneRow + rowCost tail).publicColumns = 0 ∧
          (Cost.oneRow + rowCost tail).auxiliaryColumns = 0
      simpa only [Cost.add_committedColumns, Cost.add_publicColumns,
        Cost.add_auxiliaryColumns, Cost.oneRow, Nat.zero_add] using
        inductionHypothesis

/-- The number of emitted owned columns is exactly the sum of the three
column components in the receipt-derived physical cost. -/
theorem columnIds_length_eq_cost_columns
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    encoding.columnIds.length =
      encoding.cost.committedColumns +
        encoding.cost.publicColumns +
        encoding.cost.auxiliaryColumns := by
  have columns := columnCost_total encoding.columns
  have rows := rowCost_has_no_columns encoding.rows
  unfold Encoding.columnIds Encoding.cost physicalCost
  rw [List.length_map]
  simp only [Cost.add_committedColumns, Cost.add_publicColumns,
    Cost.add_auxiliaryColumns]
  omega

/-- The number of emitted selective source rows is exactly the recurring-row
component in the receipt-derived physical cost. -/
theorem rows_length_eq_cost_rows
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    (EncodingRows.program encoding).length =
      encoding.cost.recurringRows :=
  Ownership.compiledRowCount_eq_cost encoding

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.EncodingCostCount
