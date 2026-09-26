import NightstreamFPrime.Layout.MatrixProgram.PlanBridge
import NightstreamFPrime.Layout.ProductionRelation.ColumnMap

/-! Exact interpretation of a matrix program, with ordered composition and
checked coordinate projection. The source-row accessor stays explicit. -/

namespace NightstreamFPrime.Layout.MatrixProgram

open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation

/-- Exact compact interpretation of one semantic plan. -/
structure Exact {logicalWidth : Nat}
    (matrixProgram : MatrixProgram.Program)
    (plan : ProductionRelation.Plan logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) : Prop where
  rowCount : matrixProgram.rowCount = plan.rowCount
  row? : ∀ row : Fin plan.rowCount,
    matrixProgram.row? logicalWidth sourceRow row.val = some (plan.forms row)

theorem Exact.append {logicalWidth : Nat}
    {leftProgram rightProgram : MatrixProgram.Program}
    {leftPlan rightPlan : ProductionRelation.Plan logicalWidth}
    {sourceRow : Nat → Option R1CS.Row}
    (left : Exact leftProgram leftPlan sourceRow)
    (right : Exact rightProgram rightPlan sourceRow)
    (fits : leftPlan.rowCount + rightPlan.rowCount ≤
      2 ^ Lifecycle.cubeVariables) :
    Exact (leftProgram.append rightProgram)
      (ProductionRelation.Plan.append leftPlan rightPlan fits) sourceRow := by
  refine ⟨?_, ?_⟩
  · rw [MatrixProgram.Program.append_rowCount,
      ProductionRelation.Plan.append_rowCount, left.rowCount, right.rowCount]
  · intro global
    exact MatrixProgram.Program.append_plan_row? leftProgram rightProgram
      leftPlan rightPlan fits sourceRow left.rowCount left.row? right.row?
      global

theorem Exact.mapColumns {source target : Nat} {predicate : Fin source → Prop}
    {program : Program} {plan : ProductionRelation.Plan source} {sourceRow : Nat → Option R1CS.Row}
    (exact : Exact program plan sourceRow) (projection : SourceProjection)
    (column : ∀ index, predicate index → Fin target)
    (supported : ∀ row port entry, entry ∈ (plan.forms row port).entries → predicate entry.column)
    (mapped : ∀ index live, projection.column? index.val = some (column index live).val) :
    Exact (program.mapColumns source projection) (plan.mapColumnsChecked column supported) sourceRow := by
  refine ⟨?_, ?_⟩
  · rw [Program.mapColumns_rowCount]
    exact exact.rowCount
  · intro row
    rw [Program.mapColumns_row?, exact.row? row]
    exact SourceProjection.ports?_checked projection column (plan.forms row) (supported row) mapped

end NightstreamFPrime.Layout.MatrixProgram
