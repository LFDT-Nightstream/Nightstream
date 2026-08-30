import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Layout.ProductionRelation.PlanComposition

/-!
Owns the generic exact-row bridge for ordered compact-program composition.
If two child programs return the exact rows of two child plans, their ordered
append returns the exact rows of the canonical appended plan.
-/

namespace NightstreamFPrime.Export.MatrixProgram

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

/-- Exact child row bridges compose through the same ordered append used by
the canonical production plan. -/
theorem Program.append_plan_row?
    {logicalWidth : Nat}
    (leftProgram rightProgram : Program)
    (leftPlan rightPlan : ProductionRelation.Plan logicalWidth)
    (planFits : leftPlan.rowCount + rightPlan.rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables)
    (sourceRow : Nat → Option R1CS.Row)
    (leftCount : leftProgram.rowCount = leftPlan.rowCount)
    (leftExact : ∀ row : Fin leftPlan.rowCount,
      leftProgram.row? logicalWidth sourceRow row.val =
        some (leftPlan.forms row))
    (rightExact : ∀ row : Fin rightPlan.rowCount,
      rightProgram.row? logicalWidth sourceRow row.val =
        some (rightPlan.forms row))
    (global : Fin (ProductionRelation.Plan.append
      leftPlan rightPlan planFits).rowCount) :
    (leftProgram.append rightProgram).row? logicalWidth sourceRow global.val =
      some ((ProductionRelation.Plan.append
        leftPlan rightPlan planFits).forms global) := by
  cases selected : ProductionRelation.Plan.splitIndex
      leftPlan.rowCount rightPlan.rowCount global with
  | inl leftRow =>
      have globalEq := ProductionRelation.Plan.leftIndex_of_splitIndex_eq
        leftPlan.rowCount rightPlan.rowCount global leftRow selected
      rw [← globalEq]
      have bound : leftRow.val < leftProgram.rowCount := by
        rw [leftCount]
        exact leftRow.isLt
      calc
        (leftProgram.append rightProgram).row? logicalWidth sourceRow
            (ProductionRelation.Plan.leftIndex leftPlan.rowCount
              rightPlan.rowCount leftRow).val =
            leftProgram.row? logicalWidth sourceRow leftRow.val := by
              simpa only [ProductionRelation.Plan.leftIndex_val] using
                (Program.append_left_row? leftProgram rightProgram
                  logicalWidth sourceRow leftRow.val bound)
        _ = some (leftPlan.forms leftRow) := leftExact leftRow
        _ = some ((ProductionRelation.Plan.append
              leftPlan rightPlan planFits).forms
                (ProductionRelation.Plan.leftIndex leftPlan.rowCount
                  rightPlan.rowCount leftRow)) := by
              apply congrArg some
              funext port
              exact (ProductionRelation.Plan.append_forms_left
                leftPlan rightPlan planFits leftRow port).symm
  | inr rightRow =>
      have globalEq := ProductionRelation.Plan.rightIndex_of_splitIndex_eq
        leftPlan.rowCount rightPlan.rowCount global rightRow selected
      rw [← globalEq]
      have selectedProgram := Program.append_right_row? leftProgram
        rightProgram logicalWidth sourceRow rightRow.val
      rw [leftCount] at selectedProgram
      calc
        (leftProgram.append rightProgram).row? logicalWidth sourceRow
            (ProductionRelation.Plan.rightIndex leftPlan.rowCount
              rightPlan.rowCount rightRow).val =
            rightProgram.row? logicalWidth sourceRow rightRow.val := by
              simpa only [ProductionRelation.Plan.rightIndex_val] using
                selectedProgram
        _ = some (rightPlan.forms rightRow) := rightExact rightRow
        _ = some ((ProductionRelation.Plan.append
              leftPlan rightPlan planFits).forms
                (ProductionRelation.Plan.rightIndex leftPlan.rowCount
                  rightPlan.rowCount rightRow)) := by
              apply congrArg some
              funext port
              exact (ProductionRelation.Plan.append_forms_right
                leftPlan rightPlan planFits rightRow port).symm

end NightstreamFPrime.Export.MatrixProgram
