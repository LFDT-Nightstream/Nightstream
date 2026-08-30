import NightstreamFPrime.Export.Stage1.PiRLCFirst54MatrixProgramAllRows
import NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgramAllRows

/-!
Owns the complete compact matrix program for the canonical retained PiRLC
plan. Product-family rows precede First54 rows, as required by the Lean plan.

This module does not compose the PiRLC sampler rows or later Stage 1 phases.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

/-- Product rows followed by First54 rows. -/
def matrixProgram {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    MatrixProgram.Program :=
  (PiRLCProductMatrixProgram.matrixProgram geometry).append
    (PiRLCFirst54MatrixProgram.matrixProgram geometry)

@[simp] theorem matrixProgram_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 1773933 := by
  simp [matrixProgram]

/-- Every row in the compact PiRLC program is the exact row in the canonical
retained PiRLC plan. -/
theorem matrixProgram_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PiRLCRetainedPlan.plan geometry).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiRLCRetainedPlan.plan geometry).forms global) := by
  let productPlan := PiRLCProductPlan.plan
    (PiRLCRetainedInputs.productInputs geometry)
  let first54Plan := PiRLCFirst54DirectPlan.plan
    (PiRLCRetainedInputs.first54Inputs geometry)
  cases selected : ProductionRelation.Plan.splitIndex
      productPlan.rowCount first54Plan.rowCount global with
  | inl productRow =>
      have globalEq := ProductionRelation.Plan.leftIndex_of_splitIndex_eq
        productPlan.rowCount first54Plan.rowCount global productRow selected
      rw [← globalEq]
      calc
        (matrixProgram geometry).row? logicalWidth sourceRow
            (ProductionRelation.Plan.leftIndex productPlan.rowCount
              first54Plan.rowCount productRow).val =
            (PiRLCProductMatrixProgram.matrixProgram geometry).row?
              logicalWidth sourceRow productRow.val := by
                simpa only [matrixProgram,
                  ProductionRelation.Plan.leftIndex_val] using
                    (MatrixProgram.Program.append_left_row?
                      (PiRLCProductMatrixProgram.matrixProgram geometry)
                      (PiRLCFirst54MatrixProgram.matrixProgram geometry)
                      logicalWidth sourceRow productRow.val productRow.isLt)
        _ = some (productPlan.forms productRow) := by
              exact PiRLCProductMatrixProgram.matrixProgram_plan_row?
                geometry sourceRow productRow
        _ = some ((PiRLCRetainedPlan.plan geometry).forms
              (ProductionRelation.Plan.leftIndex productPlan.rowCount
                first54Plan.rowCount productRow)) := by
              apply congrArg some
              funext port
              exact (ProductionRelation.Plan.append_forms_left
                productPlan first54Plan
                (PiRLCRetainedPlan.childRowCount_le geometry)
                productRow port).symm
  | inr first54Row =>
      have globalEq := ProductionRelation.Plan.rightIndex_of_splitIndex_eq
        productPlan.rowCount first54Plan.rowCount global first54Row selected
      rw [← globalEq]
      have productCount :
          (PiRLCProductMatrixProgram.matrixProgram geometry).rowCount =
            productPlan.rowCount := by
        simp [productPlan]
      have selectedProgram := MatrixProgram.Program.append_right_row?
        (PiRLCProductMatrixProgram.matrixProgram geometry)
        (PiRLCFirst54MatrixProgram.matrixProgram geometry)
        logicalWidth sourceRow first54Row.val
      rw [productCount] at selectedProgram
      calc
        (matrixProgram geometry).row? logicalWidth sourceRow
            (ProductionRelation.Plan.rightIndex productPlan.rowCount
              first54Plan.rowCount first54Row).val =
            (PiRLCFirst54MatrixProgram.matrixProgram geometry).row?
              logicalWidth sourceRow first54Row.val := by
                simpa only [matrixProgram,
                  ProductionRelation.Plan.rightIndex_val] using selectedProgram
        _ = some (first54Plan.forms first54Row) := by
              exact PiRLCFirst54MatrixProgram.matrixProgram_row?
                geometry sourceRow first54Row
        _ = some ((PiRLCRetainedPlan.plan geometry).forms
              (ProductionRelation.Plan.rightIndex productPlan.rowCount
                first54Plan.rowCount first54Row)) := by
              apply congrArg some
              funext port
              exact (ProductionRelation.Plan.append_forms_right
                productPlan first54Plan
                (PiRLCRetainedPlan.childRowCount_le geometry)
                first54Row port).symm

end NightstreamFPrime.Export.Stage1.PiRLCMatrixProgram
