import NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PiRLCRetainedPlan

/-!
Connects every row of the compact PiRLC product matrix program to the exact
canonical product plan. This is literal sparse-form equality for all 13
meaningful matrix ports.

This module does not compose the First54 rows or later Stage 1 phases.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

/-- Every compact product-program row is the same row as the canonical
invocation-major PiRLC product plan. -/
theorem matrixProgram_plan_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PiRLCProductPlan.plan
      (inputs geometry)).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiRLCProductPlan.plan
        (inputs geometry)).forms global) := by
  change Fin (PiRLCProductSchedule.invocationCount * 34) at global
  simpa [PiRLCProductPlan.plan, Phi81ProductFamilyPlan.plan,
    ProductionRelation.Plan.indexed] using
      (matrixProgram_row? geometry sourceRow global)

end NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgram
