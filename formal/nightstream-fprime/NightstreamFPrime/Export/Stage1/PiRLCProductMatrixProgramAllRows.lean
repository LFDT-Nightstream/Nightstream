import NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PiRLCRetainedPlan

namespace NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgram

open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

theorem matrixProgram_plan_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PiRLCProductPlan.plan
      (inputs geometry)).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiRLCProductPlan.plan
        (inputs geometry)).forms global) := by
  change Fin (PiRLCProductRingSchedule.invocationCount * 108) at global
  simpa [PiRLCProductPlan.plan, Phi81ProductFamilyPlan.plan,
    ProductionRelation.Plan.indexed] using
      (matrixProgram_row? geometry sourceRow global)

end NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgram
