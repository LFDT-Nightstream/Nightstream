import NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgramAllRows

/-! The PiRLC arithmetic matrix program uses the four canonical Phi81 product
families. Sampler rows are owned by the preceding sampler block. -/

namespace NightstreamFPrime.Export.Stage1.PiRLCMatrixProgram

open NightstreamFPrime.Layout NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout.ProductionRelation
open PiRLCProductMatrixProgram

def matrixProgram {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) : MatrixProgram.Program :=
  PiRLCProductMatrixProgram.matrixProgram geometry

@[simp] theorem matrixProgram_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 67932 :=
  PiRLCProductMatrixProgram.matrixProgram_rowCount geometry

theorem matrixProgram_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PiRLCRetainedPlan.plan
      (PiRLCValueWiring.form geometry) (prefixGeometry geometry)).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiRLCRetainedPlan.plan
        (PiRLCValueWiring.form geometry) (prefixGeometry geometry)).forms global) :=
  PiRLCProductMatrixProgram.matrixProgram_plan_row? geometry sourceRow global

end NightstreamFPrime.Export.Stage1.PiRLCMatrixProgram
