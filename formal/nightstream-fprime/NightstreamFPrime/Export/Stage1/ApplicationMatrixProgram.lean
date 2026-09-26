import NightstreamFPrime.Export.Stage1.ApplicationDirectPlan
import NightstreamFPrime.Export.Stage1.ApplicationOrdinaryMatrixProgram

/-! The executable matrix program for the shared application layout. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram

open NightstreamFPrime.Layout NightstreamFPrime.Lifecycle ApplicationOrdinaryGeometry

def matrixProgram {application : Stage1.Application.Program} {columns : Nat}
    (geometry : Geometry application columns) : MatrixProgram.Program :=
  ApplicationOrdinaryMatrixProgram.matrixProgram geometry

@[simp] theorem matrixProgram_rowCount {application : Stage1.Application.Program} {columns : Nat}
    (geometry : Geometry application columns) :
    (matrixProgram geometry).rowCount = ApplicationDirectPlan.rowCount application :=
  ApplicationOrdinaryMatrixProgram.matrixProgram_rowCount geometry

end NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram
