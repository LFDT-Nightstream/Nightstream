import NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram
import NightstreamFPrime.Export.Stage1.ApplicationOrdinaryMatrixSemantics

/-! The executable program reconstructs the complete application plan. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram

open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open ApplicationOrdinaryGeometry

theorem matrixProgram_row? {application : Stage1.Application.Program} {columns : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application) (geometry : Geometry application columns)
    (sourceRow : Nat → Option R1CS.Row)
    (loaded : ∀ index : Fin (ApplicationDirectSource.program application fits).rowCount,
      sourceRow (PerApplicationPackage.basePackage.layout.rowCount + index.val) =
        some ((ApplicationDirectSource.program application fits).row index))
    (global : Fin (ApplicationDirectPlan.plan fits geometry).rowCount) :
    (matrixProgram geometry).row? columns sourceRow global.val =
      some ((ApplicationDirectPlan.plan fits geometry).forms global) :=
  ApplicationOrdinaryMatrixProgram.matrixProgram_row? fits geometry sourceRow loaded global

end NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram
