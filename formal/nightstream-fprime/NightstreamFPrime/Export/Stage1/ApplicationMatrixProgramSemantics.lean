import NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram
import NightstreamFPrime.Export.Stage1.ApplicationOrdinaryMatrixSemantics
import NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixRows

/-! Both executable backends reconstruct the exact selected application plan. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram

open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open ApplicationRetainedGeometry

theorem matrixProgram_row? {application : Stage1.Application.Program} {columns : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application) (geometry : Geometry application columns)
    (sourceRow : Nat → Option R1CS.Row)
    (loaded : ∀ index : Fin (ApplicationDirectSource.program application fits).rowCount,
      sourceRow (PerApplicationPackage.basePackage.layout.rowCount + index.val) =
        some ((ApplicationDirectSource.program application fits).row index))
    (global : Fin (ApplicationDirectPlan.plan fits geometry).rowCount) :
    (matrixProgram geometry).row? columns sourceRow global.val =
      some ((ApplicationDirectPlan.plan fits geometry).forms global) := by
  revert global
  cases selected : application.compactHashChain with
  | none =>
    rw [matrixProgram_none geometry selected, ApplicationDirectPlan.plan_none fits geometry selected]
    exact ApplicationOrdinaryMatrixProgram.matrixProgram_row? fits (ordinaryGeometry geometry selected)
      sourceRow loaded
  | some certificate =>
    rw [matrixProgram_some geometry certificate selected,
      ApplicationDirectPlan.plan_some fits geometry certificate selected]
    exact ApplicationPoseidonMatrixProgram.matrixProgram_row?
      (poseidonGeometry geometry certificate selected) sourceRow

end NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram
