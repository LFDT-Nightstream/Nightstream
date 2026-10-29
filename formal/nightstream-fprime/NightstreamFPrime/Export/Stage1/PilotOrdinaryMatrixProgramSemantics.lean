import NightstreamFPrime.Export.Stage1.PilotOrdinaryMatrixProgram

namespace NightstreamFPrime.Export.Stage1.PilotOrdinaryMatrixProgram

open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open PilotOrdinaryRetainedGeometry

def directForms
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 686) :
    OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow
    (PilotOrdinaryDirectPlan.sourceMap geometry) (oneColumn geometry)
    (PilotOrdinaryDirectSource.programRow index)
    (PilotOrdinaryDirectSource.programRow_bounded index)

theorem plan_forms
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 686) :
    (PilotOrdinaryDirectPlan.plan geometry).forms index =
      (directForms geometry index).meaningfulForm := by
  rfl

theorem compile_programRow?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 686) :
    Ordinary.compileRow? (substitution program) logicalWidth
        (oneColumn geometry).val
        (PilotOrdinaryDirectSource.programRow index) =
      some (directForms geometry index) := by
  rcases substitution_agrees_on_programRow geometry index with
    ⟨agreesA, agreesB, agreesC⟩
  exact Ordinary.compileRow?_eq_compileRow (substitution program)
    (PilotOrdinaryDirectPlan.sourceMap geometry) (oneColumn geometry)
    (PilotOrdinaryDirectSource.programRow index)
    (PilotOrdinaryDirectSource.programRow_bounded index)
    agreesA agreesB agreesC

theorem block_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) (index : Fin 686)
    (loaded : sourceRow (rowIndexAt index) =
      some (PerApplicationSourceProjection.pilotPackageRow program
        (PilotOrdinaryDirectSource.programRow index))) :
    (block geometry).row? logicalWidth sourceRow index.val =
      some (directForms geometry index) := by
  rcases substitution_agrees_on_programRow geometry index with
    ⟨agreesA, agreesB, agreesC⟩
  have projected : (block geometry).projection.row?
      (PerApplicationSourceProjection.pilotPackageRow program
        (PilotOrdinaryDirectSource.programRow index)) =
        some (PilotOrdinaryDirectSource.programRow index) := by
    apply PerApplicationSourceProjection.pilot_row
    exact PilotOrdinaryDirectSource.programRow_bounded index
  exact Ordinary.Block.row?_eq_compileRow (block geometry) sourceRow
    index.val (rowIndexAt index)
    (PilotOrdinaryDirectSource.programRow index)
    (rowSchedule_indexAt index) _ loaded projected
    (PilotOrdinaryDirectPlan.sourceMap geometry) (oneColumn geometry) rfl
    (PilotOrdinaryDirectSource.programRow_bounded index)
    agreesA agreesB agreesC

theorem matrixProgram_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loaded : ∀ index : Fin 686,
      sourceRow (rowIndexAt index) =
        some (PerApplicationSourceProjection.pilotPackageRow program
          (PilotOrdinaryDirectSource.programRow index)))
    (global : Fin (PilotOrdinaryDirectPlan.plan geometry).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PilotOrdinaryDirectPlan.plan geometry).forms global) := by
  change Fin 686 at global
  have blockBound : global.val <
      (MatrixProgram.Block.ordinary (block geometry)).rowCount := by
    change global.val < (block geometry).rowCount
    rw [block_rowCount]
    exact global.isLt
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.ordinary (block geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, ite_eq_left blockBound]
  change (do
    let forms ← (block geometry).row? logicalWidth sourceRow global.val
    pure forms.meaningfulForm) = _
  have loadedRow := loaded global
  have exactRow := block_row? geometry sourceRow global loadedRow
  rw [exactRow]
  exact congrArg some (plan_forms geometry global).symm

end NightstreamFPrime.Export.Stage1.PilotOrdinaryMatrixProgram
