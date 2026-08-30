import NightstreamFPrime.Export.Stage1.ApplicationMatrixProgramSubstitution

/-!
Proves row-by-row equality between the compact application matrix program and
the canonical direct 14-matrix plan for the selected Lean application. The
package row accessor remains an explicit identity-checked premise.
-/

namespace NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Spec
open ApplicationRetainedGeometry

theorem rowSchedule_index?
    (application : ApplicationProgram) (fits : FitsTwoPow28 application)
    (index : Fin (ApplicationDirectSource.program application fits).rowCount) :
    (rowSchedule application).index? index.val =
      some (PerApplicationPackage.basePackage.layout.rowCount + index.val) := by
  have count := ApplicationDirectSource.program_rowCount application fits
  have bound : index.val <
      (PerApplicationPackage.applicationPlan application).rowCount := by
    calc
      index.val < (ApplicationDirectSource.program application fits).rowCount :=
        index.isLt
      _ = (PerApplicationPackage.applicationPlan application).rowCount := count
  simp [rowSchedule, bound]

private theorem programRow_support
    (application : ApplicationProgram) (fits : FitsTwoPow28 application)
    (index : Fin (ApplicationDirectSource.program application fits).rowCount) :
    ((ApplicationDirectSource.program application fits).row index).VarsSatisfy
      (ApplicationDirectSource.SourceAllowed application) := by
  exact ApplicationDirectSource.sourceRows_varsSatisfy application _
    (List.get_mem _ index)

private theorem substitution_agrees_on_row
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry application logicalWidth) (row : R1CS.Row)
    (scope : row.VarsSatisfy
      (ApplicationDirectSource.SourceAllowed application)) :
    Ordinary.AgreesOnTerms (substitution application)
        (ApplicationDirectPlan.sourceMap geometry) row.a.terms ∧
      Ordinary.AgreesOnTerms (substitution application)
        (ApplicationDirectPlan.sourceMap geometry) row.b.terms ∧
      Ordinary.AgreesOnTerms (substitution application)
        (ApplicationDirectPlan.sourceMap geometry) row.c.terms := by
  refine ⟨?_, ?_, ?_⟩
  · intro term member bounded
    exact substitution_agrees_on_target geometry ⟨term.1, bounded⟩
      (scope.1 term member)
  · intro term member bounded
    exact substitution_agrees_on_target geometry ⟨term.1, bounded⟩
      (scope.2.1 term member)
  · intro term member bounded
    exact substitution_agrees_on_target geometry ⟨term.1, bounded⟩
      (scope.2.2 term member)

def directForms
    {application : ApplicationProgram} {logicalWidth : Nat}
    (fits : FitsTwoPow28 application)
    (geometry : Geometry application logicalWidth)
    (index : Fin (ApplicationDirectSource.program application fits).rowCount) :
    OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (ApplicationDirectPlan.sourceMap geometry)
    (oneColumn geometry)
    ((ApplicationDirectSource.program application fits).row index)
    ((ApplicationDirectSource.program application fits).bounded index)

theorem plan_forms
    {application : ApplicationProgram} {logicalWidth : Nat}
    (fits : FitsTwoPow28 application)
    (geometry : Geometry application logicalWidth)
    (index : Fin (ApplicationDirectSource.program application fits).rowCount) :
    (ApplicationDirectPlan.plan fits geometry).forms index =
      (directForms fits geometry index).meaningfulForm := by
  simpa only [ApplicationDirectPlan.plan, directForms,
    ApplicationDirectPlan.inputs] using
      (OrdinarySourcePlan.Program.compile_toPlan_forms
        (ApplicationDirectSource.program application fits)
        (ApplicationDirectPlan.inputs fits geometry) index)

theorem ordinaryBlock_row?
    {application : ApplicationProgram} {logicalWidth : Nat}
    (fits : FitsTwoPow28 application)
    (geometry : Geometry application logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (index : Fin (ApplicationDirectSource.program application fits).rowCount)
    (loaded : sourceRow
        (PerApplicationPackage.basePackage.layout.rowCount + index.val) =
      some ((ApplicationDirectSource.program application fits).row index)) :
    (ordinaryBlock geometry).row? logicalWidth sourceRow index.val =
      some (directForms fits geometry index) := by
  rcases substitution_agrees_on_row geometry _
      (programRow_support application fits index) with
    ⟨agreesA, agreesB, agreesC⟩
  have projected : (ordinaryBlock geometry).projection.row?
      ((ApplicationDirectSource.program application fits).row index) =
        some ((ApplicationDirectSource.program application fits).row index) := by
    exact PerApplicationSourceProjection.application_row _
  exact Ordinary.Block.row?_eq_compileRow (ordinaryBlock geometry) sourceRow
    index.val (PerApplicationPackage.basePackage.layout.rowCount + index.val)
    ((ApplicationDirectSource.program application fits).row index)
    (rowSchedule_index? application fits index) _ loaded projected
    (ApplicationDirectPlan.sourceMap geometry) (oneColumn geometry) rfl
    ((ApplicationDirectSource.program application fits).bounded index)
    agreesA agreesB agreesC

/-- Every row of the selected application's compact program is the exact row
of its canonical direct Lean plan. -/
theorem matrixProgram_row?
    {application : ApplicationProgram} {logicalWidth : Nat}
    (fits : FitsTwoPow28 application)
    (geometry : Geometry application logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loaded : ∀ index :
      Fin (ApplicationDirectSource.program application fits).rowCount,
      sourceRow (PerApplicationPackage.basePackage.layout.rowCount +
          index.val) =
        some ((ApplicationDirectSource.program application fits).row index))
    (global : Fin (ApplicationDirectPlan.plan fits geometry).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((ApplicationDirectPlan.plan fits geometry).forms global) := by
  have blockBound : global.val <
      (MatrixProgram.Block.ordinary (ordinaryBlock geometry)).rowCount := by
    change global.val <
      (PerApplicationPackage.applicationPlan application).rowCount
    have bound := global.isLt
    have count := ApplicationDirectPlan.plan_rowCount fits geometry
    omega
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.ordinary (ordinaryBlock geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos blockBound]
  change (do
    let forms ← (ordinaryBlock geometry).row? logicalWidth sourceRow global.val
    pure forms.meaningfulForm) = _
  rw [ordinaryBlock_row? fits geometry sourceRow global (loaded global)]
  apply congrArg some
  exact (plan_forms fits geometry global).symm

end NightstreamFPrime.Export.Stage1.ApplicationMatrixProgram
