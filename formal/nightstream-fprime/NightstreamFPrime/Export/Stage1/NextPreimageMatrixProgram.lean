import NightstreamFPrime.Export.MatrixProgram.PlanBridge
import NightstreamFPrime.Export.Stage1.NextPreimageDirectPlan
import NightstreamFPrime.Export.Stage1.PerApplicationPackage
import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryMatrixProgram

/-!
Owns the compact ordinary-row program for the five Construction 2
next-preimage equations. It reuses the exact PiCCS retained substitution and
selects the final five rows of the Lean-authored per-application package.
-/

namespace NightstreamFPrime.Export.Stage1.NextPreimageMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1

abbrev ApplicationProgram := Lifecycle.Stage1.Application.Program

def rowSchedule (application : ApplicationProgram) : IndexSchedule :=
  .rangeList [
    ⟨PerApplicationPackage.nextPreimageRowStart application, 5⟩]

@[simp] theorem rowSchedule_count (application : ApplicationProgram) :
    (rowSchedule application).count = 5 := by
  rfl

theorem rowSchedule_index? (application : ApplicationProgram)
    (index : Fin NextPreimageDirectPlan.program.rowCount) :
    (rowSchedule application).index? index.val =
      some (PerApplicationPackage.nextPreimageRowStart application +
        index.val) := by
  have bound : index.val < 5 := by
    simpa using index.isLt
  simp [rowSchedule, bound]

def ordinaryBlock
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : Ordinary.Block where
  rows := rowSchedule application
  oneColumn := (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val
  substitution := PiCCSOrdinaryMatrixProgram.substitution application
  projection := .identity

def matrixProgram
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application
      logicalWidth) : MatrixProgram.Program where
  blocks := [.ordinary (ordinaryBlock geometry)]

@[simp] theorem matrixProgram_rowCount
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application
      logicalWidth) :
    (matrixProgram geometry).rowCount = 5 := by
  rfl

private theorem programRow_support
    (index : Fin NextPreimageDirectPlan.program.rowCount) :
    (NextPreimageDirectPlan.program.row index).VarsSatisfy
      PiCCSOrdinarySourceSupport.Target := by
  exact NextPreimageDirectPlan.sourceRows_varsSatisfy _
    (List.get_mem _ index)

private theorem substitution_agrees_on_row
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application
      logicalWidth)
    (row : R1CS.Row)
    (scope : row.VarsSatisfy PiCCSOrdinarySourceSupport.Target) :
    Ordinary.AgreesOnTerms
        (PiCCSOrdinaryMatrixProgram.substitution application)
        (PiCCSOrdinaryDirectPlan.sourceMap geometry) row.a.terms ∧
      Ordinary.AgreesOnTerms
        (PiCCSOrdinaryMatrixProgram.substitution application)
        (PiCCSOrdinaryDirectPlan.sourceMap geometry) row.b.terms ∧
      Ordinary.AgreesOnTerms
        (PiCCSOrdinaryMatrixProgram.substitution application)
        (PiCCSOrdinaryDirectPlan.sourceMap geometry) row.c.terms := by
  refine ⟨?_, ?_, ?_⟩
  · intro term member bounded
    exact PiCCSOrdinaryMatrixProgram.substitution_agrees_on_target geometry
      ⟨term.1, bounded⟩ (scope.1 term member)
  · intro term member bounded
    exact PiCCSOrdinaryMatrixProgram.substitution_agrees_on_target geometry
      ⟨term.1, bounded⟩ (scope.2.1 term member)
  · intro term member bounded
    exact PiCCSOrdinaryMatrixProgram.substitution_agrees_on_target geometry
      ⟨term.1, bounded⟩ (scope.2.2 term member)

def directForms
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application
      logicalWidth)
    (index : Fin NextPreimageDirectPlan.program.rowCount) :
    OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (PiCCSOrdinaryDirectPlan.sourceMap geometry)
    (PiCCSOrdinaryRetainedGeometry.oneColumn geometry)
    (NextPreimageDirectPlan.program.row index)
    (NextPreimageDirectPlan.program.bounded index)

theorem plan_forms
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application
      logicalWidth)
    (index : Fin NextPreimageDirectPlan.program.rowCount) :
    (NextPreimageDirectPlan.plan geometry).forms index =
      (directForms geometry index).meaningfulForm := by
  simpa only [NextPreimageDirectPlan.plan, directForms,
    NextPreimageDirectPlan.inputs] using
      (OrdinarySourcePlan.Program.compile_toPlan_forms
        NextPreimageDirectPlan.program
        (NextPreimageDirectPlan.inputs geometry) index)

theorem ordinaryBlock_row?
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application
      logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (index : Fin NextPreimageDirectPlan.program.rowCount)
    (loaded : sourceRow
        (PerApplicationPackage.nextPreimageRowStart application + index.val) =
      some (NextPreimageDirectPlan.program.row index)) :
    (ordinaryBlock geometry).row? logicalWidth sourceRow index.val =
      some (directForms geometry index) := by
  rcases substitution_agrees_on_row geometry _ (programRow_support index) with
    ⟨agreesA, agreesB, agreesC⟩
  exact Ordinary.Block.row?_eq_compileRow (ordinaryBlock geometry) sourceRow
    index.val
    (PerApplicationPackage.nextPreimageRowStart application + index.val)
    (NextPreimageDirectPlan.program.row index)
    (rowSchedule_index? application index)
    (NextPreimageDirectPlan.program.row index) loaded (by simp [ordinaryBlock])
    (PiCCSOrdinaryDirectPlan.sourceMap geometry)
    (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) rfl
    (NextPreimageDirectPlan.program.bounded index) agreesA agreesB agreesC

/-- Every compact next-preimage row equals the corresponding canonical Lean
plan row. -/
theorem matrixProgram_row?
    {application : ApplicationProgram} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry application
      logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loaded : ∀ index : Fin NextPreimageDirectPlan.program.rowCount,
      sourceRow
          (PerApplicationPackage.nextPreimageRowStart application + index.val) =
        some (NextPreimageDirectPlan.program.row index))
    (global : Fin (NextPreimageDirectPlan.plan geometry).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((NextPreimageDirectPlan.plan geometry).forms global) := by
  have blockBound : global.val <
      (MatrixProgram.Block.ordinary (ordinaryBlock geometry)).rowCount := by
    change global.val < 5
    have bound := global.isLt
    have count := NextPreimageDirectPlan.plan_rowCount geometry
    omega
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.ordinary (ordinaryBlock geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos blockBound]
  change (do
    let forms ←
      (ordinaryBlock geometry).row? logicalWidth sourceRow global.val
    pure forms.meaningfulForm) = _
  rw [ordinaryBlock_row? geometry sourceRow global (loaded global)]
  apply congrArg some
  exact (plan_forms geometry global).symm

end NightstreamFPrime.Export.Stage1.NextPreimageMatrixProgram
