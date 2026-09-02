import NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgramSubstitution

/-!
Proves row-by-row equality between the compact running-transition matrix
program and the canonical direct 14-matrix plan. The package row accessor is
an explicit identity-checked premise.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open RunningTransitionRetainedGeometry

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

theorem rowSchedule_index?
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (index : Fin (RunningTransitionDirectSource.program relation).rowCount) :
    rowSchedule.index? index.val =
      some (RunningTransitionArithmetic.rowStart + index.val) := by
  have count := RunningTransitionDirectSource.program_rowCount relation
  have bound : index.val < 345495 := by
    calc
      index.val < (RunningTransitionDirectSource.program relation).rowCount :=
        index.isLt
      _ = 345495 := count
  simp [rowSchedule, bound]

private theorem programRow_support
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (index : Fin (RunningTransitionDirectSource.program relation).rowCount) :
    ((RunningTransitionDirectSource.program relation).row index).VarsSatisfy
      RunningTransitionSourceSupport.Target := by
  apply RunningTransitionSourceSupport.remappedRows_varsSatisfy relation
  exact List.get_mem _ index

private theorem substitution_agrees_on_row
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (row : R1CS.Row)
    (scope : row.VarsSatisfy RunningTransitionSourceSupport.Target) :
    Ordinary.AgreesOnTerms (substitution program)
        (RunningTransitionDirectPlan.sourceMap geometry) row.a.terms ∧
      Ordinary.AgreesOnTerms (substitution program)
        (RunningTransitionDirectPlan.sourceMap geometry) row.b.terms ∧
      Ordinary.AgreesOnTerms (substitution program)
        (RunningTransitionDirectPlan.sourceMap geometry) row.c.terms := by
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
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (index : Fin (RunningTransitionDirectSource.program relation).rowCount) :
    OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (RunningTransitionDirectPlan.sourceMap geometry)
    (oneColumn geometry)
    ((RunningTransitionDirectSource.program relation).row index)
    ((RunningTransitionDirectSource.program relation).bounded index)

theorem plan_forms
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (index : Fin (RunningTransitionDirectSource.program relation).rowCount) :
    (RunningTransitionDirectPlan.plan relation geometry).forms index =
      (directForms relation geometry index).meaningfulForm := by
  simpa only [RunningTransitionDirectPlan.plan, directForms,
    RunningTransitionDirectPlan.inputs] using
      (OrdinarySourcePlan.Program.compile_toPlan_forms
        (RunningTransitionDirectSource.program relation)
        (RunningTransitionDirectPlan.inputs relation geometry) index)

theorem ordinaryBlock_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (index : Fin (RunningTransitionDirectSource.program relation).rowCount)
    (loaded : sourceRow
        (RunningTransitionArithmetic.rowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow program
        ((RunningTransitionDirectSource.program relation).row index))) :
    (ordinaryBlock geometry).row? logicalWidth sourceRow index.val =
      some (directForms relation geometry index) := by
  rcases substitution_agrees_on_row geometry _
      (programRow_support relation index) with
    ⟨agreesA, agreesB, agreesC⟩
  have projected : (ordinaryBlock geometry).projection.row?
      (PerApplicationSourceProjection.basePackageRow program
        ((RunningTransitionDirectSource.program relation).row index)) =
        some ((RunningTransitionDirectSource.program relation).row index) := by
    apply PerApplicationSourceProjection.base_row
    simpa [SourceCompiler.RowBounded, SourceCompiler.CombinationBounded,
      R1CS.Row.VarsBelow, R1CS.LinearCombination.VarsBelow,
      PerApplicationPackage.basePackage, Data.circuitPackage_layout,
      Data.physicalLayout] using
      ((RunningTransitionDirectSource.program relation).bounded index)
  exact Ordinary.Block.row?_eq_compileRow (ordinaryBlock geometry) sourceRow
    index.val (RunningTransitionArithmetic.rowStart + index.val)
    ((RunningTransitionDirectSource.program relation).row index)
    (rowSchedule_index? relation index) _ loaded projected
    (RunningTransitionDirectPlan.sourceMap geometry) (oneColumn geometry) rfl
    ((RunningTransitionDirectSource.program relation).bounded index)
    agreesA agreesB agreesC

/-- Every row of the compact running-transition program is the exact row of
the canonical direct Lean plan. -/
theorem matrixProgram_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loaded : ∀ index :
      Fin (RunningTransitionDirectSource.program relation).rowCount,
      sourceRow (RunningTransitionArithmetic.rowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          ((RunningTransitionDirectSource.program relation).row index)))
    (global : Fin
      (RunningTransitionDirectPlan.plan relation geometry).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((RunningTransitionDirectPlan.plan relation geometry).forms
        global) := by
  have blockBound : global.val <
      (MatrixProgram.Block.ordinary (ordinaryBlock geometry)).rowCount := by
    change global.val < 345495
    have bound := global.isLt
    have count := RunningTransitionDirectPlan.plan_rowCount relation geometry
    omega
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.ordinary (ordinaryBlock geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos blockBound]
  change (do
    let forms ← (ordinaryBlock geometry).row? logicalWidth sourceRow global.val
    pure forms.meaningfulForm) = _
  rw [ordinaryBlock_row? relation geometry sourceRow global (loaded global)]
  apply congrArg some
  exact (plan_forms relation geometry global).symm

end NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgram
