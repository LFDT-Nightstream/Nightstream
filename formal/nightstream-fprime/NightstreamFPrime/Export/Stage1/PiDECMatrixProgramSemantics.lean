import NightstreamFPrime.Export.Stage1.PiDECMatrixProgramSubstitution

/-!
Proves row-by-row equality between the compact PiDEC matrix program and the
canonical direct 14-matrix PiDEC plan. The package row accessor is an explicit
identity-checked premise.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.Stage1.PiDECSourceSupport
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiDECRetainedGeometry

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

theorem publicSchedule_index? (index : Fin 22680) :
    publicSchedule.index? index.val =
      some (PiDECStarts.publicInputRowStart + index.val) := by
  simp [publicSchedule, index.isLt]

theorem commitmentSchedule_index? (index : Fin 972) :
    commitmentSchedule.index? index.val =
      some (PiDECStarts.commitmentRowStart + index.val) := by
  simp [commitmentSchedule, index.isLt]

theorem evalKSchedule_index? (index : Fin 108) :
    evalKSchedule.index? index.val =
      some (PiDECStarts.evalKRowStart + index.val) := by
  simp [evalKSchedule, index.isLt]

theorem evalASchedule_index? (index : Fin 1512) :
    evalASchedule.index? index.val =
      some (PiDECStarts.evalARowStart + index.val) := by
  simp [evalASchedule, index.isLt]

private theorem substitution_agrees_on_row
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (row : R1CS.Row)
    (scope : row.VarsSatisfy PiDECSourceSupport.Target) :
    Ordinary.AgreesOnTerms (substitution program)
        (PiDECDirectPlan.sourceMap geometry) row.a.terms ∧
      Ordinary.AgreesOnTerms (substitution program)
        (PiDECDirectPlan.sourceMap geometry) row.b.terms ∧
      Ordinary.AgreesOnTerms (substitution program)
        (PiDECDirectPlan.sourceMap geometry) row.c.terms := by
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

private theorem projection_recovers_row
    (program : ApplicationProgram) (row : R1CS.Row)
    (bounded : SourceCompiler.RowBounded Spartan.spartanColumnCount row) :
    (PerApplicationSourceProjection.base program).row?
        (PerApplicationSourceProjection.basePackageRow program row) =
      some row := by
  apply PerApplicationSourceProjection.base_row
  simpa [SourceCompiler.RowBounded, SourceCompiler.CombinationBounded,
    R1CS.Row.VarsBelow, R1CS.LinearCombination.VarsBelow,
    PerApplicationPackage.basePackage, Data.circuitPackage_layout,
    Data.physicalLayout] using bounded

def publicDirectForms
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) (index : Fin 22680) :
    OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (PiDECDirectPlan.sourceMap geometry)
    (oneColumn geometry) (PiDECOrdinaryDirectSource.publicProgramRow relation index)
    (PiDECOrdinaryDirectSource.publicProgramRow_bounded relation index)

def commitmentDirectForms
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) (index : Fin 972) :
    OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (PiDECDirectPlan.sourceMap geometry)
    (oneColumn geometry)
    (PiDECOrdinaryDirectSource.commitmentProgramRow relation index)
    (PiDECOrdinaryDirectSource.commitmentProgramRow_bounded relation index)

def evalKDirectForms
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) (index : Fin 108) :
    OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (PiDECDirectPlan.sourceMap geometry)
    (oneColumn geometry) (PiDECOrdinaryDirectSource.evalKProgramRow relation index)
    (PiDECOrdinaryDirectSource.evalKProgramRow_bounded relation index)

def evalADirectForms
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) (index : Fin 1512) :
    OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (PiDECDirectPlan.sourceMap geometry)
    (oneColumn geometry) (PiDECOrdinaryDirectSource.evalAProgramRow relation index)
    (PiDECOrdinaryDirectSource.evalAProgramRow_bounded relation index)

theorem publicPlan_forms
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) (index : Fin 22680) :
    (PiDECDirectPlan.publicPlan relation geometry).forms index =
      (publicDirectForms relation geometry index).meaningfulForm := by
  simpa only [PiDECDirectPlan.publicPlan, publicDirectForms,
    PiDECDirectPlan.inputs, PiDECDirectPlan.publicSource,
    PiDECDirectPlan.SupportedProgram.toProgram] using
      (OrdinarySourcePlan.Program.compile_toPlan_forms
        (PiDECDirectPlan.publicSource relation).toProgram
        (PiDECDirectPlan.inputs
          (PiDECDirectPlan.publicSource relation).toProgram geometry) index)

theorem commitmentPlan_forms
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) (index : Fin 972) :
    (PiDECDirectPlan.commitmentPlan relation geometry).forms index =
      (commitmentDirectForms relation geometry index).meaningfulForm := by
  simpa only [PiDECDirectPlan.commitmentPlan, commitmentDirectForms,
    PiDECDirectPlan.inputs, PiDECDirectPlan.commitmentSource,
    PiDECDirectPlan.SupportedProgram.toProgram] using
      (OrdinarySourcePlan.Program.compile_toPlan_forms
        (PiDECDirectPlan.commitmentSource relation).toProgram
        (PiDECDirectPlan.inputs
          (PiDECDirectPlan.commitmentSource relation).toProgram geometry) index)

theorem evalKPlan_forms
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) (index : Fin 108) :
    (PiDECDirectPlan.evalKPlan relation geometry).forms index =
      (evalKDirectForms relation geometry index).meaningfulForm := by
  simpa only [PiDECDirectPlan.evalKPlan, evalKDirectForms,
    PiDECDirectPlan.inputs, PiDECDirectPlan.evalKSource,
    PiDECDirectPlan.SupportedProgram.toProgram] using
      (OrdinarySourcePlan.Program.compile_toPlan_forms
        (PiDECDirectPlan.evalKSource relation).toProgram
        (PiDECDirectPlan.inputs
          (PiDECDirectPlan.evalKSource relation).toProgram geometry) index)

theorem evalAPlan_forms
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) (index : Fin 1512) :
    (PiDECDirectPlan.evalAPlan relation geometry).forms index =
      (evalADirectForms relation geometry index).meaningfulForm := by
  simpa only [PiDECDirectPlan.evalAPlan, evalADirectForms,
    PiDECDirectPlan.inputs, PiDECDirectPlan.evalASource,
    PiDECDirectPlan.SupportedProgram.toProgram] using
      (OrdinarySourcePlan.Program.compile_toPlan_forms
        (PiDECDirectPlan.evalASource relation).toProgram
        (PiDECDirectPlan.inputs
          (PiDECDirectPlan.evalASource relation).toProgram geometry) index)

theorem publicBlock_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) (index : Fin 22680)
    (loaded : sourceRow (PiDECStarts.publicInputRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow program
        (PiDECOrdinaryDirectSource.publicProgramRow relation index))) :
    (publicBlock geometry).row? logicalWidth sourceRow index.val =
      some (publicDirectForms relation geometry index) := by
  rcases substitution_agrees_on_row geometry _
      (PiDECOrdinaryDirectSource.publicProgramRow_varsSatisfy relation index) with
    ⟨agreesA, agreesB, agreesC⟩
  have projected := projection_recovers_row program
    (PiDECOrdinaryDirectSource.publicProgramRow relation index)
    (PiDECOrdinaryDirectSource.publicProgramRow_bounded relation index)
  exact Ordinary.Block.row?_eq_compileRow (publicBlock geometry) sourceRow
    index.val (PiDECStarts.publicInputRowStart + index.val)
    (PiDECOrdinaryDirectSource.publicProgramRow relation index)
    (publicSchedule_index? index) _ loaded projected
    (PiDECDirectPlan.sourceMap geometry)
    (oneColumn geometry) rfl
    (PiDECOrdinaryDirectSource.publicProgramRow_bounded relation index)
    agreesA agreesB agreesC

theorem commitmentBlock_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) (index : Fin 972)
    (loaded : sourceRow (PiDECStarts.commitmentRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow program
        (PiDECOrdinaryDirectSource.commitmentProgramRow relation index))) :
    (commitmentBlock geometry).row? logicalWidth sourceRow index.val =
      some (commitmentDirectForms relation geometry index) := by
  rcases substitution_agrees_on_row geometry _
      (PiDECOrdinaryDirectSource.commitmentProgramRow_varsSatisfy relation index) with
    ⟨agreesA, agreesB, agreesC⟩
  have projected := projection_recovers_row program
    (PiDECOrdinaryDirectSource.commitmentProgramRow relation index)
    (PiDECOrdinaryDirectSource.commitmentProgramRow_bounded relation index)
  exact Ordinary.Block.row?_eq_compileRow (commitmentBlock geometry) sourceRow
    index.val (PiDECStarts.commitmentRowStart + index.val)
    (PiDECOrdinaryDirectSource.commitmentProgramRow relation index)
    (commitmentSchedule_index? index) _ loaded projected
    (PiDECDirectPlan.sourceMap geometry) (oneColumn geometry) rfl
    (PiDECOrdinaryDirectSource.commitmentProgramRow_bounded relation index)
    agreesA agreesB agreesC

theorem evalKBlock_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) (index : Fin 108)
    (loaded : sourceRow (PiDECStarts.evalKRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow program
        (PiDECOrdinaryDirectSource.evalKProgramRow relation index))) :
    (evalKBlock geometry).row? logicalWidth sourceRow index.val =
      some (evalKDirectForms relation geometry index) := by
  rcases substitution_agrees_on_row geometry _
      (PiDECOrdinaryDirectSource.evalKProgramRow_varsSatisfy relation index) with
    ⟨agreesA, agreesB, agreesC⟩
  have projected := projection_recovers_row program
    (PiDECOrdinaryDirectSource.evalKProgramRow relation index)
    (PiDECOrdinaryDirectSource.evalKProgramRow_bounded relation index)
  exact Ordinary.Block.row?_eq_compileRow (evalKBlock geometry) sourceRow
    index.val (PiDECStarts.evalKRowStart + index.val)
    (PiDECOrdinaryDirectSource.evalKProgramRow relation index)
    (evalKSchedule_index? index) _ loaded projected
    (PiDECDirectPlan.sourceMap geometry)
    (oneColumn geometry) rfl
    (PiDECOrdinaryDirectSource.evalKProgramRow_bounded relation index)
    agreesA agreesB agreesC

theorem evalABlock_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) (index : Fin 1512)
    (loaded : sourceRow (PiDECStarts.evalARowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow program
        (PiDECOrdinaryDirectSource.evalAProgramRow relation index))) :
    (evalABlock geometry).row? logicalWidth sourceRow index.val =
      some (evalADirectForms relation geometry index) := by
  rcases substitution_agrees_on_row geometry _
      (PiDECOrdinaryDirectSource.evalAProgramRow_varsSatisfy relation index) with
    ⟨agreesA, agreesB, agreesC⟩
  have projected := projection_recovers_row program
    (PiDECOrdinaryDirectSource.evalAProgramRow relation index)
    (PiDECOrdinaryDirectSource.evalAProgramRow_bounded relation index)
  exact Ordinary.Block.row?_eq_compileRow (evalABlock geometry) sourceRow
    index.val (PiDECStarts.evalARowStart + index.val)
    (PiDECOrdinaryDirectSource.evalAProgramRow relation index)
    (evalASchedule_index? index) _ loaded projected
    (PiDECDirectPlan.sourceMap geometry)
    (oneColumn geometry) rfl
    (PiDECOrdinaryDirectSource.evalAProgramRow_bounded relation index)
    agreesA agreesB agreesC

theorem publicProgram_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loaded : ∀ index : Fin 22680,
      sourceRow (PiDECStarts.publicInputRowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.publicProgramRow relation index)))
    (index : Fin (PiDECDirectPlan.publicPlan relation geometry).rowCount) :
    (publicProgram geometry).row? logicalWidth sourceRow index.val =
      some ((PiDECDirectPlan.publicPlan relation geometry).forms index) := by
  change Fin 22680 at index
  have blockBound : index.val <
      (MatrixProgram.Block.ordinary (publicBlock geometry)).rowCount := by
    change index.val < 22680
    exact index.isLt
  rw [show publicProgram geometry =
      MatrixProgram.Program.mk [.ordinary (publicBlock geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos blockBound]
  change (do
    let forms ← (publicBlock geometry).row? logicalWidth sourceRow index.val
    pure forms.meaningfulForm) = _
  rw [publicBlock_row? relation geometry sourceRow index (loaded index)]
  apply congrArg some
  exact (publicPlan_forms relation geometry index).symm

theorem commitmentProgram_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loaded : ∀ index : Fin 972,
      sourceRow (PiDECStarts.commitmentRowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.commitmentProgramRow relation index)))
    (index : Fin (PiDECDirectPlan.commitmentPlan relation geometry).rowCount) :
    (commitmentProgram geometry).row? logicalWidth sourceRow index.val =
      some ((PiDECDirectPlan.commitmentPlan relation geometry).forms index) := by
  change Fin 972 at index
  have blockBound : index.val <
      (MatrixProgram.Block.ordinary (commitmentBlock geometry)).rowCount := by
    change index.val < 972
    exact index.isLt
  rw [show commitmentProgram geometry =
      MatrixProgram.Program.mk [.ordinary (commitmentBlock geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos blockBound]
  change (do
    let forms ← (commitmentBlock geometry).row? logicalWidth sourceRow index.val
    pure forms.meaningfulForm) = _
  rw [commitmentBlock_row? relation geometry sourceRow index (loaded index)]
  apply congrArg some
  exact (commitmentPlan_forms relation geometry index).symm

theorem evalKProgram_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loaded : ∀ index : Fin 108,
      sourceRow (PiDECStarts.evalKRowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.evalKProgramRow relation index)))
    (index : Fin (PiDECDirectPlan.evalKPlan relation geometry).rowCount) :
    (evalKProgram geometry).row? logicalWidth sourceRow index.val =
      some ((PiDECDirectPlan.evalKPlan relation geometry).forms index) := by
  change Fin 108 at index
  have blockBound : index.val <
      (MatrixProgram.Block.ordinary (evalKBlock geometry)).rowCount := by
    change index.val < 108
    exact index.isLt
  rw [show evalKProgram geometry =
      MatrixProgram.Program.mk [.ordinary (evalKBlock geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos blockBound]
  change (do
    let forms ← (evalKBlock geometry).row? logicalWidth sourceRow index.val
    pure forms.meaningfulForm) = _
  rw [evalKBlock_row? relation geometry sourceRow index (loaded index)]
  apply congrArg some
  exact (evalKPlan_forms relation geometry index).symm

theorem evalAProgram_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loaded : ∀ index : Fin 1512,
      sourceRow (PiDECStarts.evalARowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.evalAProgramRow relation index)))
    (index : Fin (PiDECDirectPlan.evalAPlan relation geometry).rowCount) :
    (evalAProgram geometry).row? logicalWidth sourceRow index.val =
      some ((PiDECDirectPlan.evalAPlan relation geometry).forms index) := by
  change Fin 1512 at index
  have blockBound : index.val <
      (MatrixProgram.Block.ordinary (evalABlock geometry)).rowCount := by
    change index.val < 1512
    exact index.isLt
  rw [show evalAProgram geometry =
      MatrixProgram.Program.mk [.ordinary (evalABlock geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos blockBound]
  change (do
    let forms ← (evalABlock geometry).row? logicalWidth sourceRow index.val
    pure forms.meaningfulForm) = _
  rw [evalABlock_row? relation geometry sourceRow index (loaded index)]
  apply congrArg some
  exact (evalAPlan_forms relation geometry index).symm

private theorem evalPlans_fit
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) :
    (PiDECDirectPlan.evalKPlan relation geometry).rowCount +
        (PiDECDirectPlan.evalAPlan relation geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  simp [Lifecycle.cubeVariables]

private theorem recompositionPlans_fit
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) :
    (PiDECDirectPlan.commitmentPlan relation geometry).rowCount +
        (PiDECDirectPlan.evaluationPlan relation geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  simp [Lifecycle.cubeVariables]

private theorem allPlans_fit
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) :
    (PiDECDirectPlan.publicPlan relation geometry).rowCount +
        (PiDECDirectPlan.recompositionPlan relation geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  simp [Lifecycle.cubeVariables]

theorem evaluationProgram_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loadedK : ∀ index : Fin 108,
      sourceRow (PiDECStarts.evalKRowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.evalKProgramRow relation index)))
    (loadedA : ∀ index : Fin 1512,
      sourceRow (PiDECStarts.evalARowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.evalAProgramRow relation index)))
    (global : Fin (PiDECDirectPlan.evaluationPlan relation geometry).rowCount) :
    (evaluationProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiDECDirectPlan.evaluationPlan relation geometry).forms global) := by
  simpa [evaluationProgram, PiDECDirectPlan.evaluationPlan] using
    (MatrixProgram.Program.append_plan_row?
      (evalKProgram geometry) (evalAProgram geometry)
      (PiDECDirectPlan.evalKPlan relation geometry)
      (PiDECDirectPlan.evalAPlan relation geometry)
      (evalPlans_fit relation geometry) sourceRow
      (by simp)
      (evalKProgram_row? relation geometry sourceRow loadedK)
      (evalAProgram_row? relation geometry sourceRow loadedA) global)

theorem recompositionProgram_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loadedCommitment : ∀ index : Fin 972,
      sourceRow (PiDECStarts.commitmentRowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.commitmentProgramRow relation index)))
    (loadedK : ∀ index : Fin 108,
      sourceRow (PiDECStarts.evalKRowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.evalKProgramRow relation index)))
    (loadedA : ∀ index : Fin 1512,
      sourceRow (PiDECStarts.evalARowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.evalAProgramRow relation index)))
    (global : Fin
      (PiDECDirectPlan.recompositionPlan relation geometry).rowCount) :
    (recompositionProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiDECDirectPlan.recompositionPlan relation geometry).forms
        global) := by
  simpa [recompositionProgram, PiDECDirectPlan.recompositionPlan] using
    (MatrixProgram.Program.append_plan_row?
      (commitmentProgram geometry) (evaluationProgram geometry)
      (PiDECDirectPlan.commitmentPlan relation geometry)
      (PiDECDirectPlan.evaluationPlan relation geometry)
      (recompositionPlans_fit relation geometry) sourceRow
      (by simp)
      (commitmentProgram_row? relation geometry sourceRow loadedCommitment)
      (evaluationProgram_row? relation geometry sourceRow loadedK loadedA)
      global)

/-- Every row of the complete compact PiDEC program is the exact canonical
PiDEC plan row. -/
theorem matrixProgram_row?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loadedPublic : ∀ index : Fin 22680,
      sourceRow (PiDECStarts.publicInputRowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.publicProgramRow relation index)))
    (loadedCommitment : ∀ index : Fin 972,
      sourceRow (PiDECStarts.commitmentRowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.commitmentProgramRow relation index)))
    (loadedK : ∀ index : Fin 108,
      sourceRow (PiDECStarts.evalKRowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.evalKProgramRow relation index)))
    (loadedA : ∀ index : Fin 1512,
      sourceRow (PiDECStarts.evalARowStart + index.val) =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiDECOrdinaryDirectSource.evalAProgramRow relation index)))
    (global : Fin (PiDECDirectPlan.plan relation geometry).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiDECDirectPlan.plan relation geometry).forms global) := by
  simpa [matrixProgram, PiDECDirectPlan.plan] using
    (MatrixProgram.Program.append_plan_row?
      (publicProgram geometry) (recompositionProgram geometry)
      (PiDECDirectPlan.publicPlan relation geometry)
      (PiDECDirectPlan.recompositionPlan relation geometry)
      (allPlans_fit relation geometry) sourceRow
      (by simp)
      (publicProgram_row? relation geometry sourceRow loadedPublic)
      (recompositionProgram_row? relation geometry sourceRow loadedCommitment
        loadedK loadedA) global)

end NightstreamFPrime.Export.Stage1.PiDECMatrixProgram
