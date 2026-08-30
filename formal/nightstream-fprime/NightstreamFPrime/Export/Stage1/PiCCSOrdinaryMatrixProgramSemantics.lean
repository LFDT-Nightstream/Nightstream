import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryMatrixProgram

/-!
Proves row-by-row equality between the compact PiCCS ordinary matrix program
and the canonical direct 14-matrix plan. The package row accessor is an
explicit identity-checked premise.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSOrdinaryMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiCCSOrdinaryRetainedGeometry

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

def directForms
    {program : Program} {logicalWidth : Nat}
    (relation : Lifecycle.ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) (index : Fin 811669) :
    OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (PiCCSOrdinaryDirectPlan.sourceMap geometry)
    (oneColumn geometry) (PiCCSOrdinaryDirectSource.programRow relation index)
    (PiCCSOrdinaryDirectSource.programRow_bounded relation index)

theorem plan_forms
    {program : Program} {logicalWidth : Nat}
    (relation : Lifecycle.ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) (index : Fin 811669) :
    (PiCCSOrdinaryDirectPlan.plan relation geometry).forms index =
      (directForms relation geometry index).meaningfulForm := by
  rfl

theorem block_row?
    {program : Program} {logicalWidth : Nat}
    (relation : Lifecycle.ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) (index : Fin 811669)
    (sourceIndex : Nat)
    (selected : rowSchedule.index? index.val = some sourceIndex)
    (loaded : sourceRow sourceIndex =
      some (PerApplicationSourceProjection.basePackageRow program
        (PiCCSOrdinaryDirectSource.programRow relation index))) :
    (block geometry).row? logicalWidth sourceRow index.val =
      some (directForms relation geometry index) := by
  rcases substitution_agrees_on_programRow relation geometry index with
    ⟨agreesA, agreesB, agreesC⟩
  have projected : (block geometry).projection.row?
      (PerApplicationSourceProjection.basePackageRow program
        (PiCCSOrdinaryDirectSource.programRow relation index)) =
        some (PiCCSOrdinaryDirectSource.programRow relation index) := by
    apply PerApplicationSourceProjection.base_row
    simpa [PerApplicationPackage.basePackage, Data.circuitPackage_layout,
      Data.physicalLayout] using
      (PiCCSOrdinaryDirectSource.programRow_bounded relation index)
  exact Ordinary.Block.row?_eq_compileRow (block geometry) sourceRow
    index.val sourceIndex (PiCCSOrdinaryDirectSource.programRow relation index)
    selected _ loaded projected (PiCCSOrdinaryDirectPlan.sourceMap geometry)
    (oneColumn geometry) rfl
    (PiCCSOrdinaryDirectSource.programRow_bounded relation index)
    agreesA agreesB agreesC

/-- Every compact PiCCS ordinary row is the exact row of the canonical direct
Lean plan. -/
theorem matrixProgram_row?
    {program : Program} {logicalWidth : Nat}
    (relation : Lifecycle.ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loaded : ∀ index : Fin 811669, ∀ sourceIndex,
      rowSchedule.index? index.val = some sourceIndex →
      sourceRow sourceIndex =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiCCSOrdinaryDirectSource.programRow relation index)))
    (global : Fin (PiCCSOrdinaryDirectPlan.plan relation geometry).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiCCSOrdinaryDirectPlan.plan relation geometry).forms global) := by
  change Fin 811669 at global
  have scheduleBound : global.val < rowSchedule.indices.length := by
    rw [IndexSchedule.indices_length, rowSchedule_count]
    exact global.isLt
  let scheduleIndex : Fin rowSchedule.indices.length :=
    ⟨global.val, scheduleBound⟩
  let sourceIndex := rowSchedule.indices.get scheduleIndex
  have selected : rowSchedule.index? global.val = some sourceIndex := by
    rw [IndexSchedule.index?_eq_getElem?,
      List.getElem?_eq_getElem scheduleBound]
    rfl
  have blockBound : global.val <
      (MatrixProgram.Block.ordinary (block geometry)).rowCount := by
    change global.val < 811669
    exact global.isLt
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.ordinary (block geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos blockBound]
  change (do
    let forms ← (block geometry).row? logicalWidth sourceRow global.val
    pure forms.meaningfulForm) = _
  rw [block_row? relation geometry sourceRow global sourceIndex selected
    (loaded global sourceIndex selected)]
  apply congrArg some
  exact (plan_forms relation geometry global).symm

end NightstreamFPrime.Export.Stage1.PiCCSOrdinaryMatrixProgram
