import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixProgram

/-!
Proves row-by-row equality between the compact PiRLC sampler ordinary matrix
block and the canonical Lean direct-row compiler. The package row accessor is
an explicit identity-checked premise.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiRLCSamplerOrdinaryMatrixSchedule
open PiRLCSamplerOrdinaryMatrixSubstitution
open PiRLCSamplerOrdinaryRetainedGeometry

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

def directForms
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 220881) :
    OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow
    (PiRLCSamplerOrdinaryDirectPlan.sourceMap geometry)
    (oneColumn geometry)
    (PiRLCSamplerOrdinaryDirectSource.programRow
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index)
    (PiRLCSamplerOrdinaryDirectSource.programRow_bounded
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index)

theorem plan_forms
    {program : Program} {logicalWidth : Nat}
    (relation : Lifecycle.ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth) (index : Fin 220881) :
    (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).forms index =
      (directForms
        (relationLogicalWidth := relationLogicalWidth)
        (relationPublicFits := relationPublicFits) geometry index
      ).meaningfulForm := by
  rfl

theorem compile_programRow?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 220881) :
    Ordinary.compileRow? (substitution program) logicalWidth
        (oneColumn geometry).val
        (PiRLCSamplerOrdinaryDirectSource.programRow
          (logicalWidth := relationLogicalWidth)
          (publicFits := relationPublicFits) index) =
      some (directForms
        (relationLogicalWidth := relationLogicalWidth)
        (relationPublicFits := relationPublicFits) geometry index) := by
  rcases substitution_agrees_on_programRow
      (relationLogicalWidth := relationLogicalWidth)
      (relationPublicFits := relationPublicFits) geometry index with
    ⟨agreesA, agreesB, agreesC⟩
  exact Ordinary.compileRow?_eq_compileRow (substitution program)
    (PiRLCSamplerOrdinaryDirectPlan.sourceMap geometry) (oneColumn geometry)
    (PiRLCSamplerOrdinaryDirectSource.programRow
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index)
    (PiRLCSamplerOrdinaryDirectSource.programRow_bounded
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index)
    agreesA agreesB agreesC

/-- The compact block returns the exact direct form when its schedule-selected
physical row is loaded from the identity-checked package. -/
theorem block_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) (index : Fin 220881)
    (sourceIndex : Nat)
    (selected : rowSchedule.index? index.val = some sourceIndex)
    (loaded : sourceRow sourceIndex =
      some (PerApplicationSourceProjection.basePackageRow program
        (PiRLCSamplerOrdinaryDirectSource.programRow
          (logicalWidth := relationLogicalWidth)
          (publicFits := relationPublicFits) index))) :
    (block geometry).row? logicalWidth sourceRow index.val =
      some (directForms
        (relationLogicalWidth := relationLogicalWidth)
        (relationPublicFits := relationPublicFits) geometry index) := by
  rcases substitution_agrees_on_programRow
      (relationLogicalWidth := relationLogicalWidth)
      (relationPublicFits := relationPublicFits) geometry index with
    ⟨agreesA, agreesB, agreesC⟩
  have projected : (block geometry).projection.row?
      (PerApplicationSourceProjection.basePackageRow program
        (PiRLCSamplerOrdinaryDirectSource.programRow
          (logicalWidth := relationLogicalWidth)
          (publicFits := relationPublicFits) index)) =
        some (PiRLCSamplerOrdinaryDirectSource.programRow
          (logicalWidth := relationLogicalWidth)
          (publicFits := relationPublicFits) index) := by
    apply PerApplicationSourceProjection.base_row
    simpa [PerApplicationPackage.basePackage, Data.circuitPackage_layout,
      Data.physicalLayout] using
      (PiRLCSamplerOrdinaryDirectSource.programRow_bounded
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index)
  exact Ordinary.Block.row?_eq_compileRow (block geometry) sourceRow
    index.val sourceIndex
    (PiRLCSamplerOrdinaryDirectSource.programRow
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index)
    selected _ loaded projected
    (PiRLCSamplerOrdinaryDirectPlan.sourceMap geometry)
    (oneColumn geometry) rfl
    (PiRLCSamplerOrdinaryDirectSource.programRow_bounded
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index)
    agreesA agreesB agreesC

/-- Every compact sampler ordinary row is the exact row of the canonical
direct Lean plan. -/
theorem matrixProgram_row?
    {program : Program} {logicalWidth : Nat}
    (relation : Lifecycle.ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (loaded : ∀ index : Fin 220881, ∀ sourceIndex,
      rowSchedule.index? index.val = some sourceIndex →
      sourceRow sourceIndex =
        some (PerApplicationSourceProjection.basePackageRow program
          (PiRLCSamplerOrdinaryDirectSource.programRow
            (logicalWidth := relationLogicalWidth)
            (publicFits := relationPublicFits) index)))
    (global : Fin
      (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).forms
        global) := by
  change Fin 220881 at global
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
    change global.val < (block geometry).rowCount
    rw [block_rowCount]
    exact global.isLt
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.ordinary (block geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos blockBound]
  change (do
    let forms ← (block geometry).row? logicalWidth sourceRow global.val
    pure forms.meaningfulForm) = _
  rw [block_row? geometry sourceRow global sourceIndex selected
    (loaded global sourceIndex selected)]
  apply congrArg some
  exact (plan_forms relation geometry global).symm

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixProgram
