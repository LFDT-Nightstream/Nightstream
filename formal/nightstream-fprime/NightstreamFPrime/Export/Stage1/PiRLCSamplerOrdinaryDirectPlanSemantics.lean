import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlan

/-!
Owns the row-count, shape-independence, and row-semantics theorems for the
direct PiRLC sampler ordinary plan. The executable resolver and plan remain in
`PiRLCSamplerOrdinaryDirectPlan`.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

@[simp] theorem plan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) :
    (plan relation geometry).rowCount = 220881 := by
  rfl

theorem plan_eq_of_same_shape
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (left right : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) :
    plan left geometry = plan right geometry := by
  rfl

/-- Direct matrix acceptance is exactly the canonical Lean-lowered sampler
ordinary row relation under the assignment-derived zero-copy source view. -/
theorem rowsZero_iff_rowsHold
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (one : assignment
      (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1) :
    (plan relation geometry).RowsZero assignment ↔
      R1CS.RowsHold (resolvedEnv geometry assignment)
        (PiRLCSamplerOrdinaryDirectSource.sourceRows
          (logicalWidth := relationLogicalWidth)
          (publicFits := relationPublicFits)) := by
  rw [← PiRLCSamplerOrdinaryDirectSource.programRows_hold_iff_rowsHold
    (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
    (resolvedEnv geometry assignment)]
  constructor
  · intro rows index
    have preserves := OrdinarySourcePlan.compileRow_preserves_local
      (sourceMap geometry)
      (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry)
      (PiRLCSamplerOrdinaryDirectSource.programRow
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index)
      (PiRLCSamplerOrdinaryDirectSource.programRow_bounded
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index)
      assignment (resolvedEnv geometry assignment) one
      (programRow_preserve geometry assignment index)
    exact (OrdinaryRow.planOfForms_residual_zero_iff
      (by norm_num [Lifecycle.cubeVariables]) (rowForms relation geometry)
      assignment
      (resolvedEnv geometry assignment) index
      (PiRLCSamplerOrdinaryDirectSource.programRow
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index)
      preserves).mp (rows index)
  · intro rows index
    have preserves := OrdinarySourcePlan.compileRow_preserves_local
      (sourceMap geometry)
      (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry)
      (PiRLCSamplerOrdinaryDirectSource.programRow
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index)
      (PiRLCSamplerOrdinaryDirectSource.programRow_bounded
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index)
      assignment (resolvedEnv geometry assignment) one
      (programRow_preserve geometry assignment index)
    exact (OrdinaryRow.planOfForms_residual_zero_iff
      (by norm_num [Lifecycle.cubeVariables]) (rowForms relation geometry)
      assignment
      (resolvedEnv geometry assignment) index
      (PiRLCSamplerOrdinaryDirectSource.programRow
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index)
      preserves).mpr (rows index)

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlan
