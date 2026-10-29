import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryMatrixProgram
import NightstreamFPrime.Layout.PiCCS.v1_2.Assumptions

namespace NightstreamFPrime.Export.Stage1.PiCCSAssignmentSoundness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

def decodedEnv
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Env :=
  SourceCompiler.sourceEnv fun column =>
    ((PiCCSOrdinaryDirectPlan.sourceMap geometry).form column).eval assignment

theorem decodedEnv_preserves
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (PiCCSOrdinaryDirectPlan.sourceMap geometry).Preserves assignment
      (decodedEnv geometry assignment) := by
  intro column
  exact (SourceCompiler.sourceEnv_at
    (fun column => ((PiCCSOrdinaryDirectPlan.sourceMap geometry).form column).eval
      assignment) column).symm

theorem rowsZero_implies_sourceRows
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiCCSOrdinaryDirectPlan.plan relation geometry).RowsZero assignment) :
    R1CS.RowsHold (decodedEnv geometry assignment)
      (PiCCSOrdinaryDirectSource.sourceRows relationLogicalWidth relationPublicFits) := by
  rw [← PiCCSOrdinaryDirectSource.programRows_hold_iff_rowsHold relation
    (decodedEnv geometry assignment)]
  intro index
  have preserves := SourceCompiler.compileRow_preserves
    (PiCCSOrdinaryDirectPlan.sourceMap geometry)
    (PiCCSOrdinaryRetainedGeometry.oneColumn geometry)
    (PiCCSOrdinaryDirectSource.programRow relation index)
    (PiCCSOrdinaryDirectSource.programRow_bounded relation index)
    assignment (decodedEnv geometry assignment) one
    (decodedEnv_preserves geometry assignment)
  exact (OrdinaryRow.planOfForms_residual_zero_iff
    (by norm_num [Lifecycle.cubeVariables])
    (PiCCSOrdinaryDirectPlan.rowForms relation geometry)
    assignment (decodedEnv geometry assignment) index
    (PiCCSOrdinaryDirectSource.programRow relation index) preserves).mp (rows index)

theorem decodedEnv_location
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (location : PiCCSOrdinaryDirectPlan.Location) :
    decodedEnv geometry assignment
        (Spartan.sourceToSpartan location.sourceColumn) =
      (location.form geometry).eval assignment := by
  let column : Fin Spartan.spartanColumnCount :=
    ⟨Spartan.sourceToSpartan location.sourceColumn,
      Spartan.sourceToSpartan_lt _ location.sourceColumn_lt⟩
  have support : PiCCSOrdinarySourceSupport.Target column.val :=
    ⟨location.sourceColumn, location.sourceSupport, rfl⟩
  have mapped := PiCCSOrdinaryMatrixProgram.substitution_agrees_on_target
    geometry column support
  have selected := PiCCSOrdinaryMatrixProgram.substitution_location_form?
    geometry location
  have sameForms :
      (PiCCSOrdinaryDirectPlan.sourceMap geometry).form column =
        location.form geometry := by
    exact Option.some.inj (mapped.symm.trans selected)
  change SourceCompiler.sourceEnv
      (fun c => ((PiCCSOrdinaryDirectPlan.sourceMap geometry).form c).eval assignment)
      column.val = _
  rw [SourceCompiler.sourceEnv_at, sameForms]

theorem rowsZero_implies_arithmeticSpecs
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiCCSOrdinaryDirectPlan.plan relation geometry).RowsZero assignment) :
    PiCCSArithmetic.ArithmeticSpecs relationLogicalWidth relationPublicFits relation
      (decodedEnv geometry assignment) := by
  have packets := PiCCSArithmetic.arithmeticRows_imply_packetHolds
    relationLogicalWidth relationPublicFits (decodedEnv geometry assignment)
    (rowsZero_implies_sourceRows relation geometry assignment one rows)
  have assumptions := NightstreamFPrime.Layout.PiCCS.v1_2.Assumptions.production
    relation (PiCCSInvocations.parentInterface relationLogicalWidth relationPublicFits)
    PiCCSInputs.phaseOffset
    (PiCCSInputs.externalInputsLinear relationLogicalWidth relationPublicFits)
    (Spartan.pullback (decodedEnv geometry assignment))
  exact PiCCSArithmetic.packetHolds_imply_arithmeticSpecs
    relationLogicalWidth relationPublicFits relation (decodedEnv geometry assignment)
    assumptions packets

end NightstreamFPrime.Export.Stage1.PiCCSAssignmentSoundness
