import NightstreamFPrime.Export.Stage1.PiDECEvaluationSparseBlock
import NightstreamFPrime.Export.Stage1.PiDECEvaluationSourceEntry
import NightstreamFPrime.Export.Stage1.PiDECCommitmentFold
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1MatrixRows

/-!
Connect the selected structural-plan row product to its existing Phi81
block-row semantics on the same complete stored child assignment. Matrix
agreement is discharged by the selected plan theorem, not supplied by callers.
No point weighting, row accumulation or expected evaluation is introduced.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECEvaluationSelectedRows

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle

private theorem rowBlock_eq_blockRowRing_of_coefficients
    {shape : Phi81Relation.Shape} (system : Phi81Relation.Structure shape)
    (form : SparseForm shape.logicalWidth)
    (matrix : Fin shape.matrixCount) (vertex : BooleanVertex shape.rowVariables)
    (same : ∀ column, form.coefficient column = system.matrices matrix vertex column)
    (assignments : Vector (Vector F shape.carrierWidth) productionGlobalParams.k)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (child : Fin productionGlobalParams.k) :
    ((PiDECEvaluationBlock.rowBlock form block.val
        (PiDECCommitmentFold.childBlocks (shape := shape) assignments block)).get child).get =
      PiRLC.blockRowRing system (assignments.get child).get matrix vertex block := by
  funext output
  rw [PiDECEvaluationSparseBlock.rowBlock_coefficient_value,
    PiDECCommitmentFold.childBlocks_value]
  change _ = sumRange baseOps ringDegree (fun index =>
    if indexLt : index < ringDegree then
      system.matrixSource.paddedMatrixEntry baseOps matrix vertex block ⟨index, indexLt⟩ *
        CarrierAction.kernelImage ⟨index, indexLt⟩
          (CarrierAction.assignmentBlock (logicalWidth := shape.logicalWidth)
            (assignments.get child).get block) output
    else 0)
  apply sumRange_congr baseOps ringDegree
  intro index indexLt
  rw [dif_pos indexLt, dif_pos indexLt]
  rw [Phi81Relation.Structure.matrixSource]
  dsimp only [Phi81Relation.Shape.sourceShape, Phi81Relation.Shape.carrierWidth]
  erw [PiDECEvaluationSourceEntry.source_paddedMatrixEntry
    shape.rowVariables 0 0 shape.matrixCount shape.logicalWidth
    system.matrices system.constraintPolynomial matrix vertex block ⟨index, indexLt⟩]
  by_cases within : block.val * ringDegree + index < shape.logicalWidth
  · rw [dif_pos within, dif_pos within, same]
  · rw [dif_neg within, dif_neg within, Fin.zero_mul]

private abbrev selectedShape :=
  PaperAlgebra.FullShape
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)

/-- At each selected active row and all fourteen ports, the computed product
is exactly the corresponding block contribution of the key-facing system.
All children retain their complete carrier block, including carried tail
coordinates. No matrix-agreement, norm, opening or split premise is required. -/
theorem rowBlock_eq_blockRowRing
    (row : Fin (PerApplicationFixedPoint.structuralPlan
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits).rowCount)
    (matrix : Fin Spec.ProductionRelation.matrixCount)
    (assignments : Vector (Vector F selectedShape.carrierWidth) productionGlobalParams.k)
    (block : Fin (Phi81ColumnLayout.blockCount selectedShape.carrierWidth))
    (child : Fin productionGlobalParams.k) :
    ((PiDECEvaluationBlock.rowBlock
        ((PerApplicationFixedPoint.structuralPlan
          Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits).portForm row matrix)
        block.val (PiDECCommitmentFold.childBlocks
          (shape := selectedShape) assignments block)).get child).get =
      PiRLC.blockRowRing
        (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits).system
        (assignments.get child).get matrix
        ((PerApplicationFixedPoint.structuralPlan
          Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits).rowLayout.toVertex row) block := by
  have matrixProjection {logicalWidth : Nat}
      {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth logicalWidth}
      (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
      relation.system.matrices = relation.matrices := rfl
  refine rowBlock_eq_blockRowRing_of_coefficients
    (shape := selectedShape)
    (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits).system
    ((PerApplicationFixedPoint.structuralPlan
      Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits).portForm row matrix)
    matrix ((PerApplicationFixedPoint.structuralPlan
      Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits).rowLayout.toVertex row)
    ?_ assignments block child
  intro column
  rw [matrixProjection (PerApplicationFixedPoint.relation
    Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits)]
  exact Poseidon2HashChainV1MatrixRows.allPort_coefficient_eq_logicalRelation_matrix
    row matrix column

end NightstreamFPrime.Export.Stage1.PiDECEvaluationSelectedRows
