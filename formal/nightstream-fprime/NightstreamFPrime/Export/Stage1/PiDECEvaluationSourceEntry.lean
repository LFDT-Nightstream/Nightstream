import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81MatrixSource

/-!
Reduce one original matrix entry at a complete Phi81 carrier block/lane to
its logical-width bound check. The completed matrix suffix is zero; this
statement imposes no condition on the carried assignment suffix.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECEvaluationSourceEntry

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- Every complete carrier block/lane first encodes to its actual carrier
column, then reads the original matrix exactly when that column is logical.
No divisibility, selected-plan, matrix-equality or assignment premise is used. -/
theorem source_paddedMatrixEntry
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount → BooleanMatrix F cubeVariables logicalWidth)
    (polynomial : CCSResidualTable.ConstraintPolynomial F matrixCount)
    (matrix : Fin matrixCount) (vertex : BooleanVertex cubeVariables)
    (block : Fin (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth)))
    (lane : Fin ringDegree) :
    (Phi81MatrixSource.source cubeVariables freshCount runningCount matrixCount
        logicalWidth matrices polynomial).paddedMatrixEntry
        baseOps matrix vertex block lane =
      if within : block.val * ringDegree + lane.val < logicalWidth then
        matrices matrix vertex ⟨block.val * ringDegree + lane.val, within⟩
      else 0 := by
  simp only [MatrixCoefficientSource.MatrixSource.paddedMatrixEntry,
    MatrixCoefficientSource.MatrixSource.paddedEntry, Phi81MatrixSource.source]
  rw [Phi81CarrierLayout.layout_encode?_isSome]
  by_cases within : block.val * ringDegree + lane.val < logicalWidth
  · simp only [Phi81CarrierLayout.extendMatrix, Phi81CarrierLayout.logicalColumn?,
      Phi81ColumnLayout.flatIndex, dif_pos within]
  · simp only [Phi81CarrierLayout.extendMatrix, Phi81CarrierLayout.logicalColumn?,
      Phi81ColumnLayout.flatIndex, dif_neg within]

end NightstreamFPrime.Export.Stage1.PiDECEvaluationSourceEntry
