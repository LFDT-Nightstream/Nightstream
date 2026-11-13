import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RingCarrierLayout
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RingCoefficientKernel

/-! SuperNeo Section 5; Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RingMatrixSource

open NightstreamFPrime.Spec
open PaperLinearAlgebra
open MatrixCoefficientSource

@[implicit_reducible]
def ringShape
    (cubeVariables freshCount runningCount matrixCount : Nat) : Shape where
  cubeVariables := cubeVariables
  freshCount := freshCount
  runningCount := runningCount
  matrixCount := matrixCount
  coefficientCount := ringDegree

def source
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount) :
    MatrixSource F
      (ringShape cubeVariables freshCount runningCount matrixCount)
      (RingCarrierLayout.carrierWidth logicalWidth)
      (ConcreteRingColumnLayout.blockCount
        (RingCarrierLayout.carrierWidth logicalWidth)) where
  columnLayout := RingCarrierLayout.layout logicalWidth
  matrices := fun matrix =>
    RingCarrierLayout.extendMatrix 0 (matrices matrix)
  constraintPolynomial := constraintPolynomial
  kernel := RingCoefficientKernel.ringKernel

theorem ringShape_coefficientCount
    (cubeVariables freshCount runningCount matrixCount : Nat) :
    (ringShape cubeVariables freshCount runningCount matrixCount).coefficientCount =
      ringDegree := by
  rfl

theorem source_kernel_eq
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount) :
    (source cubeVariables freshCount runningCount matrixCount logicalWidth
      matrices constraintPolynomial).kernel =
      RingCoefficientKernel.ringKernel := by
  rfl

theorem source_matrix_embedLogical
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount)
    (matrix : Fin matrixCount)
    (vertex : BooleanVertex cubeVariables)
    (column : Fin logicalWidth) :
    (source cubeVariables freshCount runningCount matrixCount logicalWidth
        matrices constraintPolynomial).matrices matrix vertex
        (RingCarrierLayout.embedLogical column) =
      matrices matrix vertex column := by
  exact RingCarrierLayout.extendMatrix_embedLogical 0
    (matrices matrix) vertex column

theorem source_matrix_tail_zero
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount)
    (matrix : Fin matrixCount)
    (vertex : BooleanVertex cubeVariables)
    (column : Fin (RingCarrierLayout.carrierWidth logicalWidth))
    (tail : logicalWidth <= column.val) :
    (source cubeVariables freshCount runningCount matrixCount logicalWidth
        matrices constraintPolynomial).matrices matrix vertex column = 0 := by
  exact RingCarrierLayout.extendMatrix_tail_zero 0
    (matrices matrix) vertex column tail

theorem coefficientMatrix_constant_apply
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount)
    (matrix : Fin matrixCount)
    (vertex : BooleanVertex cubeVariables)
    (column : Fin (RingCarrierLayout.carrierWidth logicalWidth)) :
    let matrixSource :=
      source cubeVariables freshCount runningCount matrixCount logicalWidth
        matrices constraintPolynomial
    matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix
        matrixSource.kernel.constant vertex column =
      matrixSource.matrices matrix vertex column := by
  dsimp only
  exact MatrixSource.coefficientMatrix_constant_apply
    ConcreteCarrier.baseOps ConcreteCarrier.baseLaws
    (source cubeVariables freshCount runningCount matrixCount logicalWidth
      matrices constraintPolynomial)
    RingCoefficientKernel.ringConstantTermLaw matrix vertex column

theorem coefficientMatrix_constant_embedLogical
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount)
    (matrix : Fin matrixCount)
    (vertex : BooleanVertex cubeVariables)
    (column : Fin logicalWidth) :
    let matrixSource :=
      source cubeVariables freshCount runningCount matrixCount logicalWidth
        matrices constraintPolynomial
    matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix
        matrixSource.kernel.constant vertex
        (RingCarrierLayout.embedLogical column) =
      matrices matrix vertex column := by
  dsimp only
  rw [coefficientMatrix_constant_apply]
  exact source_matrix_embedLogical cubeVariables freshCount runningCount
    matrixCount logicalWidth matrices constraintPolynomial matrix vertex column

theorem coefficientMatrix_constant_tail_zero
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount)
    (matrix : Fin matrixCount)
    (vertex : BooleanVertex cubeVariables)
    (column : Fin (RingCarrierLayout.carrierWidth logicalWidth))
    (tail : logicalWidth <= column.val) :
    let matrixSource :=
      source cubeVariables freshCount runningCount matrixCount logicalWidth
        matrices constraintPolynomial
    matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix
        matrixSource.kernel.constant vertex column = 0 := by
  dsimp only
  rw [coefficientMatrix_constant_apply]
  exact source_matrix_tail_zero cubeVariables freshCount runningCount
    matrixCount logicalWidth matrices constraintPolynomial matrix vertex column
    tail

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RingMatrixSource
