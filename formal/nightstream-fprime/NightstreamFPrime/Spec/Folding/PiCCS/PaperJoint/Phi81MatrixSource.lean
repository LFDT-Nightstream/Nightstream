import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

/-!
Phi81 single-matrix source over the complete coefficient carrier.

Protocol: SuperNeo coefficient embedding (Section 5) and `Pi_CCS`
(SuperNeo v1.2 Section 7.3 / Appendix B.2).
Phase: original CCS matrix to all carried CE coefficient matrices.
Constraint family: logical matrix prefix / completed matrix suffix / Phi81
coefficient image.

Owns: the paper shape with `d = 54`; construction of one authoritative matrix
source over the complete carrier; zero extension of the original CCS matrix;
derivation of every carried coefficient matrix through the independent Phi81
kernel; and exact constant-coefficient agreement with the completed CCS
matrix.

Does not own: fresh or carried assignments, proof that production Rust builds
this source, the runtime bar-matrix artifact, matrix-cache loops, transcript,
SumCheck, R1CS lowering, row removal, or constraint counts.

Emits constraints: no.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81MatrixSource

open NightstreamFPrime.Spec
open PaperLinearAlgebra
open MatrixCoefficientSource

def phi81Shape
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
      (phi81Shape cubeVariables freshCount runningCount matrixCount)
      (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount
        (Phi81CarrierLayout.carrierWidth logicalWidth)) where
  columnLayout := Phi81CarrierLayout.layout logicalWidth
  matrices := fun matrix =>
    Phi81CarrierLayout.extendMatrix 0 (matrices matrix)
  constraintPolynomial := constraintPolynomial
  kernel := Phi81CoefficientKernel.phi81Kernel

theorem phi81Shape_coefficientCount
    (cubeVariables freshCount runningCount matrixCount : Nat) :
    (phi81Shape cubeVariables freshCount runningCount matrixCount).coefficientCount =
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
      Phi81CoefficientKernel.phi81Kernel := by
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
        (Phi81CarrierLayout.embedLogical column) =
      matrices matrix vertex column := by
  exact Phi81CarrierLayout.extendMatrix_embedLogical 0
    (matrices matrix) vertex column

theorem source_matrix_tail_zero
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount)
    (matrix : Fin matrixCount)
    (vertex : BooleanVertex cubeVariables)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth))
    (tail : logicalWidth <= column.val) :
    (source cubeVariables freshCount runningCount matrixCount logicalWidth
        matrices constraintPolynomial).matrices matrix vertex column = 0 := by
  exact Phi81CarrierLayout.extendMatrix_tail_zero 0
    (matrices matrix) vertex column tail

theorem coefficientMatrix_constant_apply
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount)
    (matrix : Fin matrixCount)
    (vertex : BooleanVertex cubeVariables)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)) :
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
    Phi81CoefficientKernel.phi81ConstantTermLaw matrix vertex column

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
        (Phi81CarrierLayout.embedLogical column) =
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
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth))
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

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81MatrixSource
