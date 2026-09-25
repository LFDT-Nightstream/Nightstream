import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/Phi81MatrixSource.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Phi81 single-matrix source over the complete coefficient carrier.

Protocol: SuperNeo coefficient embedding (Section 5) and `Pi_CCS`
(Section 7.3 / Appendix D.4).
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

Authority boundary: callers provide only the original-width CCS matrices and
constraint polynomial. The completed matrix suffix and all coefficient images
are definitions. A carried assignment may use every completed coordinate, but
there is no separately settable coefficient-matrix view.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| coefficient embedding | shape | `d` | `phi81Shape.coefficientCount = 54` |
| CCS structure | carrier completion | logical prefix | every original matrix entry is preserved |
| CCS structure | carrier completion | completed suffix | every new matrix entry is canonical zero |
| carried CE | coefficient expansion | output / row / assignment lane | `source.kernel = phi81Kernel` |
| carried CE | constant coefficient | completed matrix | coefficient zero equals the sole completed CCS matrix |
| assurance | production refinement | Rust cache / R1CS | explicitly not closed here |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81MatrixSource

open NightstreamFPrime.Spec
open PaperLinearAlgebra
open MatrixCoefficientSource

/-- Paper dimensions specialized to the concrete Phi81 coefficient count. -/
def phi81Shape
    (cubeVariables freshCount runningCount matrixCount : Nat) : Shape where
  cubeVariables := cubeVariables
  freshCount := freshCount
  runningCount := runningCount
  matrixCount := matrixCount
  coefficientCount := ringDegree

/-- The sole original matrix family, completed to full 54-lane blocks and
paired with the independently defined Phi81 coefficient kernel. -/
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

/-- The concrete source has exactly 54 coefficient lanes. -/
theorem phi81Shape_coefficientCount
    (cubeVariables freshCount runningCount matrixCount : Nat) :
    (phi81Shape cubeVariables freshCount runningCount matrixCount).coefficientCount =
      ringDegree := by
  rfl

/-- The source stores the independent Phi81 kernel, not caller-provided
coefficient matrices. -/
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

/-- Every original-width matrix entry survives carrier completion exactly. -/
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

/-- Matrix entries created only by carrier completion are canonical zero. -/
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

/-- The constant coefficient matrix is exactly the completed sole matrix at
every carrier coordinate, including the zero-derived matrix suffix. -/
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

/-- On the original prefix, the constant carried coefficient is exactly the
original CCS matrix entry. -/
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

/-- On the completed matrix suffix, the constant carried coefficient is zero.
This does not constrain the carried assignment suffix. -/
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
