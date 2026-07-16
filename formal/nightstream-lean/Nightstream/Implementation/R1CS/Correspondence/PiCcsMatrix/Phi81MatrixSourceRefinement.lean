import Nightstream.Implementation.R1CS.Correspondence.PiCcsMatrix.Phi81BarMatrixRefinement
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81MatrixSource

/-!
Runtime-bar refinement of the complete Phi81 single-matrix source.

Protocol: SuperNeo coefficient embedding (Section 5) and `Pi_CCS`
(Section 7.3 / Appendix D.4).
Phase: Rust-exported bar matrix to complete carried coefficient matrices.
Constraint family: completed matrix entry / kernel weight / derived coefficient
matrix leaf.

Owns: a matrix source using the Rust-exported bar kernel; exact agreement of
its completed sole matrix with the independent source; and cell-by-cell
equality of every derived coefficient matrix with the paper-derived Phi81
source.

Does not own: conformance of Rust `superneo_bar_block`,
`build_superneo_ring_forms`, cache construction, `Mat` traversal, mixed CE
assignment construction, R1CS lowering, row removal, or constraint counts.

Emits constraints: no.

Assurance tier: artifact-checked model correspondence. This closes the bar
matrix value used by the semantic source, conditional on the Rust artifact
drift test. It does not yet prove that production loops instantiate this
source.

Authority boundary: callers still provide only the original CCS matrix and
constraint polynomial. The carrier suffix, coefficient matrices, and runtime
kernel are derived. Equality with the independent source is proved per leaf;
the generated artifact is never used to define the mathematical target.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| CCS structure | carrier completion | logical prefix / zero suffix | runtime-bar and independent sources share the same sole completed matrix |
| coefficient embedding | kernel leaf | output / row / assignment | runtime weight equals independent weight |
| carried CE | coefficient matrix | matrix / output / Boolean row / carrier column | every derived leaf is equal |
| open implementation | Rust loop/cache | block traversal / storage order | explicitly not closed here |
| open R1CS | lowering | row ownership / constraints | explicitly not closed here |
-/

namespace Nightstream.Implementation.R1CS.Phi81MatrixSourceRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra
open MatrixCoefficientSource

set_option maxRecDepth 100000

/-- Complete source obtained by replacing the independent kernel with the
kernel derived from the Rust-exported bar matrix. -/
def runtimeSource
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount) :
    MatrixSource F
      (Phi81MatrixSource.phi81Shape cubeVariables freshCount runningCount
        matrixCount)
      (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount
        (Phi81CarrierLayout.carrierWidth logicalWidth)) where
  columnLayout := Phi81CarrierLayout.layout logicalWidth
  matrices := fun matrix =>
    Phi81CarrierLayout.extendMatrix 0 (matrices matrix)
  constraintPolynomial := constraintPolynomial
  kernel := Phi81BarMatrixRefinement.runtimeKernel

/-- Both sources read the same sole completed matrix entry. -/
theorem runtimeSource_matrix_eq_semantic
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount)
    (matrix : Fin matrixCount)
    (vertex : BooleanVertex cubeVariables)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    (runtimeSource cubeVariables freshCount runningCount matrixCount
        logicalWidth matrices constraintPolynomial).matrices matrix vertex
        column =
      (Phi81MatrixSource.source cubeVariables freshCount runningCount
        matrixCount logicalWidth matrices constraintPolynomial).matrices matrix
        vertex column := by
  rfl

/-- Padded matrix reads agree before applying either coefficient kernel. -/
theorem runtimeSource_paddedMatrixEntry_eq_semantic
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount)
    (matrix : Fin matrixCount)
    (vertex : BooleanVertex cubeVariables)
    (block : Fin (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth)))
    (coefficient : Fin ringDegree) :
    (runtimeSource cubeVariables freshCount runningCount matrixCount
        logicalWidth matrices constraintPolynomial).paddedMatrixEntry
        ConcreteCarrier.baseOps matrix vertex block coefficient =
      (Phi81MatrixSource.source cubeVariables freshCount runningCount
        matrixCount logicalWidth matrices constraintPolynomial).paddedMatrixEntry
        ConcreteCarrier.baseOps matrix vertex block coefficient := by
  unfold MatrixSource.paddedMatrixEntry
  rfl

/-- Artifact-to-semantics refinement for every coefficient-expanded matrix
leaf. This is the exact mathematical value that a future Rust-loop trace must
instantiate before any R1CS correspondence can be claimed. -/
theorem runtimeSource_coefficientMatrix_eq_semantic
    (cubeVariables freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount ->
      BooleanMatrix F cubeVariables logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F matrixCount)
    (matrix : Fin matrixCount)
    (output : Fin ringDegree)
    (vertex : BooleanVertex cubeVariables)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    (runtimeSource cubeVariables freshCount runningCount matrixCount
        logicalWidth matrices constraintPolynomial).coefficientMatrix
        ConcreteCarrier.baseOps matrix output vertex column =
      (Phi81MatrixSource.source cubeVariables freshCount runningCount
        matrixCount logicalWidth matrices constraintPolynomial).coefficientMatrix
        ConcreteCarrier.baseOps matrix output vertex column := by
  unfold MatrixSource.coefficientMatrix
  apply sumRange_congr
  intro rowIndex rowLt
  rw [dif_pos rowLt, dif_pos rowLt]
  let row : Fin ringDegree := ⟨rowIndex, rowLt⟩
  let packed := (Phi81CarrierLayout.layout logicalWidth).decode column
  change
    ConcreteCarrier.baseOps.mul
        ((runtimeSource cubeVariables freshCount runningCount matrixCount
          logicalWidth matrices constraintPolynomial).paddedMatrixEntry
          ConcreteCarrier.baseOps matrix vertex
          packed.1 row)
        (Phi81BarMatrixRefinement.runtimeKernel.weight output row
          packed.2) =
      ConcreteCarrier.baseOps.mul
        ((Phi81MatrixSource.source cubeVariables freshCount runningCount
          matrixCount logicalWidth matrices constraintPolynomial).paddedMatrixEntry
          ConcreteCarrier.baseOps matrix vertex
          packed.1 row)
        (Phi81CoefficientKernel.phi81Kernel.weight output row
          packed.2)
  rw [runtimeSource_paddedMatrixEntry_eq_semantic]
  rw [Phi81BarMatrixRefinement.runtimeKernel_weight_eq_phi81Kernel]

end Nightstream.Implementation.R1CS.Phi81MatrixSourceRefinement
