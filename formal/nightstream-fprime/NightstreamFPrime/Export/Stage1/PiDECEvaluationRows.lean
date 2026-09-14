import NightstreamFPrime.Export.Stage1.PiDECEvaluationBlockSupport
import NightstreamFPrime.Export.Stage1.PiDECEvaluationSelectedRows
import NightstreamFPrime.Export.Stage1.PiDECEvaluationPadBlock

/-!
Stored sparse accumulation of complete PiDEC matrix and Pad rows. Each
coefficient projects to the existing full sumRange. No point weighting, expected
row, IO, or selected setup/plan expansion is introduced.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECEvaluationRows

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle

/-- Visit the blocks used by a sparse row and return the selected child sum.
The block support is computed from the original entries, without changing
repeated entries or requiring a support certificate from the caller. -/
def row {shape : Phi81Relation.Shape} {columns : Nat}
    (form : SparseForm columns)
    (assignments : Vector (Vector F shape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) : StoredRing :=
  (PiDECEvaluationBlockSupport.kernel form fun block =>
    if live : block < Phi81ColumnLayout.blockCount shape.carrierWidth then
      PiDECCommitmentFold.childBlocks (shape := shape) assignments ⟨block, live⟩
    else Vector.replicate productionGlobalParams.k PiDECCommitmentFold.zero).get child

private theorem row_sumRange {shape : Phi81Relation.Shape} {columns : Nat}
    (form : SparseForm columns) (fits : columns ≤ shape.carrierWidth)
    (assignments : Vector (Vector F shape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (output : Fin ringDegree) :
    (row (shape := shape) form assignments child).get output =
      sumRange baseOps (Phi81ColumnLayout.blockCount shape.carrierWidth) (fun block =>
        if live : block < Phi81ColumnLayout.blockCount shape.carrierWidth then
          ((PiDECEvaluationBlock.rowBlock form block
            (PiDECCommitmentFold.childBlocks (shape := shape) assignments
              ⟨block, live⟩)).get child).get output
        else 0) := by
  have covered : columns ≤
      Phi81ColumnLayout.blockCount shape.carrierWidth * ringDegree := by
    calc
      columns ≤ shape.carrierWidth := fits
      _ = _ := by
        change Phi81CarrierLayout.carrierWidth shape.logicalWidth =
          Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth shape.logicalWidth) * ringDegree
        rw [Phi81CarrierLayout.blockCount_carrierWidth]
        exact Phi81CarrierLayout.carrierWidth_eq shape.logicalWidth
  rw [row, PiDECEvaluationBlockSupport.kernel_eq_fullBlockSum form covered]
  apply sumRange_congr baseOps _
  intro block live
  simp only [dif_pos live]

/-- An empty form computes zero without reading any child block. -/
theorem row_empty_value {shape : Phi81Relation.Shape} {columns : Nat}
    (assignments : Vector (Vector F shape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) :
    (row (shape := shape) (SparseForm.empty : SparseForm columns) assignments child).get =
      ringFZero := by
  simp only [row, PiDECEvaluationBlockSupport.kernel,
    PiDECEvaluationBlockSupport.blockIndices, SparseForm.empty,
    List.map_nil, List.dedup_nil, List.foldl_nil]
  funext output
  change ((Vector.replicate productionGlobalParams.k PiDECCommitmentFold.zero)[child.val]).get output = 0
  rw [Vector.getElem_replicate]
  exact congrFun PiDECCommitmentFold.zero_value output

private abbrev selectedShape :=
  PaperAlgebra.FullShape
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)

/-- Every selected active matrix row, including all fourteen ports, is the
existing complete derived-matrix row. The selected block theorem discharges
matrix correspondence; no expected-row or matrix-agreement premise remains. -/
theorem matrixRow_value
    (rowIndex : Fin (PerApplicationFixedPoint.structuralPlan
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits).rowCount)
    (matrix : Fin Spec.ProductionRelation.matrixCount)
    (assignments : Vector (Vector F selectedShape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) :
    (row (shape := selectedShape)
      ((PerApplicationFixedPoint.structuralPlan
        Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits).portForm
          rowIndex matrix) assignments child).get =
      PiRLC.rowRing
        (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits).system
        (assignments.get child).get matrix
        ((PerApplicationFixedPoint.structuralPlan
          Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits).rowLayout.toVertex rowIndex) := by
  funext output
  rw [row_sumRange (shape := selectedShape)
      ((PerApplicationFixedPoint.structuralPlan Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits).portForm rowIndex matrix)
      (Phi81CarrierLayout.logicalWidth_le_carrierWidth
        (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application))
      assignments child output,
    PiRLC.rowRing_eq_blockSum]
  change _ = sumRange baseOps
    (Phi81ColumnLayout.blockCount selectedShape.carrierWidth) (fun block =>
      if live : block < Phi81ColumnLayout.blockCount selectedShape.carrierWidth then
        PiRLC.blockRowRing
          (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
            Poseidon2HashChainV1Package.fits).system
          (assignments.get child).get matrix
          ((PerApplicationFixedPoint.structuralPlan
            Poseidon2HashChainV1Package.application
            Poseidon2HashChainV1Package.fits).rowLayout.toVertex rowIndex)
          ⟨block, live⟩ output
      else 0)
  apply sumRange_congr baseOps _
  intro block live
  simp only [dif_pos live]
  exact congrArg (fun value : RingF => value output)
    (PiDECEvaluationSelectedRows.rowBlock_eq_blockRowRing
      rowIndex matrix assignments ⟨block, live⟩ child)

/-- The complete Pad row uses the authoritative full-carrier identity
layout. This generic statement includes carrier tails and every Boolean row;
it requires no norm, split, opening, or expected-value premise. -/
theorem padRow_value {shape : Phi81Relation.Shape}
    (system : Phi81Relation.Structure shape)
    (layout : ColumnLayout shape.rowVariables shape.carrierWidth)
    (vertex : BooleanVertex shape.rowVariables)
    (assignments : Vector (Vector F shape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) :
    (row (shape := shape) (PiDECEvaluationPadBlock.form layout vertex)
      assignments child).get =
      PiRLC.ExplicitMatrix.rowRing system
        (layout.paddedIdentityEntry (0 : F) 1) (assignments.get child).get vertex := by
  funext output
  rw [row_sumRange (shape := shape) _ (Nat.le_refl _) assignments child output,
    PiRLC.ExplicitMatrix.rowRing_eq_sumRange]
  apply sumRange_congr baseOps _
  intro block live
  simp only [dif_pos live]
  have blockValue := PiDECEvaluationPadBlock.rowBlock_paddedEntry
    shape.rowVariables 0 0 shape.matrixCount shape.logicalWidth
    system.matrices system.constraintPolynomial layout vertex ⟨block, live⟩
    (PiDECCommitmentFold.childBlocks (shape := shape) assignments ⟨block, live⟩)
    child output
  rw [PiDECCommitmentFold.childBlocks_value] at blockValue
  exact blockValue

end NightstreamFPrime.Export.Stage1.PiDECEvaluationRows
