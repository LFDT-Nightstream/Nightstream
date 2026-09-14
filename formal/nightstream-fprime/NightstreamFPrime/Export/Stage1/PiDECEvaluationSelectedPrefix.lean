import NightstreamFPrime.Export.Stage1.PiDECEvaluationRows
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding

/-!
Numeric row selectors for the selected matrix and full-carrier Pad prefixes.
Their Boolean-vertex values are the existing row semantics; the remaining
numeric suffixes are zero. No point or expected evaluation is an input.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECEvaluationSelectedPrefix

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle

private theorem system_matrices {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    relation.system.matrices = relation.matrices := rfl

private theorem form_of_some {arity columns : Nat}
    (layout : UnifiedSources.ColumnLayout arity columns)
    (vertex : BooleanVertex arity) (column : Fin columns)
    (decoded : layout.toColumn? vertex = some column) :
    PiDECEvaluationPadBlock.form layout vertex = SparseForm.singleton column 1 := by
  simp only [PiDECEvaluationPadBlock.form, decoded]

private theorem form_of_none {arity columns : Nat}
    (layout : UnifiedSources.ColumnLayout arity columns)
    (vertex : BooleanVertex arity) (decoded : layout.toColumn? vertex = none) :
    PiDECEvaluationPadBlock.form layout vertex = SparseForm.empty := by
  simp only [PiDECEvaluationPadBlock.form, decoded]

private theorem zero_row_of_coefficients {shape : Phi81Relation.Shape}
    (system : Phi81Relation.Structure shape) (assignment : Phi81Relation.Assignment shape)
    (matrix : Fin shape.matrixCount) (vertex : BooleanVertex shape.rowVariables)
    (zero : ∀ column, system.matrices matrix vertex column = 0) :
    PiRLC.rowRing system assignment matrix vertex = ringFZero := by
  apply PiRLC.rowRing_eq_zero_of_padded_row_zero
  intro block lane
  rw [Phi81Relation.Structure.matrixSource]
  dsimp only [Phi81Relation.Shape.sourceShape, Phi81Relation.Shape.carrierWidth]
  erw [PiDECEvaluationSourceEntry.source_paddedMatrixEntry]
  by_cases within : block.val * ringDegree + lane.val < shape.logicalWidth
  · rw [dif_pos within, zero]
  · rw [dif_neg within]

private abbrev selectedShape :=
  PaperAlgebra.FullShape
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)

private abbrev selectedPlan := PerApplicationFixedPoint.structuralPlan
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits

private abbrev selectedRelation := PerApplicationFixedPoint.relation
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits

private abbrev selectedPadLayout :=
  (Lifecycle.PiRLC.v1_1.InputBinding.relationSource selectedRelation).cubeLayout

/-- Compute active selected matrix rows and return zero beyond their prefix. -/
def matrixRow
    (assignments : Vector (Vector F selectedShape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k)
    (matrix : Fin Spec.ProductionRelation.matrixCount) (index : Nat) : StoredRing :=
  if live : index < selectedPlan.rowCount then
    PiDECEvaluationRows.row (shape := selectedShape)
      (selectedPlan.portForm ⟨index, live⟩ matrix) assignments child
  else PiDECCommitmentFold.zero

/-- Pad uses every complete carrier column, including the carried tail. -/
def padRow
    (assignments : Vector (Vector F selectedShape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (index : Nat) : StoredRing :=
  if live : index < selectedShape.carrierWidth then
    PiDECEvaluationRows.row (shape := selectedShape)
      (SparseForm.singleton ⟨index, live⟩ 1) assignments child
  else PiDECCommitmentFold.zero

theorem matrixRow_outside
    (assignments : Vector (Vector F selectedShape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k)
    (matrix : Fin Spec.ProductionRelation.matrixCount) (index : Nat)
    (outside : selectedPlan.rowCount ≤ index) :
    (matrixRow assignments child matrix index).get = ringFZero := by
  rw [matrixRow, dif_neg (Nat.not_lt.mpr outside), PiDECCommitmentFold.zero_value]

theorem padRow_outside
    (assignments : Vector (Vector F selectedShape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (index : Nat)
    (outside : selectedShape.carrierWidth ≤ index) :
    (padRow assignments child index).get = ringFZero := by
  rw [padRow, dif_neg (Nat.not_lt.mpr outside), PiDECCommitmentFold.zero_value]

private theorem matrix_padding_zero
    (assignments : Vector (Vector F selectedShape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k)
    (matrix : Fin Spec.ProductionRelation.matrixCount)
    (vertex : BooleanVertex Lifecycle.cubeVariables)
    (outside : selectedPlan.rowCount ≤ NumericBooleanDomain.index vertex) :
    PiRLC.rowRing selectedRelation.system (assignments.get child).get matrix vertex =
      ringFZero := by
  have logicalZero (column : Fin selectedShape.logicalWidth) :
      selectedRelation.system.matrices matrix vertex column = 0 := by
    rw [system_matrices]
    have value := Poseidon2HashChainV1MatrixRows.padding_matrix_coefficient_zero
      ⟨NumericBooleanDomain.index vertex, NumericBooleanDomain.index_lt_twoPow vertex⟩
      outside matrix column
    simpa only [NumericBooleanDomain.vertex_index] using value
  exact zero_row_of_coefficients selectedRelation.system
    (assignments.get child).get matrix vertex logicalZero

/-- Numeric matrix selection has the existing semantic value at every Boolean
vertex. The selected padding theorem discharges the complete zero suffix. -/
theorem matrixRow_value
    (assignments : Vector (Vector F selectedShape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k)
    (matrix : Fin Spec.ProductionRelation.matrixCount)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    (matrixRow assignments child matrix (NumericBooleanDomain.index vertex)).get =
      PiRLC.rowRing selectedRelation.system (assignments.get child).get matrix vertex := by
  by_cases live : NumericBooleanDomain.index vertex < selectedPlan.rowCount
  · have decoded : selectedPlan.rowLayout.toColumn? vertex =
        some ⟨NumericBooleanDomain.index vertex, live⟩ :=
      (CanonicalRowLayout.toColumn?_eq_some_iff Lifecycle.cubeVariables
        selectedPlan.rowCount selectedPlan.rowCount_le vertex _).2 rfl
    have placed := selectedPlan.rowLayout.toVertex_toColumn vertex
      ⟨NumericBooleanDomain.index vertex, live⟩ decoded
    rw [matrixRow, dif_pos live, PiDECEvaluationRows.matrixRow_value, placed]
  · rw [matrixRow_outside assignments child matrix _ (Nat.le_of_not_lt live)]
    exact (matrix_padding_zero assignments child matrix vertex (Nat.le_of_not_lt live)).symm

private theorem empty_pad_row_zero
    (assignments : Vector (Vector F selectedShape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) :
    (PiDECEvaluationRows.row (shape := selectedShape)
      (SparseForm.empty : SparseForm selectedShape.carrierWidth) assignments child).get =
      ringFZero := by
  exact PiDECEvaluationRows.row_empty_value assignments child

/-- Numeric Pad selection equals the selected complete-carrier identity row
at every Boolean vertex, with no dropped tail coordinates. -/
theorem padRow_value
    (assignments : Vector (Vector F selectedShape.carrierWidth) productionGlobalParams.k)
    (child : Fin productionGlobalParams.k)
    (vertex : BooleanVertex Lifecycle.cubeVariables) :
    (padRow assignments child (NumericBooleanDomain.index vertex)).get =
      PiRLC.ExplicitMatrix.rowRing selectedRelation.system
        (PaperAlgebra.padMatrix
          (Lifecycle.PiRLC.v1_1.InputBinding.relationSource selectedRelation))
        (assignments.get child).get vertex := by
  change (padRow assignments child (NumericBooleanDomain.index vertex)).get =
    PiRLC.ExplicitMatrix.rowRing selectedRelation.system
      (selectedPadLayout.paddedIdentityEntry (0 : F) 1)
      (assignments.get child).get vertex
  have value := PiDECEvaluationRows.padRow_value (shape := selectedShape)
    selectedRelation.system selectedPadLayout vertex assignments child
  by_cases live : NumericBooleanDomain.index vertex < selectedShape.carrierWidth
  · have decoded : selectedPadLayout.toColumn? vertex =
        some ⟨NumericBooleanDomain.index vertex, live⟩ :=
      (CanonicalRowLayout.toColumn?_eq_some_iff Lifecycle.cubeVariables
        selectedShape.carrierWidth selectedRelation.cubeFits vertex _).2 rfl
    erw [form_of_some selectedPadLayout vertex _ decoded] at value
    rw [padRow, dif_pos live]
    exact value
  · have decoded : selectedPadLayout.toColumn? vertex = none :=
      (CanonicalRowLayout.toColumn?_eq_none_iff Lifecycle.cubeVariables
        selectedShape.carrierWidth selectedRelation.cubeFits vertex).2 (Nat.le_of_not_lt live)
    erw [form_of_none selectedPadLayout vertex decoded] at value
    rw [padRow_outside assignments child _ (Nat.le_of_not_lt live)]
    exact (empty_pad_row_zero assignments child).symm.trans value

end NightstreamFPrime.Export.Stage1.PiDECEvaluationSelectedPrefix
