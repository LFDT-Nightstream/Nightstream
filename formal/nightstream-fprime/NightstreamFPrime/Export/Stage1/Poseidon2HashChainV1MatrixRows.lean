import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

/-!
Owns theorem-only conformance evidence for the final 14-matrix
`Poseidon2HashChainV1` logical relation. The evidence connects the compact
program carried by the sealed package to the exact structural plan, covers
the explicit zero matrix in slot 13, and proves the complete Boolean-domain
padding suffix is zero.

This module does not expand matrix entries, emit an artifact, or define a
second relation.
-/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1MatrixRows

open NightstreamFPrime.Layout
open NightstreamFPrime.Spec.Folding.PiCCS
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem portForm_coefficient_eq_matrix
    {logicalWidth : Nat} (plan : ProductionRelation.Plan logicalWidth)
    (row : Fin plan.rowCount)
    (port : Fin Spec.ProductionRelation.matrixCount)
    (column : Fin logicalWidth) :
    (plan.portForm row port).coefficient column =
      plan.matrix port (plan.rowLayout.toVertex row) column := by
  unfold ProductionRelation.Plan.matrix
  rw [plan.rowLayout.toColumn_toVertex]

private theorem zeroPort_form_empty
    {logicalWidth : Nat} (plan : ProductionRelation.Plan logicalWidth)
    (row : Fin plan.rowCount) :
    plan.portForm row Spec.ProductionRelation.zeroPort =
      ProductionRelation.SparseForm.empty := by
  unfold ProductionRelation.Plan.portForm
  rw [ProductionRelation.meaningfulPort?_zeroPort]

private theorem padding_coefficient_zero
    {logicalWidth : Nat} (plan : ProductionRelation.Plan logicalWidth)
    (ordinal : Fin (2 ^ Lifecycle.cubeVariables))
    (padding : plan.rowCount ≤ ordinal.val)
    (port : Fin Spec.ProductionRelation.matrixCount)
    (column : Fin logicalWidth) :
    plan.matrix port
        (NumericBooleanDomain.vertex Lifecycle.cubeVariables ordinal) column =
      0 := by
  have decoded :
      plan.rowLayout.toColumn?
          (NumericBooleanDomain.vertex Lifecycle.cubeVariables ordinal) =
        none := by
    apply (CanonicalRowLayout.toColumn?_eq_none_iff
      Lifecycle.cubeVariables plan.rowCount plan.rowCount_le
      (NumericBooleanDomain.vertex Lifecycle.cubeVariables ordinal)).2
    simpa only [NumericBooleanDomain.index_vertex] using padding
  unfold ProductionRelation.Plan.matrix
  rw [decoded]

/-- The compact program carried by the sealed package returns the exact
structural-plan forms for every active row. -/
theorem compactProgram_row?_eq_structuralPlan_forms
    (row : Fin (PerApplicationFixedPoint.structuralPlan
      Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits).rowCount) :
    (PerApplicationMatrixProgram.matrixProgram
        Poseidon2HashChainV1Package.application).row?
        (PerApplicationFixedPoint.logicalWidth
          Poseidon2HashChainV1Package.application)
        (PerApplicationCanonicalPackage.sourceRow
          Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits) row.val =
      some ((PerApplicationFixedPoint.structuralPlan
        Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits).forms row) :=
  PerApplicationCanonicalPackage.matrixProgram_row?
    Poseidon2HashChainV1Package.application
    Poseidon2HashChainV1Package.fits row

/-- Every one of the 14 structural-plan ports supplies the corresponding
coefficient of the final key-facing logical relation at the canonical active
row vertex. -/
theorem allPort_coefficient_eq_logicalRelation_matrix
    (row : Fin (PerApplicationFixedPoint.structuralPlan
      Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits).rowCount)
    (port : Fin Spec.ProductionRelation.matrixCount)
    (column : Fin (PerApplicationFixedPoint.logicalWidth
      Poseidon2HashChainV1Package.application)) :
    ((PerApplicationFixedPoint.structuralPlan
        Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits).portForm row port).coefficient column =
      (PerApplicationFixedPoint.relation
          Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits).matrices port
        ((PerApplicationFixedPoint.structuralPlan
          Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits).rowLayout.toVertex row) column := by
  rw [PerApplicationFixedPoint.relation_matrices]
  exact portForm_coefficient_eq_matrix _ row port column

/-- Matrix slot 13 is the explicit empty sparse form at every active row. -/
theorem slot13_form_empty
    (row : Fin (PerApplicationFixedPoint.structuralPlan
      Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits).rowCount) :
    (PerApplicationFixedPoint.structuralPlan
        Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits).portForm row
          Spec.ProductionRelation.zeroPort =
      ProductionRelation.SparseForm.empty :=
  zeroPort_form_empty _ row

/-- Matrix slot 13 is the zero matrix on the complete Boolean domain. -/
theorem slot13_matrix_zero :
    (PerApplicationFixedPoint.relation
        Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits).matrices
        Spec.ProductionRelation.zeroPort =
      fun _ _ => 0 := by
  rw [PerApplicationFixedPoint.relation_matrices]
  exact ProductionRelation.Plan.zeroPort_matrix _

/-- Every matrix coefficient is zero after the active-row prefix and through
the end of the exact `2^28` Boolean row domain. -/
theorem padding_matrix_coefficient_zero
    (ordinal : Fin (2 ^ Lifecycle.cubeVariables))
    (padding : (PerApplicationFixedPoint.structuralPlan
      Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits).rowCount ≤ ordinal.val)
    (port : Fin Spec.ProductionRelation.matrixCount)
    (column : Fin (PerApplicationFixedPoint.logicalWidth
      Poseidon2HashChainV1Package.application)) :
    (PerApplicationFixedPoint.relation
        Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits).matrices port
        (NumericBooleanDomain.vertex Lifecycle.cubeVariables ordinal) column =
      0 := by
  rw [PerApplicationFixedPoint.relation_matrices]
  exact padding_coefficient_zero _ ordinal padding port column

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1MatrixRows
