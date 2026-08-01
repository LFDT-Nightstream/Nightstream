import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsCompiler
import Nightstream.SuperNeo.Concrete.Phi81Relation.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierMatrixVector

/-!
Contract: expose one finite native-CCS compiler result as a Phi81 relation
structure without changing its rows, columns, matrices, or polynomial.

Assurance tier: model-level.

Owns: the batch-free Phi81 shape and structure projection for a valid native
CCS program.

Does not own: a concrete program, public-width selection, relation setup,
fixed-point stability, commitments, Rust, or a security reduction.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.NativeCcsPhi81

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def shape
    (program : NativeCcsProgram.Program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length) :
    Phi81Relation.Shape where
  rowVariables := domain.rowVariables
  logicalWidth := program.columnIds.length
  matrixCount := NativeCcsSelector.matrixCount
  publicRingColumns := publicRingColumns
  publicFits := publicFits

noncomputable def relation
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length) :
    Phi81Relation.Structure
      (shape program domain publicRingColumns publicFits) where
  matrices :=
    (NativeCcsCompiler.system program valid domain).matrices
  constraintPolynomial :=
    (NativeCcsCompiler.system program valid domain).constraintPolynomial

@[simp] theorem relation_polynomial
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length) :
    (relation program valid domain publicRingColumns
      publicFits).constraintPolynomial =
        NativeCcsSelector.constraintPolynomial := by
  rfl

private theorem matrixImagesAt_completed
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length)
    (assignment : Fin program.columnIds.length → F)
    (vertex : BooleanVertex domain.rowVariables) :
    matrixImagesAt baseOps
        (relation program valid domain publicRingColumns
          publicFits).matrixSource.system
        (Phi81CarrierLayout.extendAssignment 0 assignment) vertex =
      matrixImagesAt baseOps
        (NativeCcsCompiler.system program valid domain) assignment vertex := by
  funext matrix
  change
    matrixVectorAt baseOps
        (Phi81CarrierLayout.extendMatrix 0
          ((NativeCcsCompiler.system program valid domain).matrices matrix))
        (Phi81CarrierLayout.extendAssignment 0 assignment) vertex =
      matrixVectorAt baseOps
        ((NativeCcsCompiler.system program valid domain).matrices matrix)
        assignment vertex
  exact Phi81CarrierMatrixVector.matrixVectorAt_extend_eq
    ((NativeCcsCompiler.system program valid domain).matrices matrix)
    assignment vertex

private theorem ccsSatisfied_completed_iff
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length)
    (assignment : Fin program.columnIds.length → F) :
    Phi81Relation.ccsSatisfied
        (relation program valid domain publicRingColumns publicFits)
        (Phi81CarrierLayout.extendAssignment 0 assignment) ↔
      ConstraintSatisfied baseOps
        (NativeCcsCompiler.system program valid domain) assignment := by
  unfold Phi81Relation.ccsSatisfied ConstraintSatisfied residualAt
  constructor <;> intro satisfied vertex
  · have current := satisfied vertex
    rw [matrixImagesAt_completed program valid domain publicRingColumns
      publicFits assignment vertex] at current
    exact current
  · rw [matrixImagesAt_completed program valid domain publicRingColumns
      publicFits assignment vertex]
    exact satisfied vertex

private theorem matrixImagesAt_arbitrary
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length)
    (assignment :
      Phi81Relation.Assignment
        (shape program domain publicRingColumns publicFits))
    (vertex : BooleanVertex domain.rowVariables) :
    matrixImagesAt baseOps
        (relation program valid domain publicRingColumns
          publicFits).matrixSource.system
        assignment vertex =
      matrixImagesAt baseOps
        (NativeCcsCompiler.system program valid domain)
        (fun logical =>
          assignment (Phi81CarrierLayout.embedLogical logical))
        vertex := by
  funext matrix
  change
    matrixVectorAt baseOps
        (Phi81CarrierLayout.extendMatrix 0
          ((NativeCcsCompiler.system program valid domain).matrices matrix))
        assignment vertex =
      matrixVectorAt baseOps
        ((NativeCcsCompiler.system program valid domain).matrices matrix)
        (fun logical =>
          assignment (Phi81CarrierLayout.embedLogical logical))
        vertex
  exact Phi81CarrierMatrixVector.matrixVectorAt_extendMatrix_eq
    ((NativeCcsCompiler.system program valid domain).matrices matrix)
    assignment vertex

/-- The completed Phi81 relation reads only the logical prefix of any
complete-carrier assignment. No zero-padding premise is required. -/
theorem ccsSatisfied_arbitrary_iff
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length)
    (assignment :
      Phi81Relation.Assignment
        (shape program domain publicRingColumns publicFits)) :
    Phi81Relation.ccsSatisfied
        (relation program valid domain publicRingColumns publicFits)
        assignment ↔
      NativeCcsSelector.Satisfies program.rows
        (NativeCcsCompiler.pulledAssignment program valid
          (fun logical =>
            assignment (Phi81CarrierLayout.embedLogical logical))) := by
  rw [← NativeCcsCompiler.constraintSatisfied_iff]
  unfold Phi81Relation.ccsSatisfied
    CCSResidualTable.ConstraintSatisfied CCSResidualTable.residualAt
  constructor <;> intro satisfied vertex
  · have current := satisfied vertex
    rw [matrixImagesAt_arbitrary program valid domain publicRingColumns
      publicFits assignment vertex] at current
    exact current
  · rw [matrixImagesAt_arbitrary program valid domain publicRingColumns
      publicFits assignment vertex]
    exact satisfied vertex

/-- The Phi81 relation accepts the canonical completion exactly when the
finite compiler accepts the same indexed assignment. The constant-one check
remains explicit because it is part of the physical program boundary, not
part of Definition 11's residual predicate. -/
theorem accepts_assignment_iff
    (program : NativeCcsProgram.Program)
    (valid : NativeCcsCompiler.Valid program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length)
    (assignment : Fin program.columnIds.length → F) :
    assignment
          (NativeCcsCompiler.ColumnIndex.index program valid program.one) = 1 ∧
        Phi81Relation.ccsSatisfied
          (relation program valid domain publicRingColumns publicFits)
          (Phi81CarrierLayout.extendAssignment 0 assignment) ↔
      NativeCcsCompiler.IndexedAccepts program valid domain assignment := by
  unfold NativeCcsCompiler.IndexedAccepts
  rw [ccsSatisfied_completed_iff]

end Nightstream.Implementation.Lowering.Goldilocks.NativeCcsPhi81
