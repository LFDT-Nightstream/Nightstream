import NightstreamFPrime.Layout.ProductionRelation.CcsOpening

/-!
Decode an arbitrary accepted CCS carrier into the production plan's logical
assignment. Matrix padding contributes zero for every carrier value; no
canonical padding, witness encoder or representation premise is required.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.Plan

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Logical columns are read from their exact positions in the opened carrier. -/
def logicalAssignment {logicalWidth : Nat}
    (assignment : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    Assignment F logicalWidth :=
  fun column => assignment (Phi81CarrierLayout.embedLogical column)

/-- Zero matrix columns make the padding values irrelevant to every image. -/
theorem matrixVectorAt_logicalAssignment {cubeSize logicalWidth : Nat}
    (matrix : BooleanMatrix F cubeSize logicalWidth)
    (assignment : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth))
    (vertex : BooleanVertex cubeSize) :
    matrixVectorAt baseOps (Phi81CarrierLayout.extendMatrix 0 matrix) assignment vertex =
      matrixVectorAt baseOps matrix (logicalAssignment assignment) vertex := by
  calc
    _ = matrixVectorAt baseOps (Phi81CarrierLayout.extendMatrix 0 matrix)
        (Phi81CarrierLayout.extendAssignment 0 (logicalAssignment assignment)) vertex := by
      unfold matrixVectorAt
      congr 1
      funext total column
      by_cases inside : column.val < logicalWidth
      · simp only [Phi81CarrierLayout.extendAssignment, Phi81CarrierLayout.logicalColumn?,
          dif_pos inside, logicalAssignment, Phi81CarrierLayout.embedLogical]
      · rw [Phi81CarrierLayout.extendMatrix_tail_zero 0 matrix vertex column
          (Nat.le_of_not_gt inside)]
        simp [baseOps]
    _ = _ := Phi81CarrierLayout.matrixVectorAt_extend baseOps baseLaws matrix
      (logicalAssignment assignment) vertex

/-- Restriction preserves the public input when that prefix belongs to the
logical assignment, independently of any values in the padding. -/
theorem projectPublicInput_logicalAssignment {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (publicLogicalFits : ringDegree * publicRingColumns ≤ logicalWidth)
    (assignment : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    Phi81Relation.projectPublicInput (shape := FullShape logicalWidth publicFits)
        (Phi81CarrierLayout.extendAssignment 0 (logicalAssignment assignment)) =
      Phi81Relation.projectPublicInput (shape := FullShape logicalWidth publicFits) assignment := by
  funext column
  have inside : column.val < logicalWidth :=
    Nat.lt_of_lt_of_le column.isLt publicLogicalFits
  let logicalColumn : Fin logicalWidth := ⟨column.val, inside⟩
  have same : (FullShape logicalWidth publicFits).publicColumn column =
      Phi81CarrierLayout.embedLogical logicalColumn := by
    apply Fin.ext
    rfl
  change Phi81CarrierLayout.extendAssignment 0 (logicalAssignment assignment)
      ((FullShape logicalWidth publicFits).publicColumn column) =
    assignment ((FullShape logicalWidth publicFits).publicColumn column)
  rw [same, Phi81CarrierLayout.extendAssignment_embedLogical]
  rfl

/-- Fresh CCS membership supplies both logical row acceptance and the actual
public input for the same opened carrier and verifier-selected matrix plan. -/
theorem freshHolds_implies_rowsAndPublic {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (plan : ProductionRelation.Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth))
    (publicLogicalFits : ringDegree * publicRingColumns ≤ logicalWidth)
    (accepted : CCS.Holds (semantics key) productionGlobalParams
      (freshStatement (plan.logicalRelation (publicFits := publicFits) cubeFits) fresh)
      assignment) :
    plan.RowsZero (logicalAssignment assignment) ∧
      Phi81Relation.projectPublicInput (shape := FullShape logicalWidth publicFits)
        (Phi81CarrierLayout.extendAssignment 0 (logicalAssignment assignment)) =
          fresh.publicInputs ⟨0, by decide⟩ := by
  constructor
  · have ccs := accepted.2
    change ∀ vertex, evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
      (fun matrix => matrixVectorAt baseOps
        (Phi81CarrierLayout.extendMatrix 0 (plan.matrix matrix)) assignment vertex) = 0 at ccs
    intro row
    have images :
        (fun matrix => matrixVectorAt baseOps
          (Phi81CarrierLayout.extendMatrix 0 (plan.matrix matrix)) assignment
          (plan.rowLayout.toVertex row)) =
        plan.rowImage (logicalAssignment assignment) (plan.rowLayout.toVertex row) := by
      funext matrix
      exact (matrixVectorAt_logicalAssignment (plan.matrix matrix) assignment
        (plan.rowLayout.toVertex row)).trans
        (matrixVectorAt_matrix plan (logicalAssignment assignment)
          (plan.rowLayout.toVertex row) matrix)
    simpa only [images] using ccs (plan.rowLayout.toVertex row)
  · exact (projectPublicInput_logicalAssignment publicLogicalFits assignment).trans accepted.1.2.1

end NightstreamFPrime.Layout.ProductionRelation.Plan
