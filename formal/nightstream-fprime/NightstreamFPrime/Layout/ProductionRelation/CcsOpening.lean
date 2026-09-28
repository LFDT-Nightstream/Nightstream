import NightstreamFPrime.Layout.ProductionRelation.PlanComposition
import NightstreamFPrime.Lifecycle.Relation
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLinearAlgebra

/-!
Owns the connection from literal production-plan rows to a fresh SuperNeo
v1.1 CCS opening. Inputs are row acceptance, a coordinate bound, and the exact
public projection. The commitment is computed from the same completed
assignment and verifier-owned key. No semantic representation is assumed.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.Plan

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- The exact scalar CCS residual vanishes on every Boolean row, including
the padded suffix, for the canonical complete carrier. -/
theorem rowsZero_implies_paperCcs
    {logicalWidth : Nat} (plan : ProductionRelation.Plan logicalWidth)
    (assignment : Assignment F logicalWidth)
    (rows : plan.RowsZero assignment) :
    ∀ vertex, evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
      (fun matrix => matrixVectorAt baseOps
        (Phi81CarrierLayout.extendMatrix 0 (plan.matrix matrix))
        (Phi81CarrierLayout.extendAssignment 0 assignment) vertex) = 0 := by
  intro vertex
  have images :
      (fun matrix => matrixVectorAt baseOps
        (Phi81CarrierLayout.extendMatrix 0 (plan.matrix matrix))
        (Phi81CarrierLayout.extendAssignment 0 assignment) vertex) =
      plan.rowImage assignment vertex := by
    funext matrix
    calc
      _ = matrixVectorAt baseOps (plan.matrix matrix) assignment vertex :=
        Phi81CarrierLayout.matrixVectorAt_extend baseOps baseLaws
          (plan.matrix matrix) assignment vertex
      _ = _ := matrixVectorAt_matrix plan assignment vertex matrix
  rw [images]
  cases decoded : plan.rowLayout.toColumn? vertex with
  | none =>
      have zeroImage : plan.rowImage assignment vertex = fun _ => 0 := by
        funext matrix
        simp only [rowImage, decoded]
      rw [zeroImage]
      exact Spec.ProductionRelation.polynomial_zeroImages
  | some row =>
      have sameVertex := plan.rowLayout.toVertex_toColumn vertex row decoded
      rw [← sameVertex]
      exact rows row

/-- Literal rows and bounded coordinates establish the exact fresh CCS
opening under the plan's key-facing matrices and actual Ajtai commitment. -/
theorem rowsZero_implies_freshHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (plan : ProductionRelation.Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Assignment F logicalWidth)
    (publicInput : PaperAlgebra.PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (rows : plan.RowsZero assignment)
    (bounded : ∀ column, centeredMagnitude (assignment column) < 2)
    (publicEqual : Phi81Relation.projectPublicInput
      (Phi81CarrierLayout.extendAssignment 0 assignment) = publicInput) :
    CCS.Holds (semantics key) productionGlobalParams
      (freshStatement (plan.logicalRelation (publicFits := publicFits) cubeFits)
        { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit key
            (Phi81CarrierLayout.extendAssignment 0 assignment)
          publicInputs := fun _ => publicInput })
      (Phi81CarrierLayout.extendAssignment 0 assignment) := by
  refine ⟨⟨rfl, publicEqual, ?_⟩, ?_⟩
  · intro column
    change centeredMagnitude
      (Phi81CarrierLayout.extendAssignment 0 assignment column) < 2
    by_cases inside : column.val < logicalWidth
    · simpa only [Phi81CarrierLayout.extendAssignment,
        Phi81CarrierLayout.logicalColumn?, dif_pos inside] using
        bounded ⟨column.val, inside⟩
    · simp only [Phi81CarrierLayout.extendAssignment,
        Phi81CarrierLayout.logicalColumn?, dif_neg inside]
      simp
  · change ∀ vertex, evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
      (fun matrix => matrixVectorAt baseOps
        (Phi81CarrierLayout.extendMatrix 0 (plan.matrix matrix))
        (Phi81CarrierLayout.extendAssignment 0 assignment) vertex) = 0
    exact rowsZero_implies_paperCcs plan assignment rows

end NightstreamFPrime.Layout.ProductionRelation.Plan
