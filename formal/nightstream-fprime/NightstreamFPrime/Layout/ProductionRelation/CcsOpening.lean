import NightstreamFPrime.Layout.ProductionRelation.PlanComposition
import NightstreamFPrime.Lifecycle.Relation
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RingCarrierLinearAlgebra
import NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.Commitment

namespace NightstreamFPrime.Layout.ProductionRelation.Plan

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

theorem rowsZero_implies_paperCcs
    {logicalWidth : Nat} (plan : ProductionRelation.Plan logicalWidth)
    (assignment : Assignment F logicalWidth)
    (rows : plan.RowsZero assignment) :
    ∀ vertex, evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
      (fun matrix => matrixVectorAt baseOps
        (RingCarrierLayout.extendMatrix 0 (plan.matrix matrix))
        (RingCarrierLayout.extendAssignment 0 assignment) vertex) = 0 := by
  intro vertex
  have images :
      (fun matrix => matrixVectorAt baseOps
        (RingCarrierLayout.extendMatrix 0 (plan.matrix matrix))
        (RingCarrierLayout.extendAssignment 0 assignment) vertex) =
      plan.rowImage assignment vertex := by
    funext matrix
    calc
      _ = matrixVectorAt baseOps (plan.matrix matrix) assignment vertex :=
        RingCarrierLayout.matrixVectorAt_extend baseOps baseLaws
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

theorem rowsZero_implies_freshHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (plan : ProductionRelation.Plan logicalWidth)
    (cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Assignment F logicalWidth)
    (publicInput : PaperAlgebra.PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (rows : plan.RowsZero assignment)
    (bounded : ∀ column, centeredMagnitude (assignment column) < 2)
    (publicEqual : RingRelation.projectPublicInput
      (RingCarrierLayout.extendAssignment 0 assignment) = publicInput) :
    CCS.Holds (semantics key) productionGlobalParams
      (freshStatement (plan.logicalRelation (publicFits := publicFits) cubeFits)
        { commitments := fun _ => RingRelation.PiRLCAlgebra.Commitment.commit key
            (RingCarrierLayout.extendAssignment 0 assignment)
          publicInputs := fun _ => publicInput })
      (RingCarrierLayout.extendAssignment 0 assignment) := by
  refine ⟨⟨rfl, publicEqual, ?_⟩, ?_⟩
  · intro column
    change centeredMagnitude
      (RingCarrierLayout.extendAssignment 0 assignment column) < 2
    by_cases inside : column.val < logicalWidth
    · simpa only [RingCarrierLayout.extendAssignment,
        RingCarrierLayout.logicalColumn?, dite_eq_left inside] using
        bounded ⟨column.val, inside⟩
    · simp only [RingCarrierLayout.extendAssignment,
        RingCarrierLayout.logicalColumn?, dite_eq_right inside]
      simp
  · change ∀ vertex, evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
      (fun matrix => matrixVectorAt baseOps
        (RingCarrierLayout.extendMatrix 0 (plan.matrix matrix))
        (RingCarrierLayout.extendAssignment 0 assignment) vertex) = 0
    exact rowsZero_implies_paperCcs plan assignment rows

end NightstreamFPrime.Layout.ProductionRelation.Plan
