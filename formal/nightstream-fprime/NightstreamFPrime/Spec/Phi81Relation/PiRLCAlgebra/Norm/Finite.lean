import NightstreamFPrime.Spec.Profile
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiRLCFinite
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Product

/-!
Finite-batch production norm growth for the typed Phi81 `PiRLC.Algebra`.

Protocol: SuperNeo v1.2 Definition 22 and `Pi_RLC`.

Owns: the coordinatewise `216` action bound, centered accumulation over the
exact head-first `PiRLCFinite.combineAssignments` fold, production arity
arithmetic, and the theorem with the exact `PiRLC.Algebra.norm_growth` field
shape.

Does not own: transcript derivation of valid challenges, commitments,
evaluation/public-input homomorphisms, Rust/R1CS refinement, row removal, or
constraint counts.
-/

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Finite

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism

theorem act_coordinate_le_expansion
    {shape : Shape} (challenge : RingF) (assignment : Assignment shape)
    (challengeValid : Challenge.challengeValid challenge)
    (assignmentFresh : assignmentNormBounded 2 assignment)
    (column : Fin shape.carrierWidth) :
    centeredMagnitude (CarrierAction.act challenge assignment column) <= 216 := by
  let packed :=
    NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.decode column
  change centeredMagnitude
      (ringFMul challenge
        (CarrierAction.assignmentBlock assignment packed.1) packed.2) <= 216
  apply Product.ringFMul_le_expansion challenge
      (CarrierAction.assignmentBlock assignment packed.1)
  · exact challengeValid
  · intro lane
    exact assignmentFresh (CarrierAction.carrierColumn packed.1 lane)

theorem combineAssignments_le
    {shape : Shape} {count : Nat}
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape)
    (challengesValid : forall index,
      Challenge.challengeValid (challenges index))
    (assignmentsFresh : forall index,
      assignmentNormBounded 2 (assignments index))
    (column : Fin shape.carrierWidth) :
    centeredMagnitude
        (PiRLCFinite.combineAssignments challenges assignments column) <=
      count * 216 := by
  induction count with
  | zero =>
      simp [PiRLCFinite.combineAssignments, BaseLinear.assignmentZero,
        BaseLinear.Raw.assignmentZero, Centered.centeredMagnitude_zero]
  | succ count inductionHypothesis =>
      have headBound := act_coordinate_le_expansion
        (challenges 0) (assignments 0) (challengesValid 0)
        (assignmentsFresh 0) column
      have tailBound := inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => assignments index.succ)
        (fun index => challengesValid index.succ)
        (fun index => assignmentsFresh index.succ)
      rw [PiRLCFinite.combineAssignments]
      change centeredMagnitude
          (CarrierAction.act (challenges 0) (assignments 0) column +
            PiRLCFinite.combineAssignments
              (fun index => challenges index.succ)
              (fun index => assignments index.succ) column) <=
        (count + 1) * 216
      exact Nat.le_trans
        (Centered.centeredMagnitude_add_le _ _)
        (by
          have added := Nat.add_le_add headBound tailBound
          omega)

/-- Definition 22 specialized to any total arity below the production cap. -/
theorem production_total_bound {count : Nat}
    (totalBound : count <=
      productionGlobalParams.maxFresh + productionGlobalParams.k) :
    count * 216 < productionGlobalParams.bigB := by
  have countLe : count <= 303 := by
    simpa [productionGlobalParams] using totalBound
  change count * 216 < 65536
  omega

theorem relation_norm_growth
    {shape : Shape} {Commitment : Type}
    (commit : Assignment shape -> Commitment)
    {count : Nat}
    (totalBound : count <=
      productionGlobalParams.maxFresh + productionGlobalParams.k)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape)
    (challengesValid : forall index,
      Challenge.challengeValid (challenges index))
    (assignmentsFresh : forall index,
      (relationSemantics commit).normBounded productionGlobalParams.b
        (assignments index)) :
    (relationSemantics commit).normBounded productionGlobalParams.bigB
      (PiRLCFinite.combineAssignments challenges assignments) := by
  have fresh : forall index,
      assignmentNormBounded 2 (assignments index) := by
    intro index
    simpa [relationSemantics, productionGlobalParams] using
      assignmentsFresh index
  have finiteBound := combineAssignments_le challenges assignments
    challengesValid fresh
  have strictBound := production_total_bound totalBound
  change assignmentNormBounded productionGlobalParams.bigB
    (PiRLCFinite.combineAssignments challenges assignments)
  intro column
  exact Nat.lt_of_le_of_lt (finiteBound column) strictBound

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Finite
