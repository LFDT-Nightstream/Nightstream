import Nightstream.SuperNeo.Concrete.Parameters
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Product

/-!
Finite-batch production norm growth for the typed Phi81 `PiRLC.Algebra`.

Protocol: SuperNeo Definition 14 and `Pi_RLC`.
Phase: valid challenge actions followed by the canonical finite assignment
combination.
Constraint family: semantic norm only; this file emits no rows.

Owns: the coordinatewise `216` action bound, centered accumulation over the
exact head-first `PiRLCFinite.combineAssignments` fold, production arity
arithmetic, and the theorem with the exact `PiRLC.Algebra.norm_growth` field
shape.

Does not own: transcript derivation of valid challenges, commitments,
evaluation/public-input homomorphisms, Rust/R1CS refinement, row removal, or
constraint counts.

Emits constraints: no.

Authority boundary: the final theorem accepts only the algebra field's
verifier-owned arity cap, exact production-set challenge predicate, and fresh
relation norm facts. It calls the executable quotient-ring proof; no expansion
oracle or circuit result is an input.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.norm_growth.assignment.action` | every coordinate of one valid action is at most `216` | derived | `act_coordinate_le_expansion` |
| `nifs.pi_rlc.verify.norm_growth.assignment.finite` | `n` canonical actions accumulate to at most `n * 216` | derived | `combineAssignments_le` |
| `nifs.pi_rlc.verify.norm_growth.assignment.production` | `n <= 61 + 14` implies `n * 216 < 2^14` | verifier parameter theorem | `production_total_bound` |
| `nifs.pi_rlc.verify.norm_growth.algebra` | exact concrete `PiRLC.Algebra.norm_growth` field | derived | `relation_norm_growth` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Finite

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism

/-- One coordinate of a valid challenge action satisfies the executable
production expansion bound. -/
theorem act_coordinate_le_expansion
    {shape : Shape} (challenge : RingF) (assignment : Assignment shape)
    (challengeValid : Challenge.challengeValid challenge)
    (assignmentFresh : assignmentNormBounded 2 assignment)
    (column : Fin shape.carrierWidth) :
    centeredMagnitude (CarrierAction.act challenge assignment column) <= 216 := by
  let packed :=
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.decode column
  change centeredMagnitude
      (ringFMul challenge
        (CarrierAction.assignmentBlock assignment packed.1) packed.2) <= 216
  apply Product.ringFMul_le_expansion challenge
      (CarrierAction.assignmentBlock assignment packed.1)
  · exact challengeValid
  · intro lane
    exact assignmentFresh (CarrierAction.carrierColumn packed.1 lane)

/-- The exact head-first finite assignment combination has coordinate norm at
most the number of sources times the production expansion factor. -/
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

/-- Definition 14 specialized to any total arity below the production cap. -/
theorem production_total_bound {count : Nat}
    (totalBound : count <=
      productionGlobalParams.maxFresh + productionGlobalParams.k) :
    count * 216 < productionGlobalParams.bigB := by
  have countLe : count <= 303 := by
    simpa [productionGlobalParams] using totalBound
  change count * 216 < 65536
  omega

/-- Exact theorem supplied to the production `PiRLC.Algebra.norm_growth`
field. Its arguments and conclusion deliberately match that field after
specializing `params = productionGlobalParams` and
`combineAssignment = PiRLCFinite.combineAssignments`. -/
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

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Finite
