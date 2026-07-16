import Nightstream.SuperNeo.Concrete.Necessity.Phi81OutputMismatch
import Nightstream.SuperNeo.Concrete.Phi81Relation

/-!
Focused regressions for the batch-invariant paper Phi81 relation carrier.
The fixture deliberately reuses nonzero completed-carrier data so the adapter
test exercises the derived coefficient image, not only a zero value.

| Protocol | Phase | Family | Regression |
|---|---|---|---|
| CCS/CE | carrier | public alignment | raw width 257 is excluded by the type |
| CE | evaluation | source adapter | a nonzero paper Phi81 lane survives batch erasure |
| `Pi_RLC` | evaluation | fixed `RingK` action | row-wise action commutes with Boolean MLE |
| CE | authority | matrix / lane coverage | the canonical array binds exact size and every lane |
| assurance | API | membership / completeness | typed relation theorems remain on the public surface |
| assurance | necessity | one-family countermodels | every retained value-level family has an isolated invalid acceptance witness |
-/

namespace tests.SuperNeoPhi81Relation

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Necessity.Phi81OutputMismatch
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

#check Shape.publicWidth_aligned
#check Shape.publicWidth_ne_257
#check Shape.publicColumn
#check Structure.matrixSource_kernel_eq
#check evaluations_size
#check evaluations_get
#check matrixEvaluation_apply_ofSourceData
#check matrixEvaluation_ofSourceData
#check evaluations_get_ofSourceData
#check Phi81Relation.EvaluationHomomorphism.BaseLinear.matrixEvaluation_zero
#check Phi81Relation.EvaluationHomomorphism.BaseLinear.matrixEvaluation_add
#check Phi81Relation.EvaluationHomomorphism.BaseLinear.matrixEvaluation_scale
#check Phi81Relation.EvaluationHomomorphism.BaseLinear.matrixEvaluation_combine
#check Phi81Relation.EvaluationHomomorphism.BaseLinear.evaluations_combine
#check Phi81Relation.EvaluationHomomorphism.RingKAction.ringKMul_right_scale
#check Phi81Relation.EvaluationHomomorphism.RingKAction.evaluateRows_action
#check Phi81Relation.EvaluationHomomorphism.RingKAction.evaluateRows_embeddedChallenge_action
#check Phi81Relation.EvaluationHomomorphism.RingFLaws.rawMulCoeffF_monomial
#check Phi81Relation.EvaluationHomomorphism.RingFLaws.ringFMul_basis_basis
#check Phi81Relation.EvaluationHomomorphism.RingFLaws.monomialReduce_recurrence
#check Phi81Relation.EvaluationHomomorphism.RingFLaws.ringFMul_leftActionComm
#check Phi81Relation.EvaluationHomomorphism.RingFLaws.ringFMul_barBasis_productOrder
#check Phi81Relation.EvaluationHomomorphism.PiDEC.evaluations_hom
#check Phi81Relation.EvaluationHomomorphism.PiDEC.relation_evaluations_hom
#check Phi81Relation.assignmentNormBounded
#check Phi81Relation.publicInputMatches
#check Phi81Relation.ccsSatisfied
#check Phi81Relation.evaluationPointValid_holds
#check Phi81Relation.ccsMembership_iff
#check Phi81Relation.ceMembership_iff
#check Phi81Relation.canonicalCCS_holds
#check Phi81Relation.canonicalCE_holds
#check Phi81Relation.evaluationsBound_iff_eq
#check Phi81Relation.ceMembership_iff_evaluationsBound
#check Phi81Relation.ce_evaluations_size_of_holds
#check Phi81Relation.Necessity.commitment_check_is_necessary
#check Phi81Relation.Necessity.public_input_check_is_necessary
#check Phi81Relation.Necessity.norm_check_is_necessary
#check Phi81Relation.Necessity.ccs_relation_check_is_necessary
#check Phi81Relation.Necessity.evaluation_size_check_is_necessary
#check Phi81Relation.Necessity.evaluation_lane_check_is_necessary
#check Phi81Relation.Necessity.no_invalid_typed_point
#check Phi81Relation.Minimality.cePlan_exact
#check Phi81Relation.Minimality.cePlan_inclusionMinimalSound

/-- One public ring exactly fills this fixture's completed 54-field carrier. -/
def alignedShape : Phi81Relation.Shape :=
  Phi81Relation.Shape.ofSemantic modelShape 1 (by decide)

/-- The paper public carrier is ring-aligned by construction. -/
example : alignedShape.publicWidth = ringDegree := by
  decide

/-- In particular, the old raw 257-field prefix is not accepted as this
typed paper public carrier. -/
example : alignedShape.publicWidth ≠ 257 := by
  exact alignedShape.publicWidth_ne_257

/-- The relation adapter preserves the independently derived nonzero Phi81
coefficient evaluation for an arbitrary semantic source batch. -/
example :
    matrixEvaluation
        (Phi81Relation.Structure.ofSourceData 1 (by decide) sourceData)
        (sourceData.assignment source) verifierPoints.rPrime matrix laneOne =
      K.one := by
  rw [Phi81Relation.matrixEvaluation_apply_ofSourceData]
  exact canonicalYRing_laneOne_eq_one

/-- Exact array size and all-lane authority hold for the canonical evaluation
array without appealing to a digest or a default value. -/
example :
    Phi81Relation.EvaluationsBound
      (Phi81Relation.Structure.ofSourceData 1 (by decide) sourceData)
      (sourceData.assignment source) verifierPoints.rPrime
      (Phi81Relation.evaluations
        (Phi81Relation.Structure.ofSourceData 1 (by decide) sourceData)
        (sourceData.assignment source) verifierPoints.rPrime) := by
  apply (Phi81Relation.evaluationsBound_iff_eq _ _ _ _).2
  rfl

end tests.SuperNeoPhi81Relation
