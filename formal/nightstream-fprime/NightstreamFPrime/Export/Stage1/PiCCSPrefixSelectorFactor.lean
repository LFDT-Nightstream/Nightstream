import NightstreamFPrime.Export.Stage1.PiCCSPrefixSelector
import NightstreamFPrime.Export.Stage1.PiCCSCarriedMoments

/-! Factor the existing prefix selector into its consumed/head polynomial
and the numeric suffix weight used by both retained contribution runners. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPrefixSelector

open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiCCSPolynomialRange (coefficient_ext coefficient_scale)

universe uField
variable {Field : Type uField}

/-- The exact consumed factor and current-coordinate head used by composePrefix. -/
def headPrefix (ops : InterpolationOps Field) {arity : Nat}
    (challenges : List Field) (target : CubePoint Field arity) : FixedPolynomial Field 1 :=
  FixedPolynomial.scale ops.toOps (consumedFactor ops challenges target)
    (PiCCSCarriedMoments.headSelector ops (dropPoint target challenges.length))

/-- The exact suffix coordinates used by the retained scalar accumulator. -/
def tailWeight (ops : InterpolationOps Field) {arity : Nat}
    (challenges : List Field) (target : CubePoint Field arity) : Nat → Field :=
  NumericBooleanDomain.tensorWeightCoordinates ops
    (target.coordinates.drop (challenges.length + 1))

private theorem tail_drop (values : List Field) (count : Nat) :
    (values.drop count).tail = values.drop (count + 1) := by
  induction count generalizing values with
  | zero => cases values <;> rfl
  | succ count ih =>
      cases values with
      | nil => rfl
      | cons value values => exact ih values

/-- Every coefficient retains the consumed factor, current head, and exact
numeric suffix weight. This also covers an empty Boolean suffix. -/
theorem selector_factor (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {arity remaining : Nat}
    (challenges : List Field) (suffix : BooleanVertex remaining)
    (target : CubePoint Field arity)
    (dimension : arity = challenges.length + remaining + 1) :
    selector ops challenges suffix target =
      FixedPolynomial.scale ops.toOps
        (tailWeight ops challenges target (NumericBooleanDomain.index suffix))
        (headPrefix ops challenges target) := by
  have restDimension : arity - challenges.length = remaining + 1 := by omega
  unfold selector headPrefix tailWeight
  rw [PiCCSCarriedMoments.equalitySelector_factor ops laws restDimension]
  have tail : (dropPoint target challenges.length).coordinates.tail =
      target.coordinates.drop (challenges.length + 1) :=
    tail_drop target.coordinates challenges.length
  rw [tail]
  apply coefficient_ext
  intro index
  simp only [coefficient_scale]
  rw [← laws.mul_assoc, laws.mul_comm (consumedFactor ops challenges target),
    laws.mul_assoc]

end NightstreamFPrime.Export.Stage1.PiCCSPrefixSelector
