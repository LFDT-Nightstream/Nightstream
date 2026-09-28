import NightstreamFPrime.Export.Stage1.PiCCSCachedSelector

/-! Equality selectors after a list of prover challenges. The production
shape is unchanged: consumed coordinates contribute a scalar factor, and
the existing selector handles the next coordinate and Boolean suffix. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPrefixSelector

open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open ProtocolPolynomialDegree.Support (polynomialLaws)

universe uField
variable {Field : Type uField}

private theorem equality_append (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left first rest tail : List Field) (same : left.length = first.length) :
    SumCheckTruthPath.pointEqualityCoordinates ops (left ++ rest) (first ++ tail) =
      ops.mul (SumCheckTruthPath.pointEqualityCoordinates ops left first)
        (SumCheckTruthPath.pointEqualityCoordinates ops rest tail) := by
  induction left generalizing first with
  | nil =>
      cases first with
      | nil =>
          simp only [List.nil_append, SumCheckTruthPath.pointEqualityCoordinates, laws.one_mul]
      | cons target first => simp at same
  | cons value left ih =>
      cases first with
      | nil => simp at same
      | cons target first =>
          have lengths : left.length = first.length := by simpa using same
          simp only [List.cons_append, SumCheckTruthPath.pointEqualityCoordinates]
          rw [ih first lengths]
          exact (laws.mul_assoc _ _ _).symm

/-- The remaining coordinates of the original point, in the original order. -/
def dropPoint {arity : Nat} (target : CubePoint Field arity) (count : Nat) :
    CubePoint Field (arity - count) where
  coordinates := target.coordinates.drop count
  dimension := by rw [List.length_drop, target.dimension]

/-- The consumed equality factors. This is computed from challenges, not
from a supplied prover result or an asserted scalar factor. -/
def consumedFactor (ops : InterpolationOps Field) {arity : Nat}
    (challenges : List Field) (target : CubePoint Field arity) : Field :=
  SumCheckTruthPath.pointEqualityCoordinates ops challenges
    (target.coordinates.take challenges.length)

/-- Extend the existing affine selector by all consumed equality factors. -/
def selector (ops : InterpolationOps Field) {arity remaining : Nat}
    (challenges : List Field) (suffix : BooleanVertex remaining)
    (target : CubePoint Field arity) : FixedPolynomial Field 1 :=
  FixedPolynomial.scale ops.toOps (consumedFactor ops challenges target)
    (PiCCSFirstRound.equalitySelector ops suffix (dropPoint target challenges.length))

/-- The selector is exactly the original equality polynomial at the
concatenated challenge prefix, current coordinate and Boolean suffix.
No nonzero-factor or challenge-distribution hypothesis is required. -/
theorem selector_evaluate (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {arity remaining : Nat}
    (challenges : List Field) (suffix : BooleanVertex remaining)
    (target : CubePoint Field arity)
    (dimension : arity = challenges.length + remaining + 1) (value : Field) :
    (selector ops challenges suffix target).evaluate ops.toOps value =
      SumCheckTruthPath.pointEqualityCoordinates ops
        (challenges ++ value :: SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix)
        target.coordinates := by
  have restDimension : arity - challenges.length = remaining + 1 := by omega
  have fits : challenges.length ≤ target.coordinates.length := by
    rw [target.dimension]
    omega
  have takeLength : challenges.length = (target.coordinates.take challenges.length).length := by
    rw [List.length_take, Nat.min_eq_left fits]
  rw [selector, FixedPolynomial.evaluate_scale ops.toOps (polynomialLaws laws),
    PiCCSFirstRound.equalitySelector_evaluate ops laws restDimension]
  change ops.mul
      (SumCheckTruthPath.pointEqualityCoordinates ops challenges
        (target.coordinates.take challenges.length))
      (SumCheckTruthPath.pointEqualityCoordinates ops
        (value :: SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix)
        (target.coordinates.drop challenges.length)) = _
  calc
    _ = SumCheckTruthPath.pointEqualityCoordinates ops
        (challenges ++ value :: SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix)
        (target.coordinates.take challenges.length ++ target.coordinates.drop challenges.length) :=
      (equality_append ops laws challenges (target.coordinates.take challenges.length)
        (value :: SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix)
        (target.coordinates.drop challenges.length) takeLength).symm
    _ = _ := by rw [List.take_append_drop]

/-- Use the existing prepared tensor weights for the remaining suffix. -/
def cachedSelector (ops : InterpolationOps Field) {arity remaining : Nat}
    (challenges : List Field) (suffix : BooleanVertex remaining)
    (target : CubePoint Field arity) (tables : Array Field × Array Field) :
    FixedPolynomial Field 1 :=
  FixedPolynomial.scale ops.toOps (consumedFactor ops challenges target)
    (PiCCSCachedSelector.equalitySelector ops suffix
      (dropPoint target challenges.length) tables)

/-- Prepared suffix tables preserve every coefficient of the selector. -/
theorem cachedSelector_prepare (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {arity remaining : Nat}
    (challenges : List Field) (suffix : BooleanVertex remaining)
    (target : CubePoint Field arity)
    (dimension : arity = challenges.length + remaining + 1) :
    cachedSelector ops challenges suffix target
        (PiCCSTensorWeights.prepare ops (dropPoint target challenges.length).coordinates.tail) =
      selector ops challenges suffix target := by
  have restDimension : arity - challenges.length = remaining + 1 := by omega
  unfold cachedSelector selector
  rw [PiCCSCachedSelector.equalitySelector_prepare ops laws restDimension]

end NightstreamFPrime.Export.Stage1.PiCCSPrefixSelector
