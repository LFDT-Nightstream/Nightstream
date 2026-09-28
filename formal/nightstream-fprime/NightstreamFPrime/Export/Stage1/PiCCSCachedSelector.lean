import NightstreamFPrime.Export.Stage1.PiCCSTensorWeights
import NightstreamFPrime.Export.Stage1.PiCCSFirstRound

/-! Reuse the prepared suffix tensor weights in the existing first-round
selector. The head affine factor and empty-target result are unchanged. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCachedSelector

open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uField
variable {Field : Type uField}

/-- The same selector construction with a prepared suffix-weight lookup. -/
def equalitySelector (ops : InterpolationOps Field) {arity remaining : Nat}
    (suffix : BooleanVertex remaining) (target : CubePoint Field arity)
    (tables : Array Field × Array Field) : FixedPolynomial Field 1 :=
  match target.coordinates with
  | [] => FixedPolynomial.zero ops.toOps 1
  | head :: tail =>
      FixedPolynomial.scale ops.toOps
        (PiCCSTensorWeights.lookup ops tail tables (NumericBooleanDomain.index suffix))
        (FixedPolynomial.affine (ops.sub ops.one head)
          (ops.sub head (ops.sub ops.one head)))

private theorem suffixWeight_value (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {remaining : Nat}
    (suffix : BooleanVertex remaining) (coordinates : List Field)
    (dimension : coordinates.length = remaining) :
    PiCCSTensorWeights.lookup ops coordinates (PiCCSTensorWeights.prepare ops coordinates)
        (NumericBooleanDomain.index suffix) =
      SumCheckTruthPath.pointEqualityCoordinates ops
        (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix) coordinates := by
  calc
    _ = NumericBooleanDomain.tensorWeightCoordinates ops coordinates
        (NumericBooleanDomain.index suffix) :=
      PiCCSTensorWeights.lookup_prepare ops
        (NumericBooleanDomain.WeightProductLaws.ofInterpolationEvaluationLaws laws)
        coordinates (NumericBooleanDomain.index suffix)
    _ = suffix.equalityWeight ops (⟨coordinates, dimension⟩ : CubePoint Field remaining) := by
      simpa only [NumericBooleanDomain.tensorWeight, NumericBooleanDomain.vertex_index] using
        NumericBooleanDomain.tensorWeight_eq_equalityWeight ops
          ⟨NumericBooleanDomain.index suffix, NumericBooleanDomain.index_lt_twoPow suffix⟩
          (⟨coordinates, dimension⟩ : CubePoint Field remaining)
    _ = SumCheckTruthPath.pointEquality ops
        (SumCheckTruthPath.VertexEncoding.toCubePoint ops suffix)
        (⟨coordinates, dimension⟩ : CubePoint Field remaining) :=
      (SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight ops laws suffix
        (⟨coordinates, dimension⟩ : CubePoint Field remaining)).symm
    _ = _ := rfl

/-- Prepared suffix weights preserve the complete original polynomial.
The sole dimension premise is the existing first-coordinate/suffix split. -/
theorem equalitySelector_prepare (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {arity remaining : Nat}
    (dimension : arity = remaining + 1) (suffix : BooleanVertex remaining)
    (target : CubePoint Field arity) :
    equalitySelector ops suffix target
        (PiCCSTensorWeights.prepare ops target.coordinates.tail) =
      PiCCSFirstRound.equalitySelector ops suffix target := by
  rcases target with ⟨coordinates, length⟩
  cases coordinates with
  | nil => rfl
  | cons head tail =>
      have tailDimension : tail.length = remaining := by
        simp only [List.length_cons] at length
        omega
      change FixedPolynomial.scale ops.toOps
          (PiCCSTensorWeights.lookup ops tail (PiCCSTensorWeights.prepare ops tail)
            (NumericBooleanDomain.index suffix))
          (FixedPolynomial.affine (ops.sub ops.one head)
            (ops.sub head (ops.sub ops.one head))) =
        FixedPolynomial.scale ops.toOps
          (SumCheckTruthPath.pointEqualityCoordinates ops
            (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops suffix) tail)
          (FixedPolynomial.affine (ops.sub ops.one head)
            (ops.sub head (ops.sub ops.one head)))
      rw [suffixWeight_value ops laws suffix tail tailDimension]

end NightstreamFPrime.Export.Stage1.PiCCSCachedSelector
