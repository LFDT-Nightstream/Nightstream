import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetPolynomial
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra

/-! Factor the existing canonical carried-coordinate gamma sums. Source
weights can be applied before a linear image evaluation. The matrix total
is local; the enclosing pair kernel owns its matrixEvaluationOffset shift. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSGammaAggregation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open FiniteSumAlgebra (sumMap)

universe uField uLeft uRight
variable {Field : Type uField} {shape : Shape}

private def shiftLaws (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) : TargetPolynomial.ShiftLaws ops.toOps where
  one_mul := laws.one_mul
  mul_assoc := laws.mul_assoc
  mul_zero := laws.mul_zero
  mul_add := laws.left_distrib

private theorem sumMap_append (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {Index : Type uLeft}
    (left right : List Index) (value : Index → Field) :
    sumMap ops (left ++ right) value = ops.add (sumMap ops left value) (sumMap ops right value) := by
  induction left with
  | nil => exact (laws.zero_add _).symm
  | cons index left ih =>
      change ops.add (value index) (sumMap ops (left ++ right) value) =
        ops.add (ops.add (value index) (sumMap ops left value)) (sumMap ops right value)
      rw [ih]
      exact (laws.add_assoc _ _ _).symm

private theorem sumMap_map (ops : InterpolationOps Field)
    {Left : Type uLeft} {Right : Type uRight}
    (indices : List Left) (map : Left → Right) (value : Right → Field) :
    sumMap ops (indices.map map) value = sumMap ops indices (fun index => value (map index)) := by
  simp only [FiniteSumAlgebra.sumMap, List.map_map, Function.comp_def]

private theorem sumMap_flatMap (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {Left : Type uLeft} {Right : Type uRight}
    (indices : List Left) (next : Left → List Right) (value : Right → Field) :
    sumMap ops (indices.flatMap next) value =
      sumMap ops indices (fun index => sumMap ops (next index) value) := by
  induction indices with
  | nil => rfl
  | cons index indices ih =>
      rw [List.flatMap_cons, sumMap_append ops laws]
      exact congrArg (ops.add (sumMap ops (next index) value)) ih

private theorem sumPadCoordinates (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (value : PadCoordinate shape → Field) :
    sumMap ops (canonicalPadCoordinates shape) value =
      sumMap ops (canonicalFinIndices shape.coefficientCount) (fun coefficient =>
        sumMap ops (canonicalFinIndices shape.runningCount) (fun source => value ⟨source, coefficient⟩)) := by
  simp only [canonicalPadCoordinates, sumMap_flatMap ops laws, sumMap_map ops]

private theorem sumMatrixCoordinates (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (value : MatrixCoordinate shape → Field) :
    sumMap ops (canonicalMatrixCoordinates shape) value =
      sumMap ops (canonicalFinIndices shape.coefficientCount) (fun coefficient =>
        sumMap ops (canonicalFinIndices shape.matrixCount) (fun matrix =>
          sumMap ops (canonicalFinIndices shape.runningCount)
            (fun source => value ⟨source, matrix, coefficient⟩))) := by
  simp only [canonicalMatrixCoordinates, sumMap_flatMap ops laws, sumMap_map ops]

private theorem power_product (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (gamma : Field) (left right : Nat) (value : Field) :
    ops.mul (TargetPolynomial.power ops.toOps gamma left)
        (ops.mul (TargetPolynomial.power ops.toOps gamma right) value) =
      ops.mul (TargetPolynomial.power ops.toOps gamma (right + left)) value := by
  calc
    _ = ops.mul (ops.mul (TargetPolynomial.power ops.toOps gamma left)
        (TargetPolynomial.power ops.toOps gamma right)) value := (laws.mul_assoc _ _ _).symm
    _ = ops.mul (ops.mul (TargetPolynomial.power ops.toOps gamma right)
        (TargetPolynomial.power ops.toOps gamma left)) value :=
      congrArg (fun factor => ops.mul factor value) (laws.mul_comm _ _)
    _ = _ := congrArg (fun factor => ops.mul factor value)
      (TargetPolynomial.power_add ops.toOps (shiftLaws ops laws) gamma right left).symm

private theorem power_product_three (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (gamma : Field)
    (source matrix coefficient : Nat) (value : Field) :
    ops.mul (TargetPolynomial.power ops.toOps gamma matrix)
      (ops.mul (TargetPolynomial.power ops.toOps gamma coefficient)
        (ops.mul (TargetPolynomial.power ops.toOps gamma source) value)) =
      ops.mul (TargetPolynomial.power ops.toOps gamma (source + matrix + coefficient)) value := by
  rw [power_product ops laws gamma coefficient source value,
    power_product ops laws gamma matrix (source + coefficient) value]
  rw [show (source + coefficient) + matrix = source + matrix + coefficient by omega]

/-- Coefficient then source order; source gamma weights are applied inside
before multiplication by the coefficient gamma weight. -/
def padTotal (ops : InterpolationOps Field) (powers : Nat → Field)
    (image : PadCoordinate shape → Field) : Field :=
  sumMap ops (canonicalFinIndices shape.coefficientCount) fun coefficient =>
    ops.mul (powers (shape.runningCount * coefficient.val))
      (sumMap ops (canonicalFinIndices shape.runningCount) fun source =>
        ops.mul (powers source.val) (image ⟨source, coefficient⟩))

/-- Matrix then coefficient then source order. These are precisely the
three factors of the existing local MatrixCoordinate gamma exponent. -/
def matrixTotal (ops : InterpolationOps Field) (powers : Nat → Field)
    (image : MatrixCoordinate shape → Field) : Field :=
  sumMap ops (canonicalFinIndices shape.matrixCount) fun matrix =>
    ops.mul (powers (shape.runningCount * matrix.val))
      (sumMap ops (canonicalFinIndices shape.coefficientCount) fun coefficient =>
        ops.mul (powers (shape.runningCount * shape.matrixCount * coefficient.val))
          (sumMap ops (canonicalFinIndices shape.runningCount) fun source =>
            ops.mul (powers source.val) (image ⟨source, matrix, coefficient⟩)))

/-- The factored Pad computation is the unchanged canonical-coordinate sum. -/
theorem padTotal_power (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (gamma : Field)
    (image : PadCoordinate shape → Field) :
    padTotal ops (TargetPolynomial.power ops.toOps gamma) image =
      sumMap ops (canonicalPadCoordinates shape) (fun coordinate =>
        ops.mul (TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent)
          (image coordinate)) := by
  unfold padTotal
  rw [sumPadCoordinates ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro coefficient _
  rw [← FiniteSumAlgebra.sumMap_mul_left ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro source _
  exact power_product ops laws gamma (shape.runningCount * coefficient.val) source.val _

/-- The factored matrix computation preserves every local exponent. The
sole traversal change is justified by the existing finite-sum swap theorem. -/
theorem matrixTotal_power (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (gamma : Field)
    (image : MatrixCoordinate shape → Field) :
    matrixTotal ops (TargetPolynomial.power ops.toOps gamma) image =
      sumMap ops (canonicalMatrixCoordinates shape) (fun coordinate =>
        ops.mul (TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent)
          (image coordinate)) := by
  unfold matrixTotal
  rw [sumMatrixCoordinates ops laws]
  calc
    _ = sumMap ops (canonicalFinIndices shape.matrixCount) (fun matrix =>
        sumMap ops (canonicalFinIndices shape.coefficientCount) (fun coefficient =>
          sumMap ops (canonicalFinIndices shape.runningCount) (fun source =>
            ops.mul (TargetPolynomial.power ops.toOps gamma (shape.runningCount * matrix.val))
              (ops.mul (TargetPolynomial.power ops.toOps gamma
                (shape.runningCount * shape.matrixCount * coefficient.val))
                (ops.mul (TargetPolynomial.power ops.toOps gamma source.val)
                  (image ⟨source, matrix, coefficient⟩)))))) := by
      simp only [← FiniteSumAlgebra.sumMap_mul_left ops laws]
    _ = sumMap ops (canonicalFinIndices shape.coefficientCount) (fun coefficient =>
        sumMap ops (canonicalFinIndices shape.matrixCount) (fun matrix =>
          sumMap ops (canonicalFinIndices shape.runningCount) (fun source =>
            ops.mul (TargetPolynomial.power ops.toOps gamma (shape.runningCount * matrix.val))
              (ops.mul (TargetPolynomial.power ops.toOps gamma
                (shape.runningCount * shape.matrixCount * coefficient.val))
                (ops.mul (TargetPolynomial.power ops.toOps gamma source.val)
                  (image ⟨source, matrix, coefficient⟩)))))) :=
      FiniteSumAlgebra.sumMap_swap ops laws
        (canonicalFinIndices shape.matrixCount) (canonicalFinIndices shape.coefficientCount) _
    _ = _ := by
      apply FiniteSumAlgebra.sumMap_congr
      intro coefficient _
      apply FiniteSumAlgebra.sumMap_congr
      intro matrix _
      apply FiniteSumAlgebra.sumMap_congr
      intro source _
      exact power_product_three ops laws gamma source.val (shape.runningCount * matrix.val)
        (shape.runningCount * shape.matrixCount * coefficient.val) _

end NightstreamFPrime.Export.Stage1.PiCCSGammaAggregation
