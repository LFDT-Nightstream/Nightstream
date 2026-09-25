import Std
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/TargetPolynomial.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Finite SuperNeo v1.1 target polynomial. The target is exactly
`T_K + gamma^(k*d) * T_A`; Pad and matrix coefficients have separate typed
owners and canonical traversals.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetPolynomial

open NightstreamFPrime.Spec.SumCheck

universe uField

/-- Exactly the algebraic laws used by the finite target-shift proof. They are
a strict fragment of the field laws available in the paper model. -/
structure ShiftLaws
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field) : Prop where
  one_mul : forall value, ops.mul ops.one value = value
  mul_assoc : forall left middle right,
    ops.mul (ops.mul left middle) right =
      ops.mul left (ops.mul middle right)
  mul_zero : forall value, ops.mul value ops.zero = ops.zero
  mul_add : forall left middle right,
    ops.mul left (ops.add middle right) =
      ops.add (ops.mul left middle) (ops.mul left right)

/-- Finite exponentiation using only the verifier-selected operations. -/
def power
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field)
    (value : Field) : Nat -> Field
  | 0 => ops.one
  | exponent + 1 => ops.mul value (power ops value exponent)

/-- Powers split exactly over addition under the stated minimal laws. -/
theorem power_add
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ShiftLaws ops)
    (value : Field)
    (left right : Nat) :
    power ops value (left + right) =
      ops.mul (power ops value left) (power ops value right) := by
  induction left with
  | zero =>
      simpa [power] using (laws.one_mul (power ops value right)).symm
  | succ left inductionHypothesis =>
      rw [Nat.succ_add]
      simp only [power]
      rw [inductionHypothesis]
      exact (laws.mul_assoc value (power ops value left)
        (power ops value right)).symm

/-- One target coefficient for every v1.1 Pad and matrix coordinate. -/
structure TargetCoefficients
    (Field : Type uField)
    (shape : Shape) where
  pad : PadCoordinate shape -> Field
  matrix : MatrixCoordinate shape -> Field

/-- One `T_K` term. -/
def padTerm
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : TargetCoefficients Field shape)
    (gamma : Field)
    (coordinate : PadCoordinate shape) : Field :=
  ops.mul
    (power ops gamma coordinate.localGammaExponent)
    (coefficients.pad coordinate)

/-- One local `T_A` term before its `k*d` shift. -/
def matrixLocalTerm
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : TargetCoefficients Field shape)
    (gamma : Field)
    (coordinate : MatrixCoordinate shape) : Field :=
  ops.mul
    (power ops gamma coordinate.localGammaExponent)
    (coefficients.matrix coordinate)

/-- One absolute matrix target term at `k*d + I_A`. -/
def matrixTerm
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : TargetCoefficients Field shape)
    (gamma : Field)
    (coordinate : MatrixCoordinate shape) : Field :=
  ops.mul
    (power ops gamma coordinate.gammaExponent)
    (coefficients.matrix coordinate)

private def sumTerms
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field)
    (values : List Field) : Field :=
  values.foldr ops.add ops.zero

/-- The unshifted Pad target `T_K`. -/
def evaluatePad
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : TargetCoefficients Field shape)
    (gamma : Field) : Field :=
  sumTerms ops <|
    (canonicalPadCoordinates shape).map fun coordinate =>
      padTerm ops coefficients gamma coordinate

/-- The local matrix target `T_A` before the `k*d` shift. -/
def evaluateMatrixLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : TargetCoefficients Field shape)
    (gamma : Field) : Field :=
  sumTerms ops <|
    (canonicalMatrixCoordinates shape).map fun coordinate =>
      matrixLocalTerm ops coefficients gamma coordinate

/-- The shifted matrix target `gamma^(k*d) * T_A`. -/
def evaluateMatrix
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : TargetCoefficients Field shape)
    (gamma : Field) : Field :=
  sumTerms ops <|
    (canonicalMatrixCoordinates shape).map fun coordinate =>
      matrixTerm ops coefficients gamma coordinate

/-- Exact v1.1 claimed sum `T_K + gamma^(k*d) * T_A`. -/
def evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : TargetCoefficients Field shape)
    (gamma : Field) : Field :=
  ops.add (evaluatePad ops coefficients gamma)
    (evaluateMatrix ops coefficients gamma)

theorem evaluatePad_eq_foldr
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : TargetCoefficients Field shape)
    (gamma : Field) :
    evaluatePad ops coefficients gamma =
      ((canonicalPadCoordinates shape).map fun coordinate =>
        padTerm ops coefficients gamma coordinate).foldr ops.add ops.zero := by
  rfl

theorem evaluateMatrixLocal_eq_foldr
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : TargetCoefficients Field shape)
    (gamma : Field) :
    evaluateMatrixLocal ops coefficients gamma =
      ((canonicalMatrixCoordinates shape).map fun coordinate =>
        matrixLocalTerm ops coefficients gamma coordinate).foldr
          ops.add ops.zero := by
  rfl

private theorem matrixTerm_eq_shift_mul_local
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ShiftLaws ops)
    (coefficients : TargetCoefficients Field shape)
    (gamma : Field)
    (coordinate : MatrixCoordinate shape) :
    matrixTerm ops coefficients gamma coordinate =
      ops.mul
        (power ops gamma shape.padEvaluationCount)
        (matrixLocalTerm ops coefficients gamma coordinate) := by
  unfold matrixTerm matrixLocalTerm MatrixCoordinate.gammaExponent
  change ops.mul
      (power ops gamma
        (shape.padEvaluationCount + coordinate.localGammaExponent))
      (coefficients.matrix coordinate) = _
  rw [power_add ops laws gamma shape.padEvaluationCount
    coordinate.localGammaExponent]
  exact laws.mul_assoc
    (power ops gamma shape.padEvaluationCount)
    (power ops gamma coordinate.localGammaExponent)
    (coefficients.matrix coordinate)

private theorem sumTerms_map_mul_left
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ShiftLaws ops)
    (factor : Field) : forall values : List Field,
    sumTerms ops (values.map (ops.mul factor)) =
      ops.mul factor (sumTerms ops values)
  | [] => by
      change ops.zero = ops.mul factor ops.zero
      exact (laws.mul_zero factor).symm
  | value :: values => by
      change ops.add (ops.mul factor value)
          (sumTerms ops (values.map (ops.mul factor))) =
        ops.mul factor (ops.add value (sumTerms ops values))
      rw [sumTerms_map_mul_left ops laws factor values]
      exact (laws.mul_add factor value (sumTerms ops values)).symm

private theorem matrixTerms_eq_map_shiftedLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ShiftLaws ops)
    (coefficients : TargetCoefficients Field shape)
    (gamma : Field) :
    (canonicalMatrixCoordinates shape).map
        (matrixTerm ops coefficients gamma) =
      ((canonicalMatrixCoordinates shape).map
        (matrixLocalTerm ops coefficients gamma)).map
          (ops.mul (power ops gamma shape.padEvaluationCount)) := by
  rw [List.map_map]
  apply List.map_congr_left
  intro coordinate _
  exact matrixTerm_eq_shift_mul_local ops laws coefficients gamma coordinate

/-- Exact v1.1 matrix-target shift theorem. -/
theorem evaluateMatrix_eq_shift_mul_evaluateMatrixLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ShiftLaws ops)
    (coefficients : TargetCoefficients Field shape)
    (gamma : Field) :
    evaluateMatrix ops coefficients gamma =
      ops.mul
        (power ops gamma shape.padEvaluationCount)
        (evaluateMatrixLocal ops coefficients gamma) := by
  unfold evaluateMatrix evaluateMatrixLocal
  rw [matrixTerms_eq_map_shiftedLocal ops laws coefficients gamma]
  exact sumTerms_map_mul_left ops laws
    (power ops gamma shape.padEvaluationCount)
    ((canonicalMatrixCoordinates shape).map
      (matrixLocalTerm ops coefficients gamma))

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetPolynomial
