import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedJointIdentity
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra

/-!
Finite signed gamma-coefficient polynomial for paper-level joint `Pi_CCS`.

Protocol: SuperNeo v1.2 `Pi_CCS` (Section 7.3 / Appendix B.2).
Phase: pre-SumCheck alpha specialization and gamma mixing.
Constraint family: exact constant-first serialization of Pad, matrix, CCS,
and norm residual blocks.

Owns: the four explicit coefficient lists, their exact block lengths, the
constant-first finite polynomial, and the theorem that executable Horner
evaluation equals the independently derived signed joint identity.

Does not own: construction of concrete CCS/norm/ring tables, the paper-source
target audit, SumCheck initial or terminal acceptance, root-counting
probability, Fiat--Shamir, Rust, R1CS, or counts.

Emits constraints: no.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial

universe uField uIndex

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.SumCheck

def ccsValues
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) : List Field :=
  (canonicalFinIndices shape.freshCount).map fun source =>
    (data.ccs source).equalityWeightedSum ops alpha

def normValues
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) : List Field :=
  (canonicalFinIndices shape.sourceCount).map fun source =>
    (data.norm source).equalityWeightedSum ops alpha

def padValues
    {Field : Type uField}
    {shape : Shape}
  (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) : List Field :=
  (canonicalPadCoordinates shape).map fun coordinate =>
    ops.sub
      (data.claimedPadCoefficient coordinate)
      ((data.padImage coordinate).equalityWeightedSum
        ops data.priorPoint)

def matrixValues
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) : List Field :=
  (canonicalMatrixCoordinates shape).map fun coordinate =>
    ops.sub
      (data.claimedMatrixCoefficient coordinate)
      ((data.matrixImage coordinate).equalityWeightedSum
        ops data.priorPoint)

def ccsCoefficients
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) : List Field :=
  (ccsValues ops data alpha).map ops.neg

def normCoefficients
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) : List Field :=
  (normValues ops data alpha).map ops.neg

def padCoefficients
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) : List Field :=
  padValues ops data

def matrixCoefficients
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) : List Field :=
  matrixValues ops data

def coefficients
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) : List Field :=
  padCoefficients ops data ++
    (matrixCoefficients ops data ++
      (ccsCoefficients ops data alpha ++
        normCoefficients ops data alpha))

def polynomial
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) :
    SumCheck.Finite.Message Field where
  coefficients := coefficients ops data alpha

theorem ccsCoefficients_length
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) :
    (ccsCoefficients ops data alpha).length = shape.freshCount := by
  simp [ccsCoefficients, ccsValues, canonicalFinIndices_length]

theorem normCoefficients_length
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) :
    (normCoefficients ops data alpha).length = shape.sourceCount := by
  simp [normCoefficients, normValues, canonicalFinIndices_length]

theorem padCoefficients_length
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) :
    (padCoefficients ops data).length = shape.padEvaluationCount := by
  simp [padCoefficients, padValues, canonicalPadCoordinates_length]

theorem matrixCoefficients_length
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) :
    (matrixCoefficients ops data).length = shape.matrixEvaluationCount := by
  simp [matrixCoefficients, matrixValues, canonicalMatrixCoordinates_length]

theorem coefficients_length
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) :
    (coefficients ops data alpha).length =
      shape.jointCoefficientCount := by
  simp only [coefficients, List.length_append,
    padCoefficients_length, matrixCoefficients_length,
    ccsCoefficients_length, normCoefficients_length,
    Shape.jointCoefficientCount, Shape.constraintOffset]
  omega

theorem polynomial_degreeUpperBound
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) :
    (polynomial ops data alpha).degreeUpperBound =
      shape.jointCoefficientCount - 1 := by
  simp [polynomial, SumCheck.Finite.Message.degreeUpperBound,
    coefficients_length]

private theorem neg_zero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) :
    ops.neg ops.zero = ops.zero := by
  have inverse := laws.add_neg ops.zero
  simpa only [laws.zero_add] using inverse

private theorem mul_neg
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left right : Field) :
    ops.mul left (ops.neg right) = ops.neg (ops.mul left right) := by
  rw [laws.mul_comm left (ops.neg right), laws.neg_mul,
    laws.mul_comm right left]

private def positionalSumFrom
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (gamma : Field) : Nat -> List Field -> Field
  | _, [] => ops.zero
  | exponent, value :: values =>
      ops.add
        (SignedJointIdentity.gammaTerm ops gamma exponent value)
        (positionalSumFrom ops gamma (exponent + 1) values)

private theorem power_mul_evaluateCoefficients_eq_positionalSumFrom
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (gamma : Field) : forall (exponent : Nat) (values : List Field),
    ops.mul (TargetPolynomial.power ops.toOps gamma exponent)
        (SumCheck.Finite.Message.evaluateCoefficients
          ops.toOps gamma values) =
      positionalSumFrom ops gamma exponent values
  | _, [] => by
      simp [SumCheck.Finite.Message.evaluateCoefficients,
        positionalSumFrom, laws.mul_zero]
  | exponent, value :: values => by
      simp only [SumCheck.Finite.Message.evaluateCoefficients,
        positionalSumFrom]
      rw [laws.left_distrib]
      rw [← laws.mul_assoc]
      have nextPower :
          ops.mul
              (TargetPolynomial.power ops.toOps gamma exponent)
              gamma =
            TargetPolynomial.power ops.toOps gamma (exponent + 1) := by
        simp only [TargetPolynomial.power]
        exact laws.mul_comm _ _
      rw [nextPower]
      rw [power_mul_evaluateCoefficients_eq_positionalSumFrom
        ops laws gamma (exponent + 1) values]
      rfl

private theorem evaluateCoefficients_eq_positionalSumFrom
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (gamma : Field)
    (values : List Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma values =
      positionalSumFrom ops gamma 0 values := by
  have expanded :=
    power_mul_evaluateCoefficients_eq_positionalSumFrom
      ops laws gamma 0 values
  simpa only [TargetPolynomial.power, laws.one_mul] using expanded

private theorem positionalSumFrom_map_eq_indexed
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (gamma : Field)
    (indices : List Index)
    (position : Index -> Nat)
    (value : Index -> Field)
    (offset : Nat)
    (positions :
      indices.map position = List.range' offset indices.length) :
    positionalSumFrom ops gamma offset (indices.map value) =
      BooleanTable.finiteSum ops
        (indices.map fun index =>
          SignedJointIdentity.gammaTerm ops gamma
            (position index) (value index)) := by
  induction indices generalizing offset with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.map_cons, List.length_cons, List.range'_succ,
        List.cons.injEq] at positions
      rcases positions with ⟨headPosition, tailPositions⟩
      simp only [List.map_cons, positionalSumFrom,
        BooleanTable.finiteSum]
      rw [headPosition]
      congr 1
      exact inductionHypothesis (offset + 1) tailPositions

theorem evaluate_map_eq_indexed
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (gamma : Field)
    (indices : List Index)
    (position : Index -> Nat)
    (value : Index -> Field)
    (positions : indices.map position = List.range' 0 indices.length) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (indices.map value) =
      BooleanTable.finiteSum ops
        (indices.map fun index =>
          SignedJointIdentity.gammaTerm ops gamma
            (position index) (value index)) := by
  rw [evaluateCoefficients_eq_positionalSumFrom ops laws]
  exact positionalSumFrom_map_eq_indexed
    ops gamma indices position value 0 positions

theorem evaluate_map_neg
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (gamma : Field) : forall values : List Field,
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (values.map ops.neg) =
      ops.neg
        (SumCheck.Finite.Message.evaluateCoefficients
          ops.toOps gamma values)
  | [] => by
      change ops.zero = ops.neg ops.zero
      exact (neg_zero ops laws).symm
  | value :: values => by
      simp only [List.map_cons,
        SumCheck.Finite.Message.evaluateCoefficients]
      rw [evaluate_map_neg ops laws gamma values]
      rw [mul_neg ops laws]
      exact (laws.neg_add _ _).symm

theorem evaluate_append
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (gamma : Field) : forall left right : List Field,
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (left ++ right) =
      ops.add
        (SumCheck.Finite.Message.evaluateCoefficients
          ops.toOps gamma left)
        (ops.mul
          (TargetPolynomial.power ops.toOps gamma left.length)
          (SumCheck.Finite.Message.evaluateCoefficients
            ops.toOps gamma right))
  | [], right => by
      simp only [List.nil_append, List.length_nil,
        SumCheck.Finite.Message.evaluateCoefficients,
        TargetPolynomial.power, laws.one_mul, laws.zero_add]
  | value :: values, right => by
      simp only [List.cons_append,
        SumCheck.Finite.Message.evaluateCoefficients, List.length_cons]
      rw [evaluate_append ops laws gamma values right]
      rw [laws.left_distrib]
      rw [← laws.mul_assoc]
      simp only [TargetPolynomial.power]
      exact (laws.add_assoc _ _ _).symm

private theorem canonicalFinPositions (count : Nat) :
    (canonicalFinIndices count).map (fun index => index.val) =
      List.range' 0 (canonicalFinIndices count).length := by
  simpa [List.range_eq_range', canonicalFinIndices_length] using
    canonicalFinIndices_values count

theorem evaluate_canonicalFinMap_eq_gammaSum
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (gamma : Field)
    (count : Nat)
    (value : Fin count → Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        ((canonicalFinIndices count).map value) =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices count) fun index =>
          SignedJointIdentity.gammaTerm ops gamma index.val (value index) := by
  unfold FiniteSumAlgebra.sumMap
  exact evaluate_map_eq_indexed ops laws gamma
    (canonicalFinIndices count) (fun index => index.val) value
    (canonicalFinPositions count)

private theorem canonicalPadPositions (shape : Shape) :
    (canonicalPadCoordinates shape).map PadCoordinate.localGammaExponent =
      List.range' 0 (canonicalPadCoordinates shape).length := by
  simpa [List.range_eq_range', canonicalPadCoordinates_length] using
    canonicalPadCoordinates_localGammaExponents shape

private theorem canonicalMatrixPositions (shape : Shape) :
    (canonicalMatrixCoordinates shape).map
        MatrixCoordinate.localGammaExponent =
      List.range' 0 (canonicalMatrixCoordinates shape).length := by
  simpa [List.range_eq_range', canonicalMatrixCoordinates_length] using
    canonicalMatrixCoordinates_localGammaExponents shape

private theorem ccsValues_evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (ccsValues ops data alpha) =
      SignedJointIdentity.ccsResidualBlock ops data alpha gamma := by
  unfold ccsValues
  change SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
      ((canonicalFinIndices shape.freshCount).map fun source =>
        (data.ccs source).equalityWeightedSum ops alpha) =
    BooleanTable.finiteSum ops
      ((canonicalFinIndices shape.freshCount).map fun source =>
        SignedJointIdentity.gammaTerm ops gamma source.val
          ((data.ccs source).equalityWeightedSum ops alpha))
  exact evaluate_map_eq_indexed ops laws gamma
    (canonicalFinIndices shape.freshCount)
    (fun source => source.val)
    (fun source => (data.ccs source).equalityWeightedSum ops alpha)
    (canonicalFinPositions shape.freshCount)

private theorem normValues_evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (normValues ops data alpha) =
      SignedJointIdentity.normResidualLocal ops data alpha gamma := by
  unfold normValues
  change SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
      ((canonicalFinIndices shape.sourceCount).map fun source =>
        (data.norm source).equalityWeightedSum ops alpha) =
    BooleanTable.finiteSum ops
      ((canonicalFinIndices shape.sourceCount).map fun source =>
        SignedJointIdentity.gammaTerm ops gamma source.val
          ((data.norm source).equalityWeightedSum ops alpha))
  exact evaluate_map_eq_indexed ops laws gamma
    (canonicalFinIndices shape.sourceCount)
    (fun source => source.val)
    (fun source => (data.norm source).equalityWeightedSum ops alpha)
    (canonicalFinPositions shape.sourceCount)

private theorem padValues_evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (padValues ops data) =
      SignedJointIdentity.padResidualLocal ops data gamma := by
  unfold padValues
  change SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
      ((canonicalPadCoordinates shape).map fun coordinate =>
        ops.sub
          (data.claimedPadCoefficient coordinate)
            ((data.padImage coordinate).equalityWeightedSum
              ops data.priorPoint)) =
    BooleanTable.finiteSum ops
      ((canonicalPadCoordinates shape).map fun coordinate =>
        SignedJointIdentity.gammaTerm ops gamma
          coordinate.localGammaExponent
          (ops.sub
            (data.claimedPadCoefficient coordinate)
            ((data.padImage coordinate).equalityWeightedSum
              ops data.priorPoint)))
  exact evaluate_map_eq_indexed ops laws gamma
    (canonicalPadCoordinates shape)
    PadCoordinate.localGammaExponent
    (fun coordinate =>
      ops.sub
        (data.claimedPadCoefficient coordinate)
        ((data.padImage coordinate).equalityWeightedSum
          ops data.priorPoint))
    (canonicalPadPositions shape)

private theorem matrixValues_evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (matrixValues ops data) =
      SignedJointIdentity.matrixResidualLocal ops data gamma := by
  unfold matrixValues
  change SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
      ((canonicalMatrixCoordinates shape).map fun coordinate =>
        ops.sub
          (data.claimedMatrixCoefficient coordinate)
            ((data.matrixImage coordinate).equalityWeightedSum
              ops data.priorPoint)) =
    BooleanTable.finiteSum ops
      ((canonicalMatrixCoordinates shape).map fun coordinate =>
        SignedJointIdentity.gammaTerm ops gamma
          coordinate.localGammaExponent
          (ops.sub
            (data.claimedMatrixCoefficient coordinate)
            ((data.matrixImage coordinate).equalityWeightedSum
              ops data.priorPoint)))
  exact evaluate_map_eq_indexed ops laws gamma
    (canonicalMatrixCoordinates shape)
    MatrixCoordinate.localGammaExponent
    (fun coordinate =>
      ops.sub
        (data.claimedMatrixCoefficient coordinate)
        ((data.matrixImage coordinate).equalityWeightedSum
          ops data.priorPoint))
    (canonicalMatrixPositions shape)

private theorem ccsCoefficients_evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (ccsCoefficients ops data alpha) =
      ops.neg (SignedJointIdentity.ccsResidualBlock
        ops data alpha gamma) := by
  unfold ccsCoefficients
  rw [evaluate_map_neg ops laws]
  rw [ccsValues_evaluate ops laws]

private theorem normCoefficients_evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (normCoefficients ops data alpha) =
      ops.neg (SignedJointIdentity.normResidualLocal
        ops data alpha gamma) := by
  unfold normCoefficients
  rw [evaluate_map_neg ops laws]
  rw [normValues_evaluate ops laws]

private theorem padCoefficients_evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (padCoefficients ops data) =
      SignedJointIdentity.padResidualLocal ops data gamma := by
  unfold padCoefficients
  exact padValues_evaluate ops laws data gamma

private theorem matrixCoefficients_evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (matrixCoefficients ops data) =
      SignedJointIdentity.matrixResidualLocal ops data gamma := by
  unfold matrixCoefficients
  exact matrixValues_evaluate ops laws data gamma

theorem evaluateCoefficients_eq_signedResidualBlocks
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (coefficients ops data alpha) =
      SignedJointIdentity.signedResidualBlocks ops data alpha gamma := by
  unfold coefficients
  rw [evaluate_append ops laws]
  rw [evaluate_append ops laws]
  rw [evaluate_append ops laws]
  rw [padCoefficients_length, matrixCoefficients_length,
    ccsCoefficients_length]
  rw [padCoefficients_evaluate ops laws]
  rw [matrixCoefficients_evaluate ops laws]
  rw [ccsCoefficients_evaluate ops laws]
  rw [normCoefficients_evaluate ops laws]
  let powerLaws : TargetPolynomial.ShiftLaws ops.toOps :=
    { one_mul := laws.one_mul
      mul_assoc := laws.mul_assoc
      mul_zero := laws.mul_zero
      mul_add := laws.left_distrib }
  have blockPower :
      TargetPolynomial.power ops.toOps gamma shape.constraintOffset =
        ops.mul
          (TargetPolynomial.power ops.toOps gamma shape.padEvaluationCount)
          (TargetPolynomial.power ops.toOps gamma
            shape.matrixEvaluationCount) := by
    simpa [Shape.constraintOffset] using
      TargetPolynomial.power_add ops.toOps powerLaws gamma
        shape.padEvaluationCount shape.matrixEvaluationCount
  have negConstraint :
      ops.mul
          (TargetPolynomial.power ops.toOps gamma shape.constraintOffset)
          (ops.add
            (ops.neg (SignedJointIdentity.ccsResidualBlock
              ops data alpha gamma))
            (ops.mul
              (TargetPolynomial.power ops.toOps gamma shape.freshCount)
              (ops.neg (SignedJointIdentity.normResidualLocal
                ops data alpha gamma)))) =
        ops.neg
          (ops.mul
            (TargetPolynomial.power ops.toOps gamma shape.constraintOffset)
            (ops.add
              (SignedJointIdentity.ccsResidualBlock ops data alpha gamma)
              (ops.mul
                (TargetPolynomial.power ops.toOps gamma shape.freshCount)
                (SignedJointIdentity.normResidualLocal
                  ops data alpha gamma)))) := by
    rw [mul_neg ops laws
      (TargetPolynomial.power ops.toOps gamma shape.freshCount)]
    rw [← laws.neg_add]
    rw [mul_neg ops laws]
  unfold SignedJointIdentity.signedResidualBlocks
    SignedJointIdentity.matrixResidualBlock
    SignedJointIdentity.constraintResidualBlock
    SignedJointIdentity.normResidualBlock
    SignedJointIdentity.gammaTerm
  rw [laws.left_distrib]
  rw [← laws.mul_assoc]
  rw [← blockPower]
  rw [negConstraint]
  unfold Shape.matrixEvaluationOffset
  rfl

theorem evaluate_eq_signedResidualBlocks
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    (polynomial ops data alpha).evaluate ops.toOps gamma =
      SignedJointIdentity.signedResidualBlocks ops data alpha gamma := by
  exact evaluateCoefficients_eq_signedResidualBlocks
    ops laws data alpha gamma

theorem paperDifference_eq_evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    SignedJointIdentity.paperDifference ops data alpha gamma =
      (polynomial ops data alpha).evaluate ops.toOps gamma := by
  rw [SignedJointIdentity.paperDifference_eq_signedResidualBlocks
    ops laws data alpha gamma]
  exact (evaluate_eq_signedResidualBlocks
    ops laws data alpha gamma).symm

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial
