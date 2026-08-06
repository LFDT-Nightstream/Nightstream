import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedJointIdentity
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra

/-!
Finite signed gamma-coefficient polynomial for paper-level joint `Pi_CCS`.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: pre-SumCheck alpha specialization and gamma mixing.
Constraint family: exact constant-first serialization of the signed CCS, norm,
and carried-evaluation residual blocks.

Owns: the three explicit coefficient lists, their exact block lengths, the
constant-first finite polynomial, and the theorem that executable Horner
evaluation equals the independently derived signed joint identity.

Does not own: construction of concrete CCS/norm/ring tables, the paper-source
target audit, SumCheck initial or terminal acceptance, root-counting
probability, Fiat--Shamir, Rust, R1CS, or counts.

Emits constraints: no.

Authority boundary: coefficient values come only from explicit `JointData`
tables and claims. Block signs, offsets, lengths, and Horner positions are
derived here. No evaluator callback, caller-supplied degree, claimed identity,
Rust trace, or existing circuit enters the theorem.

| Protocol | Phase | Coefficient family | Exact positions / meaning |
|---|---|---|---|
| `Pi_CCS` | gamma serialization | CCS | `0 .. K-1`, negative alpha-specialized CCS residuals |
| `Pi_CCS` | gamma serialization | norm | `K .. 2K+k-1`, negative alpha-specialized norm residuals |
| `Pi_CCS` | gamma serialization | carried | `2K+k .. 2K+k+ktd-1`, claimed minus derived evaluations |
| `Pi_CCS` | executable evaluation | all blocks | Horner evaluation equals `T_abs - sum_x Q` for every `alpha,gamma` |
| shared | canonical indexed evaluation | canonical `Fin n` coefficients evaluate as `sum_i gamma^i c_i` | `evaluate_canonicalFinMap_eq_gammaSum` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial

universe uField uIndex

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck

/-- Unsigned CCS values before the sign forced by `T_abs - sum Q`. -/
def ccsValues
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) : List Field :=
  (canonicalFinIndices shape.freshCount).map fun source =>
    (data.ccs source).equalityWeightedSum ops alpha

/-- Unsigned norm values before the sign and `K`-position shift. -/
def normValues
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) : List Field :=
  (canonicalFinIndices shape.sourceCount).map fun source =>
    (data.norm source).equalityWeightedSum ops alpha

/-- Claimed-minus-derived carried values before the `2K+k` position shift. -/
def carriedValues
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) : List Field :=
  (canonicalCarriedCoordinates shape).map fun coordinate =>
    ops.sub
      (data.claimedCoefficient coordinate)
      ((data.carriedImage coordinate).equalityWeightedSum
        ops data.priorPoint)

/-- Negative CCS coefficient block. -/
def ccsCoefficients
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) : List Field :=
  (ccsValues ops data alpha).map ops.neg

/-- Negative norm coefficient block. Its list position supplies the `K`
shift when concatenated after the CCS block. -/
def normCoefficients
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) : List Field :=
  (normValues ops data alpha).map ops.neg

/-- Positive claimed-minus-derived carried coefficient block. Its list
position supplies the `2K+k` shift. -/
def carriedCoefficients
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) : List Field :=
  carriedValues ops data

/-- Exact constant-first signed gamma coefficients. -/
def coefficients
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) : List Field :=
  ccsCoefficients ops data alpha ++
    normCoefficients ops data alpha ++
    carriedCoefficients ops data

/-- Finite executable polynomial with degree metadata derived from its list. -/
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

theorem carriedCoefficients_length
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) :
    (carriedCoefficients ops data).length =
      shape.carriedEvaluationCount := by
  simp [carriedCoefficients, carriedValues,
    canonicalCarriedCoordinates_length]

/-- The list contains exactly the three declared blocks and no hidden
coefficient family. -/
theorem coefficients_length
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) :
    (coefficients ops data alpha).length =
      shape.jointCoefficientCount := by
  simp only [coefficients, List.length_append,
    ccsCoefficients_length, normCoefficients_length,
    carriedCoefficients_length, Shape.jointCoefficientCount]

/-- Degree is derived from the exact signed coefficient-list length. This is
not a claim that the last coefficient is nonzero. -/
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

/-- Explicit monomial sum beginning at one absolute exponent. -/
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

private theorem evaluate_map_eq_indexed
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
      simp only [List.nil_append,
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

/-- A constant-first coefficient list in canonical `Fin count` order is
exactly the explicit paper-relative gamma sum. This shared theorem makes the
coefficient polynomial visible to split-protocol soundness proofs. -/
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

private theorem canonicalCarriedPositions (shape : Shape) :
    (canonicalCarriedCoordinates shape).map
        CarriedCoordinate.localGammaExponent =
      List.range' 0 (canonicalCarriedCoordinates shape).length := by
  simpa [List.range_eq_range', canonicalCarriedCoordinates_length] using
    canonicalCarriedCoordinates_localGammaExponents shape

private theorem finiteSum_eq_foldr
    {Field : Type uField}
    (ops : InterpolationOps Field) :
    ∀ values : List Field,
      BooleanTable.finiteSum ops values = values.foldr ops.add ops.zero
  | [] => rfl
  | _ :: values => by
      simp only [BooleanTable.finiteSum, List.foldr]
      rw [finiteSum_eq_foldr ops values]

/-- A carried-coordinate list in the canonical paper order evaluates exactly
as the local carried-target polynomial.  This is the list-level bridge used by
the Lean-owned verifier encoding: no caller supplies exponents or a reordered
coefficient vector. -/
theorem evaluate_canonicalCarriedMap_eq_targetLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (gamma : Field)
    (coefficients : TargetPolynomial.CarriedTargetCoefficients Field shape) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        ((canonicalCarriedCoordinates shape).map coefficients.coefficient) =
      TargetPolynomial.evaluateLocal ops.toOps coefficients gamma := by
  rw [TargetPolynomial.evaluateLocal_eq_foldr]
  rw [evaluate_map_eq_indexed ops laws gamma
    (canonicalCarriedCoordinates shape)
    CarriedCoordinate.localGammaExponent coefficients.coefficient
    (canonicalCarriedPositions shape)]
  rw [finiteSum_eq_foldr]
  rfl

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

private theorem carriedValues_evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (carriedValues ops data) =
      SignedJointIdentity.carriedResidualLocal ops data gamma := by
  unfold carriedValues
  change SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
      ((canonicalCarriedCoordinates shape).map fun coordinate =>
        ops.sub
          (data.claimedCoefficient coordinate)
            ((data.carriedImage coordinate).equalityWeightedSum
              ops data.priorPoint)) =
    BooleanTable.finiteSum ops
      ((canonicalCarriedCoordinates shape).map fun coordinate =>
        SignedJointIdentity.gammaTerm ops gamma
          coordinate.localGammaExponent
          (ops.sub
            (data.claimedCoefficient coordinate)
            ((data.carriedImage coordinate).equalityWeightedSum
              ops data.priorPoint)))
  exact evaluate_map_eq_indexed ops laws gamma
    (canonicalCarriedCoordinates shape)
    CarriedCoordinate.localGammaExponent
    (fun coordinate =>
      ops.sub
        (data.claimedCoefficient coordinate)
        ((data.carriedImage coordinate).equalityWeightedSum
          ops data.priorPoint))
    (canonicalCarriedPositions shape)

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

private theorem carriedCoefficients_evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field) :
    SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
        (carriedCoefficients ops data) =
      SignedJointIdentity.carriedResidualLocal ops data gamma := by
  unfold carriedCoefficients
  exact carriedValues_evaluate ops laws data gamma

/-- Executable Horner evaluation of the exact signed coefficient list equals
the independently derived signed residual blocks. -/
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
  rw [List.length_append]
  rw [ccsCoefficients_length, normCoefficients_length]
  rw [ccsCoefficients_evaluate ops laws]
  rw [normCoefficients_evaluate ops laws]
  rw [carriedCoefficients_evaluate ops laws]
  unfold SignedJointIdentity.signedResidualBlocks
  unfold SignedJointIdentity.normResidualBlock
  unfold SignedJointIdentity.carriedResidualBlock
  unfold SignedJointIdentity.gammaTerm
  rw [mul_neg ops laws]
  unfold Shape.carriedEvaluationOffset
  exact laws.add_assoc _ _ _

/-- The finite message wrapper has exactly the same evaluation. -/
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

/-- End-to-end table-level coefficient identity:

`T_abs(gamma) - sum_x Q(x, alpha, gamma)` is exactly executable Horner
evaluation of the signed constant-first coefficient list.

This remains model-level and does not instantiate production tables or a
SumCheck verifier. -/
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

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial
