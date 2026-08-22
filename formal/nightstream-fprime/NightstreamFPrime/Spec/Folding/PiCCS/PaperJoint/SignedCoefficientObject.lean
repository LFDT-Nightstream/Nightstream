import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TableResiduals

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/SignedCoefficientObject.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Unsampled signed coefficient object for paper-level joint `Pi_CCS`.

Protocol: SuperNeo v1.1 `Pi_CCS` (Section 7.3 / Appendix B.2).
Phase: alpha/gamma compression boundary before SumCheck.
Constraint family: finite Pad, matrix, CCS, and norm residuals before
the verifier samples `alpha` and `gamma`.

Owns: a finite signed coefficient type, its canonical four-block object,
coefficient truth, exact alpha specialization into the executable gamma
polynomial, table-obligation equivalence, and the deterministic signed
mixing-root event.

Does not own: construction of concrete CCS/norm/ring tables, target-convention
approval, root-counting probability, SumCheck round or terminal truth,
Fiat--Shamir, Rust, R1CS, or counts.

Emits constraints: no.

Authority boundary: negative alpha coefficients contain the verifier-derived
finite canonical alpha polynomial, not a function-valued evaluator. Scalars
are explicit Pad and matrix claimed-minus-derived residuals. Signs, block order,
specialization, and the bad-root event are derived here; no prover-supplied
degree, identity, or implementation artifact is accepted.

| Protocol | Phase | Coefficient owner | Exact guarantee |
|---|---|---|---|
| `Pi_CCS` | before alpha | negative CCS / norm alpha polynomials | finite canonical coefficient vectors |
| `Pi_CCS` | before alpha | Pad and matrix scalars | claimed minus derived prior-point evaluations |
| `Pi_CCS` | alpha specialization | all four blocks | exact equality with the executable signed gamma coefficient list |
| `Pi_CCS` | semantic truth | explicit residual tables | coefficient truth iff every table obligation holds |
| `Pi_CCS` | gamma sampling | nonzero signed object | sampled zero is the named `MixingRoot` event |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedCoefficientObject

universe uField

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.SumCheck

/-- One unsampled signed gamma coefficient. CCS and norm entries retain their
finite alpha coefficient vectors; evaluation entries are alpha-free scalars. -/
inductive Coefficient
    (Field : Type uField)
    (shape : Shape) where
  | negativeAlpha
      (polynomial : AlphaPolynomial Field (canonicalAlphaBasis shape))
  | scalar (value : Field)

namespace Coefficient

/-- Coefficient-level zero before alpha sampling. The negative sign cannot
change whether an alpha polynomial is identically zero. -/
def Zero
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field) : Coefficient Field shape -> Prop
  | .negativeAlpha polynomial => polynomial.CoefficientZero ops.toOps
  | .scalar value => value = ops.zero

/-- Verifier-owned alpha specialization. -/
def specialize
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (alpha : CubePoint Field shape.cubeVariables) :
    Coefficient Field shape -> Field
  | .negativeAlpha polynomial =>
      ops.neg (polynomial.evaluate ops.toOps alpha)
  | .scalar value => value

end Coefficient

/-- The exact residual tables underlying one explicit `JointData`. -/
def toTableResidualData
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) :
    TableResidualData Field shape where
  ccs := data.ccs
  norm := data.norm
  padEvaluation := fun coordinate =>
    ops.sub
      (data.claimedPadCoefficient coordinate)
      ((data.padImage coordinate).equalityWeightedSum
        ops data.priorPoint)
  matrixEvaluation := fun coordinate =>
    ops.sub
      (data.claimedMatrixCoefficient coordinate)
      ((data.matrixImage coordinate).equalityWeightedSum
        ops data.priorPoint)

/-- Unsigned residual carrier reused only for its finite alpha polynomials and
table-level truth theorem. Signs are applied by `coefficients` below. -/
def residuals
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) :
    Residuals Field shape (canonicalAlphaBasis shape) :=
  (toTableResidualData ops data).toResiduals ops

/-- Exact constant-first unsampled signed coefficient object. -/
def coefficients
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) :
    List (Coefficient Field shape) :=
  (residuals ops data).padEvaluation.map Coefficient.scalar ++
    ((residuals ops data).matrixEvaluation.map Coefficient.scalar ++
      ((residuals ops data).ccs.map Coefficient.negativeAlpha ++
        (residuals ops data).norm.map Coefficient.negativeAlpha))

/-- Every unsampled coefficient is identically zero. -/
def CoefficientTruth
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) : Prop :=
  forall coefficient,
    coefficient ∈ coefficients ops data -> coefficient.Zero ops

/-- Alpha specialization of the complete signed object. -/
def specializedCoefficients
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) : List Field :=
  (coefficients ops data).map (Coefficient.specialize ops alpha)

theorem coefficients_length
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) :
    (coefficients ops data).length = shape.jointCoefficientCount := by
  simp [coefficients, residuals, TableResidualData.toResiduals,
    Shape.jointCoefficientCount, Shape.constraintOffset,
    canonicalFinIndices_length,
    TableResidualData.orderedPadEvaluation_length,
    TableResidualData.orderedMatrixEvaluation_length, Nat.add_assoc]

private theorem coefficientTruth_iff_families
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) :
    CoefficientTruth ops data ↔
      (forall value, value ∈ (residuals ops data).padEvaluation ->
        value = ops.zero) ∧
      (forall value, value ∈ (residuals ops data).matrixEvaluation ->
        value = ops.zero) ∧
      (forall polynomial, polynomial ∈ (residuals ops data).ccs ->
        polynomial.CoefficientZero ops.toOps) ∧
      forall polynomial, polynomial ∈ (residuals ops data).norm ->
        polynomial.CoefficientZero ops.toOps := by
  constructor
  · intro truth
    refine ⟨?_, ?_, ?_, ?_⟩
    · intro value member
      exact truth (.scalar value) (by simp [coefficients, member])
    · intro value member
      exact truth (.scalar value) (by simp [coefficients, member])
    · intro polynomial member
      exact truth (.negativeAlpha polynomial) (by
        simp [coefficients, member])
    · intro polynomial member
      exact truth (.negativeAlpha polynomial) (by
        simp [coefficients, member])
  · rintro ⟨padTruth, matrixTruth, ccsTruth, normTruth⟩ coefficient member
    simp only [coefficients, List.mem_append, List.mem_map] at member
    rcases member with
      ⟨value, valueMember, rfl⟩ |
      ⟨value, valueMember, rfl⟩ |
      ⟨polynomial, polynomialMember, rfl⟩ |
      ⟨polynomial, polynomialMember, rfl⟩
    · exact padTruth value valueMember
    · exact matrixTruth value valueMember
    · exact ccsTruth polynomial polynomialMember
    · exact normTruth polynomial polynomialMember

/-- Signed coefficient truth is exactly the pre-existing unsigned family
truth because signs do not alter zero obligations. -/
theorem coefficientTruth_iff_residualCoefficientTruth
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape) :
    CoefficientTruth ops data ↔
      (residuals ops data).CoefficientTruth ops.toOps := by
  rw [coefficientTruth_iff_families]
  exact (Residuals.coefficientTruth_iff_residualFamilies
    ops.toOps (residuals ops data)).symm

/-- The signed object is identically zero exactly when every explicit Pad,
matrix, CCS, and norm residual table obligation holds. -/
theorem coefficientTruth_iff_tableObligations
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : SignedJointIdentity.JointData Field shape) :
    CoefficientTruth ops data ↔
      ((toTableResidualData ops data).toTableObligations ops).AllHold := by
  rw [coefficientTruth_iff_residualCoefficientTruth]
  exact TableResidualData.coefficientTruth_iff_tableObligations
    ops zeroLaws (toTableResidualData ops data)

private theorem specialize_ccs_eq
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) :
    ((residuals ops data).ccs.map Coefficient.negativeAlpha).map
        (Coefficient.specialize ops alpha) =
      SignedCoefficientPolynomial.ccsCoefficients ops data alpha := by
  unfold residuals toTableResidualData TableResidualData.toResiduals
  unfold SignedCoefficientPolynomial.ccsCoefficients
    SignedCoefficientPolynomial.ccsValues
  simp only [List.map_map]
  apply List.map_congr_left
  intro source _
  change ops.neg
      ((BooleanTable.toAlphaPolynomial ops (data.ccs source)).evaluate
        ops.toOps alpha) =
    ops.neg ((data.ccs source).equalityWeightedSum ops alpha)
  rw [BooleanTable.toAlphaPolynomial_evaluate_eq_equalityWeightedSum
    ops laws (data.ccs source) alpha]

private theorem specialize_norm_eq
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) :
    ((residuals ops data).norm.map Coefficient.negativeAlpha).map
        (Coefficient.specialize ops alpha) =
      SignedCoefficientPolynomial.normCoefficients ops data alpha := by
  unfold residuals toTableResidualData TableResidualData.toResiduals
  unfold SignedCoefficientPolynomial.normCoefficients
    SignedCoefficientPolynomial.normValues
  simp only [List.map_map]
  apply List.map_congr_left
  intro source _
  change ops.neg
      ((BooleanTable.toAlphaPolynomial ops (data.norm source)).evaluate
        ops.toOps alpha) =
    ops.neg ((data.norm source).equalityWeightedSum ops alpha)
  rw [BooleanTable.toAlphaPolynomial_evaluate_eq_equalityWeightedSum
    ops laws (data.norm source) alpha]

private theorem specialize_pad_eq
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) :
    ((residuals ops data).padEvaluation.map Coefficient.scalar).map
        (Coefficient.specialize ops alpha) =
      SignedCoefficientPolynomial.padCoefficients ops data := by
  unfold residuals toTableResidualData TableResidualData.toResiduals
  unfold TableResidualData.orderedPadEvaluation
  unfold SignedCoefficientPolynomial.padCoefficients
    SignedCoefficientPolynomial.padValues
  simp [List.map_map, Coefficient.specialize, Function.comp_def]

private theorem specialize_matrix_eq
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) :
    ((residuals ops data).matrixEvaluation.map Coefficient.scalar).map
        (Coefficient.specialize ops alpha) =
      SignedCoefficientPolynomial.matrixCoefficients ops data := by
  unfold residuals toTableResidualData TableResidualData.toResiduals
  unfold TableResidualData.orderedMatrixEvaluation
  unfold SignedCoefficientPolynomial.matrixCoefficients
    SignedCoefficientPolynomial.matrixValues
  simp [List.map_map, Coefficient.specialize, Function.comp_def]

/-- Exact alpha-specialization bridge into the executable signed gamma list. -/
theorem specializedCoefficients_eq
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) :
    specializedCoefficients ops data alpha =
      SignedCoefficientPolynomial.coefficients ops data alpha := by
  unfold specializedCoefficients coefficients
  simp only [List.map_append]
  rw [specialize_pad_eq ops, specialize_matrix_eq ops,
    specialize_ccs_eq ops laws, specialize_norm_eq ops laws]
  rfl

private theorem neg_zero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) :
    ops.neg ops.zero = ops.zero := by
  have inverse := laws.add_neg ops.zero
  simpa only [laws.zero_add] using inverse

private theorem specialize_eq_zero_of_zero
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (alpha : CubePoint Field shape.cubeVariables)
    (coefficient : Coefficient Field shape)
    (zero : coefficient.Zero ops) :
    coefficient.specialize ops alpha = ops.zero := by
  cases coefficient with
  | negativeAlpha polynomial =>
      change ops.neg (polynomial.evaluate ops.toOps alpha) = ops.zero
      rw [AlphaPolynomial.evaluate_eq_zero_of_coefficientZero
        ops.toOps
        { zero_add := laws.zero_add
          zero_mul := fun value => by
            rw [laws.mul_comm]
            exact laws.mul_zero value
          mul_zero := laws.mul_zero }
        polynomial alpha zero]
      exact neg_zero ops laws
  | scalar value => exact zero

private theorem evaluateCoefficients_eq_zero_of_all_zero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (gamma : Field) : forall values : List Field,
    (forall value, value ∈ values -> value = ops.zero) ->
      SumCheck.Finite.Message.evaluateCoefficients
        ops.toOps gamma values = ops.zero
  | [], _ => rfl
  | value :: values, allZero => by
      have headZero : value = ops.zero := allZero value (by simp)
      have tailZero : forall prior, prior ∈ values -> prior = ops.zero := by
        intro prior member
        exact allZero prior (by simp [member])
      simp only [SumCheck.Finite.Message.evaluateCoefficients, headZero,
        evaluateCoefficients_eq_zero_of_all_zero ops laws gamma values tailZero,
        laws.mul_zero, laws.zero_add]

/-- Identically zero signed coefficients force sampled equality for every
alpha and gamma. -/
theorem evaluate_eq_zero_of_coefficientTruth
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (truth : CoefficientTruth ops data) :
    (SignedCoefficientPolynomial.polynomial ops data alpha).evaluate
        ops.toOps gamma = ops.zero := by
  change SumCheck.Finite.Message.evaluateCoefficients ops.toOps gamma
      (SignedCoefficientPolynomial.coefficients ops data alpha) = ops.zero
  rw [← specializedCoefficients_eq ops laws]
  apply evaluateCoefficients_eq_zero_of_all_zero ops laws gamma
  intro value member
  rcases List.mem_map.mp member with
    ⟨coefficient, coefficientMember, rfl⟩
  exact specialize_eq_zero_of_zero ops laws alpha coefficient
    (truth coefficient coefficientMember)

/-- Exact signed mixing bad event: a nonzero finite alpha/gamma coefficient
object vanishes at the verifier's sampled point. -/
structure MixingRoot
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) : Prop where
  coefficientNonzero : Not (CoefficientTruth ops data)
  sampledZero :
    (SignedCoefficientPolynomial.polynomial ops data alpha).evaluate
      ops.toOps gamma = ops.zero

/-- Sampled equality is exactly coefficient truth or the named signed mixing
root. This is deterministic and makes no probability claim. -/
theorem evaluate_eq_zero_iff_coefficientTruth_or_mixingRoot
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    (SignedCoefficientPolynomial.polynomial ops data alpha).evaluate
        ops.toOps gamma = ops.zero ↔
      CoefficientTruth ops data ∨ MixingRoot ops data alpha gamma := by
  constructor
  · intro sampledZero
    by_cases truth : CoefficientTruth ops data
    · exact Or.inl truth
    · exact Or.inr ⟨truth, sampledZero⟩
  · intro conclusion
    cases conclusion with
    | inl truth =>
        exact evaluate_eq_zero_of_coefficientTruth
          ops laws data alpha gamma truth
    | inr bad => exact bad.sampledZero

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedCoefficientObject
