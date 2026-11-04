import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Coefficients

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

open NightstreamFPrime.Spec.SumCheck

universe uField

structure ZeroLaws
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field) : Prop where
  zero_add : forall value, ops.add ops.zero value = value
  zero_mul : forall value, ops.mul ops.zero value = ops.zero
  mul_zero : forall value, ops.mul value ops.zero = ops.zero

structure CubePoint (Field : Type uField) (variables : Nat) where
  coordinates : List Field
  dimension : coordinates.length = variables

structure Sample (Field : Type uField) (shape : Shape) where
  alpha : CubePoint Field shape.cubeVariables
  gamma : Field

private def power
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field)
    (value : Field) : Nat -> Field
  | 0 => ops.one
  | exponent + 1 => ops.mul value (power ops value exponent)

private def evaluatePowers
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field) :
    List Field -> List Nat -> Field
  | [], [] => ops.one
  | value :: values, exponent :: exponents =>
      ops.mul (power ops value exponent)
        (evaluatePowers ops values exponents)
  | _, _ => ops.zero

def AlphaMonomial.evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (monomial : AlphaMonomial shape)
    (point : CubePoint Field shape.cubeVariables) : Field :=
  evaluatePowers ops point.coordinates monomial.exponents

private def evaluateAlphaTerms
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (point : CubePoint Field shape.cubeVariables) :
    List Field -> List (AlphaMonomial shape) -> Field
  | [], [] => ops.zero
  | coefficient :: coefficients, monomial :: monomials =>
      ops.add
        (ops.mul coefficient (monomial.evaluate ops point))
        (evaluateAlphaTerms ops point coefficients monomials)
  | _, _ => ops.zero

private theorem evaluateAlphaTerms_eq_zero_of_all_zero
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ZeroLaws ops)
    (point : CubePoint Field shape.cubeVariables)
    (coefficients : List Field)
    (monomials : List (AlphaMonomial shape))
    (allZero : forall coefficient,
      coefficient ∈ coefficients -> coefficient = ops.zero) :
    evaluateAlphaTerms ops point coefficients monomials = ops.zero := by
  induction coefficients generalizing monomials with
  | nil =>
      cases monomials <;> rfl
  | cons coefficient coefficients inductionHypothesis =>
      cases monomials with
      | nil => rfl
      | cons monomial monomials =>
          have headZero : coefficient = ops.zero :=
            allZero coefficient (by simp)
          have tailZero :
              forall value, value ∈ coefficients -> value = ops.zero := by
            intro value member
            exact allZero value (by simp [member])
          simp only [evaluateAlphaTerms, headZero, laws.zero_mul,
            laws.zero_add]
          exact inductionHypothesis monomials tailZero

def AlphaPolynomial.evaluate
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (polynomial : AlphaPolynomial Field basis)
    (point : CubePoint Field shape.cubeVariables) : Field :=
  evaluateAlphaTerms ops point polynomial.coefficients basis.monomials

theorem AlphaPolynomial.evaluate_eq_zero_of_coefficientZero
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ZeroLaws ops)
    (polynomial : AlphaPolynomial Field basis)
    (point : CubePoint Field shape.cubeVariables)
    (zero : polynomial.CoefficientZero ops) :
    polynomial.evaluate ops point = ops.zero := by
  unfold AlphaPolynomial.evaluate
  exact evaluateAlphaTerms_eq_zero_of_all_zero ops laws point
    polynomial.coefficients basis.monomials zero

def JointCoefficient.specializeAlpha
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (alpha : CubePoint Field shape.cubeVariables) :
    JointCoefficient Field basis -> Field
  | .alpha polynomial => polynomial.evaluate ops alpha
  | .scalar value => value

theorem JointCoefficient.specializeAlpha_eq_zero_of_zero
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ZeroLaws ops)
    (alpha : CubePoint Field shape.cubeVariables)
    (coefficient : JointCoefficient Field basis)
    (zero : coefficient.Zero ops) :
    coefficient.specializeAlpha ops alpha = ops.zero := by
  cases coefficient with
  | alpha polynomial =>
      exact polynomial.evaluate_eq_zero_of_coefficientZero ops laws alpha zero
  | scalar value =>
      exact zero

namespace Residuals

def specializedGammaPolynomial
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis)
    (alpha : CubePoint Field shape.cubeVariables) :
    SumCheck.Finite.Message Field where
  coefficients := residuals.jointCoefficients.map
    (JointCoefficient.specializeAlpha ops alpha)

theorem specializedGammaPolynomial_degreeUpperBound
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis)
    (alpha : CubePoint Field shape.cubeVariables) :
    (residuals.specializedGammaPolynomial ops alpha).degreeUpperBound =
      shape.jointCoefficientCount - 1 := by
  simp [specializedGammaPolynomial,
    SumCheck.Finite.Message.degreeUpperBound,
    residuals.jointCoefficients_length]

def evaluateAtSample
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis)
    (sample : Sample Field shape) : Field :=
  (residuals.specializedGammaPolynomial ops sample.alpha).evaluate
    ops sample.gamma

private theorem evaluateCoefficients_eq_zero_of_all_zero
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ZeroLaws ops)
    (point : Field)
    (coefficients : List Field)
    (allZero : forall coefficient,
      coefficient ∈ coefficients -> coefficient = ops.zero) :
    SumCheck.Finite.Message.evaluateCoefficients ops point coefficients =
      ops.zero := by
  induction coefficients with
  | nil => rfl
  | cons coefficient coefficients inductionHypothesis =>
      have headZero : coefficient = ops.zero :=
        allZero coefficient (by simp)
      have tailZero :
          forall value, value ∈ coefficients -> value = ops.zero := by
        intro value member
        exact allZero value (by simp [member])
      simp only [SumCheck.Finite.Message.evaluateCoefficients,
        inductionHypothesis tailZero, laws.mul_zero, headZero, laws.zero_add]

theorem evaluateAtSample_eq_zero_of_coefficientTruth
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ZeroLaws ops)
    (residuals : Residuals Field shape basis)
    (sample : Sample Field shape)
    (truth : residuals.CoefficientTruth ops) :
    residuals.evaluateAtSample ops sample = ops.zero := by
  apply evaluateCoefficients_eq_zero_of_all_zero ops laws sample.gamma
  intro value valueMember
  rcases List.mem_map.mp valueMember with
    ⟨coefficient, coefficientMember, valueEq⟩
  rw [← valueEq]
  exact coefficient.specializeAlpha_eq_zero_of_zero ops laws sample.alpha
    (truth coefficient coefficientMember)

structure MixingRoot
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis)
    (sample : Sample Field shape) : Prop where
  coefficientNonzero : Not (residuals.CoefficientTruth ops)
  sampledZero : residuals.evaluateAtSample ops sample = ops.zero

theorem sampledZero_iff_coefficientTruth_or_mixingRoot
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ZeroLaws ops)
    (residuals : Residuals Field shape basis)
    (sample : Sample Field shape) :
    residuals.evaluateAtSample ops sample = ops.zero ↔
      residuals.CoefficientTruth ops ∨ MixingRoot ops residuals sample := by
  constructor
  · intro sampledZero
    by_cases truth : residuals.CoefficientTruth ops
    · exact Or.inl truth
    · exact Or.inr ⟨truth, sampledZero⟩
  · intro conclusion
    cases conclusion with
    | inl truth =>
        exact residuals.evaluateAtSample_eq_zero_of_coefficientTruth
          ops laws sample truth
    | inr bad => exact bad.sampledZero

theorem sampledZero_iff_allObligations_or_mixingRoot
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ZeroLaws ops)
    (residuals : Residuals Field shape basis)
    (obligations : Obligations shape)
    (boundary : ResidualizationBoundary ops residuals obligations)
    (sample : Sample Field shape) :
    residuals.evaluateAtSample ops sample = ops.zero ↔
      obligations.AllHold ∨ MixingRoot ops residuals sample := by
  rw [sampledZero_iff_coefficientTruth_or_mixingRoot ops laws]
  rw [residuals.coefficientTruth_iff_allObligations ops obligations boundary]

end Residuals

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
