import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Coefficients

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/Sampling.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Deterministic alpha/gamma specialization of the paper's joint `Pi_CCS`
residual polynomial.

Owns: finite monomial evaluation, alpha specialization, constant-first gamma
evaluation, the exact named mixing-root event, and the deterministic
soundness dichotomy after sampling.

Does not own: how alpha or gamma are sampled, probability or
Schwartz--Zippel bounds, SumCheck soundness, transcript derivation,
Fiat--Shamir, concrete CCS arithmetization, Rust, or R1CS.

Emits constraints: no.

Authority boundary: `Sample` contains verifier challenges only. The polynomial
being evaluated is the finite joint coefficient object from `Coefficients`;
there is no caller-supplied evaluation oracle.

| Sampling level | Finite input | Computation | Failure boundary |
|---|---|---|---|
| alpha | fixed monomial basis and coefficient vectors | explicit powers, products, and sums | a nonzero alpha polynomial may vanish at the chosen point |
| gamma | constant-first specialized coefficient list | Horner evaluation | the nonzero joint residual may vanish at `(alpha, gamma)` |
| conditional table conclusion | explicit residual coefficient object | coefficient theorem plus evaluation | `MixingRoot`, with no probability claim |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

open NightstreamFPrime.Spec.SumCheck

universe uField

/-- Only the zero laws needed to prove that a coefficient-zero polynomial
evaluates to zero. General ring/field laws and root counting are intentionally
outside this deterministic slice. -/
structure ZeroLaws
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field) : Prop where
  zero_add : forall value, ops.add ops.zero value = value
  zero_mul : forall value, ops.mul ops.zero value = ops.zero
  mul_zero : forall value, ops.mul value ops.zero = ops.zero

/-- A dimension-checked point in `K^(log m)`. Alpha and the SumCheck output
point use this same finite representation, while remaining distinct values. -/
structure CubePoint (Field : Type uField) (variables : Nat) where
  coordinates : List Field
  dimension : coordinates.length = variables

/-- The paper's two pre-SumCheck verifier challenges. -/
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

/-- Explicit evaluation of one finite alpha monomial. The shape proofs on the
basis and point make the mismatch branches unreachable for well-formed data;
they merely keep the raw recursive function total. -/
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

/-- Evaluate finite coefficient data against its fixed alpha basis. -/
def AlphaPolynomial.evaluate
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (polynomial : AlphaPolynomial Field basis)
    (point : CubePoint Field shape.cubeVariables) : Field :=
  evaluateAlphaTerms ops point polynomial.coefficients basis.monomials

/-- Coefficient-zero alpha polynomials evaluate to zero at every point. -/
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

/-- Specialize one joint gamma coefficient at alpha. -/
def JointCoefficient.specializeAlpha
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (alpha : CubePoint Field shape.cubeVariables) :
    JointCoefficient Field basis -> Field
  | .alpha polynomial => polynomial.evaluate ops alpha
  | .scalar value => value

/-- A zero joint coefficient remains zero after alpha specialization. -/
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

/-- Finite constant-first gamma polynomial after specializing alpha. The
degree upper bound is derived from this coefficient-list length by
`SumCheck.Finite.Message.degreeUpperBound`; no degree metadata is carried. -/
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

/-- The gamma degree upper bound is derived from the paper block sizes. It is
an upper bound from finite list length, not a claim that the high coefficient
is nonzero or that a SumCheck round used this degree. -/
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

/-- Evaluate the unsigned, formula-agnostic coefficient object using Horner
evaluation. This is not yet `T_abs(gamma) - sum_x Q(x, alpha, gamma)`:
identifying it with that signed paper residual additionally requires concrete
table construction, the target-shift convention, and the signed joint
polynomial identity. -/
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

/-- Coefficient truth forces sampled equality for every alpha and gamma. -/
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

/-- Exact deterministic bad event for this assembled coefficient object: a
nonzero polynomial in `(A, C)` vanishes at the verifier's sampled
`(alpha, gamma)`. Relating this event to Appendix D.4 still requires concrete
residual construction and the signed joint identity; this definition makes no
claim about that bridge or about how likely the event is. -/
structure MixingRoot
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis)
    (sample : Sample Field shape) : Prop where
  coefficientNonzero : Not (residuals.CoefficientTruth ops)
  sampledZero : residuals.evaluateAtSample ops sample = ops.zero

/-- Sampled equality is exactly either coefficient truth or the named
mixing-root event. This is deterministic model-level assurance, not a
Schwartz--Zippel theorem. -/
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

/-- Given an explicit residualization boundary, sampled equality implies every
supplied obligation except precisely when `(alpha, gamma)` is a root of the
nonzero assembled coefficient object. This remains conditional and is not a
concrete paper-semantics theorem. -/
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
