import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CoefficientRootCounting
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MultilinearRootCounting
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts

/-!
Finite alpha/gamma mixing soundness for the causal paper `Pi_CCS` experiment.

Assurance tier: model-level.

Owns: coefficient-level selection of one false CCS, norm, or carried
obligation; multilinear alpha root counting; constant-first gamma root
counting; the exact alpha-then-gamma Cartesian union bound; transport through
the verifier's factorized coin support; and construction of the existing
`MixingRootProbabilityContract`.

Does not own: SumCheck soundness, first-success conditioning, Fiat--Shamir,
Poseidon2, the production Split-NC challenge carrier, Rust, R1CS, artifacts,
or costs.

Emits constraints: no.

| Random coordinate | Exact source | Root charge |
|---|---|---|
| `alpha : S^ell` | verifier word before prover rounds | `ell / |S|` |
| `gamma : S` | shared joint-polynomial scalar | `(jointCoefficientCount-1) / |S|` |
| SumCheck word | independent later verifier word | absent from this event |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.SumCheck.Finite

universe uExtension uCommitment uPublicInput uProverSeed uTargetSeed uProverTape
  uAlpha uGamma

private def wordPoint
    {Extension : Type uExtension}
    {variables : Nat}
    (word : Fin variables -> Extension) : CubePoint Extension variables where
  coordinates := List.ofFn word
  dimension := by simp

private theorem neg_eq_zero_implies_eq_zero
    {Extension : Type uExtension}
    {ops : InterpolationOps Extension}
    (laws : InterpolationEvaluationLaws ops)
    {value : Extension}
    (negZero : ops.neg value = ops.zero) :
    value = ops.zero := by
  calc
    value = ops.add value ops.zero := (laws.add_zero value).symm
    _ = ops.add value (ops.neg value) := by rw [negZero]
    _ = ops.zero := laws.add_neg value

private theorem negativeAlpha_origin
    {Extension : Type uExtension}
    {shape : Shape}
    (ops : InterpolationOps Extension)
    (data : SignedJointIdentity.JointData Extension shape)
    (polynomial : AlphaPolynomial Extension (canonicalAlphaBasis shape))
    (member :
      SignedCoefficientObject.Coefficient.negativeAlpha polynomial ∈
        SignedCoefficientObject.coefficients ops data) :
    (exists source : Fin shape.freshCount,
      polynomial = (data.ccs source).toAlphaPolynomial ops) \/
    (exists source : Fin shape.sourceCount,
      polynomial = (data.norm source).toAlphaPolynomial ops) := by
  unfold SignedCoefficientObject.coefficients at member
  simp only [List.mem_append, List.mem_map] at member
  rcases member with
    (⟨prior, priorMember, coefficientEq⟩ |
      ⟨prior, priorMember, coefficientEq⟩) |
      ⟨value, valueMember, coefficientEq⟩
  · unfold SignedCoefficientObject.residuals
      SignedCoefficientObject.toTableResidualData
      TableResidualData.toResiduals at priorMember
    simp only [List.mem_map] at priorMember
    rcases priorMember with ⟨source, _sourceMember, priorEq⟩
    cases coefficientEq
    exact Or.inl ⟨source, by simpa using priorEq.symm⟩
  · unfold SignedCoefficientObject.residuals
      SignedCoefficientObject.toTableResidualData
      TableResidualData.toResiduals at priorMember
    simp only [List.mem_map] at priorMember
    rcases priorMember with ⟨source, _sourceMember, priorEq⟩
    cases coefficientEq
    exact Or.inr ⟨source, by simpa using priorEq.symm⟩
  · cases coefficientEq

private theorem signedCoefficients_nonzero_of_negativeAlpha
    {Extension : Type uExtension}
    {shape : Shape}
    (ops : InterpolationOps Extension)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Extension shape)
    (polynomial : AlphaPolynomial Extension (canonicalAlphaBasis shape))
    (member :
      SignedCoefficientObject.Coefficient.negativeAlpha polynomial ∈
        SignedCoefficientObject.coefficients ops data)
    (alpha : CubePoint Extension shape.cubeVariables)
    (evaluationNonzero :
      polynomial.evaluate ops.toOps alpha ≠ ops.zero) :
    Not (CoefficientRootCounting.AllZero ops
      (SignedCoefficientPolynomial.coefficients ops data alpha)) := by
  intro allZero
  have specializedMember :
      ops.neg (polynomial.evaluate ops.toOps alpha) ∈
        SignedCoefficientObject.specializedCoefficients ops data alpha := by
    exact List.mem_map.mpr ⟨
      SignedCoefficientObject.Coefficient.negativeAlpha polynomial,
      member, rfl⟩
  have signedMember :
      ops.neg (polynomial.evaluate ops.toOps alpha) ∈
        SignedCoefficientPolynomial.coefficients ops data alpha := by
    rw [← SignedCoefficientObject.specializedCoefficients_eq ops laws]
    exact specializedMember
  exact evaluationNonzero
    (neg_eq_zero_implies_eq_zero laws
      (allZero _ signedMember))

private theorem signedCoefficients_nonzero_of_scalar
    {Extension : Type uExtension}
    {shape : Shape}
    (ops : InterpolationOps Extension)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Extension shape)
    (value : Extension)
    (member :
      SignedCoefficientObject.Coefficient.scalar value ∈
        SignedCoefficientObject.coefficients ops data)
    (valueNonzero : value ≠ ops.zero)
    (alpha : CubePoint Extension shape.cubeVariables) :
    Not (CoefficientRootCounting.AllZero ops
      (SignedCoefficientPolynomial.coefficients ops data alpha)) := by
  intro allZero
  have specializedMember :
      value ∈ SignedCoefficientObject.specializedCoefficients ops data alpha :=
    List.mem_map.mpr ⟨
      SignedCoefficientObject.Coefficient.scalar value, member, rfl⟩
  have signedMember :
      value ∈ SignedCoefficientPolynomial.coefficients ops data alpha := by
    rw [← SignedCoefficientObject.specializedCoefficients_eq ops laws]
    exact specializedMember
  exact valueNonzero (allZero value signedMember)

/-- A single verifier-independent residual controls all bad alpha points.
The controller is extracted from coefficient nontruth; it is not prover data
and it adds no premise to the final protocol theorem. -/
private inductive Controller
    {Extension : Type uExtension}
    {shape : Shape}
    (ops : InterpolationOps Extension)
    (data : SignedJointIdentity.JointData Extension shape) : Prop where
  | table
      (residual : BooleanTable Extension shape.cubeVariables)
      (nonzero : Not (residual.AllEntriesZero ops))
      (controls : forall alpha,
        residual.evaluate ops alpha ≠ ops.zero ->
        Not (CoefficientRootCounting.AllZero ops
          (SignedCoefficientPolynomial.coefficients ops data alpha)))
  | scalar
      (controls : forall alpha,
        Not (CoefficientRootCounting.AllZero ops
          (SignedCoefficientPolynomial.coefficients ops data alpha)))

private theorem controller_of_coefficientNonzero
    {Extension : Type uExtension}
    {shape : Shape}
    (ops : InterpolationOps Extension)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (data : SignedJointIdentity.JointData Extension shape)
    (nonzero :
      Not (SignedCoefficientObject.CoefficientTruth ops data)) :
    Controller ops data := by
  classical
  obtain ⟨coefficient, coefficientFailure⟩ :=
    Classical.not_forall.mp nonzero
  have coefficientParts :=
    Classical.not_imp.mp coefficientFailure
  rcases coefficientParts with ⟨member, coefficientNonzero⟩
  cases coefficient with
  | scalar value =>
      exact .scalar fun alpha =>
        signedCoefficients_nonzero_of_scalar ops laws data value member
          coefficientNonzero alpha
  | negativeAlpha polynomial =>
      rcases negativeAlpha_origin ops data polynomial member with
        ⟨source, polynomialEq⟩ | ⟨source, polynomialEq⟩
      · subst polynomial
        have tableNonzero :
            Not ((data.ccs source).AllEntriesZero ops) := by
          intro tableZero
          exact coefficientNonzero
            ((BooleanTable.toAlphaPolynomial_coefficientZero_iff_allEntriesZero
              ops zeroLaws (data.ccs source)).2 tableZero)
        exact .table (data.ccs source) tableNonzero (by
          intro alpha evaluationNonzero
          apply signedCoefficients_nonzero_of_negativeAlpha ops laws data
            ((data.ccs source).toAlphaPolynomial ops) member alpha
          rw [BooleanTable.toAlphaPolynomial_evaluate_eq_evaluate
            ops laws]
          exact evaluationNonzero)
      · subst polynomial
        have tableNonzero :
            Not ((data.norm source).AllEntriesZero ops) := by
          intro tableZero
          exact coefficientNonzero
            ((BooleanTable.toAlphaPolynomial_coefficientZero_iff_allEntriesZero
              ops zeroLaws (data.norm source)).2 tableZero)
        exact .table (data.norm source) tableNonzero (by
          intro alpha evaluationNonzero
          apply signedCoefficients_nonzero_of_negativeAlpha ops laws data
            ((data.norm source).toAlphaPolynomial ops) member alpha
          rw [BooleanTable.toAlphaPolynomial_evaluate_eq_evaluate
            ops laws]
          exact evaluationNonzero)

private theorem sum_fibers_le_bad_mul_length_add_degree
    {Alpha : Type uAlpha}
    {Gamma : Type uGamma}
    (alphas : List Alpha)
    (gammas : List Gamma)
    (bad : Alpha -> Bool)
    (event : Alpha -> Gamma -> Bool)
    (degree : Nat)
    (goodBound :
      forall alpha, alpha ∈ alphas -> bad alpha = false ->
        gammas.countP (event alpha) <= degree) :
    (alphas.map fun alpha => gammas.countP (event alpha)).sum <=
      alphas.countP bad * gammas.length + alphas.length * degree := by
  induction alphas with
  | nil => simp
  | cons alpha alphas inductionHypothesis =>
      have tailBound :
          forall prior, prior ∈ alphas -> bad prior = false ->
            gammas.countP (event prior) <= degree := by
        intro prior member priorGood
        exact goodBound prior (by simp [member]) priorGood
      have remainder := inductionHypothesis tailBound
      cases alphaBad : bad alpha with
      | false =>
          have current := goodBound alpha (by simp) alphaBad
          have combined := Nat.add_le_add current remainder
          simpa [alphaBad, Nat.add_mul, Nat.mul_add, Nat.add_assoc,
            Nat.add_comm, Nat.add_left_comm] using combined
      | true =>
          have current : gammas.countP (event alpha) <= gammas.length :=
            List.countP_le_length
          have withSlack :
              gammas.countP (event alpha) <= gammas.length + degree :=
            Nat.le_trans current (Nat.le_add_right _ _)
          have combined := Nat.add_le_add withSlack remainder
          simpa [alphaBad, Nat.add_mul, Nat.mul_add, Nat.add_assoc,
            Nat.add_comm, Nat.add_left_comm] using combined

private theorem sum_map_le_constant
    {Element : Type uAlpha}
    (values : List Element)
    (cost : Element -> Nat)
    (bound : Nat)
    (bounded : forall value, value ∈ values -> cost value <= bound) :
    (values.map cost).sum <= values.length * bound := by
  induction values with
  | nil => simp
  | cons value values inductionHypothesis =>
      have current := bounded value List.mem_cons_self
      have tailBound :
          forall prior, prior ∈ values -> cost prior <= bound := by
        intro prior member
        exact bounded prior (List.mem_cons_of_mem value member)
      have remainder := inductionHypothesis tailBound
      simpa [Nat.add_mul, Nat.add_comm] using
        Nat.add_le_add current remainder

private theorem alphaGamma_count_le
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {shape : Shape}
    (ops : InterpolationOps Extension)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (data : SignedJointIdentity.JointData Extension shape)
    (alphabet : Support Extension)
    (coefficientNonzero :
      Not (SignedCoefficientObject.CoefficientTruth ops data)) :
    let alphas := vectors alphabet.values shape.cubeVariables
    (alphas.flatMap fun alpha =>
      alphabet.values.map fun gamma => (alpha, gamma)).countP
        (fun sample =>
          decide
            ((SignedCoefficientPolynomial.polynomial ops data
              (wordPoint sample.1)).evaluate ops.toOps sample.2 =
                ops.zero)) <=
      (shape.cubeVariables + (shape.jointCoefficientCount - 1)) *
        alphabet.cardinality ^ shape.cubeVariables := by
  let alphas := vectors alphabet.values shape.cubeVariables
  let gammaDegree := shape.jointCoefficientCount - 1
  let event : (Fin shape.cubeVariables -> Extension) -> Extension -> Bool :=
    fun alpha gamma =>
      decide
        ((SignedCoefficientPolynomial.polynomial ops data
          (wordPoint alpha)).evaluate ops.toOps gamma = ops.zero)
  have gammaBound
      (alpha : Fin shape.cubeVariables -> Extension)
      (nonzero :
        Not (CoefficientRootCounting.AllZero ops
          (SignedCoefficientPolynomial.coefficients ops data
            (wordPoint alpha)))) :
      alphabet.values.countP (event alpha) <= gammaDegree := by
    let coefficients :=
      SignedCoefficientPolynomial.coefficients ops data (wordPoint alpha)
    have coefficientsNonempty : coefficients ≠ [] := by
      intro empty
      apply nonzero
      intro coefficient member
      change coefficient ∈ coefficients at member
      rw [empty] at member
      exact False.elim (List.not_mem_nil member)
    have coefficientCount :
        coefficients.length = coefficients.length.pred + 1 := by
      have positive := List.length_pos_iff.mpr coefficientsNonempty
      exact (Nat.succ_pred_eq_of_pos positive).symm
    have rootBound :=
      CoefficientRootCounting.roots_count_le_degree ops laws noZeroDivisors
        coefficients.length.pred coefficients coefficientCount
        alphabet.values alphabet.nodup nonzero
    simpa [event, gammaDegree, coefficients,
      SignedCoefficientPolynomial.coefficients_length] using rootBound
  change
    (alphas.flatMap fun alpha =>
      alphabet.values.map fun gamma => (alpha, gamma)).countP
        (fun sample => event sample.1 sample.2) <=
      (shape.cubeVariables + gammaDegree) *
        alphabet.cardinality ^ shape.cubeVariables
  rw [List.countP_flatMap]
  have normalized :
      (alphas.map fun alpha =>
        (alphabet.values.map fun gamma => (alpha, gamma)).countP
          (fun sample => event sample.1 sample.2)).sum =
        (alphas.map fun alpha =>
          alphabet.values.countP (event alpha)).sum := by
    apply congrArg List.sum
    apply List.map_congr_left
    intro alpha _member
    rw [List.countP_map]
    rfl
  change
    (alphas.map fun alpha =>
      (alphabet.values.map fun gamma => (alpha, gamma)).countP
        (fun sample => event sample.1 sample.2)).sum <=
      (shape.cubeVariables + gammaDegree) *
        alphabet.cardinality ^ shape.cubeVariables
  rw [normalized]
  cases controller_of_coefficientNonzero ops laws zeroLaws data
      coefficientNonzero with
  | scalar controls =>
      have eachBound :
          forall alpha, alpha ∈ alphas ->
            alphabet.values.countP (event alpha) <= gammaDegree := by
        intro alpha _member
        exact gammaBound alpha (controls (wordPoint alpha))
      have sumBound :
          (alphas.map fun alpha =>
            alphabet.values.countP (event alpha)).sum <=
            alphas.length * gammaDegree := by
        exact sum_map_le_constant alphas
          (fun alpha => alphabet.values.countP (event alpha))
          gammaDegree eachBound
      refine Nat.le_trans sumBound ?_
      rw [vectors_length]
      unfold gammaDegree Support.cardinality
      rw [Nat.add_mul, Nat.mul_comm
        (alphabet.values.length ^ shape.cubeVariables)
        (shape.jointCoefficientCount - 1)]
      exact Nat.le_add_left _ _
  | table residual residualNonzero controls =>
      let alphaBad : (Fin shape.cubeVariables -> Extension) -> Bool :=
        fun alpha =>
          decide
            (residual.evaluateCoordinates ops (List.ofFn alpha) = ops.zero)
      have badBound :
          alphas.countP alphaBad <=
            shape.cubeVariables *
              alphabet.cardinality ^ shape.cubeVariables.pred := by
        simpa [alphas, alphaBad, Support.cardinality] using
          MultilinearRootCounting.zeros_count_le ops laws noZeroDivisors
            residual alphabet.values alphabet.nodup residualNonzero
      have goodBound :
          forall alpha, alpha ∈ alphas -> alphaBad alpha = false ->
            alphabet.values.countP (event alpha) <= gammaDegree := by
        intro alpha _member alphaGood
        have evaluationNonzero :
            residual.evaluate ops (wordPoint alpha) ≠ ops.zero := by
          simpa [alphaBad, BooleanTable.evaluate, wordPoint] using alphaGood
        exact gammaBound alpha
          (controls (wordPoint alpha) evaluationNonzero)
      have fiberBound :=
        sum_fibers_le_bad_mul_length_add_degree alphas alphabet.values
          alphaBad event gammaDegree goodBound
      have combined := Nat.add_le_add
        (Nat.mul_le_mul_right alphabet.values.length badBound)
        (Nat.le_refl (alphas.length * gammaDegree))
      refine Nat.le_trans fiberBound (Nat.le_trans combined ?_)
      rw [vectors_length]
      unfold gammaDegree Support.cardinality
      cases shape.cubeVariables with
      | zero => simp
      | succ variables =>
          apply Nat.le_of_eq
          simp only [Nat.pred_succ, Nat.pow_succ, Nat.add_mul]
          ac_rfl

/-- Exact Boolean event on the verifier's alpha/gamma product. -/
noncomputable def alphaGammaZeroEvent
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {shape : Shape}
    (ops : InterpolationOps Extension)
    (data : SignedJointIdentity.JointData Extension shape) :
    (VerifierCoins.Word Extension shape.cubeVariables × Extension) -> Bool :=
  fun sample =>
    decide
      ((SignedCoefficientPolynomial.polynomial ops data
        (wordPoint sample.1)).evaluate ops.toOps sample.2 = ops.zero)

/-- The finite alpha/gamma product obeys the additive multivariate-plus-
univariate root bound. The denominator is exactly the sampled scalar support
cardinality. -/
theorem alphaGammaZero_probability_le
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {shape : Shape}
    (ops : InterpolationOps Extension)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (data : SignedJointIdentity.JointData Extension shape)
    (alphabet : Support Extension)
    (coefficientNonzero :
      Not (SignedCoefficientObject.CoefficientTruth ops data)) :
    let alphaSupport :=
      FiniteWords.Support.challengeVectors alphabet shape.cubeVariables
    (alphaSupport.product alphabet).uniform.probabilityBool
        (alphaGammaZeroEvent ops data) <=
      ratio
        (shape.cubeVariables + (shape.jointCoefficientCount - 1))
        alphabet.cardinality := by
  let alphaSupport :=
    FiniteWords.Support.challengeVectors alphabet shape.cubeVariables
  let numerator :=
    shape.cubeVariables + (shape.jointCoefficientCount - 1)
  have countBound :=
    alphaGamma_count_le ops laws zeroLaws noZeroDivisors data alphabet
      coefficientNonzero
  have denominatorPos :
      0 < (((alphaSupport.product alphabet).cardinality : Nat) : Rat) :=
    Rat.natCast_pos.mpr (alphaSupport.product alphabet).cardinality_pos
  unfold Experiment.probabilityBool Experiment.countBool
  apply (div_le_iff_of_pos denominatorPos).2
  have castCountBound :
      (((alphaSupport.product alphabet).values.countP
        (alphaGammaZeroEvent ops data) : Nat) : Rat) <=
        (numerator * alphabet.cardinality ^ shape.cubeVariables : Nat) := by
    exact Rat.natCast_le_natCast.mpr (by
      simpa [alphaSupport, alphaGammaZeroEvent, numerator,
        FiniteWords.Support.challengeVectors_values,
        Support.product_values] using countBound)
  refine Rat.le_trans castCountBound ?_
  have alphabetNeZero : (alphabet.cardinality : Rat) ≠ 0 :=
    Rat.ne_of_gt (Rat.natCast_pos.mpr alphabet.cardinality_pos)
  have ratioTimesDenominator :
      ratio numerator alphabet.cardinality *
          (((alphaSupport.product alphabet).cardinality : Nat) : Rat) =
        ((numerator * alphabet.cardinality ^ shape.cubeVariables : Nat) :
          Rat) := by
    unfold ratio
    rw [Support.product_cardinality,
      FiniteWords.Support.challengeVectors_cardinality]
    simp only [Rat.natCast_mul, Rat.natCast_pow]
    calc
      ((numerator : Rat) / (alphabet.cardinality : Rat)) *
          ((alphabet.cardinality : Rat) ^ shape.cubeVariables *
            (alphabet.cardinality : Rat)) =
        (((numerator : Rat) / (alphabet.cardinality : Rat)) *
          (alphabet.cardinality : Rat)) *
            (alphabet.cardinality : Rat) ^ shape.cubeVariables := by
          rw [Rat.mul_comm
            ((alphabet.cardinality : Rat) ^ shape.cubeVariables)
            (alphabet.cardinality : Rat)]
          rw [← Rat.mul_assoc]
      _ = (numerator : Rat) *
          (alphabet.cardinality : Rat) ^ shape.cubeVariables := by
        rw [Rat.div_mul_cancel alphabetNeZero]
  rw [ratioTimesDenominator]
  exact Rat.le_refl

private theorem nested_alphaGamma_count
    {Alpha : Type uAlpha}
    {Gamma : Type uGamma}
    {Round : Type uProverSeed}
    (alphas : List Alpha)
    (gammas : List Gamma)
    (rounds : List Round)
    (event : Alpha -> Gamma -> Bool) :
    (alphas.flatMap fun alpha =>
      (gammas.flatMap fun gamma =>
        rounds.map fun round => (gamma, round)).map fun gammaRound =>
          (alpha, gammaRound)).countP
        (fun seed => event seed.1 seed.2.1) =
      (alphas.flatMap fun alpha =>
        gammas.map fun gamma => (alpha, gamma)).countP
          (fun seed => event seed.1 seed.2) * rounds.length := by
  have innerCount (alpha : Alpha) :
      (gammas.flatMap fun gamma =>
        rounds.map fun round => (gamma, round)).countP
          (fun seed => event alpha seed.1) =
        gammas.countP (event alpha) * rounds.length := by
    induction gammas with
    | nil => simp
    | cons gamma gammas inductionHypothesis =>
        have headCount :
            (rounds.map fun round => (gamma, round)).countP
                (fun seed => event alpha seed.1) =
              if event alpha gamma then rounds.length else 0 := by
          rw [List.countP_map]
          cases eventAtGamma : event alpha gamma <;>
            simp [eventAtGamma]
        rw [List.flatMap_cons, List.countP_append, headCount,
          inductionHypothesis]
        cases eventAtGamma : event alpha gamma <;>
          simp [eventAtGamma, Nat.add_mul, Nat.add_comm]
  induction alphas with
  | nil => simp
  | cons alpha alphas inductionHypothesis =>
      rw [List.flatMap_cons, List.countP_append,
        List.flatMap_cons, List.countP_append]
      have leftHead :
          (((gammas.flatMap fun gamma =>
            rounds.map fun round => (gamma, round)).map fun gammaRound =>
              (alpha, gammaRound))).countP
              (fun seed => event seed.1 seed.2.1) =
            gammas.countP (event alpha) * rounds.length := by
        rw [List.countP_map]
        simpa using innerCount alpha
      have rightHead :
          (gammas.map fun gamma => (alpha, gamma)).countP
              (fun seed => event seed.1 seed.2) =
            gammas.countP (event alpha) := by
        rw [List.countP_map]
        rfl
      rw [leftHead, rightHead, inductionHypothesis, Nat.add_mul]

/-- The complete verifier seed has the exact alpha/gamma marginal. The
independent SumCheck word is enumerated after gamma and cancels from both
event count and support cardinality. -/
theorem verifierAlphaGamma_marginal
    {Extension : Type uExtension}
    (alphabet : Support Extension)
    (variables : Nat)
    (event : VerifierCoins.Word Extension variables -> Extension -> Bool) :
    ((VerifierCoins.support alphabet variables).uniform).probabilityBool
        (fun seed =>
          event (VerifierCoins.alphaWord seed) (VerifierCoins.gamma seed)) =
      let alphaSupport :=
        FiniteWords.Support.challengeVectors alphabet variables
      (alphaSupport.product alphabet).uniform.probabilityBool
        (fun seed => event seed.1 seed.2) := by
  let alphaSupport :=
    FiniteWords.Support.challengeVectors alphabet variables
  have countIdentity :
      (VerifierCoins.support alphabet variables).values.countP
          (fun seed =>
            event (VerifierCoins.alphaWord seed)
              (VerifierCoins.gamma seed)) =
        (alphaSupport.product alphabet).values.countP
            (fun seed => event seed.1 seed.2) *
          alphaSupport.cardinality := by
    change
      (alphaSupport.values.flatMap fun alpha =>
        (alphabet.values.flatMap fun gamma =>
          alphaSupport.values.map fun round => (gamma, round)).map
            fun gammaRound => (alpha, gammaRound)).countP
          (fun seed => event seed.1 seed.2.1) =
        (alphaSupport.values.flatMap fun alpha =>
          alphabet.values.map fun gamma => (alpha, gamma)).countP
            (fun seed => event seed.1 seed.2) *
          alphaSupport.values.length
    exact nested_alphaGamma_count alphaSupport.values alphabet.values
      alphaSupport.values event
  change
    (((VerifierCoins.support alphabet variables).values.countP
        (fun seed =>
          event (VerifierCoins.alphaWord seed)
            (VerifierCoins.gamma seed)) : Nat) : Rat) /
      ((VerifierCoins.support alphabet variables).cardinality : Rat) =
    (((alphaSupport.product alphabet).values.countP
        (fun seed => event seed.1 seed.2) : Nat) : Rat) /
      ((alphaSupport.product alphabet).cardinality : Rat)
  rw [countIdentity]
  have verifierCardinality :
      (VerifierCoins.support alphabet variables).cardinality =
        alphaSupport.cardinality *
          (alphabet.cardinality * alphaSupport.cardinality) := by
    simp [alphaSupport]
  rw [verifierCardinality, Support.product_cardinality]
  change
    ((((alphaSupport.product alphabet).values.countP
        (fun seed => event seed.1 seed.2) *
          alphaSupport.cardinality : Nat) : Rat) /
      ((alphaSupport.cardinality *
        (alphabet.cardinality * alphaSupport.cardinality) : Nat) : Rat)) =
    ((((alphaSupport.product alphabet).values.countP
        (fun seed => event seed.1 seed.2) : Nat) : Rat) /
      ((alphaSupport.cardinality * alphabet.cardinality : Nat) : Rat))
  have alphaNonzero : (alphaSupport.cardinality : Rat) ≠ 0 :=
    Rat.ne_of_gt (Rat.natCast_pos.mpr alphaSupport.cardinality_pos)
  simp only [Rat.natCast_mul, Rat.div_def, Rat.inv_mul_rev]
  calc
    (((alphaSupport.product alphabet).values.countP
          (fun seed => event seed.1 seed.2) : Nat) : Rat) *
          (alphaSupport.cardinality : Rat) *
        (((alphaSupport.cardinality : Rat)⁻¹ *
          (alphabet.cardinality : Rat)⁻¹) *
            (alphaSupport.cardinality : Rat)⁻¹) =
      (((alphaSupport.product alphabet).values.countP
          (fun seed => event seed.1 seed.2) : Nat) : Rat) *
        ((alphabet.cardinality : Rat)⁻¹ *
          (alphaSupport.cardinality : Rat)⁻¹) := by
      calc
        (((alphaSupport.product alphabet).values.countP
              (fun seed => event seed.1 seed.2) : Nat) : Rat) *
              (alphaSupport.cardinality : Rat) *
            (((alphaSupport.cardinality : Rat)⁻¹ *
              (alphabet.cardinality : Rat)⁻¹) *
                (alphaSupport.cardinality : Rat)⁻¹) =
          (((alphaSupport.product alphabet).values.countP
              (fun seed => event seed.1 seed.2) : Nat) : Rat) *
            ((alphaSupport.cardinality : Rat) *
              (alphaSupport.cardinality : Rat)⁻¹) *
            ((alphabet.cardinality : Rat)⁻¹ *
              (alphaSupport.cardinality : Rat)⁻¹) := by
            rw [Rat.mul_assoc
              (((alphaSupport.product alphabet).values.countP
                (fun seed => event seed.1 seed.2) : Nat) : Rat)
              (alphaSupport.cardinality : Rat)]
            rw [Rat.mul_assoc
              (alphaSupport.cardinality : Rat)⁻¹
              (alphabet.cardinality : Rat)⁻¹
              (alphaSupport.cardinality : Rat)⁻¹]
            rw [← Rat.mul_assoc
              (alphaSupport.cardinality : Rat)
              (alphaSupport.cardinality : Rat)⁻¹]
            rw [← Rat.mul_assoc
              (((alphaSupport.product alphabet).values.countP
                (fun seed => event seed.1 seed.2) : Nat) : Rat)
              ((alphaSupport.cardinality : Rat) *
                (alphaSupport.cardinality : Rat)⁻¹)]
        _ = (((alphaSupport.product alphabet).values.countP
              (fun seed => event seed.1 seed.2) : Nat) : Rat) *
            ((alphabet.cardinality : Rat)⁻¹ *
              (alphaSupport.cardinality : Rat)⁻¹) := by
          rw [Rat.mul_inv_cancel _ alphaNonzero, Rat.mul_one]
    _ = _ := rfl

private theorem mixingRootEvent_run_eq_alphaGammaZeroEvent
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape)
    (witness : OutputWitness shape columns)
    (coefficientNonzero :
      Not (SignedCoefficientObject.CoefficientTruth context.extensionOps
        ((context.statement.sourceProtocolData context.lift witness
          ).toJointData context.extensionOps)))
    (seed : RunSeed Extension shape ProverSeed TargetSeed) :
    mixingRootEvent context witness (run context adversary seed) =
      alphaGammaZeroEvent context.extensionOps
        ((context.statement.sourceProtocolData context.lift witness
          ).toJointData context.extensionOps)
        (VerifierCoins.alphaWord seed.2.2,
          VerifierCoins.gamma seed.2.2) := by
  apply Bool.eq_iff_iff.mpr
  rw [mixingRootEvent_eq_true_iff]
  simp only [alphaGammaZeroEvent, decide_eq_true_eq,
    MixingFailure, run_causalRun, execute_probe_coins,
    VerifierCoins.toPublicCoins, VerifierCoins.alphaWord,
    VerifierCoins.gamma, wordPoint]
  constructor
  · exact fun root => root.sampledZero
  · exact fun sampledZero => ⟨coefficientNonzero, sampledZero⟩

/-- The exact fixed-witness mixing-root event in the complete causal
experiment obeys the finite alpha/gamma root bound. Prover and target tapes
cancel as product marginals; the later SumCheck word cancels separately. -/
theorem mixingRoot_probability_le
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (noZeroDivisors :
      FiniteRootCounting.NoZeroDivisors context.extensionOps)
    (alphabet : Support Extension)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape)
    (witness : OutputWitness shape columns) :
    (experiment context alphabet adversary).probabilityBool
        (mixingRootEvent context witness) <=
      ratio
        (shape.cubeVariables + (shape.jointCoefficientCount - 1))
        alphabet.cardinality := by
  let data :=
    (context.statement.sourceProtocolData context.lift witness).toJointData
      context.extensionOps
  by_cases coefficientTruth :
      SignedCoefficientObject.CoefficientTruth context.extensionOps data
  · have eventFalse :
        mixingRootEvent context witness =
          fun _execution => false := by
      funext execution
      cases eventValue :
          mixingRootEvent context witness execution with
      | false => rfl
      | true =>
          have root :=
            (mixingRootEvent_eq_true_iff context witness execution).mp
              eventValue
          exact False.elim (root.coefficientNonzero coefficientTruth)
    rw [eventFalse]
    have countFalse
        (values :
          List (RunSeed Extension shape ProverSeed TargetSeed)) :
        values.countP (fun _seed => false) = 0 := by
      induction values with
      | nil => rfl
      | cons _ values inductionHypothesis =>
          simp only [List.countP_cons, Bool.false_eq_true, ↓reduceIte]
          exact inductionHypothesis
    unfold Experiment.probabilityBool Experiment.countBool
    change
      (((experiment context alphabet adversary).support.values.countP
        (fun _seed => false) : Nat) : Rat) /
          ((experiment context alphabet adversary).support.cardinality : Rat) <=
        ratio
          (shape.cubeVariables + (shape.jointCoefficientCount - 1))
          alphabet.cardinality
    have countIsZero :
        (experiment context alphabet adversary).support.values.countP
          (fun _seed => false) = 0 :=
      countFalse (experiment context alphabet adversary).support.values
    rw [countIsZero]
    change
      (0 : Rat) *
          (((experiment context alphabet adversary).support.cardinality :
            Nat) : Rat)⁻¹ <=
        ratio
          (shape.cubeVariables + (shape.jointCoefficientCount - 1))
          alphabet.cardinality
    rw [Rat.zero_mul]
    unfold ratio
    rw [Rat.div_def]
    exact Rat.mul_nonneg Rat.natCast_nonneg
      (Rat.le_of_lt (Rat.inv_pos.mpr
        (Rat.natCast_pos.mpr alphabet.cardinality_pos)))
  · let verifierEvent :
        VerifierCoins.Seed Extension shape.cubeVariables -> Bool :=
      fun seed =>
        alphaGammaZeroEvent context.extensionOps data
          (VerifierCoins.alphaWord seed, VerifierCoins.gamma seed)
    have eventEquality :
        (fun seed =>
          mixingRootEvent context witness
            (run context adversary seed)) =
        (fun seed => verifierEvent seed.2.2) := by
      funext seed
      exact mixingRootEvent_run_eq_alphaGammaZeroEvent
        context adversary witness coefficientTruth seed
    change
      ((runSupport context alphabet adversary).uniform).probabilityBool
          (fun seed =>
            mixingRootEvent context witness
              (run context adversary seed)) <=
        ratio
          (shape.cubeVariables + (shape.jointCoefficientCount - 1))
          alphabet.cardinality
    rw [eventEquality]
    calc
      _ =
          ((adversary.targetSupport.product
            (VerifierCoins.support alphabet shape.cubeVariables)).uniform
            ).probabilityBool (fun seed => verifierEvent seed.2) := by
        simpa only [runSupport] using
          Support.product_uniform_probabilityBool_second
            adversary.proverSupport
            (adversary.targetSupport.product
              (VerifierCoins.support alphabet shape.cubeVariables))
            (fun seed => verifierEvent seed.2)
      _ =
          ((VerifierCoins.support alphabet shape.cubeVariables).uniform
            ).probabilityBool verifierEvent := by
        exact Support.product_uniform_probabilityBool_second
          adversary.targetSupport
          (VerifierCoins.support alphabet shape.cubeVariables)
          verifierEvent
      _ =
          let alphaSupport :=
            FiniteWords.Support.challengeVectors alphabet
              shape.cubeVariables
          (alphaSupport.product alphabet).uniform.probabilityBool
            (alphaGammaZeroEvent context.extensionOps data) := by
        simpa only [verifierEvent] using
          verifierAlphaGamma_marginal alphabet shape.cubeVariables
            (fun alpha gamma =>
              alphaGammaZeroEvent context.extensionOps data (alpha, gamma))
      _ <=
          ratio
            (shape.cubeVariables + (shape.jointCoefficientCount - 1))
            alphabet.cardinality := by
        exact alphaGammaZero_probability_le context.extensionOps
          context.extensionLaws context.extensionZeroLaws noZeroDivisors
          data alphabet coefficientTruth

/-- Concrete construction of the repository's mixing-root contract from
finite multilinear and univariate root counting. The contract is a conclusion,
not a premise. -/
theorem mixingRootProbabilityContract_of_rootCounting
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (noZeroDivisors :
      FiniteRootCounting.NoZeroDivisors context.extensionOps)
    (alphabet : Support Extension)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape) :
    MixingRootProbabilityContract context alphabet adversary
      (ratio
        (shape.cubeVariables + (shape.jointCoefficientCount - 1))
        alphabet.cardinality) := by
  intro witness
  exact mixingRoot_probability_le context noZeroDivisors alphabet adversary
    witness

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness
