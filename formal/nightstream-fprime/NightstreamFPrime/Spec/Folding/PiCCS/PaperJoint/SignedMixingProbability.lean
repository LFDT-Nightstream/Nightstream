import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanMixingProbability
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedCoefficientObject

/-!
Mixing probability for the exact signed PiCCS coefficient object. The data,
including every source witness and claimed prior evaluation, is fixed before
independent uniform alpha/gamma draws. A fixed nonzero coefficient gives the
alpha loss; the actual specialized gamma list gives the gamma loss.

This is the interactive law. No conditioning on source agreement or a later
successful execution changes these fresh coins. No Fiat–Shamir law is assumed.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedMixingProbability

open scoped BigOperators
open NightstreamFPrime.Spec SumCheck.Finite
open GoldilocksCausal ConcreteCarrier SignedCoefficientObject

attribute [local instance] Classical.propDecidable

private theorem neg_zero_iff (value : K) : extensionOps.neg value = K.zero ↔ value = K.zero := by
  constructor
  · intro zero
    change extensionOps.neg value = extensionOps.zero at zero
    have inverse := extensionLaws.add_neg value
    rw [zero, extensionLaws.add_zero] at inverse
    exact inverse
  · intro zero
    rw [zero]
    exact extensionZeroLaws.neg_zero

/-- Every alpha-dependent signed coefficient comes from one of the exact CCS
or norm tables in this data object. -/
private theorem negative_coefficient_is_table {shape : Shape}
    (data : SignedJointIdentity.JointData K shape)
    (polynomial : AlphaPolynomial K (canonicalAlphaBasis shape))
    (inside : Coefficient.negativeAlpha polynomial ∈ coefficients extensionOps data) :
    ∃ table : BooleanTable K shape.cubeVariables, polynomial = table.toAlphaPolynomial extensionOps := by
  simp only [coefficients, residuals, TableResidualData.toResiduals, List.map_map,
    Function.comp_def, List.mem_append, List.mem_map] at inside
  rcases inside with ⟨value, _member, equal⟩ | ⟨value, _member, equal⟩ |
    ⟨source, _member, equal⟩ | ⟨source, _member, equal⟩
  · cases equal
  · cases equal
  · exact ⟨data.ccs source, (Coefficient.negativeAlpha.inj equal).symm⟩
  · exact ⟨data.norm source, (Coefficient.negativeAlpha.inj equal).symm⟩

private noncomputable def coefficientZeroProbability {shape : Shape}
    (samples : Finset K) (coefficient : Coefficient K shape) : ℝ :=
  uniformAverage samples shape.cubeVariables (fun coordinates =>
    if dimension : coordinates.length = shape.cubeVariables then
      if coefficient.specialize extensionOps ⟨coordinates, dimension⟩ = K.zero then 1 else 0
    else 0)

private theorem coefficientZeroProbability_le {shape : Shape}
    (samples : Finset K) (nonempty : samples.Nonempty)
    (data : SignedJointIdentity.JointData K shape) (coefficient : Coefficient K shape)
    (inside : coefficient ∈ coefficients extensionOps data)
    (nonzero : ¬ coefficient.Zero extensionOps) :
    coefficientZeroProbability samples coefficient ≤ (shape.cubeVariables : ℝ) / samples.card := by
  cases coefficient with
  | scalar value =>
      have different : value ≠ K.zero := nonzero
      have zero : coefficientZeroProbability (shape := shape) samples (Coefficient.scalar value) = 0 := by
        unfold coefficientZeroProbability
        calc
          _ = uniformAverage samples shape.cubeVariables (fun _ => 0) := by
            apply uniformAverage_congr
            intro coordinates dimension
            simp only [dif_pos dimension, Coefficient.specialize, different, ↓reduceIte]
          _ = 0 := uniformAverage_const samples nonempty _ _
      rw [zero]
      positivity
  | negativeAlpha polynomial =>
      obtain ⟨table, rfl⟩ := negative_coefficient_is_table data polynomial inside
      have tableNonzero : ¬ table.AllEntriesZero extensionOps := by
        intro zero
        exact nonzero ((BooleanTable.toAlphaPolynomial_coefficientZero_iff_allEntriesZero
          extensionOps extensionZeroLaws table).mpr zero)
      have same : coefficientZeroProbability samples
          (Coefficient.negativeAlpha (table.toAlphaPolynomial extensionOps)) =
          BooleanMixingProbability.zeroProbability samples table := by
        apply uniformAverage_congr
        intro coordinates dimension
        simp only [dif_pos dimension, Coefficient.specialize,
          BooleanTable.toAlphaPolynomial_evaluate_eq_evaluate extensionOps extensionLaws,
          neg_zero_iff, BooleanTable.evaluate]
      rw [same]
      exact BooleanMixingProbability.zeroProbability_le samples nonempty table tableNonzero

/-- Alpha is a product of independent uniform K draws; gamma is one further
independent draw. Both are sampled after the exact signed data are fixed. -/
noncomputable def mixingProbability {shape : Shape}
    (samples : Finset K) (data : SignedJointIdentity.JointData K shape) : ℝ :=
  uniformAverage samples shape.cubeVariables (fun coordinates =>
    if dimension : coordinates.length = shape.cubeVariables then
      𝔼 gamma ∈ samples,
        if MixingRoot extensionOps data ⟨coordinates, dimension⟩ gamma then 1 else 0
    else 0)

/-- Exact signed mixing loss: gamma degree plus the alpha MLE dimension.
No unrelated polynomial or nonzero-specialization premise is supplied. -/
theorem mixingProbability_le {shape : Shape} (samples : Finset K) (nonempty : samples.Nonempty)
    (data : SignedJointIdentity.JointData K shape) :
    mixingProbability samples data ≤
      ((shape.jointCoefficientCount - 1 + shape.cubeVariables : Nat) : ℝ) / samples.card := by
  by_cases truth : CoefficientTruth extensionOps data
  · have noRoot (alpha : CubePoint K shape.cubeVariables) (gamma : K) :
        ¬ MixingRoot extensionOps data alpha gamma := fun root => root.coefficientNonzero truth
    have zero : mixingProbability samples data = 0 := by
      unfold mixingProbability
      calc
        _ = uniformAverage samples shape.cubeVariables (fun _ => 0) := by
          apply uniformAverage_congr
          intro coordinates dimension
          simp only [dif_pos dimension, noRoot, ↓reduceIte, Finset.expect_const nonempty]
        _ = 0 := uniformAverage_const samples nonempty _ _
    rw [zero]
    positivity
  · have selected : ∃ coefficient ∈ coefficients extensionOps data, ¬ coefficient.Zero extensionOps := by
      by_contra absent
      apply truth
      intro coefficient inside
      by_contra nonzero
      exact absent ⟨coefficient, inside, nonzero⟩
    obtain ⟨coefficient, inside, nonzero⟩ := selected
    have each (alpha : CubePoint K shape.cubeVariables) :
        (𝔼 gamma ∈ samples, if MixingRoot extensionOps data alpha gamma then (1 : ℝ) else 0) ≤
        (if coefficient.specialize extensionOps alpha = K.zero then 1 else 0) +
          (shape.jointCoefficientCount - 1 : Nat) / (samples.card : ℝ) := by
      by_cases zero : coefficient.specialize extensionOps alpha = K.zero
      · have atMostOne :
            (𝔼 gamma ∈ samples, if MixingRoot extensionOps data alpha gamma then (1 : ℝ) else 0) ≤ 1 := by
          apply Finset.expect_le nonempty
          intro gamma _
          split <;> norm_num
        rw [if_pos zero]
        have nonnegative : (0 : ℝ) ≤ (shape.jointCoefficientCount - 1 : Nat) / (samples.card : ℝ) := by
          positivity
        linarith
      · have specializedMember : coefficient.specialize extensionOps alpha ∈
            SignedCoefficientPolynomial.coefficients extensionOps data alpha := by
          rw [← specializedCoefficients_eq extensionOps extensionLaws]
          exact List.mem_map.mpr ⟨coefficient, inside, rfl⟩
        have gammaBound := SignedMixingRoots.signed_gamma_probability_le data alpha samples
          ⟨coefficient.specialize extensionOps alpha, specializedMember, zero⟩
        rw [if_neg zero, zero_add]
        apply le_trans _ gammaBound
        apply Finset.expect_le_expect
        intro gamma _
        by_cases root : MixingRoot extensionOps data alpha gamma
        · have sampled :
              (SignedCoefficientPolynomial.polynomial extensionOps data alpha).evaluate
                extensionOps.toOps gamma = K.zero := root.sampledZero
          simp only [if_pos root, if_pos sampled, le_refl]
        · rw [if_neg root]
          split <;> norm_num
    have averaged := uniformAverage_mono samples shape.cubeVariables
      (fun coordinates => if dimension : coordinates.length = shape.cubeVariables then
        𝔼 gamma ∈ samples,
          if MixingRoot extensionOps data ⟨coordinates, dimension⟩ gamma then 1 else 0 else 0)
      (fun coordinates =>
        (if dimension : coordinates.length = shape.cubeVariables then
          if coefficient.specialize extensionOps ⟨coordinates, dimension⟩ = K.zero then 1 else 0
          else 0) + (shape.jointCoefficientCount - 1 : Nat) / (samples.card : ℝ))
      (fun coordinates dimension => by simpa only [dif_pos dimension] using each ⟨coordinates, dimension⟩)
    apply averaged.trans
    rw [uniformAverage_add, uniformAverage_const samples nonempty]
    have alphaBound := coefficientZeroProbability_le samples nonempty data coefficient inside nonzero
    have combined := _root_.add_le_add_right alphaBound
      ((shape.jointCoefficientCount - 1 : Nat) / (samples.card : ℝ))
    simpa only [coefficientZeroProbability, Nat.cast_add, add_div, add_comm] using combined

/-- The paper's `k*d*(t+1) + 2*K + k - 1 + log2(m)` numerator over the exact
Goldilocks extension-field size. -/
theorem mixingProbability_fullField_le {shape : Shape}
    (data : SignedJointIdentity.JointData K shape) :
    mixingProbability GoldilocksRoots.fullChallengeSet data ≤
      ((shape.runningCount * shape.coefficientCount * (shape.matrixCount + 1) +
        2 * shape.freshCount + shape.runningCount - 1 + shape.cubeVariables : Nat) : ℝ) /
        (goldilocksModulus ^ 2 : Nat) := by
  have count : shape.jointCoefficientCount =
      shape.runningCount * shape.coefficientCount * (shape.matrixCount + 1) +
        2 * shape.freshCount + shape.runningCount := by
    rw [Shape.jointCoefficientCount, Shape.constraintOffset_eq, Shape.sourceCount]
    omega
  simpa only [count, GoldilocksRoots.fullChallengeSet_card] using
    mixingProbability_le GoldilocksRoots.fullChallengeSet sampleSpace_nonempty data

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedMixingProbability
