import NightstreamFPrime.Spec.Folding.Nifs.VerifierCoinSpace
import Mathlib.Data.ENNReal.BigOperators

/-!
The finite mathematical PMF for the existing independent-uniform PiCCS
verifier mean. Its masses are that mean's singleton indicators; every real
test function has the same mean. This is not an executable sampler or a
runtime or cryptographic assumption.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.VerifierCoinLaw

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction

attribute [local instance] Classical.propDecidable

private noncomputable def indicator {shape : Shape}
    (selected : VerifierCoinSpace.Request shape)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) : ℝ :=
  if VerifierCoinSpace.request ⟨alpha, gamma, point⟩ = selected then 1 else 0

private noncomputable def weight {shape : Shape}
    (selected : VerifierCoinSpace.Request shape) : ℝ :=
  StrongProbability.verifierMean (indicator selected)

private theorem single_indicator_sum {Index : Type*} [Fintype Index] [DecidableEq Index]
    (selected : Index) (value : Index → ℝ) :
    (∑ index, (if selected = index then (1 : ℝ) else 0) * value index) = value selected := by
  classical
  rw [Finset.sum_eq_single selected]
  · simp
  · intro index _ different
    rw [if_neg (Ne.symm different), zero_mul]
  · intro absent
    exact False.elim (absent (Finset.mem_univ selected))

private theorem weight_nonnegative {shape : Shape}
    (selected : VerifierCoinSpace.Request shape) : 0 ≤ weight selected := by
  apply (StrongProbability.verifierMean_range (indicator selected) 1 ?_).1
  intro alpha gamma point
  unfold indicator
  split <;> norm_num

private theorem weight_sum (shape : Shape) :
    (∑ selected : VerifierCoinSpace.Request shape, weight selected) = 1 := by
  classical
  have partition :
      (fun alpha gamma point =>
        ∑ selected : VerifierCoinSpace.Request shape, indicator selected alpha gamma point) =
      (fun _ _ _ => (1 : ℝ)) := by
    funext alpha gamma point
    simpa only [indicator, mul_one] using
      single_indicator_sum (VerifierCoinSpace.request ⟨alpha, gamma, point⟩) (fun _ => 1)
  calc
    _ = StrongProbability.verifierMean (fun alpha gamma point =>
        ∑ selected : VerifierCoinSpace.Request shape, indicator selected alpha gamma point) :=
      (StrongProbability.verifierMean_sum Finset.univ indicator).symm
    _ = 1 := by rw [partition, StrongProbability.verifierMean_const]

/-- The PMF realizes the existing verifier mean on its finite request space.
No enumeration or sampling of that space is part of the protocol program. -/
noncomputable def law (shape : Shape) : PMF (VerifierCoinSpace.Request shape) := by
  classical
  refine PMF.ofFintype (fun selected => ENNReal.ofReal (weight selected)) ?_
  rw [← ENNReal.ofReal_sum_of_nonneg (fun selected _ => weight_nonnegative selected),
    weight_sum, ENNReal.ofReal_one]

private theorem law_toReal {shape : Shape} (selected : VerifierCoinSpace.Request shape) :
    (law shape selected).toReal = weight selected := by
  change (ENNReal.ofReal (weight selected)).toReal = weight selected
  exact ENNReal.toReal_ofReal (weight_nonnegative selected)

private theorem value_expansion {shape : Shape}
    (value : CubePoint K shape.cubeVariables → K →
      CubePoint K shape.cubeVariables → ℝ)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) :
    value alpha gamma point =
      ∑ selected : VerifierCoinSpace.Request shape, indicator selected alpha gamma point *
        value (VerifierCoinSpace.coins selected).alpha (VerifierCoinSpace.coins selected).gamma
          (VerifierCoinSpace.coins selected).roundPoint := by
  classical
  simpa only [indicator, VerifierCoinSpace.coins_request] using
    (single_indicator_sum (VerifierCoinSpace.request ⟨alpha, gamma, point⟩)
      (fun selected => value (VerifierCoinSpace.coins selected).alpha
        (VerifierCoinSpace.coins selected).gamma (VerifierCoinSpace.coins selected).roundPoint)).symm

/-- Every real test function has its original independent-uniform verifier
mean under this PMF. No boundedness premise or new coin law is assumed. -/
theorem verifierMean_eq_requestMean {shape : Shape}
    (value : CubePoint K shape.cubeVariables → K →
      CubePoint K shape.cubeVariables → ℝ) :
    StrongProbability.verifierMean value =
      ∑' selected, (law shape selected).toReal *
        value (VerifierCoinSpace.coins selected).alpha (VerifierCoinSpace.coins selected).gamma
          (VerifierCoinSpace.coins selected).roundPoint := by
  classical
  rw [tsum_fintype]
  simp only [law_toReal]
  calc
    _ = StrongProbability.verifierMean (fun alpha gamma point =>
        ∑ selected : VerifierCoinSpace.Request shape, indicator selected alpha gamma point *
          value (VerifierCoinSpace.coins selected).alpha (VerifierCoinSpace.coins selected).gamma
            (VerifierCoinSpace.coins selected).roundPoint) := by
      congr 1
      funext alpha gamma point
      exact value_expansion value alpha gamma point
    _ = ∑ selected : VerifierCoinSpace.Request shape,
        StrongProbability.verifierMean (fun alpha gamma point =>
          indicator selected alpha gamma point *
            value (VerifierCoinSpace.coins selected).alpha (VerifierCoinSpace.coins selected).gamma
              (VerifierCoinSpace.coins selected).roundPoint) :=
      StrongProbability.verifierMean_sum Finset.univ _
    _ = _ := by
      apply Finset.sum_congr rfl
      intro selected _
      exact StrongProbability.verifierMean_mul_const (indicator selected)
        (value (VerifierCoinSpace.coins selected).alpha (VerifierCoinSpace.coins selected).gamma
          (VerifierCoinSpace.coins selected).roundPoint)

end NightstreamFPrime.Spec.Folding.Nifs.VerifierCoinLaw
