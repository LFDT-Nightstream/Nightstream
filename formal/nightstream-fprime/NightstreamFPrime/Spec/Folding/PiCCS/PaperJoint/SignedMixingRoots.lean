import NightstreamFPrime.Spec.SumCheck.GoldilocksRoots
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Data.Real.Basic

/-!
Root count for the actual signed gamma coefficient list over K. Nonzeroness
is proved from a coefficient of that list, rather than assumed as a distinct
polynomial function. This also covers degrees at or above the field size.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedMixingRoots

open scoped BigOperators
open NightstreamFPrime.Spec SumCheck.Finite
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing


private noncomputable def polynomial : List K → Polynomial K
  | [] => 0
  | coefficient :: rest => Polynomial.C coefficient + Polynomial.X * polynomial rest

private theorem polynomial_eval (coefficients : List K) (point : K) :
    (polynomial coefficients).eval point =
      Message.evaluateCoefficients GoldilocksRoots.ops point coefficients := by
  induction coefficients with
  | nil => simp only [polynomial, Polynomial.eval_zero, GoldilocksExtensionRing.zero_eq]; rfl
  | cons coefficient rest ih =>
      simp only [polynomial, Polynomial.eval_add, Polynomial.eval_C,
        Polynomial.eval_mul, Polynomial.eval_X, ih, GoldilocksExtensionRing.add_eq,
        GoldilocksExtensionRing.mul_eq]
      rfl

private theorem polynomial_degree (coefficients : List K) :
    (polynomial coefficients).natDegree ≤ coefficients.length - 1 := by
  induction coefficients with
  | nil => simp [polynomial]
  | cons coefficient rest ih =>
      cases rest with
      | nil => simp [polynomial]
      | cons next rest =>
          have bound := Polynomial.natDegree_add_le (Polynomial.C coefficient)
            (Polynomial.X * polynomial (next :: rest))
          have productBound := Polynomial.natDegree_mul_le
            (p := (Polynomial.X : Polynomial K)) (q := polynomial (next :: rest))
          simp only [Polynomial.natDegree_C, Polynomial.natDegree_X,
            List.length_cons] at bound productBound ih ⊢
          change (Polynomial.C coefficient + Polynomial.X * polynomial (next :: rest)).natDegree ≤ _
          omega

private theorem coefficients_zero_of_polynomial_zero (coefficients : List K)
    (zero : polynomial coefficients = 0) :
    ∀ coefficient ∈ coefficients, coefficient = K.zero := by
  induction coefficients with
  | nil => simp
  | cons coefficient rest ih =>
      have head : coefficient = (0 : K) := by
        simpa [polynomial] using congrArg (fun value : Polynomial K => value.coeff 0) zero
      have tail : polynomial rest = 0 := by
        have product : (Polynomial.X : Polynomial K) * polynomial rest = 0 := by
          simpa only [polynomial, head, Polynomial.C_0, zero_add] using zero
        exact (mul_eq_zero.mp product).resolve_left Polynomial.X_ne_zero
      intro value member
      rcases List.mem_cons.mp member with same | inside
      · simpa only [same, GoldilocksExtensionRing.zero_eq] using head
      · exact ih tail value inside

/-- A nonzero actual coefficient list has at most its list-derived degree
many roots in the chosen K challenge set. -/
theorem coefficient_root_count_le (coefficients : List K) (samples : Finset K)
    (nonzero : ∃ coefficient ∈ coefficients, coefficient ≠ K.zero) :
    (samples.filter fun gamma =>
      Message.evaluateCoefficients GoldilocksRoots.ops gamma coefficients = K.zero).card ≤
      coefficients.length - 1 := by
  classical
  have notZero : polynomial coefficients ≠ 0 := by
    intro zero
    obtain ⟨coefficient, inside, nonzero⟩ := nonzero
    exact nonzero (coefficients_zero_of_polynomial_zero coefficients zero coefficient inside)
  apply (Polynomial.card_le_degree_of_subset_roots (p := polynomial coefficients) ?_).trans
    (polynomial_degree coefficients)
  intro gamma member
  apply (Polynomial.mem_roots notZero).mpr
  change (polynomial coefficients).eval gamma = 0
  rw [polynomial_eval, GoldilocksExtensionRing.zero_eq]
  exact (Finset.mem_filter.mp member).2

/-- Uniform gamma is sampled after the signed list is fixed. -/
theorem coefficient_root_probability_le (coefficients : List K) (samples : Finset K)
    (nonzero : ∃ coefficient ∈ coefficients, coefficient ≠ K.zero) :
    (𝔼 gamma ∈ samples, if
      Message.evaluateCoefficients GoldilocksRoots.ops gamma coefficients = K.zero
      then (1 : ℝ) else 0) ≤ (coefficients.length - 1 : Nat) / (samples.card : ℝ) := by
  classical
  rw [Finset.expect_eq_sum_div_card, ← Finset.sum_filter]
  simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg _)
  exact_mod_cast coefficient_root_count_le coefficients samples nonzero

/-- The bound is for the protocol's own signed gamma polynomial and its
proved four-block length. -/
theorem signed_gamma_probability_le {shape : Shape}
    (data : SignedJointIdentity.JointData K shape)
    (alpha : CubePoint K shape.cubeVariables) (samples : Finset K)
    (nonzero : ∃ coefficient ∈
      SignedCoefficientPolynomial.coefficients ConcreteCarrier.extensionOps data alpha,
      coefficient ≠ K.zero) :
    (𝔼 gamma ∈ samples, if
      (SignedCoefficientPolynomial.polynomial ConcreteCarrier.extensionOps data alpha).evaluate
        ConcreteCarrier.extensionOps.toOps gamma = K.zero then (1 : ℝ) else 0) ≤
      (shape.jointCoefficientCount - 1 : Nat) / (samples.card : ℝ) := by
  simpa only [SignedCoefficientPolynomial.polynomial, Message.evaluate,
    GoldilocksRoots.ops, SignedCoefficientPolynomial.coefficients_length] using!
    coefficient_root_probability_le
      (SignedCoefficientPolynomial.coefficients ConcreteCarrier.extensionOps data alpha) samples nonzero

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedMixingRoots
