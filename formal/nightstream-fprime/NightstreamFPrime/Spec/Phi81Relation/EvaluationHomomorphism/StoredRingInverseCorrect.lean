import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverse
import Mathlib.Tactic.Ring

/-! Polynomial correctness of the executed normalized-GCD candidate under
explicit coprimality. This does not identify RingF units with coprime
polynomial representatives and gives no execution-work bound. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverseCorrect

open NightstreamFPrime.Spec
open CompPoly (CPolynomial)
open StoredRingInverse

private abbrev Base := ZMod goldilocksModulus
local instance : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩

private theorem modulus_monic : (modulus ()).toPoly.Monic :=
  (CPolynomial.monic_toPoly_iff _).mp divisor_monic

theorem modulus_degree :
    (modulus ()).toPoly.degree = (ringDegree : WithBot Nat) := by
  have lower :
      (Polynomial.X ^ ringMiddleDegree + 1 : Polynomial Base).degree =
        (ringMiddleDegree : WithBot Nat) := by
    simpa only [ringMiddleDegree] using
      (Polynomial.degree_X_pow_add_C (R := Base) (n := 27) (by decide) (1 : Base))
  rw [modulus_toPolynomial, Polynomial.degree_add_eq_left_of_degree_lt]
  · exact Polynomial.degree_X_pow ringDegree
  · rw [lower, Polynomial.degree_X_pow]
    decide

/-- Monic reduction bounds the candidate's degree on every stored input. -/
theorem candidatePolynomial_degree_lt (value : StoredRing) :
    (candidatePolynomial value).toPoly.degree < (ringDegree : WithBot Nat) := by
  rw [candidatePolynomial_toPolynomial, ← modulus_degree]
  exact Polynomial.degree_modByMonic_lt _ modulus_monic

private theorem one_modulus :
    (1 : Polynomial Base) %ₘ (modulus ()).toPoly = 1 := by
  apply (Polynomial.modByMonic_eq_self_iff modulus_monic).mpr
  rw [Polynomial.degree_one, modulus_degree]
  decide

/-- Coprimality makes the cofactor from the actual normalized-GCD call a
polynomial inverse. The stored RingF unit premise is not established here. -/
theorem candidatePolynomial_mul_mod (value : StoredRing)
    (coprime : IsCoprime (encode value).toPoly (modulus ()).toPoly) :
    ((candidatePolynomial value).toPoly * (encode value).toPoly) %ₘ
      (modulus ()).toPoly = 1 := by
  let result := CPolynomial.normXgcd (encode value) (modulus ()) 0
  have gcdOne : result.1.toPoly = 1 := by
    dsimp only [result]
    rw [CPolynomial.normXgcd_fst_toPoly]
    exact normalize_eq_one.mpr (EuclideanDomain.gcd_isUnit_iff.mpr coprime)
  have bezout : (1 : Polynomial Base) =
      result.2.1.toPoly * (encode value).toPoly +
        result.2.2.toPoly * (modulus ()).toPoly := by
    rw [← gcdOne]
    exact normalized_bezout value
  have sameRemainder :
      ((candidatePolynomial value).toPoly * (encode value).toPoly) %ₘ
          (modulus ()).toPoly =
        (1 : Polynomial Base) %ₘ (modulus ()).toPoly := by
    apply Polynomial.modByMonic_eq_of_dvd_sub modulus_monic
    refine ⟨-(result.2.1.toPoly /ₘ (modulus ()).toPoly) * (encode value).toPoly -
      result.2.2.toPoly, ?_⟩
    rw [candidatePolynomial_toPolynomial, bezout, Polynomial.modByMonic_eq_sub_mul_div]
    change (result.2.1.toPoly - (modulus ()).toPoly *
        (result.2.1.toPoly /ₘ (modulus ()).toPoly)) * (encode value).toPoly -
        (result.2.1.toPoly * (encode value).toPoly +
          result.2.2.toPoly * (modulus ()).toPoly) =
      (modulus ()).toPoly *
        (-(result.2.1.toPoly /ₘ (modulus ()).toPoly) * (encode value).toPoly -
          result.2.2.toPoly)
    ring
  exact sameRemainder.trans one_modulus

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverseCorrect
