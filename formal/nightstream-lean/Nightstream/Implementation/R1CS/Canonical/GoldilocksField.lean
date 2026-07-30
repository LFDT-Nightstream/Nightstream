import Mathlib.NumberTheory.LucasPrimality
import Mathlib.Tactic.NormNum.Prime
import Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest
import Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate

/-!
Contract: close the Goldilocks field boundary from Lean-owned certificate data.

Owns the proof that the Goldilocks modulus is prime, the project-local
`EuclidPrime` theorem, and the global inverse witness used by honest canonical
assignments.

Mathlib supplies Lucas's theorem and finite-field inversion. It is proof
infrastructure, not protocol authority. The modulus, generator, factorisation,
and modular exponentiation residues come from `GoldilocksCertificate`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.GoldilocksField

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate

/-- The certificate evaluator agrees with exponentiation in `ZMod` whenever
the fuel strictly exceeds the exponent bit length. -/
theorem powMod_cast
    (modulus fuel base exponent : Nat)
    (hexponent : exponent < 2 ^ fuel) :
    ((powMod modulus (fuel + 1) base exponent : Nat) : ZMod modulus) =
      (base : ZMod modulus) ^ exponent := by
  induction fuel generalizing base exponent with
  | zero =>
      have hexponent_zero : exponent = 0 := by
        simpa using hexponent
      subst exponent
      simp [powMod]
  | succ fuel ih =>
      by_cases hexponent_zero : exponent = 0
      · subst exponent
        simp [powMod]
      · have hhalf : exponent / 2 < 2 ^ fuel := by
          apply (Nat.div_lt_iff_lt_mul (by decide : 0 < 2)).2
          simpa [pow_succ, Nat.mul_comm] using hexponent
        rw [show fuel + 1 + 1 = (fuel + 1) + 1 by omega]
        simp only [powMod]
        have hsquare :
            ((base * base % modulus : Nat) : ZMod modulus) =
              (base : ZMod modulus) * (base : ZMod modulus) := by
          simp
        rcases Nat.mod_two_eq_zero_or_one exponent with heven | hodd
        · simp [heven]
          rw [ih (base * base % modulus) (exponent / 2) hhalf]
          rw [hsquare, ← pow_two, ← pow_mul]
          congr
          omega
        · rw [if_pos hodd]
          simp
          rw [ih (base * base % modulus) (exponent / 2) hhalf]
          rw [hsquare, ← pow_two, ← pow_mul, ← pow_succ]
          congr
          omega

theorem fermat_zmod :
    ((generator : ZMod goldilocksP) ^ (goldilocksP - 1)) = 1 := by
  rw [← powMod_cast goldilocksP 69 generator (goldilocksP - 1) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksP 70 generator (goldilocksP - 1) = 1 := by
    simpa [certificateFuel] using fermat
  rw [hcertificate]
  simp

private theorem zmod_natCast_ne_one
    {value : Nat} (hvalue : value < goldilocksP) (hne : value ≠ 1) :
    (value : ZMod goldilocksP) ≠ 1 := by
  letI : Fact (1 < goldilocksP) := ⟨by decide⟩
  intro hequal
  have hvalues := congrArg ZMod.val hequal
  rw [ZMod.val_natCast_of_lt hvalue] at hvalues
  have hone : (1 : ZMod goldilocksP).val = 1 := by
    exact ZMod.val_one goldilocksP
  rw [hone] at hvalues
  exact hne hvalues

theorem order_not_halved_zmod :
    ((generator : ZMod goldilocksP) ^ ((goldilocksP - 1) / 2)) ≠ 1 := by
  rw [← powMod_cast goldilocksP 69 generator ((goldilocksP - 1) / 2) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksP 70 generator ((goldilocksP - 1) / 2) =
        18446744069414584320 := by
    simpa [certificateFuel] using order_not_halved
  rw [hcertificate]
  exact zmod_natCast_ne_one (by decide) (by decide)

theorem order_not_thirded_zmod :
    ((generator : ZMod goldilocksP) ^ ((goldilocksP - 1) / 3)) ≠ 1 := by
  rw [← powMod_cast goldilocksP 69 generator ((goldilocksP - 1) / 3) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksP 70 generator ((goldilocksP - 1) / 3) =
        18446744065119617025 := by
    simpa [certificateFuel] using order_not_thirded
  rw [hcertificate]
  exact zmod_natCast_ne_one (by decide) (by decide)

theorem order_not_fifthed_zmod :
    ((generator : ZMod goldilocksP) ^ ((goldilocksP - 1) / 5)) ≠ 1 := by
  rw [← powMod_cast goldilocksP 69 generator ((goldilocksP - 1) / 5) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksP 70 generator ((goldilocksP - 1) / 5) =
        1373043270956696022 := by
    simpa [certificateFuel] using order_not_fifthed
  rw [hcertificate]
  exact zmod_natCast_ne_one (by decide) (by decide)

theorem order_not_seventeenthed_zmod :
    ((generator : ZMod goldilocksP) ^ ((goldilocksP - 1) / 17)) ≠ 1 := by
  rw [← powMod_cast goldilocksP 69 generator ((goldilocksP - 1) / 17) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksP 70 generator ((goldilocksP - 1) / 17) =
        16301593560560007290 := by
    simpa [certificateFuel] using order_not_seventeenthed
  rw [hcertificate]
  exact zmod_natCast_ne_one (by decide) (by decide)

theorem order_not_two_five_seventhed_zmod :
    ((generator : ZMod goldilocksP) ^ ((goldilocksP - 1) / 257)) ≠ 1 := by
  rw [← powMod_cast goldilocksP 69 generator ((goldilocksP - 1) / 257) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksP 70 generator ((goldilocksP - 1) / 257) =
        995085315851368103 := by
    simpa [certificateFuel] using order_not_two_five_seventhed
  rw [hcertificate]
  exact zmod_natCast_ne_one (by decide) (by decide)

theorem order_not_sixtyfive_five_three_seventhed_zmod :
    ((generator : ZMod goldilocksP) ^ ((goldilocksP - 1) / 65537)) ≠ 1 := by
  rw [← powMod_cast goldilocksP 69 generator ((goldilocksP - 1) / 65537) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksP 70 generator ((goldilocksP - 1) / 65537) =
        8478886009461009681 := by
    simpa [certificateFuel] using order_not_sixtyfive_five_three_seventhed
  rw [hcertificate]
  exact zmod_natCast_ne_one (by decide) (by decide)

theorem prime_divisor_order
    {q : Nat} (hq : q.Prime) (hdivides : q ∣ goldilocksP - 1) :
    q = 2 ∨ q = 3 ∨ q = 5 ∨ q = 17 ∨ q = 257 ∨ q = 65537 := by
  rw [order_factorisation] at hdivides
  rcases hq.dvd_mul.mp hdivides with hleft | h65537
  · rcases hq.dvd_mul.mp hleft with hleft | h257
    · rcases hq.dvd_mul.mp hleft with hleft | h17
      · rcases hq.dvd_mul.mp hleft with hleft | h5
        · rcases hq.dvd_mul.mp hleft with h2pow | h3
          · have h2 : q ∣ 2 := hq.dvd_of_dvd_pow h2pow
            rcases (Nat.dvd_prime Nat.prime_two).mp h2 with hq_one | hq_two
            · exact (hq.ne_one hq_one).elim
            · exact Or.inl hq_two
          · rcases (Nat.dvd_prime (by norm_num : Nat.Prime 3)).mp h3 with
              hq_one | hq_three
            · exact (hq.ne_one hq_one).elim
            · exact Or.inr (Or.inl hq_three)
        · rcases (Nat.dvd_prime (by norm_num : Nat.Prime 5)).mp h5 with
            hq_one | hq_five
          · exact (hq.ne_one hq_one).elim
          · exact Or.inr (Or.inr (Or.inl hq_five))
      · rcases (Nat.dvd_prime (by norm_num : Nat.Prime 17)).mp h17 with
          hq_one | hq_seventeen
        · exact (hq.ne_one hq_one).elim
        · exact Or.inr (Or.inr (Or.inr (Or.inl hq_seventeen)))
    · rcases (Nat.dvd_prime (by norm_num : Nat.Prime 257)).mp h257 with
        hq_one | hq_257
      · exact (hq.ne_one hq_one).elim
      · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl hq_257))))
  · rcases (Nat.dvd_prime (by norm_num : Nat.Prime 65537)).mp h65537 with
      hq_one | hq_65537
    · exact (hq.ne_one hq_one).elim
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr hq_65537))))

/-- Primality follows from Lucas's theorem applied to the exact certificate. -/
theorem goldilocks_natPrime : Nat.Prime goldilocksP := by
  apply lucas_primality goldilocksP (generator : ZMod goldilocksP) fermat_zmod
  intro q hq hdivides
  rcases prime_divisor_order hq hdivides with
      hq_two | hq_three | hq_five | hq_seventeen | hq_257 | hq_65537
  · subst q
    exact order_not_halved_zmod
  · subst q
    exact order_not_thirded_zmod
  · subst q
    exact order_not_fifthed_zmod
  · subst q
    exact order_not_seventeenthed_zmod
  · subst q
    exact order_not_two_five_seventhed_zmod
  · subst q
    exact order_not_sixtyfive_five_three_seventhed_zmod

/-- The project-local divisor property follows from primality. -/
theorem goldilocks_euclidPrime : EuclidPrime goldilocksP := by
  intro a b hproduct
  have hdivides : goldilocksP ∣ a * b := Nat.dvd_of_mod_eq_zero hproduct
  rcases goldilocks_natPrime.dvd_mul.mp hdivides with ha | hb
  · exact Or.inl (Nat.mod_eq_zero_of_dvd ha)
  · exact Or.inr (Nat.mod_eq_zero_of_dvd hb)

private instance goldilocksPrimeFact : Fact (Nat.Prime goldilocksP) :=
  ⟨goldilocks_natPrime⟩

def goldilocksInverseValue (value : Nat) : Nat :=
  ((value : ZMod goldilocksP)⁻¹).val

theorem goldilocksInverseValue_canonical (value : Nat) :
    goldilocksInverseValue value < goldilocksP :=
  ZMod.val_lt _

theorem goldilocksInverseValue_zero :
    goldilocksInverseValue 0 = 0 := by
  simp [goldilocksInverseValue]

theorem goldilocksInverseValue_correct
    (value : Nat) (hvalue : value < goldilocksP) (hne : value ≠ 0) :
    value * goldilocksInverseValue value % goldilocksP = 1 := by
  have hcast_ne : (value : ZMod goldilocksP) ≠ 0 := by
    intro hzero
    have hvalues := congrArg ZMod.val hzero
    rw [ZMod.val_natCast_of_lt hvalue, ZMod.val_zero] at hvalues
    exact hne hvalues
  have hinverse :
      (value : ZMod goldilocksP) * (value : ZMod goldilocksP)⁻¹ = 1 :=
    mul_inv_cancel₀ hcast_ne
  have hvalues := congrArg ZMod.val hinverse
  rw [ZMod.val_mul, ZMod.val_natCast_of_lt hvalue] at hvalues
  have hone : (1 : ZMod goldilocksP).val = 1 := by
    exact ZMod.val_one goldilocksP
  rw [hone] at hvalues
  exact hvalues

/-- One global inverse witness for every honest canonical assignment. -/
def goldilocksFieldInverse :
    CanonicalU64RecipeHonest.FieldInverse where
  inverse := goldilocksInverseValue
  canonical := goldilocksInverseValue_canonical
  zero := goldilocksInverseValue_zero
  correct := goldilocksInverseValue_correct

end Nightstream.Implementation.R1CS.Canonical.GoldilocksField
