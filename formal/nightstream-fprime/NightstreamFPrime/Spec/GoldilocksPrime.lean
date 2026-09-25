import Mathlib.NumberTheory.LucasPrimality
import Mathlib.Tactic.NormNum.Prime
import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NormRange

/-!
Owns the kernel-checked Lucas certificate that the Goldilocks modulus is prime
and the derived `EuclidPrime` / `BaseFieldNoZeroDivisors` facts the NIFS key
consumes. The modular exponentiation is fuel-structural so the kernel reduces
it; every residue is stated exactly so fuel exhaustion (poison `0`) fails.
Cost axis: 64-bit exponents, a protocol constant.

Provenance: adapted from
`formal/nightstream-lean/Nightstream/Implementation/R1CS/Canonical/GoldilocksCertificate.lean`
and `.../GoldilocksField.lean` at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; the R1CS and inverse-witness
dependencies are not copied.
-/

namespace NightstreamFPrime.Spec.GoldilocksPrime

open NightstreamFPrime.Spec

/-- Modular exponentiation by repeated squaring, structural on `fuel` so the
kernel can reduce it. Fuel exhaustion returns the poison value `0`. -/
def powMod (modulus : Nat) : Nat → Nat → Nat → Nat
  | 0, _, _ => 0
  | _, _, 0 => 1 % modulus
  | fuel + 1, base, exponent =>
      let half := powMod modulus fuel (base * base % modulus) (exponent / 2)
      if exponent % 2 = 1 then half * base % modulus else half

/-- Fuel large enough for any exponent below `2 ^ 70`. -/
def certificateFuel : Nat := 70

/-- The proposed generator. -/
def generator : Nat := 7

/-! ## The factorisation of `q - 1`

Fully known, which is what makes a Lucas certificate available at all. -/

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_factorisation :
    goldilocksModulus - 1 = 2 ^ 32 * 3 * 5 * 17 * 257 * 65537 := by
  decide

/-! ## The certificate

`7 ^ (q - 1) = 1`, and `7 ^ ((q - 1) / p)` is a specific residue other than `1`
for each prime `p` dividing `q - 1`. Together these say the order of `7` is
exactly `q - 1`.

Each is stated as an exact residue, not as `≠ 1`, so that a poisoned fuel value
cannot satisfy it. -/

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem fermat :
    powMod goldilocksModulus certificateFuel generator (goldilocksModulus - 1) = 1 := by
  decide

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_halved :
    powMod goldilocksModulus certificateFuel generator ((goldilocksModulus - 1) / 2)
      = 18446744069414584320 := by
  decide

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_thirded :
    powMod goldilocksModulus certificateFuel generator ((goldilocksModulus - 1) / 3)
      = 18446744065119617025 := by
  decide

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_fifthed :
    powMod goldilocksModulus certificateFuel generator ((goldilocksModulus - 1) / 5)
      = 1373043270956696022 := by
  decide

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_seventeenthed :
    powMod goldilocksModulus certificateFuel generator ((goldilocksModulus - 1) / 17)
      = 16301593560560007290 := by
  decide

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_two_five_seventhed :
    powMod goldilocksModulus certificateFuel generator ((goldilocksModulus - 1) / 257)
      = 995085315851368103 := by
  decide

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_sixtyfive_five_three_seventhed :
    powMod goldilocksModulus certificateFuel generator ((goldilocksModulus - 1) / 65537)
      = 8478886009461009681 := by
  decide

/-! ## The six residues are not one

Immediate from the exact values, and stated separately because it is the form
Lucas's theorem consumes. -/

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem certificate_residues_ne_one :
    (18446744069414584320 : Nat) ≠ 1
      ∧ (18446744065119617025 : Nat) ≠ 1
      ∧ (1373043270956696022 : Nat) ≠ 1
      ∧ (16301593560560007290 : Nat) ≠ 1
      ∧ (995085315851368103 : Nat) ≠ 1
      ∧ (8478886009461009681 : Nat) ≠ 1 := by
  decide


set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
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

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem fermat_zmod :
    ((generator : ZMod goldilocksModulus) ^ (goldilocksModulus - 1)) = 1 := by
  rw [← powMod_cast goldilocksModulus 69 generator (goldilocksModulus - 1) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksModulus 70 generator (goldilocksModulus - 1) = 1 := by
    simpa [certificateFuel] using fermat
  rw [hcertificate]
  simp

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
private theorem zmod_natCast_ne_one
    {value : Nat} (hvalue : value < goldilocksModulus) (hne : value ≠ 1) :
    (value : ZMod goldilocksModulus) ≠ 1 := by
  letI : Fact (1 < goldilocksModulus) := ⟨by decide⟩
  intro hequal
  have hvalues := congrArg ZMod.val hequal
  rw [ZMod.val_natCast_of_lt hvalue] at hvalues
  have hone : (1 : ZMod goldilocksModulus).val = 1 := by
    exact ZMod.val_one goldilocksModulus
  rw [hone] at hvalues
  exact hne hvalues

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_halved_zmod :
    ((generator : ZMod goldilocksModulus) ^ ((goldilocksModulus - 1) / 2)) ≠ 1 := by
  rw [← powMod_cast goldilocksModulus 69 generator ((goldilocksModulus - 1) / 2) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksModulus 70 generator ((goldilocksModulus - 1) / 2) =
        18446744069414584320 := by
    simpa [certificateFuel] using order_not_halved
  rw [hcertificate]
  exact zmod_natCast_ne_one (by decide) (by decide)

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_thirded_zmod :
    ((generator : ZMod goldilocksModulus) ^ ((goldilocksModulus - 1) / 3)) ≠ 1 := by
  rw [← powMod_cast goldilocksModulus 69 generator ((goldilocksModulus - 1) / 3) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksModulus 70 generator ((goldilocksModulus - 1) / 3) =
        18446744065119617025 := by
    simpa [certificateFuel] using order_not_thirded
  rw [hcertificate]
  exact zmod_natCast_ne_one (by decide) (by decide)

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_fifthed_zmod :
    ((generator : ZMod goldilocksModulus) ^ ((goldilocksModulus - 1) / 5)) ≠ 1 := by
  rw [← powMod_cast goldilocksModulus 69 generator ((goldilocksModulus - 1) / 5) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksModulus 70 generator ((goldilocksModulus - 1) / 5) =
        1373043270956696022 := by
    simpa [certificateFuel] using order_not_fifthed
  rw [hcertificate]
  exact zmod_natCast_ne_one (by decide) (by decide)

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_seventeenthed_zmod :
    ((generator : ZMod goldilocksModulus) ^ ((goldilocksModulus - 1) / 17)) ≠ 1 := by
  rw [← powMod_cast goldilocksModulus 69 generator ((goldilocksModulus - 1) / 17) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksModulus 70 generator ((goldilocksModulus - 1) / 17) =
        16301593560560007290 := by
    simpa [certificateFuel] using order_not_seventeenthed
  rw [hcertificate]
  exact zmod_natCast_ne_one (by decide) (by decide)

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_two_five_seventhed_zmod :
    ((generator : ZMod goldilocksModulus) ^ ((goldilocksModulus - 1) / 257)) ≠ 1 := by
  rw [← powMod_cast goldilocksModulus 69 generator ((goldilocksModulus - 1) / 257) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksModulus 70 generator ((goldilocksModulus - 1) / 257) =
        995085315851368103 := by
    simpa [certificateFuel] using order_not_two_five_seventhed
  rw [hcertificate]
  exact zmod_natCast_ne_one (by decide) (by decide)

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_sixtyfive_five_three_seventhed_zmod :
    ((generator : ZMod goldilocksModulus) ^ ((goldilocksModulus - 1) / 65537)) ≠ 1 := by
  rw [← powMod_cast goldilocksModulus 69 generator ((goldilocksModulus - 1) / 65537) (by decide)]
  norm_num only [Nat.reduceAdd]
  have hcertificate :
      powMod goldilocksModulus 70 generator ((goldilocksModulus - 1) / 65537) =
        8478886009461009681 := by
    simpa [certificateFuel] using order_not_sixtyfive_five_three_seventhed
  rw [hcertificate]
  exact zmod_natCast_ne_one (by decide) (by decide)

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem prime_divisor_order
    {q : Nat} (hq : q.Prime) (hdivides : q ∣ goldilocksModulus - 1) :
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

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
/-- Primality follows from Lucas's theorem applied to the exact certificate. -/
theorem goldilocks_natPrime : Nat.Prime goldilocksModulus := by
  apply lucas_primality goldilocksModulus (generator : ZMod goldilocksModulus) fermat_zmod
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

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
/-- The project-local divisor property follows from primality. -/
theorem goldilocks_euclidPrime : Folding.PiCCS.PaperJoint.NormRange.GoldilocksModulusEuclid := by
  intro a b hproduct
  have hdivides : goldilocksModulus ∣ a * b := Nat.dvd_of_mod_eq_zero hproduct
  rcases goldilocks_natPrime.dvd_mul.mp hdivides with ha | hb
  · exact Or.inl (Nat.mod_eq_zero_of_dvd ha)
  · exact Or.inr (Nat.mod_eq_zero_of_dvd hb)


set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem baseFieldNoZeroDivisors : Folding.PiCCS.PaperJoint.NormRange.BaseFieldNoZeroDivisors :=
  Folding.PiCCS.PaperJoint.NormRange.baseFieldNoZeroDivisors_of_modulusEuclid goldilocks_euclidPrime

end NightstreamFPrime.Spec.GoldilocksPrime
