import Mathlib.NumberTheory.LucasPrimality
import Mathlib.Tactic.NormNum.Prime
import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NormRange

/-! The field modulus `2^64 − 59` is prime: a Lucas certificate with generator 2 over
`q − 1 = 2² · 11 · 137 · 547 · 5594472617641`, and a Lucas certificate with generator 13 for
the factor `5594472617641`. -/

namespace NightstreamFPrime.Spec.FieldPrime

open NightstreamFPrime.Spec

def powMod (modulus : Nat) : Nat → Nat → Nat → Nat
  | 0, _, _ => 0
  | _, _, 0 => 1 % modulus
  | fuel + 1, base, exponent =>
      let half := powMod modulus fuel (base * base % modulus) (exponent / 2)
      if exponent % 2 = 1 then half * base % modulus else half

def certificateFuel : Nat := 70

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
        · rw [ite_eq_left hodd]
          simp
          rw [ih (base * base % modulus) (exponent / 2) hhalf]
          rw [hsquare, ← pow_two, ← pow_mul, ← pow_succ]
          congr
          omega

/-- A `powMod` certificate with the value 1 gives the power in `ZMod`. -/
theorem pow_eq_one_of_powMod
    {modulus base exponent : Nat}
    (exponentLt : exponent < 2 ^ (certificateFuel - 1))
    (certificate : powMod modulus certificateFuel base exponent = 1) :
    (base : ZMod modulus) ^ exponent = 1 := by
  rw [← powMod_cast modulus (certificateFuel - 1) base exponent exponentLt]
  change ((powMod modulus certificateFuel base exponent : Nat) : ZMod modulus) = 1
  rw [certificate, Nat.cast_one]

/-- A `powMod` certificate with a value above 1 gives a power that is not 1 in `ZMod`. -/
theorem pow_ne_one_of_powMod
    {modulus base exponent residue : Nat}
    (exponentLt : exponent < 2 ^ (certificateFuel - 1))
    (certificate : powMod modulus certificateFuel base exponent = residue)
    (residueLt : residue < modulus) (residueGt : 1 < residue) :
    (base : ZMod modulus) ^ exponent ≠ 1 := by
  have : Fact (1 < modulus) := ⟨by omega⟩
  rw [← powMod_cast modulus (certificateFuel - 1) base exponent exponentLt]
  change ((powMod modulus certificateFuel base exponent : Nat) : ZMod modulus) ≠ 1
  rw [certificate]
  intro equal
  have values := congrArg ZMod.val equal
  rw [ZMod.val_natCast_of_lt residueLt, ZMod.val_one] at values
  omega

private theorem dvd_mul_prime {divisor left prime : Nat}
    (divisorPrime : divisor.Prime) (primePrime : prime.Prime) (divides : divisor ∣ left * prime) :
    divisor ∣ left ∨ divisor = prime :=
  (divisorPrime.dvd_mul.mp divides).imp id (Nat.prime_dvd_prime_iff_eq divisorPrime primePrime).mp

private theorem eq_two_of_dvd_two_pow {divisor exponent : Nat}
    (divisorPrime : divisor.Prime) (divides : divisor ∣ 2 ^ exponent) : divisor = 2 :=
  (Nat.prime_dvd_prime_iff_eq divisorPrime Nat.prime_two).mp (divisorPrime.dvd_of_dvd_pow divides)

/-! ### The factor `5594472617641` of `q − 1` -/

def largeOrderFactor : Nat := 5594472617641

def largeOrderFactorGenerator : Nat := 13

theorem largeOrderFactor_order_factorisation :
    largeOrderFactor - 1 = 2 ^ 3 * 3 * 5 * 1427 * 2131 * 15331 := by
  decide

set_option maxRecDepth 100000 in -- fixed-size: 43-bit Lucas certificate, not artifact data
theorem largeOrderFactor_fermat :
    powMod largeOrderFactor certificateFuel largeOrderFactorGenerator (largeOrderFactor - 1) = 1 := by
  decide

set_option maxRecDepth 100000 in -- fixed-size: 43-bit Lucas certificate, not artifact data
theorem largeOrderFactor_order_residues :
    powMod largeOrderFactor certificateFuel largeOrderFactorGenerator ((largeOrderFactor - 1) / 2)
        = 5594472617640
      ∧ powMod largeOrderFactor certificateFuel largeOrderFactorGenerator
          ((largeOrderFactor - 1) / 3) = 633520002390
      ∧ powMod largeOrderFactor certificateFuel largeOrderFactorGenerator
          ((largeOrderFactor - 1) / 5) = 5539104407118
      ∧ powMod largeOrderFactor certificateFuel largeOrderFactorGenerator
          ((largeOrderFactor - 1) / 1427) = 693744849281
      ∧ powMod largeOrderFactor certificateFuel largeOrderFactorGenerator
          ((largeOrderFactor - 1) / 2131) = 2614630137118
      ∧ powMod largeOrderFactor certificateFuel largeOrderFactorGenerator
          ((largeOrderFactor - 1) / 15331) = 4170830691369 := by
  decide

theorem largeOrderFactor_prime_divisor
    {divisor : Nat} (divisorPrime : divisor.Prime) (divides : divisor ∣ largeOrderFactor - 1) :
    divisor = 2 ∨ divisor = 3 ∨ divisor = 5 ∨ divisor = 1427 ∨ divisor = 2131 ∨
      divisor = 15331 := by
  rw [largeOrderFactor_order_factorisation] at divides
  rcases dvd_mul_prime divisorPrime (by norm_num) divides with divides | equal
  · rcases dvd_mul_prime divisorPrime (by norm_num) divides with divides | equal
    · rcases dvd_mul_prime divisorPrime (by norm_num) divides with divides | equal
      · rcases dvd_mul_prime divisorPrime (by norm_num) divides with divides | equal
        · rcases dvd_mul_prime divisorPrime (by norm_num) divides with divides | equal
          · exact Or.inl (eq_two_of_dvd_two_pow divisorPrime divides)
          · exact Or.inr (Or.inl equal)
        · exact Or.inr (Or.inr (Or.inl equal))
      · exact Or.inr (Or.inr (Or.inr (Or.inl equal)))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl equal))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr equal))))

theorem largeOrderFactor_natPrime : Nat.Prime largeOrderFactor := by
  apply lucas_primality largeOrderFactor (largeOrderFactorGenerator : ZMod largeOrderFactor)
    (pow_eq_one_of_powMod (by decide) largeOrderFactor_fermat)
  intro divisor divisorPrime divides
  obtain ⟨half, third, fifth, first, second, last⟩ := largeOrderFactor_order_residues
  rcases largeOrderFactor_prime_divisor divisorPrime divides with
    rfl | rfl | rfl | rfl | rfl | rfl
  · exact pow_ne_one_of_powMod (by decide) half (by decide) (by decide)
  · exact pow_ne_one_of_powMod (by decide) third (by decide) (by decide)
  · exact pow_ne_one_of_powMod (by decide) fifth (by decide) (by decide)
  · exact pow_ne_one_of_powMod (by decide) first (by decide) (by decide)
  · exact pow_ne_one_of_powMod (by decide) second (by decide) (by decide)
  · exact pow_ne_one_of_powMod (by decide) last (by decide) (by decide)

/-! ### The field modulus -/

def generator : Nat := 2

theorem order_factorisation :
    fieldModulus - 1 = 2 ^ 2 * 11 * 137 * 547 * largeOrderFactor := by
  decide

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem fermat :
    powMod fieldModulus certificateFuel generator (fieldModulus - 1) = 1 := by
  decide

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_not_halved :
    powMod fieldModulus certificateFuel generator ((fieldModulus - 1) / 2)
      = 18446744073709551556 := by
  decide

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Lucas certificate, not artifact data
theorem order_residues :
    powMod fieldModulus certificateFuel generator ((fieldModulus - 1) / 11)
        = 7233656613878036887
      ∧ powMod fieldModulus certificateFuel generator ((fieldModulus - 1) / 137)
        = 808296807897363856
      ∧ powMod fieldModulus certificateFuel generator ((fieldModulus - 1) / 547)
        = 16285151747536493802
      ∧ powMod fieldModulus certificateFuel generator ((fieldModulus - 1) / largeOrderFactor)
        = 11493017446059487016 := by
  decide

theorem fermat_zmod :
    ((generator : ZMod fieldModulus) ^ (fieldModulus - 1)) = 1 :=
  pow_eq_one_of_powMod (by decide) fermat

theorem order_not_halved_zmod :
    ((generator : ZMod fieldModulus) ^ ((fieldModulus - 1) / 2)) ≠ 1 :=
  pow_ne_one_of_powMod (by decide) order_not_halved (by decide) (by decide)

theorem prime_divisor_order
    {divisor : Nat} (divisorPrime : divisor.Prime) (divides : divisor ∣ fieldModulus - 1) :
    divisor = 2 ∨ divisor = 11 ∨ divisor = 137 ∨ divisor = 547 ∨ divisor = largeOrderFactor := by
  rw [order_factorisation] at divides
  rcases dvd_mul_prime divisorPrime largeOrderFactor_natPrime divides with divides | equal
  · rcases dvd_mul_prime divisorPrime (by norm_num) divides with divides | equal
    · rcases dvd_mul_prime divisorPrime (by norm_num) divides with divides | equal
      · rcases dvd_mul_prime divisorPrime (by norm_num) divides with divides | equal
        · exact Or.inl (eq_two_of_dvd_two_pow divisorPrime divides)
        · exact Or.inr (Or.inl equal)
      · exact Or.inr (Or.inr (Or.inl equal))
    · exact Or.inr (Or.inr (Or.inr (Or.inl equal)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr equal)))

theorem fieldModulus_natPrime : Nat.Prime fieldModulus := by
  apply lucas_primality fieldModulus (generator : ZMod fieldModulus) fermat_zmod
  intro divisor divisorPrime divides
  obtain ⟨eleventh, first, second, last⟩ := order_residues
  rcases prime_divisor_order divisorPrime divides with rfl | rfl | rfl | rfl | rfl
  · exact order_not_halved_zmod
  · exact pow_ne_one_of_powMod (by decide) eleventh (by decide) (by decide)
  · exact pow_ne_one_of_powMod (by decide) first (by decide) (by decide)
  · exact pow_ne_one_of_powMod (by decide) second (by decide) (by decide)
  · exact pow_ne_one_of_powMod (by decide) last (by decide) (by decide)

theorem fieldModulus_euclidPrime : Folding.PiCCS.PaperJoint.NormRange.FieldModulusEuclid := by
  intro a b hproduct
  have hdivides : fieldModulus ∣ a * b := Nat.dvd_of_mod_eq_zero hproduct
  rcases fieldModulus_natPrime.dvd_mul.mp hdivides with ha | hb
  · exact Or.inl (Nat.mod_eq_zero_of_dvd ha)
  · exact Or.inr (Nat.mod_eq_zero_of_dvd hb)

theorem baseFieldNoZeroDivisors : Folding.PiCCS.PaperJoint.NormRange.BaseFieldNoZeroDivisors :=
  Folding.PiCCS.PaperJoint.NormRange.baseFieldNoZeroDivisors_of_modulusEuclid fieldModulus_euclidPrime

end NightstreamFPrime.Spec.FieldPrime
