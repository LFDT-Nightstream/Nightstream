import Mathlib.FieldTheory.Finite.Basic
import NightstreamFPrime.Spec.FieldPrime
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.NoZeroDivisors

namespace NightstreamFPrime.Spec.FieldExtension

open NightstreamFPrime.Spec
open Folding.PiCCS.PaperJoint.ConcreteCarrier

set_option maxRecDepth 100000 in -- fixed-size: 64-bit Euler criterion, not artifact data
/-- Euler's criterion for 7: `7 ^ ((q − 1) / 2) = q − 1`. -/
theorem seven_euler_residue :
    FieldPrime.powMod fieldModulus FieldPrime.certificateFuel 7 ((fieldModulus - 1) / 2)
      = 18446744073709551556 := by
  decide

private theorem seven_not_square : ¬ IsSquare (7 : ZMod fieldModulus) := by
  let : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩
  rintro ⟨root, square⟩
  have rootNonzero : root ≠ 0 := by
    intro zero
    have sevenNonzero : (7 : ZMod fieldModulus) ≠ 0 := by decide
    exact sevenNonzero (by simpa [zero] using square)
  have halfPower : ((7 : Nat) : ZMod fieldModulus) ^ ((fieldModulus - 1) / 2) ≠ 1 :=
    FieldPrime.pow_ne_one_of_powMod (by decide) seven_euler_residue (by decide) (by decide)
  apply halfPower
  change (7 : ZMod fieldModulus) ^ ((fieldModulus - 1) / 2) = 1
  rw [square, ← pow_two, ← pow_mul]
  have exponent : 2 * ((fieldModulus - 1) / 2) = fieldModulus - 1 := by
    decide
  rw [exponent]
  exact ZMod.pow_card_sub_one_eq_one rootNonzero

theorem sevenProjectiveNonresidue : SevenProjectiveNonresidue := by
  let : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩
  change ∀ real imaginary : ZMod fieldModulus,
    real * real - 7 * imaginary * imaginary = 0 → real = 0 ∧ imaginary = 0
  intro real imaginary normZero
  have equation : real ^ 2 = 7 * imaginary ^ 2 := by
    simpa [pow_two, mul_assoc] using sub_eq_zero.mp normZero
  by_cases imaginaryZero : imaginary = 0
  · refine ⟨?_, imaginaryZero⟩
    have realSquareZero : real * real = 0 := by
      simpa [imaginaryZero, pow_two] using equation
    exact (mul_self_eq_zero.mp realSquareZero)
  · exfalso
    apply seven_not_square
    refine ⟨real / imaginary, ?_⟩
    rw [← pow_two, div_pow]
    exact ((div_eq_iff (pow_ne_zero 2 imaginaryZero)).mpr equation).symm

theorem extensionNoZeroDivisors : ExtensionNoZeroDivisors :=
  extensionNoZeroDivisors_of_base_and_seven
    FieldPrime.baseFieldNoZeroDivisors sevenProjectiveNonresidue

end NightstreamFPrime.Spec.FieldExtension
