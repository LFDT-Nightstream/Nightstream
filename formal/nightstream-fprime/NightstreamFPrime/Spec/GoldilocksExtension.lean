import Mathlib.FieldTheory.Finite.Basic
import NightstreamFPrime.Spec.GoldilocksPrime
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.NoZeroDivisors

/-! Closes the quadratic-extension cancellation premise for the existing
`F = Fin q` and `K` carriers, using the checked Goldilocks certificate. -/

namespace NightstreamFPrime.Spec.GoldilocksExtension

open NightstreamFPrime.Spec
open Folding.PiCCS.PaperJoint.ConcreteCarrier

private theorem seven_not_square : ¬ IsSquare (7 : ZMod goldilocksModulus) := by
  letI : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩
  rintro ⟨root, square⟩
  have rootNonzero : root ≠ 0 := by
    intro zero
    have sevenNonzero : (7 : ZMod goldilocksModulus) ≠ 0 := by decide
    exact sevenNonzero (by simpa [zero] using square)
  apply GoldilocksPrime.order_not_halved_zmod
  change (7 : ZMod goldilocksModulus) ^ ((goldilocksModulus - 1) / 2) = 1
  rw [square, ← pow_two, ← pow_mul]
  have exponent : 2 * ((goldilocksModulus - 1) / 2) = goldilocksModulus - 1 := by
    decide
  rw [exponent]
  exact ZMod.pow_card_sub_one_eq_one rootNonzero

/-- The existing `Fin q` arithmetic is definitionally the nonzero-modulus
`ZMod q` arithmetic; no second field representation is introduced. -/
theorem sevenProjectiveNonresidue : SevenProjectiveNonresidue := by
  letI : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩
  change ∀ real imaginary : ZMod goldilocksModulus,
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

/-- Unconditional cancellation for the actual quadratic-extension carrier. -/
theorem extensionNoZeroDivisors : ExtensionNoZeroDivisors :=
  extensionNoZeroDivisors_of_base_and_seven
    GoldilocksPrime.baseFieldNoZeroDivisors sevenProjectiveNonresidue

end NightstreamFPrime.Spec.GoldilocksExtension
