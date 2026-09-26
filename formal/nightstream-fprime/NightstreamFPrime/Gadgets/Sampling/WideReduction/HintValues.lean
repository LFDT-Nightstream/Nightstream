import NightstreamFPrime.Gadgets.Sampling.WideReduction.LimbArithmetic
import NightstreamFPrime.Gadgets.Sampling.WideReduction.WitnessArithmetic

/-! Exact values of the small expressions used by the hint program.
Every division is justified by a bound on its integer input. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.HintValues

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
open Finset

theorem bit_hint (env : Env) (source : Expr) (value bit : Nat)
    (bound : value < goldilocksModulus) (meaning : source.eval env = fieldOfNat value) :
    Hint.eval env (.bit source bit) = fieldOfNat (value / 2 ^ bit % 2) := by
  simp only [Hint.eval, meaning, fieldOfNat, Nat.mod_eq_of_lt bound,
    Nat.and_one_is_mod, Nat.shiftRight_eq_div_pow]
  rfl

theorem bit_terms (env : Env) (start count value shift : Nat)
    (bits : ∀ bit, bit < count →
      (env (start + bit)).val = value / 2 ^ (shift + bit) % 2) :
    linearValue env (HintProgram.bitTerms start count) = value / 2 ^ shift % 2 ^ count := by
  rw [HintProgram.bitTerms, linearValue_rangeMap]
  calc
    _ = ∑ bit ∈ range count, 2 ^ bit * (value / 2 ^ shift / 2 ^ bit % 2) := by
      apply sum_congr rfl
      intro bit inside
      change 2 ^ bit * (env (start + bit)).val = _
      rw [bits bit (mem_range.mp inside), Nat.div_div_eq_div_mul, ← pow_add]
    _ = _ := digit_sum 2 (value / 2 ^ shift) count

theorem source_word (env : Env) (offset : Nat) (draw : Draw) (lane word : Fin 4)
    (bits : ∀ bit, bit < 64 →
      (env (offset + lane.val * 64 + bit)).val = (draw lane).val / 2 ^ bit % 2) :
    linearValue env (HintProgram.bitTerms (offset + lane.val * 64 + word.val * 16) 16) =
      LimbArithmetic.sourceLimb draw lane word := by
  have value := bit_terms env (offset + lane.val * 64 + word.val * 16) 16
    (draw lane).val (word.val * 16) (by
      intro bit below
      have source := bits (word.val * 16 + bit) (by have := word.isLt; omega)
      simpa only [Nat.add_assoc] using source)
  rw [value]
  unfold LimbArithmetic.sourceLimb LimbArithmetic.base
  rw [← pow_mul, Nat.mul_comm 16 word.val]

theorem carry_word (env : Env) (offset position : Nat) (draw : Draw)
    (bits : ∀ bit, bit < 37 →
      (env (HintProgram.limbStart offset position + 1 + bit)).val =
        LimbArithmetic.accumulator draw position / 2 ^ bit % 2) :
    linearValue env
      (HintProgram.bitTerms (HintProgram.limbStart offset position + 1 + 16) 21) =
        LimbArithmetic.carry draw (position + 1) := by
  have value := bit_terms env (HintProgram.limbStart offset position + 1 + 16) 21
    (LimbArithmetic.accumulator draw position) 16 (by
      intro bit below
      simpa only [Nat.add_assoc] using bits (16 + bit) (by omega))
  rw [value, Nat.mod_eq_of_lt]
  · rfl
  · change LimbArithmetic.carry draw (position + 1) < 2 ^ 21
    exact LimbArithmetic.carry_bound draw (position + 1)

def limb (value position : Nat) : Nat := value / WitnessArithmetic.radix ^ position %
  WitnessArithmetic.radix

def divisionCarry (value position : Nat) : Nat :=
  value / WitnessArithmetic.radix ^ (position + 1) % 5

theorem division_step (value position : Nat) :
    let input := divisionCarry value position * WitnessArithmetic.radix + limb value position
    input / 5 = limb (value / 5) position ∧
      input % 5 = value / WitnessArithmetic.radix ^ position % 5 := by
  simp only [divisionCarry, limb, pow_succ, ← Nat.div_div_eq_div_mul]
  have commute : value / 5 / WitnessArithmetic.radix ^ position =
      value / WitnessArithmetic.radix ^ position / 5 := by
    rw [Nat.div_div_eq_div_mul, Nat.div_div_eq_div_mul, Nat.mul_comm]
  rw [commute]
  unfold WitnessArithmetic.radix
  omega

theorem division_first_carry (value : Nat) (bound : value < 2 ^ 256) :
    divisionCarry value 4 = 0 := by
  unfold divisionCarry WitnessArithmetic.radix
  have below : value < (2 ^ 60) ^ (4 + 1) := lt_trans bound (by decide)
  rw [Nat.div_eq_of_lt below]

theorem division_source_bound (value position : Nat) :
    divisionCarry value position * WitnessArithmetic.radix + limb value position <
      goldilocksModulus :=
  WitnessArithmetic.accumulator_lt_field _ _ (Nat.mod_lt _ (by decide))
    (Nat.mod_lt _ (by decide))

theorem quotient_bit (value bit : Nat) :
    limb value (bit / 60) / 2 ^ (bit % 60) % 2 = value / 2 ^ bit % 2 := by
  unfold limb WitnessArithmetic.radix
  rw [LimbArithmetic.bit_below_mod_power _ 60 _ (Nat.mod_lt _ (by decide)),
    Nat.div_div_eq_div_mul, ← pow_mul, ← pow_add]
  congr 3
  omega

end NightstreamFPrime.Gadgets.Sampling.WideReduction.HintValues
