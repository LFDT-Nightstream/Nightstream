import Mathlib.Tactic.Ring
import NightstreamFPrime.Gadgets.Sampling.WideReduction.Linear

/-! Integer arithmetic for the existing-hint witness program. Division
proceeds from the most significant 60-bit limb. Bounds precede conversion
to field expressions, so field reduction cannot change the quotient. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.WitnessArithmetic

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

def radix : Nat := 2 ^ 60

theorem accumulator_lt (carry limb : Nat) (hc : carry < 5) (hl : limb < radix) :
    carry * radix + limb < 5 * radix := by
  unfold radix at *
  omega

theorem accumulator_lt_field (carry limb : Nat) (hc : carry < 5) (hl : limb < radix) :
    carry * radix + limb < goldilocksModulus :=
  lt_trans (accumulator_lt carry limb hc hl) (by decide)

theorem quotient_lt (carry limb : Nat) (hc : carry < 5) (hl : limb < radix) :
    (carry * radix + limb) / 5 < radix := by
  have h := accumulator_lt carry limb hc hl
  omega

theorem quotient_hint (env : Env) (source : Expr) (integer : Nat)
    (bound : integer < goldilocksModulus) (meaning : source.eval env = fieldOfNat integer) :
    Hint.eval env (.quotientFive source) = fieldOfNat (integer / 5) := by
  simp only [Hint.eval, meaning, fieldOfNat, Nat.mod_eq_of_lt bound]
  rfl

theorem remainder_hint (env : Env) (source : Expr) (integer : Nat)
    (bound : integer < goldilocksModulus) (meaning : source.eval env = fieldOfNat integer) :
    Hint.eval env (.remainderFive source) = fieldOfNat (integer % 5) := by
  simp only [Hint.eval, meaning, fieldOfNat, Nat.mod_eq_of_lt bound]
  rfl

end NightstreamFPrime.Gadgets.Sampling.WideReduction.WitnessArithmetic
