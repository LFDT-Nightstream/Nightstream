import Mathlib.Data.Nat.Count
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import NightstreamFPrime.Spec.AjtaiSetupV1

/-! Event-frequency error for uniform integer sampling followed by modular
reduction. No premise about ChaCha20 output distribution is introduced. -/

namespace NightstreamFPrime.Spec.AjtaiSetupV1.ReductionBias

private theorem count_blocks (modulus blocks : Nat) (event : Nat → Prop)
    [DecidablePred event] :
    Nat.count (fun value => event (value % modulus)) (modulus * blocks) =
      blocks * Nat.count (fun value => event (value % modulus)) modulus := by
  induction blocks with
  | zero => simp
  | succ blocks ih =>
    rw [Nat.mul_succ, Nat.count_add, ih]
    have shift : (fun value => event ((modulus * blocks + value) % modulus)) =
        (fun value => event (value % modulus)) := by
      funext value
      simp [Nat.add_mod]
    simp only [shift, Nat.succ_mul]

/-- Exact number of successful samples: complete residue blocks and a tail. -/
theorem event_count (samples modulus : Nat) (event : Nat → Prop)
    [DecidablePred event] :
    Nat.count (fun value => event (value % modulus)) samples =
      samples / modulus * Nat.count (fun value => event (value % modulus)) modulus +
      Nat.count (fun value => event (value % modulus)) (samples % modulus) := by
  conv_lhs => rw [← Nat.mod_add_div samples modulus, Nat.add_comm]
  rw [Nat.count_add, count_blocks]
  congr 1
  congr 1
  funext value
  simp [Nat.add_mod]

/-- Probability of an event when a uniform integer in `[0,samples)` is
reduced modulo `modulus`. Positive sample counts are required by the theorem. -/
def frequency (samples modulus : Nat) (event : Nat → Prop) [DecidablePred event] : ℚ :=
  (Nat.count (fun value => event (value % modulus)) samples : ℚ) / samples

/-- Uniform modular reduction changes any event probability by at most the
fraction of inputs in the incomplete final residue block. -/
theorem frequency_error_le (samples modulus : Nat)
    (samplesPositive : 0 < samples) (modulusPositive : 0 < modulus)
    (event : Nat → Prop) [DecidablePred event] :
    |frequency samples modulus event - frequency modulus modulus event| ≤
      (samples % modulus : Nat) / (samples : ℚ) := by
  let count := Nat.count (fun value => event (value % modulus))
  have fullBound : (count modulus : ℚ) ≤ modulus := by exact_mod_cast Nat.count_le _
  have tailBound : (count (samples % modulus) : ℚ) ≤ (samples % modulus : Nat) := by
    exact_mod_cast Nat.count_le _
  have fullNonnegative : (0 : ℚ) ≤ count modulus := by positivity
  have tailNonnegative : (0 : ℚ) ≤ count (samples % modulus) := by positivity
  have counts : (count samples : ℚ) =
      (samples / modulus : Nat) * (count modulus : ℚ) + count (samples % modulus) := by
    exact_mod_cast event_count samples modulus event
  have division : (samples : ℚ) =
      (modulus : ℚ) * (samples / modulus : Nat) + (samples % modulus : Nat) := by
    exact_mod_cast (Nat.div_add_mod samples modulus).symm
  have samplesPos : (0 : ℚ) < samples := by exact_mod_cast samplesPositive
  have modulusPos : (0 : ℚ) < modulus := by exact_mod_cast modulusPositive
  have remainderNonnegative : (0 : ℚ) ≤ (samples % modulus : Nat) := by positivity
  change |(count samples : ℚ) / samples - (count modulus : ℚ) / modulus| ≤ _
  rw [abs_sub_le_iff]
  constructor
  · apply (le_div_iff₀ samplesPos).mpr
    field_simp
    nlinarith [mul_le_mul_of_nonneg_left tailBound modulusPos.le,
      mul_nonneg remainderNonnegative fullNonnegative]
  · apply (le_div_iff₀ samplesPos).mpr
    field_simp
    nlinarith [mul_le_mul_of_nonneg_left fullBound remainderNonnegative,
      mul_nonneg modulusPos.le tailNonnegative]

theorem wide_remainder_eq : 2 ^ 256 % goldilocksModulus = 4294967295 := by decide

theorem wide_frequency_error_le (event : Nat → Prop) [DecidablePred event] :
    |frequency (2 ^ 256) goldilocksModulus event -
      frequency goldilocksModulus goldilocksModulus event| ≤
      (4294967295 : ℚ) / 2 ^ 256 := by
  simpa only [wide_remainder_eq, Nat.cast_pow, Nat.cast_ofNat] using
    frequency_error_le (2 ^ 256) goldilocksModulus (by decide) (by decide) event

end NightstreamFPrime.Spec.AjtaiSetupV1.ReductionBias
