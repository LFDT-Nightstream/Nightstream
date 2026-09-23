import NightstreamFPrime.Gadgets.Sampling.WideReduction.HintProgram
import NightstreamFPrime.Gadgets.Sampling.WideReduction.Completeness

/-! Exact base-2^16 accumulation of the four base-p input words.
The carry bounds are integer bounds, before any conversion to the field. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.LimbArithmetic

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
open Finset

def base : Nat := 2 ^ 16

def sourceLimb (draw : Draw) (lane word : Fin 4) : Nat :=
  (draw lane).val / base ^ word.val % base

def coefficient (draw : Draw) (position : Nat) : Nat :=
  ∑ lane : Fin 4, ∑ word : Fin 4,
    if word.val ≤ position then
      HintProgram.coefficient lane.val (position - word.val) * sourceLimb draw lane word
    else 0

def carry (draw : Draw) : Nat → Nat
  | 0 => 0
  | position + 1 => (carry draw position + coefficient draw position) / base

def accumulator (draw : Draw) (position : Nat) : Nat :=
  carry draw position + coefficient draw position

theorem coefficient_bound (draw : Draw) (position : Nat) :
    coefficient draw position ≤ 16 * (base - 1) ^ 2 := by
  have each (lane word : Fin 4) :
      (if word.val ≤ position then
        HintProgram.coefficient lane.val (position - word.val) * sourceLimb draw lane word
      else 0) ≤ (base - 1) ^ 2 := by
    split
    · have left : HintProgram.coefficient lane.val (position - word.val) ≤ base - 1 :=
        Nat.le_sub_one_of_lt (Nat.mod_lt _ (by decide))
      have right : sourceLimb draw lane word ≤ base - 1 :=
        Nat.le_sub_one_of_lt (Nat.mod_lt _ (by decide))
      simpa only [pow_two] using Nat.mul_le_mul left right
    · exact Nat.zero_le _
  calc
    coefficient draw position ≤ ∑ _lane : Fin 4, ∑ _word : Fin 4, (base - 1) ^ 2 :=
      sum_le_sum fun lane _ => sum_le_sum fun word _ => each lane word
    _ = 16 * (base - 1) ^ 2 := by simp; ring

theorem carry_bound (draw : Draw) (position : Nat) : carry draw position < 2 ^ 21 := by
  induction position with
  | zero => norm_num [carry]
  | succ position ih =>
      have bound := coefficient_bound draw position
      rw [carry]
      norm_num [base] at bound ⊢
      norm_num at ih
      omega

theorem accumulator_bound (draw : Draw) (position : Nat) : accumulator draw position < 2 ^ 37 := by
  have hc := carry_bound draw position
  have hcoef := coefficient_bound draw position
  unfold accumulator
  norm_num [base] at hcoef ⊢
  norm_num at hc
  omega

theorem accumulator_times_five_lt_field (draw : Draw) (position : Nat) :
    5 * accumulator draw position < goldilocksModulus := by
  have bound := accumulator_bound draw position
  norm_num [goldilocksModulus] at bound ⊢
  omega

def lowValue (draw : Draw) (count : Nat) : Nat :=
  ∑ position ∈ range count, base ^ position * (accumulator draw position % base)

theorem carry_identity (draw : Draw) (count : Nat) :
    lowValue draw count + base ^ count * carry draw count =
      ∑ position ∈ range count, base ^ position * coefficient draw position := by
  induction count with
  | zero => simp [lowValue, carry]
  | succ count ih =>
      have division := Nat.mod_add_div (accumulator draw count) base
      rw [lowValue, sum_range_succ, sum_range_succ, pow_succ, carry]
      change lowValue draw count + base ^ count * (accumulator draw count % base) +
        (base ^ count * base) * (accumulator draw count / base) = _
      calc
        _ = lowValue draw count + base ^ count *
            (accumulator draw count % base + base * (accumulator draw count / base)) := by ring
        _ = lowValue draw count + base ^ count * accumulator draw count := by rw [division]
        _ = (lowValue draw count + base ^ count * carry draw count) +
            base ^ count * coefficient draw count := by unfold accumulator; ring
        _ = _ := by rw [ih]

theorem source_expansion (draw : Draw) (lane : Fin 4) :
    (draw lane).val = ∑ word : Fin 4, base ^ word.val * sourceLimb draw lane word := by
  have bound : (draw lane).val < base ^ 4 :=
    lt_trans (draw lane).isLt (by decide)
  have expansion := digit_sum base (draw lane).val 4
  rw [Nat.mod_eq_of_lt bound] at expansion
  simpa only [sourceLimb, Fin.sum_univ_eq_sum_range] using expansion.symm

private theorem weighted_matrix (words : Fin 4 → Fin 4 → Nat) :
    (∑ position ∈ range 16, base ^ position *
      (∑ lane : Fin 4, ∑ word : Fin 4,
        if word.val ≤ position then
          HintProgram.coefficient lane.val (position - word.val) * words lane word else 0)) =
      ∑ lane : Fin 4, goldilocksModulus ^ lane.val *
        (∑ word : Fin 4, base ^ word.val * words lane word) := by
  norm_num [sum_range_succ, Fin.sum_univ_four, HintProgram.coefficient,
    HintProgram.limbBits, base, goldilocksModulus]
  ring

theorem coefficient_sum (draw : Draw) :
    (∑ position ∈ range 16, base ^ position * coefficient draw position) =
      (drawIndex draw).val := by
  unfold coefficient
  rw [weighted_matrix, drawIndex_val]
  apply sum_congr rfl
  intro lane _
  rw [← source_expansion]
  exact Nat.mul_comm _ _

theorem lowValue_exact (draw : Draw) : lowValue draw 16 = (drawIndex draw).val := by
  have equation := carry_identity draw 16
  rw [coefficient_sum] at equation
  have bound : (drawIndex draw).val < base ^ 16 :=
    lt_trans (drawIndex draw).isLt (by decide)
  have zero : carry draw 16 = 0 := by
    norm_num [base] at equation bound
    omega
  simpa [zero] using equation

theorem limb_exact (draw : Draw) (position : Fin 16) :
    accumulator draw position.val % base = (drawIndex draw).val / base ^ position.val % base := by
  let limbs : Fin 16 → Fin base := fun index =>
    ⟨accumulator draw index.val % base, Nat.mod_lt _ (by decide)⟩
  have encoded : (finFunctionFinEquiv limbs).val = (drawIndex draw).val := by
    rw [finFunctionFinEquiv_apply]
    simpa only [limbs, Fin.sum_univ_eq_sum_range, Nat.mul_comm] using lowValue_exact draw
  have decoded := congrArg (fun values : Fin 16 → Fin base => (values position).val)
    (finFunctionFinEquiv.symm_apply_apply limbs)
  change (finFunctionFinEquiv limbs).val / base ^ position.val % base =
    accumulator draw position.val % base at decoded
  rw [encoded] at decoded
  exact decoded.symm

theorem digit_below_mod_power (radix value width digit : Nat) (below : digit < width) :
    value % radix ^ width / radix ^ digit % radix = value / radix ^ digit % radix := by
  rw [← Nat.mod_mul_right_div_self, ← Nat.mod_mul_right_div_self]
  have divides : radix ^ digit * radix ∣ radix ^ width := by
    rw [← pow_succ]
    exact pow_dvd_pow radix (by omega)
  rw [Nat.mod_mod_of_dvd _ divides]

theorem bit_below_mod_power (value width bit : Nat) (below : bit < width) :
    value % 2 ^ width / 2 ^ bit % 2 = value / 2 ^ bit % 2 :=
  digit_below_mod_power 2 value width bit below

theorem integerBit_exact (draw : Draw) (index : Nat) (below : index < 256) :
    accumulator draw (index / 16) / 2 ^ (index % 16) % 2 =
      (drawIndex draw).val / 2 ^ index % 2 := by
  have limb := limb_exact draw ⟨index / 16, by omega⟩
  change accumulator draw (index / 16) % base =
    (drawIndex draw).val / base ^ (index / 16) % base at limb
  have bit := Nat.mod_lt index (by decide : 0 < 16)
  rw [← bit_below_mod_power _ 16 _ bit]
  change accumulator draw (index / 16) % base / 2 ^ (index % 16) % 2 = _
  rw [limb]
  unfold base
  rw [bit_below_mod_power _ 16 _ bit, Nat.div_div_eq_div_mul, ← pow_mul, ← pow_add]
  congr 3
  omega

end NightstreamFPrime.Gadgets.Sampling.WideReduction.LimbArithmetic
