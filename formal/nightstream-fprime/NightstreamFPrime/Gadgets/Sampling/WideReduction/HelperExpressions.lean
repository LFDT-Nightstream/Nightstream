import NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperValues

/-! Evaluation of helper expressions in their integer reference environment. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperExpressions

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
open Finset

theorem source_bit_val (base : Env) (start : Nat) (draw : Draw) (lane : Fin 4) (bit : Nat)
    (bound : bit < 64) :
    (HelperValues.environment base start draw (start + lane.val * 64 + bit)).val =
      (draw lane).val / 2 ^ bit % 2 := by
  rw [HelperValues.source_bit base start draw lane bit bound]
  exact honestBit_val _

theorem accumulator_bit_val (base : Env) (start : Nat) (draw : Draw) (position bit : Nat)
    (positionBound : position < 16) (bitBound : bit < 37) :
    (HelperValues.environment base start draw (HintProgram.limbStart start position + 1 + bit)).val =
      LimbArithmetic.accumulator draw position / 2 ^ bit % 2 := by
  rw [HelperValues.accumulator_bit base start draw position bit positionBound bitBound]
  exact honestBit_val _

private theorem linearValue_scale (env : Env) (terms : List (Nat × Expr)) (scale : Nat) :
    linearValue env (terms.map fun term => (scale * term.1, term.2)) =
      scale * linearValue env terms := by
  induction terms with
  | nil => simp [linearValue]
  | cons term rest ih =>
      simp only [List.map_cons, linearValue, ih]
      ring

private theorem word_value (base : Env) (start : Nat) (draw : Draw)
    (lane word : Fin 4) (scale : Nat) :
    linearValue (HelperValues.environment base start draw)
      ((List.range 16).map fun bit =>
        (scale * 2 ^ bit, Expr.var (start + lane.val * 64 + word.val * 16 + bit))) =
      scale * LimbArithmetic.sourceLimb draw lane word := by
  have forms : (List.range 16).map (fun bit =>
        (scale * 2 ^ bit, Expr.var (start + lane.val * 64 + word.val * 16 + bit))) =
      (HintProgram.bitTerms (start + lane.val * 64 + word.val * 16) 16).map
        (fun term => (scale * term.1, term.2)) := by
    simp only [HintProgram.bitTerms, List.map_map, Function.comp_def]
  rw [forms, linearValue_scale, HintValues.source_word]
  exact source_bit_val base start draw lane

theorem limb_terms (base : Env) (start : Nat) (draw : Draw) (position : Nat)
    (bound : position < 16) :
    linearValue (HelperValues.environment base start draw) (HintProgram.limbTerms start position) =
      LimbArithmetic.accumulator draw position := by
  have carry : linearValue (HelperValues.environment base start draw)
      (if position = 0 then [] else
        HintProgram.bitTerms (HintProgram.limbStart start (position - 1) + 1 + HintProgram.limbBits)
          (HintProgram.accumulatorBits - HintProgram.limbBits)) =
      LimbArithmetic.carry draw position := by
    cases position with
    | zero => rfl
    | succ position =>
        simp only [Nat.succ_ne_zero, if_false, Nat.add_sub_cancel, HintProgram.limbBits,
          HintProgram.accumulatorBits]
        exact HintValues.carry_word _ start position draw
          (accumulator_bit_val base start draw position · (by omega))
  unfold HintProgram.limbTerms
  rw [linearValue_append, carry, linearValue_flatMap_range]
  change LimbArithmetic.carry draw position + _ =
    LimbArithmetic.carry draw position + LimbArithmetic.coefficient draw position
  apply congrArg (fun value => LimbArithmetic.carry draw position + value)
  unfold LimbArithmetic.coefficient
  change (∑ lane ∈ range 4, _) = _
  rw [← Fin.sum_univ_eq_sum_range]
  apply sum_congr rfl
  intro lane _
  rw [linearValue_flatMap_range, ← Fin.sum_univ_eq_sum_range]
  apply sum_congr rfl
  intro word _
  by_cases included : word.val ≤ position
  · rw [if_pos included, if_pos included]
    exact word_value base start draw lane word _
  · rw [if_neg included, if_neg included]
    rfl

theorem integer_bit (base : Env) (start : Nat) (draw : Draw) (index : Nat)
    (bound : index < 256) :
    ((HintProgram.integerBit start index).eval (HelperValues.environment base start draw)).val =
      (drawIndex draw).val / 2 ^ index % 2 := by
  change (HelperValues.environment base start draw
    (HintProgram.limbStart start (index / 16) + 1 + index % 16)).val = _
  rw [accumulator_bit_val base start draw _ _ (by omega) (by omega)]
  exact LimbArithmetic.integerBit_exact draw index bound

private theorem linearValue_filterMap (env : Env) (terms : Nat → Option (Nat × Expr))
    (count : Nat) :
    linearValue env ((List.range count).filterMap terms) =
      ∑ index ∈ range count, match terms index with
        | none => 0
        | some term => term.1 * (term.2.eval env).val := by
  induction count with
  | zero => rfl
  | succ count ih =>
      rw [List.range_succ, List.filterMap_append, linearValue_append, ih, sum_range_succ]
      cases value : terms count <;> simp [value, linearValue]

theorem initial_limb (base : Env) (start : Nat) (draw : Draw) (position : Nat) :
    (HintProgram.initialDivisionLimb start position).eval
        (HelperValues.environment base start draw) =
      fieldOfNat (HintValues.limb (drawIndex draw).val position) := by
  rw [HintProgram.initialDivisionLimb, linearExpr_eval, linearValue_filterMap]
  apply congrArg fieldOfNat
  simp only [HintProgram.divisionBits, HintProgram.limbCount, HintProgram.limbBits, Nat.reduceMul]
  calc
    _ = ∑ bit ∈ range 60, 2 ^ bit * ((drawIndex draw).val / 2 ^ (position * 60 + bit) % 2) := by
      apply sum_congr rfl
      intro bit _
      by_cases included : position * 60 + bit < 256
      · rw [if_pos included]
        exact congrArg (fun value => 2 ^ bit * value) (integer_bit base start draw _ included)
      · rw [if_neg included]
        have valueBound : (drawIndex draw).val < 2 ^ 256 :=
          lt_trans (drawIndex draw).isLt (by decide)
        have high : (drawIndex draw).val < 2 ^ (position * 60 + bit) :=
          lt_of_lt_of_le valueBound (Nat.pow_le_pow_right (by decide) (by omega))
        rw [Nat.div_eq_of_lt high]
        simp
    _ = (drawIndex draw).val / 2 ^ (position * 60) % 2 ^ 60 := by
      simp only [pow_add, ← Nat.div_div_eq_div_mul]
      exact digit_sum 2 _ 60
    _ = _ := by
      unfold HintValues.limb WitnessArithmetic.radix
      rw [← pow_mul, Nat.mul_comm 60 position]

theorem division_input (base : Env) (start : Nat) (draw : Draw) (round position : Nat)
    (roundBound : round < 54) (positionBound : position < 5) :
    (HintProgram.divisionInput start round position).eval
        (HelperValues.environment base start draw) =
      fieldOfNat
        (HintValues.divisionCarry ((drawIndex draw).val / 5 ^ round) (4 - position) *
          WitnessArithmetic.radix + HintValues.limb ((drawIndex draw).val / 5 ^ round) (4 - position)) := by
  have carryMeaning :
      (if position = 0 then Expr.const 0 else
        Expr.var (HintProgram.divisionColumn start round (position - 1) + 1)).eval
          (HelperValues.environment base start draw) =
        fieldOfNat (HintValues.divisionCarry ((drawIndex draw).val / 5 ^ round) (4 - position)) := by
    by_cases first : position = 0
    · subst position
      rw [if_pos rfl]
      have valueBound : (drawIndex draw).val / 5 ^ round < 2 ^ 256 :=
        lt_of_le_of_lt (Nat.div_le_self _ _) (lt_trans (drawIndex draw).isLt (by decide))
      rw [Nat.sub_zero, HintValues.division_first_carry _ valueBound]
      rfl
    · rw [if_neg first]
      change HelperValues.environment base start draw _ = _
      rw [(HelperValues.division base start draw round (position - 1) roundBound (by omega)).2]
      apply congrArg fieldOfNat
      unfold HelperValues.remainder HintValues.divisionCarry
      rw [show 4 - (position - 1) = 4 - position + 1 by omega]
  have limbMeaning :
      (if round = 0 then HintProgram.initialDivisionLimb start (4 - position) else
        Expr.var (HintProgram.divisionColumn start (round - 1) position)).eval
          (HelperValues.environment base start draw) =
        fieldOfNat (HintValues.limb ((drawIndex draw).val / 5 ^ round) (4 - position)) := by
    by_cases first : round = 0
    · subst round
      rw [if_pos rfl, initial_limb]
      simp only [pow_zero, Nat.div_one]
    · rw [if_neg first]
      change HelperValues.environment base start draw _ = _
      rw [(HelperValues.division base start draw (round - 1) position (by omega) positionBound).1]
      unfold HelperValues.quotient
      rw [show round - 1 + 1 = round by omega]
  unfold HintProgram.divisionInput
  change (Expr.const (fieldOfNat WitnessArithmetic.radix) * _ + _).eval _ = _
  simp only [HintProgram.divisionLimbs, Nat.reduceSub]
  rw [Expr.eval_hadd, Expr.eval_hmul, Expr.eval_const, carryMeaning, limbMeaning,
    fieldOfNat_mul, fieldOfNat_add, Nat.mul_comm WitnessArithmetic.radix]

end NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperExpressions
