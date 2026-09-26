import NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperExecution
import NightstreamFPrime.Gadgets.Sampling.WideReduction.CheckQuotient

/-! The result-bit hints read the proved temporary quotient and remainders.
Their values agree with the gadget's explicit honest completion. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.OutputHintValues

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler

theorem division_values (env : Env) (start : Nat) (draw : Draw)
    (present : HelperValues.Present env start draw) (round position : Nat)
    (roundBound : round < 54) (positionBound : position < 5) :
    env (HintProgram.divisionColumn start round position) =
        fieldOfNat (HelperValues.quotient draw round position) ∧
      env (HintProgram.divisionColumn start round position + 1) =
        fieldOfNat (HelperValues.remainder draw round position) := by
  have lower : start ≤ HintProgram.divisionColumn start round position := by
    unfold HintProgram.divisionColumn HintProgram.divisionStart
    omega
  have upper : HintProgram.divisionColumn start round position + 1 < start + HintProgram.helperCount := by
    simp only [HintProgram.divisionColumn, HintProgram.divisionStart, HintProgram.divisionStride,
      HintProgram.divisionLimbs, HintProgram.sourceBitCount, HintProgram.limbCount,
      HintProgram.limbStride, HintProgram.accumulatorBits, HintProgram.helperCount, digitCount]
    omega
  rw [present _ lower (by omega), present _ (by omega) upper]
  exact HelperValues.division env start draw round position roundBound positionBound

theorem quotient_hint (env : Env) (start : Nat) (draw : Draw)
    (present : HelperValues.Present env start draw) (bit : Nat) (bound : bit < quotientBitCount) :
    (HintProgram.quotientHint start bit).eval env =
      fieldOfNat ((drawIndex draw).val / scalarCount / 2 ^ bit % 2) := by
  have positionBound : 4 - bit / 60 < 5 := by omega
  have value := (division_values env start draw present 53 (4 - bit / 60) (by decide) positionBound).1
  have small : HelperValues.quotient draw 53 (4 - bit / 60) < goldilocksModulus :=
    lt_trans (Nat.mod_lt _ (by decide : 0 < WitnessArithmetic.radix)) (by decide)
  unfold HintProgram.quotientHint
  simp only [digitCount, HintProgram.divisionLimbs, HintProgram.divisionBits, Nat.reduceSub]
  rw [HintValues.bit_hint env (.var (HintProgram.divisionColumn start 53 (4 - bit / 60)))
    _ (bit % 60) small value]
  unfold HelperValues.quotient
  have bitBound : bit < 131 := bound
  rw [show 4 - (4 - bit / 60) = bit / 60 by omega]
  change fieldOfNat (HintValues.limb ((drawIndex draw).val / scalarCount) (bit / 60) /
    2 ^ (bit % 60) % 2) = _
  rw [HintValues.quotient_bit]

theorem digit_hint (env : Env) (start : Nat) (draw : Draw)
    (present : HelperValues.Present env start draw) (digit bit : Nat)
    (bound : digit < digitCount) :
    (HintProgram.digitHint start digit bit).eval env =
      fieldOfNat ((drawIndex draw).val % scalarCount / 5 ^ digit % 5 / 2 ^ bit % 2) := by
  have value := (division_values env start draw present digit 4 bound (by decide)).2
  have small : HelperValues.remainder draw digit 4 < goldilocksModulus :=
    lt_trans (Nat.mod_lt _ (by decide : 0 < 5)) (by decide)
  unfold HintProgram.digitHint
  simp only [HintProgram.divisionLimbs, Nat.reduceSub]
  rw [HintValues.bit_hint env (.var (HintProgram.divisionColumn start digit 4 + 1))
    _ bit small value]
  unfold HelperValues.remainder
  simp only [Nat.sub_self, pow_zero, Nat.div_one]
  have digitMeaning := LimbArithmetic.digit_below_mod_power 5 (drawIndex draw).val 54 digit bound
  change (drawIndex draw).val % scalarCount / 5 ^ digit % 5 = _ at digitMeaning
  rw [digitMeaning]

theorem completeNew_helpers (interface : Interface) (base : Env) (helperOffset offset : Nat)
    (before : helperOffset + HintProgram.helperCount ≤ offset)
    (present : HelperValues.Present base helperOffset (drawOf interface base offset)) :
    HelperValues.Present (completeNew interface base offset) helperOffset (drawOf interface base offset) := by
  apply HelperValues.present_of_agree base _ _ _ present
  intro index _ upper
  exact completeNew_agreesOutside interface base offset index (Or.inl (by
    unfold quotientStart
    omega))

theorem check_hint (interface : Interface) (hints : Nat → List Hint) (base : Env) (offset : Nat)
    (children : ∀ index, Range.CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) base)
    (childRows : holdsFlat base (childOps interface offset))
    (childScope : ∀ expression ∈ flatConstraints (childOps interface offset),
      expression.VarsBelow (quotientStart offset)) (check : Fin checkCount) :
    (HintProgram.checkQuotient offset check).eval (completeNew interface base offset) =
      fieldOfNat (checkQuotients base offset (drawIndex (drawOf interface base offset)).val check.val) := by
  let draw := (drawIndex (drawOf interface base offset)).val
  let target := completeNew interface base offset
  let quotient := checkQuotients base offset draw check.val
  have certified := completeNew_certificate interface hints base offset children childRows childScope
  have bound : quotient < 2 ^ checkBitCount := lt_trans (certified.2 check) (by decide)
  have checkValue : linearValue target (checkTerms offset check) = modulus check * quotient := by
    rw [checkTerms, linearValue_rangeMap]
    calc
      _ = ∑ bit ∈ Finset.range checkBitCount,
          modulus check * (2 ^ bit * (quotient / 2 ^ bit % 2)) := by
        apply Finset.sum_congr rfl
        intro bit member
        have value := checkBit_eval (base := base) (offset := offset) (draw := draw)
          (check := checkQuotients base offset draw) check.val bit (Finset.mem_range.mp member) check.isLt
        change ((checkBit offset check bit).eval target).val = quotient / 2 ^ bit % 2 at value
        change modulus check * 2 ^ bit * ((checkBit offset check bit).eval target).val = _
        rw [value]
        ring
      _ = modulus check * quotient := by
        rw [← Finset.mul_sum, digit_sum, Nat.mod_eq_of_lt bound]
  have holds := holdsFlat_implies_holds _ _ certified.1
  have row := holds (Op.assertZero (checkRow offset check)) (by
    simp only [operations, rowOps, List.mem_append, List.mem_map, List.mem_finRange, true_and]
    exact Or.inr (Or.inr ⟨check, rfl⟩))
  change (linearExpr (reduceTerms (modulus check) (drawTerms offset)) +
    Expr.const (fieldOfNat (modulus check * checkBias)) -
    (linearExpr (reduceTerms (modulus check) (resultTerms offset)) +
      linearExpr (checkTerms offset check))).eval target = 0 at row
  rw [Expr.eval_sub, sub_eq_zero] at row
  simp only [Expr.eval_hadd] at row
  have difference : (linearExpr (reduceTerms (modulus check) (drawTerms offset)) +
      Expr.const (fieldOfNat (modulus check * checkBias)) -
      linearExpr (reduceTerms (modulus check) (resultTerms offset))).eval target =
        fieldOfNat (modulus check * quotient) := by
    rw [Expr.eval_sub]
    apply sub_eq_iff_eq_add.mpr
    rw [Expr.eval_hadd, row, linearExpr_eval target (checkTerms offset check), checkValue, add_comm]
  unfold HintProgram.checkQuotient
  rw [Expr.eval_hmul, Expr.eval_const, difference, ← fieldOfNat_mul]
  change (fieldOfNat (modulus check) : ZMod goldilocksModulus) * fieldOfNat quotient * _ = _
  rw [mul_comm (fieldOfNat (modulus check)), mul_assoc, CheckQuotient.modulus_inverse, mul_one]

end NightstreamFPrime.Gadgets.Sampling.WideReduction.OutputHintValues
