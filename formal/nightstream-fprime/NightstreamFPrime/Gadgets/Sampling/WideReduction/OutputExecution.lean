import NightstreamFPrime.Gadgets.Sampling.WideReduction.OutputHintValues
import NightstreamFPrime.Gadgets.Sampling.WideReduction.HintSupport
import NightstreamFPrime.Circuit.Execution

/-! The 434 retained result-bit hints compute the existing gadget's explicit
honest assignment. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.OutputExecution

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler

private theorem assemble {holds : Nat → Hint → Prop} (helperOffset offset : Nat)
    (quotients : HintsEach holds (quotientStart offset)
      ((List.range quotientBitCount).map (HintProgram.quotientHint helperOffset)))
    (digits : HintsEach holds (digitStart offset)
      ((List.range digitCount).flatMap fun digit =>
        (List.range digitBitCount).map (HintProgram.digitHint helperOffset digit)))
    (checks : HintsEach holds (checkStart offset)
      ((List.finRange checkCount).flatMap fun check =>
        (List.range checkBitCount).map fun bit => .bit (HintProgram.checkQuotient offset check) bit)) :
    HintsEach holds (quotientStart offset) (HintProgram.resultHints helperOffset offset) := by
  have first := quotients.append (by
    rw [List.length_map, List.length_range]
    exact digits)
  exact first.append (by
    simp only [List.length_append, List.length_map, List.length_range, List.length_flatMap,
      List.map_const', List.sum_replicate, smul_eq_mul]
    simpa only [checkStart, digitStart, Nat.add_assoc] using checks)

theorem causal (helperOffset offset : Nat)
    (before : helperOffset + HintProgram.helperCount ≤ offset) :
    HintsCausal (quotientStart offset) (HintProgram.resultHints helperOffset offset) := by
  refine assemble helperOffset offset (HintsEach.range_map fun bit _ => ?_)
    (HintsEach.range_flatMap (stride := digitBitCount) (fun digit _ => by simp) fun digit _ =>
      HintsEach.range_map fun bit _ => ?_)
    (HintsEach.finRange_flatMap (stride := checkBitCount) (fun check => by simp) fun check =>
      HintsEach.range_map fun bit _ => ?_)
  · change HintProgram.divisionColumn helperOffset (digitCount - 1)
      (HintProgram.divisionLimbs - 1 - bit / HintProgram.divisionBits) < quotientStart offset + bit
    simp only [HintProgram.divisionColumn, HintProgram.divisionStart, HintProgram.divisionStride,
      HintProgram.divisionLimbs, HintProgram.sourceBitCount, HintProgram.limbCount,
      HintProgram.limbStride, HintProgram.accumulatorBits, HintProgram.helperCount,
      HintProgram.divisionBits, digitCount, quotientStart, childWidth,
      Range.CanonicalU64.auxiliaryCount, fieldCount] at *
    omega
  · change HintProgram.divisionColumn helperOffset digit (HintProgram.divisionLimbs - 1) + 1 <
      digitStart offset + digit * digitBitCount + bit
    simp only [HintProgram.divisionColumn, HintProgram.divisionStart, HintProgram.divisionStride,
      HintProgram.divisionLimbs, HintProgram.sourceBitCount, HintProgram.limbCount,
      HintProgram.limbStride, HintProgram.accumulatorBits, HintProgram.helperCount,
      digitCount, digitBitCount, digitStart, quotientStart, childWidth,
      Range.CanonicalU64.auxiliaryCount, fieldCount, quotientBitCount] at *
    omega
  · exact Expr.VarsBelow.mono _ (HintSupport.checkQuotient_below offset check) (by omega)

theorem quotient_bits (interface : Interface) (base : Env) (helperOffset offset : Nat)
    (before : helperOffset + HintProgram.helperCount ≤ offset)
    (helpers : HelperValues.Present base helperOffset (drawOf interface base offset)) :
    HintsComputed (completeNew interface base offset) (quotientStart offset)
      ((List.range quotientBitCount).map (HintProgram.quotientHint helperOffset)) := by
  have present := OutputHintValues.completeNew_helpers interface base helperOffset offset before helpers
  refine HintsEach.range_map fun bit below => Eq.symm ?_
  rw [OutputHintValues.quotient_hint _ helperOffset _ present bit below]
  apply Fin.ext
  rw [show (fieldOfNat ((drawIndex (drawOf interface base offset)).val / scalarCount /
      2 ^ bit % 2)).val = (drawIndex (drawOf interface base offset)).val / scalarCount /
        2 ^ bit % 2 from honestBit_val _]
  exact (quotientBit_eval (base := base) (offset := offset)
    (draw := (drawIndex (drawOf interface base offset)).val)
    (check := checkQuotients base offset (drawIndex (drawOf interface base offset)).val) bit below).symm

theorem digit_bits (interface : Interface) (base : Env) (helperOffset offset : Nat)
    (before : helperOffset + HintProgram.helperCount ≤ offset)
    (helpers : HelperValues.Present base helperOffset (drawOf interface base offset)) :
    HintsComputed (completeNew interface base offset) (digitStart offset)
      ((List.range digitCount).flatMap fun digit =>
        (List.range digitBitCount).map (HintProgram.digitHint helperOffset digit)) := by
  have present := OutputHintValues.completeNew_helpers interface base helperOffset offset before helpers
  refine HintsEach.range_flatMap (stride := digitBitCount) (fun digit _ => by simp)
    fun digit digitBelow => HintsEach.range_map fun bit bitBelow => Eq.symm ?_
  rw [OutputHintValues.digit_hint _ helperOffset _ present digit bit digitBelow]
  apply Fin.ext
  rw [show (fieldOfNat ((drawIndex (drawOf interface base offset)).val % scalarCount /
      5 ^ digit % 5 / 2 ^ bit % 2)).val =
      (drawIndex (drawOf interface base offset)).val % scalarCount / 5 ^ digit % 5 /
        2 ^ bit % 2 from honestBit_val _]
  have value := digitBit_eval (base := base) (offset := offset)
    (draw := (drawIndex (drawOf interface base offset)).val)
    (check := checkQuotients base offset (drawIndex (drawOf interface base offset)).val)
    digit bit digitBelow bitBelow
  simpa only [digitBit, Expr.eval_var, Nat.mul_comm digit digitBitCount] using! value.symm

theorem check_bits (interface : Interface) (hints : Nat → List Hint) (base : Env) (offset : Nat)
    (children : ∀ index, Range.CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) base)
    (childRows : holdsFlat base (childOps interface offset))
    (childScope : ∀ expression ∈ flatConstraints (childOps interface offset),
      expression.VarsBelow (quotientStart offset)) :
    HintsComputed (completeNew interface base offset) (checkStart offset)
      ((List.finRange checkCount).flatMap fun check =>
        (List.range checkBitCount).map fun bit => .bit (HintProgram.checkQuotient offset check) bit) := by
  have certified := completeNew_certificate interface hints base offset children childRows childScope
  refine HintsEach.finRange_flatMap (stride := checkBitCount) (fun check => by simp) fun check =>
    HintsEach.range_map fun bit bitBelow => Eq.symm ?_
  have value := OutputHintValues.check_hint interface hints base offset children childRows childScope check
  rw [HintValues.bit_hint _ _ _ bit (lt_trans (certified.2 check) (by decide)) value]
  apply Fin.ext
  rw [show (fieldOfNat (checkQuotients base offset (drawIndex (drawOf interface base offset)).val
      check.val / 2 ^ bit % 2)).val =
      checkQuotients base offset (drawIndex (drawOf interface base offset)).val check.val /
        2 ^ bit % 2 from honestBit_val _]
  have result := checkBit_eval (base := base) (offset := offset)
    (draw := (drawIndex (drawOf interface base offset)).val)
    (check := checkQuotients base offset (drawIndex (drawOf interface base offset)).val)
    check.val bit bitBelow check.isLt
  simpa only [checkBit, Expr.eval_var, Nat.mul_comm check.val checkBitCount] using! result.symm

theorem reference (interface : Interface) (env : Env) (helperOffset offset : Nat)
    (before : helperOffset + HintProgram.helperCount ≤ offset)
    (helpers : HelperValues.Present env helperOffset (drawOf interface env offset))
    (children : ∀ index, Range.CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) env)
    (childRows : holdsFlat env (childOps interface offset))
    (childScope : ∀ expression ∈ flatConstraints (childOps interface offset),
      expression.VarsBelow (quotientStart offset))
    (computed : HintsComputed env (quotientStart offset)
      (HintProgram.resultHints helperOffset offset)) :
    completeNew interface env offset = env := by
  have same := hintsComputed_unique (causal helperOffset offset before)
    (assemble helperOffset offset (quotient_bits interface env helperOffset offset before helpers)
      (digit_bits interface env helperOffset offset before helpers)
      (check_bits interface (HintProgram.resultHints helperOffset) env offset children childRows
        childScope))
    computed fun index below => completeNew_agreesOutside interface env offset index (Or.inl below)
  funext index
  by_cases inside : index < quotientStart offset + newBitCount
  · exact same index (by rwa [HintProgram.resultHints_length])
  · exact completeNew_agreesOutside interface env offset index (Or.inr (by omega))

end NightstreamFPrime.Gadgets.Sampling.WideReduction.OutputExecution
