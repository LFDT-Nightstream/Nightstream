import NightstreamFPrime.Gadgets.Sampling.WideReduction.OutputHintValues

/-! Execution of the 353 retained result-bit hints agrees with the
existing gadget's explicit honest assignment. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.OutputExecution

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
open HintCertificate

theorem quotient_bits (interface : Interface) (base : Env) (helperOffset offset : Nat)
    (before : helperOffset + HintProgram.helperCount ≤ offset)
    (helpers : HelperValues.Present base helperOffset (drawOf interface base offset)) :
    Valid (completeNew interface base offset) (quotientStart offset)
      ((List.range quotientBitCount).map (HintProgram.quotientHint helperOffset)) := by
  have present := OutputHintValues.completeNew_helpers interface base helperOffset offset before helpers
  apply range_map
  · intro bit below
    change HintProgram.divisionColumn helperOffset (digitCount - 1)
      (HintProgram.divisionLimbs - 1 - bit / HintProgram.divisionBits) < quotientStart offset + bit
    simp only [HintProgram.divisionColumn, HintProgram.divisionStart, HintProgram.divisionStride,
      HintProgram.divisionLimbs, HintProgram.sourceBitCount, HintProgram.limbCount,
      HintProgram.limbStride, HintProgram.accumulatorBits, HintProgram.helperCount,
      HintProgram.divisionBits, digitCount, quotientStart, childWidth,
      Range.CanonicalU64.auxiliaryCount, fieldCount] at *
    omega
  · intro bit below
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
    Valid (completeNew interface base offset) (digitStart offset)
      ((List.range digitCount).flatMap fun digit =>
        (List.range digitBitCount).map (HintProgram.digitHint helperOffset digit)) := by
  have present := OutputHintValues.completeNew_helpers interface base helperOffset offset before helpers
  apply range_flatMap _ _ _ digitBitCount
  · intro digit _
    simp
  · intro digit digitBelow
    apply range_map
    · intro bit _
      change HintProgram.divisionColumn helperOffset digit (HintProgram.divisionLimbs - 1) + 1 <
        digitStart offset + digit * digitBitCount + bit
      simp only [HintProgram.divisionColumn, HintProgram.divisionStart, HintProgram.divisionStride,
        HintProgram.divisionLimbs, HintProgram.sourceBitCount, HintProgram.limbCount,
        HintProgram.limbStride, HintProgram.accumulatorBits, HintProgram.helperCount,
        digitCount, digitBitCount, digitStart, quotientStart, childWidth,
        Range.CanonicalU64.auxiliaryCount, fieldCount, quotientBitCount] at *
      omega
    · intro bit bitBelow
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
      simpa only [digitBit, Expr.eval_var, Nat.mul_comm digit digitBitCount] using value.symm

theorem check_bits (interface : Interface) (hints : Nat → List Hint) (base : Env) (offset : Nat)
    (children : ∀ index, Range.CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) base)
    (childRows : holdsFlat base (childOps interface offset))
    (childScope : ∀ expression ∈ flatConstraints (childOps interface offset),
      expression.VarsBelow (quotientStart offset)) :
    Valid (completeNew interface base offset) (checkStart offset)
      ((List.finRange checkCount).flatMap fun check =>
        (List.range checkBitCount).map fun bit => .bit (HintProgram.checkQuotient offset check) bit) := by
  have certified := completeNew_certificate interface hints base offset children childRows childScope
  apply finRange_flatMap _ _ _ checkBitCount
  · intro check
    simp
  · intro check
    apply range_map
    · intro bit _
      exact Expr.VarsBelow.mono _ (HintSupport.checkQuotient_below offset check) (by omega)
    · intro bit bitBelow
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
      simpa only [checkBit, Expr.eval_var, Nat.mul_comm check.val checkBitCount] using result.symm

theorem certificate (interface : Interface) (base : Env) (helperOffset offset : Nat)
    (before : helperOffset + HintProgram.helperCount ≤ offset)
    (helpers : HelperValues.Present base helperOffset (drawOf interface base offset))
    (children : ∀ index, Range.CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) base)
    (childRows : holdsFlat base (childOps interface offset))
    (childScope : ∀ expression ∈ flatConstraints (childOps interface offset),
      expression.VarsBelow (quotientStart offset)) :
    Valid (completeNew interface base offset) (quotientStart offset)
      (HintProgram.resultHints helperOffset offset) := by
  let target := completeNew interface base offset
  have quotients := quotient_bits interface base helperOffset offset before helpers
  have digits := digit_bits interface base helperOffset offset before helpers
  have checks := check_bits interface (HintProgram.resultHints helperOffset) base offset
    children childRows childScope
  have first := append target (quotientStart offset) _ _ quotients (by
    rw [List.length_map, List.length_range]
    exact digits)
  have all := append target (quotientStart offset) _ _ first (by
    simp only [List.length_append, List.length_map, List.length_range, List.length_flatMap,
      List.map_const', List.sum_replicate, smul_eq_mul]
    simpa only [checkStart, digitStart, Nat.add_assoc] using checks)
  simpa only [HintProgram.resultHints, List.append_assoc] using all

/-- Exact execution of the retained bits, with the canonical children and
temporary helpers already present. No constraint assumes hint correctness. -/
theorem execute_reference (interface : Interface) (base : Env) (helperOffset offset : Nat)
    (before : helperOffset + HintProgram.helperCount ≤ offset)
    (helpers : HelperValues.Present base helperOffset (drawOf interface base offset))
    (children : ∀ index, Range.CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) base)
    (childRows : holdsFlat base (childOps interface offset))
    (childScope : ∀ expression ∈ flatConstraints (childOps interface offset),
      expression.VarsBelow (quotientStart offset)) :
    executeHints base (quotientStart offset) (HintProgram.resultHints helperOffset offset) =
      completeNew interface base offset := by
  have valid := certificate interface base helperOffset offset before helpers children childRows childScope
  have assigned := HintCertificate.execute _ base (quotientStart offset) _ valid (by
    intro index below
    exact (completeNew_agreesOutside interface base offset index (Or.inl below)).symm)
  funext index
  by_cases inside : index < quotientStart offset + newBitCount
  · exact assigned index (by rwa [HintProgram.resultHints_length])
  · rw [executeHints_agrees_above _ _ _ index (by rw [HintProgram.resultHints_length]; omega)]
    exact (completeNew_agreesOutside interface base offset index (Or.inr (by omega))).symm

end NightstreamFPrime.Gadgets.Sampling.WideReduction.OutputExecution
