import Nightstream.Implementation.R1CS.Ownership.ShiftedTernary.ShiftedTernary

/-!
Soundness of the shifted-base-3 canonical opening gadget.

This module owns the arithmetic argument behind the generated borrow rows:
the final zero borrow forces the 41 ordinary trits to encode a number below
the Goldilocks modulus.  Exact correspondence between these semantics and the
generated rows remains inherited from `ShiftedTernary.canonicalRows_eq_artifact`.
-/

namespace Nightstream.Implementation.R1CS.ShiftedTernarySound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler

set_option maxRecDepth 262144

/-- Little-endian radix-three value of the first `count` digits. -/
def lowValue (digits : Nat → Nat) : Nat → Nat
  | 0 => 0
  | count + 1 => lowValue digits count + digits count * 3 ^ count

/-- Borrow after processing the first `count` little-endian digits. -/
def expectedBorrow (digits bounds : Nat → Nat) (initial : Nat) : Nat → Nat
  | 0 => initial
  | count + 1 =>
      if digits count + expectedBorrow digits bounds initial count > bounds count
      then 1 else 0

theorem expectedBorrow_le_one
    (digits bounds : Nat → Nat) (initial : Nat) (initialLe : initial ≤ 1) :
    ∀ count, expectedBorrow digits bounds initial count ≤ 1 := by
  intro count
  cases count with
  | zero => exact initialLe
  | succ count => simp only [expectedBorrow]; split <;> omega

theorem lowValue_lt_pow
    {digits : Nat → Nat} {count : Nat}
    (digitLt : ∀ index, index < count → digits index < 3) :
    lowValue digits count < 3 ^ count := by
  induction count with
  | zero => simp [lowValue]
  | succ count ih =>
      have prefixLt := ih (fun index indexLt => digitLt index (by omega))
      have lastLt := digitLt count (by omega)
      have cases : digits count = 0 ∨ digits count = 1 ∨ digits count = 2 := by
        omega
      rcases cases with zero | one | two
      · simp [lowValue, zero, Nat.pow_succ]
        omega
      · simp [lowValue, one, Nat.pow_succ]
        omega
      · simp [lowValue, two, Nat.pow_succ]
        omega

private theorem borrow_extension_iff
    {prefixValue boundPrefix digit boundDigit current previous power : Nat}
    (digitLt : digit < 3) (boundDigitLt : boundDigit < 3)
    (currentLe : current ≤ 1) (previousLe : previous ≤ 1)
    (prefixLt : prefixValue < power) (boundPrefixLt : boundPrefix < power)
    (powerPos : 0 < power)
    (previousIff : previous = 1 ↔ prefixValue + current > boundPrefix) :
    (if digit + previous > boundDigit then 1 else 0) = 1 ↔
      prefixValue + digit * power + current >
        boundPrefix + boundDigit * power := by
  have digitCases : digit = 0 ∨ digit = 1 ∨ digit = 2 := by omega
  have boundCases : boundDigit = 0 ∨ boundDigit = 1 ∨ boundDigit = 2 := by
    omega
  rcases digitCases with hd | hd | hd <;>
    rcases boundCases with hb | hb | hb <;>
    subst digit <;> subst boundDigit
  all_goals
    by_cases lower : prefixValue + current > boundPrefix
    · have previousEq : previous = 1 := previousIff.mpr lower
      simp only [previousEq]
      split <;> simp <;> omega
    · have notOne : previous ≠ 1 := fun equal => lower (previousIff.mp equal)
      have previousEq : previous = 0 := by omega
      simp only [previousEq]
      split <;> simp <;> omega

/-- The borrow recursion is exactly comparison of the represented prefixes. -/
theorem expectedBorrow_eq_one_iff
    {digits bounds : Nat → Nat} {initial count : Nat}
    (initialLe : initial ≤ 1)
    (digitLt : ∀ index, index < count → digits index < 3)
    (boundLt : ∀ index, index < count → bounds index < 3) :
    expectedBorrow digits bounds initial count = 1 ↔
      lowValue digits count + initial > lowValue bounds count := by
  induction count with
  | zero => simp [expectedBorrow, lowValue]; omega
  | succ count ih =>
      have prefixDigitLt : ∀ index, index < count → digits index < 3 :=
        fun index indexLt => digitLt index (by omega)
      have prefixBoundLt : ∀ index, index < count → bounds index < 3 :=
        fun index indexLt => boundLt index (by omega)
      have previousIff := ih prefixDigitLt prefixBoundLt
      have previousLe := expectedBorrow_le_one digits bounds initial initialLe count
      have digitPrefixLt := lowValue_lt_pow prefixDigitLt
      have boundPrefixLt := lowValue_lt_pow prefixBoundLt
      simp only [expectedBorrow, lowValue]
      exact borrow_extension_iff
        (digitLt count (by omega)) (boundLt count (by omega))
        initialLe previousLe digitPrefixLt boundPrefixLt (Nat.pow_pos (by decide))
        previousIff

theorem lowValue_add_le_of_expectedBorrow_zero
    {digits bounds : Nat → Nat} {initial count : Nat}
    (initialLe : initial ≤ 1)
    (digitLt : ∀ index, index < count → digits index < 3)
    (boundLt : ∀ index, index < count → bounds index < 3)
    (borrowZero : expectedBorrow digits bounds initial count = 0) :
    lowValue digits count + initial ≤ lowValue bounds count := by
  apply Nat.le_of_not_gt
  intro greater
  have borrowOne :=
    (expectedBorrow_eq_one_iff initialLe digitLt boundLt).mpr greater
  rw [borrowZero] at borrowOne
  exact Nat.zero_ne_one borrowOne

theorem lowValue_le_of_expectedBorrow_zero
    {digits bounds : Nat → Nat} {count : Nat}
    (digitLt : ∀ index, index < count → digits index < 3)
    (boundLt : ∀ index, index < count → bounds index < 3)
    (borrowZero : expectedBorrow digits bounds 0 count = 0) :
    lowValue digits count ≤ lowValue bounds count := by
  simpa only [Nat.add_zero] using
    lowValue_add_le_of_expectedBorrow_zero (initial := 0)
      (Nat.zero_le 1) digitLt boundLt borrowZero

theorem expectedBorrow_zero_of_lowValue_le
    {digits bounds : Nat → Nat} {count : Nat}
    (digitLt : ∀ index, index < count → digits index < 3)
    (boundLt : ∀ index, index < count → bounds index < 3)
    (bounded : lowValue digits count ≤ lowValue bounds count) :
    expectedBorrow digits bounds 0 count = 0 := by
  have leOne := expectedBorrow_le_one digits bounds 0 (by omega) count
  have notOne : expectedBorrow digits bounds 0 count ≠ 1 := by
    intro one
    have greater :=
      (expectedBorrow_eq_one_iff (by omega) digitLt boundLt).mp one
    omega
  omega

theorem Digit.tritValue_lt_three
    {value negative : Nat} (digit : Digit value negative) :
    tritValue value < 3 := by
  cases digit with
  | neg valueEq _ => simp [tritValue, valueEq]
  | zero valueEq _ => simp [tritValue, valueEq, goldilocksP]
  | pos valueEq _ => simp [tritValue, valueEq, goldilocksP]

theorem base3Digits_getD_lt_three
    (value count index : Nat) (indexLt : index < count) :
    (base3Digits value count).getD index 0 < 3 := by
  induction count generalizing value index with
  | zero => omega
  | succ count ih =>
      cases index with
      | zero =>
          simp only [base3Digits, List.getD_cons_zero]
          exact Nat.mod_lt _ (by decide)
      | succ index =>
          simp only [base3Digits, List.getD_cons_succ]
          exact ih (value / 3) index (by omega)

theorem boundDigit_lt_three {index : Nat} (indexLt : index < digitCount) :
    boundDigits.getD index 0 < 3 :=
  base3Digits_getD_lt_three (goldilocksP - 1) digitCount index indexLt

/-- Borrow wire before digit `index`, with the verifier-fixed zero sentinels. -/
def borrowAt (assignment : Nat → Nat) (index : Nat) : Nat :=
  if index = 0 then 0
  else if index = digitCount then 0
  else assignment (ShiftedTernary.borrowCols.getD (index - 1) 0)

theorem borrowAt_le_one_of_internal
    {assignment : Nat → Nat}
    (internalBits : ∀ index, index < digitCount - 1 →
      assignment (ShiftedTernary.borrowCols.getD index 0) ≤ 1) :
    ∀ index, index ≤ digitCount → borrowAt assignment index ≤ 1 := by
  intro index indexLe
  by_cases zero : index = 0
  · simp [borrowAt, zero]
  by_cases terminal : index = digitCount
  · simp [borrowAt, terminal]
  simp only [borrowAt, zero, terminal, ↓reduceIte]
  apply internalBits (index - 1)
  omega

theorem currentBorrowTerms_eval
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    {index : Nat} (indexLt : index < digitCount) :
    lcEval assignment (currentBorrowTerms index) = borrowAt assignment index := by
  by_cases zero : index = 0
  · subst index
    simp [currentBorrowTerms, borrowAt, lcEval]
  · have notTerminal : index ≠ digitCount := by omega
    have wireLt := canonical
      (ShiftedTernary.borrowCols.getD (index - 1) 0)
    simp only [currentBorrowTerms, zero, borrowAt, notTerminal, ↓reduceIte,
      lcEval, List.foldl, Nat.zero_add, Nat.one_mul]
    exact Nat.mod_eq_of_lt wireLt

theorem nextBorrowTerms_eval
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    {index : Nat} :
    lcEval assignment (nextBorrowTerms index 1) =
      borrowAt assignment (index + 1) := by
  by_cases terminal : index + 1 = digitCount
  · simp [nextBorrowTerms, terminal, borrowAt, lcEval]
  · have notZero : index + 1 ≠ 0 := by omega
    have wireLt := canonical (ShiftedTernary.borrowCols.getD index 0)
    simp only [nextBorrowTerms, terminal, borrowAt, notZero, ↓reduceIte,
      lcEval, List.foldl, Nat.zero_add, Nat.one_mul]
    exact Nat.mod_eq_of_lt wireLt

private theorem goldilocks_neg_residue {value : Nat}
    (valueLt : value < goldilocksP) :
    ((goldilocksP - 1) * value) % goldilocksP =
      if value = 0 then 0 else goldilocksP - value := by
  simp only [goldilocksP] at valueLt ⊢
  by_cases zero : value = 0
  · simp [zero]
  · have valuePos : 0 < value := Nat.pos_of_ne_zero zero
    have remainderLt :
        18446744069414584321 - value < 18446744069414584321 := by omega
    have decomposition :
        (18446744069414584321 - 1) * value =
          18446744069414584321 * (value - 1) +
            (18446744069414584321 - value) := by
      omega
    rw [decomposition, Nat.add_mod]
    simp [Nat.mod_eq_of_lt remainderLt, zero]

private theorem goldilocks_one_minus_eq_zero {value : Nat}
    (valueLt : value < goldilocksP)
    (zero : ((goldilocksP - 1) * value + 1) % goldilocksP = 0) :
    value = 1 := by
  rw [Nat.add_mod, goldilocks_neg_residue valueLt] at zero
  simp only [goldilocksP] at valueLt zero ⊢
  have oneMod : 1 % 18446744069414584321 = 1 :=
    Nat.mod_eq_of_lt (by decide)
  rw [oneMod] at zero
  by_cases valueZero : value = 0
  · simp [valueZero] at zero
  · simp only [valueZero, ↓reduceIte] at zero
    by_cases valueOne : value = 1
    · exact valueOne
    · have valueGtOne : 1 < value := by omega
      have sumLt :
          18446744069414584321 - value + 1 < 18446744069414584321 := by omega
      rw [Nat.mod_eq_of_lt sumLt] at zero
      omega

private theorem goldilocks_one_minus_eq_one {value : Nat}
    (valueLt : value < goldilocksP)
    (one : ((goldilocksP - 1) * value + 1) % goldilocksP = 1) :
    value = 0 := by
  rw [Nat.add_mod, goldilocks_neg_residue valueLt] at one
  simp only [goldilocksP] at valueLt one ⊢
  have oneMod : 1 % 18446744069414584321 = 1 :=
    Nat.mod_eq_of_lt (by decide)
  rw [oneMod] at one
  by_cases valueZero : value = 0
  · exact valueZero
  · simp only [valueZero, ↓reduceIte] at one
    by_cases valueOne : value = 1
    · simp [valueOne] at one
    · have valueGtOne : 1 < value := by omega
      have sumLt :
          18446744069414584321 - value + 1 < 18446744069414584321 := by omega
      rw [Nat.mod_eq_of_lt sumLt] at one
      omega

private theorem foldl_linearCombination_add
    (assignment : Nat → Nat) (terms : List (Nat × Nat)) (start : Nat) :
    terms.foldl (fun total term => total + term.2 * assignment term.1) start =
      start + terms.foldl
        (fun total term => total + term.2 * assignment term.1) 0 := by
  induction terms generalizing start with
  | nil => simp
  | cons term terms ih =>
      simp only [List.foldl]
      rw [ih]
      simp only [Nat.zero_add]
      rw [ih (term.2 * assignment term.1)]
      omega

theorem lcEval_append (assignment : Nat → Nat)
    (left right : List (Nat × Nat)) :
    lcEval assignment (left ++ right) =
      (lcEval assignment left + lcEval assignment right) % goldilocksP := by
  unfold lcEval
  rw [List.foldl_append, foldl_linearCombination_add, Nat.add_mod]

theorem negCurrentBorrowTerms_eval
    {assignment : Nat → Nat}
    {index : Nat} (indexLt : index < digitCount) :
    lcEval assignment ((currentBorrowTerms index).map
      (fun term => (term.1, goldilocksP - 1))) =
      ((goldilocksP - 1) * borrowAt assignment index) % goldilocksP := by
  by_cases zero : index = 0
  · subst index
    simp [currentBorrowTerms, borrowAt, lcEval]
  · have notTerminal : index ≠ digitCount := by omega
    simp only [currentBorrowTerms, zero, borrowAt, notTerminal, ↓reduceIte,
      List.map, lcEval, List.foldl, Nat.zero_add]

theorem negNextBorrowTerms_eval
    {assignment : Nat → Nat}
    {index : Nat} :
    lcEval assignment (nextBorrowTerms index (goldilocksP - 1)) =
      ((goldilocksP - 1) * borrowAt assignment (index + 1)) %
        goldilocksP := by
  by_cases terminal : index + 1 = digitCount
  · simp [nextBorrowTerms, terminal, borrowAt, lcEval]
  · have notZero : index + 1 ≠ 0 := by omega
    simp only [nextBorrowTerms, terminal, borrowAt, notZero,
      ↓reduceIte, lcEval, List.foldl, Nat.zero_add]
    simp

def fieldOneMinus (value : Nat) : Nat :=
  ((goldilocksP - 1) * value + 1) % goldilocksP

private theorem fieldOneMinus_eq_zero {value : Nat}
    (valueLt : value < goldilocksP)
    (zero : fieldOneMinus value = 0) : value = 1 :=
  goldilocks_one_minus_eq_zero valueLt zero

private theorem fieldOneMinus_eq_one {value : Nat}
    (valueLt : value < goldilocksP)
    (one : fieldOneMinus value = 1) : value = 0 :=
  goldilocks_one_minus_eq_one valueLt one

theorem borrowAt_lt_modulus
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (index : Nat) : borrowAt assignment index < goldilocksP := by
  by_cases zero : index = 0
  · simp [borrowAt, zero, goldilocksP]
  by_cases terminal : index = digitCount
  · simp [borrowAt, terminal, goldilocksP]
  simp only [borrowAt, zero, terminal, ↓reduceIte]
  exact canonical _

private theorem borrowRow_bound_zero_equation
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {index : Nat} (indexLt : index < digitCount)
    (boundZero : boundDigits.getD index 0 = 0)
    (holds : RowHolds assignment (borrowRow index)) :
    assignment (ShiftedTernary.negativeCols.getD index 0) *
          fieldOneMinus (borrowAt assignment index) % goldilocksP =
      fieldOneMinus (borrowAt assignment (index + 1)) := by
  have equation := holds
  simp only [borrowRow, boundZero, RowHolds] at equation
  rw [lcEval_append, lcEval_append,
    negCurrentBorrowTerms_eval indexLt,
    negNextBorrowTerms_eval] at equation
  have negativeLt := canonical
    (ShiftedTernary.negativeCols.getD index 0)
  simpa [fieldOneMinus, lcEval, one, Nat.mod_eq_of_lt negativeLt,
    Nat.add_mod, goldilocksP] using equation

private theorem borrowRow_bound_zero_forces
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    {index : Nat}
    (digit : Digit
      (assignment (ShiftedTernary.digitCols.getD index 0))
      (assignment (ShiftedTernary.negativeCols.getD index 0)))
    (currentLe : borrowAt assignment index ≤ 1)
    (equation :
      assignment (ShiftedTernary.negativeCols.getD index 0) *
            fieldOneMinus (borrowAt assignment index) % goldilocksP =
        fieldOneMinus (borrowAt assignment (index + 1))) :
    borrowAt assignment (index + 1) =
      if tritValue
            (assignment (ShiftedTernary.digitCols.getD index 0)) +
          borrowAt assignment index > 0
      then 1 else 0 := by
  have nextLt := borrowAt_lt_modulus canonical (index + 1)
  have currentCases :
      borrowAt assignment index = 0 ∨ borrowAt assignment index = 1 := by
    omega
  cases digit with
  | neg valueEq negativeEq =>
      rcases currentCases with currentEq | currentEq
      · rw [negativeEq, currentEq] at equation
        have nextEq : borrowAt assignment (index + 1) = 0 :=
          fieldOneMinus_eq_one nextLt (by
            simpa [fieldOneMinus, goldilocksP] using equation.symm)
        rw [valueEq, currentEq, nextEq]
        simp [tritValue]
      · rw [negativeEq, currentEq] at equation
        have nextEq : borrowAt assignment (index + 1) = 1 :=
          fieldOneMinus_eq_zero nextLt (by
            simpa [fieldOneMinus, goldilocksP] using equation.symm)
        rw [valueEq, currentEq, nextEq]
        simp [tritValue]
  | zero valueEq negativeEq =>
      rw [negativeEq] at equation
      have nextEq : borrowAt assignment (index + 1) = 1 :=
        fieldOneMinus_eq_zero nextLt (by
          simpa [fieldOneMinus, goldilocksP] using equation.symm)
      rcases currentCases with currentEq | currentEq <;>
        rw [valueEq, currentEq, nextEq] <;> simp [tritValue, goldilocksP]
  | pos valueEq negativeEq =>
      rw [negativeEq] at equation
      have nextEq : borrowAt assignment (index + 1) = 1 :=
        fieldOneMinus_eq_zero nextLt (by
          simpa [fieldOneMinus, goldilocksP] using equation.symm)
      rcases currentCases with currentEq | currentEq <;>
        rw [valueEq, currentEq, nextEq] <;> simp [tritValue, goldilocksP]

def fieldPositive (value negative : Nat) : Nat :=
  (value + negative) % goldilocksP

def fieldZeroIndicator (value negative : Nat) : Nat :=
  ((goldilocksP - 1) * value +
    (goldilocksP - 2) * negative + 1) % goldilocksP

def fieldNextMinusPositive (next value negative : Nat) : Nat :=
  (next + (goldilocksP - 1) * value +
    (goldilocksP - 1) * negative) % goldilocksP

private theorem goldilocks_add_pred_mod_zero {value : Nat}
    (valueLt : value < goldilocksP)
    (zero : (value + (goldilocksP - 1)) % goldilocksP = 0) :
    value = 1 := by
  simp only [goldilocksP] at valueLt zero ⊢
  cases value with
  | zero =>
      simp only [Nat.zero_add] at zero
      rw [Nat.mod_eq_of_lt (by decide)] at zero
      omega
  | succ predecessor =>
      have predecessorLt : predecessor < 18446744069414584321 := by omega
      have sumEq :
          Nat.succ predecessor + (18446744069414584321 - 1) =
            18446744069414584321 + predecessor := by omega
      rw [sumEq, Nat.add_mod] at zero
      simp only [Nat.mod_self, Nat.zero_add,
        Nat.mod_eq_of_lt predecessorLt] at zero
      omega

private theorem fieldZeroIndicator_neg :
    fieldZeroIndicator (goldilocksP - 1) 1 = 0 := by
  native_decide

private theorem fieldZeroIndicator_zero :
    fieldZeroIndicator 0 0 = 1 := by
  native_decide

private theorem fieldZeroIndicator_pos :
    fieldZeroIndicator 1 0 = 0 := by
  native_decide

private theorem fieldNextMinusPositive_neg {next : Nat}
    (nextLt : next < goldilocksP) :
    fieldNextMinusPositive next (goldilocksP - 1) 1 = next := by
  have constant :
      ((18446744069414584321 - 1) * (18446744069414584321 - 1) +
        (18446744069414584321 - 1) * 1) % 18446744069414584321 = 0 := by
    native_decide
  simp only [goldilocksP] at nextLt ⊢
  unfold fieldNextMinusPositive
  simp only [goldilocksP]
  rw [Nat.add_assoc, Nat.add_mod, constant]
  simp [Nat.mod_eq_of_lt nextLt]

private theorem fieldNextMinusPositive_zero {next : Nat}
    (nextLt : next < goldilocksP) :
    fieldNextMinusPositive next 0 0 = next := by
  simp [fieldNextMinusPositive, Nat.mod_eq_of_lt nextLt]

private theorem fieldNextMinusPositive_pos (next : Nat) :
    fieldNextMinusPositive next 1 0 =
      (next + (goldilocksP - 1)) % goldilocksP := by
  simp [fieldNextMinusPositive]

private theorem borrowRow_bound_one_equation
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {index : Nat} (indexLt : index < digitCount)
    (boundOne : boundDigits.getD index 0 = 1)
    (holds : RowHolds assignment (borrowRow index)) :
    fieldZeroIndicator
          (assignment (ShiftedTernary.digitCols.getD index 0))
          (assignment (ShiftedTernary.negativeCols.getD index 0)) *
        borrowAt assignment index % goldilocksP =
      fieldNextMinusPositive
        (borrowAt assignment (index + 1))
        (assignment (ShiftedTernary.digitCols.getD index 0))
        (assignment (ShiftedTernary.negativeCols.getD index 0)) := by
  have equation := holds
  simp only [borrowRow, boundOne, RowHolds] at equation
  rw [currentBorrowTerms_eval canonical indexLt,
    lcEval_append, nextBorrowTerms_eval canonical] at equation
  simpa [fieldZeroIndicator, fieldNextMinusPositive, lcEval, one,
    Nat.add_mod, Nat.add_assoc, goldilocksP] using equation

private theorem borrowRow_bound_one_forces
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    {index : Nat}
    (digit : Digit
      (assignment (ShiftedTernary.digitCols.getD index 0))
      (assignment (ShiftedTernary.negativeCols.getD index 0)))
    (currentLe : borrowAt assignment index ≤ 1)
    (equation :
      fieldZeroIndicator
            (assignment (ShiftedTernary.digitCols.getD index 0))
            (assignment (ShiftedTernary.negativeCols.getD index 0)) *
          borrowAt assignment index % goldilocksP =
        fieldNextMinusPositive
          (borrowAt assignment (index + 1))
          (assignment (ShiftedTernary.digitCols.getD index 0))
          (assignment (ShiftedTernary.negativeCols.getD index 0))) :
    borrowAt assignment (index + 1) =
      if tritValue
            (assignment (ShiftedTernary.digitCols.getD index 0)) +
          borrowAt assignment index > 1
      then 1 else 0 := by
  have nextLt := borrowAt_lt_modulus canonical (index + 1)
  have currentCases :
      borrowAt assignment index = 0 ∨ borrowAt assignment index = 1 := by
    omega
  cases digit with
  | neg valueEq negativeEq =>
      rw [valueEq, negativeEq] at equation
      rw [fieldZeroIndicator_neg,
        fieldNextMinusPositive_neg nextLt] at equation
      have nextEq : borrowAt assignment (index + 1) = 0 := by
        simpa using equation.symm
      rcases currentCases with currentEq | currentEq <;>
        rw [valueEq, currentEq, nextEq] <;> simp [tritValue]
  | zero valueEq negativeEq =>
      rcases currentCases with currentEq | currentEq
      · rw [valueEq, negativeEq, currentEq] at equation
        rw [fieldZeroIndicator_zero,
          fieldNextMinusPositive_zero nextLt] at equation
        have nextEq : borrowAt assignment (index + 1) = 0 := by
          simpa [goldilocksP] using equation.symm
        rw [valueEq, currentEq, nextEq]
        simp [tritValue, goldilocksP]
      · rw [valueEq, negativeEq, currentEq] at equation
        rw [fieldZeroIndicator_zero,
          fieldNextMinusPositive_zero nextLt] at equation
        have nextEq : borrowAt assignment (index + 1) = 1 := by
          simpa [goldilocksP] using equation.symm
        rw [valueEq, currentEq, nextEq]
        simp [tritValue, goldilocksP]
  | pos valueEq negativeEq =>
      rw [valueEq, negativeEq] at equation
      rw [fieldZeroIndicator_pos,
        fieldNextMinusPositive_pos] at equation
      have nextEq : borrowAt assignment (index + 1) = 1 :=
        goldilocks_add_pred_mod_zero nextLt (by
          simpa using equation.symm)
      rcases currentCases with currentEq | currentEq <;>
        rw [valueEq, currentEq, nextEq] <;> simp [tritValue, goldilocksP]

private theorem borrowRow_bound_two_equation
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    {index : Nat} (indexLt : index < digitCount)
    (boundTwo : boundDigits.getD index 0 = 2)
    (holds : RowHolds assignment (borrowRow index)) :
    fieldPositive
          (assignment (ShiftedTernary.digitCols.getD index 0))
          (assignment (ShiftedTernary.negativeCols.getD index 0)) *
        borrowAt assignment index % goldilocksP =
      borrowAt assignment (index + 1) := by
  have equation := holds
  simp only [borrowRow, boundTwo, RowHolds] at equation
  rw [currentBorrowTerms_eval canonical indexLt,
    nextBorrowTerms_eval canonical] at equation
  simpa [fieldPositive, lcEval, Nat.add_mod, goldilocksP] using equation

private theorem borrowRow_bound_two_forces
    {assignment : Nat → Nat}
    {index : Nat}
    (digit : Digit
      (assignment (ShiftedTernary.digitCols.getD index 0))
      (assignment (ShiftedTernary.negativeCols.getD index 0)))
    (currentLe : borrowAt assignment index ≤ 1)
    (equation :
      fieldPositive
            (assignment (ShiftedTernary.digitCols.getD index 0))
            (assignment (ShiftedTernary.negativeCols.getD index 0)) *
          borrowAt assignment index % goldilocksP =
        borrowAt assignment (index + 1)) :
    borrowAt assignment (index + 1) =
      if tritValue
            (assignment (ShiftedTernary.digitCols.getD index 0)) +
          borrowAt assignment index > 2
      then 1 else 0 := by
  have currentCases :
      borrowAt assignment index = 0 ∨ borrowAt assignment index = 1 := by
    omega
  cases digit with
  | neg valueEq negativeEq =>
      rcases currentCases with currentEq | currentEq
      · rw [valueEq, negativeEq, currentEq] at equation
        rw [valueEq, currentEq]
        simp [fieldPositive, tritValue, goldilocksP] at equation ⊢
        omega
      · rw [valueEq, negativeEq, currentEq] at equation
        rw [valueEq, currentEq]
        simp [fieldPositive, tritValue, goldilocksP] at equation ⊢
        omega
  | zero valueEq negativeEq =>
      rcases currentCases with currentEq | currentEq
      · rw [valueEq, negativeEq, currentEq] at equation
        rw [valueEq, currentEq]
        simp [fieldPositive, tritValue, goldilocksP] at equation ⊢
        omega
      · rw [valueEq, negativeEq, currentEq] at equation
        rw [valueEq, currentEq]
        simp [fieldPositive, tritValue, goldilocksP] at equation ⊢
        omega
  | pos valueEq negativeEq =>
      rcases currentCases with currentEq | currentEq
      · rw [valueEq, negativeEq, currentEq] at equation
        rw [valueEq, currentEq]
        simp [fieldPositive, tritValue, goldilocksP] at equation ⊢
        omega
      · rw [valueEq, negativeEq, currentEq] at equation
        rw [valueEq, currentEq]
        simp [fieldPositive, tritValue, goldilocksP] at equation ⊢
        omega

/-- Each generated borrow row enforces exactly one radix-three comparison
step; no Boolean annotation is assumed for the next wire. -/
theorem borrowRow_forces_step
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {index : Nat} (indexLt : index < digitCount)
    (digit : Digit
      (assignment (ShiftedTernary.digitCols.getD index 0))
      (assignment (ShiftedTernary.negativeCols.getD index 0)))
    (currentLe : borrowAt assignment index ≤ 1)
    (holds : RowHolds assignment (borrowRow index)) :
    borrowAt assignment (index + 1) =
      if tritValue
            (assignment (ShiftedTernary.digitCols.getD index 0)) +
          borrowAt assignment index > boundDigits.getD index 0
      then 1 else 0 := by
  have boundLt := boundDigit_lt_three indexLt
  have boundCases :
      boundDigits.getD index 0 = 0 ∨
      boundDigits.getD index 0 = 1 ∨
      boundDigits.getD index 0 = 2 := by
    omega
  rcases boundCases with boundZero | boundOne | boundTwo
  · rw [boundZero]
    exact borrowRow_bound_zero_forces canonical digit currentLe
      (borrowRow_bound_zero_equation canonical one indexLt boundZero holds)
  · rw [boundOne]
    exact borrowRow_bound_one_forces canonical digit currentLe
      (borrowRow_bound_one_equation canonical one indexLt boundOne holds)
  · rw [boundTwo]
    exact borrowRow_bound_two_forces digit currentLe
      (borrowRow_bound_two_equation canonical indexLt boundTwo holds)

def assignmentTrit (assignment : Nat → Nat) (index : Nat) : Nat :=
  tritValue (assignment (ShiftedTernary.digitCols.getD index 0))

def boundDigit (index : Nat) : Nat := boundDigits.getD index 0

theorem borrowRow_holds_of_satisfies
    {assignment : Nat → Nat}
    (satisfies : Satisfies canonicalRows assignment)
    {index : Nat} (indexLt : index < digitCount) :
    RowHolds assignment (borrowRow index) := by
  apply satisfies
  unfold canonicalRows
  apply List.mem_append_right
  unfold borrowRows
  apply List.mem_map.mpr
  exact ⟨index, List.mem_range.mpr indexLt, rfl⟩

theorem borrowTrace_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies canonicalRows assignment) :
    ∀ count, count ≤ digitCount →
      borrowAt assignment count =
        expectedBorrow (assignmentTrit assignment) boundDigit 0 count := by
  have digits := allDigits_sound_of_canonicalRows prime canonical one satisfies
  intro count countLe
  induction count with
  | zero => simp [borrowAt, expectedBorrow]
  | succ count ih =>
      have countLt : count < digitCount := by omega
      have prefixEq := ih (by omega)
      have currentLe : borrowAt assignment count ≤ 1 := by
        rw [prefixEq]
        exact expectedBorrow_le_one _ _ 0 (by omega) count
      have step := borrowRow_forces_step canonical one countLt
        (digits.atIndex countLt) currentLe
        (borrowRow_holds_of_satisfies satisfies countLt)
      simpa [expectedBorrow, assignmentTrit, boundDigit, prefixEq] using step

theorem boundLowValue :
    lowValue boundDigit digitCount = goldilocksP - 1 := by
  native_decide

theorem foldl_range_eq_lowValue
    (digits : Nat → Nat) (start count : Nat) :
    (List.range count).foldl
        (fun total index => total + 3 ^ index * digits index) start =
      start + lowValue digits count := by
  induction count with
  | zero => simp [lowValue]
  | succ count ih =>
      rw [List.range_succ, List.foldl_append]
      simp only [List.foldl]
      rw [ih]
      rw [Nat.mul_comm (3 ^ count) (digits count)]
      change start + lowValue digits count + digits count * 3 ^ count =
        start + (lowValue digits count + digits count * 3 ^ count)
      omega

theorem encodedValue_eq_lowValue (assignment : Nat → Nat) :
    encodedValue assignment = lowValue (assignmentTrit assignment) digitCount := by
  unfold encodedValue assignmentTrit
  simpa only [Nat.zero_add] using foldl_range_eq_lowValue
    (fun index => tritValue
      (assignment (ShiftedTernary.digitCols.getD index 0))) 0 digitCount

theorem assignmentTrit_lt_three
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies canonicalRows assignment) :
    ∀ index, index < digitCount → assignmentTrit assignment index < 3 := by
  have digits := allDigits_sound_of_canonicalRows prime canonical one satisfies
  intro index indexLt
  exact Digit.tritValue_lt_three (digits.atIndex indexLt)

theorem expectedBorrow_final_zero
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
  (satisfies : Satisfies canonicalRows assignment) :
    expectedBorrow (assignmentTrit assignment) boundDigit 0 digitCount = 0 := by
  rw [← borrowTrace_sound prime canonical one satisfies digitCount Nat.le.refl]
  simp [borrowAt]

theorem encodedLowValue_le_bound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies canonicalRows assignment) :
    lowValue (assignmentTrit assignment) digitCount ≤
      lowValue boundDigit digitCount := by
  exact lowValue_le_of_expectedBorrow_zero
    (digits := assignmentTrit assignment) (bounds := boundDigit)
    (count := digitCount)
    (assignmentTrit_lt_three (assignment := assignment)
      prime canonical one satisfies)
    (fun index indexLt => boundDigit_lt_three indexLt)
    (expectedBorrow_final_zero (assignment := assignment)
      prime canonical one satisfies)

private theorem goldilocks_pred_lt : goldilocksP - 1 < goldilocksP := by
  native_decide

theorem encodedValue_lt_modulus
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies canonicalRows assignment) :
    encodedValue assignment < goldilocksP := by
  rw [encodedValue_eq_lowValue]
  have bounded := encodedLowValue_le_bound prime canonical one satisfies
  rw [boundLowValue] at bounded
  exact Nat.lt_of_le_of_lt bounded goldilocks_pred_lt

def centeredDigit (assignment : Nat → Nat) (index : Nat) : Nat :=
  assignment (ShiftedTernary.digitCols.getD index 0)

def negativeWeightedValue (assignment : Nat → Nat) : Nat → Nat
  | 0 => 0
  | count + 1 =>
      negativeWeightedValue assignment count +
        (goldilocksP - 3 ^ count) * centeredDigit assignment count

theorem power_lt_modulus {index : Nat} (indexLt : index < digitCount) :
    3 ^ index < goldilocksP := by
  have indexLe : index ≤ 40 := by
    simp only [digitCount] at indexLt
    omega
  have powerLe : 3 ^ index ≤ 3 ^ 40 :=
    Nat.pow_le_pow_right (by decide) indexLe
  exact Nat.lt_of_le_of_lt powerLe (by native_decide)

private theorem negative_term_add_centered
    {power value : Nat} (powerLe : power ≤ goldilocksP) :
    (goldilocksP - power) * value + power * value =
      goldilocksP * value := by
  rw [← Nat.add_mul, Nat.sub_add_cancel powerLe]

theorem negative_add_centered_mod_zero
    (assignment : Nat → Nat) : ∀ count, count ≤ digitCount →
    (negativeWeightedValue assignment count +
      lowValue (centeredDigit assignment) count) % goldilocksP = 0 := by
  intro count countLe
  induction count with
  | zero => simp [negativeWeightedValue, lowValue]
  | succ count ih =>
      have countLt : count < digitCount := by omega
      have prefixZero := ih (by omega)
      have term := negative_term_add_centered
        (Nat.le_of_lt (power_lt_modulus countLt))
        (value := centeredDigit assignment count)
      rw [negativeWeightedValue, lowValue]
      rw [Nat.mul_comm (centeredDigit assignment count) (3 ^ count)]
      calc
        (negativeWeightedValue assignment count +
            (goldilocksP - 3 ^ count) * centeredDigit assignment count +
            (lowValue (centeredDigit assignment) count +
              3 ^ count * centeredDigit assignment count)) % goldilocksP =
            ((negativeWeightedValue assignment count +
                lowValue (centeredDigit assignment) count) +
              ((goldilocksP - 3 ^ count) * centeredDigit assignment count +
                3 ^ count * centeredDigit assignment count)) % goldilocksP := by
                  congr 1
                  omega
        _ = ((negativeWeightedValue assignment count +
                lowValue (centeredDigit assignment) count) +
              goldilocksP * centeredDigit assignment count) % goldilocksP := by
                rw [term]
        _ = 0 := by
              rw [Nat.add_mod, prefixZero]
              simp

theorem foldl_range_eq_negativeWeightedValue
    (assignment : Nat → Nat) (start count : Nat) :
    (List.range count).foldl (fun total index =>
        total + (goldilocksP - 3 ^ index) * centeredDigit assignment index) start =
      start + negativeWeightedValue assignment count := by
  induction count with
  | zero => simp [negativeWeightedValue]
  | succ count ih =>
      rw [List.range_succ, List.foldl_append]
      simp only [List.foldl]
      rw [ih]
      change start + negativeWeightedValue assignment count +
          (goldilocksP - 3 ^ count) * centeredDigit assignment count =
        start + (negativeWeightedValue assignment count +
          (goldilocksP - 3 ^ count) * centeredDigit assignment count)
      omega

theorem reconstructionRow_holds_of_satisfies
    {assignment : Nat → Nat}
    (satisfies : Satisfies canonicalRows assignment) :
    RowHolds assignment reconstructionRow := by
  apply satisfies
  unfold canonicalRows
  apply List.mem_append_left
  apply List.mem_append_right
  simp

theorem reconstruction_equation_of_satisfies
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (satisfies : Satisfies canonicalRows assignment) :
    (assignment ShiftedTernary.fieldCol +
      negativeWeightedValue assignment digitCount) % goldilocksP = 0 := by
  have equation := reconstructionRow_holds_of_satisfies satisfies
  simp only [RowHolds, reconstructionRow, lcEval, List.foldl, one,
    Nat.one_mul, Nat.mul_one, Nat.zero_add, Nat.zero_mod] at equation
  simp only [List.foldl_map] at equation
  have folded := foldl_range_eq_negativeWeightedValue assignment
    (assignment ShiftedTernary.fieldCol) digitCount
  unfold centeredDigit at folded
  rw [folded] at equation
  simpa [goldilocksP] using equation

private theorem mod_eq_of_complement
    {value complement target modulus : Nat}
    (valueComplement : (value + complement) % modulus = 0)
    (complementTarget : (complement + target) % modulus = 0) :
    value % modulus = target % modulus := by
  have appendTarget :
      (value + complement + target) % modulus = target % modulus := by
    rw [Nat.add_mod, valueComplement]
    simp
  have appendValue :
      (value + complement + target) % modulus = value % modulus := by
    rw [Nat.add_assoc, Nat.add_mod, complementTarget]
    simp
  exact appendValue.symm.trans appendTarget

theorem field_eq_centered_mod
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (satisfies : Satisfies canonicalRows assignment) :
    assignment ShiftedTernary.fieldCol % goldilocksP =
      lowValue (centeredDigit assignment) digitCount % goldilocksP := by
  exact mod_eq_of_complement
    (reconstruction_equation_of_satisfies one satisfies)
    (negative_add_centered_mod_zero assignment digitCount Nat.le.refl)

theorem Digit.add_one_mod_eq_tritValue
    {value negative : Nat} (digit : Digit value negative) :
    (value + 1) % goldilocksP = tritValue value := by
  cases digit with
  | neg valueEq _ => simp [valueEq, tritValue, goldilocksP]
  | zero valueEq _ => simp [valueEq, tritValue, goldilocksP]
  | pos valueEq _ => simp [valueEq, tritValue, goldilocksP]

theorem lowValue_mod_congr
    {left right : Nat → Nat} {count : Nat}
    (pointwise : ∀ index, index < count →
      left index % goldilocksP = right index % goldilocksP) :
    lowValue left count % goldilocksP =
      lowValue right count % goldilocksP := by
  induction count with
  | zero => simp [lowValue]
  | succ count ih =>
      rw [lowValue, lowValue]
      have prefixEq := ih (fun index indexLt => pointwise index (by omega))
      have lastEq := pointwise count (by omega)
      calc
        (lowValue left count + left count * 3 ^ count) % goldilocksP =
            (lowValue left count % goldilocksP +
              (left count % goldilocksP *
                (3 ^ count % goldilocksP)) % goldilocksP) % goldilocksP := by
                  simp [Nat.add_mod, Nat.mul_mod]
        _ = (lowValue right count % goldilocksP +
              (right count % goldilocksP *
                (3 ^ count % goldilocksP)) % goldilocksP) % goldilocksP := by
                  rw [prefixEq, lastEq]
        _ = (lowValue right count + right count * 3 ^ count) %
              goldilocksP := by
                  simp [Nat.add_mod, Nat.mul_mod]

theorem lowValue_pointwise_add (left right : Nat → Nat) :
    ∀ count,
    lowValue (fun index => left index + right index) count =
      lowValue left count + lowValue right count := by
  intro count
  induction count with
  | zero => simp [lowValue]
  | succ count ih =>
      rw [lowValue, lowValue, lowValue, ih, Nat.add_mul]
      omega

theorem shift_eq_ones_lowValue :
    shift = lowValue (fun _ => 1) digitCount := by
  native_decide

theorem encodedValue_centered_shift_mod_of_digits
    {assignment : Nat → Nat}
    (digits : DigitsHold assignment) :
    encodedValue assignment % goldilocksP =
      (lowValue (centeredDigit assignment) digitCount + shift) % goldilocksP := by
  rw [encodedValue_eq_lowValue]
  calc
    lowValue (assignmentTrit assignment) digitCount % goldilocksP =
        lowValue (fun index => centeredDigit assignment index + 1)
          digitCount % goldilocksP := by
            symm
            apply lowValue_mod_congr
            intro index indexLt
            have digit := digits.atIndex indexLt
            have tritLt : tritValue
                (assignment (ShiftedTernary.digitCols.getD index 0)) <
                goldilocksP := Nat.lt_trans
                  (Digit.tritValue_lt_three digit) (by native_decide)
            unfold centeredDigit assignmentTrit
            rw [Nat.mod_eq_of_lt tritLt]
            exact Digit.add_one_mod_eq_tritValue digit
    _ = (lowValue (centeredDigit assignment) digitCount +
          lowValue (fun _ => 1) digitCount) % goldilocksP := by
            rw [lowValue_pointwise_add]
    _ = (lowValue (centeredDigit assignment) digitCount + shift) %
          goldilocksP := by rw [shift_eq_ones_lowValue]

theorem encodedValue_centered_shift_mod
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies canonicalRows assignment) :
    encodedValue assignment % goldilocksP =
      (lowValue (centeredDigit assignment) digitCount + shift) % goldilocksP :=
  encodedValue_centered_shift_mod_of_digits
    (allDigits_sound_of_canonicalRows prime canonical one satisfies)

theorem fieldMatches_of_satisfies
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies canonicalRows assignment) :
    (assignment ShiftedTernary.fieldCol + shift) % goldilocksP =
      encodedValue assignment := by
  have fieldCongruence := field_eq_centered_mod one satisfies
  have encodedCongruence :=
    encodedValue_centered_shift_mod prime canonical one satisfies
  have encodedLt := encodedValue_lt_modulus prime canonical one satisfies
  calc
    (assignment ShiftedTernary.fieldCol + shift) % goldilocksP =
        (assignment ShiftedTernary.fieldCol % goldilocksP +
          shift % goldilocksP) % goldilocksP := Nat.add_mod ..
    _ = (lowValue (centeredDigit assignment) digitCount % goldilocksP +
          shift % goldilocksP) % goldilocksP := by rw [fieldCongruence]
    _ = (lowValue (centeredDigit assignment) digitCount + shift) %
          goldilocksP := (Nat.add_mod ..).symm
    _ = encodedValue assignment % goldilocksP := encodedCongruence.symm
    _ = encodedValue assignment := Nat.mod_eq_of_lt encodedLt

/-- Exact one-field soundness: every satisfying assignment to the generated
canonical-encoding rows determines one unique shifted-base-3 opening. -/
theorem canonicalOpening_of_canonicalRows
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies canonicalRows assignment) :
    CanonicalOpening assignment where
  digits := fun _ indexLt =>
    (allDigits_sound_of_canonicalRows prime canonical one satisfies).atIndex indexLt
  encodedLt := encodedValue_lt_modulus prime canonical one satisfies
  fieldMatches := fieldMatches_of_satisfies prime canonical one satisfies

theorem canonicalOpening_of_satisfies
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies ShiftedTernary.rows assignment) :
    CanonicalOpening assignment :=
  canonicalOpening_of_canonicalRows prime canonical one
    (canonicalRows_satisfy satisfies)

def dimensionRow : Row :=
  ⟨[(ShiftedTernary.commitmentDCol, 1),
      (0, goldilocksP - SeededPhi81.dimension)], [(0, 1)], []⟩

def kappaRow : Row :=
  ⟨[(ShiftedTernary.commitmentKappaCol, 1),
      (0, goldilocksP - 1)], [(0, 1)], []⟩

theorem shapeRows_eq_artifact :
    [dimensionRow, kappaRow] = ShiftedTernary.rows.take 2 := by
  decide

theorem shapeRows_satisfy
    {assignment : Nat → Nat}
    (satisfies : Satisfies ShiftedTernary.rows assignment) :
    Satisfies [dimensionRow, kappaRow] assignment := by
  rw [shapeRows_eq_artifact]
  intro row member
  exact satisfies row (List.mem_of_mem_take member)

private theorem add_complement_mod_zero
    {value constant : Nat}
    (valueLt : value < goldilocksP)
    (constantPos : 0 < constant)
    (constantLt : constant < goldilocksP)
    (zero : (value + (goldilocksP - constant)) % goldilocksP = 0) :
    value = constant := by
  by_cases below : value + (goldilocksP - constant) < goldilocksP
  · rw [Nat.mod_eq_of_lt below] at zero
    omega
  · have modulusLe :
        goldilocksP ≤ value + (goldilocksP - constant) :=
      Nat.le_of_not_gt below
    rw [Nat.mod_eq_sub_mod modulusLe] at zero
    have reducedLt :
        value + (goldilocksP - constant) - goldilocksP < goldilocksP := by
      omega
    rw [Nat.mod_eq_of_lt reducedLt] at zero
    omega

private theorem shapeRow_sound
    {assignment : Nat → Nat} {column constant : Nat}
    (canonical : ∀ wire, assignment wire < goldilocksP)
    (one : assignment 0 = 1)
    (constantPos : 0 < constant)
    (constantLt : constant < goldilocksP)
    (holds : RowHolds assignment
      ⟨[(column, 1), (0, goldilocksP - constant)], [(0, 1)], []⟩) :
    assignment column = constant := by
  have equation := holds
  have valueLt := canonical column
  simp only [RowHolds, lcEval, List.foldl, one, Nat.zero_add,
    Nat.one_mul, Nat.mul_one, Nat.zero_mod] at equation
  have zero :
      (assignment column + (goldilocksP - constant)) % goldilocksP = 0 := by
    simpa [goldilocksP] using equation
  exact add_complement_mod_zero valueLt constantPos constantLt zero

theorem commitmentShape_of_satisfies
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies ShiftedTernary.rows assignment) :
    assignment ShiftedTernary.commitmentDCol = SeededPhi81.dimension ∧
      assignment ShiftedTernary.commitmentKappaCol = 1 := by
  have shape := shapeRows_satisfy satisfies
  constructor
  · apply shapeRow_sound canonical one (by decide) (by native_decide)
    exact shape dimensionRow (by simp)
  · apply shapeRow_sound canonical one (by decide) (by native_decide)
    exact shape kappaRow (by simp)

theorem commitmentBlock_valid : ShiftedTernary.commitmentBlock.Valid := by
  native_decide

theorem commitmentRows_eq_artifact :
    ShiftedTernary.commitmentBlock.rows = ShiftedTernary.rows.drop 126 := by
  native_decide

theorem commitmentHolds_of_satisfies
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies ShiftedTernary.rows assignment) :
    ShiftedTernary.commitmentBlock.Holds assignment := by
  apply SeededPhi81.sound canonical one
  rw [commitmentRows_eq_artifact]
  intro row member
  exact satisfies row (List.mem_of_mem_drop member)

structure OneFieldSound (assignment : Nat → Nat) : Prop where
  opening : CanonicalOpening assignment
  dimension : assignment ShiftedTernary.commitmentDCol = SeededPhi81.dimension
  kappa : assignment ShiftedTernary.commitmentKappaCol = 1
  commitment : ShiftedTernary.commitmentBlock.Holds assignment

/-- Complete semantic soundness contract for all 180 exact rows emitted by the
one-field production SIS compiler tracer. -/
theorem oneField_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies ShiftedTernary.rows assignment) :
    OneFieldSound assignment := by
  have shape := commitmentShape_of_satisfies canonical one satisfies
  exact
    { opening := canonicalOpening_of_satisfies prime canonical one satisfies
      dimension := shape.1
      kappa := shape.2
      commitment := commitmentHolds_of_satisfies canonical one satisfies }

end Nightstream.Implementation.R1CS.ShiftedTernarySound
