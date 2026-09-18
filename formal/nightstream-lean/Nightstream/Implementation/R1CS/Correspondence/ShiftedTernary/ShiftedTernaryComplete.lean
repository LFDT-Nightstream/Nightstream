import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.ShiftedTernarySound

/-!
Completeness of the production shifted-base-3 witness generator.

`CanonicalWitness` describes values written by the Rust generator, not R1CS
acceptance: digits are the radix-three digits of `(x + shift) mod p`, negative
indicators are derived from those digits, and borrow wires are the executable
comparison trace.  The final theorem compiles that independent witness
description to the exact 124 canonical rows.
-/

namespace Nightstream.Implementation.R1CS.ShiftedTernaryComplete

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound

set_option maxRecDepth 262144

def targetValue (assignment : Nat → Nat) : Nat :=
  (assignment ShiftedTernary.fieldCol + shift) % goldilocksP

def nativeTrit (assignment : Nat → Nat) (index : Nat) : Nat :=
  targetValue assignment / 3 ^ index % 3

def nativeDigit (assignment : Nat → Nat) (index : Nat) : Nat :=
  match nativeTrit assignment index with
  | 0 => goldilocksP - 1
  | 1 => 0
  | _ => 1

def nativeNegative (assignment : Nat → Nat) (index : Nat) : Nat :=
  if nativeTrit assignment index = 0 then 1 else 0

structure CanonicalWitness (assignment : Nat → Nat) : Prop where
  canonical : ∀ column, assignment column < goldilocksP
  one : assignment 0 = 1
  digitEq : ∀ index, index < digitCount →
    assignment (ShiftedTernary.digitCols.getD index 0) =
      nativeDigit assignment index
  negativeEq : ∀ index, index < digitCount →
    assignment (ShiftedTernary.negativeCols.getD index 0) =
      nativeNegative assignment index
  borrowEq : ∀ index, index < digitCount - 1 →
    assignment (ShiftedTernary.borrowCols.getD index 0) =
      expectedBorrow (nativeTrit assignment) boundDigit 0 (index + 1)

theorem nativeTrit_lt_three (assignment : Nat → Nat) (index : Nat) :
    nativeTrit assignment index < 3 := by
  unfold nativeTrit
  exact Nat.mod_lt _ (by decide)

theorem lowValue_nativeTrit (assignment : Nat → Nat) : ∀ count,
    lowValue (nativeTrit assignment) count =
      targetValue assignment % 3 ^ count := by
  intro count
  induction count with
  | zero => exact (Nat.mod_one _).symm
  | succ count ih =>
      rw [lowValue, ih, Nat.pow_succ, Nat.mod_mul]
      simp only [nativeTrit]
      rw [Nat.mul_comm]

private theorem target_lt_radix :
    goldilocksP < 3 ^ digitCount := by
  native_decide

theorem lowValue_nativeTrit_full (assignment : Nat → Nat) :
    lowValue (nativeTrit assignment) digitCount = targetValue assignment := by
  rw [lowValue_nativeTrit]
  apply Nat.mod_eq_of_lt
  exact Nat.lt_trans (Nat.mod_lt _ (by native_decide)) target_lt_radix

theorem nativeDigit_semantic (assignment : Nat → Nat)
    {index : Nat} (_indexLt : index < digitCount) :
    Digit (nativeDigit assignment index) (nativeNegative assignment index) := by
  have tritLt := nativeTrit_lt_three assignment index
  have cases : nativeTrit assignment index = 0 ∨
      nativeTrit assignment index = 1 ∨ nativeTrit assignment index = 2 := by
    omega
  rcases cases with zero | one | two
  · exact Digit.neg (by simp [nativeDigit, zero])
      (by simp [nativeNegative, zero])
  · exact Digit.zero (by simp [nativeDigit, one])
      (by simp [nativeNegative, one])
  · exact Digit.pos (by simp [nativeDigit, two])
      (by simp [nativeNegative, two])

theorem nativeDigit_tritValue (assignment : Nat → Nat)
    {index : Nat} (_indexLt : index < digitCount) :
    tritValue (nativeDigit assignment index) = nativeTrit assignment index := by
  have tritLt := nativeTrit_lt_three assignment index
  have cases : nativeTrit assignment index = 0 ∨
      nativeTrit assignment index = 1 ∨ nativeTrit assignment index = 2 := by
    omega
  rcases cases with zero | one | two
  · simp [nativeDigit, tritValue, zero, goldilocksP]
  · simp [nativeDigit, tritValue, one, goldilocksP]
  · simp [nativeDigit, tritValue, two, goldilocksP]

theorem lowValue_congr {left right : Nat → Nat} {count : Nat}
    (pointwise : ∀ index, index < count → left index = right index) :
    lowValue left count = lowValue right count := by
  induction count with
  | zero => simp [lowValue]
  | succ count ih =>
      rw [lowValue, lowValue, ih (fun index indexLt =>
        pointwise index (by omega)), pointwise count (by omega)]

theorem CanonicalWitness.digits {assignment : Nat → Nat}
    (witness : CanonicalWitness assignment) : DigitsHold assignment := by
  intro pair member
  have lengths : ShiftedTernary.digitCols.length = digitCount ∧
      ShiftedTernary.negativeCols.length = digitCount := by decide
  rcases List.mem_iff_getElem.mp member with ⟨index, indexLt, pairEq⟩
  have digitLt : index < digitCount := by
    simpa [List.length_zip, lengths.1, lengths.2] using indexLt
  have columns :
      (ShiftedTernary.digitCols.zip ShiftedTernary.negativeCols)[index] =
        (ShiftedTernary.digitCols.getD index 0,
          ShiftedTernary.negativeCols.getD index 0) := by
    rw [List.getElem_zip]
    simp only [List.getD_eq_getElem?_getD]
    have hd : index < ShiftedTernary.digitCols.length := by
      simpa [lengths.1] using digitLt
    have hn : index < ShiftedTernary.negativeCols.length := by
      simpa [lengths.2] using digitLt
    rw [List.getElem?_eq_getElem hd, List.getElem?_eq_getElem hn]
    simp
  rw [← pairEq]
  have semantic :
      Digit
        (assignment (ShiftedTernary.digitCols.getD index 0))
        (assignment (ShiftedTernary.negativeCols.getD index 0)) := by
    rw [witness.digitEq index digitLt, witness.negativeEq index digitLt]
    exact nativeDigit_semantic assignment digitLt
  exact columns.symm ▸ semantic

theorem CanonicalWitness.encodedValue_eq_target
    {assignment : Nat → Nat} (witness : CanonicalWitness assignment) :
    encodedValue assignment = targetValue assignment := by
  rw [encodedValue_eq_lowValue, ← lowValue_nativeTrit_full assignment]
  apply lowValue_congr
  intro index indexLt
  unfold assignmentTrit
  rw [witness.digitEq index indexLt]
  exact nativeDigit_tritValue assignment indexLt

theorem CanonicalWitness.opening
    {assignment : Nat → Nat} (witness : CanonicalWitness assignment) :
    CanonicalOpening assignment where
  digits := fun _ indexLt => witness.digits.atIndex indexLt
  encodedLt := by
    rw [witness.encodedValue_eq_target]
    exact Nat.mod_lt _ (by native_decide)
  fieldMatches := by
    rw [witness.encodedValue_eq_target]
    rfl

theorem digitDefinition_complete
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    {digitCol negativeCol : Nat}
    (digit : Digit (assignment digitCol) (assignment negativeCol)) :
    RowHolds assignment (negativeDefinitionRow digitCol negativeCol) := by
  cases digit with
  | neg valueEq negativeEq =>
      simp [RowHolds, negativeDefinitionRow, lcEval, one,
        valueEq, negativeEq, goldilocksP]
  | zero valueEq negativeEq =>
      simp [RowHolds, negativeDefinitionRow, lcEval, one,
        valueEq, negativeEq, goldilocksP]
  | pos valueEq negativeEq =>
      simp [RowHolds, negativeDefinitionRow, lcEval, one,
        valueEq, negativeEq, goldilocksP]

theorem digitSupport_complete
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    {digitCol negativeCol : Nat}
    (digit : Digit (assignment digitCol) (assignment negativeCol)) :
    RowHolds assignment (negativeSupportRow digitCol negativeCol) := by
  cases digit with
  | neg valueEq negativeEq =>
      simp [RowHolds, negativeSupportRow, lcEval, one,
        valueEq, negativeEq, goldilocksP]
  | zero valueEq negativeEq =>
      simp [RowHolds, negativeSupportRow, lcEval, one,
        valueEq, negativeEq, goldilocksP]
  | pos valueEq negativeEq =>
      simp [RowHolds, negativeSupportRow, lcEval, one,
        valueEq, negativeEq, goldilocksP]

theorem CanonicalWitness.digitRows_complete
    {assignment : Nat → Nat} (witness : CanonicalWitness assignment) :
    Satisfies digitRows assignment := by
  intro row member
  unfold digitRows at member
  rw [List.mem_flatMap] at member
  rcases member with ⟨pair, pairMember, rowMember⟩
  have digit := witness.digits pair pairMember
  simp only [List.mem_cons, List.not_mem_nil, or_false] at rowMember
  rcases rowMember with rfl | rfl
  · exact digitDefinition_complete witness.one digit
  · exact digitSupport_complete witness.one digit

private theorem shift_lt_modulus : shift < goldilocksP := by
  native_decide

private theorem add_shift_inverse_mod (value : Nat) :
    ((value + shift) + (goldilocksP - shift)) % goldilocksP =
      value % goldilocksP := by
  calc
    ((value + shift) + (goldilocksP - shift)) % goldilocksP =
        (value + (shift + (goldilocksP - shift))) % goldilocksP := by
          rw [Nat.add_assoc]
    _ = (value + goldilocksP) % goldilocksP := by
          rw [Nat.add_sub_of_le (Nat.le_of_lt shift_lt_modulus)]
    _ = value % goldilocksP := by simp

theorem CanonicalWitness.field_eq_centered_mod
    {assignment : Nat → Nat} (witness : CanonicalWitness assignment) :
    assignment ShiftedTernary.fieldCol % goldilocksP =
      lowValue (centeredDigit assignment) digitCount % goldilocksP := by
  have shifted :
      (assignment ShiftedTernary.fieldCol + shift) % goldilocksP =
        (lowValue (centeredDigit assignment) digitCount + shift) %
          goldilocksP := by
    rw [witness.opening.fieldMatches]
    rw [← encodedValue_centered_shift_mod_of_digits witness.digits]
    exact (Nat.mod_eq_of_lt witness.opening.encodedLt).symm
  have appended := congrArg
    (fun value => (value + (goldilocksP - shift)) % goldilocksP) shifted
  change
    (((assignment ShiftedTernary.fieldCol + shift) % goldilocksP +
        (goldilocksP - shift)) % goldilocksP) =
      (((lowValue (centeredDigit assignment) digitCount + shift) %
          goldilocksP + (goldilocksP - shift)) % goldilocksP) at appended
  rw [Nat.mod_add_mod, Nat.mod_add_mod,
    add_shift_inverse_mod, add_shift_inverse_mod] at appended
  exact appended

theorem CanonicalWitness.reconstruction_complete
    {assignment : Nat → Nat} (witness : CanonicalWitness assignment) :
    RowHolds assignment reconstructionRow := by
  have equation :
      (assignment ShiftedTernary.fieldCol +
        negativeWeightedValue assignment digitCount) % goldilocksP = 0 := by
    rw [Nat.add_mod, witness.field_eq_centered_mod]
    rw [← Nat.add_mod]
    simpa [Nat.add_comm] using
      negative_add_centered_mod_zero assignment digitCount Nat.le.refl
  simp only [RowHolds, reconstructionRow, lcEval, List.foldl, witness.one,
    Nat.one_mul, Nat.mul_one, Nat.zero_add, Nat.zero_mod]
  simp only [List.foldl_map]
  have folded := foldl_range_eq_negativeWeightedValue assignment
    (assignment ShiftedTernary.fieldCol) digitCount
  unfold centeredDigit at folded
  rw [folded]
  simpa [goldilocksP] using equation

theorem CanonicalWitness.finalBorrow_zero
    {assignment : Nat → Nat} (_witness : CanonicalWitness assignment) :
    expectedBorrow (nativeTrit assignment) boundDigit 0 digitCount = 0 := by
  apply expectedBorrow_zero_of_lowValue_le
    (fun index _ => nativeTrit_lt_three assignment index)
    (fun _ indexLt => boundDigit_lt_three indexLt)
  rw [lowValue_nativeTrit_full]
  change targetValue assignment ≤ lowValue boundDigit digitCount
  rw [boundLowValue]
  have targetLt : targetValue assignment < goldilocksP :=
    Nat.mod_lt _ (by native_decide)
  omega

theorem CanonicalWitness.borrowAt_eq_expected
    {assignment : Nat → Nat} (witness : CanonicalWitness assignment) :
    ∀ count, count ≤ digitCount →
      borrowAt assignment count =
        expectedBorrow (nativeTrit assignment) boundDigit 0 count := by
  intro count countLe
  by_cases zero : count = 0
  · subst count
    simp [borrowAt, expectedBorrow]
  by_cases terminal : count = digitCount
  · subst count
    simp only [borrowAt, ↓reduceIte]
    exact witness.finalBorrow_zero.symm
  have indexLt : count - 1 < digitCount - 1 := by omega
  simp only [borrowAt, zero, terminal, ↓reduceIte]
  rw [witness.borrowEq (count - 1) indexLt]
  congr 2
  omega

theorem CanonicalWitness.borrowStep
    {assignment : Nat → Nat} (witness : CanonicalWitness assignment)
    {index : Nat} (indexLt : index < digitCount) :
    borrowAt assignment (index + 1) =
      if tritValue
            (assignment (ShiftedTernary.digitCols.getD index 0)) +
          borrowAt assignment index > boundDigit index
      then 1 else 0 := by
  rw [witness.borrowAt_eq_expected index (by omega),
    witness.borrowAt_eq_expected (index + 1) (by omega),
    expectedBorrow]
  rw [witness.digitEq index indexLt,
    nativeDigit_tritValue assignment indexLt]

private theorem borrowEquation_zero
    {digit negative current next : Nat}
    (semantic : Digit digit negative) (currentLe : current ≤ 1)
    (nextEq : next = if tritValue digit + current > 0 then 1 else 0) :
    negative * fieldOneMinus current % goldilocksP =
      fieldOneMinus next := by
  have currentCases : current = 0 ∨ current = 1 := by omega
  cases semantic with
  | neg digitEq negativeEq =>
      rcases currentCases with currentEq | currentEq <;>
        subst digit <;> subst negative <;> subst current <;>
        simp [tritValue, fieldOneMinus, goldilocksP] at nextEq ⊢ <;>
        omega
  | zero digitEq negativeEq =>
      rcases currentCases with currentEq | currentEq <;>
        subst digit <;> subst negative <;> subst current <;>
        simp [tritValue, fieldOneMinus, goldilocksP] at nextEq ⊢ <;>
        omega
  | pos digitEq negativeEq =>
      rcases currentCases with currentEq | currentEq <;>
        subst digit <;> subst negative <;> subst current <;>
        simp [tritValue, fieldOneMinus, goldilocksP] at nextEq ⊢ <;>
        omega

private theorem borrowEquation_one
    {digit negative current next : Nat}
    (semantic : Digit digit negative) (currentLe : current ≤ 1)
    (nextEq : next = if tritValue digit + current > 1 then 1 else 0) :
    fieldZeroIndicator digit negative * current % goldilocksP =
      fieldNextMinusPositive next digit negative := by
  have currentCases : current = 0 ∨ current = 1 := by omega
  cases semantic with
  | neg digitEq negativeEq =>
      rcases currentCases with currentEq | currentEq <;>
        subst digit <;> subst negative <;> subst current <;>
        simp [tritValue, fieldZeroIndicator, fieldNextMinusPositive,
          goldilocksP] at nextEq ⊢ <;> omega
  | zero digitEq negativeEq =>
      rcases currentCases with currentEq | currentEq <;>
        subst digit <;> subst negative <;> subst current <;>
        simp [tritValue, fieldZeroIndicator, fieldNextMinusPositive,
          goldilocksP] at nextEq ⊢ <;> omega
  | pos digitEq negativeEq =>
      rcases currentCases with currentEq | currentEq <;>
        subst digit <;> subst negative <;> subst current <;>
        simp [tritValue, fieldZeroIndicator, fieldNextMinusPositive,
          goldilocksP] at nextEq ⊢ <;> omega

private theorem borrowEquation_two
    {digit negative current next : Nat}
    (semantic : Digit digit negative) (currentLe : current ≤ 1)
    (nextEq : next = if tritValue digit + current > 2 then 1 else 0) :
    fieldPositive digit negative * current % goldilocksP = next := by
  have currentCases : current = 0 ∨ current = 1 := by omega
  cases semantic with
  | neg digitEq negativeEq =>
      rcases currentCases with currentEq | currentEq <;>
        subst digit <;> subst negative <;> subst current <;>
        simp [tritValue, fieldPositive, goldilocksP] at nextEq ⊢ <;> omega
  | zero digitEq negativeEq =>
      rcases currentCases with currentEq | currentEq <;>
        subst digit <;> subst negative <;> subst current <;>
        simp [tritValue, fieldPositive, goldilocksP] at nextEq ⊢ <;> omega
  | pos digitEq negativeEq =>
      rcases currentCases with currentEq | currentEq <;>
        subst digit <;> subst negative <;> subst current <;>
        simp [tritValue, fieldPositive, goldilocksP] at nextEq ⊢ <;> omega

theorem CanonicalWitness.borrowRow_complete
    {assignment : Nat → Nat} (witness : CanonicalWitness assignment)
    {index : Nat} (indexLt : index < digitCount) :
    RowHolds assignment (borrowRow index) := by
  have digit := witness.digits.atIndex indexLt
  have currentLe : borrowAt assignment index ≤ 1 := by
    rw [witness.borrowAt_eq_expected index (by omega)]
    exact expectedBorrow_le_one _ _ 0 (by omega) index
  have step := witness.borrowStep indexLt
  have boundLt := boundDigit_lt_three indexLt
  have boundCases : boundDigits.getD index 0 = 0 ∨
      boundDigits.getD index 0 = 1 ∨
        boundDigits.getD index 0 = 2 := by omega
  rcases boundCases with rawBound | rawBound | rawBound
  · have stepZero := step
    change borrowAt assignment (index + 1) =
      (if tritValue
          (assignment (ShiftedTernary.digitCols.getD index 0)) +
        borrowAt assignment index > boundDigits.getD index 0
      then 1 else 0) at stepZero
    rw [rawBound] at stepZero
    have equation := borrowEquation_zero digit currentLe stepZero
    simp only [borrowRow, rawBound, RowHolds]
    rw [lcEval_append, lcEval_append,
      negCurrentBorrowTerms_eval indexLt,
      negNextBorrowTerms_eval]
    have negativeLt := witness.canonical
      (ShiftedTernary.negativeCols.getD index 0)
    simpa [fieldOneMinus, lcEval, witness.one,
      Nat.mod_eq_of_lt negativeLt, Nat.add_mod, goldilocksP] using equation
  · have stepOne := step
    change borrowAt assignment (index + 1) =
      (if tritValue
          (assignment (ShiftedTernary.digitCols.getD index 0)) +
        borrowAt assignment index > boundDigits.getD index 0
      then 1 else 0) at stepOne
    rw [rawBound] at stepOne
    have equation := borrowEquation_one digit currentLe stepOne
    simp only [borrowRow, rawBound, RowHolds]
    rw [currentBorrowTerms_eval witness.canonical indexLt,
      lcEval_append, nextBorrowTerms_eval witness.canonical]
    simpa [fieldZeroIndicator, fieldNextMinusPositive, lcEval, witness.one,
      Nat.add_mod, Nat.add_assoc, goldilocksP] using equation
  · have stepTwo := step
    change borrowAt assignment (index + 1) =
      (if tritValue
          (assignment (ShiftedTernary.digitCols.getD index 0)) +
        borrowAt assignment index > boundDigits.getD index 0
      then 1 else 0) at stepTwo
    rw [rawBound] at stepTwo
    have equation := borrowEquation_two digit currentLe stepTwo
    simp only [borrowRow, rawBound, RowHolds]
    rw [currentBorrowTerms_eval witness.canonical indexLt,
      nextBorrowTerms_eval witness.canonical]
    simpa [fieldPositive, lcEval, Nat.add_mod, goldilocksP] using equation

theorem CanonicalWitness.borrowRows_complete
    {assignment : Nat → Nat} (witness : CanonicalWitness assignment) :
    Satisfies borrowRows assignment := by
  intro row member
  unfold borrowRows at member
  rw [List.mem_map] at member
  rcases member with ⟨index, indexMember, rfl⟩
  exact witness.borrowRow_complete (List.mem_range.mp indexMember)

/-- Completeness against the exact 124-row compiler: the independent native
witness description constructs a satisfying assignment without assuming any
row equation or verifier acceptance bit. -/
theorem canonicalRows_complete
    {assignment : Nat → Nat} (witness : CanonicalWitness assignment) :
    Satisfies canonicalRows assignment := by
  intro row member
  unfold canonicalRows at member
  rw [List.mem_append] at member
  rcases member with prefixMember | borrowMember
  · rw [List.mem_append] at prefixMember
    rcases prefixMember with digitMember | reconstructionMember
    · exact witness.digitRows_complete row digitMember
    · simp only [List.mem_singleton] at reconstructionMember
      subst row
      exact witness.reconstruction_complete
  · exact witness.borrowRows_complete row borrowMember

end Nightstream.Implementation.R1CS.ShiftedTernaryComplete
