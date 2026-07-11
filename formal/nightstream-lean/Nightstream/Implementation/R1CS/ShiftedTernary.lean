import Nightstream.Implementation.R1CS.ShiftedTernaryArtifact

/-!
Contract: semantic compiler for the canonical 41-trit SIS field encoding.

The Rust gadget represents `N = (x + M) mod p` in ordinary base three and
stores centered trits `dᵢ = tᵢ - 1`.  These definitions reconstruct the exact
generated range, reconstruction, and borrow rows.  The seeded commitment rows
remain a separate linear-map obligation.
-/

namespace Nightstream.Implementation.R1CS.ShiftedTernaryCompiler

open Nightstream.Implementation.R1CS

def digitCount : Nat := 41
def shift : Nat := (3 ^ digitCount - 1) / 2

def base3Digits : Nat → Nat → List Nat
  | _, 0 => []
  | value, count + 1 => value % 3 :: base3Digits (value / 3) count

def boundDigits : List Nat := base3Digits (goldilocksP - 1) digitCount

def negativeDefinitionRow (digit negative : Nat) : Row :=
  ⟨[(digit, 1)], [(digit, 1), (0, goldilocksP - 1)], [(negative, 2)]⟩

def negativeSupportRow (digit negative : Nat) : Row :=
  ⟨[(negative, 1)], [(digit, 1), (0, 1)], []⟩

def digitRows : List Row :=
  (ShiftedTernary.digitCols.zip ShiftedTernary.negativeCols).flatMap
    (fun pair => [negativeDefinitionRow pair.1 pair.2,
      negativeSupportRow pair.1 pair.2])

def reconstructionRow : Row :=
  ⟨(ShiftedTernary.fieldCol, 1) ::
      (List.range digitCount).map (fun index =>
        (ShiftedTernary.digitCols.getD index 0,
          goldilocksP - 3 ^ index)),
    [(0, 1)], []⟩

def currentBorrowTerms (index : Nat) : List (Nat × Nat) :=
  if index = 0 then []
  else [(ShiftedTernary.borrowCols.getD (index - 1) 0, 1)]

def nextBorrowTerms (index coefficient : Nat) : List (Nat × Nat) :=
  if index + 1 = digitCount then []
  else [(ShiftedTernary.borrowCols.getD index 0, coefficient)]

def borrowRow (index : Nat) : Row :=
  let digit := ShiftedTernary.digitCols.getD index 0
  let negative := ShiftedTernary.negativeCols.getD index 0
  match boundDigits.getD index 0 with
  | 0 =>
      ⟨[(negative, 1)],
        (currentBorrowTerms index).map
          (fun term => (term.1, goldilocksP - 1)) ++ [(0, 1)],
        nextBorrowTerms index (goldilocksP - 1) ++ [(0, 1)]⟩
  | 1 =>
      ⟨[(digit, goldilocksP - 1),
         (negative, goldilocksP - 2), (0, 1)],
        currentBorrowTerms index,
        nextBorrowTerms index 1 ++
          [(digit, goldilocksP - 1),
           (negative, goldilocksP - 1)]⟩
  | _ =>
      ⟨[(digit, 1), (negative, 1)],
        currentBorrowTerms index,
        nextBorrowTerms index 1⟩

def borrowRows : List Row :=
  (List.range digitCount).map borrowRow

def canonicalRows : List Row :=
  digitRows ++ [reconstructionRow] ++ borrowRows

theorem boundDigits_value :
    boundDigits =
      [0, 1, 0, 0, 0, 1, 2, 1, 2, 1, 0, 0, 1, 2, 0, 1, 1, 0, 2, 0,
       1, 1, 0, 1, 0, 2, 1, 2, 2, 1, 2, 2, 0, 0, 2, 2, 2, 1, 1, 1, 1] := by
  native_decide

/-- Kernel-checked equality between the compact compiler and rows 2-125 of
the exact generated artifact. -/
theorem canonicalRows_eq_artifact :
    canonicalRows = (ShiftedTernary.rows.drop 2).take 124 := by
  decide

/-- Centered residue plus its exact negative indicator. -/
inductive Digit (value negative : Nat) : Prop where
  | neg : value = goldilocksP - 1 → negative = 1 → Digit value negative
  | zero : value = 0 → negative = 0 → Digit value negative
  | pos : value = 1 → negative = 0 → Digit value negative

private theorem goldilocks_succ_mod_zero {value : Nat}
    (valueLt : value < 18446744069414584321)
    (zero : (value + 1) % 18446744069414584321 = 0) :
    value = 18446744069414584320 := by
  have valueSuccLe : value + 1 ≤ 18446744069414584321 := by omega
  rcases Nat.lt_or_eq_of_le valueSuccLe with strict | equal
  · rw [Nat.mod_eq_of_lt strict] at zero
    omega
  · omega

private theorem goldilocks_add_pred_mod_zero {value : Nat}
    (valueLt : value < 18446744069414584321)
    (zero :
      (value + (18446744069414584321 - 1)) %
        18446744069414584321 = 0) :
    value = 1 := by
  cases value with
  | zero =>
      simp only [Nat.zero_add] at zero
      rw [Nat.mod_eq_of_lt (by omega)] at zero
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

private theorem goldilocks_double_mod_two {value : Nat}
    (valueLt : value < 18446744069414584321)
    (two : 2 * value % 18446744069414584321 = 2) :
    value = 1 := by
  by_cases below : 2 * value < 18446744069414584321
  · rw [Nat.mod_eq_of_lt below] at two
    omega
  · have modulusLe : 18446744069414584321 ≤ 2 * value :=
      Nat.le_of_not_gt below
    rw [Nat.mod_eq_sub_mod modulusLe] at two
    have reducedLt :
        2 * value - 18446744069414584321 < 18446744069414584321 := by
      apply (Nat.sub_lt_iff_lt_add' modulusLe).mpr
      simpa [Nat.two_mul] using
        (Nat.mul_lt_mul_left (a := 2) (b := value)
          (c := 18446744069414584321) (by decide)).mpr valueLt
    rw [Nat.mod_eq_of_lt reducedLt] at two
    have impossible :
        2 * value = 18446744069414584321 + 2 :=
      (Nat.sub_eq_iff_eq_add' modulusLe).mp two
    have parity := congrArg (fun number => number % 2) impossible
    simp at parity

private theorem goldilocks_negative_product :
    (18446744069414584321 - 1) *
          (((18446744069414584321 - 1) +
            (18446744069414584321 - 1)) % 18446744069414584321) %
        18446744069414584321 = 2 := by
  native_decide

theorem digitRows_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) {digit negative : Nat}
    (definitionHolds : RowHolds assignment
      (negativeDefinitionRow digit negative))
    (supportHolds : RowHolds assignment
      (negativeSupportRow digit negative)) :
    Digit (assignment digit) (assignment negative) := by
  have definition := definitionHolds
  have support := supportHolds
  simp only [RowHolds, negativeDefinitionRow, negativeSupportRow, lcEval,
    List.foldl, one, goldilocksP] at definition support
  have digitLt := canonical digit
  have negativeLt := canonical negative
  simp only [goldilocksP] at digitLt negativeLt
  simp only [Nat.zero_add, Nat.one_mul, Nat.mul_one,
    Nat.mod_eq_of_lt digitLt, Nat.mod_eq_of_lt negativeLt,
    Nat.zero_mod] at definition support
  rcases prime _ _ support with negativeZero | plusOneZero
  · have negativeEq : assignment negative = 0 := by
      simp only [goldilocksP, Nat.mod_eq_of_lt negativeLt] at negativeZero
      exact negativeZero
    rw [negativeEq] at definition
    simp only [Nat.mul_zero, Nat.zero_mod] at definition
    rcases prime _ _ definition with digitZero | minusOneZero
    · apply Digit.zero
      · simp only [goldilocksP, Nat.mod_eq_of_lt digitLt] at digitZero
        exact digitZero
      · exact negativeEq
    · apply Digit.pos
      · simp only [goldilocksP, Nat.mod_mod] at minusOneZero
        exact goldilocks_add_pred_mod_zero digitLt minusOneZero
      · exact negativeEq
  · have digitEq : assignment digit = goldilocksP - 1 := by
      simp only [goldilocksP, Nat.mod_mod] at plusOneZero
      exact goldilocks_succ_mod_zero digitLt plusOneZero
    apply Digit.neg digitEq
    rw [digitEq] at definition
    simp only [goldilocksP, goldilocks_negative_product] at definition
    exact goldilocks_double_mod_two negativeLt definition.symm

def DigitsHold (assignment : Nat → Nat) : Prop :=
  ∀ pair ∈ ShiftedTernary.digitCols.zip ShiftedTernary.negativeCols,
    Digit (assignment pair.1) (assignment pair.2)

theorem canonicalRows_satisfy
    {assignment : Nat → Nat}
    (satisfies : Satisfies ShiftedTernary.rows assignment) :
    Satisfies canonicalRows assignment := by
  rw [canonicalRows_eq_artifact]
  intro row member
  exact satisfies row
    (List.mem_of_mem_drop (List.mem_of_mem_take member))

theorem allDigits_sound_of_canonicalRows
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies canonicalRows assignment) :
    DigitsHold assignment := by
  intro pair member
  apply digitRows_sound prime canonical one
  · apply satisfies
    unfold canonicalRows digitRows
    apply List.mem_append_left
    apply List.mem_append_left
    apply List.mem_flatMap.mpr
    exact ⟨pair, member, by simp⟩
  · apply satisfies
    unfold canonicalRows digitRows
    apply List.mem_append_left
    apply List.mem_append_left
    apply List.mem_flatMap.mpr
    exact ⟨pair, member, by simp⟩

theorem allDigits_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies ShiftedTernary.rows assignment) :
    DigitsHold assignment :=
  allDigits_sound_of_canonicalRows prime canonical one
    (canonicalRows_satisfy satisfies)

theorem DigitsHold.atIndex
    {assignment : Nat → Nat} (holds : DigitsHold assignment)
    {index : Nat} (indexLt : index < digitCount) :
    Digit
      (assignment (ShiftedTernary.digitCols.getD index 0))
      (assignment (ShiftedTernary.negativeCols.getD index 0)) := by
  have digitLt : index < ShiftedTernary.digitCols.length := by
    simpa [digitCount, ShiftedTernary.digitCols] using indexLt
  have negativeLt : index < ShiftedTernary.negativeCols.length := by
    simpa [digitCount, ShiftedTernary.negativeCols] using indexLt
  have zipLt : index <
      (ShiftedTernary.digitCols.zip ShiftedTernary.negativeCols).length := by
    simp only [List.length_zip]
    omega
  have member := List.getElem_mem (l :=
    ShiftedTernary.digitCols.zip ShiftedTernary.negativeCols) zipLt
  have pairEq :
      (ShiftedTernary.digitCols.zip
        ShiftedTernary.negativeCols)[index] =
      (ShiftedTernary.digitCols.getD index 0,
        ShiftedTernary.negativeCols.getD index 0) := by
    rw [List.getElem_zip]
    simp only [List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem digitLt,
      List.getElem?_eq_getElem negativeLt]
    simp
  rw [pairEq] at member
  exact holds _ member

def Digit.trit {value negative : Nat} (_ : Digit value negative) : Nat :=
  if value = goldilocksP - 1 then 0 else value + 1

def tritValue (value : Nat) : Nat :=
  if value = goldilocksP - 1 then 0 else value + 1

def encodedValue (assignment : Nat → Nat) : Nat :=
  (List.range digitCount).foldl (fun total index =>
    total + 3 ^ index * tritValue
      (assignment (ShiftedTernary.digitCols.getD index 0))) 0

structure CanonicalOpening (assignment : Nat → Nat) : Prop where
  digits : ∀ index, index < digitCount →
    Digit
      (assignment (ShiftedTernary.digitCols.getD index 0))
      (assignment (ShiftedTernary.negativeCols.getD index 0))
  encodedLt : encodedValue assignment < goldilocksP
  fieldMatches :
    (assignment ShiftedTernary.fieldCol + shift) % goldilocksP =
      encodedValue assignment

/-- Executable centered-trit predicate used by independent compiler checks. -/
def digitCheck (value negative : Nat) : Bool :=
  decide
    ((value = goldilocksP - 1 ∧ negative = 1) ∨
      (value = 0 ∧ negative = 0) ∨
      (value = 1 ∧ negative = 0))

theorem digitCheck_eq_true_iff (value negative : Nat) :
    digitCheck value negative = true ↔ Digit value negative := by
  simp only [digitCheck, decide_eq_true_eq]
  constructor
  · rintro (negativeDigit | zeroDigit | positiveDigit)
    · exact .neg negativeDigit.1 negativeDigit.2
    · exact .zero zeroDigit.1 zeroDigit.2
    · exact .pos positiveDigit.1 positiveDigit.2
  · intro digit
    cases digit with
    | neg valueEq negativeEq => exact Or.inl ⟨valueEq, negativeEq⟩
    | zero valueEq negativeEq => exact Or.inr (Or.inl ⟨valueEq, negativeEq⟩)
    | pos valueEq negativeEq => exact Or.inr (Or.inr ⟨valueEq, negativeEq⟩)

/-- Executable semantic check for one canonical shifted-ternary opening. -/
def canonicalOpeningCheck (assignment : Nat → Nat) : Bool :=
  ((List.range digitCount).all fun index =>
      digitCheck
        (assignment (ShiftedTernary.digitCols.getD index 0))
        (assignment (ShiftedTernary.negativeCols.getD index 0))) &&
    (decide (encodedValue assignment < goldilocksP) &&
      decide
        ((assignment ShiftedTernary.fieldCol + shift) % goldilocksP =
          encodedValue assignment))

theorem canonicalOpeningCheck_eq_true_iff (assignment : Nat → Nat) :
    canonicalOpeningCheck assignment = true ↔
      CanonicalOpening assignment := by
  simp only [canonicalOpeningCheck, Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨digitsChecked, encodedLt, fieldMatches⟩
    refine ⟨?_, encodedLt, fieldMatches⟩
    intro index indexLt
    exact (digitCheck_eq_true_iff _ _).mp
      ((List.all_eq_true.mp digitsChecked) index (List.mem_range.mpr indexLt))
  · intro opening
    refine ⟨?_, opening.encodedLt, opening.fieldMatches⟩
    apply List.all_eq_true.mpr
    intro index indexMember
    exact (digitCheck_eq_true_iff _ _).mpr
      (opening.digits index (List.mem_range.mp indexMember))

theorem canonicalOpening_unique
    {left right : Nat → Nat}
    (leftOpening : CanonicalOpening left)
    (rightOpening : CanonicalOpening right)
    (sameField : left ShiftedTernary.fieldCol =
      right ShiftedTernary.fieldCol) :
    encodedValue left = encodedValue right := by
  rw [← leftOpening.fieldMatches, ← rightOpening.fieldMatches, sameField]

end Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
