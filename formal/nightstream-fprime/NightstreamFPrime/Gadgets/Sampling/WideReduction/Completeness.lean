import NightstreamFPrime.Gadgets.Sampling.WideReduction.Footprint
import NightstreamFPrime.Gadgets.Sampling.WideReduction.Values
import NightstreamFPrime.Gadgets.Sampling.WideReduction.Soundness

/-!
Owns the completeness of the whole-vector sampler constraints (V4). From any
environment whose four inputs lie below the offset, it completes the four
canonical-u64 children and assigns the new bits: the binary digits of
`Q = X div 5^54`, the base-five digits of `R = X mod 5^54` in binary, and the
check quotients `K_r = (L_r - R_r) / m_r`, which lie in `[0, 549)`. Every row
then holds and no variable outside the gadget changes.

Completeness is existential; the exported witness program is not owned here.
-/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet ProductionStrongSet
open Finset

/-! ### Digit expansions -/

theorem digit_sum (base value count : Nat) :
    ∑ index ∈ Finset.range count, base ^ index * (value / base ^ index % base) =
      value % base ^ count := by
  induction count with
  | zero => simp [Nat.mod_one]
  | succ count inductionHypothesis =>
      rw [Finset.sum_range_succ, inductionHypothesis, pow_succ, Nat.mod_mul]

/-! ### Honest assignment of the new bits -/

/-- A field bit. -/
def honestBit (value : Nat) : F := fieldOfNat (value % 2)

theorem honestBit_val (value : Nat) : (honestBit value).val = value % 2 := by
  have small : value % 2 < goldilocksModulus :=
    lt_of_lt_of_le (Nat.mod_lt _ (by decide)) (by decide)
  simp [honestBit, fieldOfNat, Nat.mod_eq_of_lt small]

/-- Honest new bit number `k`: quotient bits, digit bits, then check bits. -/
def honestNew (draw : Nat) (check : Nat → Nat) (position : Nat) : F :=
  if position < quotientBitCount then
    honestBit (draw / scalarCount / 2 ^ position)
  else if position < checkBias then
    honestBit (draw % scalarCount / 5 ^ ((position - quotientBitCount) / digitBitCount) % 5 /
      2 ^ ((position - quotientBitCount) % digitBitCount))
  else
    honestBit (check ((position - checkBias) / checkBitCount) /
      2 ^ ((position - checkBias) % checkBitCount))

/-- Overwrite the new-bit region of `base` with honest values. -/
def completedEnv (base : Env) (offset draw : Nat) (check : Nat → Nat) : Env :=
  fun index =>
    if quotientStart offset ≤ index ∧ index < quotientStart offset + newBitCount then
      honestNew draw check (index - quotientStart offset)
    else base index

section Assignment

variable {base : Env} {offset draw : Nat} {check : Nat → Nat}

theorem completedEnv_below (index : Nat) (below : index < quotientStart offset) :
    completedEnv base offset draw check index = base index := by
  unfold completedEnv
  rw [if_neg (by omega)]

theorem quotientBit_eval (bit : Nat) (below : bit < quotientBitCount) :
    ((quotientBit offset bit).eval (completedEnv base offset draw check)).val =
      draw / scalarCount / 2 ^ bit % 2 := by
  change (completedEnv base offset draw check (quotientStart offset + bit)).val = _
  unfold completedEnv honestNew
  rw [if_pos (by simp [newBitCount] at *; omega), Nat.add_sub_cancel_left, if_pos below,
    honestBit_val]

theorem digitBit_eval (digit bit : Nat) (digitBelow : digit < digitCount)
    (bitBelow : bit < digitBitCount) :
    ((digitBit offset digit bit).eval (completedEnv base offset draw check)).val =
      draw % scalarCount / 5 ^ digit % 5 / 2 ^ bit % 2 := by
  change (completedEnv base offset draw check
    (digitStart offset + digitBitCount * digit + bit)).val = _
  unfold completedEnv honestNew
  simp only [digitStart, digitCount, digitBitCount, quotientBitCount, checkBias,
    newBitCount, checkBitCount] at *
  rw [if_pos (by omega), if_neg (by omega), if_pos (by omega), honestBit_val]
  have quotientDigit :
      (quotientStart offset + 131 + 3 * digit + bit - quotientStart offset - 131) / 3 = digit := by
    omega
  have remainderBit :
      (quotientStart offset + 131 + 3 * digit + bit - quotientStart offset - 131) % 3 = bit := by
    omega
  rw [quotientDigit, remainderBit]

theorem checkBit_eval (index : Nat) (bit : Nat) (bitBelow : bit < checkBitCount)
    (indexBelow : index < checkCount) :
    ((checkBit offset index bit).eval (completedEnv base offset draw check)).val =
      check index / 2 ^ bit % 2 := by
  change (completedEnv base offset draw check
    (checkStart offset + checkBitCount * index + bit)).val = _
  unfold completedEnv honestNew
  simp only [checkStart, digitStart, digitCount, digitBitCount, quotientBitCount, checkBias,
    newBitCount, checkBitCount, checkCount] at *
  rw [if_pos (by omega), if_neg (by omega), if_neg (by omega), honestBit_val]
  have quotientIndex :
      (quotientStart offset + 131 + 54 * 3 + 10 * index + bit - quotientStart offset -
        (131 + 54 * 3)) / 10 = index := by
    omega
  have remainderBit :
      (quotientStart offset + 131 + 54 * 3 + 10 * index + bit - quotientStart offset -
        (131 + 54 * 3)) % 10 = bit := by
    omega
  rw [quotientIndex, remainderBit]

/-- Every new variable holds a bit. -/
theorem newVariable_bit (position : Nat) :
    (completedEnv base offset draw check (quotientStart offset + position)) *
        ((completedEnv base offset draw check (quotientStart offset + position)) - 1) = 0 ∨
      ¬ position < newBitCount := by
  by_cases below : position < newBitCount
  · left
    unfold completedEnv
    rw [if_pos (by omega), Nat.add_sub_cancel_left]
    have bitValue : ∀ value : Nat, honestBit value * (honestBit value - 1) = 0 := by
      intro value
      rcases Nat.mod_two_eq_zero_or_one value with zero | one
      · have : honestBit value = 0 := by
          apply Fin.ext; rw [honestBit_val, zero]; rfl
        rw [this, zero_mul]
      · have : honestBit value = 1 := by
          apply Fin.ext; rw [honestBit_val, one]; rfl
        rw [this, sub_self, mul_zero]
    unfold honestNew
    split_ifs <;> exact bitValue _
  · exact Or.inr below

end Assignment

/-! ### Linear-form values under the completed environment -/

theorem linearValue_congr (left right : Env) (terms : List (Nat × Expr))
    (same : ∀ term ∈ terms, term.2.eval left = term.2.eval right) :
    linearValue left terms = linearValue right terms := by
  induction terms with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      simp only [linearValue]
      rw [same term (by simp), inductionHypothesis (fun t member => same t (by simp [member]))]

private theorem binary_sum (value count : Nat) (small : value < 2 ^ count) :
    ∑ index ∈ Finset.range count, 2 ^ index * (value / 2 ^ index % 2) = value := by
  rw [digit_sum, Nat.mod_eq_of_lt small]

private theorem digit_bits (digit : Nat) (small : digit < 5) :
    digit % 2 + 2 * (digit / 2 % 2) + 4 * (digit / 4 % 2) = digit := by
  interval_cases digit <;> rfl

/-- `Y = Q N + R` for the honest bits. -/
theorem resultTerms_honest (base : Env) (offset draw : Nat) (check : Nat → Nat)
    (drawBound : draw < drawCount) :
    linearValue (completedEnv base offset draw check) (resultTerms offset) = draw := by
  rw [resultTerms_value]
  have quotientSmall : draw / scalarCount < 2 ^ quotientBitCount := by
    have : (drawCount - 1) / scalarCount < 2 ^ quotientBitCount := by decide
    calc draw / scalarCount ≤ (drawCount - 1) / scalarCount := Nat.div_le_div_right (by omega)
      _ < 2 ^ quotientBitCount := this
  have quotient : quotientValue (completedEnv base offset draw check) offset = draw / scalarCount := by
    unfold quotientValue
    rw [← binary_sum (draw / scalarCount) quotientBitCount quotientSmall]
    refine Finset.sum_congr rfl fun bit member => ?_
    rw [quotientBit_eval bit (Finset.mem_range.mp member), Nat.div_div_eq_div_mul,
      ← Nat.div_div_eq_div_mul]
  have digitValue_eq : ∀ digit, digit < digitCount →
      digitValue (completedEnv base offset draw check) offset digit =
        draw % scalarCount / 5 ^ digit % 5 := by
    intro digit below
    unfold digitValue
    rw [digitBit_eval digit 0 below (by decide), digitBit_eval digit 1 below (by decide),
      digitBit_eval digit 2 below (by decide)]
    simpa using digit_bits (draw % scalarCount / 5 ^ digit % 5) (Nat.mod_lt _ (by decide))
  have remainderEq : remainderValue (completedEnv base offset draw check) offset =
      draw % scalarCount := by
    unfold remainderValue
    have expansion := digit_sum 5 (draw % scalarCount) digitCount
    rw [Nat.mod_eq_of_lt (Nat.mod_lt _ scalarCount_pos |>.trans_le (by rfl))] at expansion
    rw [← expansion]
    refine Finset.sum_congr rfl fun digit member => ?_
    rw [digitValue_eq digit (Finset.mem_range.mp member)]
  rw [quotient, remainderEq, Nat.div_add_mod]

theorem resultTerms_bits_honest (base : Env) (offset draw : Nat) (check : Nat → Nat) :
    ∀ term ∈ resultTerms offset, (term.2.eval (completedEnv base offset draw check)).val ≤ 1 := by
  intro term member
  simp only [resultTerms, quotientTerms, digitTerms, List.mem_append, List.mem_map,
    List.mem_range, List.mem_flatMap] at member
  rcases member with ⟨bit, below, rfl⟩ | ⟨digit, digitBelow, bit, bitBelow, rfl⟩
  · rw [quotientBit_eval bit below]; exact Nat.le_of_lt_succ (Nat.mod_lt _ (by decide))
  · rw [digitBit_eval digit bit digitBelow bitBelow]
    exact Nat.le_of_lt_succ (Nat.mod_lt _ (by decide))

/-! ### Children -/

/-- The operation of child `index`. -/
def childOpAt (interface : Interface) (offset : Nat) (index : Fin fieldCount) : Op :=
  Sequence.childOp (childName index) (CanonicalU64.circuit (childInterface interface offset index))
    (childOffset offset index)

private theorem appendChild (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset) (current : Sequence.Prefix env offset)
    (index : Fin fieldCount) (atEnd : localLength current.operations = childWidth * index.val) :
    ∃ next : Sequence.Prefix env offset,
      next.operations = current.operations ++ [childOpAt interface offset index] ∧
      localLength next.operations = childWidth * (index.val + 1) := by
  let child := CanonicalU64.circuit (childInterface interface offset index)
  have start : offset + localLength current.operations = childOffset offset index := by
    rw [atEnd]; rfl
  have sourceBelow : ((childInterface interface offset index).source
      (offset + localLength current.operations)).VarsBelow
        (offset + localLength current.operations) :=
    Expr.VarsBelow.mono _ (assumptions index) (by omega)
  obtain ⟨after, agrees, rows⟩ := CanonicalU64.complete (childInterface interface offset index)
    current.current (offset + localLength current.operations) sourceBelow
  have childLength : localLength (Circuit.ops child.main (offset + localLength current.operations)) =
      childWidth := CanonicalU64.localLength_eq _ _
  obtain ⟨next, operations, _, _⟩ := Sequence.appendBuilt current child
    (childOpAt interface offset index)
    (by
      change CanonicalU64.auxiliaryCount = _
      rw [childLength]; rfl)
    (by
      change flatConstraints (CanonicalU64.operations (childInterface interface offset index)
        (childOffset offset index)) = _
      rw [← start]; rfl)
    (by
      intro expression member
      have scope := CanonicalU64.flatConstraints_varsBelow (childInterface interface offset index)
        (offset + localLength current.operations) sourceBelow expression member
      rw [childLength]
      exact scope)
    after agrees rows
  refine ⟨next, operations, ?_⟩
  rw [operations, Sequence.localLength_append, Sequence.localLength_singleton, atEnd]
  change childWidth * index.val + CanonicalU64.auxiliaryCount = _
  simp only [childWidth]
  ring

private theorem completeChildren (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset) :
    ∃ done : Sequence.Prefix env offset,
      done.operations = childOps interface offset ∧
      localLength done.operations = childWidth * fieldCount := by
  obtain ⟨first, firstOps, firstLength⟩ := appendChild interface offset assumptions
    (Sequence.empty env offset) 0 rfl
  obtain ⟨second, secondOps, secondLength⟩ := appendChild interface offset assumptions first 1
    firstLength
  obtain ⟨third, thirdOps, thirdLength⟩ := appendChild interface offset assumptions second 2
    secondLength
  obtain ⟨fourth, fourthOps, fourthLength⟩ := appendChild interface offset assumptions third 3
    thirdLength
  refine ⟨fourth, ?_, fourthLength⟩
  rw [fourthOps, thirdOps, secondOps, firstOps]
  rfl

/-! ### Completeness -/

/-- Integer check quotients after the quotient and digit bits have been assigned. -/
def checkQuotients (base : Env) (offset draw : Nat) : Nat → Nat := fun index =>
  if below : index < checkCount then
    let check : Fin checkCount := ⟨index, below⟩
    let left := linearValue base (reduceTerms (modulus check) (drawTerms offset)) +
      modulus check * checkBias
    let right := linearValue (completedEnv base offset draw (fun _ => 0))
      (reduceTerms (modulus check) (resultTerms offset))
    (left - right) / modulus check
  else 0

/-- The deterministic honest assignment of the 353 checked result bits. -/
def completeNew (interface : Interface) (base : Env) (offset : Nat) : Env :=
  let draw := (drawIndex (drawOf interface base offset)).val
  completedEnv base offset draw (checkQuotients base offset draw)

theorem completeNew_agreesOutside (interface : Interface) (base : Env) (offset : Nat) :
    AgreesOutside base (completeNew interface base offset) (quotientStart offset) newBitCount := by
  intro index outside
  unfold completeNew completedEnv
  rw [if_neg (by omega)]

/-- Once the canonical children are complete, the explicit result assignment
satisfies the unchanged gadget rows. -/
theorem completeNew_certificate (interface : Interface) (hints : Nat → List Hint) (base : Env)
    (offset : Nat)
    (children : ∀ index, CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) base)
    (childRows : holdsFlat base (childOps interface offset))
    (childScope : ∀ expression ∈ flatConstraints (childOps interface offset),
      expression.VarsBelow (quotientStart offset)) :
    holdsFlat (completeNew interface base offset) (operations interface hints offset) ∧
      ∀ check : Fin checkCount,
        checkQuotients base offset (drawIndex (drawOf interface base offset)).val check.val < 549 := by
  let draw := (drawIndex (drawOf interface base offset)).val
  have drawBound : draw < drawCount := (drawIndex (drawOf interface base offset)).isLt
  have drawValue : linearValue base (drawTerms offset) = draw := drawTerms_value children
  let zeroEnv := completedEnv base offset draw (fun _ => 0)
  let left : Fin checkCount → Nat := fun index =>
    linearValue base (reduceTerms (modulus index) (drawTerms offset)) + modulus index * checkBias
  let right : Fin checkCount → Nat := fun index =>
    linearValue zeroEnv (reduceTerms (modulus index) (resultTerms offset))
  let quotients := checkQuotients base offset draw
  let completed := completedEnv base offset draw quotients
  -- Result atoms do not depend on the check quotients.
  have resultAtoms : ∀ term ∈ resultTerms offset, term.2.eval completed = term.2.eval zeroEnv := by
    intro term member
    simp only [resultTerms, quotientTerms, digitTerms, List.mem_append, List.mem_map,
      List.mem_range, List.mem_flatMap] at member
    apply Fin.ext
    rcases member with ⟨bit, below, rfl⟩ | ⟨digit, digitBelow, bit, bitBelow, rfl⟩
    · rw [quotientBit_eval bit below, quotientBit_eval bit below]
    · rw [digitBit_eval digit bit digitBelow bitBelow, digitBit_eval digit bit digitBelow bitBelow]
  have resultSame : ∀ modulusValue,
      linearValue completed (reduceTerms modulusValue (resultTerms offset)) =
        linearValue zeroEnv (reduceTerms modulusValue (resultTerms offset)) := by
    intro modulusValue
    apply linearValue_congr
    intro term member
    simp only [reduceTerms, List.mem_map] at member
    obtain ⟨original, originalMember, rfl⟩ := member
    exact resultAtoms original originalMember
  -- Draw atoms are child bits below the new region.
  have drawAtoms : ∀ term ∈ drawTerms offset, term.2.eval completed = term.2.eval base := by
    intro term member
    have below : ∀ index bit, index < fieldCount → bit < CanonicalU64.bitCount →
        (fieldBit offset index bit).eval completed = (fieldBit offset index bit).eval base := by
      intro index bit indexBelow bitBelow
      apply completedEnv_below
      simp only [childOffset, quotientStart, childWidth, CanonicalU64.auxiliaryCount,
        CanonicalU64.bitCount, fieldCount] at *
      omega
    simp only [drawTerms, fieldTerms, List.mem_append, List.mem_map, List.mem_range] at member
    rcases member with (((⟨bit, bitBelow, rfl⟩ | ⟨bit, bitBelow, rfl⟩) | ⟨bit, bitBelow, rfl⟩) |
      ⟨bit, bitBelow, rfl⟩)
    · exact below 0 bit (by decide) bitBelow
    · exact below 1 bit (by decide) bitBelow
    · exact below 2 bit (by decide) bitBelow
    · exact below 3 bit (by decide) bitBelow
  have drawSame : ∀ modulusValue,
      linearValue completed (reduceTerms modulusValue (drawTerms offset)) =
        linearValue base (reduceTerms modulusValue (drawTerms offset)) := by
    intro modulusValue
    apply linearValue_congr
    intro term member
    simp only [reduceTerms, List.mem_map] at member
    obtain ⟨original, originalMember, rfl⟩ := member
    exact drawAtoms original originalMember
  -- Each check row closes with its quotient.
  have checkClosed : ∀ index : Fin checkCount,
      left index = right index + modulus index * quotients index.val ∧
        quotients index.val < 549 := by
    intro index
    let m := modulus index
    have positive := modulus_pos index
    have leftModEq : left index ≡ draw [MOD m] := by
      have biasZero : m * checkBias ≡ 0 [MOD m] := (Nat.modEq_zero_iff_dvd).mpr (Dvd.intro _ rfl)
      have sum := (linearValue_reduce_modEq base m (drawTerms offset)).add biasZero
      rw [Nat.add_zero, drawValue] at sum
      exact sum
    have rightModEq : right index ≡ draw [MOD m] := by
      have := linearValue_reduce_modEq zeroEnv m (resultTerms offset)
      rwa [resultTerms_honest base offset draw (fun _ => 0) drawBound] at this
    have rightBound : right index ≤ 293 * (m - 1) := by
      have := linearValue_le zeroEnv (reduceTerms m (resultTerms offset)) (m - 1)
        (reduceTerms_bound m positive _)
        (reduceTerms_bits zeroEnv m _ (resultTerms_bits_honest base offset draw (fun _ => 0)))
      rwa [reduceTerms_length, length_resultTerms] at this
    have leftLower : 293 * m ≤ left index := by
      change 293 * m ≤ linearValue base (reduceTerms m (drawTerms offset)) + m * checkBias
      have : checkBias = 293 := rfl
      rw [this]; omega
    have leftUpper : left index ≤ 256 * (m - 1) + 293 * m := by
      have := linearValue_le base (reduceTerms m (drawTerms offset)) (m - 1)
        (reduceTerms_bound m positive _) (reduceTerms_bits base m _ (by
          intro term member
          simp only [drawTerms, fieldTerms, List.mem_append, List.mem_map, List.mem_range] at member
          rcases member with (((⟨bit, below, rfl⟩ | ⟨bit, below, rfl⟩) | ⟨bit, below, rfl⟩) |
            ⟨bit, below, rfl⟩)
          · exact Nat.le_of_lt_succ ((children 0).bit_lt_two bit below)
          · exact Nat.le_of_lt_succ ((children 1).bit_lt_two bit below)
          · exact Nat.le_of_lt_succ ((children 2).bit_lt_two bit below)
          · exact Nat.le_of_lt_succ ((children 3).bit_lt_two bit below)))
      rw [reduceTerms_length, length_drawTerms] at this
      change linearValue base (reduceTerms m (drawTerms offset)) + m * checkBias ≤ _
      have : checkBias = 293 := rfl
      rw [this]; omega
    have ordered : right index ≤ left index := by
      calc right index ≤ 293 * (m - 1) := rightBound
        _ ≤ 293 * m := Nat.mul_le_mul_left _ (Nat.sub_le _ _)
        _ ≤ left index := leftLower
    have divides : m ∣ left index - right index :=
      (Nat.modEq_iff_dvd' ordered).mp (rightModEq.trans leftModEq.symm)
    have quotientEq : quotients index.val = (left index - right index) / m := by
      simp only [quotients, checkQuotients, dif_pos index.isLt, Fin.eta]
      rfl
    refine ⟨?_, ?_⟩
    · rw [quotientEq, Nat.mul_div_cancel' divides]
      omega
    · rw [quotientEq]
      have : (left index - right index) / m < 549 := by
        apply Nat.div_lt_of_lt_mul
        calc left index - right index ≤ left index := Nat.sub_le _ _
          _ ≤ 256 * (m - 1) + 293 * m := leftUpper
          _ < m * 549 := by
            have : 1 ≤ m := positive
            omega
      exact this
  refine ⟨?_, ?_⟩
  · change holdsFlat completed (operations interface hints offset)
    intro expression member
    simp only [operations, flatConstraints, List.flatMap_append, List.mem_append] at member
    rcases member with (childMember | witnessMember) | rowMember
    · have childHolds : ConstraintsHold completed (flatConstraints (childOps interface offset)) := by
        apply constraintsHold_of_agree_below base completed _ (quotientStart offset)
        · exact childScope
        · intro index below
          exact completedEnv_below index below
        · exact childRows
      exact childHolds expression childMember
    · simp [Op.flatConstraints, recipeConstraints, WitnessBatch.hinted] at witnessMember
    · obtain ⟨operation, operationMember, expressionMember⟩ := List.mem_flatMap.mp rowMember
      simp only [rowOps, List.mem_append, List.mem_map] at operationMember
      rcases operationMember with (⟨atom, atomMember, rfl⟩ | ⟨digit, digitMember, rfl⟩) |
          ⟨index, _, rfl⟩ <;>
        (have same := List.mem_singleton.mp expressionMember; subst same)
      · -- Booleanity of a new bit.
        have position : ∃ position, position < newBitCount ∧
            atom = Expr.var (quotientStart offset + position) := by
          simp only [newBits, List.mem_append, List.mem_map, List.mem_range, List.mem_flatMap,
            List.mem_finRange, true_and] at atomMember
          rcases atomMember with (⟨bit, below, rfl⟩ | ⟨digit, digitBelow, bit, bitBelow, rfl⟩) |
              ⟨check, bit, bitBelow, rfl⟩
          · exact ⟨bit, by simp [newBitCount, quotientBitCount] at *; omega, rfl⟩
          · refine ⟨quotientBitCount + digitBitCount * digit + bit, ?_, ?_⟩
            · simp [newBitCount, quotientBitCount, digitBitCount, digitCount, checkBitCount] at *
              omega
            · simp only [digitBit, digitStart]
              exact congrArg Expr.var (by omega)
          · refine ⟨quotientBitCount + digitCount * digitBitCount + checkBitCount * check.val + bit,
              ?_, ?_⟩
            · have := check.isLt
              simp [newBitCount, quotientBitCount, digitBitCount, digitCount, checkBitCount] at *
              omega
            · simp only [checkBit, checkStart, digitStart]
              exact congrArg Expr.var (by omega)
        obtain ⟨position, below, rfl⟩ := position
        change completed (quotientStart offset + position) *
          (Expr.var (quotientStart offset + position) - 1).eval completed = 0
        rw [Expr.eval_sub]
        rcases newVariable_bit (base := base) (offset := offset) (draw := draw)
          (check := quotients) position with zero | outside
        · simpa [Expr.eval] using! zero
        · exact absurd below outside
      · -- Digit range.
        have below := List.mem_range.mp digitMember
        change (digitBit offset digit 2).eval completed *
          ((digitBit offset digit 0).eval completed + (digitBit offset digit 1).eval completed) = 0
        have b0 := digitBit_eval (base := base) (offset := offset) (draw := draw) (check := quotients) digit 0 below
          (by decide)
        have b1 := digitBit_eval (base := base) (offset := offset) (draw := draw) (check := quotients) digit 1 below
          (by decide)
        have b2 := digitBit_eval (base := base) (offset := offset) (draw := draw) (check := quotients) digit 2 below
          (by decide)
        set value := draw % scalarCount / 5 ^ digit % 5 with valueDef
        have small : value < 5 := Nat.mod_lt _ (by decide)
        by_cases four : value = 4
        · have zero0 : (digitBit offset digit 0).eval completed = 0 := by
            apply Fin.ext; rw [b0, four]; rfl
          have zero1 : (digitBit offset digit 1).eval completed = 0 := by
            apply Fin.ext; rw [b1, four]; rfl
          rw [zero0, zero1, add_zero, mul_zero]
        · have zero2 : (digitBit offset digit 2).eval completed = 0 := by
            apply Fin.ext; rw [b2]
            have : value / 2 ^ 2 = 0 := Nat.div_eq_of_lt (by change value < 4; omega)
            rw [this]; rfl
          rw [zero2, zero_mul]
      · -- Check row.
        obtain ⟨closed, bound⟩ := checkClosed index
        change (linearExpr (reduceTerms (modulus index) (drawTerms offset)) +
              Expr.const (fieldOfNat (modulus index * checkBias)) -
            (linearExpr (reduceTerms (modulus index) (resultTerms offset)) +
              linearExpr (checkTerms offset index))).eval completed = 0
        rw [Expr.eval_sub, sub_eq_zero]
        change (linearExpr (reduceTerms (modulus index) (drawTerms offset))).eval completed +
            fieldOfNat (modulus index * checkBias) =
          (linearExpr (reduceTerms (modulus index) (resultTerms offset))).eval completed +
            (linearExpr (checkTerms offset index)).eval completed
        rw [linearExpr_eval, linearExpr_eval, linearExpr_eval, fieldOfNat_add, fieldOfNat_add,
          drawSame, resultSame]
        have checkValue : linearValue completed (checkTerms offset index) =
            modulus index * quotients index.val := by
          unfold checkTerms
          rw [linearValue_rangeMap]
          have bits : ∀ bit ∈ Finset.range checkBitCount,
              modulus index * 2 ^ bit * ((checkBit offset index bit).eval completed).val =
                modulus index * (2 ^ bit * (quotients index.val / 2 ^ bit % 2)) := by
            intro bit member
            rw [checkBit_eval index bit (Finset.mem_range.mp member) index.isLt]
            ring
          rw [Finset.sum_congr rfl bits, ← Finset.mul_sum, binary_sum _ _ (lt_trans bound (by decide : 549 < 2 ^ checkBitCount))]
        rw [checkValue]
        exact congrArg fieldOfNat closed
  · exact fun check => (checkClosed check).2

theorem completeNew_holds (interface : Interface) (hints : Nat → List Hint) (base : Env)
    (offset : Nat)
    (children : ∀ index, CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) base)
    (childRows : holdsFlat base (childOps interface offset))
    (childScope : ∀ expression ∈ flatConstraints (childOps interface offset),
      expression.VarsBelow (quotientStart offset)) :
    holdsFlat (completeNew interface base offset) (operations interface hints offset) := by
  exact (completeNew_certificate interface hints base offset children childRows childScope).1

/-- V4: from any environment whose inputs lie below the offset, an honest
completion changes only the gadget's variables and satisfies every row. -/
theorem completeness (interface : Interface) (hints : Nat → List Hint) (env : Env)
    (offset : Nat) (assumptions : Assumptions interface offset)
    (allocates : (hints offset).length = newBitCount) :
    ∃ completed,
      AgreesOutside env completed offset (localLength (operations interface hints offset)) ∧
      holdsFlat completed (operations interface hints offset) := by
  obtain ⟨done, doneOps, doneLength⟩ := completeChildren interface offset env assumptions
  let base := done.current
  have childRows : holds base (childOps interface offset) := by
    rw [← doneOps]
    exact holdsFlat_implies_holds _ _ done.rows
  have children : ∀ index, CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) base := by
    intro index
    have callHolds := childRows (childOpAt interface offset index) (by
      simp only [childOps, List.mem_map, List.mem_finRange, true_and]
      exact ⟨index, rfl⟩)
    change CanonicalU64.Assumptions (childInterface interface offset index)
        (childOffset offset index) base →
      CanonicalU64.SpecHolds (childInterface interface offset index) (childOffset offset index) base
      at callHolds
    exact callHolds (Expr.VarsBelow.mono _ (assumptions index) (by simp [childOffset]))
  let completed := completeNew interface base offset
  refine ⟨completed, ?_, completeNew_holds interface hints base offset children ?_ ?_⟩
  · rw [localLength_eq_privateCount interface hints offset allocates]
    have childAgreement := done.agrees
    rw [doneLength] at childAgreement
    exact childAgreement.append (completeNew_agreesOutside interface base offset)
  · rw [← doneOps]
    exact done.rows
  · intro expression member
    rw [← doneOps] at member
    have scope := done.scope expression member
    rwa [doneLength] at scope

/-! ### The proved circuit -/

/-- The whole-vector sampler as one opaque proved circuit. -/
def circuit (interface : Interface) (hints : Nat → List Hint)
    (allocates : ∀ offset, (hints offset).length = newBitCount) : FormalCircuit where
  main := fun offset => ((), offset + privateCount, operations interface hints offset)
  assumptions := fun offset _ => Assumptions interface offset
  spec := fun offset env => SpecHolds interface offset env
  privateCount := fun _ => privateCount
  rowCount := fun _ => WideReduction.rowCount
  privateCount_eq := fun offset =>
    localLength_eq_privateCount interface hints offset (allocates offset)
  rowCount_eq := fun offset => rowCount_eq interface hints offset
  soundness := fun env offset assumptions rows => soundness interface hints env offset assumptions rows
  completeness := fun env offset assumptions _ =>
    completeness interface hints env offset assumptions (allocates offset)

end NightstreamFPrime.Gadgets.Sampling.WideReduction
