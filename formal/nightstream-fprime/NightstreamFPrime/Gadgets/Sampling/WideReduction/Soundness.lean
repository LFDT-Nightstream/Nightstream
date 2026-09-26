import NightstreamFPrime.Gadgets.Sampling.WideReduction.Values

/-!
Owns the soundness of the whole-vector sampler constraints (V3): every
assignment accepted by the rows decodes the digits of `sample`. The witness
program is arbitrary. The argument is: children give `h_i` as bits, check rows
give `Σ p^i h_i ≡ Q N + R` modulo six coprime moduli, the Chinese remainder
theorem and range bounds give equality, and `R < 5^54` fixes the digits.
-/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet ProductionStrongSet
open Finset

/-! ### Row extraction -/

section Rows

variable {interface : Interface} {hints : Nat → List Hint} {offset : Nat} {env : Env}

private theorem child_spec (assumptions : Assumptions interface offset)
    (rows : holds env (operations interface hints offset)) (index : Fin fieldCount) :
    CanonicalU64.SpecHolds (childInterface interface offset index) (childOffset offset index) env := by
  have callHolds := rows (Sequence.childOp (childName index)
    (CanonicalU64.circuit (childInterface interface offset index)) (childOffset offset index)) (by
      simp only [operations, childOps, List.mem_append, List.mem_map, List.mem_finRange, true_and]
      exact Or.inl (Or.inl ⟨index, rfl⟩))
  change CanonicalU64.Assumptions (childInterface interface offset index)
      (childOffset offset index) env →
    CanonicalU64.SpecHolds (childInterface interface offset index) (childOffset offset index) env
    at callHolds
  exact callHolds (Expr.VarsBelow.mono _ (assumptions index) (by simp [childOffset]))

private theorem row_holds (rows : holds env (operations interface hints offset)) (row : Expr)
    (member : Op.assertZero row ∈ rowOps offset) : row.eval env = 0 := by
  exact rows (Op.assertZero row) (by simp only [operations, List.mem_append]; exact Or.inr member)

private theorem newBit_le_one (rows : holds env (operations interface hints offset)) (atom : Expr)
    (member : atom ∈ newBits offset) : (atom.eval env).val ≤ 1 :=
  bit_of_boolean env atom (row_holds rows (booleanRow atom) (by
    simp only [rowOps, List.mem_append, List.mem_map]
    exact Or.inl (Or.inl ⟨atom, member, rfl⟩)))

private theorem quotientBit_le_one (rows : holds env (operations interface hints offset))
    (bit : Nat) (below : bit < quotientBitCount) : ((quotientBit offset bit).eval env).val ≤ 1 :=
  newBit_le_one rows _ (by
    simp only [newBits, List.mem_append, List.mem_map, List.mem_range]
    exact Or.inl (Or.inl ⟨bit, below, rfl⟩))

private theorem digitBit_le_one (rows : holds env (operations interface hints offset))
    (digit bit : Nat) (digitBelow : digit < digitCount) (bitBelow : bit < digitBitCount) :
    ((digitBit offset digit bit).eval env).val ≤ 1 :=
  newBit_le_one rows _ (by
    simp only [newBits, List.mem_append, List.mem_map, List.mem_range, List.mem_flatMap]
    exact Or.inl (Or.inr ⟨digit, digitBelow, bit, bitBelow, rfl⟩))

private theorem checkBit_le_one (rows : holds env (operations interface hints offset))
    (check : Fin checkCount) (bit : Nat) (below : bit < checkBitCount) :
    ((checkBit offset check.val bit).eval env).val ≤ 1 :=
  newBit_le_one rows _ (by
    simp only [newBits, List.mem_append, List.mem_map, List.mem_range, List.mem_flatMap,
      List.mem_finRange, true_and]
    exact Or.inr ⟨check, bit, below, rfl⟩)

private theorem digitValue_le (rows : holds env (operations interface hints offset))
    (digit : Nat) (below : digit < digitCount) : digitValue env offset digit ≤ 4 := by
  have zero := row_holds rows (digitRangeRow offset digit) (by
    simp only [rowOps, List.mem_append, List.mem_map, List.mem_range]
    exact Or.inl (Or.inr ⟨digit, below, rfl⟩))
  have b0 := digitBit_le_one rows digit 0 below (by decide)
  have b1 := digitBit_le_one rows digit 1 below (by decide)
  have b2 := digitBit_le_one rows digit 2 below (by decide)
  change (digitBit offset digit 2).eval env *
      ((digitBit offset digit 0).eval env + (digitBit offset digit 1).eval env) = 0 at zero
  unfold digitValue
  rcases NightstreamFPrime.Spec.GoldilocksPrime.baseFieldNoZeroDivisors _ _ zero with high | low
  · rw [high]
    change _ + 2 * _ + 4 * 0 ≤ 4
    omega
  · have sum := congrArg Fin.val low
    rw [Fin.val_add] at sum
    have small : ((digitBit offset digit 0).eval env).val +
        ((digitBit offset digit 1).eval env).val < goldilocksModulus := by
      have : (2 : Nat) < goldilocksModulus := by decide
      omega
    rw [Nat.mod_eq_of_lt small] at sum
    change _ = 0 at sum
    omega

end Rows

/-- All retained values except each canonical child's two auxiliary fields
are Boolean under the authoritative rows. -/
theorem retained_bit_le_one (interface : Interface) (hints : Nat → List Hint)
    (env : Env) (offset column : Nat) (assumptions : Assumptions interface offset)
    (rows : holds env (operations interface hints offset)) (bound : column < privateCount)
    (bitColumn : column < childWidth * fieldCount → column % childWidth < CanonicalU64.bitCount) :
    (env (offset + column)).val ≤ 1 := by
  change column < 617 at bound
  by_cases inChild : column < childWidth * fieldCount
  · have bitBound := bitColumn inChild
    have indexBound : column / childWidth < fieldCount := by
      change column < 264 at inChild
      change column / 66 < 4
      omega
    have value := (child_spec assumptions rows ⟨column / childWidth, indexBound⟩).bit_lt_two
      (column % childWidth) bitBound
    change (env (offset + childWidth * (column / childWidth) + column % childWidth)).val < 2 at value
    rw [Nat.add_assoc, Nat.div_add_mod] at value
    omega
  · change ¬column < 264 at inChild
    by_cases inQuotient : column < 395
    · have value := quotientBit_le_one rows (column - 264) (by change _ < 131; omega)
      change (env (quotientStart offset + (column - 264))).val ≤ 1 at value
      have coordinate : quotientStart offset + (column - 264) = offset + column := by
        change offset + 264 + (column - 264) = _
        omega
      rwa [coordinate] at value
    · by_cases inDigit : column < 557
      · have digitBound : (column - 395) / 3 < digitCount := by change _ < 54; omega
        have value := digitBit_le_one rows ((column - 395) / 3) ((column - 395) % 3)
          digitBound (by exact Nat.mod_lt _ (by decide))
        change (env (offset + 395 + 3 * ((column - 395) / 3) + (column - 395) % 3)).val ≤ 1 at value
        have coordinate : offset + 395 + 3 * ((column - 395) / 3) + (column - 395) % 3 =
            offset + column := by omega
        rwa [coordinate] at value
      · have checkBound : (column - 557) / 10 < checkCount := by change _ < 6; omega
        have value := checkBit_le_one rows ⟨(column - 557) / 10, checkBound⟩ ((column - 557) % 10)
          (by exact Nat.mod_lt _ (by decide))
        change (env (offset + 557 + 10 * ((column - 557) / 10) + (column - 557) % 10)).val ≤ 1 at value
        have coordinate : offset + 557 + 10 * ((column - 557) / 10) + (column - 557) % 10 =
            offset + column := by omega
        rwa [coordinate] at value

/-! ### Check rows -/

section Checks

variable {interface : Interface} {hints : Nat → List Hint} {offset : Nat} {env : Env}

private theorem drawTerms_bits (assumptions : Assumptions interface offset)
    (rows : holds env (operations interface hints offset)) :
    ∀ term ∈ drawTerms offset, (term.2.eval env).val ≤ 1 := by
  intro term member
  have fieldBits : ∀ index : Fin fieldCount, ∀ term ∈ fieldTerms offset index.val,
      (term.2.eval env).val ≤ 1 := by
    intro index term member
    simp only [fieldTerms, List.mem_map, List.mem_range] at member
    obtain ⟨bit, below, rfl⟩ := member
    exact Nat.le_of_lt_succ ((child_spec assumptions rows index).bit_lt_two bit below)
  simp only [drawTerms, List.mem_append] at member
  rcases member with ((first | second) | third) | fourth
  · exact fieldBits 0 term first
  · exact fieldBits 1 term second
  · exact fieldBits 2 term third
  · exact fieldBits 3 term fourth

private theorem resultTerms_bits (rows : holds env (operations interface hints offset)) :
    ∀ term ∈ resultTerms offset, (term.2.eval env).val ≤ 1 := by
  intro term member
  simp only [resultTerms, quotientTerms, digitTerms, List.mem_append, List.mem_map,
    List.mem_range, List.mem_flatMap] at member
  rcases member with ⟨bit, below, rfl⟩ | ⟨digit, digitBelow, bit, bitBelow, rfl⟩
  · exact quotientBit_le_one rows bit below
  · exact digitBit_le_one rows digit bit digitBelow bitBelow

private theorem check_modEq (assumptions : Assumptions interface offset)
    (rows : holds env (operations interface hints offset)) (check : Fin checkCount) :
    linearValue env (drawTerms offset) ≡ linearValue env (resultTerms offset)
      [MOD modulus check] := by
  let m := modulus check
  have positive := modulus_pos check
  have small := modulus_lt check
  have zero := row_holds rows (checkRow offset check) (by
    simp only [rowOps, List.mem_append, List.mem_map, List.mem_finRange, true_and]
    exact Or.inr ⟨check, rfl⟩)
  unfold checkRow at zero
  rw [Expr.eval_sub, sub_eq_zero] at zero
  change (linearExpr (reduceTerms m (drawTerms offset))).eval env +
      fieldOfNat (m * checkBias) =
    (linearExpr (reduceTerms m (resultTerms offset))).eval env +
      (linearExpr (checkTerms offset check)).eval env at zero
  rw [linearExpr_eval, linearExpr_eval, linearExpr_eval, fieldOfNat_add, fieldOfNat_add] at zero
  have leftBound : linearValue env (reduceTerms m (drawTerms offset)) ≤ 256 * (m - 1) := by
    have := linearValue_le env (reduceTerms m (drawTerms offset)) (m - 1)
      (reduceTerms_bound m positive _) (reduceTerms_bits env m _ (drawTerms_bits assumptions rows))
    rwa [reduceTerms_length, length_drawTerms] at this
  have rightBound : linearValue env (reduceTerms m (resultTerms offset)) ≤ 293 * (m - 1) := by
    have := linearValue_le env (reduceTerms m (resultTerms offset)) (m - 1)
      (reduceTerms_bound m positive _) (reduceTerms_bits env m _ (resultTerms_bits rows))
    rwa [reduceTerms_length, length_resultTerms] at this
  have checkBound : linearValue env (checkTerms offset check) ≤ 10 * (m * 2 ^ 9) := by
    have := linearValue_le env (checkTerms offset check) (m * 2 ^ 9)
      (by
        intro term member
        simp only [checkTerms, List.mem_map, List.mem_range] at member
        obtain ⟨bit, below, rfl⟩ := member
        exact Nat.mul_le_mul_left _ (Nat.pow_le_pow_right (by decide) (by
          simp [checkBitCount] at below; omega)))
      (by
        intro term member
        simp only [checkTerms, List.mem_map, List.mem_range] at member
        obtain ⟨bit, below, rfl⟩ := member
        exact checkBit_le_one rows check bit below)
    rwa [length_checkTerms] at this
  have modulusValue : goldilocksModulus = 18446744069414584321 := rfl
  have bias : checkBias = 293 := rfl
  have equal := fieldOfNat_inj (by rw [modulusValue, bias]; omega) (by rw [modulusValue]; omega) zero
  have left : linearValue env (reduceTerms m (drawTerms offset)) + m * checkBias ≡
      linearValue env (drawTerms offset) [MOD m] := by
    have biasZero : m * checkBias ≡ 0 [MOD m] :=
      (Nat.modEq_zero_iff_dvd).mpr (Dvd.intro _ rfl)
    have sum := (linearValue_reduce_modEq env m (drawTerms offset)).add biasZero
    rwa [Nat.add_zero] at sum
  have right : linearValue env (reduceTerms m (resultTerms offset)) +
      linearValue env (checkTerms offset check) ≡ linearValue env (resultTerms offset) [MOD m] := by
    have sum := (linearValue_reduce_modEq env m (resultTerms offset)).add
      (checkTerms_modEq (env := env) (offset := offset) check)
    rwa [Nat.add_zero] at sum
  rw [equal] at left
  exact left.symm.trans right

end Checks

/-! ### Soundness -/

/-- V3: every accepted assignment decodes the sampled scalar. -/
theorem soundness (interface : Interface) (hints : Nat → List Hint) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset)
    (rows : holds env (operations interface hints offset)) :
    SpecHolds interface offset env := by
  let digits : Scalar := fun digit =>
    ⟨digitValue env offset digit.val, by
      have := digitValue_le rows digit.val digit.isLt
      change _ < 5
      omega⟩
  have remainder : (scalarIndex digits).val = remainderValue env offset := by
    change (finFunctionFinEquiv digits : Nat) = _
    rw [finFunctionFinEquiv_apply]
    unfold remainderValue
    rw [← Fin.sum_univ_eq_sum_range (fun digit => 5 ^ digit * digitValue env offset digit)]
    refine Finset.sum_congr rfl fun digit _ => ?_
    change digitValue env offset digit.val * 5 ^ digit.val = _
    ring
  have quotientBound : quotientValue env offset < 2 ^ quotientBitCount :=
    binary_sum_lt (fun bit => ((quotientBit offset bit).eval env).val) quotientBitCount
      (fun bit below => quotientBit_le_one rows bit below)
  have congruent := crt (check_modEq assumptions rows)
  rw [drawTerms_value (fun index => child_spec (hints := hints) assumptions rows index),
    resultTerms_value, ← remainder] at congruent
  have drawBound := (drawIndex (drawOf interface env offset)).isLt
  have resultBound : scalarCount * quotientValue env offset + (scalarIndex digits).val <
      modulusProduct := by
    have := (scalarIndex digits).isLt
    calc scalarCount * quotientValue env offset + (scalarIndex digits).val
        < scalarCount * quotientValue env offset + scalarCount := by omega
      _ = scalarCount * (quotientValue env offset + 1) := by ring
      _ ≤ scalarCount * 2 ^ quotientBitCount := Nat.mul_le_mul_left _ quotientBound
      _ ≤ modulusProduct := result_le_product
  have equal : (drawIndex (drawOf interface env offset)).val =
      scalarCount * quotientValue env offset + (scalarIndex digits).val := by
    have := congruent
    unfold Nat.ModEq at this
    rwa [Nat.mod_eq_of_lt (lt_of_lt_of_le drawBound drawCount_le_product),
      Nat.mod_eq_of_lt resultBound] at this
  have sampled : sample (drawOf interface env offset) = digits := by
    rw [sample_eq_iff, equal, Nat.mul_add_mod, Nat.mod_eq_of_lt (scalarIndex digits).isLt]
  intro digit
  rw [sampled]

end NightstreamFPrime.Gadgets.Sampling.WideReduction
