import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Nat.ModEq
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Ring
import NightstreamFPrime.Gadgets.Sampling.WideReduction

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

/-! ### Generic sums -/

private theorem foldl_range (term : Nat → Nat) (count : Nat) :
    (List.range count).foldl (fun value index => value + term index) 0 =
      ∑ index ∈ Finset.range count, term index := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.foldl_append, inductionHypothesis, Finset.sum_range_succ]
      rfl

private theorem linearValue_flatMap_range (env : Env) (count : Nat)
    (block : Nat → List (Nat × Expr)) :
    linearValue env ((List.range count).flatMap block) =
      ∑ index ∈ Finset.range count, linearValue env (block index) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append, linearValue_append, inductionHypothesis,
        Finset.sum_range_succ]
      simp [List.flatMap_cons]

private theorem binary_sum_lt (bit : Nat → Nat) (count : Nat)
    (bits : ∀ index, index < count → bit index ≤ 1) :
    ∑ index ∈ Finset.range count, 2 ^ index * bit index < 2 ^ count := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [Finset.sum_range_succ, pow_succ]
      have earlier := inductionHypothesis fun index below => bits index (by omega)
      have last : 2 ^ count * bit count ≤ 2 ^ count := by
        calc 2 ^ count * bit count ≤ 2 ^ count * 1 :=
              Nat.mul_le_mul_left _ (bits count (by omega))
          _ = 2 ^ count := Nat.mul_one _
      omega

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

/-! ### Values of the linear forms -/

section Values

variable {interface : Interface} {hints : Nat → List Hint} {offset : Nat} {env : Env}
  {index : Fin fieldCount}

private theorem fieldWord (spec : CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) env) :
    ∑ bit ∈ Finset.range CanonicalU64.bitCount,
        2 ^ bit * ((fieldBit offset index bit).eval env).val =
      ((interface.source index offset).eval env).val := by
  have window := CanonicalU64.windowValue_eq (childInterface interface offset index) env
    (childOffset offset index) 0 CanonicalU64.bitCount spec (by simp)
  have bound : ((interface.source index offset).eval env).val < 2 ^ CanonicalU64.bitCount :=
    lt_of_lt_of_le ((interface.source index offset).eval env).isLt (by decide)
  change CanonicalU64.weightedValue env (childOffset offset index) 0 CanonicalU64.bitCount =
    ((interface.source index offset).eval env).val / 2 ^ 0 % 2 ^ CanonicalU64.bitCount at window
  rw [pow_zero, Nat.div_one, Nat.mod_eq_of_lt bound] at window
  rw [← window]
  unfold CanonicalU64.weightedValue
  rw [foldl_range]
  simp [CanonicalU64.bitValue, fieldBit, CanonicalU64.bitExpr]

private theorem fieldTerms_value (spec : CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) env) :
    linearValue env (fieldTerms offset index.val) =
      goldilocksModulus ^ index.val * ((interface.source index offset).eval env).val := by
  unfold fieldTerms
  rw [linearValue_rangeMap, ← fieldWord spec, Finset.mul_sum]
  refine Finset.sum_congr rfl fun bit _ => ?_
  ring

private theorem drawTerms_value (assumptions : Assumptions interface offset)
    (rows : holds env (operations interface hints offset)) :
    linearValue env (drawTerms offset) = (drawIndex (drawOf interface env offset)).val := by
  have value := fun index => fieldTerms_value (child_spec (hints := hints) assumptions rows index)
  have v0 : linearValue env (fieldTerms offset 0) =
      goldilocksModulus ^ 0 * ((interface.source 0 offset).eval env).val := value 0
  have v1 : linearValue env (fieldTerms offset 1) =
      goldilocksModulus ^ 1 * ((interface.source 1 offset).eval env).val := value 1
  have v2 : linearValue env (fieldTerms offset 2) =
      goldilocksModulus ^ 2 * ((interface.source 2 offset).eval env).val := value 2
  have v3 : linearValue env (fieldTerms offset 3) =
      goldilocksModulus ^ 3 * ((interface.source 3 offset).eval env).val := value 3
  rw [drawIndex_val]
  change linearValue env (fieldTerms offset 0 ++ fieldTerms offset 1 ++ fieldTerms offset 2 ++
      fieldTerms offset 3) =
    ∑ index : Fin 4, ((interface.source index offset).eval env).val * goldilocksModulus ^ index.val
  rw [Fin.sum_univ_four]
  simp only [linearValue_append]
  rw [v0, v1, v2, v3]
  change _ = ((interface.source 0 offset).eval env).val * goldilocksModulus ^ 0 +
    ((interface.source 1 offset).eval env).val * goldilocksModulus ^ 1 +
    ((interface.source 2 offset).eval env).val * goldilocksModulus ^ 2 +
    ((interface.source 3 offset).eval env).val * goldilocksModulus ^ 3
  ring

/-- `Q = Σ 2^b q_b`. -/
def quotientValue (env : Env) (offset : Nat) : Nat :=
  ∑ bit ∈ Finset.range quotientBitCount, 2 ^ bit * ((quotientBit offset bit).eval env).val

/-- `R = Σ 5^j D_j`. -/
def remainderValue (env : Env) (offset : Nat) : Nat :=
  ∑ digit ∈ Finset.range digitCount, 5 ^ digit * digitValue env offset digit

private theorem resultTerms_value :
    linearValue env (resultTerms offset) =
      scalarCount * quotientValue env offset + remainderValue env offset := by
  unfold resultTerms quotientTerms
  rw [linearValue_append, linearValue_rangeMap, linearValue_flatMap_range]
  unfold quotientValue remainderValue
  have quotientPart : ∑ bit ∈ Finset.range quotientBitCount,
        2 ^ bit * scalarCount * ((quotientBit offset bit).eval env).val =
      scalarCount * ∑ bit ∈ Finset.range quotientBitCount,
        2 ^ bit * ((quotientBit offset bit).eval env).val := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun bit _ => ?_
    ring
  have digitPart : ∑ digit ∈ Finset.range digitCount, linearValue env (digitTerms offset digit) =
      ∑ digit ∈ Finset.range digitCount, 5 ^ digit * digitValue env offset digit := by
    refine Finset.sum_congr rfl fun digit _ => ?_
    unfold digitTerms
    rw [linearValue_rangeMap]
    simp only [digitBitCount, Finset.sum_range_succ, Finset.sum_range_zero, digitValue]
    ring
  rw [quotientPart, digitPart]

private theorem checkTerms_modEq (check : Fin checkCount) :
    linearValue env (checkTerms offset check) ≡ 0 [MOD modulus check] := by
  unfold checkTerms
  rw [linearValue_rangeMap]
  have factor : ∑ bit ∈ Finset.range checkBitCount,
        modulus check * 2 ^ bit * ((checkBit offset check bit).eval env).val =
      modulus check * ∑ bit ∈ Finset.range checkBitCount,
        2 ^ bit * ((checkBit offset check bit).eval env).val := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun bit _ => ?_
    ring
  rw [factor]
  exact (Nat.modEq_zero_iff_dvd).mpr (Dvd.intro _ rfl)

end Values

/-! ### Check rows and the Chinese remainder theorem -/

section Checks

variable {interface : Interface} {hints : Nat → List Hint} {offset : Nat} {env : Env}

private theorem length_drawTerms : (drawTerms offset).length = 256 := by
  simp [drawTerms, fieldTerms, CanonicalU64.bitCount]

private theorem length_resultTerms : (resultTerms offset).length = 293 := by
  simp [resultTerms, quotientTerms, digitTerms, quotientBitCount, digitCount, digitBitCount,
    List.length_flatMap]

private theorem length_checkTerms (check : Fin checkCount) :
    (checkTerms offset check).length = 10 := by
  simp [checkTerms, checkBitCount]

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

private def modulusProduct : Nat :=
  modulus 0 * modulus 1 * modulus 2 * modulus 3 * modulus 4 * modulus 5

private theorem crt {left right : Nat}
    (congruent : ∀ check : Fin checkCount, left ≡ right [MOD modulus check]) :
    left ≡ right [MOD modulusProduct] := by
  have c01 := (Nat.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨congruent 0, congruent 1⟩
  have c012 := (Nat.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨c01, congruent 2⟩
  have c0123 := (Nat.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨c012, congruent 3⟩
  have c01234 := (Nat.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨c0123, congruent 4⟩
  exact (Nat.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨c01234, congruent 5⟩

private theorem drawCount_le_product : drawCount ≤ modulusProduct := by decide

private theorem result_le_product : scalarCount * 2 ^ quotientBitCount ≤ modulusProduct := by
  decide

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
  rw [drawTerms_value assumptions rows, resultTerms_value, ← remainder] at congruent
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
