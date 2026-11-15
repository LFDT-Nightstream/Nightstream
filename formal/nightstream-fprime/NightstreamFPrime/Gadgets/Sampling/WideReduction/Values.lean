import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Nat.ModEq
import Mathlib.Tactic.Ring
import NightstreamFPrime.Gadgets.Sampling.WideReduction

/-!
Owns the values of the whole-vector sampler linear forms under an arbitrary
environment: the draw form equals the base-`p` draw integer when the five
children hold, the result form equals `Q 5^64 + R`, check terms are multiples
of their modulus, and the seven moduli combine by the Chinese remainder theorem.
Soundness and completeness both use these facts.
-/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open NightstreamFPrime.Spec.RingStrongSet
open Finset

theorem foldl_range (term : Nat → Nat) (count : Nat) :
    (List.range count).foldl (fun value index => value + term index) 0 =
      ∑ index ∈ Finset.range count, term index := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.foldl_append, inductionHypothesis, Finset.sum_range_succ]
      rfl

theorem linearValue_flatMap_range (env : Env) (count : Nat)
    (block : Nat → List (Nat × Expr)) :
    linearValue env ((List.range count).flatMap block) =
      ∑ index ∈ Finset.range count, linearValue env (block index) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append, linearValue_append, inductionHypothesis,
        Finset.sum_range_succ]
      simp [List.flatMap_cons]

theorem binary_sum_lt (bit : Nat → Nat) (count : Nat)
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

section Values

variable {interface : Interface} {hints : Nat → List Hint} {offset : Nat} {env : Env}
  {index : Fin fieldCount}

theorem fieldWord (spec : CanonicalU64.SpecHolds (childInterface interface offset index)
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

theorem fieldTerms_value (spec : CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) env) :
    linearValue env (fieldTerms offset index.val) =
      fieldModulus ^ index.val * ((interface.source index offset).eval env).val := by
  unfold fieldTerms
  rw [linearValue_rangeMap, ← fieldWord spec, Finset.mul_sum]
  refine Finset.sum_congr rfl fun bit _ => ?_
  ring

theorem drawTerms_value
    (children : ∀ index, CanonicalU64.SpecHolds (childInterface interface offset index)
      (childOffset offset index) env) :
    linearValue env (drawTerms offset) = (drawIndex (drawOf interface env offset)).val := by
  have value := fun index => fieldTerms_value (children index)
  have v0 : linearValue env (fieldTerms offset 0) =
      fieldModulus ^ 0 * ((interface.source 0 offset).eval env).val := value 0
  have v1 : linearValue env (fieldTerms offset 1) =
      fieldModulus ^ 1 * ((interface.source 1 offset).eval env).val := value 1
  have v2 : linearValue env (fieldTerms offset 2) =
      fieldModulus ^ 2 * ((interface.source 2 offset).eval env).val := value 2
  have v3 : linearValue env (fieldTerms offset 3) =
      fieldModulus ^ 3 * ((interface.source 3 offset).eval env).val := value 3
  have v4 : linearValue env (fieldTerms offset 4) =
      fieldModulus ^ 4 * ((interface.source 4 offset).eval env).val := value 4
  rw [drawIndex_val]
  change linearValue env (fieldTerms offset 0 ++ fieldTerms offset 1 ++ fieldTerms offset 2 ++
      fieldTerms offset 3 ++ fieldTerms offset 4) =
    ∑ index : Fin 5, ((interface.source index offset).eval env).val * fieldModulus ^ index.val
  rw [Fin.sum_univ_five]
  simp only [linearValue_append]
  rw [v0, v1, v2, v3, v4]
  change _ = ((interface.source 0 offset).eval env).val * fieldModulus ^ 0 +
    ((interface.source 1 offset).eval env).val * fieldModulus ^ 1 +
    ((interface.source 2 offset).eval env).val * fieldModulus ^ 2 +
    ((interface.source 3 offset).eval env).val * fieldModulus ^ 3 +
    ((interface.source 4 offset).eval env).val * fieldModulus ^ 4
  ring

def quotientValue (env : Env) (offset : Nat) : Nat :=
  ∑ bit ∈ Finset.range quotientBitCount, 2 ^ bit * ((quotientBit offset bit).eval env).val

def remainderValue (env : Env) (offset : Nat) : Nat :=
  ∑ digit ∈ Finset.range digitCount, 5 ^ digit * digitValue env offset digit

theorem resultTerms_value :
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

theorem checkTerms_modEq (check : Fin checkCount) :
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

section Checks

variable {offset : Nat}

theorem length_drawTerms : (drawTerms offset).length = 320 := by
  simp [drawTerms, fieldTerms, CanonicalU64.bitCount]

theorem length_resultTerms : (resultTerms offset).length = 364 := by
  simp [resultTerms, quotientTerms, digitTerms, quotientBitCount, digitCount, digitBitCount,
    List.length_flatMap]

theorem length_checkTerms (check : Fin checkCount) :
    (checkTerms offset check).length = 10 := by
  simp [checkTerms, checkBitCount]

def modulusProduct : Nat :=
  modulus 0 * modulus 1 * modulus 2 * modulus 3 * modulus 4 * modulus 5 * modulus 6

theorem crt {left right : Nat}
    (congruent : ∀ check : Fin checkCount, left ≡ right [MOD modulus check]) :
    left ≡ right [MOD modulusProduct] := by
  have c01 := (Nat.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨congruent 0, congruent 1⟩
  have c012 := (Nat.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨c01, congruent 2⟩
  have c0123 := (Nat.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨c012, congruent 3⟩
  have c01234 := (Nat.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨c0123, congruent 4⟩
  have c012345 := (Nat.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨c01234, congruent 5⟩
  exact (Nat.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨c012345, congruent 6⟩

theorem drawCount_le_product : drawCount ≤ modulusProduct := by decide

theorem result_le_product : scalarCount * 2 ^ quotientBitCount ≤ modulusProduct := by
  decide

end Checks

end NightstreamFPrime.Gadgets.Sampling.WideReduction
