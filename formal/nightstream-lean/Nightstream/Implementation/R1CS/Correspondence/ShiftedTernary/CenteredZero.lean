import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.ShiftedTernarySound

/-!
Contract: model-level uniqueness of the zero centered 41-trit SIS word.

Owns: the arithmetic fact needed before an inactive shared opening may omit
rows: centered-unit coordinates whose weighted Goldilocks sum is zero are all
zero.

Does not own: R1CS row selection, production witness materialization, or any
row-removal claim.

Emits constraints: no.

Authority boundary: the theorem consumes the mathematical centered-digit
predicate and weighted sum directly; no digest or prover-supplied layout is
authority.

| Predicate/theorem | Mathematical obligation | Guarantee | Tier |
|---|---|---|---|
| `lowValue_injective` | bounded radix-3 uniqueness | Equal 41-trit values have equal coordinates | model-level |
| `centered_zero_unique` | zero weighted centered word | Every centered coordinate is zero | model-level |
-/

namespace Nightstream.Implementation.R1CS.ShiftedTernaryCenteredZero

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound

theorem lowValue_succ_eq_head_tail (digits : Nat → Nat) (count : Nat) :
    lowValue digits (count + 1) =
      digits 0 + 3 * lowValue (fun index => digits (index + 1)) count := by
  induction count with
  | zero => simp [lowValue]
  | succ count inductionHypothesis =>
      rw [lowValue, inductionHypothesis, lowValue]
      simp only [Nat.pow_succ]
      rw [Nat.mul_add]
      have termOrder :
          digits (count + 1) * (3 ^ count * 3) =
            3 * (digits (count + 1) * 3 ^ count) := by
        ac_rfl
      rw [termOrder]
      omega

/-- Bounded little-endian radix-three words have unique coordinates. -/
theorem lowValue_injective
    {left right : Nat → Nat} {count : Nat}
    (leftLt : ∀ index, index < count → left index < 3)
    (rightLt : ∀ index, index < count → right index < 3)
    (equal : lowValue left count = lowValue right count) :
    ∀ index, index < count → left index = right index := by
  induction count generalizing left right with
  | zero => intro index indexLt; omega
  | succ count inductionHypothesis =>
      have decomposition := equal
      rw [lowValue_succ_eq_head_tail, lowValue_succ_eq_head_tail] at decomposition
      have headEqual : left 0 = right 0 := by
        have leftHeadLt := leftLt 0 (by omega)
        have rightHeadLt := rightLt 0 (by omega)
        omega
      have tailEqual :
          lowValue (fun index => left (index + 1)) count =
            lowValue (fun index => right (index + 1)) count := by
        omega
      intro index indexLt
      cases index with
      | zero => exact headEqual
      | succ index =>
          apply inductionHypothesis
            (fun tailIndex tailIndexLt => leftLt (tailIndex + 1) (by omega))
            (fun tailIndex tailIndexLt => rightLt (tailIndex + 1) (by omega))
            tailEqual index
          omega

/-- A centered-unit 41-coordinate word cannot alias zero in Goldilocks.
This is the model theorem required before an inactive-opening reduction can
omit any of the current 124 source rows. -/
theorem centered_zero_unique
    {value negative : Nat → Nat}
    (digits : ∀ index, index < digitCount → Digit (value index) (negative index))
    (weightedZero : lowValue value digitCount % goldilocksP = 0) :
    ∀ index, index < digitCount → value index = 0 := by
  let trits := fun index => tritValue (value index)
  have tritLt : ∀ index, index < digitCount → trits index < 3 := by
    intro index indexLt
    exact Digit.tritValue_lt_three (digits index indexLt)
  have shiftedCongruence :
      lowValue trits digitCount % goldilocksP = shift := by
    calc
      lowValue trits digitCount % goldilocksP =
          lowValue (fun index => value index + 1) digitCount % goldilocksP := by
            apply lowValue_mod_congr
            intro index indexLt
            have tritLtP : trits index < goldilocksP :=
              Nat.lt_trans (tritLt index indexLt) (by native_decide)
            rw [Nat.mod_eq_of_lt tritLtP]
            exact (Digit.add_one_mod_eq_tritValue (digits index indexLt)).symm
      _ = (lowValue value digitCount + lowValue (fun _ => 1) digitCount) %
            goldilocksP := by rw [lowValue_pointwise_add]
      _ = (lowValue value digitCount + shift) % goldilocksP := by
            rw [shift_eq_ones_lowValue]
      _ = shift := by
            rw [Nat.add_mod, weightedZero]
            native_decide
  have tritValueLt : lowValue trits digitCount < 3 ^ digitCount :=
    lowValue_lt_pow tritLt
  have tritValueEqShift : lowValue trits digitCount = shift := by
    by_cases below : lowValue trits digitCount < goldilocksP
    · simpa [Nat.mod_eq_of_lt below] using shiftedCongruence
    · have modulusLe : goldilocksP ≤ lowValue trits digitCount :=
        Nat.le_of_not_gt below
      have reducedLt :
          lowValue trits digitCount - goldilocksP < goldilocksP := by
        have shiftLt : shift < goldilocksP := by native_decide
        have powerLt : 3 ^ digitCount < goldilocksP + shift := by native_decide
        omega
      have reducedEq :
          lowValue trits digitCount - goldilocksP = shift := by
        rw [Nat.mod_eq_sub_mod modulusLe, Nat.mod_eq_of_lt reducedLt]
          at shiftedCongruence
        exact shiftedCongruence
      have powerLt : 3 ^ digitCount < goldilocksP + shift := by native_decide
      omega
  have tritsAreOne : ∀ index, index < digitCount → trits index = 1 := by
    apply lowValue_injective tritLt
      (fun _ _ => by decide)
    rw [tritValueEqShift, shift_eq_ones_lowValue]
  intro index indexLt
  have tritOne := tritsAreOne index indexLt
  have digit := digits index indexLt
  cases digit with
  | neg valueEq _ => simp [trits, tritValue, valueEq, goldilocksP] at tritOne
  | zero valueEq _ => exact valueEq
  | pos valueEq _ => simp [trits, tritValue, valueEq, goldilocksP] at tritOne

end Nightstream.Implementation.R1CS.ShiftedTernaryCenteredZero
