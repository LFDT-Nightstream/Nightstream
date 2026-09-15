import Mathlib.Tactic.SplitIfs
import Init.Data.ByteArray.Basic
import Init.Data.ByteArray.Lemmas
import Init.Data.Array.Bootstrap
import Init.Data.Array.Lemmas
import NightstreamFPrime.Export.NativePoseidon2RoundCore
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.CarrierAction

/-!
Prepare the signed support of one 54-coefficient child ring.
Offsets are the exact bytes 81 - input, hence lie in 28..81. Unsupported
field digits return none. Native folds read bytes, never stored UInt64 digits.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECSignedDigits

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Export.NativePoseidon2

/-- Exact reverse offset for one original child coefficient. -/
@[inline] def offset (lane : Fin ringDegree) : UInt8 :=
  UInt8.ofNat (81 - lane.val)

theorem offset_toNat (lane : Fin ringDegree) :
    (offset lane).toNat = 81 - lane.val := by
  apply UInt8.toNat_ofNat_of_lt'
  change 81 - lane.val < 256
  have live := lane.isLt
  change lane.val < 54 at live
  omega

theorem offset_bounds (lane : Fin ringDegree) :
    28 ≤ (offset lane).toNat ∧ (offset lane).toNat ≤ 81 := by
  rw [offset_toNat]
  have live := lane.isLt
  change lane.val < 54 at live
  omega

/-- Only preparation constructs these arrays and their byte bounds. -/
structure Prepared where
  private mk ::
  private positive : ByteArray
  private negative : ByteArray
  private positiveBound : ∀ byte ∈ positive.data,
    28 ≤ byte.toNat ∧ byte.toNat ≤ 81
  private negativeBound : ∀ byte ∈ negative.data,
    28 ≤ byte.toNat ∧ byte.toNat ≤ 81

private def empty : Prepared where
  positive := ByteArray.empty
  negative := ByteArray.empty
  positiveBound := by intro byte member; cases (show False by simpa using member)
  negativeBound := by intro byte member; cases (show False by simpa using member)

private def pushPositive (value : Prepared) (lane : Fin ringDegree) : Prepared where
  positive := value.positive.push (offset lane)
  negative := value.negative
  positiveBound := by
    intro byte member
    rw [ByteArray.data_push, Array.mem_push] at member
    rcases member with old | rfl
    · exact value.positiveBound byte old
    · exact offset_bounds lane
  negativeBound := value.negativeBound

private def pushNegative (value : Prepared) (lane : Fin ringDegree) : Prepared where
  positive := value.positive
  negative := value.negative.push (offset lane)
  positiveBound := value.positiveBound
  negativeBound := by
    intro byte member
    rw [ByteArray.data_push, Array.mem_push] at member
    rcases member with old | rfl
    · exact value.negativeBound byte old
    · exact offset_bounds lane

private def pushDigit (value : Prepared) (lane : Fin ringDegree) (digit : F) :
    Option Prepared :=
  if digit = 0 then some value
  else if digit = 1 then some (pushPositive value lane)
  else if digit = -1 then some (pushNegative value lane)
  else none

private def preparePrefix (digit : StoredRing) :
    (count : Nat) → count ≤ ringDegree → Option Prepared
  | 0, _ => some empty
  | count + 1, bound => do
      let value ← preparePrefix digit count (Nat.le_trans (Nat.le_succ count) bound)
      let lane : Fin ringDegree :=
        ⟨count, Nat.lt_of_lt_of_le (Nat.lt_succ_self count) bound⟩
      pushDigit value lane (digit.get lane)

/-- Inspect every original lane; any unsupported digit selects the caller's
generic path. An all-zero ring produces two empty byte arrays. -/
def prepare (digit : StoredRing) : Option Prepared :=
  preparePrefix digit ringDegree (Nat.le_refl _)

/-- Compute the signed field fold, positive offsets first. -/
@[inline] def foldF (value : Prepared) (weight : UInt8 → F) (initial : F) : F :=
  let positive := value.positive.foldl (fun total byte => total + weight byte) initial
  value.negative.foldl (fun total byte => total - weight byte) positive

/-- Native accumulation contains only byte reads and canonical add/sub.
The callback receives the stored 28..81 offset, not the original lane. -/
@[inline] def fold64 (value : Prepared) (weight : UInt8 → UInt64)
    (initial : UInt64) : UInt64 :=
  let positive := value.positive.foldl (fun total byte => add64 total (weight byte)) initial
  value.negative.foldl (fun total byte => sub64 total (weight byte)) positive

-- ByteArray.foldlM and Array.foldlM have the same logical loop. This equality
-- is used only in proofs; executable folds call ByteArray.foldl directly.
private theorem byteFold_loop_eq_array {Alpha : Type}
    (bytes : ByteArray) (step : Alpha → UInt8 → Alpha)
    (stop : Nat) (bound : stop ≤ bytes.size) (count : Nat) :
    ∀ (index : Nat) (initial : Alpha),
      ByteArray.foldlM.loop (m := Id) step bytes stop bound count index initial =
        Array.foldlM.loop (m := Id) step bytes.data stop bound count index initial := by
  induction count with
  | zero =>
      intro index initial
      rw [ByteArray.foldlM.loop, Array.foldlM.loop]
  | succ count ih =>
      intro index initial
      rw [ByteArray.foldlM.loop, Array.foldlM.loop]
      by_cases live : index < stop
      · simp only [dif_pos live]
        exact ih (index + 1)
          (step initial (bytes.get index (Nat.lt_of_lt_of_le live bound)))
      · simp only [dif_neg live]

private theorem byteFold_eq_array {Alpha : Type} (bytes : ByteArray)
    (step : Alpha → UInt8 → Alpha) (initial : Alpha) :
    bytes.foldl step initial = bytes.data.foldl step initial := by
  change ByteArray.foldlM (m := Id) step initial bytes 0 bytes.size =
    Array.foldlM (m := Id) step initial bytes.data 0 bytes.size
  rw [ByteArray.foldlM, Array.foldlM]
  simp only [ByteArray.size, dif_pos (Nat.le_refl bytes.data.size), Nat.sub_zero]
  exact byteFold_loop_eq_array bytes step bytes.size (Nat.le_refl _) bytes.size 0 initial

private theorem byteFold_eq_list {Alpha : Type} (bytes : ByteArray)
    (step : Alpha → UInt8 → Alpha) (initial : Alpha) :
    bytes.foldl step initial = bytes.data.toList.foldl step initial := by
  rw [byteFold_eq_array, ← Array.foldl_toList]

private theorem byteFold_push {Alpha : Type} (bytes : ByteArray)
    (step : Alpha → UInt8 → Alpha) (initial : Alpha) (byte : UInt8) :
    (bytes.push byte).foldl step initial = step (bytes.foldl step initial) byte := by
  simp only [byteFold_eq_array, ByteArray.data_push, Array.foldl_push]

private theorem subtractList_add (bytes : List UInt8) (weight : UInt8 → F) :
    ∀ initial extra : F,
      bytes.foldl (fun total byte => total - weight byte) (initial + extra) =
        bytes.foldl (fun total byte => total - weight byte) initial + extra := by
  induction bytes with
  | nil => intro initial extra; rfl
  | cons byte bytes ih =>
      intro initial extra
      rw [List.foldl_cons, List.foldl_cons]
      have rearrange : initial + extra - weight byte = (initial - weight byte) + extra := by
        grind only
      rw [rearrange, ih]

private theorem foldF_pushPositive (value : Prepared) (lane : Fin ringDegree)
    (weight : UInt8 → F) (initial : F) :
    foldF (pushPositive value lane) weight initial =
      foldF value weight initial + weight (offset lane) := by
  unfold foldF pushPositive
  dsimp only
  rw [byteFold_push]
  simp only [byteFold_eq_list]
  exact subtractList_add _ _ _ _

private theorem foldF_pushNegative (value : Prepared) (lane : Fin ringDegree)
    (weight : UInt8 → F) (initial : F) :
    foldF (pushNegative value lane) weight initial =
      foldF value weight initial - weight (offset lane) := by
  unfold foldF pushNegative
  dsimp only
  exact byteFold_push _ _ _ _

private theorem pushDigit_foldF (value result : Prepared) (lane : Fin ringDegree)
    (digit : F) (success : pushDigit value lane digit = some result)
    (weight : UInt8 → F) (initial : F) :
    foldF result weight initial = foldF value weight initial + weight (offset lane) * digit := by
  unfold pushDigit at success
  split_ifs at success with zero one negative
  · cases Option.some.inj success
    rw [zero]
    grind only
  · cases Option.some.inj success
    rw [foldF_pushPositive, one]
    grind only
  · cases Option.some.inj success
    rw [foldF_pushNegative, negative]
    grind only

/-- Exact canonical 54-term target. The weight is arbitrary. -/
def linearCombination (digit : StoredRing) (weight : UInt8 → F) : F :=
  sumRange ConcreteCarrier.baseOps ringDegree fun index =>
    if live : index < ringDegree then
      weight (offset ⟨index, live⟩) * digit.get ⟨index, live⟩
    else 0

private theorem preparePrefix_foldF (digit : StoredRing) (weight : UInt8 → F)
    (initial : F) : ∀ (count : Nat) (bound : count ≤ ringDegree) (result : Prepared),
    preparePrefix digit count bound = some result →
    foldF result weight initial = initial +
      sumRange ConcreteCarrier.baseOps count (fun index =>
        if live : index < ringDegree then
          weight (offset ⟨index, live⟩) * digit.get ⟨index, live⟩
        else 0)
  | 0, _, result, success => by
      have same : empty = result := Option.some.inj success
      rw [← same]
      change initial = initial + 0
      grind only
  | count + 1, bound, result, success => by
      let previousBound := Nat.le_trans (Nat.le_succ count) bound
      let lane : Fin ringDegree :=
        ⟨count, Nat.lt_of_lt_of_le (Nat.lt_succ_self count) bound⟩
      cases previous : preparePrefix digit count previousBound with
      | none =>
          simp [preparePrefix, previousBound, previous] at success
      | some value =>
          have step : pushDigit value lane (digit.get lane) = some result := by
            simpa only [preparePrefix, previous, Option.bind_some, lane, previousBound] using success
          rw [pushDigit_foldF value result lane (digit.get lane) step weight initial,
            preparePrefix_foldF digit weight initial count previousBound value previous,
            sumRange]
          rw [dif_pos lane.isLt]
          change (_ + _) + weight (offset lane) * digit.get lane =
            _ + (_ + weight (offset lane) * digit.get lane)
          grind only

/-- Successful preparation gives the complete original linear combination.
No supplied expected output or coefficient equality enters preparation. -/
theorem prepare_foldF (digit : StoredRing) (value : Prepared)
    (success : prepare digit = some value) (weight : UInt8 → F) (initial : F) :
    foldF value weight initial = initial + linearCombination digit weight :=
  preparePrefix_foldF digit weight initial ringDegree (Nat.le_refl _) value success

private theorem addList64_correct (weight : UInt8 → UInt64)
    (canonical : ∀ byte, (weight byte).toNat < goldilocksModulus) :
    ∀ (bytes : List UInt8) (initial : UInt64), initial.toNat < goldilocksModulus →
      (bytes.foldl (fun total byte => add64 total (weight byte)) initial).toNat < goldilocksModulus ∧
      (bytes.foldl (fun total byte => add64 total (weight byte)) initial).denote =
        bytes.foldl (fun total byte => total + (weight byte).denote) initial.denote
  | [], initial, bound => ⟨bound, rfl⟩
  | byte :: bytes, initial, bound => by
      simp only [List.foldl_cons]
      have tail := addList64_correct weight canonical bytes (add64 initial (weight byte))
        (add64_canonical _ _ bound (canonical byte))
      refine ⟨tail.1, ?_⟩
      rw [tail.2, add64_denote _ _ bound (canonical byte)]

private theorem subList64_correct (weight : UInt8 → UInt64)
    (canonical : ∀ byte, (weight byte).toNat < goldilocksModulus) :
    ∀ (bytes : List UInt8) (initial : UInt64), initial.toNat < goldilocksModulus →
      (bytes.foldl (fun total byte => sub64 total (weight byte)) initial).toNat < goldilocksModulus ∧
      (bytes.foldl (fun total byte => sub64 total (weight byte)) initial).denote =
        bytes.foldl (fun total byte => total - (weight byte).denote) initial.denote
  | [], initial, bound => ⟨bound, rfl⟩
  | byte :: bytes, initial, bound => by
      simp only [List.foldl_cons]
      have tail := subList64_correct weight canonical bytes (sub64 initial (weight byte))
        (sub64_canonical _ _ bound (canonical byte))
      refine ⟨tail.1, ?_⟩
      rw [tail.2, sub64_denote _ _ bound (canonical byte)]

/-- Packed native folding stays canonical and equals its field fold. -/
theorem fold64_correct (value : Prepared) (weight : UInt8 → UInt64)
    (canonical : ∀ byte, (weight byte).toNat < goldilocksModulus)
    (initial : UInt64) (initialBound : initial.toNat < goldilocksModulus) :
    (fold64 value weight initial).toNat < goldilocksModulus ∧
      (fold64 value weight initial).denote =
        foldF value (fun byte => (weight byte).denote) initial.denote := by
  unfold fold64 foldF
  simp only [byteFold_eq_list]
  have positive := addList64_correct weight canonical value.positive.data.toList initial initialBound
  have negative := subList64_correct weight canonical value.negative.data.toList
    (value.positive.data.toList.foldl (fun total byte => add64 total (weight byte)) initial) positive.1
  exact ⟨negative.1, negative.2.trans (by rw [positive.2])⟩

/-- Native signed-offset execution agrees with all 54 original weighted digits. -/
theorem prepare_fold64 (digit : StoredRing) (value : Prepared)
    (success : prepare digit = some value) (weight : UInt8 → UInt64)
    (canonical : ∀ byte, (weight byte).toNat < goldilocksModulus)
    (initial : UInt64) (initialBound : initial.toNat < goldilocksModulus) :
    (fold64 value weight initial).denote =
      initial.denote + linearCombination digit (fun byte => (weight byte).denote) := by
  rw [(fold64_correct value weight canonical initial initialBound).2]
  exact prepare_foldF digit value success _ _

private theorem preparePrefix_none_of_unsupported (digit : StoredRing)
    (badLane : Fin ringDegree) (notZero : digit.get badLane ≠ 0)
    (notOne : digit.get badLane ≠ 1) (notNegative : digit.get badLane ≠ -1) :
    ∀ (count : Nat) (bound : count ≤ ringDegree), badLane.val < count →
      preparePrefix digit count bound = none
  | 0, _, inside => by omega
  | count + 1, bound, inside => by
      let previousBound := Nat.le_trans (Nat.le_succ count) bound
      by_cases earlier : badLane.val < count
      · rw [preparePrefix,
          preparePrefix_none_of_unsupported digit badLane notZero notOne notNegative
            count previousBound earlier]
        rfl
      · have sameIndex : badLane.val = count := by omega
        let lane : Fin ringDegree :=
          ⟨count, Nat.lt_of_lt_of_le (Nat.lt_succ_self count) bound⟩
        have sameLane : lane = badLane := Fin.ext sameIndex.symm
        unfold preparePrefix
        cases previous : preparePrefix digit count previousBound with
        | none => simp [previous, previousBound]
        | some value =>
            simp [previous, previousBound, pushDigit, lane, sameLane,
              notZero, notOne, notNegative]

/-- A non-signed coefficient always selects the generic fallback. -/
theorem prepare_none_of_unsupported (digit : StoredRing) (lane : Fin ringDegree)
    (notZero : digit.get lane ≠ 0) (notOne : digit.get lane ≠ 1)
    (notNegative : digit.get lane ≠ -1) : prepare digit = none :=
  preparePrefix_none_of_unsupported digit lane notZero notOne notNegative
    ringDegree (Nat.le_refl _) lane.isLt

private theorem preparePrefix_some_of_signed (digit : StoredRing)
    (signed : ∀ lane, digit.get lane = 0 ∨ digit.get lane = 1 ∨ digit.get lane = -1) :
    ∀ (count : Nat) (bound : count ≤ ringDegree),
      ∃ value, preparePrefix digit count bound = some value
  | 0, _ => ⟨empty, rfl⟩
  | count + 1, bound => by
      let previousBound := Nat.le_trans (Nat.le_succ count) bound
      let lane : Fin ringDegree :=
        ⟨count, Nat.lt_of_lt_of_le (Nat.lt_succ_self count) bound⟩
      obtain ⟨value, previous⟩ := preparePrefix_some_of_signed digit signed count previousBound
      rcases signed lane with zero | one | negative
      · refine ⟨value, ?_⟩
        simp [preparePrefix, previous, previousBound, pushDigit, lane, zero]
      · refine ⟨pushPositive value lane, ?_⟩
        have oneNotZero : (1 : F) ≠ 0 := by decide
        simp [preparePrefix, previous, previousBound, pushDigit, lane, one, oneNotZero]
      · refine ⟨pushNegative value lane, ?_⟩
        have negativeNotZero : (-1 : F) ≠ 0 := by decide
        have negativeNotOne : (-1 : F) ≠ 1 := by decide
        simp [preparePrefix, previous, previousBound, pushDigit, lane, negative,
          negativeNotZero, negativeNotOne]

/-- Every signed ring reaches the prepared path, including the zero ring. -/
theorem prepare_some_of_signed (digit : StoredRing)
    (signed : ∀ lane, digit.get lane = 0 ∨ digit.get lane = 1 ∨ digit.get lane = -1) :
    ∃ value, prepare digit = some value :=
  preparePrefix_some_of_signed digit signed ringDegree (Nat.le_refl _)

end NightstreamFPrime.Export.Stage1.PiDECSignedDigits
