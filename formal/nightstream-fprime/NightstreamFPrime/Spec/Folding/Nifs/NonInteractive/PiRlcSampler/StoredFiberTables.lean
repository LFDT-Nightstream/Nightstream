import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldDecoderFiberCount
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import Init.Data.Vector.OfFn

/-!
Stored success/abort tables for the complete bounded scalar decoder. Input
digits are stored in a Vector; semantic function views occur only in proofs.
The named clock charges arithmetic, access, construction, executed literals,
and prefix-copy allowances. Proofs, local variable references, and clock
instrumentation are excluded. Counts and arithmetic operands have proved
finite bounds; this is not a machine-bit-time or random-sampling implementation.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.StoredFiberTables

open ProductionAlphabet FieldPairLaw FieldDecoderFiberCount
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

def rowWidth : Nat := coefficientCount + 1
def levelCount : Nat := FieldShortfall.fieldLaneCount + 1

abbrev Target := Vector Coefficient coefficientCount

structure Cell where
  success : Nat
  aborts : Nat

abbrev Row := Vector Cell rowWidth
abbrev Tables := Vector Row levelCount

private def symbols (target : Target) : List Coefficient := List.ofFn target.get

private theorem symbols_length (target : Target) : (symbols target).length = coefficientCount := by
  simp only [symbols, List.length_ofFn]

private theorem symbols_drop_end (target : Target) : (symbols target).drop coefficientCount = [] :=
  List.drop_eq_nil_of_le (symbols_length target).le

private theorem symbols_drop_cons (target : Target) (position : Nat) (inside : position < coefficientCount) :
    (symbols target).drop position =
      target.toArray[position]'(by simpa only [Vector.size_toArray] using inside) ::
        (symbols target).drop (position + 1) := by
  have inList : position < (symbols target).length := by simpa only [symbols_length] using inside
  rw [List.drop_eq_getElem_cons inList]
  simp only [symbols, List.getElem_ofFn] <;> rfl

private theorem window_card (lanes : Nat) :
    Nat.card (FieldDecoderFiberCount.Window lanes) = goldilocksModulus ^ lanes := by
  have fieldCard : Nat.card F = goldilocksModulus := Nat.card_fin goldilocksModulus
  rw [Nat.card_fun, fieldCard, Nat.card_fin]

/-- Successful matching windows and all aborting windows are disjoint. -/
theorem count_sum_le (lanes : Nat) (target : List Coefficient) :
    successCount lanes target + abortCount lanes target.length ≤ goldilocksModulus ^ lanes := by
  let succeeds := fun fields : FieldDecoderFiberCount.Window lanes =>
    Sampling.FirstAccepted.boundedSample verifier target.length (candidateList fields) = some target
  let aborts := fun fields : FieldDecoderFiberCount.Window lanes =>
    Sampling.FirstAccepted.boundedSample verifier target.length (candidateList fields) = none
  have separate (fields : FieldDecoderFiberCount.Window lanes)
      (success : succeeds fields) (failure : aborts fields) : False :=
    Option.some_ne_none target (success.symm.trans failure)
  have counted := disjoint_event_card succeeds aborts separate
  have bounded : Nat.card {fields : FieldDecoderFiberCount.Window lanes // succeeds fields ∨ aborts fields} ≤
      Nat.card (FieldDecoderFiberCount.Window lanes) :=
    Nat.card_le_card_of_injective Subtype.val Subtype.val_injective
  rw [counted, window_card lanes] at bounded
  change Nat.card (SuccessFiber lanes target) + Nat.card (AbortFiber lanes target.length) ≤ _ at bounded
  simpa only [success_fiber_card, abort_fiber_card] using bounded

/-- Every target that fits in the remaining ordered pairs has a positive fiber. -/
theorem successCount_pos (lanes : Nat) (target : List Coefficient)
    (fits : target.length ≤ 2 * lanes) : 0 < successCount lanes target := by
  induction lanes generalizing target with
  | zero =>
      have empty : target = [] := List.length_eq_zero_iff.mp (by omega)
      rw [empty, successCount]
      decide
  | succ lanes ih =>
      cases target with
      | nil => rw [successCount_nil]; exact pow_pos (by decide) _
      | cons first tail =>
          cases tail with
          | nil =>
              rw [successCount_one]
              have coefficientPositive : 0 <
                  (pairModulus - 1) * acceptedQuotientCount * (chunkModulus + 1) +
                    if first.val = 0 then 1 else 0 :=
                lt_of_lt_of_le (by decide) (Nat.le_add_right _ _)
              have tailPositive : 0 < successCount lanes [] := by
                rw [successCount_nil]
                exact pow_pos (by decide) _
              have productPositive := Nat.mul_pos coefficientPositive tailPositive
              omega
          | cons second suffix =>
              have tailFits : suffix.length ≤ 2 * lanes := by
                simp only [List.length_cons] at fits
                omega
              have tailPositive := ih suffix tailFits
              rw [successCount_two]
              have coefficientPositive : 0 <
                  (pairModulus - 1) * acceptedQuotientCount ^ 2 +
                    if first.val = 0 ∧ second.val = 0 then 1 else 0 :=
                lt_of_lt_of_le (by decide) (Nat.le_add_right _ _)
              have productPositive := Nat.mul_pos coefficientPositive tailPositive
              omega

/-- The all-rejection branch gives a positive abort fiber whenever an accept is needed. -/
theorem abortCount_pos (lanes need : Nat) (needed : 0 < need) : 0 < abortCount lanes need := by
  induction lanes with
  | zero => simp only [abortCount, needed, if_true]; decide
  | succ lanes ih =>
      obtain ⟨remaining, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt needed)
      rw [abortCount_step]
      have previousPositive : 0 < abortCount lanes (remaining + 1) := by
        simpa only [Nat.succ_eq_add_one] using ih
      have productPositive := Nat.mul_pos (show 0 < pairModulus - 1 by decide) previousPositive
      omega

/-- The finite-count width follows from the selected 32 lanes and 64-bit field modulus. -/
theorem field_window_count_lt_twoPow2048 : goldilocksModulus ^ FieldShortfall.fieldLaneCount < 2 ^ 2048 := by
  change goldilocksModulus ^ 32 < _
  have smaller : goldilocksModulus < 2 ^ 64 := by decide
  have powered : goldilocksModulus ^ 32 < (2 ^ 64) ^ 32 := Nat.pow_lt_pow_left smaller (by decide)
  simpa only [← pow_mul, show (64 : Nat) * 32 = 2048 from by decide] using powered

private structure Weights where
  reject : Nat
  finishOne : Nat
  successOne : Nat
  successTwo : Nat
  abortOne : Nat
  abortTwo : Nat

private def expectedWeights : Weights where
  reject := pairModulus - 1
  finishOne := (pairModulus - 1) * acceptedQuotientCount * (chunkModulus + 1)
  successOne := 2 * ((pairModulus - 1) * acceptedQuotientCount)
  successTwo := (pairModulus - 1) * acceptedQuotientCount * acceptedQuotientCount
  abortOne := 2 * ((pairModulus - 1) * rejectionBucket)
  abortTwo := (pairModulus - 1) * rejectionBucket * rejectionBucket + 1

/-- Eleven word operations, twelve literal occurrences, and two constructors.
The caller charges Unit and call. -/
private def makeWeights (_ : Unit) : Result Weights :=
  let reject := chunkModulus * chunkModulus - 1
  let success := reject * acceptedQuotientCount
  let aborts := reject * rejectionBucket
  ⟨⟨reject, success * (chunkModulus + 1), 2 * success,
    success * acceptedQuotientCount, 2 * aborts, aborts * rejectionBucket + 1⟩, 25⟩

private theorem makeWeights_value : (makeWeights ()).value = expectedWeights := rfl

private theorem weights_bounded :
    expectedWeights.reject ≤ goldilocksModulus ∧
    expectedWeights.finishOne + 1 ≤ goldilocksModulus ∧
    expectedWeights.successOne ≤ goldilocksModulus ∧
    expectedWeights.successTwo + 1 ≤ goldilocksModulus ∧
    expectedWeights.abortOne ≤ goldilocksModulus ∧
    expectedWeights.abortTwo ≤ goldilocksModulus := by decide

private def baseCell (position : Fin rowWidth) : Result Cell :=
  if position.val = coefficientCount then ⟨⟨1, 0⟩, 8⟩ else ⟨⟨0, 1⟩, 8⟩

/-- Actual array reads. The three branch clocks include reads, arithmetic,
tests, literals and constructors, and are 12, 31, and 53. Zero flags are raw indices. -/
private def stepCell (weights : Weights) (target : Target) (previous : Row)
    (position : Fin rowWidth) : Result Cell :=
  let j := position.val
  let row := previous.toArray
  have rowSize : row.size = rowWidth := previous.size_toArray
  let here := row[j]'(by rw [rowSize]; exact position.isLt)
  if saturated : j = coefficientCount then
    ⟨⟨goldilocksModulus * here.success, 0⟩, 12⟩
  else
    have inside : j < coefficientCount := by have := position.isLt; unfold rowWidth at this; omega
    let j1 := j + 1
    let next := row[j1]'(by rw [rowSize]; unfold rowWidth; omega)
    let digits := target.toArray
    have digitsSize : digits.size = coefficientCount := target.size_toArray
    let digit := digits[j]'(by rw [digitsSize]; exact inside)
    let reject := weights.reject
    if final : j1 = coefficientCount then
      let extra := if digit.val = 0 then 1 else 0
      ⟨⟨reject * here.success + (weights.finishOne + extra) * next.success,
        reject * here.aborts⟩, 31⟩
    else
      have more : j1 < coefficientCount := by omega
      let j2 := j + 2
      let later := row[j2]'(by rw [rowSize]; unfold rowWidth; omega)
      let second := digits[j1]'(by rw [digitsSize]; exact more)
      let firstZero := if digit.val = 0 then 1 else 0
      let secondZero := if second.val = 0 then 1 else 0
      let extra := firstZero * secondZero
      ⟨⟨reject * here.success + weights.successOne * next.success +
          (weights.successTwo + extra) * later.success,
        reject * here.aborts + weights.abortOne * next.aborts + weights.abortTwo * later.aborts⟩, 53⟩

private def RowCorrect (target : Target) (lanes : Nat) (row : Row) : Prop :=
  ∀ position : Fin rowWidth,
    (row.get position).success = successCount lanes ((symbols target).drop position.val) ∧
    (row.get position).aborts = abortCount lanes (coefficientCount - position.val)

private theorem baseCell_value (target : Target) (position : Fin rowWidth) :
    (baseCell position).value.success = successCount 0 ((symbols target).drop position.val) ∧
    (baseCell position).value.aborts = abortCount 0 (coefficientCount - position.val) := by
  by_cases saturated : position.val = coefficientCount
  · simp [baseCell, saturated, symbols_drop_end, successCount, abortCount]
  · have inside : position.val < coefficientCount := by
      have := position.isLt
      unfold rowWidth at this
      omega
    rw [symbols_drop_cons target position.val inside]
    have missing : 0 < coefficientCount - position.val := by omega
    simp [baseCell, saturated, successCount, abortCount, missing]

private theorem zero_flags (first second : Coefficient) :
    (if first.val = 0 then 1 else 0 : Nat) * (if second.val = 0 then 1 else 0) =
      if first.val = 0 ∧ second.val = 0 then 1 else 0 := by
  by_cases a : first.val = 0 <;> by_cases b : second.val = 0 <;> simp [a, b]

private theorem stepCell_value (target : Target) (previous : Row) (lanes : Nat)
    (correct : RowCorrect target lanes previous) (position : Fin rowWidth) :
    (stepCell expectedWeights target previous position).value.success =
        successCount (lanes + 1) ((symbols target).drop position.val) ∧
    (stepCell expectedWeights target previous position).value.aborts =
        abortCount (lanes + 1) (coefficientCount - position.val) := by
  have here := correct position
  by_cases saturated : position.val = coefficientCount
  · have empty : (symbols target).drop position.val = [] := by rw [saturated, symbols_drop_end]
    have missing : coefficientCount - position.val = 0 := by omega
    have hereSuccess : (previous.get position).success = successCount lanes [] := by rw [here.1, empty]
    simp only [stepCell, dif_pos saturated, empty, missing, abortCount_zero]
    constructor
    · change goldilocksModulus * (previous.get position).success = _
      rw [successCount, hereSuccess]
    · exact True.intro
  · have inside : position.val < coefficientCount := by
      have := position.isLt
      unfold rowWidth at this
      omega
    let j1 : Fin rowWidth := ⟨position.val + 1, by unfold rowWidth; omega⟩
    have next := correct j1
    have digitInside : position.val < target.toArray.size := by
      simpa only [Vector.size_toArray] using inside
    by_cases final : position.val + 1 = coefficientCount
    · have leftOne : coefficientCount - position.val = 1 := by omega
      have atEnd : (symbols target).drop (position.val + 1) = [] := by rw [final, symbols_drop_end]
      rw [symbols_drop_cons target position.val inside, atEnd, successCount_one, leftOne]
      have hereSuccess : (previous.get position).success =
          successCount lanes [target.toArray[position.val]'digitInside] := by
        rw [here.1, symbols_drop_cons target position.val inside, atEnd]
      have hereAbort : (previous.get position).aborts = abortCount lanes 1 := by rw [here.2, leftOne]
      have nextSuccess : (previous.get j1).success = successCount lanes [] := by
        rw [next.1]
        exact congrArg (successCount lanes) atEnd
      simp only [stepCell, dif_neg saturated, dif_pos final]
      constructor
      · change expectedWeights.reject * (previous.get position).success +
          (expectedWeights.finishOne + _) * (previous.get j1).success = _
        rw [hereSuccess, nextSuccess] <;> rfl
      · change expectedWeights.reject * (previous.get position).aborts = _
        rw [hereAbort, abortCount_step]
        simp only [Nat.zero_sub, Nat.zero_add, abortCount_zero, Nat.mul_zero, Nat.add_zero, expectedWeights]
    · have more : position.val + 1 < coefficientCount := by omega
      have secondInside : position.val + 1 < target.toArray.size := by
        simpa only [Vector.size_toArray] using more
      let j2 : Fin rowWidth := ⟨position.val + 2, by unfold rowWidth; omega⟩
      have later := correct j2
      have suffixes : (symbols target).drop position.val =
          target.toArray[position.val]'digitInside :: target.toArray[position.val + 1]'secondInside ::
            (symbols target).drop (position.val + 2) := by
        rw [symbols_drop_cons target position.val inside, symbols_drop_cons target (position.val + 1) more] <;>
          simp only [Nat.add_assoc]
      have nextSuffix : (symbols target).drop (position.val + 1) =
          target.toArray[position.val + 1]'secondInside :: (symbols target).drop (position.val + 2) := by
        rw [symbols_drop_cons target (position.val + 1) more] <;> simp only [Nat.add_assoc]
      have needSplit : coefficientCount - position.val = (coefficientCount - (position.val + 1)) + 1 := by omega
      have needTail : coefficientCount - (position.val + 1) - 1 = coefficientCount - (position.val + 2) := by omega
      rw [suffixes, successCount_two, needSplit, abortCount_step, needTail]
      simp only [stepCell, dif_neg saturated, dif_neg final]
      constructor
      · change expectedWeights.reject * (previous.get position).success +
          expectedWeights.successOne * (previous.get j1).success +
          (expectedWeights.successTwo + _) * (previous.get j2).success = _
        rw [here.1, next.1, later.1, suffixes, nextSuffix, zero_flags]
        simp only [j1, j2, expectedWeights]
        ring_nf
      · change expectedWeights.reject * (previous.get position).aborts +
          expectedWeights.abortOne * (previous.get j1).aborts +
          expectedWeights.abortTwo * (previous.get j2).aborts = _
        rw [here.2, next.2, later.2]
        simp only [j1, j2, expectedWeights, needSplit]
        ring_nf

private theorem baseCell_work (position : Fin rowWidth) : (baseCell position).work ≤ 53 := by
  unfold baseCell
  split <;> decide

private theorem stepCell_work (weights : Weights) (target : Target) (previous : Row)
    (position : Fin rowWidth) : (stepCell weights target previous position).work ≤ 53 := by
  unfold stepCell
  dsimp only
  split
  · change (12 : Nat) ≤ 53
    decide
  · split
    · change (31 : Nat) ≤ 53
      decide
    · change (53 : Nat) ≤ 53
      decide

/- The concrete builders reserve the complete output capacity. The append
allowance is `capacity + 2*prefixLength + 3`: the append call, allocation
including all reserved slots, a read/write for each possible copied entry,
and the appended write.
The empty-prefix allowance is `capacity + 5`: dispatch, capacity literal,
one named `Array.emptyWithCapacity` allocation, Vector and Result construction.
The capacity term covers reserved slots; there is no second allocator charge.
An in-place append is covered by the same bound. It is a named array-operation
allowance, not a claim about compiler instructions or physical allocation. -/

private def rowPrefix (program : Fin rowWidth → Result Cell) :
    (count : Nat) → count ≤ rowWidth → Result (Vector Cell count)
  | 0, _ => ⟨⟨Array.emptyWithCapacity rowWidth, rfl⟩, rowWidth + 5⟩
  | count + 1, within =>
      let earlier := rowPrefix program count (by omega)
      let item := program ⟨count, by omega⟩
      let buffer := earlier.value.toArray.push item.value
      ⟨⟨buffer, by simp [buffer]⟩,
        earlier.work + item.work + rowWidth + 2 * count + 13⟩

private theorem rowPrefix_value (program : Fin rowWidth → Result Cell)
    (count : Nat) (within : count ≤ rowWidth) (position : Fin count) :
    ((rowPrefix program count within).value.get position) =
      (program (position.castLE within)).value := by
  induction count with
  | zero => exact Fin.elim0 position
  | succ count ih =>
      by_cases earlier : position.val < count
      · simpa only [rowPrefix, Vector.get, Fin.coe_cast, Array.getElem_push, Vector.size_toArray,
          dif_pos earlier, Fin.val_castLE] using
          ih (by omega) ⟨position.val, earlier⟩
      · have last : position.val = count := by have := position.isLt; omega
        simp only [rowPrefix, Vector.get, Fin.coe_cast, Array.getElem_push, Vector.size_toArray, dif_neg earlier]
        apply congrArg (fun coordinate : Fin rowWidth => (program coordinate).value)
        exact Fin.ext last.symm

private def rowPrefixBound (count : Nat) : Nat :=
  rowWidth + 5 + count * (53 + 3 * rowWidth + 13)

private theorem rowPrefix_work (program : Fin rowWidth → Result Cell)
    (bounded : ∀ position, (program position).work ≤ 53)
    (count : Nat) (within : count ≤ rowWidth) :
    (rowPrefix program count within).work ≤ rowPrefixBound count := by
  induction count with
  | zero => simp only [rowPrefix, rowPrefixBound, Nat.zero_mul, Nat.add_zero, Nat.le_refl]
  | succ count ih =>
      have previous := ih (by omega)
      have item := bounded ⟨count, by omega⟩
      change (rowPrefix program count _).work + (program ⟨count, _⟩).work +
        rowWidth + 2 * count + 13 ≤ rowPrefixBound (count + 1)
      unfold rowPrefixBound at previous ⊢
      nlinarith

attribute [irreducible] rowPrefix

private def baseRow (_ : Unit) : Result Row :=
  let result := rowPrefix baseCell rowWidth (Nat.le_refl _)
  ⟨result.value, result.work + 4⟩

private def nextRow (weights : Weights) (target : Target) (previous : Row) : Result Row :=
  let result := rowPrefix (fun position => stepCell weights target previous position) rowWidth (Nat.le_refl _)
  ⟨result.value, result.work + 5⟩

private theorem baseRow_value (target : Target) : RowCorrect target 0 (baseRow ()).value := by
  intro position
  change ((rowPrefix baseCell rowWidth _).value.get position).success = _ ∧
    ((rowPrefix baseCell rowWidth _).value.get position).aborts = _
  rw [rowPrefix_value]
  exact baseCell_value target position

private theorem nextRow_value (target : Target) (previous : Row) (lanes : Nat)
    (correct : RowCorrect target lanes previous) :
    RowCorrect target (lanes + 1) (nextRow expectedWeights target previous).value := by
  intro position
  change ((rowPrefix _ rowWidth _).value.get position).success = _ ∧
    ((rowPrefix _ rowWidth _).value.get position).aborts = _
  rw [rowPrefix_value]
  exact stepCell_value target previous lanes correct position

/-- Both row constructors are bounded, including their private prefix builder. -/
private def rowWork : Nat := rowPrefixBound rowWidth + 5

private theorem baseRow_work : (baseRow ()).work ≤ rowWork := by
  have bound := rowPrefix_work baseCell baseCell_work rowWidth (Nat.le_refl _)
  change (rowPrefix baseCell rowWidth _).work + 4 ≤ rowPrefixBound rowWidth + 5
  omega

private theorem nextRow_work (weights : Weights) (target : Target) (previous : Row) :
    (nextRow weights target previous).work ≤ rowWork := by
  exact Nat.add_le_add_right
    (rowPrefix_work _ (stepCell_work weights target previous) rowWidth (Nat.le_refl _)) 5

private def tablePrefix (weights : Weights) (target : Target) :
    (count : Nat) → count ≤ levelCount → Result (Vector Row count)
  | 0, _ => ⟨⟨Array.emptyWithCapacity levelCount, rfl⟩, levelCount + 5⟩
  | count + 1, within =>
      let earlier := tablePrefix weights target count (by omega)
      let row := if first : count = 0 then baseRow () else
        let previous := earlier.value.toArray[count - 1]'(by
          simpa only [Vector.size_toArray] using (show count - 1 < count by omega))
        nextRow weights target previous
      let buffer := earlier.value.toArray.push row.value
      ⟨⟨buffer, by simp [buffer]⟩,
        earlier.work + row.work + levelCount + 2 * count + 20⟩

private theorem tablePrefix_value (target : Target)
    (count : Nat) (within : count ≤ levelCount) (level : Fin count) :
    RowCorrect target level.val ((tablePrefix expectedWeights target count within).value.get level) := by
  induction count with
  | zero => exact Fin.elim0 level
  | succ count ih =>
      by_cases earlier : level.val < count
      · simpa only [tablePrefix, Vector.get, Fin.coe_cast, Array.getElem_push, Vector.size_toArray,
          dif_pos earlier] using ih (by omega) ⟨level.val, earlier⟩
      · have last : level.val = count := by have := level.isLt; omega
        simp only [tablePrefix, Vector.get, Fin.coe_cast, Array.getElem_push, Vector.size_toArray, dif_neg earlier]
        by_cases first : count = 0
        · rw [dif_pos first]
          simpa only [last, first] using baseRow_value target
        · rw [dif_neg first]
          have positive : 0 < count := Nat.pos_of_ne_zero first
          have previous := ih (by omega) ⟨count - 1, by omega⟩
          have built := nextRow_value target
            ((tablePrefix expectedWeights target count (by omega)).value.get ⟨count - 1, by omega⟩)
            (count - 1) previous
          have restored : count - 1 + 1 = count := by omega
          simpa only [last, restored, Vector.get, Fin.coe_cast] using built

private def tablePrefixBound (count : Nat) : Nat :=
  levelCount + 5 + count * (rowWork + 3 * levelCount + 20)

private theorem tablePrefix_work (weights : Weights) (target : Target)
    (count : Nat) (within : count ≤ levelCount) :
    (tablePrefix weights target count within).work ≤ tablePrefixBound count := by
  induction count with
  | zero => simp only [tablePrefix, tablePrefixBound, Nat.zero_mul, Nat.add_zero, Nat.le_refl]
  | succ count ih =>
      have previous := ih (by omega)
      have rowBound :
          (if first : count = 0 then baseRow () else
            nextRow weights target
              ((tablePrefix weights target count (by omega)).value.toArray[count - 1]'(by
                simpa only [Vector.size_toArray] using (show count - 1 < count by omega)))).work ≤
            rowWork := by
        split
        · exact baseRow_work
        · exact nextRow_work _ _ _
      change (tablePrefix weights target count _).work +
        (if first : count = 0 then baseRow () else
          nextRow weights target
            ((tablePrefix weights target count (by omega)).value.toArray[count - 1]'(by
              simpa only [Vector.size_toArray] using (show count - 1 < count by omega)))).work +
          levelCount + 2 * count + 20 ≤ tablePrefixBound (count + 1)
      unfold tablePrefixBound at previous ⊢
      nlinarith

attribute [irreducible] tablePrefix baseRow nextRow

/-- Build one stored base row and all 32 successor rows. The only target
accesses are the actual array reads in `stepCell`. -/
def build (target : Target) : Result Tables :=
  let weights := makeWeights ()
  let result := tablePrefix weights.value target levelCount (Nat.le_refl _)
  ⟨result.value, weights.work + result.work + 7⟩

/-- The selected 33-by-55 table bound evaluates to 425407 named operations. -/
def buildWork : Nat := 25 + tablePrefixBound levelCount + 7

/-- Exact table meaning at every selected lane and prefix index. -/
theorem build_value (target : Target) (level : Fin levelCount) (position : Fin rowWidth) :
    (((build target).value.get level).get position).success =
        successCount level.val ((List.ofFn target.get).drop position.val) ∧
    (((build target).value.get level).get position).aborts =
        abortCount level.val (coefficientCount - position.val) := by
  have correct : RowCorrect target level.val
      ((tablePrefix (makeWeights ()).value target levelCount (Nat.le_refl _)).value.get level) := by
    rw [makeWeights_value]
    exact tablePrefix_value target levelCount (Nat.le_refl _) level
  exact correct position

/-- Full named-operation bound, with input access and both prefix-copy axes. -/
theorem build_work_le (target : Target) : (build target).work ≤ buildWork := by
  have bounded := tablePrefix_work (makeWeights ()).value target levelCount (Nat.le_refl _)
  change 25 + (tablePrefix (makeWeights ()).value target levelCount _).work + 7 ≤
    25 + tablePrefixBound levelCount + 7
  omega

/-- Each stored sum counts disjoint events, including at the saturated index. -/
theorem build_sum_le (target : Target) (level : Fin levelCount) (position : Fin rowWidth) :
    (((build target).value.get level).get position).success +
        (((build target).value.get level).get position).aborts ≤ goldilocksModulus ^ level.val := by
  rw [(build_value target level position).1, (build_value target level position).2]
  have length : ((List.ofFn target.get).drop position.val).length = coefficientCount - position.val := by
    simp only [List.length_drop, List.length_ofFn]
  simpa only [length] using count_sum_le level.val ((List.ofFn target.get).drop position.val)

theorem build_sum_lt_twoPow2048 (target : Target) (level : Fin levelCount) (position : Fin rowWidth) :
    (((build target).value.get level).get position).success +
        (((build target).value.get level).get position).aborts < 2 ^ 2048 := by
  have inside : level.val ≤ FieldShortfall.fieldLaneCount := by
    have := level.isLt
    unfold levelCount at this
    omega
  exact (build_sum_le target level position).trans_lt
    ((Nat.pow_le_pow_right (by decide) inside).trans_lt field_window_count_lt_twoPow2048)

/-- All six stored recurrence weights, including the largest raw-zero bonuses, fit one field word. -/
theorem recurrence_weights_le_modulus :
    pairModulus - 1 ≤ goldilocksModulus ∧
    (pairModulus - 1) * acceptedQuotientCount * (chunkModulus + 1) + 1 ≤ goldilocksModulus ∧
    2 * ((pairModulus - 1) * acceptedQuotientCount) ≤ goldilocksModulus ∧
    (pairModulus - 1) * acceptedQuotientCount * acceptedQuotientCount + 1 ≤ goldilocksModulus ∧
    2 * ((pairModulus - 1) * rejectionBucket) ≤ goldilocksModulus ∧
    (pairModulus - 1) * rejectionBucket * rejectionBucket + 1 ≤ goldilocksModulus := weights_bounded

/-- A counted read of the stored table. The two Fin values, two Vector-array
projections, two array reads, and Result construction cost seven operations. -/
def read (tables : Tables) (level : Fin levelCount) (position : Fin rowWidth) : Result Cell :=
  let n := level.val
  let j := position.val
  let row := tables.toArray[n]'(by simpa only [Vector.size_toArray] using level.isLt)
  ⟨row.toArray[j]'(by simpa only [Vector.size_toArray] using position.isLt), 7⟩

theorem read_work (tables : Tables) (level : Fin levelCount) (position : Fin rowWidth) :
    (read tables level position).work = 7 := rfl

theorem read_build_value (target : Target) (level : Fin levelCount) (position : Fin rowWidth) :
    (read (build target).value level position).value.success =
        successCount level.val ((List.ofFn target.get).drop position.val) ∧
    (read (build target).value level position).value.aborts =
        abortCount level.val (coefficientCount - position.val) := build_value target level position

/-- Every complete 54-symbol target has a positive successful 32-field fiber. -/
theorem initial_success_pos (target : Target) :
    0 < (read (build target).value (Fin.last FieldShortfall.fieldLaneCount) ⟨0, by decide⟩).value.success := by
  rw [(read_build_value target _ _).1, List.drop_zero]
  apply successCount_pos
  simp only [List.length_ofFn, Fin.val_last]
  decide

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.StoredFiberTables
