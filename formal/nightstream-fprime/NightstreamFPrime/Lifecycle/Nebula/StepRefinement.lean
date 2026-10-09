import NightstreamFPrime.Lifecycle.Nebula.StepDecode

/-! Owns Ob4 for the memory rows of one invocation: when the rows of
`StepWitness.RowsHold` hold and the input carry is reachable, the model's
`invoke` maps the decoded input carry and the decoded records to the decoded
output carry, which is reachable again. It does not own the machine rows or
the circuit. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing
open Fin.CommRing

/-- The invariant of every carry between invocations: counters in range, and
an open segment below `S_max`. -/
structure Reach (p : Plan) (c : MemoryCarry) : Prop where
  seg : c.segIdx ≤ p.sMax
  idx : c.idx ≤ p.n
  openSeg : c.idx ≠ p.n → c.segIdx < p.sMax
  ts : c.ts < 2 ^ p.wTs

theorem reach_startCarry {p : Plan} : Reach p (startCarry (context p)) where
  seg := Nat.zero_le _
  idx := le_rfl
  openSeg := fun h => absurd rfl h
  ts := Nat.two_pow_pos _

/-! ### Small values -/

theorem two_pow_lt_modulus {k : ℕ} (small : k ≤ 63) : 2 ^ k < goldilocksModulus :=
  lt_of_le_of_lt (Nat.pow_le_pow_right (by norm_num) small) (by decide)

theorem Plan.n_lt {p : Plan} (valid : p.Valid) : p.n < goldilocksModulus := by
  have cover := valid.exactCover
  have positive := valid.positive.2.1
  have below := valid.belowModulus
  calc p.n ≤ p.n * p.bScan := Nat.le_mul_of_pos_right _ positive
    _ = p.cells := cover
    _ < goldilocksModulus := below

theorem Plan.wTs_small {p : Plan} (valid : p.Valid) : 2 ^ p.wTs ≤ 2 ^ 62 :=
  Nat.pow_le_pow_right (by norm_num) valid.fieldEncoding

theorem Plan.sMax_small {p : Plan} (valid : p.Valid) : p.sMax < 2 ^ 62 := by
  have range := valid.timestampRange
  have positive := valid.positive
  have : p.sMax ≤ p.sMax * p.n * p.bOps := by
    calc p.sMax = p.sMax * 1 * 1 := by ring
      _ ≤ p.sMax * p.n * p.bOps := by
        gcongr
        · exact positive.2.2.1
        · exact positive.1
  exact lt_of_le_of_lt this (lt_of_lt_of_le range (Plan.wTs_small valid))

theorem Plan.bOps_small {p : Plan} (valid : p.Valid) : p.bOps < 2 ^ 62 := by
  have range := valid.timestampRange
  have positive := valid.positive
  have : p.bOps ≤ p.sMax * p.n * p.bOps := Nat.le_mul_of_pos_left _
    (Nat.mul_pos positive.2.2.2 positive.2.2.1)
  exact lt_of_le_of_lt this (lt_of_lt_of_le range (Plan.wTs_small valid))

namespace StepWitness

variable {p : Plan} (w : StepWitness p)

/-- The decoded input carry. -/
def inCarry : MemoryCarry := decodeCarry w.carryIn

/-- The decoded output carry. -/
def outCarry : MemoryCarry := decodeCarry w.carryOut

theorem cIn_eq {k : ℕ} (h : k < 39) : w.cIn k = w.carryIn ⟨k, h⟩ := by
  simp [cIn, carryWord, h]

theorem cOut_eq {k : ℕ} (h : k < 39) : w.cOut k = w.carryOut ⟨k, h⟩ := by
  simp [cOut, carryWord, h]

variable {w} {zIn : List F}

/-- Spec §12: the rows select the reopen arm exactly when the input carry is
closed. -/
theorem RowsHold.arm (valid : p.Valid) (rows : w.RowsHold zIn) :
    (w.isOpen = 1 ∧ w.inCarry.idx = p.n) ∨ (w.isOpen = 0 ∧ w.inCarry.idx ≠ p.n) := by
  have word : w.inCarry.idx = (w.cIn 1).val := by
    rw [cIn_eq w (by decide)]
    rfl
  rcases rows.openBit with zero | one
  · refine Or.inr ⟨zero, fun closed => ?_⟩
    have test := rows.openTest
    rw [zero, sub_zero] at test
    have equal : w.cIn 1 = natWord p.n := by
      apply Fin.ext
      rw [← word, closed, natWord_val (Plan.n_lt valid)]
    rw [equal, sub_self, zero_mul] at test
    exact absurd test (by decide)
  · refine Or.inl ⟨one, ?_⟩
    have zero := rows.openZero
    rw [one, mul_one, sub_eq_zero] at zero
    rw [word, zero, natWord_val (Plan.n_lt valid)]

/-! ### Bits -/

theorem modulus_ne_one : goldilocksModulus ≠ 1 := by decide

theorem toBool_true {x : F} : toBool x = true ↔ x = 1 := by simp [toBool]

theorem toBool_false {x : F} (bit : IsBit x) : toBool x = false ↔ x = 0 := by
  rcases bit with rfl | rfl <;> simp [toBool, modulus_ne_one]

private theorem chunkValue_replicate (m : ℕ) : chunkValue (List.replicate m false) = 0 := by
  induction m with
  | zero => rfl
  | succ m ih => simp [List.replicate_succ, chunkValue, ih]

/-- A bit vector whose bits from `r` on are zero has a value below `2 ^ r`. -/
theorem bitsNat_lt_of_high_zero {n r : ℕ} {bits : Fin n → F} (le : r ≤ n)
    (high : ∀ k : Fin n, r ≤ k.val → bits k = 0) : bitsNat bits < 2 ^ r := by
  have split : (List.ofFn bits).map toBool =
      ((List.ofFn bits).map toBool).take r ++ List.replicate (n - r) false := by
    conv_lhs => rw [← List.take_append_drop r ((List.ofFn bits).map toBool)]
    congr 1
    apply List.ext_getElem (by simp)
    intro k h₁ h₂
    simp only [List.getElem_drop, List.getElem_map, List.getElem_ofFn, List.getElem_replicate]
    have hk : r + k < n := by simp at h₁; omega
    simp [toBool, high ⟨r + k, hk⟩ (by simp), modulus_ne_one]
  have value : ∀ (l : List Bool) (m : ℕ),
      chunkValue (l ++ List.replicate m false) = chunkValue l := by
    intro l m
    induction l with
    | nil => exact (List.nil_append _).symm ▸ chunkValue_replicate m
    | cons b l ih => simp [chunkValue, ih]
  unfold bitsNat
  rw [split, value]
  have := chunkValue_lt (((List.ofFn bits).map toBool).take r)
  refine lt_of_lt_of_le this (Nat.pow_le_pow_right (by norm_num) ?_)
  simp

/-- A bit vector of zeros has value `0`. -/
theorem bitsNat_zero {n : ℕ} {bits : Fin n → F} (zero : ∀ k, bits k = 0) : bitsNat bits = 0 := by
  have := bitsNat_lt_of_high_zero (r := 0) (Nat.zero_le n) (fun k _ => zero k)
  simpa using this

theorem bitsWord_inj {n : ℕ} {a b : Fin n → F} (small : n ≤ 63)
    (bitsA : ∀ k, IsBit (a k)) (bitsB : ∀ k, IsBit (b k)) (same : bitsWord a = bitsWord b) :
    bitsNat a = bitsNat b := by
  rw [bitsWord_eq bitsA, bitsWord_eq bitsB] at same
  exact natWord_injective (lt_trans (bitsNat_lt a) (two_pow_lt_modulus small))
    (lt_trans (bitsNat_lt b) (two_pow_lt_modulus small)) same

/-! ### Slot bits from the rows -/

variable (w)

/-- The lane bits of slot `j` are bits. -/
theorem RowsHold.slotBits (rows : w.RowsHold zIn) (j : Fin p.bOps) :
    IsBit (w.ops j).pad ∧ IsBit (w.ops j).isWrite ∧ IsBit (w.ops j).isRam ∧
      (∀ k, IsBit ((w.ops j).addr k)) ∧ (∀ k, IsBit ((w.ops j).vr k)) ∧
      (∀ k, IsBit ((w.ops j).vw k)) ∧ (∀ k, IsBit ((w.ops j).rt k)) := by
  have all := rows.opsBits j
  simp only [OpSlotBits.lane, List.mem_append, List.mem_cons, List.mem_ofFn] at all
  refine ⟨all _ (by simp), all _ (by simp), all _ (by simp), fun k => all _ ?_, fun k => all _ ?_,
    fun k => all _ ?_, fun k => all _ ?_⟩ <;> simp

/-- The other lane bits of slot `j` are zero when it is a pad slot (O7). -/
theorem RowsHold.padTail (rows : w.RowsHold zIn) (j : Fin p.bOps) (pad : (w.ops j).pad = 1) :
    (w.ops j).isWrite = 0 ∧ (w.ops j).isRam = 0 ∧ (∀ k, (w.ops j).addr k = 0) ∧
      (∀ k, (w.ops j).vr k = 0) ∧ (∀ k, (w.ops j).vw k = 0) ∧ (∀ k, (w.ops j).rt k = 0) := by
  have zero : ∀ x ∈ ((w.ops j).lane).tail, x = 0 := fun x hx => by
    have := rows.padZero j x hx
    rwa [StepWitness.pad, pad, one_mul] at this
  simp only [OpSlotBits.lane, List.cons_append, List.tail_cons, List.mem_append, List.mem_cons,
    List.mem_ofFn] at zero
  exact ⟨zero _ (by simp), zero _ (by simp), fun k => zero _ (by simp), fun k => zero _ (by simp),
    fun k => zero _ (by simp), fun k => zero _ (by simp)⟩

/-- Rows O3, O5, O6, O7 give the typed slot rows. -/
theorem RowsHold.slotRows (valid : p.Valid) (rows : w.RowsHold zIn) (j : Fin p.bOps) :
    (w.ops j).decode.Rows p := by
  obtain ⟨padBit, writeBit, ramBit, addrBits, vrBits, vwBits, rtBits⟩ := rows.slotBits w j
  refine ⟨fun readSlot => ?_, fun writeSlot => ?_, fun romSlot => ?_, fun padSlot => ?_⟩
  · have zero : (w.ops j).isWrite = 0 := (toBool_false writeBit).mp readSlot
    have keeps := rows.readKeeps j
    rw [StepWitness.isWrite, zero, sub_zero, one_mul, sub_eq_zero] at keeps
    exact bitsWord_inj (by norm_num) vwBits vrBits keeps
  · have one : (w.ops j).isWrite = 1 := toBool_true.mp writeSlot
    have rom := rows.noRomWrite j
    rw [StepWitness.isWrite, one, one_mul, sub_eq_zero] at rom
    exact toBool_true.mpr rom.symm
  · have zero : (w.ops j).isRam = 0 := (toBool_false ramBit).mp romSlot
    refine bitsNat_lt_of_high_zero valid.addressWidth fun k high => ?_
    have range := rows.romRange j k high
    rwa [StepWitness.isRam, zero, sub_zero, one_mul] at range
  · have one : (w.ops j).pad = 1 := toBool_true.mp padSlot
    obtain ⟨write, ram, addr, vr, vw, rt⟩ := rows.padTail w j one
    simp only [OpSlotBits.decode, OpSlot.padSlot, one, write, ram, toBool, bitsNat_zero addr,
      bitsNat_zero vr, bitsNat_zero vw, bitsNat_zero rt, decide_true, OpSlot.mk.injEq,
      true_and]
    simp [modulus_ne_one]

/-! ### Active operations -/

/-- An active operation of a slot list is an active slot, with its write
stamp after the active slots before it. -/
theorem mem_activeOps_exists : ∀ {ts : ℕ} {l : List OpSlot} {o : MemOp},
    o ∈ activeOps ts l → ∃ k, ∃ h : k < l.length, l[k].port = some o.access ∧
      o.rt = l[k].rt ∧ o.wt = ts + activeLength (l.take k) + 1
  | _, [], _, member => by simp [activeOps] at member
  | ts, s :: rest, o, member => by
    cases hp : s.port with
    | none =>
      simp only [activeOps, hp] at member
      obtain ⟨k, hk, port, rt, wt⟩ := mem_activeOps_exists member
      refine ⟨k + 1, by simp; omega, port, rt, ?_⟩
      simp only [List.take_succ_cons, activeLength, List.filterMap_cons, hp] at wt ⊢
      exact wt
    | some a =>
      simp only [activeOps, hp, List.mem_cons] at member
      rcases member with rfl | member
      · exact ⟨0, by simp, hp, rfl, by simp [activeLength]⟩
      · obtain ⟨k, hk, port, rt, wt⟩ := mem_activeOps_exists member
        refine ⟨k + 1, by simp; omega, port, rt, ?_⟩
        simp only [List.take_succ_cons, activeLength, List.filterMap_cons, hp,
          List.length_cons] at wt ⊢
        omega

theorem take_succ_ofFn {α : Type} {n : ℕ} (f : Fin n → α) {k : ℕ} (hk : k < n) :
    (List.ofFn f).take (k + 1) = (List.ofFn f).take k ++ [f ⟨k, hk⟩] := by
  rw [List.take_add_one, List.getElem?_ofFn]
  simp [hk]

theorem records_ops : w.records.ops = List.ofFn fun j => (w.ops j).decode := rfl

/-- Row O2: the active count is the number of active slots. -/
theorem RowsHold.cntBefore_eq (rows : w.RowsHold zIn) :
    ∀ k ≤ p.bOps, w.cntBefore k = natWord (activeLength (w.records.ops.take k))
  | 0, _ => by simp [cntBefore, activeLength, natWord_zero]
  | k + 1, hk => by
    have previous := RowsHold.cntBefore_eq rows k (by omega)
    have padBit := (rows.slotBits w ⟨k, hk⟩).1
    unfold cntBefore at previous ⊢
    rw [take_succ_ofFn _ hk, List.sum_append, previous, records_ops, take_succ_ofFn _ hk,
      activeLength_append, natWord_add]
    congr 1
    rcases padBit with zero | one
    · simp [StepWitness.pad, OpSlotBits.decode, zero, toBool, modulus_ne_one]; rfl
    · simp [StepWitness.pad, OpSlotBits.decode, one, toBool]; rfl

theorem activeLength_le (l : List OpSlot) : activeLength l ≤ l.length :=
  List.length_filterMap_le _ _

/-- Row O4: every active operation reads a stamp below its write stamp. -/
theorem RowsHold.activeFresh (valid : p.Valid) (rows : w.RowsHold zIn)
    (tsRange : (w.cIn 2).val < 2 ^ p.wTs) :
    ∀ o ∈ activeOps (w.cIn 2).val w.records.ops, o.rt < o.wt := by
  intro o member
  obtain ⟨k, hk, port, rt, wt⟩ := mem_activeOps_exists member
  have length : w.records.ops.length = p.bOps := by simp [records]
  have hk' : k < p.bOps := length ▸ hk
  let j : Fin p.bOps := ⟨k, hk'⟩
  have slotEq : w.records.ops[k] = (w.ops j).decode := by simp [records, j]
  rw [slotEq] at port rt
  have active : (w.ops j).pad = 0 := by
    have padBit := (rows.slotBits w j).1
    rcases padBit with zero | one
    · exact zero
    · simp [OpSlotBits.decode, OpSlot.port, toBool, one] at port
  obtain ⟨-, -, -, -, -, -, rtBits⟩ := rows.slotBits w j
  have diffBits : ∀ k, IsBit ((w.ops j).diff k) := rows.diffBits j
  have rowZero := rows.fresh j
  rw [StepWitness.pad, active, sub_zero, one_mul] at rowZero
  have row : w.wt j = w.rt j + 1 + w.diff j := by linear_combination rowZero
  have count := rows.cntBefore_eq w (k + 1) hk'
  have countStep : activeLength (w.records.ops.take (k + 1)) =
      activeLength (w.records.ops.take k) + 1 := by
    rw [records_ops, take_succ_ofFn _ hk', activeLength_append]
    have active' : (w.ops ⟨k, hk'⟩).pad = 0 := active
    simp [OpSlotBits.decode, active', toBool, modulus_ne_one]
  have wtWord : w.wt j = natWord o.wt := by
    rw [StepWitness.wt, count, countStep, wt, Nat.add_assoc]
    conv_rhs => rw [natWord_add, natWord_of_val]
  rw [wtWord, StepWitness.rt, StepWitness.diff, bitsWord_eq rtBits, bitsWord_eq diffBits,
    show (1 : F) = natWord 1 from rfl, ← natWord_add, ← natWord_add] at row
  have small62 := Plan.wTs_small valid
  have opsSmall := Plan.bOps_small valid
  have prefixSmall : activeLength (w.records.ops.take k) ≤ p.bOps := by
    have := activeLength_le (w.records.ops.take k)
    simp only [List.length_take] at this
    omega
  have rtSmall := bitsNat_lt (w.ops j).rt
  have diffSmall := bitsNat_lt (w.ops j).diff
  have equal := natWord_injective (by rw [wt]; unfold goldilocksModulus; omega)
    (by unfold goldilocksModulus; omega) row
  rw [rt]
  change bitsNat (w.ops j).rt < o.wt
  omega

theorem activeCount_records : activeCount w.records = activeLength w.records.ops := rfl

/-- Row §8.4: the output timestamp is `ts + cnt`, below `2 ^ W_ts`. -/
theorem RowsHold.boundary (valid : p.Valid) (rows : w.RowsHold zIn)
    (tsRange : (w.cIn 2).val < 2 ^ p.wTs) :
    (w.cIn 2).val + activeCount w.records < 2 ^ p.wTs ∧
      w.cOut 2 = natWord ((w.cIn 2).val + activeCount w.records) := by
  have length : w.records.ops.length = p.bOps := by simp [records]
  have count := rows.cntBefore_eq w p.bOps le_rfl
  rw [List.take_of_length_le length.le, ← activeCount_records] at count
  have out : w.cOut 2 = natWord ((w.cIn 2).val + activeCount w.records) := by
    rw [rows.tsOut, count, natWord_add, natWord_of_val]
  refine ⟨?_, out⟩
  have range := rows.tsRange
  rw [out, bitsWord_eq rows.tsBits] at range
  have small62 := Plan.wTs_small valid
  have opsSmall := Plan.bOps_small valid
  have active : activeCount w.records ≤ p.bOps := by
    rw [activeCount_records, ← length]
    exact activeLength_le _
  have bitsSmall := bitsNat_lt w.tsBits
  have equal := natWord_injective (by unfold goldilocksModulus; omega)
    (by unfold goldilocksModulus; omega) range
  omega

/-- The rows give the typed step rows of spec §8 (Lemma 4's premise). -/
theorem RowsHold.stepRows (valid : p.Valid) (rows : w.RowsHold zIn)
    (tsRange : (w.cIn 2).val < 2 ^ p.wTs) : StepRows p (w.cIn 2).val w.records where
  shaped := w.records_shaped
  slots := by
    intro s member
    obtain ⟨j, rfl⟩ := List.mem_ofFn.mp member
    exact rows.slotRows w valid j
  fresh := rows.activeFresh w valid tsRange
  boundary := (rows.boundary w valid tsRange).1

/-! ### The carry at the start of the step -/

/-- The proposals of the step. -/
def proposals : Digest × Digest := (w.proposalDigest 0, w.proposalDigest 4)

/-- The carry after the arm's open, read from the circuit's effective words:
the input carry with the step's challenges, start products, proposals, and
chain starts. -/
def stepStart : MemoryCarry where
  segIdx := w.inCarry.segIdx
  idx := w.idxEff.val
  ts := w.inCarry.ts
  eta := (w.eta1, w.eta2)
  products := ⟨w.startProduct 7, w.startProduct 9, w.startProduct 11, w.startProduct 13⟩
  proposed := (carryDigest w.carryOut 15, carryDigest w.carryOut 19)
  seen := ⟨w.previousDigest 0, w.previousDigest 4, w.previousDigest 8⟩
  memRoot := w.inCarry.memRoot

theorem readDigest_eq (v : Fin 39 → F) (start : ℕ) (fits : start + 3 < 39) :
    readDigest v start fits = carryDigest v start := by
  funext i
  simp [readDigest, carryDigest, carryWord, show start + i < 39 by omega]

theorem startProduct_open (word : ℕ) (opens : w.isOpen = 1) : w.startProduct word = 1 := by
  rw [GoldilocksExtensionRing.one_eq]
  simp [startProduct, opens, embed, K.add, K.mul, K.one]

theorem startProduct_continue (word : ℕ) (continues : w.isOpen = 0) :
    w.startProduct word = kOf (w.cIn word) (w.cIn (word + 1)) := by
  simp [startProduct, continues, embed, K.add, K.mul, kOf]

theorem carry_ext {E D : Type} {c c' : Carry E D} (segIdx : c.segIdx = c'.segIdx)
    (idx : c.idx = c'.idx) (ts : c.ts = c'.ts) (eta : c.eta = c'.eta)
    (products : c.products = c'.products) (proposed : c.proposed = c'.proposed)
    (seen : c.seen = c'.seen) (memRoot : c.memRoot = c'.memRoot) : c = c' := by
  cases c
  cases c'
  simp_all

/-- The step's segment is below `S_max`: by the S_max row on a reopen, by the
reach invariant on a continue. -/
theorem RowsHold.segBelow (valid : p.Valid) (rows : w.RowsHold zIn)
    (reach : Reach p w.inCarry) : w.inCarry.segIdx < p.sMax := by
  rcases rows.arm valid with ⟨opens, -⟩ | ⟨-, notClosed⟩
  · have row := rows.segRange
    rw [opens, one_mul, sub_sub, sub_eq_zero] at row
    have segBits := bitsWord_eq rows.segBits
    rw [segBits, show w.cIn 0 = natWord w.inCarry.segIdx by
      rw [cIn_eq w (by decide)]; exact (natWord_of_val _).symm, ← natWord_add] at row
    have sMaxSmall := Plan.sMax_small valid
    have positive : 0 < p.sMax := valid.positive.2.2.2
    have widthSmall : p.segWidth ≤ 62 := by
      have := (Nat.log2_lt positive.ne').mpr sMaxSmall
      unfold Plan.segWidth
      omega
    have bitsSmall := lt_of_lt_of_le (bitsNat_lt w.segBits)
      (Nat.pow_le_pow_right (by norm_num) widthSmall)
    have segLe : w.inCarry.segIdx ≤ p.sMax := reach.seg
    have equal := natWord_injective (by unfold goldilocksModulus; omega)
      (by unfold goldilocksModulus; omega) row
    omega
  · exact reach.openSeg notClosed

/-- The arm's open (spec §11.2) gives the step-start carry. -/
theorem RowsHold.opened (valid : p.Valid) (rows : w.RowsHold zIn)
    (reach : Reach p w.inCarry) :
    (if w.inCarry.idx = p.n then Spec.Nebula.openSegment (context p) w.inCarry w.proposals
      else some w.inCarry) = some w.stepStart := by
  rcases rows.arm valid with ⟨opens, closed⟩ | ⟨continues, notClosed⟩
  · rw [if_pos closed]
    have segBelow := rows.segBelow w valid reach
    rw [Spec.Nebula.openSegment,
      if_pos (show w.inCarry.segIdx < (context p).plan.sMax from segBelow)]
    congr 1
    have proposedOut : ∀ i : Fin 8, carryWord w.carryOut (15 + i) = w.proposal i := fun i => by
      simpa [opens, cOut] using rows.proposed i
    have etaOut : ∀ i : Fin 4, carryWord w.carryOut (3 + i) = w.etaFresh i := fun i => by
      simpa [opens, cOut] using rows.eta i
    have seenStart : ∀ i : Fin 12, w.seenPrev i = headerWords p i := fun i => by
      simpa [opens] using rows.seenPrev i
    apply carry_ext
    · rfl
    · simp [openedCarry, stepStart, rows.idxEff, opens]
    · rfl
    · show etaChallenges ⟨planDigest p, w.inCarry.ts, w.proposalDigest 0, w.inCarry.memRoot,
        w.proposalDigest 4⟩ = (w.eta1, w.eta2)
      have memRoot : w.inCarry.memRoot = carryDigest w.carryIn 35 := readDigest_eq _ _ (by decide)
      have ts : w.inCarry.ts = (w.cIn 2).val := by rw [cIn_eq w (by decide)]; rfl
      rw [memRoot, ts, rows.freshEta]
      have e0 : w.cOut 3 = w.etaFresh 0 := by simpa [cOut] using etaOut 0
      have e1 : w.cOut 4 = w.etaFresh 1 := by simpa [cOut] using etaOut 1
      have e2 : w.cOut 5 = w.etaFresh 2 := by simpa [cOut] using etaOut 2
      have e3 : w.cOut 6 = w.etaFresh 3 := by simpa [cOut] using etaOut 3
      simp only [eta1, eta2, e0, e1, e2, e3]
      rfl
    · simp [openedCarry, stepStart, startProduct_open w _ opens]
    · show (w.proposalDigest 0, w.proposalDigest 4) =
        (carryDigest w.carryOut 15, carryDigest w.carryOut 19)
      congr 1 <;> funext i <;> simp only [proposalDigest, carryDigest]
      · rw [dif_pos (by omega), ← proposedOut]
        simp
      · rw [dif_pos (by omega), ← proposedOut]
        simp only
        congr 1
        omega
    · show (⟨hash (.header .ops (planDigest p)), hash (.header .mem (planDigest p)),
          hash (.header .mem (planDigest p))⟩ : Roots Digest) =
        ⟨w.previousDigest 0, w.previousDigest 4, w.previousDigest 8⟩
      congr 1 <;> funext i <;> simp only [previousDigest] <;>
        rw [dif_pos (by omega), seenStart] <;> simp only [headerWords]
      · rw [if_pos (by omega)]
        congr 1
        exact Fin.ext (by simp; omega)
      · rw [if_neg (by omega)]
        congr 1
        exact Fin.ext (by simp; omega)
      · rw [if_neg (by omega)]
        congr 1
        exact Fin.ext (by simp; omega)
    · rfl
  · rw [if_neg notClosed]
    congr 1
    have proposedOut : ∀ i : Fin 8, carryWord w.carryOut (15 + i) = carryWord w.carryIn (15 + i) :=
      fun i => by simpa [continues, cOut, cIn] using rows.proposed i
    have etaOut : ∀ i : Fin 4, carryWord w.carryOut (3 + i) = carryWord w.carryIn (3 + i) :=
      fun i => by simpa [continues, cOut, cIn] using rows.eta i
    have seenStart : ∀ i : Fin 12, w.seenPrev i = carryWord w.carryIn (23 + i) := fun i => by
      simpa [continues, cIn] using rows.seenPrev i
    have word : ∀ (k : ℕ) (h : k < 39), carryWord w.carryIn k = w.carryIn ⟨k, h⟩ := fun k h => by
      simp [carryWord, h]
    apply carry_ext
    · rfl
    · simp [stepStart, rows.idxEff, continues, cIn, word 1 (by decide)]
      rfl
    · rfl
    · show (⟨w.carryIn 3, w.carryIn 4⟩, ⟨w.carryIn 5, w.carryIn 6⟩) = (w.eta1, w.eta2)
      have e0 : w.cOut 3 = w.carryIn 3 := by simpa [cOut, word 3 (by decide)] using etaOut 0
      have e1 : w.cOut 4 = w.carryIn 4 := by simpa [cOut, word 4 (by decide)] using etaOut 1
      have e2 : w.cOut 5 = w.carryIn 5 := by simpa [cOut, word 5 (by decide)] using etaOut 2
      have e3 : w.cOut 6 = w.carryIn 6 := by simpa [cOut, word 6 (by decide)] using etaOut 3
      simp only [eta1, eta2, kOf, e0, e1, e2, e3]
    · simp only [stepStart, startProduct_continue w _ continues, kOf, cIn, word 7 (by decide),
        word 8 (by decide), word 9 (by decide), word 10 (by decide), word 11 (by decide),
        word 12 (by decide), word 13 (by decide), word 14 (by decide)]
      rfl
    · show (readDigest w.carryIn 15 _, readDigest w.carryIn 19 _) =
        (carryDigest w.carryOut 15, carryDigest w.carryOut 19)
      rw [readDigest_eq, readDigest_eq]
      congr 1 <;> funext i <;> simp only [carryDigest]
      · exact (proposedOut ⟨i, by omega⟩).symm
      · have := proposedOut ⟨4 + i, by omega⟩
        simp only [show 15 + (4 + (i : ℕ)) = 19 + i by omega] at this
        exact this.symm
    · show (⟨readDigest w.carryIn 23 _, readDigest w.carryIn 27 _, readDigest w.carryIn 31 _⟩ :
          Roots Digest) = ⟨w.previousDigest 0, w.previousDigest 4, w.previousDigest 8⟩
      rw [readDigest_eq, readDigest_eq, readDigest_eq]
      congr 1 <;> funext i <;> simp only [carryDigest, previousDigest] <;>
        rw [dif_pos (by omega), seenStart] <;> simp only <;> congr 1 <;> omega
    · rfl

end StepWitness

/-! ### `K` values -/

theorem embed_eq (x : F) : embed x = ((x.val : ℕ) : K) := by
  rw [GoldilocksExtensionRing.natCast_eq]
  simp only [embed, K.mk.injEq, and_true]
  exact Fin.ext (by simp [Nat.mod_eq_of_lt x.isLt])

/-- The circuit fingerprint is the model fingerprint of the field values. -/
theorem fingerprintK_eq (η1 η2 η1sq : K) (square : η1sq = η1 * η1) (t g v : F) :
    fingerprintK η1 η2 η1sq t g v = fingerprint (η1, η2) (t.val, g.val, v.val) := by
  simp only [fingerprintK, fingerprint, ← GoldilocksExtensionRing.add_eq,
    ← GoldilocksExtensionRing.mul_eq, ← GoldilocksExtensionRing.sub_eq, embed_eq, square, pow_two]

/-- The O8 and O9 gate on a bit. -/
theorem gatedK_bit {pad : F} (bit : IsBit pad) (f : K) :
    gatedK pad f = if toBool pad then 1 else f := by
  rcases bit with rfl | rfl
  · have : toBool (0 : F) = false := by simp [toBool, StepWitness.modulus_ne_one]
    rw [this, if_neg (by simp)]
    simp [gatedK, embed, K.add, K.mul]
  · rw [show toBool (1 : F) = true from by simp [toBool], if_pos rfl, GoldilocksExtensionRing.one_eq]
    simp [gatedK, embed, K.add, K.mul, K.one]

namespace StepWitness

variable {p : Plan} (w : StepWitness p) {zIn : List F}

/-! ### Field values of a slot -/

theorem bitsWord_val {n : ℕ} {bits : Fin n → F} (allBits : ∀ k, IsBit (bits k)) (small : n ≤ 63) :
    (bitsWord bits).val = bitsNat bits := by
  rw [bitsWord_eq allBits, natWord_val (lt_trans (bitsNat_lt bits) (two_pow_lt_modulus small))]

theorem RowsHold.rt_val (valid : p.Valid) (rows : w.RowsHold zIn) (j : Fin p.bOps) :
    (w.rt j).val = (w.ops j).decode.rt :=
  bitsWord_val (rows.slotBits w j).2.2.2.2.2.2 (by have := valid.fieldEncoding; omega)

theorem RowsHold.vr_val (rows : w.RowsHold zIn) (j : Fin p.bOps) :
    (w.vr j).val = (w.ops j).decode.vr :=
  bitsWord_val (rows.slotBits w j).2.2.2.2.1 (by norm_num)

theorem RowsHold.vw_val (rows : w.RowsHold zIn) (j : Fin p.bOps) :
    (w.vw j).val = (w.ops j).decode.vw :=
  bitsWord_val (rows.slotBits w j).2.2.2.2.2.1 (by norm_num)

/-- The model's global index of a decoded slot. -/
def slotIndex (j : Fin p.bOps) : ℕ :=
  PortAccess.globalIndex p ⟨(w.ops j).decode.isWrite, (w.ops j).decode.isRam,
    (w.ops j).decode.addr, (w.ops j).decode.vr, (w.ops j).decode.vw⟩

theorem RowsHold.globalIndex_val (valid : p.Valid) (rows : w.RowsHold zIn) (j : Fin p.bOps) :
    (w.globalIndex j).val = w.slotIndex j := by
  obtain ⟨-, -, ramBit, addrBits, -⟩ := rows.slotBits w j
  have μSmall : p.μ ≤ 63 := by
    have below := valid.belowModulus
    have ram : 2 ^ p.μ ≤ p.cells := Nat.le_add_left _ _
    by_contra large
    have : 2 ^ 64 ≤ 2 ^ p.μ := Nat.pow_le_pow_right (by norm_num) (by omega)
    have : (2 : ℕ) ^ 64 > goldilocksModulus := by decide
    omega
  have addr : w.addr j = natWord (bitsNat (w.ops j).addr) := bitsWord_eq addrBits
  have addrSmall := bitsNat_lt (w.ops j).addr
  have cells := valid.belowModulus
  unfold Plan.cells Plan.ramSize at cells
  rcases ramBit with zero | one
  · have decodeRam : (w.ops j).decode.isRam = false := by
      simp [OpSlotBits.decode, zero, toBool, modulus_ne_one]
    simp only [globalIndex, isRam, zero, zero_mul, add_zero, addr, slotIndex, PortAccess.globalIndex,
      decodeRam]
    exact natWord_val (by unfold goldilocksModulus at *; omega)
  · have decodeRam : (w.ops j).decode.isRam = true := by simp [OpSlotBits.decode, one, toBool]
    simp only [globalIndex, isRam, one, one_mul, addr, slotIndex, PortAccess.globalIndex,
      decodeRam, if_true, ← natWord_add]
    rw [natWord_val (by unfold goldilocksModulus at *; omega)]
    simp only [OpSlotBits.decode]
    omega

theorem RowsHold.squareEq (rows : w.RowsHold zIn) : w.eta1Sq = w.eta1 * w.eta1 := by
  rw [rows.square, GoldilocksExtensionRing.mul_eq]

/-- The write stamp of an active slot (rows O2 and §6.1). -/
theorem RowsHold.wt_val (valid : p.Valid) (rows : w.RowsHold zIn) (j : Fin p.bOps)
    (active : (w.ops j).pad = 0) (tsRange : (w.cIn 2).val < 2 ^ p.wTs) :
    (w.wt j).val = (w.cIn 2).val + activeLength (w.records.ops.take j.val) + 1 := by
  have count := rows.cntBefore_eq w (j.val + 1) j.isLt
  have split : w.records.ops.take (j.val + 1) = w.records.ops.take j.val ++ [(w.ops j).decode] := by
    rw [records_ops, take_succ_ofFn _ j.isLt]
  rw [split, activeLength_append] at count
  have inactive : (w.ops j).decode.pad = false := by
    simp [OpSlotBits.decode, active, toBool, modulus_ne_one]
  rw [inactive, if_neg (by simp)] at count
  have length : w.records.ops.length = p.bOps := by simp [records]
  have prefixSmall : activeLength (w.records.ops.take j.val) ≤ p.bOps := by
    have := activeLength_le (w.records.ops.take j.val)
    simp only [List.length_take] at this
    omega
  have small62 := Plan.wTs_small valid
  have opsSmall := Plan.bOps_small valid
  have sumSmall : (w.cIn 2).val + activeLength (w.records.ops.take j.val) + 1 <
      goldilocksModulus := by
    have tsSmall := lt_of_lt_of_le tsRange small62
    unfold goldilocksModulus
    omega
  rw [StepWitness.wt, count, show w.cIn 2 + natWord (activeLength (w.records.ops.take j.val) + 1) =
      natWord ((w.cIn 2).val + activeLength (w.records.ops.take j.val) + 1) by
    rw [Nat.add_assoc, natWord_add (w.cIn 2).val, natWord_of_val]]
  exact natWord_val sumSmall

theorem RowsHold.idxEff_lt (valid : p.Valid) (rows : w.RowsHold zIn) (reach : Reach p w.inCarry) :
    w.idxEff.val < p.n := by
  have positive := valid.positive.2.2.1
  rcases rows.arm valid with ⟨opens, -⟩ | ⟨continues, notClosed⟩
  · rw [rows.idxEff, opens, sub_self, zero_mul]
    exact positive
  · have idx : w.inCarry.idx = (w.cIn 1).val := by rw [cIn_eq w (by decide)]; rfl
    rw [rows.idxEff, continues, sub_zero, one_mul, ← idx]
    exact lt_of_le_of_ne reach.idx notClosed

/-- Rows O8: the running read product after the first `k` slots. -/
theorem RowsHold.readProducts (valid : p.Valid) (rows : w.RowsHold zIn) :
    ∀ k ≤ p.bOps, w.opsProductAfter 0 k = w.startProduct 7 *
      (opsFactors p (w.eta1, w.eta2) (w.cIn 2).val (w.records.ops.take k)).1
  | 0, _ => by simp [opsProductAfter, opsFactors]
  | k + 1, hk => by
    have hk' : k < p.bOps := hk
    have previous := RowsHold.readProducts valid rows k (by omega)
    have split : w.records.ops.take (k + 1) =
        w.records.ops.take k ++ [(w.ops ⟨k, hk'⟩).decode] := by
      rw [records_ops, take_succ_ofFn _ hk']
    rw [rows.readProduct ⟨k, hk'⟩, previous, ← GoldilocksExtensionRing.mul_eq, split,
      opsFactors_append, mul_assoc]
    congr 2
    rw [StepWitness.pad, gatedK_bit (rows.slotBits w ⟨k, hk'⟩).1,
      fingerprintK_eq _ _ _ rows.squareEq, rows.rt_val w valid, rows.globalIndex_val w valid,
      rows.vr_val w]
    rfl

/-- Rows O9: the running write product after the first `k` slots. -/
theorem RowsHold.writeProducts (valid : p.Valid) (rows : w.RowsHold zIn)
    (tsRange : (w.cIn 2).val < 2 ^ p.wTs) :
    ∀ k ≤ p.bOps, w.opsProductAfter 1 k = w.startProduct 9 *
      (opsFactors p (w.eta1, w.eta2) (w.cIn 2).val (w.records.ops.take k)).2
  | 0, _ => by simp [opsProductAfter, opsFactors]
  | k + 1, hk => by
    have hk' : k < p.bOps := hk
    have previous := RowsHold.writeProducts valid rows tsRange k (by omega)
    have split : w.records.ops.take (k + 1) =
        w.records.ops.take k ++ [(w.ops ⟨k, hk'⟩).decode] := by
      rw [records_ops, take_succ_ofFn _ hk']
    rw [rows.writeProduct ⟨k, hk'⟩, previous, ← GoldilocksExtensionRing.mul_eq, split,
      opsFactors_append, mul_assoc]
    congr 2
    rw [StepWitness.pad, gatedK_bit (rows.slotBits w ⟨k, hk'⟩).1]
    rcases (rows.slotBits w ⟨k, hk'⟩).1 with zero | one
    · have inactive : toBool (w.ops ⟨k, hk'⟩).pad = false := by
        simp [zero, toBool, modulus_ne_one]
      simp only [inactive, OpSlotBits.decode, Bool.false_eq_true, ite_false]
      rw [fingerprintK_eq _ _ _ rows.squareEq, rows.wt_val w valid ⟨k, hk'⟩ zero tsRange,
        rows.globalIndex_val w valid, rows.vw_val w]
      rfl
    · have padTrue : toBool (w.ops ⟨k, hk'⟩).pad = true := by simp [one, toBool]
      simp only [padTrue, OpSlotBits.decode, if_true]

theorem scanBits {slot : ScanSlotBits p} (all : ∀ x ∈ slot.lane, IsBit x) :
    (∀ k, IsBit (slot.value k)) ∧ (∀ k, IsBit (slot.stamp k)) := by
  simp only [ScanSlotBits.lane, List.mem_append, List.mem_ofFn] at all
  exact ⟨fun k => all _ (Or.inl ⟨k, rfl⟩), fun k => all _ (Or.inr ⟨k, rfl⟩)⟩

theorem scanValue_val {slot : ScanSlotBits p} (all : ∀ x ∈ slot.lane, IsBit x) :
    (bitsWord slot.value).val = slot.decode.value :=
  bitsWord_val (scanBits all).1 (by norm_num)

theorem scanStamp_val (valid : p.Valid) {slot : ScanSlotBits p} (all : ∀ x ∈ slot.lane, IsBit x) :
    (bitsWord slot.stamp).val = slot.decode.stamp :=
  bitsWord_val (scanBits all).2 (by have := valid.fieldEncoding; omega)

theorem RowsHold.scanIndex_val (valid : p.Valid) (rows : w.RowsHold zIn)
    (reach : Reach p w.inCarry) (j : Fin p.bScan) :
    (w.scanIndex j).val = w.idxEff.val * p.bScan + j := by
  have idxLt := rows.idxEff_lt w valid reach
  have cover := valid.exactCover
  have below := valid.belowModulus
  have bound : w.idxEff.val * p.bScan + j < goldilocksModulus := by
    have : w.idxEff.val * p.bScan + j < p.n * p.bScan := by
      calc w.idxEff.val * p.bScan + j < w.idxEff.val * p.bScan + p.bScan := by omega
        _ = (w.idxEff.val + 1) * p.bScan := by ring
        _ ≤ p.n * p.bScan := Nat.mul_le_mul_right _ idxLt
    omega
  rw [scanIndex, ← natWord_of_val w.idxEff, ← natWord_mul, ← natWord_add, natWord_val bound,
    natWord_of_val]

private theorem take_scans (slots : Fin p.bScan → ScanSlotBits p) {k : ℕ} (hk : k < p.bScan) :
    (List.ofFn fun j => (slots j).decode).take (k + 1) =
      (List.ofFn fun j => (slots j).decode).take k ++ [(slots ⟨k, hk⟩).decode] :=
  take_succ_ofFn _ hk

private theorem take_scans_length (slots : Fin p.bScan → ScanSlotBits p) {k : ℕ}
    (hk : k ≤ p.bScan) : ((List.ofFn fun j => (slots j).decode).take k).length = k := by
  simp [hk]

/-- Row S2: the running IS product after the first `k` scan slots. -/
theorem RowsHold.initialProducts (valid : p.Valid) (rows : w.RowsHold zIn)
    (reach : Reach p w.inCarry) :
    ∀ k ≤ p.bScan, w.scanProductAfter 0 k = w.startProduct 11 *
      scanFactor (w.eta1, w.eta2) (w.idxEff.val * p.bScan) (w.records.initialScan.take k)
  | 0, _ => by simp [scanProductAfter, scanFactor]
  | k + 1, hk => by
    have hk' : k < p.bScan := hk
    have previous := RowsHold.initialProducts valid rows reach k (by omega)
    have all := rows.initialBits ⟨k, hk'⟩
    rw [rows.initialProduct ⟨k, hk'⟩, previous, ← GoldilocksExtensionRing.mul_eq]
    simp only [records]
    rw [take_scans _ hk', scanFactor_append, take_scans_length _ (by omega), mul_assoc,
      fingerprintK_eq _ _ _ rows.squareEq, scanStamp_val valid all, scanValue_val all,
      rows.scanIndex_val w valid reach]

/-- Row S3: the running FS product after the first `k` scan slots. -/
theorem RowsHold.finalProducts (valid : p.Valid) (rows : w.RowsHold zIn)
    (reach : Reach p w.inCarry) :
    ∀ k ≤ p.bScan, w.scanProductAfter 1 k = w.startProduct 13 *
      scanFactor (w.eta1, w.eta2) (w.idxEff.val * p.bScan) (w.records.finalScan.take k)
  | 0, _ => by simp [scanProductAfter, scanFactor]
  | k + 1, hk => by
    have hk' : k < p.bScan := hk
    have previous := RowsHold.finalProducts valid rows reach k (by omega)
    have all := rows.finalBits ⟨k, hk'⟩
    rw [rows.finalProduct ⟨k, hk'⟩, previous, ← GoldilocksExtensionRing.mul_eq]
    simp only [records]
    rw [take_scans _ hk', scanFactor_append, take_scans_length _ (by omega), mul_assoc,
      fingerprintK_eq _ _ _ rows.squareEq, scanStamp_val valid all, scanValue_val all,
      rows.scanIndex_val w valid reach]

end StepWitness

end NightstreamFPrime.Lifecycle.Nebula
