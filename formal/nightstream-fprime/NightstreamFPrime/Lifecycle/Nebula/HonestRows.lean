import NightstreamFPrime.Lifecycle.Nebula.HonestWitness

/-! Owns the polynomial rows of the honest witness (Ob5 for the relation):
for a reachable carry and an invocation that `invoke` accepts, every row of
`StepWitness.PolyRows` holds on `Honest.witness`. It does not own the chains,
the challenge transcript, the machine rows, or the word list. -/

namespace NightstreamFPrime.Lifecycle.Nebula.Honest

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing
open Fin.CommRing

variable {p : Plan} {c : MemoryCarry} {inv : Invocation Machine.State Digest}

/-! ### Carries -/

theorem start_segIdx : (start p c inv).segIdx = c.segIdx := by
  unfold start
  split <;> rfl

theorem start_memRoot : (start p c inv).memRoot = c.memRoot := by
  unfold start
  split <;> rfl

theorem start_idx_lt (valid : p.Valid) (reach : Reach p c) : (start p c inv).idx < p.n := by
  unfold start
  split
  · exact valid.positive.2.2.1
  · next h => exact lt_of_le_of_ne reach.idx h

theorem stepped_ts : (stepped p c inv).ts = c.ts + activeCount inv.records := by
  show (start p c inv).ts + activeCount inv.records = _
  rw [start_ts]

theorem stepped_segIdx : (stepped p c inv).segIdx = c.segIdx := start_segIdx

theorem stepped_memRoot : (stepped p c inv).memRoot = c.memRoot := start_memRoot

theorem finished_idx : (finished p c inv).idx = (stepped p c inv).idx := by
  simp only [finished]
  split <;> rfl

theorem finished_ts : (finished p c inv).ts = (stepped p c inv).ts := by
  simp only [finished]
  split <;> rfl

theorem finished_eta : (finished p c inv).eta = (start p c inv).eta := by
  simp only [finished]
  split <;> rfl

theorem finished_products : (finished p c inv).products = (stepped p c inv).products := by
  simp only [finished]
  split <;> rfl

theorem finished_proposed : (finished p c inv).proposed = (start p c inv).proposed := by
  simp only [finished]
  split <;> rfl

theorem finished_seen : (finished p c inv).seen = (stepped p c inv).seen := by
  simp only [finished]
  split <;> rfl

theorem finished_segIdx : (finished p c inv).segIdx =
    c.segIdx + if (stepped p c inv).idx = p.n then 1 else 0 := by
  simp only [finished]
  split
  · next h => simp [stepped_segIdx]
  · next h => simp [stepped_segIdx]

theorem finished_memRoot : (finished p c inv).memRoot =
    if (stepped p c inv).idx = p.n then (start p c inv).proposed.2 else c.memRoot := by
  simp only [finished]
  split
  · next h => simp; rfl
  · next h => simp [stepped_memRoot]

/-- Every reachable carry is canonical. -/
theorem canonical (valid : p.Valid) (reach : Reach p c) : c.Canonical := by
  have := Plan.sMax_small valid
  have := Plan.n_lt valid
  have := Plan.wTs_small valid
  have := reach.seg
  have := reach.idx
  have := reach.ts
  refine ⟨?_, ?_, ?_⟩ <;> unfold goldilocksModulus at * <;> omega

/-! ### Witness words -/

variable (s : Machine.State)

theorem cIn_val {k : ℕ} (h : k < 39) : (witness p c inv s).cIn k = carryVector c ⟨k, h⟩ :=
  StepWitness.cIn_eq _ h

theorem cOut_val {k : ℕ} (h : k < 39) :
    (witness p c inv s).cOut k = carryVector (finished p c inv) ⟨k, h⟩ :=
  StepWitness.cOut_eq _ h

theorem openZero : ((witness p c inv s).cIn 1 - natWord p.n) * (witness p c inv s).isOpen = 0 := by
  rw [cIn_val s (by decide)]
  show (natWord c.idx - natWord p.n) * bitWord (decide (c.idx = p.n)) = 0
  by_cases h : c.idx = p.n
  · rw [h, sub_self, zero_mul]
  · simp [h, bitWord]

/-- A difference of two canonical counters is zero only when they are equal. -/
theorem natWord_sub_ne {a b : ℕ} (ha : a < goldilocksModulus) (hb : b < goldilocksModulus)
    (ne : a ≠ b) : natWord a - natWord b ≠ 0 :=
  fun zero => ne (natWord_injective ha hb (sub_eq_zero.mp zero))

theorem openTest (valid : p.Valid) (reach : Reach p c) :
    ((witness p c inv s).cIn 1 - natWord p.n) * (witness p c inv s).openInverse =
      1 - (witness p c inv s).isOpen := by
  rw [cIn_val s (by decide)]
  show (natWord c.idx - natWord p.n) * fieldInverse (natWord c.idx - natWord p.n) =
    1 - bitWord (decide (c.idx = p.n))
  by_cases h : c.idx = p.n
  · simp [h, bitWord]
  · rw [mul_fieldInverse (natWord_sub_ne (canonical valid reach).2.1 (Plan.n_lt valid) h)]
    simp [h, bitWord]

theorem segRange (accepted : Accepted p c inv) :
    (witness p c inv s).isOpen * (natWord (p.sMax - 1) - (witness p c inv s).cIn 0 -
      bitsWord (witness p c inv s).segBits) = 0 := by
  rw [cIn_val s (by decide)]
  show bitWord (decide (c.idx = p.n)) * (natWord (p.sMax - 1) - natWord c.segIdx -
    bitsWord (natBits p.segWidth (p.sMax - 1 - c.segIdx))) = 0
  by_cases h : c.idx = p.n
  · have below := accepted.segBelow h
    have width : p.sMax - 1 - c.segIdx < 2 ^ p.segWidth :=
      lt_of_le_of_lt (by omega) Nat.lt_log2_self
    have split : natWord (p.sMax - 1) = natWord c.segIdx + natWord (p.sMax - 1 - c.segIdx) := by
      rw [← natWord_add]
      congr 1
      omega
    rw [bitsWord_natBits width, split]
    ring
  · simp [h, bitWord]

theorem idxEff_row :
    (witness p c inv s).idxEff = (1 - (witness p c inv s).isOpen) * (witness p c inv s).cIn 1 := by
  rw [cIn_val s (by decide)]
  show natWord (start p c inv).idx = (1 - bitWord (decide (c.idx = p.n))) * natWord c.idx
  unfold start
  by_cases h : c.idx = p.n
  · simp [h, bitWord, openedCarry, natWord_zero]
  · simp [h, bitWord]


/-! ### Carry words by field -/

theorem carryVector_eta (d : MemoryCarry) (i : Fin 4) :
    carryVector d ⟨3 + i.val, by omega⟩ = ![d.eta.1.c0, d.eta.1.c1, d.eta.2.c0, d.eta.2.c1] i := by
  fin_cases i <;> rfl

theorem carryVector_proposed (d : MemoryCarry) (i : Fin 8) :
    carryVector d ⟨15 + i.val, by omega⟩ =
      if h : i.val < 4 then d.proposed.1 ⟨i.val, h⟩ else d.proposed.2 ⟨i.val - 4, by omega⟩ := by
  fin_cases i <;> rfl

theorem carryVector_seen (d : MemoryCarry) (i : Fin 12) :
    carryVector d ⟨23 + i.val, by omega⟩ =
      if h : i.val < 4 then d.seen.ops ⟨i.val, h⟩
      else if h' : i.val < 8 then d.seen.initial ⟨i.val - 4, by omega⟩
      else d.seen.final ⟨i.val - 8, by omega⟩ := by
  fin_cases i <;> rfl

theorem carryVector_proposedOps (d : MemoryCarry) (i : Fin 4) :
    carryVector d ⟨15 + i.val, by omega⟩ = d.proposed.1 i := by fin_cases i <;> rfl

theorem carryVector_proposedFinal (d : MemoryCarry) (i : Fin 4) :
    carryVector d ⟨19 + i.val, by omega⟩ = d.proposed.2 i := by fin_cases i <;> rfl

theorem carryVector_seenOps (d : MemoryCarry) (i : Fin 4) :
    carryVector d ⟨23 + i.val, by omega⟩ = d.seen.ops i := by fin_cases i <;> rfl

theorem carryVector_seenInitial (d : MemoryCarry) (i : Fin 4) :
    carryVector d ⟨27 + i.val, by omega⟩ = d.seen.initial i := by fin_cases i <;> rfl

theorem carryVector_seenFinal (d : MemoryCarry) (i : Fin 4) :
    carryVector d ⟨31 + i.val, by omega⟩ = d.seen.final i := by fin_cases i <;> rfl

theorem carryVector_memRoot (d : MemoryCarry) (i : Fin 4) :
    carryVector d ⟨35 + i.val, by omega⟩ = d.memRoot i := by fin_cases i <;> rfl

/-! ### Open-arm rows -/

theorem seenPrev_row (i : Fin 12) :
    (witness p c inv s).seenPrev i = (witness p c inv s).isOpen * headerWords p i +
      (1 - (witness p c inv s).isOpen) * (witness p c inv s).cIn (23 + i) := by
  rw [cIn_val s (by omega)]
  show carryVector (start p c inv) ⟨23 + i.val, _⟩ = bitWord (decide (c.idx = p.n)) *
    headerWords p i + (1 - bitWord (decide (c.idx = p.n))) * carryVector c ⟨23 + i.val, _⟩
  unfold start
  by_cases h : c.idx = p.n
  · simp only [h, if_true, decide_true, bitWord, one_mul, sub_self, zero_mul, add_zero]
    fin_cases i <;> rfl
  · simp [h, bitWord]

theorem proposed_row (i : Fin 8) :
    (witness p c inv s).cOut (15 + i) = (witness p c inv s).isOpen * (witness p c inv s).proposal i +
      (1 - (witness p c inv s).isOpen) * (witness p c inv s).cIn (15 + i) := by
  rw [cIn_val s (by omega), cOut_val s (by omega), carryVector_proposed, carryVector_proposed,
    finished_proposed]
  show (if h : i.val < 4 then (start p c inv).proposed.1 ⟨i.val, h⟩
      else (start p c inv).proposed.2 ⟨i.val - 4, by omega⟩) =
    bitWord (decide (c.idx = p.n)) *
      (if h : i.val < 4 then inv.proposal.1 ⟨i.val, h⟩ else inv.proposal.2 ⟨i.val - 4, by omega⟩) +
    (1 - bitWord (decide (c.idx = p.n))) *
      (if h : i.val < 4 then c.proposed.1 ⟨i.val, h⟩ else c.proposed.2 ⟨i.val - 4, by omega⟩)
  unfold start
  by_cases h : c.idx = p.n
  · simp [h, bitWord, openedCarry]
  · simp [h, bitWord]

theorem eta_row (i : Fin 4) :
    (witness p c inv s).cOut (3 + i) = (witness p c inv s).isOpen * (witness p c inv s).etaFresh i +
      (1 - (witness p c inv s).isOpen) * (witness p c inv s).cIn (3 + i) := by
  rw [cIn_val s (by omega), cOut_val s (by omega), carryVector_eta, carryVector_eta, finished_eta]
  show ![(start p c inv).eta.1.c0, (start p c inv).eta.1.c1, (start p c inv).eta.2.c0,
      (start p c inv).eta.2.c1] i =
    bitWord (decide (c.idx = p.n)) *
      ![(fresh p c inv).1.c0, (fresh p c inv).1.c1, (fresh p c inv).2.c0, (fresh p c inv).2.c1] i +
    (1 - bitWord (decide (c.idx = p.n))) * ![c.eta.1.c0, c.eta.1.c1, c.eta.2.c0, c.eta.2.c1] i
  unfold start
  by_cases h : c.idx = p.n
  · simp only [h, if_true, decide_true, bitWord, one_mul, sub_self, zero_mul, add_zero]
    rfl
  · simp [h, bitWord]

theorem square_row :
    (witness p c inv s).eta1Sq = K.mul (witness p c inv s).eta1 (witness p c inv s).eta1 := by
  rw [← GoldilocksExtensionRing.mul_eq]
  rfl


/-! ### Operation slots -/

theorem carryVector_ts (d : MemoryCarry) : carryVector d ⟨2, by decide⟩ = natWord d.ts := rfl

theorem opSlot_mem (accepted : Accepted p c inv) (j : Fin p.bOps) :
    opSlot inv j ∈ inv.records.ops := by
  have length := accepted.rows.shaped.opsLength
  unfold opSlot
  rw [List.getD_eq_getElem _ _ (by rw [length]; exact j.isLt)]
  exact List.getElem_mem _

theorem take_succ_ops (accepted : Accepted p c inv) {k : ℕ} (hk : k < p.bOps) :
    inv.records.ops.take (k + 1) = inv.records.ops.take k ++ [opSlot inv k] := by
  have length := accepted.rows.shaped.opsLength
  rw [List.take_add_one, opSlot, List.getD_eq_getElem _ _ (by omega),
    List.getElem?_eq_getElem (by omega)]
  rfl

/-- The active operations of a concatenation. -/
theorem activeOps_append_all (ts : ℕ) (l₁ l₂ : List OpSlot) :
    activeOps ts (l₁ ++ l₂) = activeOps ts l₁ ++ activeOps (ts + activeLength l₁) l₂ := by
  induction l₁ generalizing ts with
  | nil => simp [activeOps, activeLength]
  | cons s l₁ ih =>
    cases h : s.port with
    | none => simp only [List.cons_append, activeOps, h, ih, activeLength, List.filterMap_cons]
    | some a =>
      simp only [List.cons_append, activeOps, h, ih, activeLength, List.filterMap_cons,
        List.length_cons]
      rw [show ts + 1 + (List.filterMap OpSlot.port l₁).length =
        ts + ((List.filterMap OpSlot.port l₁).length + 1) by omega]

theorem activeLength_take_le (l : List OpSlot) (k : ℕ) : activeLength (l.take k) ≤ activeLength l :=
  ((List.take_sublist k l).filterMap _).length_le

/-- Row O2: the active count before slot `k`. -/
theorem cntBefore_val (accepted : Accepted p c inv) : ∀ k ≤ p.bOps,
    (witness p c inv s).cntBefore k = natWord (activeLength (inv.records.ops.take k))
  | 0, _ => by simp [StepWitness.cntBefore, activeLength, natWord_zero]
  | k + 1, hk => by
    have previous := cntBefore_val accepted k (by omega)
    unfold StepWitness.cntBefore at previous ⊢
    rw [StepWitness.take_succ_ofFn _ hk, List.sum_append, previous, take_succ_ops accepted hk,
      activeLength_append, natWord_add, List.sum_singleton]
    congr 1
    show 1 - bitWord (opSlot inv k).pad = natWord (if (opSlot inv k).pad then 0 else 1)
    cases (opSlot inv k).pad
    · simp [bitWord]; rfl
    · simp [bitWord]; rfl

theorem readKeeps_row (accepted : Accepted p c inv) (j : Fin p.bOps) :
    (1 - (witness p c inv s).isWrite j) *
      ((witness p c inv s).vw j - (witness p c inv s).vr j) = 0 := by
  show (1 - bitWord (opSlot inv j).isWrite) *
    (bitsWord (natBits 32 (opSlot inv j).vw) - bitsWord (natBits 32 (opSlot inv j).vr)) = 0
  have rows := accepted.rows.slots _ (opSlot_mem accepted j)
  cases h : (opSlot inv j).isWrite
  · rw [rows.readKeeps h, sub_self, mul_zero]
  · simp [bitWord]

theorem noRomWrite_row (accepted : Accepted p c inv) (j : Fin p.bOps) :
    (witness p c inv s).isWrite j * (1 - (witness p c inv s).isRam j) = 0 := by
  show bitWord (opSlot inv j).isWrite * (1 - bitWord (opSlot inv j).isRam) = 0
  have rows := accepted.rows.slots _ (opSlot_mem accepted j)
  cases h : (opSlot inv j).isWrite
  · simp [bitWord]
  · rw [rows.noRomWrite h]
    simp [bitWord]

theorem romRange_row (accepted : Accepted p c inv) (j : Fin p.bOps) (k : Fin p.μ)
    (high : p.r ≤ k.val) :
    (1 - (witness p c inv s).isRam j) * ((witness p c inv s).ops j).addr k = 0 := by
  show (1 - bitWord (opSlot inv j).isRam) * natBits p.μ (opSlot inv j).addr k = 0
  have rows := accepted.rows.slots _ (opSlot_mem accepted j)
  cases h : (opSlot inv j).isRam
  · have below : (opSlot inv j).addr < 2 ^ k.val :=
      lt_of_lt_of_le (rows.romRange h) (Nat.pow_le_pow_right (by norm_num) high)
    simp [natBits, Nat.testBit_lt_two_pow below, bitWord]
  · simp [bitWord]

theorem padSlot_tail (p : Plan) : ∀ b ∈ (OpSlot.padSlot.bits p).tail, b = false := by
  intro b member
  simp only [OpSlot.padSlot, OpSlot.bits, bitsLE, Nat.zero_testBit, List.cons_append,
    List.tail_cons, List.mem_cons, List.mem_append, List.mem_map, List.mem_range] at member
  aesop

theorem padZero_row (accepted : Accepted p c inv) (j : Fin p.bOps) :
    ∀ x ∈ ((witness p c inv s).ops j).lane.tail, (witness p c inv s).pad j * x = 0 := by
  intro x member
  change x ∈ (encodeOp p (opSlot inv j) (diff c inv j)).lane.tail at member
  show bitWord (opSlot inv j).pad * x = 0
  have rows := accepted.rows.slots _ (opSlot_mem accepted j)
  cases h : (opSlot inv j).pad
  · simp [bitWord]
  · rw [lane_encodeOp, rows.padZero h, ← List.map_tail] at member
    obtain ⟨b, inTail, rfl⟩ := List.mem_map.mp member
    rw [padSlot_tail p b inTail]
    simp [bitWord]

theorem fresh_row (accepted : Accepted p c inv) (j : Fin p.bOps) :
    (1 - (witness p c inv s).pad j) * ((witness p c inv s).wt j - (witness p c inv s).rt j - 1 -
      (witness p c inv s).diff j) = 0 := by
  show (1 - bitWord (opSlot inv j).pad) * ((witness p c inv s).cIn 2 +
    (witness p c inv s).cntBefore (j.val + 1) - bitsWord (natBits p.wTs (opSlot inv j).rt) - 1 -
      bitsWord (natBits p.wTs (diff c inv j))) = 0
  cases hpad : (opSlot inv j).pad
  · have fits := accepted.rows.shaped.opsFit _ (opSlot_mem accepted j)
    have member : (⟨⟨(opSlot inv j).isWrite, (opSlot inv j).isRam, (opSlot inv j).addr,
        (opSlot inv j).vr, (opSlot inv j).vw⟩, (opSlot inv j).rt,
        c.ts + activeLength (inv.records.ops.take j) + 1⟩ : MemOp) ∈
        activeOps c.ts inv.records.ops := by
      have split := activeOps_append_all c.ts (inv.records.ops.take (j.val + 1))
        (inv.records.ops.drop (j.val + 1))
      rw [List.take_append_drop] at split
      rw [split]
      apply List.mem_append_left
      rw [take_succ_ops accepted j.isLt, activeOps_append]
      apply List.mem_append_right
      simp [OpSlot.port, hpad]
    have before := accepted.rows.fresh _ member
    simp only at before
    have step : activeLength (inv.records.ops.take (j.val + 1)) =
        activeLength (inv.records.ops.take j) + 1 := by
      rw [take_succ_ops accepted j.isLt, activeLength_append, hpad]
      simp
    have stamp : writeStamp c inv j = c.ts + activeLength (inv.records.ops.take j) + 1 := by
      unfold writeStamp
      rw [step]
      omega
    have whole : activeLength (inv.records.ops.take (j.val + 1)) ≤ activeCount inv.records :=
      activeLength_take_le _ _
    have boundary := accepted.rows.boundary
    have diffSmall : diff c inv j < 2 ^ p.wTs := by
      unfold diff
      omega
    have sum : natWord c.ts + natWord (activeLength (inv.records.ops.take (j.val + 1))) =
        natWord (opSlot inv j).rt + 1 + natWord (diff c inv j) := by
      rw [← natWord_add, ← natWord_one, ← natWord_add, ← natWord_add]
      congr 1
      unfold diff
      omega
    rw [cIn_val s (by decide), carryVector_ts, cntBefore_val s accepted (j.val + 1) j.isLt,
      bitsWord_natBits fits.2.2.2, bitsWord_natBits diffSmall, sum]
    ring
  · simp [bitWord]


/-! ### Running products -/

theorem take_succ_getD {α : Type} (l : List α) (d : α) {k : ℕ} (hk : k < l.length) :
    l.take (k + 1) = l.take k ++ [l.getD k d] := by
  rw [List.take_add_one, List.getD_eq_getElem _ _ hk, List.getElem?_eq_getElem hk]
  rfl

theorem isOpen_open (h : c.idx = p.n) : (witness p c inv s).isOpen = 1 := by
  show bitWord (decide (c.idx = p.n)) = 1
  simp [h, bitWord]

theorem isOpen_continue (h : c.idx ≠ p.n) : (witness p c inv s).isOpen = 0 := by
  show bitWord (decide (c.idx = p.n)) = 0
  simp [h, bitWord]

/-- The products at the start of the step are the start carry's. -/
theorem startProducts :
    (witness p c inv s).startProduct 7 = (start p c inv).products.read ∧
      (witness p c inv s).startProduct 9 = (start p c inv).products.write ∧
      (witness p c inv s).startProduct 11 = (start p c inv).products.initial ∧
      (witness p c inv s).startProduct 13 = (start p c inv).products.final := by
  by_cases h : c.idx = p.n
  · simp only [StepWitness.startProduct_open _ _ (isOpen_open s h), start, if_pos h]
    exact ⟨rfl, rfl, rfl, rfl⟩
  · simp only [StepWitness.startProduct_continue _ _ (isOpen_continue s h), start, if_neg h]
    exact ⟨rfl, rfl, rfl, rfl⟩

theorem opsProductAfter_val : ∀ k ≤ p.bOps,
    (witness p c inv s).opsProductAfter 0 k = (opsAfter p c inv k).1 ∧
      (witness p c inv s).opsProductAfter 1 k = (opsAfter p c inv k).2
  | 0, _ => by
    simp only [StepWitness.opsProductAfter, if_true, opsAfter, List.take_zero, opsFactors,
      mul_one]
    exact ⟨(startProducts s).1, (startProducts s).2.1⟩
  | k + 1, hk => by
    simp only [StepWitness.opsProductAfter, Nat.add_one_ne_zero, if_false,
      Nat.add_sub_cancel, dif_pos (show k < p.bOps by omega)]
    exact ⟨rfl, rfl⟩

theorem scanProductAfter_val : ∀ k ≤ p.bScan,
    (witness p c inv s).scanProductAfter 0 k = (scanAfter p c inv k).1 ∧
      (witness p c inv s).scanProductAfter 1 k = (scanAfter p c inv k).2
  | 0, _ => by
    simp only [StepWitness.scanProductAfter, if_true, scanAfter, List.take_zero, scanFactor,
      mul_one]
    exact ⟨(startProducts s).2.2.1, (startProducts s).2.2.2⟩
  | k + 1, hk => by
    simp only [StepWitness.scanProductAfter, Nat.add_one_ne_zero, if_false,
      Nat.add_sub_cancel, dif_pos (show k < p.bScan by omega)]
    exact ⟨rfl, rfl⟩

theorem eta_pair : ((witness p c inv s).eta1, (witness p c inv s).eta2) = (start p c inv).eta := by
  rw [← finished_eta]
  rfl

theorem squareEq : (witness p c inv s).eta1Sq = (witness p c inv s).eta1 * (witness p c inv s).eta1 := by
  rw [square_row, GoldilocksExtensionRing.mul_eq]

theorem natWord_val_small {n k : ℕ} (small : n < 2 ^ k) (width : k ≤ 63) :
    (natWord n).val = n :=
  natWord_val (lt_trans small (two_pow_lt_modulus width))

theorem globalIndex_val (valid : p.Valid) (accepted : Accepted p c inv) (j : Fin p.bOps) :
    ((witness p c inv s).globalIndex j).val = PortAccess.globalIndex p
      ⟨(opSlot inv j).isWrite, (opSlot inv j).isRam, (opSlot inv j).addr, (opSlot inv j).vr,
        (opSlot inv j).vw⟩ := by
  have fits := accepted.rows.shaped.opsFit _ (opSlot_mem accepted j)
  show (bitsWord (natBits p.μ (opSlot inv j).addr) + bitWord (opSlot inv j).isRam *
    natWord p.romSize).val = _
  rw [bitsWord_natBits fits.1]
  have cells := valid.belowModulus
  have addr := fits.1
  unfold Plan.cells Plan.ramSize at cells
  cases (opSlot inv j).isRam
  · simp only [bitWord, Bool.false_eq_true, if_false, zero_mul, add_zero, PortAccess.globalIndex]
    exact natWord_val (by omega)
  · simp only [bitWord, if_true, one_mul, PortAccess.globalIndex, ← natWord_add]
    rw [natWord_val (by omega)]
    omega

theorem readProduct_row (valid : p.Valid) (accepted : Accepted p c inv) (j : Fin p.bOps) :
    (witness p c inv s).opsProductAfter 0 (j.val + 1) = K.mul
      ((witness p c inv s).opsProductAfter 0 j.val)
      (gatedK ((witness p c inv s).pad j) (fingerprintK (witness p c inv s).eta1
        (witness p c inv s).eta2 (witness p c inv s).eta1Sq ((witness p c inv s).rt j)
        ((witness p c inv s).globalIndex j) ((witness p c inv s).vr j))) := by
  have fits := accepted.rows.shaped.opsFit _ (opSlot_mem accepted j)
  rw [(opsProductAfter_val s (j.val + 1) j.isLt).1, (opsProductAfter_val s j.val j.isLt.le).1,
    ← GoldilocksExtensionRing.mul_eq,
    gatedK_bit (show IsBit ((witness p c inv s).pad j) from bitWord_isBit _),
    fingerprintK_eq _ _ _ (squareEq s), eta_pair, globalIndex_val s valid accepted j]
  show _ = _ * (if toBool (bitWord (opSlot inv j).pad) then 1 else fingerprint (start p c inv).eta
    ((bitsWord (natBits p.wTs (opSlot inv j).rt)).val, _, (bitsWord (natBits 32 (opSlot inv j).vr)).val))
  rw [toBool_bitWord, bitsWord_natBits fits.2.2.2, bitsWord_natBits fits.2.1,
    natWord_val_small fits.2.2.2 (by have := valid.fieldEncoding; omega),
    natWord_val_small fits.2.1 (by norm_num)]
  simp only [opsAfter]
  rw [take_succ_ops accepted j.isLt, opsFactors_append, mul_assoc]

theorem writeProduct_row (valid : p.Valid) (accepted : Accepted p c inv) (j : Fin p.bOps) :
    (witness p c inv s).opsProductAfter 1 (j.val + 1) = K.mul
      ((witness p c inv s).opsProductAfter 1 j.val)
      (gatedK ((witness p c inv s).pad j) (fingerprintK (witness p c inv s).eta1
        (witness p c inv s).eta2 (witness p c inv s).eta1Sq ((witness p c inv s).wt j)
        ((witness p c inv s).globalIndex j) ((witness p c inv s).vw j))) := by
  have fits := accepted.rows.shaped.opsFit _ (opSlot_mem accepted j)
  rw [(opsProductAfter_val s (j.val + 1) j.isLt).2, (opsProductAfter_val s j.val j.isLt.le).2,
    ← GoldilocksExtensionRing.mul_eq,
    gatedK_bit (show IsBit ((witness p c inv s).pad j) from bitWord_isBit _),
    fingerprintK_eq _ _ _ (squareEq s), eta_pair, globalIndex_val s valid accepted j]
  show _ = _ * (if toBool (bitWord (opSlot inv j).pad) then 1 else fingerprint (start p c inv).eta
    (((witness p c inv s).cIn 2 + (witness p c inv s).cntBefore (j.val + 1)).val, _,
      (bitsWord (natBits 32 (opSlot inv j).vw)).val))
  rw [toBool_bitWord, bitsWord_natBits fits.2.2.1,
    natWord_val_small fits.2.2.1 (by norm_num)]
  simp only [opsAfter]
  rw [take_succ_ops accepted j.isLt, opsFactors_append, mul_assoc]
  cases hpad : (opSlot inv j).pad
  · have boundary := accepted.rows.boundary
    have whole : activeLength (inv.records.ops.take (j.val + 1)) ≤ activeCount inv.records :=
      activeLength_take_le _ _
    have step : activeLength (inv.records.ops.take (j.val + 1)) =
        activeLength (inv.records.ops.take j) + 1 := by
      rw [take_succ_ops accepted j.isLt, activeLength_append, hpad]
      simp
    rw [cIn_val s (by decide), carryVector_ts, cntBefore_val s accepted (j.val + 1) j.isLt,
      ← natWord_add, step,
      natWord_val_small (k := p.wTs) (by omega) (by have := valid.fieldEncoding; omega)]
    simp only [Bool.false_eq_true, if_false]
    rw [Nat.add_assoc]
  · simp only [↓reduceIte]

theorem take_scan_length {l : List ScanSlot} {k : ℕ} (le : k ≤ l.length) : (l.take k).length = k := by
  simp [le]

theorem scanIndex_val (valid : p.Valid) (reach : Reach p c) (j : Fin p.bScan) :
    ((witness p c inv s).scanIndex j).val = (start p c inv).idx * p.bScan + j.val := by
  have idxLt := start_idx_lt (inv := inv) valid reach
  have cover := valid.exactCover
  have below := valid.belowModulus
  have bound : (start p c inv).idx * p.bScan + j.val < goldilocksModulus := by
    have : (start p c inv).idx * p.bScan + j.val < p.n * p.bScan := by
      calc (start p c inv).idx * p.bScan + j.val < (start p c inv).idx * p.bScan + p.bScan := by
            omega
        _ = ((start p c inv).idx + 1) * p.bScan := by ring
        _ ≤ p.n * p.bScan := Nat.mul_le_mul_right _ idxLt
    omega
  show (natWord (start p c inv).idx * natWord p.bScan + natWord j.val).val = _
  rw [← natWord_mul, ← natWord_add, natWord_val bound]

theorem initialProduct_row (valid : p.Valid) (reach : Reach p c) (accepted : Accepted p c inv)
    (j : Fin p.bScan) :
    (witness p c inv s).scanProductAfter 0 (j.val + 1) = K.mul
      ((witness p c inv s).scanProductAfter 0 j.val)
      (fingerprintK (witness p c inv s).eta1 (witness p c inv s).eta2 (witness p c inv s).eta1Sq
        (bitsWord ((witness p c inv s).initial j).stamp) ((witness p c inv s).scanIndex j)
        (bitsWord ((witness p c inv s).initial j).value)) := by
  have length := accepted.rows.shaped.initialLength
  have member : initialSlot inv j ∈ inv.records.initialScan := by
    unfold initialSlot
    rw [List.getD_eq_getElem _ _ (by omega)]
    exact List.getElem_mem _
  have fits := accepted.rows.shaped.initialFit _ member
  rw [(scanProductAfter_val s (j.val + 1) j.isLt).1, (scanProductAfter_val s j.val j.isLt.le).1,
    ← GoldilocksExtensionRing.mul_eq, fingerprintK_eq _ _ _ (squareEq s), eta_pair,
    scanIndex_val s valid reach]
  show _ = _ * fingerprint (start p c inv).eta
    ((bitsWord (natBits p.wTs (initialSlot inv j).stamp)).val, _,
      (bitsWord (natBits 32 (initialSlot inv j).value)).val)
  rw [bitsWord_natBits fits.1, bitsWord_natBits fits.2,
    natWord_val_small fits.1 (by norm_num),
    natWord_val_small fits.2 (by have := valid.fieldEncoding; omega)]
  simp only [scanAfter]
  rw [take_succ_getD _ ⟨0, 0⟩ (by omega), scanFactor_append, mul_assoc,
    take_scan_length (by omega)]
  rfl

theorem finalProduct_row (valid : p.Valid) (reach : Reach p c) (accepted : Accepted p c inv)
    (j : Fin p.bScan) :
    (witness p c inv s).scanProductAfter 1 (j.val + 1) = K.mul
      ((witness p c inv s).scanProductAfter 1 j.val)
      (fingerprintK (witness p c inv s).eta1 (witness p c inv s).eta2 (witness p c inv s).eta1Sq
        (bitsWord ((witness p c inv s).final j).stamp) ((witness p c inv s).scanIndex j)
        (bitsWord ((witness p c inv s).final j).value)) := by
  have length := accepted.rows.shaped.finalLength
  have member : finalSlot inv j ∈ inv.records.finalScan := by
    unfold finalSlot
    rw [List.getD_eq_getElem _ _ (by omega)]
    exact List.getElem_mem _
  have fits := accepted.rows.shaped.finalFit _ member
  rw [(scanProductAfter_val s (j.val + 1) j.isLt).2, (scanProductAfter_val s j.val j.isLt.le).2,
    ← GoldilocksExtensionRing.mul_eq, fingerprintK_eq _ _ _ (squareEq s), eta_pair,
    scanIndex_val s valid reach]
  show _ = _ * fingerprint (start p c inv).eta
    ((bitsWord (natBits p.wTs (finalSlot inv j).stamp)).val, _,
      (bitsWord (natBits 32 (finalSlot inv j).value)).val)
  rw [bitsWord_natBits fits.1, bitsWord_natBits fits.2,
    natWord_val_small fits.1 (by norm_num),
    natWord_val_small fits.2 (by have := valid.fieldEncoding; omega)]
  simp only [scanAfter]
  rw [take_succ_getD _ ⟨0, 0⟩ (by omega), scanFactor_append, mul_assoc,
    take_scan_length (by omega)]
  rfl


/-! ### Boundary and close rows -/

theorem tsOut_row (accepted : Accepted p c inv) :
    (witness p c inv s).cOut 2 = (witness p c inv s).cIn 2 + (witness p c inv s).cntBefore p.bOps := by
  rw [cOut_val s (by decide), cIn_val s (by decide), carryVector_ts, carryVector_ts,
    cntBefore_val s accepted p.bOps le_rfl, finished_ts, stepped_ts, ← natWord_add,
    List.take_of_length_le accepted.rows.shaped.opsLength.le]
  rfl

theorem tsRange_row (accepted : Accepted p c inv) :
    (witness p c inv s).cOut 2 = bitsWord (witness p c inv s).tsBits := by
  have small : (stepped p c inv).ts < 2 ^ p.wTs := by
    rw [stepped_ts]
    exact accepted.rows.boundary
  rw [cOut_val s (by decide), carryVector_ts, finished_ts]
  exact (bitsWord_natBits small).symm

theorem idxOut_row : (witness p c inv s).cOut 1 = (witness p c inv s).idxEff + 1 := by
  rw [cOut_val s (by decide)]
  show natWord (finished p c inv).idx = natWord (start p c inv).idx + 1
  rw [finished_idx, ← natWord_one, ← natWord_add]
  rfl

theorem productsOut_row (accepted : Accepted p c inv) :
    kOf ((witness p c inv s).cOut 7) ((witness p c inv s).cOut 8) =
        (witness p c inv s).opsProductAfter 0 p.bOps ∧
      kOf ((witness p c inv s).cOut 9) ((witness p c inv s).cOut 10) =
        (witness p c inv s).opsProductAfter 1 p.bOps ∧
      kOf ((witness p c inv s).cOut 11) ((witness p c inv s).cOut 12) =
        (witness p c inv s).scanProductAfter 0 p.bScan ∧
      kOf ((witness p c inv s).cOut 13) ((witness p c inv s).cOut 14) =
        (witness p c inv s).scanProductAfter 1 p.bScan := by
  obtain ⟨read, write⟩ := opsProductAfter_val s p.bOps le_rfl
  obtain ⟨initial, final⟩ := scanProductAfter_val s p.bScan le_rfl
  rw [read, write, initial, final]
  simp only [opsAfter, scanAfter, List.take_of_length_le accepted.rows.shaped.opsLength.le,
    List.take_of_length_le accepted.rows.shaped.initialLength.le,
    List.take_of_length_le accepted.rows.shaped.finalLength.le]
  refine ⟨?_, ?_, ?_, ?_⟩
  · show (finished p c inv).products.read = _
    rw [finished_products]
    show (start p c inv).products.read *
      (opsFactors p (start p c inv).eta (start p c inv).ts inv.records.ops).1 = _
    rw [start_ts]
  · show (finished p c inv).products.write = _
    rw [finished_products]
    show (start p c inv).products.write *
      (opsFactors p (start p c inv).eta (start p c inv).ts inv.records.ops).2 = _
    rw [start_ts]
  · show (finished p c inv).products.initial = _
    rw [finished_products]
    rfl
  · show (finished p c inv).products.final = _
    rw [finished_products]
    rfl

theorem closeZero_row :
    ((witness p c inv s).cOut 1 - natWord p.n) * (witness p c inv s).isClose = 0 := by
  rw [cOut_val s (by decide)]
  show (natWord (finished p c inv).idx - natWord p.n) *
    bitWord (decide ((stepped p c inv).idx = p.n)) = 0
  rw [finished_idx]
  by_cases h : (stepped p c inv).idx = p.n
  · rw [h, sub_self, zero_mul]
  · simp [h, bitWord]

theorem closeTest_row (valid : p.Valid) (reach : Reach p c) :
    ((witness p c inv s).cOut 1 - natWord p.n) * (witness p c inv s).closeInverse =
      1 - (witness p c inv s).isClose := by
  rw [cOut_val s (by decide)]
  show (natWord (finished p c inv).idx - natWord p.n) *
      fieldInverse (natWord (stepped p c inv).idx - natWord p.n) =
    1 - bitWord (decide ((stepped p c inv).idx = p.n))
  rw [finished_idx]
  by_cases h : (stepped p c inv).idx = p.n
  · simp [h, bitWord]
  · have small : (stepped p c inv).idx < goldilocksModulus := by
      have := start_idx_lt (inv := inv) valid reach
      have := Plan.n_lt valid
      show (start p c inv).idx + 1 < _
      omega
    rw [mul_fieldInverse (natWord_sub_ne small (Plan.n_lt valid) h)]
    simp [h, bitWord]

theorem closeOps_row (accepted : Accepted p c inv) (i : Fin 4) :
    (witness p c inv s).isClose * ((witness p c inv s).cOut (23 + i) -
      (witness p c inv s).cOut (15 + i)) = 0 := by
  rw [cOut_val s (by omega), cOut_val s (by omega), carryVector_seenOps, carryVector_proposedOps,
    finished_seen, finished_proposed]
  show bitWord (decide ((stepped p c inv).idx = p.n)) * _ = 0
  by_cases h : (stepped p c inv).idx = p.n
  · have eq : (stepped p c inv).seen.ops = (start p c inv).proposed.1 := (accepted.closes h).1
    rw [eq, sub_self, mul_zero]
  · simp [h, bitWord]

theorem closeInitial_row (accepted : Accepted p c inv) (i : Fin 4) :
    (witness p c inv s).isClose * ((witness p c inv s).cOut (27 + i) -
      (witness p c inv s).cIn (35 + i)) = 0 := by
  rw [cOut_val s (by omega), cIn_val s (by omega), carryVector_seenInitial, carryVector_memRoot,
    finished_seen]
  show bitWord (decide ((stepped p c inv).idx = p.n)) * _ = 0
  by_cases h : (stepped p c inv).idx = p.n
  · rw [(accepted.closes h).2.1, stepped_memRoot, sub_self, mul_zero]
  · simp [h, bitWord]

theorem closeFinal_row (accepted : Accepted p c inv) (i : Fin 4) :
    (witness p c inv s).isClose * ((witness p c inv s).cOut (31 + i) -
      (witness p c inv s).cOut (19 + i)) = 0 := by
  rw [cOut_val s (by omega), cOut_val s (by omega), carryVector_seenFinal,
    carryVector_proposedFinal, finished_seen, finished_proposed]
  show bitWord (decide ((stepped p c inv).idx = p.n)) * _ = 0
  by_cases h : (stepped p c inv).idx = p.n
  · have eq : (stepped p c inv).seen.final = (start p c inv).proposed.2 := (accepted.closes h).2.2.1
    rw [eq, sub_self, mul_zero]
  · simp [h, bitWord]

theorem closeProducts_row (accepted : Accepted p c inv) :
    K.mul (embed (witness p c inv s).isClose)
      (K.sub (K.mul (kOf ((witness p c inv s).cOut 11) ((witness p c inv s).cOut 12))
          (kOf ((witness p c inv s).cOut 9) ((witness p c inv s).cOut 10)))
        (K.mul (kOf ((witness p c inv s).cOut 7) ((witness p c inv s).cOut 8))
          (kOf ((witness p c inv s).cOut 13) ((witness p c inv s).cOut 14)))) = K.zero := by
  show K.mul (embed (bitWord (decide ((stepped p c inv).idx = p.n))))
    (K.sub (K.mul (finished p c inv).products.initial (finished p c inv).products.write)
      (K.mul (finished p c inv).products.read (finished p c inv).products.final)) = K.zero
  simp only [← GoldilocksExtensionRing.mul_eq, ← GoldilocksExtensionRing.sub_eq,
    ← GoldilocksExtensionRing.zero_eq, finished_products]
  by_cases h : (stepped p c inv).idx = p.n
  · rw [(accepted.closes h).2.2.2, sub_self, mul_zero]
  · have zero : embed (bitWord (decide ((stepped p c inv).idx = p.n))) = 0 := by
      simp [h, bitWord, embed]
      rfl
    rw [zero, zero_mul]

theorem segOut_row :
    (witness p c inv s).cOut 0 = (witness p c inv s).cIn 0 + (witness p c inv s).isClose := by
  rw [cOut_val s (by decide), cIn_val s (by decide)]
  show natWord (finished p c inv).segIdx = natWord c.segIdx +
    bitWord (decide ((stepped p c inv).idx = p.n))
  rw [finished_segIdx]
  by_cases h : (stepped p c inv).idx = p.n
  · simp [h, bitWord, natWord_add, natWord_one]
  · simp [h, bitWord]

theorem memOut_row (i : Fin 4) :
    (witness p c inv s).cOut (35 + i) = (witness p c inv s).isClose *
      (witness p c inv s).cOut (19 + i) +
        (1 - (witness p c inv s).isClose) * (witness p c inv s).cIn (35 + i) := by
  rw [cOut_val s (by omega), cOut_val s (by omega), cIn_val s (by omega), carryVector_memRoot,
    carryVector_memRoot, carryVector_proposedFinal, finished_memRoot, finished_proposed]
  show _ = bitWord (decide ((stepped p c inv).idx = p.n)) * _ +
    (1 - bitWord (decide ((stepped p c inv).idx = p.n))) * _
  by_cases h : (stepped p c inv).idx = p.n
  · simp [h, bitWord]
  · simp [h, bitWord]

/-- Ob5 for the polynomial rows: the honest witness satisfies every row of
`StepWitness.PolyRows`. -/
theorem polyRows (valid : p.Valid) (reach : Reach p c) (accepted : Accepted p c inv) :
    (witness p c inv s).PolyRows where
  opsBits j := by
    show ∀ x ∈ (encodeOp p (opSlot inv j) (diff c inv j)).lane, IsBit x
    rw [lane_encodeOp]
    exact bitWord_bits
  diffBits _ _ := bitWord_isBit _
  initialBits j := by
    show ∀ x ∈ (encodeScan p (initialSlot inv j)).lane, IsBit x
    rw [lane_encodeScan]
    exact bitWord_bits
  finalBits j := by
    show ∀ x ∈ (encodeScan p (finalSlot inv j)).lane, IsBit x
    rw [lane_encodeScan]
    exact bitWord_bits
  tsBits _ := bitWord_isBit _
  segBits _ := bitWord_isBit _
  idleBit := bitWord_isBit _
  openBit := bitWord_isBit _
  closeBit := bitWord_isBit _
  openZero := openZero s
  openTest := openTest s valid reach
  segRange := segRange s accepted
  idxEff := idxEff_row s
  seenPrev := seenPrev_row s
  proposed := proposed_row s
  eta := eta_row s
  square := square_row s
  readKeeps := readKeeps_row s accepted
  fresh := fresh_row s accepted
  noRomWrite := noRomWrite_row s accepted
  romRange := romRange_row s accepted
  padZero := padZero_row s accepted
  readProduct := readProduct_row s valid accepted
  writeProduct := writeProduct_row s valid accepted
  initialProduct := initialProduct_row s valid reach accepted
  finalProduct := finalProduct_row s valid reach accepted
  tsOut := tsOut_row s accepted
  tsRange := tsRange_row s accepted
  idxOut := idxOut_row s
  productsOut := productsOut_row s accepted
  closeZero := closeZero_row s
  closeTest := closeTest_row s valid reach
  closeOps := closeOps_row s accepted
  closeInitial := closeInitial_row s accepted
  closeFinal := closeFinal_row s accepted
  closeProducts := closeProducts_row s accepted
  segOut := segOut_row s
  memOut := memOut_row s

end NightstreamFPrime.Lifecycle.Nebula.Honest
