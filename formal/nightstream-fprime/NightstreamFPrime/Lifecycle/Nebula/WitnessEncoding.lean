import NightstreamFPrime.Lifecycle.Nebula.MemoryCircuit

/-! Owns the inverse of the witness word layout: the word vector of a step
witness, which `MemoryApp.decode` reads back as the same witness. Completeness
of the relation (Ob5) uses it to turn an honest step witness into witness
words. It does not own any row. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula

variable {p : Plan}

/-! ### Slot words -/

/-- Word `i` of an operation slot: the lane bits of spec §6.3, then the O4
difference bits; `0` past the slot. -/
def OpSlotBits.word (slot : OpSlotBits p) (i : ℕ) : F :=
  if i = 0 then slot.pad
  else if i = 1 then slot.isWrite
  else if i = 2 then slot.isRam
  else if h : i - 3 < p.μ then slot.addr ⟨i - 3, h⟩
  else if h : i - 3 - p.μ < 32 then slot.vr ⟨i - 3 - p.μ, h⟩
  else if h : i - 3 - p.μ - 32 < 32 then slot.vw ⟨i - 3 - p.μ - 32, h⟩
  else if h : i - 3 - p.μ - 64 < p.wTs then slot.rt ⟨i - 3 - p.μ - 64, h⟩
  else if h : i - p.opWidth < p.wTs then slot.diff ⟨i - p.opWidth, h⟩
  else 0

/-- Word `i` of a scan slot: the value bits, then the stamp bits. -/
def ScanSlotBits.word (slot : ScanSlotBits p) (i : ℕ) : F :=
  if h : i < 32 then slot.value ⟨i, h⟩
  else if h : i - 32 < p.wTs then slot.stamp ⟨i - 32, h⟩
  else 0

theorem OpSlotBits.ofWords_word (slot : OpSlotBits p) (v : ℕ → F) (j : ℕ)
    (words : ∀ k < Words.opWords p, v (Words.op p j k) = slot.word k) :
    OpSlotBits.ofWords p v j = slot := by
  have opWords : Words.opWords p = 3 + p.μ + 64 + p.wTs + p.wTs := by
    simp [Words.opWords, Plan.opWidth]
  cases slot with
  | mk pad isWrite isRam addr vr vw rt diff =>
    simp only [OpSlotBits.ofWords, OpSlotBits.mk.injEq]
    refine ⟨?_, ?_, ?_, funext fun k => ?_, funext fun k => ?_, funext fun k => ?_,
      funext fun k => ?_, funext fun k => ?_⟩
    · rw [words _ (by simp [Words.opPad]; omega)]; simp [OpSlotBits.word, Words.opPad]
    · rw [words _ (by simp [Words.opIsWrite]; omega)]; simp [OpSlotBits.word, Words.opIsWrite]
    · rw [words _ (by simp [Words.opIsRam]; omega)]; simp [OpSlotBits.word, Words.opIsRam]
    · have := k.isLt
      rw [words _ (by simp [Words.opAddr]; omega)]
      simp only [OpSlotBits.word, Words.opAddr]
      rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), dif_pos (by omega)]
      congr 1; ext; simp only; omega
    · have := k.isLt
      rw [words _ (by simp [Words.opVr]; omega)]
      simp only [OpSlotBits.word, Words.opVr]
      rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), dif_neg (by omega),
        dif_pos (by omega)]
      congr 1; ext; simp only; omega
    · have := k.isLt
      rw [words _ (by simp [Words.opVw]; omega)]
      simp only [OpSlotBits.word, Words.opVw]
      rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), dif_neg (by omega),
        dif_neg (by omega), dif_pos (by omega)]
      congr 1; ext; simp only; omega
    · have := k.isLt
      rw [words _ (by simp [Words.opRt]; omega)]
      simp only [OpSlotBits.word, Words.opRt]
      rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), dif_neg (by omega),
        dif_neg (by omega), dif_neg (by omega), dif_pos (by omega)]
      congr 1; ext; simp only; omega
    · have := k.isLt
      rw [words _ (by simp [Words.opDiff, Plan.opWidth]; omega)]
      simp only [OpSlotBits.word, Words.opDiff, Plan.opWidth]
      rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), dif_neg (by omega),
        dif_neg (by omega), dif_neg (by omega), dif_neg (by omega), dif_pos (by omega)]
      congr 1; ext; simp only; omega

theorem ScanSlotBits.ofWords_word (slot : ScanSlotBits p) (v : ℕ → F) (base : ℕ)
    (words : ∀ k < p.scanWidth, v (base + k) = slot.word k) :
    ScanSlotBits.ofWords p v base = slot := by
  cases slot with
  | mk value stamp =>
    simp only [ScanSlotBits.ofWords, ScanSlotBits.mk.injEq]
    refine ⟨funext fun k => ?_, funext fun k => ?_⟩
    · have := k.isLt
      rw [words _ (by simp only [Words.scanValue, Plan.scanWidth]; omega)]
      simp only [ScanSlotBits.word, Words.scanValue]
      rw [dif_pos (by omega)]
    · have := k.isLt
      rw [words _ (by simp only [Words.scanStamp, Plan.scanWidth]; omega)]
      simp only [ScanSlotBits.word, Words.scanStamp]
      rw [dif_neg (by omega), dif_pos (by omega)]
      congr 1; ext; simp only; omega

/-! ### Slots of a word range -/

/-- The slot and the offset of word `k` in `count` slots of `width` words that
start at word `start`, or `none` outside them. -/
def slotAt (start width count k : ℕ) : Option (Fin count × ℕ) :=
  if h : start ≤ k ∧ (k - start) / width < count then
    some (⟨(k - start) / width, h.2⟩, (k - start) % width)
  else none

theorem slotAt_slot {start width count j i : ℕ} (hj : j < count) (hi : i < width) :
    slotAt start width count (start + j * width + i) = some (⟨j, hj⟩, i) := by
  have pos : 0 < width := by omega
  have sub : start + j * width + i - start = i + j * width := by omega
  have div : (i + j * width) / width = j := by
    rw [Nat.add_mul_div_right _ _ pos, Nat.div_eq_of_lt hi, Nat.zero_add]
  have mod : (i + j * width) % width = i := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]
  unfold slotAt
  rw [dif_pos ⟨by omega, by rw [sub, div]; exact hj⟩]
  simp only [sub, div, mod]

theorem slotAt_above {start width count k : ℕ} (pos : 0 < width)
    (above : start + count * width ≤ k) : slotAt start width count k = none := by
  unfold slotAt
  rw [dif_neg]
  rintro ⟨-, inside⟩
  have := (Nat.div_lt_iff_lt_mul pos).mp inside
  omega

/-- Word `k` of slot `j` lies below the end of `count` slots. -/
theorem slot_lt {j count width k : ℕ} (hj : j < count) (hk : k < width) :
    j * width + k < count * width :=
  calc j * width + k < j * width + width := by omega
    _ = (j + 1) * width := by ring
    _ ≤ count * width := Nat.mul_le_mul_right _ hj

namespace Words

theorem opWords_pos : 0 < Words.opWords p := by simp [Words.opWords, Plan.opWidth]

theorem scanWidth_pos : 0 < p.scanWidth := by simp [Plan.scanWidth]

theorem count_eq : Words.count p = 114 + p.bOps * Words.opWords p + p.bScan * p.scanWidth +
    p.bScan * p.scanWidth + p.wTs + p.segWidth + 4 * p.bOps + 4 * p.bScan := by
  simp only [Words.count, Words.scanProducts, Words.opsProducts, Words.segBits, Words.tsBits,
    Words.finalStart, Words.initialStart]
  omega

end Words

/-! ### The word vector -/

/-- The witness words of a step witness in the `Words` layout; `0` past
`Words.count p`. -/
def StepWitness.wordAt (w : StepWitness p) (k : ℕ) : F :=
  if h : k < 2 then w.appIn ⟨k, h⟩
  else if h : k < 41 then w.carryIn ⟨k - 2, by omega⟩
  else if h : k < 80 then w.carryOut ⟨k - 41, by omega⟩
  else if h : k < 82 then w.appOut ⟨k - 80, by omega⟩
  else if h : k < 90 then w.proposal ⟨k - 82, by omega⟩
  else if k = 90 then w.idle
  else if k = 91 then w.isOpen
  else if k = 92 then w.openInverse
  else if k = 93 then w.isClose
  else if k = 94 then w.closeInverse
  else if h : k < 99 then w.etaFresh ⟨k - 95, by omega⟩
  else if h : k < 101 then w.eta1Square ⟨k - 99, by omega⟩
  else if h : k < 113 then w.seenPrev ⟨k - 101, by omega⟩
  else if k = 113 then w.idxEff
  else match slotAt 114 (Words.opWords p) p.bOps k with
  | some (j, i) => (w.ops j).word i
  | none => match slotAt (Words.initialStart p) p.scanWidth p.bScan k with
  | some (j, i) => (w.initial j).word i
  | none => match slotAt (Words.finalStart p) p.scanWidth p.bScan k with
  | some (j, i) => (w.final j).word i
  | none =>
    if h : Words.tsBits p 0 ≤ k ∧ k < Words.segBits p 0 then
      w.tsBits ⟨k - Words.tsBits p 0, by simp only [Words.segBits, Words.tsBits] at h ⊢; omega⟩
    else if h : Words.segBits p 0 ≤ k ∧ k < Words.opsProducts p 0 0 then
      w.segBits ⟨k - Words.segBits p 0, by simp only [Words.opsProducts, Words.segBits] at h ⊢; omega⟩
    else if h : Words.opsProducts p 0 0 ≤ k ∧ k < Words.scanProducts p 0 0 then
      w.opsProducts ⟨(k - Words.opsProducts p 0 0) / 4,
          by simp only [Words.scanProducts, Words.opsProducts] at h ⊢; omega⟩
        ⟨(k - Words.opsProducts p 0 0) % 4, Nat.mod_lt _ (by decide)⟩
    else if h : Words.scanProducts p 0 0 ≤ k ∧ k < Words.count p then
      w.scanProducts ⟨(k - Words.scanProducts p 0 0) / 4,
          by simp only [Words.count, Words.scanProducts] at h ⊢; omega⟩
        ⟨(k - Words.scanProducts p 0 0) % 4, Nat.mod_lt _ (by decide)⟩
    else 0

/-- The witness word list of a step witness. -/
def StepWitness.words (w : StepWitness p) : List F :=
  List.ofFn fun k : Fin (Words.count p) => w.wordAt k

namespace StepWitness

variable (w : StepWitness p)

/-- Evaluates the branch tests of `wordAt` with `omega`. -/
macro "read_word" : tactic =>
  `(tactic| simp (disch := omega) only [StepWitness.wordAt, dif_pos, dif_neg, if_pos, if_neg,
    ↓reduceIte, ↓reduceDIte, Fin.eta, Nat.add_sub_cancel_left])

theorem wordAt_appIn (i : Fin 2) : w.wordAt (Words.appIn i) = w.appIn i := by
  have := i.isLt; simp only [Words.appIn]; read_word

theorem wordAt_carryIn (i : Fin 39) : w.wordAt (Words.carryIn i) = w.carryIn i := by
  have := i.isLt; simp only [Words.carryIn]; read_word

theorem wordAt_carryOut (i : Fin 39) : w.wordAt (Words.carryOut i) = w.carryOut i := by
  have := i.isLt; simp only [Words.carryOut]; read_word

theorem wordAt_appOut (i : Fin 2) : w.wordAt (Words.appOut i) = w.appOut i := by
  have := i.isLt; simp only [Words.appOut]; read_word

theorem wordAt_proposal (i : Fin 8) : w.wordAt (Words.proposal i) = w.proposal i := by
  have := i.isLt; simp only [Words.proposal]; read_word

theorem wordAt_idle : w.wordAt Words.idle = w.idle := by simp only [Words.idle]; read_word

theorem wordAt_isOpen : w.wordAt Words.isOpen = w.isOpen := by
  simp only [Words.isOpen]; read_word

theorem wordAt_openInverse : w.wordAt Words.openInverse = w.openInverse := by
  simp only [Words.openInverse]; read_word

theorem wordAt_isClose : w.wordAt Words.isClose = w.isClose := by
  simp only [Words.isClose]; read_word

theorem wordAt_closeInverse : w.wordAt Words.closeInverse = w.closeInverse := by
  simp only [Words.closeInverse]; read_word

theorem wordAt_etaFresh (i : Fin 4) : w.wordAt (Words.etaFresh i) = w.etaFresh i := by
  have := i.isLt; simp only [Words.etaFresh]; read_word

theorem wordAt_eta1Square (i : Fin 2) : w.wordAt (Words.eta1Square i) = w.eta1Square i := by
  have := i.isLt; simp only [Words.eta1Square]; read_word

theorem wordAt_seenPrev (i : Fin 12) : w.wordAt (Words.seenPrev i) = w.seenPrev i := by
  have := i.isLt; simp only [Words.seenPrev]; read_word

theorem wordAt_idxEff : w.wordAt Words.idxEff = w.idxEff := by
  simp only [Words.idxEff]; read_word

theorem wordAt_op (j : Fin p.bOps) {k : ℕ} (hk : k < Words.opWords p) :
    w.wordAt (Words.op p j k) = (w.ops j).word k := by
  simp only [Words.op]
  simp (disch := omega) only [StepWitness.wordAt, dif_neg, if_neg, slotAt_slot j.isLt hk, Fin.eta]

theorem wordAt_initial (j : Fin p.bScan) {k : ℕ} (hk : k < p.scanWidth) :
    w.wordAt (Words.initial p j k) = (w.initial j).word k := by
  have := Words.opWords_pos (p := p)
  have start : Words.initialStart p = 114 + p.bOps * Words.opWords p := rfl
  simp only [Words.initial]
  simp (disch := omega) only [StepWitness.wordAt, dif_neg, if_neg, slotAt_above,
    slotAt_slot j.isLt hk, Fin.eta]

theorem wordAt_final (j : Fin p.bScan) {k : ℕ} (hk : k < p.scanWidth) :
    w.wordAt (Words.final p j k) = (w.final j).word k := by
  have := Words.opWords_pos (p := p)
  have := Words.scanWidth_pos (p := p)
  have start : Words.initialStart p = 114 + p.bOps * Words.opWords p := rfl
  have final : Words.finalStart p = Words.initialStart p + p.bScan * p.scanWidth := rfl
  simp only [Words.final]
  simp (disch := omega) only [StepWitness.wordAt, dif_neg, if_neg, slotAt_above,
    slotAt_slot j.isLt hk, Fin.eta]

/-- Past the three slot ranges, `wordAt` is the counter and product cascade. -/
theorem wordAt_tail {k : ℕ} (past : Words.tsBits p 0 ≤ k) :
    w.wordAt k =
      if h : Words.tsBits p 0 ≤ k ∧ k < Words.segBits p 0 then
        w.tsBits ⟨k - Words.tsBits p 0, by simp only [Words.segBits, Words.tsBits] at h ⊢; omega⟩
      else if h : Words.segBits p 0 ≤ k ∧ k < Words.opsProducts p 0 0 then
        w.segBits ⟨k - Words.segBits p 0,
          by simp only [Words.opsProducts, Words.segBits] at h ⊢; omega⟩
      else if h : Words.opsProducts p 0 0 ≤ k ∧ k < Words.scanProducts p 0 0 then
        w.opsProducts ⟨(k - Words.opsProducts p 0 0) / 4,
            by simp only [Words.scanProducts, Words.opsProducts] at h ⊢; omega⟩
          ⟨(k - Words.opsProducts p 0 0) % 4, Nat.mod_lt _ (by decide)⟩
      else if h : Words.scanProducts p 0 0 ≤ k ∧ k < Words.count p then
        w.scanProducts ⟨(k - Words.scanProducts p 0 0) / 4,
            by simp only [Words.count, Words.scanProducts] at h ⊢; omega⟩
          ⟨(k - Words.scanProducts p 0 0) % 4, Nat.mod_lt _ (by decide)⟩
      else 0 := by
  have := Words.opWords_pos (p := p)
  have := Words.scanWidth_pos (p := p)
  simp only [Words.tsBits, Words.finalStart, Words.initialStart] at past
  conv_lhs => unfold StepWitness.wordAt
  simp (disch := omega) only [dif_neg, if_neg, slotAt_above, Words.finalStart, Words.initialStart]

theorem wordAt_tsBits (k : Fin p.wTs) : w.wordAt (Words.tsBits p k) = w.tsBits k := by
  have := k.isLt
  rw [wordAt_tail w (by simp only [Words.tsBits]; omega)]
  simp (disch := (simp only [Words.segBits, Words.tsBits]; omega)) only [dif_pos]
  congr 1; ext; simp only [Words.tsBits]; omega

theorem wordAt_segBits (k : Fin p.segWidth) : w.wordAt (Words.segBits p k) = w.segBits k := by
  have := k.isLt
  rw [wordAt_tail w (by simp only [Words.segBits, Words.tsBits]; omega)]
  rw [dif_neg (by simp only [Words.segBits, Words.tsBits]; omega),
    dif_pos (by simp only [Words.opsProducts, Words.segBits, Words.tsBits]; omega)]
  congr 1; ext; simp only [Words.segBits]; omega

theorem wordAt_opsProducts (j : Fin p.bOps) (c : Fin 4) :
    w.wordAt (Words.opsProducts p j c) = w.opsProducts j c := by
  have := j.isLt
  have := c.isLt
  rw [wordAt_tail w (by simp only [Words.opsProducts, Words.segBits, Words.tsBits]; omega)]
  rw [dif_neg (by simp only [Words.opsProducts, Words.segBits, Words.tsBits]; omega),
    dif_neg (by simp only [Words.opsProducts, Words.segBits, Words.tsBits]; omega),
    dif_pos (by simp only [Words.scanProducts, Words.opsProducts, Words.segBits, Words.tsBits]; omega)]
  congr 1 <;> ext <;> simp only [Words.opsProducts] <;> omega

theorem wordAt_scanProducts (j : Fin p.bScan) (c : Fin 4) :
    w.wordAt (Words.scanProducts p j c) = w.scanProducts j c := by
  have := j.isLt
  have := c.isLt
  have position : Words.scanProducts p j c = Words.finalStart p + p.bScan * p.scanWidth + p.wTs +
      p.segWidth + 4 * p.bOps + 4 * j + c := by
    simp only [Words.scanProducts, Words.opsProducts, Words.segBits, Words.tsBits]; omega
  have ts : Words.tsBits p 0 = Words.finalStart p + p.bScan * p.scanWidth := rfl
  have seg : Words.segBits p 0 = Words.finalStart p + p.bScan * p.scanWidth + p.wTs := rfl
  have ops : Words.opsProducts p 0 0 = Words.finalStart p + p.bScan * p.scanWidth + p.wTs +
      p.segWidth := rfl
  have scan : Words.scanProducts p 0 0 = Words.finalStart p + p.bScan * p.scanWidth + p.wTs +
      p.segWidth + 4 * p.bOps := by
    simp only [Words.scanProducts, Words.opsProducts, Words.segBits, Words.tsBits]; omega
  have count : Words.count p = Words.scanProducts p 0 0 + 4 * p.bScan := by
    simp only [Words.count, Words.scanProducts]; omega
  rw [wordAt_tail w (by omega), dif_neg (by omega), dif_neg (by omega), dif_neg (by omega),
    dif_pos (by omega)]
  congr 1 <;> ext <;> simp only <;> omega

theorem length_words : w.words.length = Words.count p := by simp [words]

theorem wordOf_words {k : ℕ} (hk : k < Words.count p) : MemoryApp.wordOf w.words k = w.wordAt k := by
  rw [MemoryApp.wordOf, List.getD_eq_getElem _ _ (by rw [length_words]; exact hk)]
  simp only [words, List.getElem_ofFn]

/-- `MemoryApp.decode` reads the word list of a witness back as the witness. -/
theorem decode_words : MemoryApp.decode p w.words = w := by
  have count := Words.count_eq (p := p)
  have read : ∀ k, k < Words.count p → MemoryApp.wordOf w.words k = w.wordAt k :=
    fun _ hk => wordOf_words w hk
  unfold MemoryApp.decode StepWitness.ofWords
  cases w
  congr 1
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.appIn]; omega), wordAt_appIn]
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.carryIn]; omega), wordAt_carryIn]
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.carryOut]; omega), wordAt_carryOut]
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.appOut]; omega), wordAt_appOut]
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.proposal]; omega), wordAt_proposal]
  · rw [read _ (by simp only [Words.idle]; omega), wordAt_idle]
  · rw [read _ (by simp only [Words.isOpen]; omega), wordAt_isOpen]
  · rw [read _ (by simp only [Words.openInverse]; omega), wordAt_openInverse]
  · rw [read _ (by simp only [Words.isClose]; omega), wordAt_isClose]
  · rw [read _ (by simp only [Words.closeInverse]; omega), wordAt_closeInverse]
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.etaFresh]; omega), wordAt_etaFresh]
  · funext i; have := i.isLt
    rw [read _ (by simp only [Words.eta1Square]; omega), wordAt_eta1Square]
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.seenPrev]; omega), wordAt_seenPrev]
  · rw [read _ (by simp only [Words.idxEff]; omega), wordAt_idxEff]
  · funext j
    apply OpSlotBits.ofWords_word
    intro k hk
    have := slot_lt j.isLt hk
    rw [read _ (by simp only [Words.op]; omega), wordAt_op _ j hk]
  · funext j
    apply ScanSlotBits.ofWords_word
    intro k hk
    have := slot_lt j.isLt hk
    have shift : Words.initial p j 0 + k = Words.initial p j k := by simp only [Words.initial]; omega
    rw [shift, read _ (by simp only [Words.initial, Words.initialStart]; omega), wordAt_initial _ j hk]
  · funext j
    apply ScanSlotBits.ofWords_word
    intro k hk
    have := slot_lt j.isLt hk
    have shift : Words.final p j 0 + k = Words.final p j k := by simp only [Words.final]; omega
    rw [shift, read _ (by simp only [Words.final, Words.finalStart, Words.initialStart]; omega),
      wordAt_final _ j hk]
  · funext k; have := k.isLt
    rw [read _ (by simp only [Words.tsBits, Words.finalStart, Words.initialStart]; omega),
      wordAt_tsBits]
  · funext k; have := k.isLt
    rw [read _ (by simp only [Words.segBits, Words.tsBits, Words.finalStart, Words.initialStart]; omega),
      wordAt_segBits]
  · funext j c; have := j.isLt; have := c.isLt
    rw [read _ (by simp only [Words.opsProducts, Words.segBits, Words.tsBits, Words.finalStart,
      Words.initialStart]; omega), wordAt_opsProducts]
  · funext j c; have := j.isLt; have := c.isLt
    rw [read _ (by simp only [Words.scanProducts, Words.opsProducts, Words.segBits, Words.tsBits,
      Words.finalStart, Words.initialStart]; omega), wordAt_scanProducts]

end StepWitness

end NightstreamFPrime.Lifecycle.Nebula
