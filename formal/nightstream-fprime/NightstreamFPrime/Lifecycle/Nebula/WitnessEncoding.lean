import NightstreamFPrime.Lifecycle.Nebula.MemoryCircuit

/-! Owns the inverse of the witness word layout: the word vector of a step
witness, which `MemoryApp.decode` reads back as the same witness. Completeness
of the relation (Ob5) uses it to turn an honest step witness into witness
words. It does not own any row. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula

variable {p : Geometry} {a : AppShape}

theorem OpSlotBits.ofWords_word (slot : OpSlotBits p) (v : ℕ → F) (j : ℕ)
    (words : ∀ k < Words.opWords p, v (Words.op p j k) = slot.word k) :
    OpSlotBits.ofWords p v j = slot := by
  have opWords : Words.opWords p = 3 + p.μ + 64 + p.wTs + p.wTs := by
    simp [Words.opWords, Geometry.opWidth]
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
      rw [words _ (by simp [Words.opDiff, Geometry.opWidth]; omega)]
      simp only [OpSlotBits.word, Words.opDiff, Geometry.opWidth]
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
      rw [words _ (by simp only [Words.scanValue, Geometry.scanWidth]; omega)]
      simp only [ScanSlotBits.word, Words.scanValue]
      rw [dif_pos (by omega)]
    · have := k.isLt
      rw [words _ (by simp only [Words.scanStamp, Geometry.scanWidth]; omega)]
      simp only [ScanSlotBits.word, Words.scanStamp]
      rw [dif_neg (by omega), dif_pos (by omega)]
      congr 1; ext; simp only; omega

namespace Words

theorem scanWidth_pos : 0 < p.scanWidth := by simp [Geometry.scanWidth]

theorem memoryCount_eq : Words.memoryCount p = 109 + p.bOps * Words.opWords p +
    p.bScan * p.scanWidth + p.bScan * p.scanWidth + p.wTs + p.segWidth + 4 * p.bOps +
    4 * p.bScan := by
  simp only [Words.memoryCount, Words.scanProducts, Words.opsProducts, Words.segBits,
    Words.tsBits, Words.finalStart, Words.initialStart]
  omega

theorem count_eq : Words.count p a = Words.memoryCount p + 2 * a.width + a.aux := rfl

end Words

/-! ### The word vector -/

/-- Word `k` past the three slot ranges: the counters, the products, and the
application words; `0` past `Words.count p a`. -/
def StepWitness.tailWord (w : StepWitness p a) (k : ℕ) : F :=
  if h : Words.tsBits p 0 ≤ k ∧ k < Words.segBits p 0 then
    w.tsBits ⟨k - Words.tsBits p 0, by simp only [Words.segBits, Words.tsBits] at h ⊢; omega⟩
  else if h : Words.segBits p 0 ≤ k ∧ k < Words.opsProducts p 0 0 then
    w.segBits ⟨k - Words.segBits p 0, by simp only [Words.opsProducts, Words.segBits] at h ⊢; omega⟩
  else if h : Words.opsProducts p 0 0 ≤ k ∧ k < Words.scanProducts p 0 0 then
    w.opsProducts ⟨(k - Words.opsProducts p 0 0) / 4,
        by simp only [Words.scanProducts, Words.opsProducts] at h ⊢; omega⟩
      ⟨(k - Words.opsProducts p 0 0) % 4, Nat.mod_lt _ (by decide)⟩
  else if h : Words.scanProducts p 0 0 ≤ k ∧ k < Words.memoryCount p then
    w.scanProducts ⟨(k - Words.scanProducts p 0 0) / 4,
        by simp only [Words.memoryCount, Words.scanProducts] at h ⊢; omega⟩
      ⟨(k - Words.scanProducts p 0 0) % 4, Nat.mod_lt _ (by decide)⟩
  else if h : Words.memoryCount p ≤ k ∧ k < Words.memoryCount p + a.width then
    w.appIn ⟨k - Words.memoryCount p, by omega⟩
  else if h : Words.memoryCount p + a.width ≤ k ∧ k < Words.memoryCount p + 2 * a.width then
    w.appOut ⟨k - Words.memoryCount p - a.width, by omega⟩
  else if h : Words.memoryCount p + 2 * a.width ≤ k ∧ k < Words.count p a then
    w.aux ⟨k - Words.memoryCount p - 2 * a.width, by simp only [Words.count] at h; omega⟩
  else 0

/-- The witness words of a step witness in the `Words` layout; `0` past
`Words.count p a`. -/
def StepWitness.wordAt (w : StepWitness p a) (k : ℕ) : F :=
  if h : k < 39 then w.carryIn ⟨k, h⟩
  else if h : k < 78 then w.carryOut ⟨k - 39, by omega⟩
  else if h : k < 86 then w.proposal ⟨k - 78, by omega⟩
  else if k = 86 then w.isOpen
  else if k = 87 then w.openInverse
  else if k = 88 then w.isClose
  else if k = 89 then w.closeInverse
  else if h : k < 94 then w.etaFresh ⟨k - 90, by omega⟩
  else if h : k < 96 then w.eta1Square ⟨k - 94, by omega⟩
  else if h : k < 108 then w.seenPrev ⟨k - 96, by omega⟩
  else if k = 108 then w.idxEff
  else match slotAt 109 (Words.opWords p) p.bOps k with
  | some (j, i) => (w.ops j).word i
  | none => match slotAt (Words.initialStart p) p.scanWidth p.bScan k with
  | some (j, i) => (w.initial j).word i
  | none => match slotAt (Words.finalStart p) p.scanWidth p.bScan k with
  | some (j, i) => (w.final j).word i
  | none => w.tailWord k

/-- The witness word list of a step witness. -/
def StepWitness.words (w : StepWitness p a) : List F :=
  List.ofFn fun k : Fin (Words.count p a) => w.wordAt k

namespace StepWitness

variable (w : StepWitness p a)

/-- Evaluates the branch tests of `wordAt` with `omega`. -/
macro "read_word" : tactic =>
  `(tactic| simp (disch := omega) only [StepWitness.wordAt, dif_pos, dif_neg, if_pos, if_neg,
    ↓reduceIte, ↓reduceDIte, Fin.eta, Nat.add_sub_cancel_left])

theorem wordAt_carryIn (i : Fin 39) : w.wordAt (Words.carryIn i) = w.carryIn i := by
  have := i.isLt; simp only [Words.carryIn]; read_word

theorem wordAt_carryOut (i : Fin 39) : w.wordAt (Words.carryOut i) = w.carryOut i := by
  have := i.isLt; simp only [Words.carryOut]; read_word

theorem wordAt_proposal (i : Fin 8) : w.wordAt (Words.proposal i) = w.proposal i := by
  have := i.isLt; simp only [Words.proposal]; read_word

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
  have start : Words.initialStart p = 109 + p.bOps * Words.opWords p := rfl
  simp only [Words.initial]
  simp (disch := omega) only [StepWitness.wordAt, dif_neg, if_neg, slotAt_above,
    slotAt_slot j.isLt hk, Fin.eta]

theorem wordAt_final (j : Fin p.bScan) {k : ℕ} (hk : k < p.scanWidth) :
    w.wordAt (Words.final p j k) = (w.final j).word k := by
  have := Words.opWords_pos (p := p)
  have := Words.scanWidth_pos (p := p)
  have start : Words.initialStart p = 109 + p.bOps * Words.opWords p := rfl
  have final : Words.finalStart p = Words.initialStart p + p.bScan * p.scanWidth := rfl
  simp only [Words.final]
  simp (disch := omega) only [StepWitness.wordAt, dif_neg, if_neg, slotAt_above,
    slotAt_slot j.isLt hk, Fin.eta]

/-- Past the three slot ranges, `wordAt` is `tailWord`. -/
theorem wordAt_tail {k : ℕ} (past : Words.tsBits p 0 ≤ k) : w.wordAt k = w.tailWord k := by
  have := Words.opWords_pos (p := p)
  have := Words.scanWidth_pos (p := p)
  simp only [Words.tsBits, Words.finalStart, Words.initialStart] at past
  unfold StepWitness.wordAt
  simp (disch := omega) only [dif_neg, if_neg, slotAt_above, Words.finalStart, Words.initialStart]

/-- The tail positions in order. -/
theorem tail_order : Words.tsBits p 0 ≤ Words.segBits p 0 ∧
    Words.segBits p 0 ≤ Words.opsProducts p 0 0 ∧
    Words.opsProducts p 0 0 ≤ Words.scanProducts p 0 0 ∧
    Words.scanProducts p 0 0 ≤ Words.memoryCount p := by
  simp only [Words.memoryCount, Words.scanProducts, Words.opsProducts, Words.segBits, Words.tsBits]
  omega

theorem wordAt_appIn (i : Fin a.width) : w.wordAt (Words.appIn p i) = w.appIn i := by
  have := i.isLt
  have := tail_order (p := p)
  rw [wordAt_tail w (by simp only [Words.appIn]; omega)]
  unfold StepWitness.tailWord
  simp (disch := (simp only [Words.appIn]; omega)) only [dif_neg, dif_pos]
  congr 1; ext; simp only [Words.appIn]; omega

theorem wordAt_appOut (i : Fin a.width) : w.wordAt (Words.appOut p a i) = w.appOut i := by
  have := i.isLt
  have := tail_order (p := p)
  rw [wordAt_tail w (by simp only [Words.appOut]; omega)]
  unfold StepWitness.tailWord
  simp (disch := (simp only [Words.appOut]; omega)) only [dif_neg, dif_pos]
  congr 1; ext; simp only [Words.appOut]; omega

theorem wordAt_aux (i : Fin a.aux) : w.wordAt (Words.aux p a i) = w.aux i := by
  have := i.isLt
  have := tail_order (p := p)
  rw [wordAt_tail w (by simp only [Words.aux]; omega)]
  unfold StepWitness.tailWord
  simp (disch := (simp only [Words.aux, Words.count]; omega)) only [dif_neg, dif_pos]
  congr 1; ext; simp only [Words.aux]; omega

theorem wordAt_tsBits (k : Fin p.wTs) : w.wordAt (Words.tsBits p k) = w.tsBits k := by
  have := k.isLt
  rw [wordAt_tail w (by simp only [Words.tsBits]; omega)]
  unfold StepWitness.tailWord
  simp (disch := (simp only [Words.segBits, Words.tsBits]; omega)) only [dif_pos]
  congr 1; ext; simp only [Words.tsBits]; omega

theorem wordAt_segBits (k : Fin p.segWidth) : w.wordAt (Words.segBits p k) = w.segBits k := by
  have := k.isLt
  rw [wordAt_tail w (by simp only [Words.segBits, Words.tsBits]; omega)]
  unfold StepWitness.tailWord
  rw [dif_neg (by simp only [Words.segBits, Words.tsBits]; omega),
    dif_pos (by simp only [Words.opsProducts, Words.segBits, Words.tsBits]; omega)]
  congr 1; ext; simp only [Words.segBits]; omega

theorem wordAt_opsProducts (j : Fin p.bOps) (c : Fin 4) :
    w.wordAt (Words.opsProducts p j c) = w.opsProducts j c := by
  have := j.isLt
  have := c.isLt
  rw [wordAt_tail w (by simp only [Words.opsProducts, Words.segBits, Words.tsBits]; omega)]
  unfold StepWitness.tailWord
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
  have count : Words.memoryCount p = Words.scanProducts p 0 0 + 4 * p.bScan := by
    simp only [Words.memoryCount, Words.scanProducts]; omega
  rw [wordAt_tail w (by omega)]
  unfold StepWitness.tailWord
  rw [dif_neg (by omega), dif_neg (by omega), dif_neg (by omega), dif_pos (by omega)]
  congr 1 <;> ext <;> simp only <;> omega

theorem length_words : w.words.length = Words.count p a := by simp [words]

theorem wordOf_words {k : ℕ} (hk : k < Words.count p a) : MemoryApp.wordOf w.words k = w.wordAt k := by
  rw [MemoryApp.wordOf, List.getD_eq_getElem _ _ (by rw [length_words]; exact hk)]
  simp only [words, List.getElem_ofFn]

/-- The word list of a witness decodes back to the witness. -/
theorem ofWords_words : StepWitness.ofWords p a (MemoryApp.wordOf w.words) = w := by
  have count := Words.memoryCount_eq (p := p)
  have total : Words.count p a = Words.memoryCount p + 2 * a.width + a.aux := rfl
  have read : ∀ k, k < Words.count p a → MemoryApp.wordOf w.words k = w.wordAt k :=
    fun _ hk => wordOf_words w hk
  unfold StepWitness.ofWords
  cases w
  congr 1
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.carryIn]; omega), wordAt_carryIn]
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.carryOut]; omega), wordAt_carryOut]
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.proposal]; omega), wordAt_proposal]
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
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.appIn]; omega), wordAt_appIn]
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.appOut]; omega), wordAt_appOut]
  · funext i; have := i.isLt; rw [read _ (by simp only [Words.aux]; omega), wordAt_aux]

/-- `MemoryApp.decode` reads the word list of a witness back as the witness. -/
theorem decode_words {app : PortApp p} (w : StepWitness p app.toAppShape) :
    MemoryApp.decode p app w.words = w :=
  ofWords_words w

end StepWitness

end NightstreamFPrime.Lifecycle.Nebula
