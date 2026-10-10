import NightstreamFPrime.Lifecycle.Nebula.RowList

/-! Owns the meaning of the polynomial row list: on a word vector, the rows of
`Rows.polyRows` evaluate to zero exactly when the decoded witness satisfies
`PolyRows` and `MachineRows`. It does not own the hash rows or the circuit. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic

namespace Rows

variable (p : Plan) (v : ℕ → F)

local notation "W" => StepWitness.ofWords p v

open Sym (word cIn cOut eta1 eta2 eta1Sq idle isOpen openInverse isClose closeInverse idxEff
  isWrite isRam addrBit vrBit vwBit rtBit diffBit addr vr vw rt diff opLane scanValue scanStamp
  scanLane tsWord segWord cntBefore wt globalIndex scanIndex opsProductAfter scanProductAfter)

theorem hold_three (a b c : Circuit.Expr) :
    ConstraintsHold v [a, b, c] ↔ a.eval v = 0 ∧ b.eval v = 0 ∧ c.eval v = 0 := by
  simp [ConstraintsHold]

theorem hold_four (a b c d : Circuit.Expr) :
    ConstraintsHold v [a, b, c, d] ↔
      a.eval v = 0 ∧ b.eval v = 0 ∧ c.eval v = 0 ∧ d.eval v = 0 := by
  simp [ConstraintsHold]

theorem hold_two (a b : Circuit.Expr) :
    ConstraintsHold v [a, b] ↔ a.eval v = 0 ∧ b.eval v = 0 := by
  simp [ConstraintsHold]

theorem keq_iff (a b : KExpr) : ConstraintsHold v (KExpr.equalities a b) ↔ a.eval v = b.eval v :=
  KExpr.equalities_hold_iff v a b

/-! ### Bits -/

theorem bitRows_iff : ConstraintsHold v (bitRows p) ↔
    ((((((∀ j, ∀ x ∈ ((W).ops j).lane, IsBit x) ∧ (∀ j k, IsBit (((W).ops j).diff k))) ∧
      (∀ j, ∀ x ∈ ((W).initial j).lane, IsBit x)) ∧ (∀ j, ∀ x ∈ ((W).final j).lane, IsBit x)) ∧
      (∀ k, IsBit ((W).tsBits k))) ∧ (∀ k, IsBit ((W).segBits k))) ∧
      (IsBit (W).idle ∧ IsBit (W).isOpen ∧ IsBit (W).isClose) := by
  rw [bitRows, hold_append, hold_append, hold_append, hold_append, hold_append, hold_append,
    hold_finRange, hold_finRange, hold_finRange, hold_finRange, hold_finRange_map,
    hold_finRange_map, hold_three, bitRow_iff, bitRow_iff, bitRow_iff]
  refine and_congr (and_congr (and_congr (and_congr (and_congr (and_congr ?_ ?_) ?_) ?_)
    (forall_congr' fun k => bitRow_iff v _)) (forall_congr' fun k => bitRow_iff v _)) Iff.rfl
  · exact forall_congr' fun j => by rw [hold_bits, Sym.eval_opLane]
  · exact forall_congr' fun j => by rw [hold_finRange_map]; exact forall_congr' fun k => bitRow_iff v _
  · exact forall_congr' fun j => by rw [hold_bits, Sym.eval_initialLane]
  · exact forall_congr' fun j => by rw [hold_bits, Sym.eval_finalLane]

/-! ### Arm and open -/

theorem armRows_iff : ConstraintsHold v (armRows p) ↔
    (((((((W).cIn 1 - natWord p.n) * (W).isOpen = 0 ∧
      ((W).cIn 1 - natWord p.n) * (W).openInverse = 1 - (W).isOpen ∧
      (W).isOpen * (natWord (p.sMax - 1) - (W).cIn 0 - bitsWord (W).segBits) = 0 ∧
      (W).idxEff = (1 - (W).isOpen) * (W).cIn 1) ∧
      (∀ i : Fin 12, (W).seenPrev i =
        (W).isOpen * headerWords p i + (1 - (W).isOpen) * (W).cIn (23 + i))) ∧
      (∀ i : Fin 8, (W).cOut (15 + i) =
        (W).isOpen * (W).proposal i + (1 - (W).isOpen) * (W).cIn (15 + i))) ∧
      (∀ i : Fin 4, (W).cOut (3 + i) =
        (W).isOpen * (W).etaFresh i + (1 - (W).isOpen) * (W).cIn (3 + i))) ∧
      (W).eta1Sq = K.mul (W).eta1 (W).eta1) := by
  rw [armRows, hold_append, hold_append, hold_append, hold_append, hold_four, hold_finRange_map,
    hold_finRange_map, hold_finRange_map, keq_iff, KExpr.eval_mul, Sym.eval_eta1 p v, Sym.eval_eta1Sq p v]
  refine and_congr (and_congr (and_congr (and_congr ?_ ?_) ?_) ?_) Iff.rfl
  · refine and_congr ?_ (and_congr ?_ (and_congr ?_ ?_))
    · simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Sym.eval_cIn p v, constE,
        Circuit.Expr.eval_const]
      rfl
    · rw [eval_sub_eq_zero]
      simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Sym.eval_cIn p v, constE,
        Circuit.Expr.eval_const, Expr.eval_one]
      rfl
    · simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Sym.eval_cIn p v, constE,
        Circuit.Expr.eval_const, Sym.eval_segWord]
      rfl
    · rw [eval_sub_eq_zero]
      simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Sym.eval_cIn p v, Expr.eval_one]
      rfl
  · refine forall_congr' fun i => ?_
    rw [eval_sub_eq_zero]
    simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd, Circuit.Expr.eval_sub, Sym.eval_cIn p v,
      constE, Circuit.Expr.eval_const, Expr.eval_one]
    rfl
  · refine forall_congr' fun i => ?_
    rw [eval_sub_eq_zero]
    simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd, Circuit.Expr.eval_sub, Sym.eval_cIn p v,
      Sym.eval_cOut p v, Expr.eval_one]
    rfl
  · refine forall_congr' fun i => ?_
    rw [eval_sub_eq_zero]
    simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd, Circuit.Expr.eval_sub, Sym.eval_cIn p v,
      Sym.eval_cOut p v, Expr.eval_one]
    rfl

/-! ### Operation rows -/

theorem slotRows_iff (j : Fin p.bOps) : ConstraintsHold v (slotRows p j) ↔
    ((((((1 - (W).isWrite j) * ((W).vw j - (W).vr j) = 0 ∧
      (1 - (W).pad j) * ((W).wt j - (W).rt j - 1 - (W).diff j) = 0 ∧
      (W).isWrite j * (1 - (W).isRam j) = 0) ∧
      (∀ k : Fin p.μ, p.r ≤ k.val → (1 - (W).isRam j) * ((W).ops j).addr k = 0)) ∧
      (∀ x ∈ (((W).ops j).lane).tail, (W).pad j * x = 0)) ∧
      (W).opsProductAfter 0 (j.val + 1) = K.mul ((W).opsProductAfter 0 j.val)
        (gatedK ((W).pad j) (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq ((W).rt j)
          ((W).globalIndex j) ((W).vr j)))) ∧
      (W).opsProductAfter 1 (j.val + 1) = K.mul ((W).opsProductAfter 1 j.val)
        (gatedK ((W).pad j) (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq ((W).wt j)
          ((W).globalIndex j) ((W).vw j)))) := by
  rw [slotRows, hold_append, hold_append, hold_append, hold_append, hold_three, keq_iff, keq_iff]
  refine and_congr (and_congr (and_congr (and_congr ?_ ?_) ?_) ?_) ?_
  · simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one, Sym.eval_vw,
      Sym.eval_vr, Sym.eval_wt, Sym.eval_rt, Sym.eval_diff]
    rfl
  · rw [hold_map]
    simp only [List.mem_filter, List.mem_finRange, true_and, decide_eq_true_eq]
    refine forall_congr' fun k => imp_congr_right fun _ => ?_
    simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one]
    rfl
  · rw [hold_map, ← Sym.eval_opLane p v j, ← List.map_tail, List.forall_mem_map]
    rfl
  · simp only [KExpr.eval_mul, eval_gatedE, eval_fingerprintE, Sym.eval_opsProductAfter,
      Sym.eval_eta1 p v, Sym.eval_eta2 p v, Sym.eval_eta1Sq p v, Sym.eval_rt, Sym.eval_globalIndex,
      Sym.eval_vr]
    rfl
  · simp only [KExpr.eval_mul, eval_gatedE, eval_fingerprintE, Sym.eval_opsProductAfter,
      Sym.eval_eta1 p v, Sym.eval_eta2 p v, Sym.eval_eta1Sq p v, Sym.eval_wt, Sym.eval_globalIndex,
      Sym.eval_vw]
    rfl

/-! ### Scan rows -/

theorem scanRows_iff (j : Fin p.bScan) : ConstraintsHold v (scanRows p j) ↔
    (W).scanProductAfter 0 (j.val + 1) = K.mul ((W).scanProductAfter 0 j.val)
      (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq (bitsWord ((W).initial j).stamp)
        ((W).scanIndex j) (bitsWord ((W).initial j).value)) ∧
    (W).scanProductAfter 1 (j.val + 1) = K.mul ((W).scanProductAfter 1 j.val)
      (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq (bitsWord ((W).final j).stamp)
        ((W).scanIndex j) (bitsWord ((W).final j).value)) := by
  rw [scanRows, hold_append, keq_iff, keq_iff]
  simp only [KExpr.eval_mul, eval_fingerprintE, Sym.eval_scanProductAfter, Sym.eval_eta1 p v,
    Sym.eval_eta2 p v, Sym.eval_eta1Sq p v, Sym.eval_scanIndex, Sym.eval_initialStamp,
    Sym.eval_initialValue, Sym.eval_finalStamp, Sym.eval_finalValue]

/-! ### Boundary and close -/

theorem closeRows_iff : ConstraintsHold v (closeRows p) ↔
    ((((((((((((W).cOut 2 = (W).cIn 2 + (W).cntBefore p.bOps ∧ (W).cOut 2 = bitsWord (W).tsBits ∧
      (W).cOut 1 = (W).idxEff + 1) ∧
      kOf ((W).cOut 7) ((W).cOut 8) = (W).opsProductAfter 0 p.bOps) ∧
      kOf ((W).cOut 9) ((W).cOut 10) = (W).opsProductAfter 1 p.bOps) ∧
      kOf ((W).cOut 11) ((W).cOut 12) = (W).scanProductAfter 0 p.bScan) ∧
      kOf ((W).cOut 13) ((W).cOut 14) = (W).scanProductAfter 1 p.bScan) ∧
      (((W).cOut 1 - natWord p.n) * (W).isClose = 0 ∧
        ((W).cOut 1 - natWord p.n) * (W).closeInverse = 1 - (W).isClose)) ∧
      (∀ i : Fin 4, (W).isClose * ((W).cOut (23 + i) - (W).cOut (15 + i)) = 0)) ∧
      (∀ i : Fin 4, (W).isClose * ((W).cOut (27 + i) - (W).cIn (35 + i)) = 0)) ∧
      (∀ i : Fin 4, (W).isClose * ((W).cOut (31 + i) - (W).cOut (19 + i)) = 0)) ∧
      K.mul (embed (W).isClose) (K.sub (K.mul (kOf ((W).cOut 11) ((W).cOut 12))
        (kOf ((W).cOut 9) ((W).cOut 10))) (K.mul (kOf ((W).cOut 7) ((W).cOut 8))
          (kOf ((W).cOut 13) ((W).cOut 14)))) = K.zero) ∧
      (W).cOut 0 = (W).cIn 0 + (W).isClose) ∧
      (∀ i : Fin 4, (W).cOut (35 + i) =
        (W).isClose * (W).cOut (19 + i) + (1 - (W).isClose) * (W).cIn (35 + i)) := by
  rw [closeRows, hold_append, hold_append, hold_append, hold_append, hold_append, hold_append,
    hold_append, hold_append, hold_append, hold_append, hold_append, hold_three, keq_iff, keq_iff,
    keq_iff, keq_iff, hold_two, hold_finRange_map, hold_finRange_map, hold_finRange_map, keq_iff,
    hold_single, hold_finRange_map]
  simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd, sub_eq_zero,
    Circuit.Expr.eval_sub, Expr.eval_one, Sym.eval_cIn p v, Sym.eval_cOut p v, Sym.eval_cntBefore,
    Sym.eval_tsWord, eval_kOfE, Sym.eval_opsProductAfter, Sym.eval_scanProductAfter, constE,
    Circuit.Expr.eval_const, KExpr.eval_mul, KExpr.eval_sub, eval_embedE, KExpr.eval_zero]
  rfl

/-! ### Machine rows -/

section Machine

variable (two : p.bOps = 2)

theorem machineRows_iff : ConstraintsHold v (machineRows p) ↔ (W).MachineRows two := by
  have addr0 : (addr p 0).eval v = (W).addr (StepWitness.fetchSlot two) := by
    rw [Sym.addr, Expr.eval_bits]
    rfl
  have addr1 : (addr p 1).eval v = (W).addr (StepWitness.dataSlot two) := by
    rw [Sym.addr, Expr.eval_bits]
    rfl
  have vr1 : (vr p 1).eval v = (W).vr (StepWitness.dataSlot two) := by
    rw [Sym.vr, Expr.eval_bits]
    rfl
  have vw1 : (vw p 1).eval v = (W).vw (StepWitness.dataSlot two) := by
    rw [Sym.vw, Expr.eval_bits]
    rfl
  have load : (isLoad p).eval v = (W).isLoad two := by
    simp only [isLoad, Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one]
    rfl
  have store : (isStore p).eval v = (W).isStore two := by
    simp only [isStore, Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one]
    rfl
  have loadi : (isLoadi p).eval v = (W).isLoadi two := by
    simp only [isLoadi, Circuit.Expr.eval_hmul]
    rfl
  have arg : (argument p).eval v = (W).argument two := by
    rw [argument, Expr.eval_chunk, List.map_drop, List.map_ofFn]
    rfl
  simp only [machineRows, ConstraintsHold, List.mem_cons, List.not_mem_nil, or_false,
    forall_eq_or_imp, forall_eq, Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd, sub_eq_zero,
    Circuit.Expr.eval_sub, Expr.eval_one, addr0, addr1, vr1, vw1, load, store, loadi, arg]
  constructor
  · rintro ⟨r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11⟩
    exact ⟨r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11⟩
  · rintro ⟨r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11⟩
    exact ⟨r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11⟩

end Machine


/-- The polynomial rows hold on a word vector exactly when its decoded witness
satisfies the memory rows and the machine rows. -/
theorem polyRows_iff (two : p.bOps = 2) :
    ConstraintsHold v (polyRows p) ↔ (W).PolyRows ∧ (W).MachineRows two := by
  rw [polyRows, memoryRows, hold_append, hold_append, hold_append, hold_append, hold_append,
    bitRows_iff, armRows_iff, hold_finRange, hold_finRange, closeRows_iff, machineRows_iff p v two]
  simp only [slotRows_iff, scanRows_iff]
  constructor
  · rintro ⟨⟨⟨⟨⟨bits, arm⟩, slots⟩, scans⟩, close⟩, machine⟩
    obtain ⟨⟨⟨⟨⟨⟨opsBits, diffBits⟩, initialBits⟩, finalBits⟩, tsBits⟩, segBits⟩, idleBit, openBit,
      closeBit⟩ := bits
    obtain ⟨⟨⟨⟨⟨openZero, openTest, segRange, idxEff⟩, seenPrev⟩, proposed⟩, eta⟩, square⟩ := arm
    obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨tsOut, tsRange, idxOut⟩, read⟩, write⟩, initial⟩, final⟩, closeZero,
      closeTest⟩, closeOps⟩, closeInitial⟩, closeFinal⟩, closeProducts⟩, segOut⟩, memOut⟩ := close
    refine ⟨?_, machine⟩
    exact {
      opsBits := opsBits, diffBits := diffBits, initialBits := initialBits,
      finalBits := finalBits, tsBits := tsBits, segBits := segBits, idleBit := idleBit,
      openBit := openBit, closeBit := closeBit, openZero := openZero, openTest := openTest,
      segRange := segRange, idxEff := idxEff, seenPrev := seenPrev, proposed := proposed,
      eta := eta, square := square,
      readKeeps := fun j => (slots j).1.1.1.1.1,
      fresh := fun j => (slots j).1.1.1.1.2.1,
      noRomWrite := fun j => (slots j).1.1.1.1.2.2,
      romRange := fun j => (slots j).1.1.1.2,
      padZero := fun j => (slots j).1.1.2,
      readProduct := fun j => (slots j).1.2,
      writeProduct := fun j => (slots j).2,
      initialProduct := fun j => (scans j).1,
      finalProduct := fun j => (scans j).2,
      tsOut := tsOut, tsRange := tsRange, idxOut := idxOut,
      productsOut := ⟨read, write, initial, final⟩,
      closeZero := closeZero, closeTest := closeTest, closeOps := closeOps,
      closeInitial := closeInitial, closeFinal := closeFinal, closeProducts := closeProducts,
      segOut := segOut, memOut := memOut }
  · rintro ⟨rows, machine⟩
    refine ⟨⟨⟨⟨⟨?_, ?_⟩, fun j => ?_⟩, fun j => ⟨rows.initialProduct j, rows.finalProduct j⟩⟩, ?_⟩,
      machine⟩
    · exact ⟨⟨⟨⟨⟨⟨rows.opsBits, rows.diffBits⟩, rows.initialBits⟩, rows.finalBits⟩, rows.tsBits⟩,
        rows.segBits⟩, rows.idleBit, rows.openBit, rows.closeBit⟩
    · exact ⟨⟨⟨⟨⟨rows.openZero, rows.openTest, rows.segRange, rows.idxEff⟩, rows.seenPrev⟩,
        rows.proposed⟩, rows.eta⟩, rows.square⟩
    · exact ⟨⟨⟨⟨⟨rows.readKeeps j, rows.fresh j, rows.noRomWrite j⟩, rows.romRange j⟩,
        rows.padZero j⟩, rows.readProduct j⟩, rows.writeProduct j⟩
    · exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨rows.tsOut, rows.tsRange, rows.idxOut⟩, rows.productsOut.1⟩,
        rows.productsOut.2.1⟩, rows.productsOut.2.2.1⟩, rows.productsOut.2.2.2⟩, rows.closeZero,
        rows.closeTest⟩, rows.closeOps⟩, rows.closeInitial⟩, rows.closeFinal⟩, rows.closeProducts⟩,
        rows.segOut⟩, rows.memOut⟩

end Rows

end NightstreamFPrime.Lifecycle.Nebula
