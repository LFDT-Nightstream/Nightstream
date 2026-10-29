import NightstreamFPrime.Lifecycle.Nebula.RowList

/-! Owns the meaning of the polynomial row list: on a word vector, the rows of
`Rows.polyRows` evaluate to zero exactly when the decoded witness satisfies
`PolyRows` and the application's rows hold on its words. It does not own the
hash rows or the circuit. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic

namespace Rows

variable (p : Plan) (a : AppShape) (v : ℕ → F)

local notation "W" => StepWitness.ofWords p a v

open _root_.NightstreamFPrime.Lifecycle.Nebula.Sym (word cIn cOut eta1 eta2 eta1Sq isOpen openInverse isClose closeInverse idxEff
  isWrite isRam addrBit vrBit vwBit rtBit diffBit addr vr vw rt diff opLane scanValue scanStamp
  scanLane tsWord segWord cntBefore wt globalIndex scanIndex opsProductAfter scanProductAfter)

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
      (IsBit (W).isOpen ∧ IsBit (W).isClose) := by
  simp only [bitRows, bitChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append]
  rw [hold_append, hold_append, hold_append, hold_append, hold_append, hold_append,
    hold_finRange, hold_finRange, hold_finRange, hold_finRange, hold_finRange_map,
    hold_finRange_map, hold_two, bitRow_iff, bitRow_iff]
  refine and_congr (and_congr (and_congr (and_congr (and_congr (and_congr ?_ ?_) ?_) ?_)
    (forall_congr' fun k => bitRow_iff v _)) (forall_congr' fun k => bitRow_iff v _)) Iff.rfl
  · exact forall_congr' fun j => by rw [hold_bits, Sym.eval_opLane p a v]
  · exact forall_congr' fun j => by rw [hold_finRange_map]; exact forall_congr' fun k => bitRow_iff v _
  · exact forall_congr' fun j => by rw [hold_bits, Sym.eval_initialLane p a v]
  · exact forall_congr' fun j => by rw [hold_bits, Sym.eval_finalLane p a v]

/-! ### Arm and open -/

theorem armRows_iff : ConstraintsHold v (armRows p) ↔
    (((((((((W).cIn 1 - natWord p.n) * (W).isOpen = 0 ∧
      ((W).cIn 1 - natWord p.n) * (W).openInverse = 1 - (W).isOpen) ∧
      (W).isOpen * (natWord (p.sMax - 1) - (W).cIn 0 - bitsWord (W).segBits) = 0) ∧
      (W).idxEff = (1 - (W).isOpen) * (W).cIn 1) ∧
      (∀ i : Fin 12, (W).seenPrev i =
        (W).isOpen * headerWords p i + (1 - (W).isOpen) * (W).cIn (23 + i))) ∧
      (∀ i : Fin 8, (W).cOut (15 + i) =
        (W).isOpen * (W).proposal i + (1 - (W).isOpen) * (W).cIn (15 + i))) ∧
      (∀ i : Fin 4, (W).cOut (3 + i) =
        (W).isOpen * (W).etaFresh i + (1 - (W).isOpen) * (W).cIn (3 + i))) ∧
      (W).eta1Sq = K.mul (W).eta1 (W).eta1) := by
  simp only [armRows, armChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append]
  rw [hold_append, hold_append, hold_append, hold_append, hold_append, hold_append, hold_two,
    hold_single, hold_single, hold_finRange_map, hold_finRange_map, hold_finRange_map, keq_iff,
    KExpr.eval_mul, Sym.eval_eta1 p a v, Sym.eval_eta1Sq p a v]
  refine and_congr (and_congr (and_congr (and_congr ?_ ?_) ?_) ?_) Iff.rfl
  · refine and_congr (and_congr (and_congr ?_ ?_) ?_) ?_
    · simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Sym.eval_cIn p a v, constE,
        Circuit.Expr.eval_const]
      rfl
    · rw [eval_sub_eq_zero]
      simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Sym.eval_cIn p a v, constE,
        Circuit.Expr.eval_const, Expr.eval_one]
      rfl
    · simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Sym.eval_cIn p a v, constE,
        Circuit.Expr.eval_const, Sym.eval_segWord p a v]
      rfl
    · rw [eval_sub_eq_zero]
      simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Sym.eval_cIn p a v, Expr.eval_one]
      rfl
  · refine forall_congr' fun i => ?_
    rw [eval_sub_eq_zero]
    simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd, Circuit.Expr.eval_sub, Sym.eval_cIn p a v,
      constE, Circuit.Expr.eval_const, Expr.eval_one]
    rfl
  · refine forall_congr' fun i => ?_
    rw [eval_sub_eq_zero]
    simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd, Circuit.Expr.eval_sub, Sym.eval_cIn p a v,
      Sym.eval_cOut p a v, Expr.eval_one]
    rfl
  · refine forall_congr' fun i => ?_
    rw [eval_sub_eq_zero]
    simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd, Circuit.Expr.eval_sub, Sym.eval_cIn p a v,
      Sym.eval_cOut p a v, Expr.eval_one]
    rfl

/-! ### Operation rows -/

theorem slotRows_iff (j : Fin p.bOps) : ConstraintsHold v (slotRows p j) ↔
    (((((((1 - (W).isWrite j) * ((W).vw j - (W).vr j) = 0 ∧
      (1 - (W).pad j) * ((W).wt j - (W).rt j - 1 - (W).diff j) = 0) ∧
      (W).isWrite j * (1 - (W).isRam j) = 0) ∧
      (∀ k : Fin p.μ, p.r ≤ k.val → (1 - (W).isRam j) * ((W).ops j).addr k = 0)) ∧
      (∀ x ∈ (((W).ops j).lane).tail, (W).pad j * x = 0)) ∧
      (W).opsProductAfter 0 (j.val + 1) = K.mul ((W).opsProductAfter 0 j.val)
        (gatedK ((W).pad j) (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq ((W).rt j)
          ((W).globalIndex j) ((W).vr j)))) ∧
      (W).opsProductAfter 1 (j.val + 1) = K.mul ((W).opsProductAfter 1 j.val)
        (gatedK ((W).pad j) (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq ((W).wt j)
          ((W).globalIndex j) ((W).vw j)))) := by
  simp only [slotRows, slotChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append]
  rw [hold_append, hold_append, hold_append, hold_append, hold_append, hold_append, hold_single,
    hold_single, hold_single, keq_iff, keq_iff]
  refine and_congr (and_congr (and_congr (and_congr ?_ ?_) ?_) ?_) ?_
  · simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one, Sym.eval_vw p a v,
      Sym.eval_vr p a v, Sym.eval_wt p a v, Sym.eval_rt p a v, Sym.eval_diff p a v]
    rfl
  · rw [hold_map]
    simp only [List.mem_filter, List.mem_finRange, true_and, decide_eq_true_eq]
    refine forall_congr' fun k => imp_congr_right fun _ => ?_
    simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one]
    rfl
  · rw [hold_map, ← Sym.eval_opLane p a v j, ← List.map_tail, List.forall_mem_map]
    rfl
  · simp only [KExpr.eval_mul, eval_gatedE, eval_fingerprintE, Sym.eval_opsProductAfter p a v,
      Sym.eval_eta1 p a v, Sym.eval_eta2 p a v, Sym.eval_eta1Sq p a v, Sym.eval_rt p a v, Sym.eval_globalIndex p a v,
      Sym.eval_vr p a v]
    rfl
  · simp only [KExpr.eval_mul, eval_gatedE, eval_fingerprintE, Sym.eval_opsProductAfter p a v,
      Sym.eval_eta1 p a v, Sym.eval_eta2 p a v, Sym.eval_eta1Sq p a v, Sym.eval_wt p a v, Sym.eval_globalIndex p a v,
      Sym.eval_vw p a v]
    rfl

/-! ### Scan rows -/

theorem scanRows_iff (j : Fin p.bScan) : ConstraintsHold v (scanRows p j) ↔
    (W).scanProductAfter 0 (j.val + 1) = K.mul ((W).scanProductAfter 0 j.val)
      (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq (bitsWord ((W).initial j).stamp)
        ((W).scanIndex j) (bitsWord ((W).initial j).value)) ∧
    (W).scanProductAfter 1 (j.val + 1) = K.mul ((W).scanProductAfter 1 j.val)
      (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq (bitsWord ((W).final j).stamp)
        ((W).scanIndex j) (bitsWord ((W).final j).value)) := by
  simp only [scanRows, scanChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append]
  rw [hold_append, keq_iff, keq_iff]
  simp only [KExpr.eval_mul, eval_fingerprintE, Sym.eval_scanProductAfter p a v, Sym.eval_eta1 p a v,
    Sym.eval_eta2 p a v, Sym.eval_eta1Sq p a v, Sym.eval_scanIndex p a v, Sym.eval_initialStamp p a v,
    Sym.eval_initialValue p a v, Sym.eval_finalStamp p a v, Sym.eval_finalValue p a v]

/-! ### Boundary and close -/

theorem closeRows_iff : ConstraintsHold v (closeRows p) ↔
    ((((((((((W).cOut 2 = (W).cIn 2 + (W).cntBefore p.bOps ∧ (W).cOut 2 = bitsWord (W).tsBits) ∧
      (W).cOut 1 = (W).idxEff + 1) ∧
      (((kOf ((W).cOut 7) ((W).cOut 8) = (W).opsProductAfter 0 p.bOps ∧
      kOf ((W).cOut 9) ((W).cOut 10) = (W).opsProductAfter 1 p.bOps) ∧
      kOf ((W).cOut 11) ((W).cOut 12) = (W).scanProductAfter 0 p.bScan) ∧
      kOf ((W).cOut 13) ((W).cOut 14) = (W).scanProductAfter 1 p.bScan)) ∧
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
  simp only [closeRows, closeChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append, hold_append]
  rw [hold_single, hold_single, hold_single, keq_iff, keq_iff, keq_iff, keq_iff, hold_two,
    hold_finRange_map, hold_finRange_map, hold_finRange_map, keq_iff, hold_single, hold_finRange_map]
  simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd, sub_eq_zero,
    Circuit.Expr.eval_sub, Expr.eval_one, Sym.eval_cIn p a v, Sym.eval_cOut p a v, Sym.eval_cntBefore p a v,
    Sym.eval_tsWord p a v, eval_kOfE, Sym.eval_opsProductAfter p a v, Sym.eval_scanProductAfter p a v, constE,
    Circuit.Expr.eval_const, KExpr.eval_mul, KExpr.eval_sub, eval_embedE, KExpr.eval_zero]
  rfl

/-! ### Application rows -/

/-- The application rows hold on a word vector exactly when the application's
rows hold on its words through `Words.appVar`. -/
theorem appRows_iff (app : PortApp p) :
    ConstraintsHold v (appRows p app) ↔ ConstraintsHold (v ∘ Words.appVar p app.toAppShape) app.rows := by
  simp only [appRows, ConstraintsHold, List.mem_map, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂, Expr.eval_subst]
  rfl

/-- The polynomial rows hold on a word vector exactly when its decoded witness
satisfies the memory rows and the application's rows hold on its words. -/
theorem polyRows_iff (app : PortApp p) :
    ConstraintsHold v (polyRows p app) ↔ (StepWitness.ofWords p app.toAppShape v).PolyRows ∧
      ConstraintsHold (v ∘ Words.appVar p app.toAppShape) app.rows := by
  rw [polyRows, memoryRows, hold_append, hold_append, hold_append, hold_append, hold_append,
    bitRows_iff p app.toAppShape v, armRows_iff p app.toAppShape v, hold_finRange, hold_finRange,
    closeRows_iff p app.toAppShape v, appRows_iff p v app]
  simp only [slotRows_iff p app.toAppShape v, scanRows_iff p app.toAppShape v]
  constructor
  · rintro ⟨⟨⟨⟨⟨bits, arm⟩, slots⟩, scans⟩, close⟩, application⟩
    obtain ⟨⟨⟨⟨⟨⟨opsBits, diffBits⟩, initialBits⟩, finalBits⟩, tsBits⟩, segBits⟩, openBit,
      closeBit⟩ := bits
    obtain ⟨⟨⟨⟨⟨⟨⟨openZero, openTest⟩, segRange⟩, idxEff⟩, seenPrev⟩, proposed⟩, eta⟩, square⟩ := arm
    obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨tsOut, tsRange⟩, idxOut⟩, ⟨⟨⟨read, write⟩, initial⟩, final⟩⟩, closeZero,
      closeTest⟩, closeOps⟩, closeInitial⟩, closeFinal⟩, closeProducts⟩, segOut⟩, memOut⟩ := close
    refine ⟨?_, application⟩
    exact {
      opsBits := opsBits, diffBits := diffBits, initialBits := initialBits,
      finalBits := finalBits, tsBits := tsBits, segBits := segBits,
      openBit := openBit, closeBit := closeBit, openZero := openZero, openTest := openTest,
      segRange := segRange, idxEff := idxEff, seenPrev := seenPrev, proposed := proposed,
      eta := eta, square := square,
      readKeeps := fun j => (slots j).1.1.1.1.1.1,
      fresh := fun j => (slots j).1.1.1.1.1.2,
      noRomWrite := fun j => (slots j).1.1.1.1.2,
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
  · rintro ⟨rows, application⟩
    refine ⟨⟨⟨⟨⟨?_, ?_⟩, fun j => ?_⟩, fun j => ⟨rows.initialProduct j, rows.finalProduct j⟩⟩, ?_⟩,
      application⟩
    · exact ⟨⟨⟨⟨⟨⟨rows.opsBits, rows.diffBits⟩, rows.initialBits⟩, rows.finalBits⟩, rows.tsBits⟩,
        rows.segBits⟩, rows.openBit, rows.closeBit⟩
    · exact ⟨⟨⟨⟨⟨⟨⟨rows.openZero, rows.openTest⟩, rows.segRange⟩, rows.idxEff⟩, rows.seenPrev⟩,
        rows.proposed⟩, rows.eta⟩, rows.square⟩
    · exact ⟨⟨⟨⟨⟨⟨rows.readKeeps j, rows.fresh j⟩, rows.noRomWrite j⟩, rows.romRange j⟩,
        rows.padZero j⟩, rows.readProduct j⟩, rows.writeProduct j⟩
    · exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨rows.tsOut, rows.tsRange⟩, rows.idxOut⟩, ⟨⟨⟨rows.productsOut.1,
        rows.productsOut.2.1⟩, rows.productsOut.2.2.1⟩, rows.productsOut.2.2.2⟩⟩, rows.closeZero,
        rows.closeTest⟩, rows.closeOps⟩, rows.closeInitial⟩, rows.closeFinal⟩, rows.closeProducts⟩,
        rows.segOut⟩, rows.memOut⟩

end Rows

end NightstreamFPrime.Lifecycle.Nebula
