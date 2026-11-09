import NightstreamFPrime.Lifecycle.Nebula.RowList

/-! Owns the meaning of the polynomial row list: on a word vector, the rows of
`Rows.polyRows` evaluate to zero exactly when the decoded witness satisfies
`PolyRows`, the auxiliary words are its honest words (`AuxHonest`), and the
application's rows hold on its words. It does not own the hash rows or the
circuit. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open Fin.CommRing

namespace Rows

variable (p : Geometry) (a : AppShape) (v : ℕ → F)

local notation "W" => StepWitness.ofWords p a v

open _root_.NightstreamFPrime.Lifecycle.Nebula.Sym (word cIn cOut eta1 eta2 eta1Sq isOpen openInverse isClose closeInverse idxEff
  isWrite isRam addrBit vrBit vwBit rtBit diffBit addr vr vw rt diff opLane scanValue scanStamp
  scanLane tsWord segWord cnt cntBefore wt globalIndex scanIndex startWord opsProductAfter scanProductAfter
  opsFactor scanFactor)

theorem hold_two (a b : Circuit.Expr) :
    ConstraintsHold v [a, b] ↔ a.eval v = 0 ∧ b.eval v = 0 := by
  simp [ConstraintsHold]

theorem keq_iff (a b : KExpr) : ConstraintsHold v (KExpr.equalities a b) ↔ a.eval v = b.eval v :=
  KExpr.equalities_hold_iff v a b

theorem hold_cons (e : Circuit.Expr) (l : List Circuit.Expr) :
    ConstraintsHold v (e :: l) ↔ e.eval v = 0 ∧ ConstraintsHold v l := by
  simp [ConstraintsHold]

theorem hold_nil : ConstraintsHold v [] ↔ True := by simp [ConstraintsHold]

/-! ### Fingerprint steps -/

theorem gatedRows_iff (f : ℕ → Circuit.Expr) (pad : Circuit.Expr) (prod next : KExpr) (t g x : Circuit.Expr) :
    ConstraintsHold v (gatedRows f pad prod next t g x) ↔
      (fun c : Fin 8 => (f c).eval v) = opsFactorWords (W).eta1 (W).eta2 (W).eta1Sq (prod.eval v) (pad.eval v)
        (t.eval v) (g.eval v) (x.eval v) ∧
      next.eval v = K.mul (prod.eval v) (gatedK (pad.eval v)
        (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq (t.eval v) (g.eval v) (x.eval v))) := by
  rw [← opsFactorWords_iff]
  simp only [gatedRows, fingerprintRows, productRows, List.cons_append, List.nil_append, hold_cons, hold_nil,
    and_true, Circuit.Expr.eval_sub, sub_eq_zero, Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd,
    Sym.eval_cOut p a v, Expr.eval_one, constE, Circuit.Expr.eval_const]
  rfl

theorem factorRows_iff (f : ℕ → Circuit.Expr) (prod next : KExpr) (t g x : Circuit.Expr) :
    ConstraintsHold v (factorRows f prod next t g x) ↔
      (fun c : Fin 6 => (f c).eval v) = scanFactorWords (W).eta1 (W).eta2 (W).eta1Sq (prod.eval v)
        (t.eval v) (g.eval v) (x.eval v) ∧
      next.eval v = K.mul (prod.eval v) (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq (t.eval v) (g.eval v)
        (x.eval v)) := by
  rw [← scanFactorWords_iff]
  simp only [factorRows, fingerprintRows, productRows, List.cons_append, List.nil_append, hold_cons, hold_nil,
    and_true, Circuit.Expr.eval_sub, sub_eq_zero, Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd,
    Sym.eval_cOut p a v, constE, Circuit.Expr.eval_const]
  rfl

/-- The two rows of start product `s` hold exactly when its words are the
product at the start of the step (`StepWitness.startProduct`). -/
theorem startRows_iff (s : Fin 4) :
    ConstraintsHold v [startWord p s 0 - ((1 - isOpen) * cIn (7 + 2 * s.val) + isOpen),
      startWord p s 1 - (1 - isOpen) * cIn (8 + 2 * s.val)] ↔
    kOf (v (Words.start p s 0)) (v (Words.start p s 1)) = (W).startProduct (7 + 2 * s.val) := by
  have isOpenEval : isOpen.eval v = (W).isOpen := rfl
  rw [hold_two]
  simp only [Circuit.Expr.eval_sub, sub_eq_zero, Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd,
    Sym.eval_cIn p a v, Expr.eval_one, isOpenEval, StepWitness.startProduct, kOf, embed, K.add, K.mul,
    K.mk.injEq, show 7 + 2 * s.val + 1 = 8 + 2 * s.val by omega]
  constructor <;> rintro ⟨h0, h1⟩
  · exact ⟨h0.trans (by ring), h1.trans (by ring)⟩
  · exact ⟨h0.trans (by ring), h1.trans (by ring)⟩


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
      (W).eta1Sq = K.mul (W).eta1 (W).eta1) ∧
      (∀ s : Fin 4, kOf (v (Words.start p s 0)) (v (Words.start p s 1)) = (W).startProduct (7 + 2 * s.val)) := by
  simp only [armRows, armChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append]
  rw [hold_append, hold_append, hold_append, hold_append, hold_append, hold_append, hold_append, hold_two,
    hold_single, hold_single, hold_finRange_map, hold_finRange_map, hold_finRange_map, keq_iff,
    KExpr.eval_mul, Sym.eval_eta1 p a v, Sym.eval_eta1Sq p a v, hold_finRange]
  refine and_congr (and_congr (and_congr (and_congr (and_congr ?_ ?_) ?_) ?_) Iff.rfl)
    (forall_congr' fun s => startRows_iff p a v s)
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
    (((((((v (Words.cnt p j) = (cntBefore p j).eval v + (1 - (W).pad j)) ∧
      ((1 - (W).isWrite j) * ((W).vw j - (W).vr j) = 0)) ∧
      ((1 - (W).pad j) * ((W).cIn 2 + v (Words.cnt p j) - (W).rt j - 1 - (W).diff j) = 0)) ∧
      ((W).isWrite j * (1 - (W).isRam j) = 0)) ∧
      (∀ k : Fin p.μ, p.r ≤ k.val → (1 - (W).isRam j) * ((W).ops j).addr k = 0)) ∧
      (∀ x ∈ (((W).ops j).lane).tail, (W).pad j * x = 0)) ∧
      ((fun c : Fin 8 => v (Words.opsFactor p j 0 c)) = opsFactorWords (W).eta1 (W).eta2 (W).eta1Sq
          ((opsProductAfter p 0 j.val).eval v) ((W).pad j) ((W).rt j) ((W).globalIndex j) ((W).vr j) ∧
        (opsProductAfter p 0 (j.val + 1)).eval v = K.mul ((opsProductAfter p 0 j.val).eval v)
          (gatedK ((W).pad j) (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq ((W).rt j) ((W).globalIndex j)
            ((W).vr j))))) ∧
      ((fun c : Fin 8 => v (Words.opsFactor p j 1 c)) = opsFactorWords (W).eta1 (W).eta2 (W).eta1Sq
          ((opsProductAfter p 1 j.val).eval v) ((W).pad j) ((W).cIn 2 + v (Words.cnt p j))
          ((W).globalIndex j) ((W).vw j) ∧
        (opsProductAfter p 1 (j.val + 1)).eval v = K.mul ((opsProductAfter p 1 j.val).eval v)
          (gatedK ((W).pad j) (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq ((W).cIn 2 + v (Words.cnt p j))
            ((W).globalIndex j) ((W).vw j)))) := by
  simp only [slotRows, slotChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append]
  rw [hold_append, hold_append, hold_append, hold_append, hold_append, hold_append, hold_append,
    hold_single, hold_single, hold_single, hold_single, gatedRows_iff p a v, gatedRows_iff p a v]
  refine and_congr (and_congr (and_congr (and_congr (and_congr (and_congr (and_congr ?_ ?_) ?_) ?_) ?_) ?_)
    ?_) ?_
  · rw [eval_sub_eq_zero]
    simp only [Circuit.Expr.eval_hadd, Circuit.Expr.eval_sub, Expr.eval_one]
    rfl
  · simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one, Sym.eval_vw p a v,
      Sym.eval_vr p a v]
    rfl
  · simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one, Sym.eval_wt p a v,
      Sym.eval_rt p a v, Sym.eval_diff p a v]
    rfl
  · simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one]
    rfl
  · rw [hold_map]
    simp only [List.mem_filter, List.mem_finRange, true_and, decide_eq_true_eq]
    refine forall_congr' fun k => imp_congr_right fun _ => ?_
    simp only [Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one]
    rfl
  · rw [hold_map, ← Sym.eval_opLane p a v j, ← List.map_tail, List.forall_mem_map]
    rfl
  · simp only [Sym.eval_rt p a v, Sym.eval_globalIndex p a v, Sym.eval_vr p a v]
    rfl
  · simp only [Sym.eval_wt p a v, Sym.eval_globalIndex p a v, Sym.eval_vw p a v]
    rfl

/-! ### Scan rows -/

theorem scanRows_iff (j : Fin p.bScan) : ConstraintsHold v (scanRows p j) ↔
    ((fun c : Fin 6 => v (Words.scanFactor p j 0 c)) = scanFactorWords (W).eta1 (W).eta2 (W).eta1Sq
        ((scanProductAfter p 0 j.val).eval v) (bitsWord ((W).initial j).stamp) ((W).scanIndex j)
        (bitsWord ((W).initial j).value) ∧
      (scanProductAfter p 0 (j.val + 1)).eval v = K.mul ((scanProductAfter p 0 j.val).eval v)
        (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq (bitsWord ((W).initial j).stamp) ((W).scanIndex j)
          (bitsWord ((W).initial j).value))) ∧
    ((fun c : Fin 6 => v (Words.scanFactor p j 1 c)) = scanFactorWords (W).eta1 (W).eta2 (W).eta1Sq
        ((scanProductAfter p 1 j.val).eval v) (bitsWord ((W).final j).stamp) ((W).scanIndex j)
        (bitsWord ((W).final j).value) ∧
      (scanProductAfter p 1 (j.val + 1)).eval v = K.mul ((scanProductAfter p 1 j.val).eval v)
        (fingerprintK (W).eta1 (W).eta2 (W).eta1Sq (bitsWord ((W).final j).stamp) ((W).scanIndex j)
          (bitsWord ((W).final j).value))) := by
  simp only [scanRows, scanChecks, joinChecks, List.foldl_cons, List.foldl_nil, List.nil_append]
  rw [hold_append, factorRows_iff p a v, factorRows_iff p a v]
  simp only [Sym.eval_scanIndex p a v, Sym.eval_initialStamp p a v, Sym.eval_initialValue p a v,
    Sym.eval_finalStamp p a v, Sym.eval_finalValue p a v]
  rfl

/-! ### Boundary and close -/

theorem closeRows_iff : ConstraintsHold v (closeRows p) ↔
    ((((((((((W).cOut 2 = (W).cIn 2 + (cntBefore p p.bOps).eval v ∧ (W).cOut 2 = bitsWord (W).tsBits) ∧
      (W).cOut 1 = (W).idxEff + 1) ∧
      (((kOf ((W).cOut 7) ((W).cOut 8) = (opsProductAfter p 0 p.bOps).eval v ∧
      kOf ((W).cOut 9) ((W).cOut 10) = (opsProductAfter p 1 p.bOps).eval v) ∧
      kOf ((W).cOut 11) ((W).cOut 12) = (scanProductAfter p 0 p.bScan).eval v) ∧
      kOf ((W).cOut 13) ((W).cOut 14) = (scanProductAfter p 1 p.bScan).eval v)) ∧
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
    Circuit.Expr.eval_sub, Expr.eval_one, Sym.eval_cIn p a v, Sym.eval_cOut p a v,
    Sym.eval_tsWord p a v, eval_kOfE, constE,
    Circuit.Expr.eval_const, KExpr.eval_mul, KExpr.eval_sub, eval_embedE, KExpr.eval_zero]
  rfl

/-! ### Counters and products -/

/-- The O2 chain holds exactly for the honest counters `cnt_j = Σ_{i ≤ j} (1 − pad_i)`. -/
theorem cnt_chain_iff :
    (∀ j : Fin p.bOps, v (Words.cnt p j) = (cntBefore p j).eval v + (1 - (W).pad j)) ↔
      ∀ j : Fin p.bOps, v (Words.cnt p j) = (W).cntBefore (j.val + 1) := by
  have before : ∀ (j : Fin p.bOps), (∀ i : Fin p.bOps, i.val < j.val →
      v (Words.cnt p i) = (W).cntBefore (i.val + 1)) → (cntBefore p j).eval v = (W).cntBefore j.val := by
    intro j earlier
    rw [Sym.eval_cntBefore]
    split
    · next zero => simp [zero, StepWitness.cntBefore]
    · next positive =>
      rw [earlier ⟨j.val - 1, by omega⟩ (show j.val - 1 < j.val by omega)]
      congr 1
      show j.val - 1 + 1 = j.val
      omega
  constructor
  · intro chain
    have upTo : ∀ n (j : Fin p.bOps), j.val = n → v (Words.cnt p j) = (W).cntBefore (j.val + 1) := by
      intro n
      induction n using Nat.strong_induction_on with
      | _ n ih =>
        intro j same
        rw [chain j, StepWitness.cntBefore_succ, before j fun i lt => ih i.val (by omega) i rfl]
    exact fun j => upTo j.val j rfl
  · intro honest j
    rw [honest j, StepWitness.cntBefore_succ, before j fun i _ => honest i]

/-- With honest counters, `cnt` before slot `k` is the witness's count. -/
theorem cntBefore_eval (honest : ∀ j : Fin p.bOps, v (Words.cnt p j) = (W).cntBefore (j.val + 1)) {k : ℕ}
    (bound : k ≤ p.bOps) : (cntBefore p k).eval v = (W).cntBefore k := by
  rw [Sym.eval_cntBefore]
  split
  · next zero => simp [zero, StepWitness.cntBefore]
  · next positive =>
    rw [honest ⟨k - 1, by omega⟩]
    congr 1
    show k - 1 + 1 = k
    omega

theorem kOf_eq_iff (x y : F) (z : K) : kOf x y = z ↔ x = z.c0 ∧ y = z.c1 := by
  cases z
  simp [kOf]

/-- With honest start words, the symbolic operation products are the witness's. -/
theorem opsProduct_eval (starts : ∀ s : Fin 4,
      kOf (v (Words.start p s 0)) (v (Words.start p s 1)) = (W).startProduct (7 + 2 * s.val))
    (pair : ℕ) (small : pair < 2) (k : ℕ) :
    (opsProductAfter p pair k).eval v = (W).opsProductAfter pair k := by
  cases k with
  | zero =>
    rw [Sym.eval_opsProductAfter_zero]
    unfold StepWitness.opsProductAfter
    rw [ite_eq_left rfl]
    exact starts ⟨pair, by omega⟩
  | succ k => exact Sym.eval_opsProductAfter_succ p a v pair k

/-- With honest start words, the symbolic scan products are the witness's. -/
theorem scanProduct_eval (starts : ∀ s : Fin 4,
      kOf (v (Words.start p s 0)) (v (Words.start p s 1)) = (W).startProduct (7 + 2 * s.val))
    (pair : ℕ) (small : pair < 2) (k : ℕ) :
    (scanProductAfter p pair k).eval v = (W).scanProductAfter pair k := by
  cases k with
  | zero =>
    rw [Sym.eval_scanProductAfter_zero]
    unfold StepWitness.scanProductAfter
    rw [ite_eq_left rfl, show 11 + 2 * pair = 7 + 2 * (2 + pair) by omega]
    exact starts ⟨2 + pair, by omega⟩
  | succ k => exact Sym.eval_scanProductAfter_succ p a v pair k

/-- `AuxHonest` field by field. -/
theorem auxHonest_iff : AuxHonest p a v ↔
    (∀ j : Fin p.bOps, v (Words.cnt p j) = (W).cntBefore (j.val + 1)) ∧
    (∀ s : Fin 4, kOf (v (Words.start p s 0)) (v (Words.start p s 1)) = (W).startProduct (7 + 2 * s.val)) ∧
    (∀ (j : Fin p.bOps) (pair : Fin 2),
      (fun c : Fin 8 => v (Words.opsFactor p j pair c)) = (W).auxWords.ops j pair) ∧
    (∀ (j : Fin p.bScan) (pair : Fin 2),
      (fun c : Fin 6 => v (Words.scanFactor p j pair c)) = (W).auxWords.scan j pair) := by
  constructor
  · intro honest
    refine ⟨fun j => congrFun (congrArg AuxWords.cnt honest) j, fun s => ?_,
      fun j pair => congrFun (congrFun (congrArg AuxWords.ops honest) j) pair,
      fun j pair => congrFun (congrFun (congrArg AuxWords.scan honest) j) pair⟩
    have start := congrFun (congrArg AuxWords.start honest) s
    rw [kOf_eq_iff]
    exact ⟨congrFun start 0, congrFun start 1⟩
  · rintro ⟨cnt, start, ops, scan⟩
    unfold AuxHonest AuxWords.ofWords StepWitness.auxWords
    congr 1
    · exact funext cnt
    · funext s c
      obtain ⟨c0, c1⟩ := (kOf_eq_iff _ _ _).mp (start s)
      fin_cases c
      · exact c0
      · exact c1
    · funext j pair
      exact ops j pair
    · funext j pair
      exact scan j pair

theorem opsAux_zero (j : Fin p.bOps) : (W).auxWords.ops j 0 = opsFactorWords (W).eta1 (W).eta2 (W).eta1Sq
    ((W).opsProductAfter 0 j.val) ((W).pad j) ((W).rt j) ((W).globalIndex j) ((W).vr j) := rfl

theorem opsAux_one (j : Fin p.bOps) : (W).auxWords.ops j 1 = opsFactorWords (W).eta1 (W).eta2 (W).eta1Sq
    ((W).opsProductAfter 1 j.val) ((W).pad j) ((W).wt j) ((W).globalIndex j) ((W).vw j) := rfl

theorem scanAux_zero (j : Fin p.bScan) : (W).auxWords.scan j 0 = scanFactorWords (W).eta1 (W).eta2 (W).eta1Sq
    ((W).scanProductAfter 0 j.val) (bitsWord ((W).initial j).stamp) ((W).scanIndex j)
    (bitsWord ((W).initial j).value) := rfl

theorem scanAux_one (j : Fin p.bScan) : (W).auxWords.scan j 1 = scanFactorWords (W).eta1 (W).eta2 (W).eta1Sq
    ((W).scanProductAfter 1 j.val) (bitsWord ((W).final j).stamp) ((W).scanIndex j)
    (bitsWord ((W).final j).value) := rfl

/-! ### Application rows -/

/-- The application rows hold on a word vector exactly when the application's
rows hold on its words through `Words.appVar`. -/
theorem appRows_iff (app : PortApp p) :
    ConstraintsHold v (appRows p app) ↔ ConstraintsHold (v ∘ Words.appVar p app.toAppShape) app.rows := by
  simp only [appRows, ConstraintsHold, List.mem_map, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂, Expr.eval_subst]
  rfl

/-- The polynomial rows hold on a word vector exactly when its decoded witness
satisfies the memory rows, its auxiliary words are honest, and the
application's rows hold on its words. -/
theorem polyRows_iff (app : PortApp p) :
    ConstraintsHold v (polyRows p app) ↔ (StepWitness.ofWords p app.toAppShape v).PolyRows ∧
      AuxHonest p app.toAppShape v ∧ ConstraintsHold (v ∘ Words.appVar p app.toAppShape) app.rows := by
  rw [polyRows, memoryRows, hold_append, hold_append, hold_append, hold_append, hold_append,
    bitRows_iff p app.toAppShape v, armRows_iff p app.toAppShape v, hold_finRange, hold_finRange,
    closeRows_iff p app.toAppShape v, appRows_iff p v app, auxHonest_iff p app.toAppShape v]
  simp only [slotRows_iff p app.toAppShape v, scanRows_iff p app.toAppShape v]
  constructor
  · rintro ⟨⟨⟨⟨⟨bits, arm⟩, slots⟩, scans⟩, close⟩, application⟩
    obtain ⟨⟨⟨⟨⟨⟨opsBits, diffBits⟩, initialBits⟩, finalBits⟩, tsBits⟩, segBits⟩, openBit,
      closeBit⟩ := bits
    obtain ⟨⟨⟨⟨⟨⟨⟨⟨openZero, openTest⟩, segRange⟩, idxEff⟩, seenPrev⟩, proposed⟩, eta⟩, square⟩,
      starts⟩ := arm
    obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨tsOut, tsRange⟩, idxOut⟩, ⟨⟨⟨read, write⟩, initial⟩, final⟩⟩, closeZero,
      closeTest⟩, closeOps⟩, closeInitial⟩, closeFinal⟩, closeProducts⟩, segOut⟩, memOut⟩ := close
    have cnt := (cnt_chain_iff p _ v).mp fun j => (slots j).1.1.1.1.1.1.1
    have wt : ∀ j : Fin p.bOps, (StepWitness.ofWords p app.toAppShape v).cIn 2 + v (Words.cnt p j) =
        (StepWitness.ofWords p app.toAppShape v).wt j := fun j => by rw [cnt j]; rfl
    have ops := opsProduct_eval p _ v starts
    have scan := scanProduct_eval p _ v starts
    refine ⟨{
      opsBits := opsBits, diffBits := diffBits, initialBits := initialBits,
      finalBits := finalBits, tsBits := tsBits, segBits := segBits,
      openBit := openBit, closeBit := closeBit, openZero := openZero, openTest := openTest,
      segRange := segRange, idxEff := idxEff, seenPrev := seenPrev, proposed := proposed,
      eta := eta, square := square,
      readKeeps := fun j => (slots j).1.1.1.1.1.1.2,
      fresh := fun j => by rw [← wt j]; exact (slots j).1.1.1.1.1.2,
      noRomWrite := fun j => (slots j).1.1.1.1.2,
      romRange := fun j => (slots j).1.1.1.2,
      padZero := fun j => (slots j).1.1.2,
      readProduct := fun j => by
        rw [← ops 0 (by decide), ← ops 0 (by decide)]
        exact (slots j).1.2.2,
      writeProduct := fun j => by
        rw [← ops 1 (by decide), ← ops 1 (by decide), ← wt j]
        exact (slots j).2.2,
      initialProduct := fun j => by
        rw [← scan 0 (by decide), ← scan 0 (by decide)]
        exact (scans j).1.2,
      finalProduct := fun j => by
        rw [← scan 1 (by decide), ← scan 1 (by decide)]
        exact (scans j).2.2,
      tsOut := by rw [tsOut, cntBefore_eval p _ v cnt le_rfl],
      tsRange := tsRange, idxOut := idxOut,
      productsOut := ⟨read.trans (ops 0 (by decide) _), write.trans (ops 1 (by decide) _),
        initial.trans (scan 0 (by decide) _), final.trans (scan 1 (by decide) _)⟩,
      closeZero := closeZero, closeTest := closeTest, closeOps := closeOps,
      closeInitial := closeInitial, closeFinal := closeFinal, closeProducts := closeProducts,
      segOut := segOut, memOut := memOut }, ⟨cnt, starts, fun j pair => ?_, fun j pair => ?_⟩, application⟩
    · fin_cases pair <;> simp only [Fin.zero_eta, Fin.mk_one, Fin.isValue]
      · rw [opsAux_zero, ← ops 0 (by decide)]
        exact (slots j).1.2.1
      · rw [opsAux_one, ← ops 1 (by decide), ← wt j]
        exact (slots j).2.1
    · fin_cases pair <;> simp only [Fin.zero_eta, Fin.mk_one, Fin.isValue]
      · rw [scanAux_zero, ← scan 0 (by decide)]
        exact (scans j).1.1
      · rw [scanAux_one, ← scan 1 (by decide)]
        exact (scans j).2.1
  · rintro ⟨rows, ⟨cnt, starts, opsAux, scanAux⟩, application⟩
    have chain := (cnt_chain_iff p _ v).mpr cnt
    have wt : ∀ j : Fin p.bOps, (StepWitness.ofWords p app.toAppShape v).cIn 2 + v (Words.cnt p j) =
        (StepWitness.ofWords p app.toAppShape v).wt j := fun j => by rw [cnt j]; rfl
    have ops := opsProduct_eval p _ v starts
    have scan := scanProduct_eval p _ v starts
    refine ⟨⟨⟨⟨⟨?_, ?_⟩, fun j => ?_⟩, fun j => ?_⟩, ?_⟩, application⟩
    · exact ⟨⟨⟨⟨⟨⟨rows.opsBits, rows.diffBits⟩, rows.initialBits⟩, rows.finalBits⟩, rows.tsBits⟩,
        rows.segBits⟩, rows.openBit, rows.closeBit⟩
    · exact ⟨⟨⟨⟨⟨⟨⟨⟨rows.openZero, rows.openTest⟩, rows.segRange⟩, rows.idxEff⟩, rows.seenPrev⟩,
        rows.proposed⟩, rows.eta⟩, rows.square⟩, starts⟩
    · rw [wt j, ops 0 (by decide), ops 0 (by decide), ops 1 (by decide), ops 1 (by decide)]
      exact ⟨⟨⟨⟨⟨⟨⟨chain j, rows.readKeeps j⟩, rows.fresh j⟩, rows.noRomWrite j⟩, rows.romRange j⟩,
        rows.padZero j⟩, (opsAux j 0).trans (opsAux_zero p _ v j), rows.readProduct j⟩,
        (opsAux j 1).trans (opsAux_one p _ v j), rows.writeProduct j⟩
    · rw [scan 0 (by decide), scan 0 (by decide), scan 1 (by decide), scan 1 (by decide)]
      exact ⟨⟨(scanAux j 0).trans (scanAux_zero p _ v j), rows.initialProduct j⟩,
        (scanAux j 1).trans (scanAux_one p _ v j), rows.finalProduct j⟩
    · rw [cntBefore_eval p _ v cnt le_rfl, ops 0 (by decide), ops 1 (by decide), scan 0 (by decide),
        scan 1 (by decide)]
      exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨rows.tsOut, rows.tsRange⟩, rows.idxOut⟩, ⟨⟨⟨rows.productsOut.1,
        rows.productsOut.2.1⟩, rows.productsOut.2.2.1⟩, rows.productsOut.2.2.2⟩⟩, rows.closeZero,
        rows.closeTest⟩, rows.closeOps⟩, rows.closeInitial⟩, rows.closeFinal⟩, rows.closeProducts⟩,
        rows.segOut⟩, rows.memOut⟩

end Rows

end NightstreamFPrime.Lifecycle.Nebula
