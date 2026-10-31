import CompressionLeanRust.Positions
import CompressionLean.Ports.Layer1.Budget
import CompressionLean.Ports.Shrink.Budget
import CompressionLean.Ports.Replay.Data

/-! The translated degrees of `neo_verified::degrees` equal the copies in
`CompressionLean/Ports/Layer1/Budget.lean`, `Ports/Shrink/Budget.lean` and `Replay.Relation.rowCount`.
Each sum is over the degrees that `neo-spartan` charges one `SecurityTerm` each.

Each `Complete` premise says that the copy's value fits in a `usize`, and for the linear charges that
`D ≥ 1`. Production meets them: `D = 54`, and each exponent is the variable count of a table that the
prover holds, so a production degree is below `2^40`. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean.Layer1 (GkrShape rustGkr rustEarly rustLinear rustLate rustDegrees earlyShape lateShape)
open CompressionLean.Shrink.Layout (rowVars cellVars)
open CompressionLean.Circuit.Block (poseidonTemplate)

/-! ## Steps of the translated code -/

theorem dadd_eq_ok {a b c : Usize} (h : degrees.add a b = ok c) : c.val = a.val + b.val :=
  add_eq_ok (by simpa only [degrees.add, lift_ok, bind_ok] using h)

theorem dadd_ok (a b : Usize) (h : a.val + b.val ≤ Usize.max) :
    ∃ c, degrees.add a b = ok c ∧ c.val = a.val + b.val := by
  obtain ⟨c, hc, hv⟩ := add_ok a b h
  exact ⟨c, by simpa only [degrees.add, lift_ok, bind_ok] using hc, hv⟩

theorem dsub_eq_ok {a b c : Usize} (h : degrees.sub a b = ok c) : c.val = a.val - b.val ∧ b.val ≤ a.val :=
  sub_eq_ok (by simpa only [degrees.sub, Usize.checked_sub, lift_ok, bind_ok] using h)

theorem dsub_ok (a b : Usize) (h : b.val ≤ a.val) : ∃ c, degrees.sub a b = ok c ∧ c.val = a.val - b.val := by
  obtain ⟨c, hc, hv⟩ := checked_sub_some a b h
  exact ⟨c, by simp only [degrees.sub, Usize.checked_sub, lift_ok, bind_ok, hc, unwrap_some], hv⟩

theorem dmul_eq_ok {a b c : Usize} (h : degrees.mul a b = ok c) : c.val = a.val * b.val :=
  mul_eq_ok (by simpa only [degrees.mul, lift_ok, bind_ok] using h)

theorem dmul_ok (a b : Usize) (h : a.val * b.val ≤ Usize.max) :
    ∃ c, degrees.mul a b = ok c ∧ c.val = a.val * b.val := by
  obtain ⟨c, hc, hv⟩ := mul_ok a b h
  exact ⟨c, by simpa only [degrees.mul, lift_ok, bind_ok] using hc, hv⟩

theorem max_eq : Usize.max = 2 ^ UScalarTy.Usize.numBits - 1 := by
  rw [← UScalar.max_USize_eq, UScalar.max]

theorem lt_numBits {v : ℕ} (h : 2 ^ v ≤ Usize.max) : v < UScalarTy.Usize.numBits := by
  rw [max_eq] at h
  have hpos : 0 < 2 ^ UScalarTy.Usize.numBits := Nat.two_pow_pos _
  exact (Nat.pow_lt_pow_iff_right (by decide : 1 < 2)).mp (by omega)

theorem bits_usize : (UScalar.cast .Usize core.num.Usize.BITS).val = UScalarTy.Usize.numBits := by
  rw [UScalar.cast_val_eq, bits_val]
  exact Nat.mod_eq_of_lt Nat.lt_two_pow_self

theorem wrapping_shl_one (k : U32) (h : k.val < UScalarTy.Usize.numBits) :
    (core.num.Usize.wrapping_shl 1#usize k).val = 2 ^ k.val := by
  have he := numBits_eq
  have hk : k.val % System.Platform.numBits = k.val := Nat.mod_eq_of_lt (by omega)
  have hlt : 2 ^ k.val < 2 ^ System.Platform.numBits := Nat.pow_lt_pow_right (by decide) (by omega)
  simp only [UScalar.val, core.num.Usize.wrapping_shl_bv_eq, BitVec.shiftLeft_eq, BitVec.toNat_shiftLeft]
  have h1 : (Usize.bv 1#usize).toNat = 1 := by simp
  have hkv : k.bv.toNat = k.val := rfl
  rw [h1, hkv, hk, Nat.one_shiftLeft, he, Nat.mod_eq_of_lt hlt]

theorem exp2_value (v : Usize) (h : v.val < UScalarTy.Usize.numBits) :
    (core.num.Usize.wrapping_shl 1#usize (UScalar.cast .U32 v)).val = 2 ^ v.val := by
  have hn := numBits_le
  have hc : (UScalar.cast .U32 v).val = v.val := by
    rw [UScalar.cast_val_eq]
    have : (64 : ℕ) < 2 ^ 32 := by norm_num
    exact Nat.mod_eq_of_lt (by simp only [UScalarTy.U32_numBits_eq]; omega)
  rw [wrapping_shl_one _ (by omega), hc]

theorem exp2_eq_ok {v p : Usize} (h : degrees.exp2 v = ok p) :
    p.val = 2 ^ v.val ∧ v.val < UScalarTy.Usize.numBits := by
  unfold degrees.exp2 at h
  simp only [lift_ok, bind_ok] at h
  obtain ⟨a, ha, h⟩ := bind_eq_ok h
  obtain ⟨c, hc, h⟩ := bind_eq_ok h
  rw [Result.ok.injEq] at h
  have ha' := dsub_eq_ok ha
  have hc' := dsub_eq_ok hc
  rw [bits_usize] at ha'
  have hlt : v.val < UScalarTy.Usize.numBits := by
    have := numBits_pos
    have : (1#usize).val = 1 := rfl
    omega
  exact ⟨h ▸ exp2_value v hlt, hlt⟩

theorem exp2_ok (v : Usize) (h : v.val < UScalarTy.Usize.numBits) :
    ∃ p, degrees.exp2 v = ok p ∧ p.val = 2 ^ v.val := by
  have hn := numBits_pos
  have h1 : (1#usize).val = 1 := rfl
  have hb := bits_usize
  obtain ⟨a, ha, hav⟩ := dsub_ok (UScalar.cast .Usize core.num.Usize.BITS) 1#usize (by omega)
  obtain ⟨c, hc, -⟩ := dsub_ok a v (by omega)
  exact ⟨_, by simp only [degrees.exp2, lift_ok, bind_ok, ha, hc], exp2_value v h⟩

/-! ## `row_count` -/

/-- A relation with these three counts. `rowCount` reads no other field. -/
def replayRelation (kappa matrices publicBlocks : ℕ) : CompressionLean.Replay.Relation :=
  { (default : CompressionLean.Replay.Relation) with
    kappa, publicBlocks,
    key := { (default : CompressionLean.Replay.Key) with
      layout := { (default : CompressionLean.Replay.Layout) with matrices } } }

def rowCountCopy (x : Usize × Usize × Usize) : ℕ :=
  (replayRelation x.1.val x.2.1.val x.2.2.val).rowCount

theorem rowCountCopy_eq (x : Usize × Usize × Usize) :
    rowCountCopy x = x.1.val + 2 + 2 * x.2.1.val + x.2.2.val := rfl

theorem row_count_eq_ok {k m p y : Usize} (h : degrees.row_count k m p = ok y) :
    y.val = k.val + 2 + 2 * m.val + p.val := by
  unfold degrees.row_count at h
  obtain ⟨a, ha, h⟩ := bind_eq_ok h
  obtain ⟨b, hb, h⟩ := bind_eq_ok h
  obtain ⟨c, hc, h⟩ := bind_eq_ok h
  rw [dadd_eq_ok h, dadd_eq_ok hc, dadd_eq_ok ha, dmul_eq_ok hb]
  rfl

theorem row_count_sound :
    Sound (fun x : Usize × Usize × Usize => degrees.row_count x.1 x.2.1 x.2.2) usizeValue
      (fun x => (replayRelation x.1.val x.2.1.val x.2.2.val).rowCount) :=
  fun x _ h => (row_count_eq_ok h).trans (rowCountCopy_eq x).symm

theorem row_count_succeeds (k m p : Usize) (h : k.val + 2 + 2 * m.val + p.val ≤ Usize.max) :
    ∃ y, degrees.row_count k m p = ok y := by
  have h2 : (2#usize).val = 2 := rfl
  obtain ⟨a, ha, hav⟩ := dadd_ok k 2#usize (by omega)
  obtain ⟨b, hb, hbv⟩ := dmul_ok 2#usize m (by rw [h2]; omega)
  rw [h2] at hav hbv
  obtain ⟨c, hc, hcv⟩ := dadd_ok a b (by omega)
  obtain ⟨d, hd, -⟩ := dadd_ok c p (by omega)
  exact ⟨d, by simp only [degrees.row_count, ha, hb, hc, hd, bind_ok]⟩

theorem row_count_complete :
    Complete (fun x : Usize × Usize × Usize => (replayRelation x.1.val x.2.1.val x.2.2.val).rowCount ≤ Usize.max)
      (fun x : Usize × Usize × Usize => degrees.row_count x.1 x.2.1 x.2.2) usizeValue
      (fun x => (replayRelation x.1.val x.2.1.val x.2.2.val).rowCount) := by
  refine complete_of row_count_sound ⟨(0#usize, 0#usize, 0#usize), ?_⟩ ?_
  · have := max_ge
    rw [U32.max_eq] at this
    show rowCountCopy _ ≤ _
    rw [rowCountCopy_eq]
    simp
    omega
  · intro x h
    exact row_count_succeeds x.1 x.2.1 x.2.2 (by rw [← rowCountCopy_eq]; exact h)

/-! ## `gkr` -/

/-- The depths of a translated slice. -/
def depthsOf (d : Slice Usize) : List ℕ := d.val.map (·.val)

/-- The copy's GKR shape of a slice of depths. `rustGkr` reads no factor. -/
def gkrShape (d : Slice Usize) : GkrShape :=
  ⟨(depthsOf d).length, fun t => (depthsOf d).getD t 0, fun _ => 0⟩

theorem sum_four (L : ℕ) : ∑ k ∈ Finset.Ico 1 (L + 1), 4 * k = 2 * L * (L + 1) := by
  induction L with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_Ico_succ_top (by omega : 1 ≤ n + 1), ih]
    ring

theorem length_filter_range (p : ℕ → Bool) (T : ℕ) :
    ((List.range T).filter p).length = ∑ t ∈ Finset.range T, if p t then 1 else 0 := by
  induction T with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.filter_append, List.length_append, ih, Finset.sum_range_succ]
    cases h : p n <;> simp [List.filter, h]

theorem count_le (m L : ℕ) : ∑ k ∈ Finset.Ico 1 (L + 1), (if k ≤ m then 1 else 0) = min m L := by
  induction L with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_Ico_succ_top (by omega : 1 ≤ n + 1), ih]
    split <;> omega

/-- Layer `k` adds `4k` and one per tree of depth at least `k`, so tree `t` adds its depth. -/
theorem rustGkr_eq (s : GkrShape) :
    rustGkr s = 2 * s.layers * (s.layers + 1) + ∑ t ∈ Finset.range s.trees, s.depth t := by
  unfold rustGkr
  rw [Finset.sum_add_distrib, sum_four]
  congr 1
  simp only [GkrShape.active, length_filter_range, decide_eq_true_eq]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun t ht => ?_
  rw [count_le, min_eq_left (s.depth_le_layers (Finset.mem_range.mp ht))]

theorem le_foldr_max {l : List ℕ} {x : ℕ} (h : x ∈ l) : x ≤ l.foldr max 0 := by
  induction l with
  | nil => simp at h
  | cons a rest ih =>
    rcases List.mem_cons.mp h with rfl | inside
    · exact le_max_left _ _
    · exact (ih inside).trans (le_max_right _ _)

theorem foldr_max_le {l : List ℕ} {B : ℕ} (h : ∀ x ∈ l, x ≤ B) : l.foldr max 0 ≤ B := by
  induction l with
  | nil => simp
  | cons a rest ih => exact max_le (h a (by simp)) (ih fun x hx => h x (List.mem_cons_of_mem a hx))

theorem sum_range_getD (l : List ℕ) : ∑ t ∈ Finset.range l.length, l.getD t 0 = l.sum := by
  induction l with
  | nil => simp
  | cons a rest ih =>
    rw [List.length_cons, Finset.sum_range_succ']
    simp only [List.getD_cons_succ, List.getD_cons_zero, ih, List.sum_cons]
    omega

theorem rustGkr_gkrShape (d : Slice Usize) :
    rustGkr (gkrShape d) = 2 * (depthsOf d).foldr max 0 * ((depthsOf d).foldr max 0 + 1) + (depthsOf d).sum := by
  have layers : (gkrShape d).layers = (depthsOf d).foldr max 0 := by
    apply le_antisymm
    · apply GkrShape.layers_le
      intro t ht
      simp only [gkrShape] at ht ⊢
      rw [List.getD_eq_getElem _ _ ht]
      exact le_foldr_max (List.getElem_mem ht)
    · apply foldr_max_le
      intro x hx
      obtain ⟨t, ht, rfl⟩ := List.getElem_of_mem hx
      have := (gkrShape d).depth_le_layers (t := t) ht
      simpa only [gkrShape, List.getD_eq_getElem _ _ ht] using this
  rw [rustGkr_eq, layers]
  simp only [gkrShape, sum_range_getD]

theorem gkr_loop_eq_ok (d : Slice Usize) :
    ∀ n (deepest sum t : Usize), (depthsOf d).length - t.val = n →
      ∀ r, degrees.gkr_loop d deepest sum t = ok r →
        r.1.val = max deepest.val (((depthsOf d).drop t.val).foldr max 0) ∧
          r.2.val = sum.val + ((depthsOf d).drop t.val).sum := by
  intro n
  induction n with
  | zero =>
    intro deepest sum t hn r h
    unfold degrees.gkr_loop loop at h
    simp only [degrees.gkr_loop.body] at h
    have hlen : (depthsOf d).length = d.val.length := List.length_map _
    have hlt : ¬ t < Slice.len d := by scalar_tac
    simp only [hlt, ite_false, bind_ok, Result.ok.injEq] at h
    subst h
    rw [List.drop_eq_nil_of_le (by omega)]
    simp
  | succ n ih =>
    intro deepest sum t hn r h
    unfold degrees.gkr_loop loop at h
    simp only [degrees.gkr_loop.body] at h
    have hlen : (depthsOf d).length = d.val.length := List.length_map _
    have ht : t.val < d.val.length := by omega
    have hlt : t < Slice.len d := by scalar_tac
    have hindex : Slice.index_usize d t = ok d.val[t.val] := by
      unfold Slice.index_usize
      rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem ht]
      rfl
    simp only [hlt, ite_true, hindex, bind_ok, lift_ok] at h
    obtain ⟨cf, hcf, h⟩ := bind_eq_ok h
    obtain ⟨s1, hs1, hcf⟩ := bind_eq_ok hcf
    obtain ⟨t1, ht1, hcf⟩ := bind_eq_ok hcf
    rw [Result.ok.injEq] at hcf
    subst hcf
    replace h : degrees.gkr_loop d (core.cmp.impls.OrdUsize.max deepest d.val[t.val]) s1 t1 = ok r := by
      unfold degrees.gkr_loop
      simp only [degrees.gkr_loop.body, lift_ok, bind_ok] at h ⊢
      exact h
    have h1 : (1#usize).val = 1 := rfl
    have ht1' := dadd_eq_ok ht1
    obtain ⟨hr1, hr2⟩ := ih _ _ t1 (by omega) r h
    have hdrop : (depthsOf d).drop t.val = d.val[t.val].val :: (depthsOf d).drop t1.val := by
      rw [ht1', h1, List.drop_eq_getElem_cons (by omega)]
      simp only [depthsOf, List.getElem_map]
      rfl
    rw [hdrop, List.foldr_cons, List.sum_cons]
    rw [hr1, hr2, core.cmp.impls.OrdUsize.max_val, dadd_eq_ok hs1]
    constructor
    · rw [max_assoc]
    · omega

theorem gkr_loop_succeeds (d : Slice Usize) :
    ∀ n (deepest sum t : Usize), (depthsOf d).length - t.val = n →
      sum.val + ((depthsOf d).drop t.val).sum ≤ Usize.max → ∃ r, degrees.gkr_loop d deepest sum t = ok r := by
  have hlen : (depthsOf d).length = d.val.length := List.length_map _
  have hbound := d.property
  intro n
  induction n with
  | zero =>
    intro deepest sum t hn _
    have hlt : ¬ t < Slice.len d := by scalar_tac
    refine ⟨(deepest, sum), ?_⟩
    unfold degrees.gkr_loop loop
    simp only [degrees.gkr_loop.body, hlt, ite_false, bind_ok]
  | succ n ih =>
    intro deepest sum t hn hfit
    have ht : t.val < d.val.length := by omega
    have hlt : t < Slice.len d := by scalar_tac
    have hdrop : (depthsOf d).drop t.val = d.val[t.val].val :: (depthsOf d).drop (t.val + 1) := by
      rw [List.drop_eq_getElem_cons (by omega)]
      simp only [depthsOf, List.getElem_map]
      rfl
    rw [hdrop, List.sum_cons] at hfit
    have h1 : (1#usize).val = 1 := rfl
    obtain ⟨s1, hs1, hs1v⟩ := dadd_ok sum d.val[t.val] (by omega)
    obtain ⟨t1, ht1, ht1v⟩ := dadd_ok t 1#usize (by omega)
    obtain ⟨r, hr⟩ := ih (core.cmp.impls.OrdUsize.max deepest d.val[t.val]) s1 t1 (by rw [ht1v, h1]; omega)
      (by rw [ht1v, h1, hs1v]; omega)
    refine ⟨r, ?_⟩
    have hindex : Slice.index_usize d t = ok d.val[t.val] := by
      unfold Slice.index_usize
      rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem ht]
      rfl
    unfold degrees.gkr_loop at hr
    unfold degrees.gkr_loop loop
    simp only [degrees.gkr_loop.body, hlt, ite_true, hindex, lift_ok, bind_ok, hs1, ht1] at hr ⊢
    exact hr

theorem gkr_eq_ok {d : Slice Usize} {y : Usize} (h : degrees.gkr d = ok y) : y.val = rustGkr (gkrShape d) := by
  unfold degrees.gkr at h
  obtain ⟨⟨D, S⟩, hl, h⟩ := bind_eq_ok h
  obtain ⟨hD, hS⟩ := gkr_loop_eq_ok d _ 0#usize 0#usize 0#usize rfl _ hl
  simp only at hD hS h
  obtain ⟨a, ha, h⟩ := bind_eq_ok h
  obtain ⟨b, hb, h⟩ := bind_eq_ok h
  obtain ⟨c, hc, h⟩ := bind_eq_ok h
  have h0 : (0#usize).val = 0 := rfl
  have h1 : (1#usize).val = 1 := rfl
  have h2 : (2#usize).val = 2 := rfl
  rw [h0, List.drop_zero, Nat.zero_max] at hD
  rw [h0, List.drop_zero, Nat.zero_add] at hS
  rw [dadd_eq_ok h, dmul_eq_ok hc, dmul_eq_ok hb, dadd_eq_ok ha, hD, hS, h1, h2, rustGkr_gkrShape]
  ring

theorem gkr_sound : Sound degrees.gkr usizeValue (fun d => rustGkr (gkrShape d)) :=
  fun _ _ h => gkr_eq_ok h

theorem gkr_succeeds (d : Slice Usize) (h : rustGkr (gkrShape d) ≤ Usize.max) : ∃ y, degrees.gkr d = ok y := by
  rw [rustGkr_gkrShape] at h
  have h0 : (0#usize).val = 0 := rfl
  have h1 : (1#usize).val = 1 := rfl
  have h2 : (2#usize).val = 2 := rfl
  obtain ⟨⟨D, S⟩, hl⟩ := gkr_loop_succeeds d _ 0#usize 0#usize 0#usize rfl (by rw [h0, List.drop_zero]; omega)
  obtain ⟨hD, hS⟩ := gkr_loop_eq_ok d _ 0#usize 0#usize 0#usize rfl _ hl
  simp only at hD hS
  rw [h0, List.drop_zero, Nat.zero_max] at hD
  rw [h0, List.drop_zero, Nat.zero_add] at hS
  generalize (depthsOf d).foldr max 0 = L at h hD
  generalize (depthsOf d).sum = T at h hS
  have hmax : 1 ≤ Usize.max := by have := max_ge; rw [U32.max_eq] at this; omega
  have hfit : D.val + (1#usize).val ≤ Usize.max := by
    rw [h1, hD]
    rcases Nat.eq_zero_or_pos L with hL | hL
    · omega
    · nlinarith
  obtain ⟨a, ha, hav⟩ := dadd_ok D 1#usize hfit
  obtain ⟨b, hb, hbv⟩ := dmul_ok D a (by rw [hav, h1, hD]; nlinarith)
  obtain ⟨c, hc, hcv⟩ := dmul_ok 2#usize b (by rw [hbv, hav, h1, hD, h2]; nlinarith)
  obtain ⟨e, he, -⟩ := dadd_ok c S (by rw [hcv, hbv, hav, h1, hD, h2, hS]; nlinarith)
  refine ⟨e, ?_⟩
  rw [degrees.gkr, hl, bind_ok]
  exact (by simp only [ha, hb, hc, he, bind_ok] : (do
    let i ← degrees.add D 1#usize
    let i1 ← degrees.mul D i
    let i2 ← degrees.mul 2#usize i1
    degrees.add i2 S) = ok e)

theorem gkr_complete :
    Complete (fun d => rustGkr (gkrShape d) ≤ Usize.max) degrees.gkr usizeValue (fun d => rustGkr (gkrShape d)) :=
  complete_of gkr_sound ⟨Slice.new Usize, by
    rw [rustGkr_gkrShape]
    simp [depthsOf, Slice.new_val]⟩ gkr_succeeds

/-! ## Arrays of degrees -/

theorem index_value_eq {n i y : Usize} {a : Aeneas.Std.Array Usize n} {l : List ℕ} (h : usizeValues a = l)
    (hy : Array.index_usize a i = ok y) : l[i.val]? = some y.val := by
  subst h
  simp [usizeValues, index_eq_ok hy]

theorem index_value {n : Usize} {a : Aeneas.Std.Array Usize n} {l : List ℕ} (h : usizeValues a = l) (i : Usize)
    (hi : i.val < l.length) : ∃ y, Array.index_usize a i = ok y ∧ l[i.val]? = some y.val := by
  subst h
  have hlen : i.val < a.val.length := by simpa [usizeValues] using hi
  refine ⟨a.val[i.val], index_ok a i _ (List.getElem?_eq_getElem hlen), ?_⟩
  rw [usizeValues, List.getElem?_map, List.getElem?_eq_getElem hlen]
  rfl

theorem values_make {n : Usize} (l : List Usize) (hl : l.length = n.val) :
    usizeValues (Aeneas.Std.Array.make n l hl) = l.map (·.val) := by
  rw [usizeValues, Array.make_val]

theorem rustGkr_congr {s s' : GkrShape} (ht : s.trees = s'.trees)
    (hd : ∀ t < s.trees, s.depth t = s'.depth t) : rustGkr s = rustGkr s' := by
  have layers : s.layers = s'.layers := by
    apply le_antisymm
    · exact GkrShape.layers_le s _ fun t h => (hd t h) ▸ s'.depth_le_layers (ht ▸ h)
    · exact GkrShape.layers_le s' _ fun t h => (hd t (ht ▸ h)).symm ▸ s.depth_le_layers (ht ▸ h)
  rw [rustGkr_eq, rustGkr_eq, layers, ht]
  congr 1
  exact Finset.sum_congr rfl fun t h => hd t (ht ▸ Finset.mem_range.mp h)

theorem rustGkr_slice {n : Usize} (l : List Usize) (hl : l.length = n.val) (s : GkrShape) (ht : l.length = s.trees)
    (hd : ∀ t (h : t < l.length), (l[t]'h).val = s.depth t) :
    rustGkr (gkrShape (Array.to_slice (Aeneas.Std.Array.make n l hl))) = rustGkr s := by
  apply rustGkr_congr
  · simp [gkrShape, depthsOf, Array.to_slice, ht]
  · intro t h
    simp only [gkrShape, depthsOf, Array.to_slice, Slice.from_val, Array.make_val, List.length_map] at h ⊢
    rw [← hd t h, List.getD_eq_getElem _ _ (by simpa using h), List.getElem_map]

/-! ## The layer-1 charges -/

/-- The copy's shape of a translated relation. The copies read no other field. -/
def layer1Shape (r : degrees.Relation) : CompressionLean.Layer1.Shape where
  D := r.d.val
  laneVars := r.lane_variables.val
  blockVars := r.block_variables.val
  blocks := 0
  kappa := r.kappa.val
  matrices := r.matrices.val
  publicBlocks := r.public_blocks.val
  normBound := r.norm_bound.val
  runVars := r.run_variables.val
  rowVars := r.row_variables.val
  slotVars := r.slot_variables.val
  matrixVars := r.matrix_variables.val
  setupVars := 0
  queries := 0

/-- The degrees of `rustEarly`, in the order of `early_terms`. -/
def earlyTerms (s : CompressionLean.Layer1.Shape) : List ℕ :=
  [2 ^ s.cubeVars + (2 * s.normBound + 1), 3 * (2 ^ s.runVars + 2 ^ s.rowVars),
    2 ^ s.runVars + 2 ^ (s.slotVars + s.matrixVars) + 1, rustGkr (earlyShape s), s.runVars]

def linearTerms (s : CompressionLean.Layer1.Shape) : List ℕ := [s.rows - 1, 2 * s.D - 2, 2 * s.cubeVars]

def lateTerms (s : CompressionLean.Layer1.Shape) : List ℕ :=
  [2 ^ (s.slotVars + 1) + 2 ^ s.blockVars, rustGkr (lateShape s)]

theorem cube_eq_ok {r : degrees.Relation} {c : Usize} (h : degrees.cube_variables r = ok c) :
    c.val = (layer1Shape r).cubeVars := by
  rw [dadd_eq_ok h, CompressionLean.Layer1.Shape.cubeVars]
  rfl

theorem early_terms_eq_ok {r : degrees.Relation} {a : Aeneas.Std.Array Usize 5#usize}
    (h : degrees.early_terms r = ok a) : usizeValues a = earlyTerms (layer1Shape r) := by
  unfold degrees.early_terms at h
  simp only [lift_ok, bind_ok] at h
  obtain ⟨cube, hcube, h⟩ := bind_eq_ok h
  obtain ⟨pairs, hpairs, h⟩ := bind_eq_ok h
  obtain ⟨i, hi, h⟩ := bind_eq_ok h
  obtain ⟨i2, hi2, h⟩ := bind_eq_ok h
  obtain ⟨i3, hi3, h⟩ := bind_eq_ok h
  obtain ⟨i4, hi4, h⟩ := bind_eq_ok h
  obtain ⟨i5, hi5, h⟩ := bind_eq_ok h
  obtain ⟨i6, hi6, h⟩ := bind_eq_ok h
  obtain ⟨i7, hi7, h⟩ := bind_eq_ok h
  obtain ⟨i8, hi8, h⟩ := bind_eq_ok h
  obtain ⟨i9, hi9, h⟩ := bind_eq_ok h
  obtain ⟨i10, hi10, h⟩ := bind_eq_ok h
  obtain ⟨i11, hi11, h⟩ := bind_eq_ok h
  obtain ⟨i12, hi12, h⟩ := bind_eq_ok h
  rw [Result.ok.injEq] at h
  subst h
  have hg := gkr_eq_ok hi12
  rw [rustGkr_slice _ _ (earlyShape (layer1Shape r)) rfl (by
    intro t ht
    have hc := cube_eq_ok hcube
    have hp := dadd_eq_ok hpairs
    simp only [List.length_cons, List.length_nil] at ht
    interval_cases t <;> simp [earlyShape, hc, hp, layer1Shape])] at hg
  have h1 : (1#usize).val = 1 := rfl
  have h2 : (2#usize).val = 2 := rfl
  have h3 : (3#usize).val = 3 := rfl
  simp only [values_make, List.map_cons, List.map_nil, earlyTerms, List.cons.injEq, and_true]
  refine ⟨?_, ?_, ?_, hg, rfl⟩
  · rw [dadd_eq_ok hi4, (exp2_eq_ok hi).1, cube_eq_ok hcube, dadd_eq_ok hi3, dmul_eq_ok hi2, cast_u32, h1, h2]
    rfl
  · rw [dmul_eq_ok hi8, dadd_eq_ok hi7, (exp2_eq_ok hi5).1, (exp2_eq_ok hi6).1, h3]
    rfl
  · rw [dadd_eq_ok hi11, dadd_eq_ok hi10, (exp2_eq_ok hi5).1, (exp2_eq_ok hi9).1, dadd_eq_ok hpairs, h1]
    rfl

theorem pow_fits {v : ℕ} (h : 2 ^ v ≤ Usize.max) : v < UScalarTy.Usize.numBits ∧ v ≤ Usize.max :=
  ⟨lt_numBits h, (Nat.lt_two_pow_self (n := v)).le.trans h⟩

theorem early_terms_succeeds (r : degrees.Relation) (h : rustEarly (layer1Shape r) ≤ Usize.max) :
    ∃ a, degrees.early_terms r = ok a := by
  obtain ⟨C, hC⟩ : ∃ C, 2 ^ (r.lane_variables.val + r.block_variables.val) = C := ⟨_, rfl⟩
  obtain ⟨U, hU⟩ : ∃ U, 2 ^ r.run_variables.val = U := ⟨_, rfl⟩
  obtain ⟨O, hO⟩ : ∃ O, 2 ^ r.row_variables.val = O := ⟨_, rfl⟩
  obtain ⟨P, hP⟩ : ∃ P, 2 ^ (r.slot_variables.val + r.matrix_variables.val) = P := ⟨_, rfl⟩
  obtain ⟨G, hG⟩ : ∃ G, rustGkr (earlyShape (layer1Shape r)) = G := ⟨_, rfl⟩
  have hfit : C + (2 * r.norm_bound.val + 1) + 3 * (U + O) + (U + P + 1) + G + r.run_variables.val ≤
      Usize.max := by
    rw [← hC, ← hU, ← hO, ← hP, ← hG]
    exact h
  have h1 : (1#usize).val = 1 := rfl
  have h2 : (2#usize).val = 2 := rfl
  have h3 : (3#usize).val = 3 := rfl
  have pc := pow_fits (v := r.lane_variables.val + r.block_variables.val) (by rw [hC]; omega)
  have pp := pow_fits (v := r.slot_variables.val + r.matrix_variables.val) (by rw [hP]; omega)
  have pu := pow_fits (v := r.run_variables.val) (by rw [hU]; omega)
  have po := pow_fits (v := r.row_variables.val) (by rw [hO]; omega)
  obtain ⟨cube, hcube, hcv⟩ := dadd_ok r.lane_variables r.block_variables pc.2
  have hcube' : degrees.cube_variables r = ok cube := by simpa only [degrees.cube_variables] using hcube
  obtain ⟨pairs, hpairs, hpv⟩ := dadd_ok r.slot_variables r.matrix_variables pp.2
  obtain ⟨i, hi, hiv⟩ := exp2_ok cube (by rw [hcv]; exact pc.1)
  rw [hcv, hC] at hiv
  have hn := cast_u32 r.norm_bound
  obtain ⟨i2, hi2, hi2v⟩ := dmul_ok 2#usize (UScalar.cast .Usize r.norm_bound) (by rw [hn, h2]; omega)
  rw [hn, h2] at hi2v
  obtain ⟨i3, hi3, hi3v⟩ := dadd_ok i2 1#usize (by rw [hi2v, h1]; omega)
  rw [hi2v, h1] at hi3v
  obtain ⟨i4, hi4, -⟩ := dadd_ok i i3 (by rw [hi3v, hiv]; omega)
  obtain ⟨i5, hi5, hi5v⟩ := exp2_ok r.run_variables pu.1
  obtain ⟨i6, hi6, hi6v⟩ := exp2_ok r.row_variables po.1
  rw [hU] at hi5v
  rw [hO] at hi6v
  obtain ⟨i7, hi7, hi7v⟩ := dadd_ok i5 i6 (by rw [hi5v, hi6v]; omega)
  rw [hi5v, hi6v] at hi7v
  obtain ⟨i8, hi8, -⟩ := dmul_ok 3#usize i7 (by rw [hi7v, h3]; omega)
  obtain ⟨i9, hi9, hi9v⟩ := exp2_ok pairs (by rw [hpv]; exact pp.1)
  rw [hpv, hP] at hi9v
  obtain ⟨i10, hi10, hi10v⟩ := dadd_ok i5 i9 (by rw [hi5v, hi9v]; omega)
  rw [hi5v, hi9v] at hi10v
  obtain ⟨i11, hi11, -⟩ := dadd_ok i10 1#usize (by rw [hi10v, h1]; omega)
  obtain ⟨i12, hi12⟩ := gkr_succeeds (Array.to_slice (Aeneas.Std.Array.make 5#usize
      [cube, r.run_variables, r.row_variables, r.run_variables, pairs])) (by
    rw [rustGkr_slice _ _ (earlyShape (layer1Shape r)) rfl (by
      intro t ht
      simp only [List.length_cons, List.length_nil] at ht
      interval_cases t <;> simp [earlyShape, hcv, hpv, layer1Shape, CompressionLean.Layer1.Shape.cubeVars]), hG]
    omega)
  exact ⟨Aeneas.Std.Array.make 5#usize [i4, i8, i11, i12, r.run_variables], by
    simp only [degrees.early_terms, lift_ok, bind_ok, hcube', hpairs, hi, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9,
      hi10, hi11, hi12]⟩

theorem earlyTerms_sum (s : CompressionLean.Layer1.Shape) : (earlyTerms s).sum = rustEarly s := by
  simp only [earlyTerms, rustEarly, List.sum_cons, List.sum_nil]
  omega

theorem early_eq_ok {r : degrees.Relation} {y : Usize} (h : degrees.early r = ok y) :
    y.val = rustEarly (layer1Shape r) := by
  unfold degrees.early at h
  obtain ⟨a, ha, h⟩ := bind_eq_ok h
  have hv := early_terms_eq_ok ha
  obtain ⟨y0, h0, h⟩ := bind_eq_ok h
  obtain ⟨y1, h1, h⟩ := bind_eq_ok h
  obtain ⟨y2, h2, h⟩ := bind_eq_ok h
  obtain ⟨y3, h3, h⟩ := bind_eq_ok h
  obtain ⟨y4, h4, h⟩ := bind_eq_ok h
  obtain ⟨b, hb, h⟩ := bind_eq_ok h
  obtain ⟨c, hc, h⟩ := bind_eq_ok h
  obtain ⟨e, he, h⟩ := bind_eq_ok h
  have e0 := index_value_eq hv h0
  have e1 := index_value_eq hv h1
  have e2 := index_value_eq hv h2
  have e3 := index_value_eq hv h3
  have e4 := index_value_eq hv h4
  rw [← earlyTerms_sum, dadd_eq_ok h, dadd_eq_ok he, dadd_eq_ok hc, dadd_eq_ok hb]
  simp only [earlyTerms] at e0 e1 e2 e3 e4 ⊢
  simp at e0 e1 e2 e3 e4
  simp only [List.sum_cons, List.sum_nil]
  omega

theorem array_list {n : Usize} (a : Aeneas.Std.Array Usize n) : a.val.length = n.val := a.property

theorem index_list {n : Usize} {a : Aeneas.Std.Array Usize n} {l : List Usize} (h : a.val = l) (i : Usize)
    (hi : i.val < l.length) : Array.index_usize a i = ok (l[i.val]'hi) :=
  index_ok a i _ (by simp [h, hi])

theorem early_succeeds (r : degrees.Relation) (h : rustEarly (layer1Shape r) ≤ Usize.max) :
    ∃ y, degrees.early r = ok y := by
  obtain ⟨a, ha⟩ := early_terms_succeeds r h
  have hv := early_terms_eq_ok ha
  have hl := array_list a
  rw [← earlyTerms_sum, ← hv, usizeValues] at h
  match hval : a.val, hl with
  | [y0, y1, y2, y3, y4], _ =>
    simp only [hval, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil] at h
    obtain ⟨b, hb, -⟩ := dadd_ok y0 y1 (by omega)
    obtain ⟨c, hc, -⟩ := dadd_ok b y2 (by rw [dadd_eq_ok hb]; omega)
    obtain ⟨d, hd, -⟩ := dadd_ok c y3 (by rw [dadd_eq_ok hc, dadd_eq_ok hb]; omega)
    obtain ⟨e, he, -⟩ := dadd_ok d y4 (by rw [dadd_eq_ok hd, dadd_eq_ok hc, dadd_eq_ok hb]; omega)
    refine ⟨e, ?_⟩
    simp only [degrees.early, ha, bind_ok, index_list hval 0#usize (by simp), index_list hval 1#usize (by simp),
      index_list hval 2#usize (by simp), index_list hval 3#usize (by simp), index_list hval 4#usize (by simp)]
    simp [hb, hc, hd, he]

theorem early_sound : Sound degrees.early usizeValue (fun r => rustEarly (layer1Shape r)) :=
  fun _ _ h => early_eq_ok h

/-- A relation with every count zero and `D = 1`. -/
def zeroRelation : degrees.Relation :=
  ⟨1#usize, 0#usize, 0#usize, 0#usize, 0#usize, 0#usize, 0#u32, 0#usize, 0#usize, 0#usize, 0#usize⟩

theorem zero_rustGkr_early : rustGkr (earlyShape (layer1Shape zeroRelation)) = 0 := by
  rw [rustGkr_eq]
  simp [earlyShape, layer1Shape, zeroRelation, CompressionLean.Layer1.GkrShape.layers,
    CompressionLean.Layer1.Shape.cubeVars, List.range_succ]

theorem early_complete :
    Complete (fun r => rustEarly (layer1Shape r) ≤ Usize.max) degrees.early usizeValue
      (fun r => rustEarly (layer1Shape r)) := by
  refine complete_of early_sound ⟨zeroRelation, ?_⟩ early_succeeds
  have := max_ge
  rw [U32.max_eq] at this
  simp only [rustEarly, zero_rustGkr_early]
  simp [layer1Shape, zeroRelation, CompressionLean.Layer1.Shape.cubeVars]
  omega

/-! ### `linear` -/

theorem linearTerms_sum (s : CompressionLean.Layer1.Shape) : (linearTerms s).sum = rustLinear s := by
  simp only [linearTerms, rustLinear, List.sum_cons, List.sum_nil]
  omega

theorem shape_rows_eq (r : degrees.Relation) :
    (layer1Shape r).rows = r.kappa.val + 2 + 2 * r.matrices.val + r.public_blocks.val := rfl

theorem linear_terms_eq_ok {r : degrees.Relation} {a : Aeneas.Std.Array Usize 3#usize}
    (h : degrees.linear_terms r = ok a) : usizeValues a = linearTerms (layer1Shape r) := by
  unfold degrees.linear_terms at h
  obtain ⟨rows, hrows, h⟩ := bind_eq_ok h
  obtain ⟨i, hi, h⟩ := bind_eq_ok h
  obtain ⟨i1, hi1, h⟩ := bind_eq_ok h
  obtain ⟨i2, hi2, h⟩ := bind_eq_ok h
  obtain ⟨i3, hi3, h⟩ := bind_eq_ok h
  obtain ⟨i4, hi4, h⟩ := bind_eq_ok h
  rw [Result.ok.injEq] at h
  subst h
  have h1 : (1#usize).val = 1 := rfl
  have h2 : (2#usize).val = 2 := rfl
  have e1 := dsub_eq_ok hi1
  simp only [values_make, List.map_cons, List.map_nil, linearTerms, List.cons.injEq, and_true]
  refine ⟨?_, ?_, ?_⟩
  · rw [(dsub_eq_ok hi).1, row_count_eq_ok hrows, shape_rows_eq, h1]
  · rw [dmul_eq_ok hi2, e1.1, h1, h2]
    show 2 * (r.d.val - 1) = 2 * r.d.val - 2
    omega
  · rw [dmul_eq_ok hi4, cube_eq_ok hi3, h2]

/-- `D ≥ 1`, and the ring rows and the linear degrees fit. With `D = 0`, Rust's `D - 1` panics and the
copy's `2·D - 2` is 0. Production has `D = 54`. -/
def LinearFits (r : degrees.Relation) : Prop :=
  1 ≤ r.d.val ∧ (layer1Shape r).rows ≤ Usize.max ∧ rustLinear (layer1Shape r) ≤ Usize.max

theorem linear_terms_succeeds (r : degrees.Relation) (h : LinearFits r) : ∃ a, degrees.linear_terms r = ok a := by
  obtain ⟨hd, hrows, hfit⟩ := h
  rw [shape_rows_eq] at hrows
  simp only [rustLinear, CompressionLean.Layer1.Shape.cubeVars, layer1Shape] at hfit
  have h1 : (1#usize).val = 1 := rfl
  have h2 : (2#usize).val = 2 := rfl
  obtain ⟨rows, hrows', hrv⟩ : ∃ y, degrees.row_count r.kappa r.matrices r.public_blocks = ok y ∧
      y.val = r.kappa.val + 2 + 2 * r.matrices.val + r.public_blocks.val := by
    obtain ⟨y, hy⟩ := row_count_succeeds r.kappa r.matrices r.public_blocks hrows
    exact ⟨y, hy, row_count_eq_ok hy⟩
  obtain ⟨i, hi, -⟩ := dsub_ok rows 1#usize (by rw [hrv, h1]; omega)
  obtain ⟨i1, hi1, hi1v⟩ := dsub_ok r.d 1#usize (by rw [h1]; omega)
  obtain ⟨i2, hi2, -⟩ := dmul_ok 2#usize i1 (by rw [hi1v, h1, h2]; omega)
  obtain ⟨i3, hi3, hi3v⟩ := dadd_ok r.lane_variables r.block_variables (by omega)
  have hi3' : degrees.cube_variables r = ok i3 := by simpa only [degrees.cube_variables] using hi3
  obtain ⟨i4, hi4, -⟩ := dmul_ok 2#usize i3 (by rw [hi3v, h2]; omega)
  exact ⟨Aeneas.Std.Array.make 3#usize [i, i2, i4], by
    simp only [degrees.linear_terms, hrows', hi, hi1, hi2, hi3', hi4, bind_ok]⟩

theorem linear_eq_ok {r : degrees.Relation} {y : Usize} (h : degrees.linear r = ok y) :
    y.val = rustLinear (layer1Shape r) := by
  unfold degrees.linear at h
  obtain ⟨a, ha, h⟩ := bind_eq_ok h
  have hv := linear_terms_eq_ok ha
  have hl := array_list a
  rw [← linearTerms_sum, ← hv, usizeValues]
  match hval : a.val, hl with
  | [y0, y1, y2], _ =>
    simp only [index_list hval 0#usize (by simp), index_list hval 1#usize (by simp),
      index_list hval 2#usize (by simp), bind_ok] at h
    obtain ⟨b, hb, h⟩ := bind_eq_ok h
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    simp at hb h
    rw [dadd_eq_ok h, dadd_eq_ok hb]
    omega

theorem linear_succeeds (r : degrees.Relation) (h : LinearFits r) : ∃ y, degrees.linear r = ok y := by
  obtain ⟨a, ha⟩ := linear_terms_succeeds r h
  have hv := linear_terms_eq_ok ha
  have hl := array_list a
  have hfit := h.2.2
  rw [← linearTerms_sum, ← hv, usizeValues] at hfit
  match hval : a.val, hl with
  | [y0, y1, y2], _ =>
    simp only [hval, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil] at hfit
    obtain ⟨b, hb, -⟩ := dadd_ok y0 y1 (by omega)
    obtain ⟨c, hc, -⟩ := dadd_ok b y2 (by rw [dadd_eq_ok hb]; omega)
    refine ⟨c, ?_⟩
    simp only [degrees.linear, ha, bind_ok, index_list hval 0#usize (by simp), index_list hval 1#usize (by simp),
      index_list hval 2#usize (by simp)]
    simp [hb, hc]

theorem linear_sound : Sound degrees.linear usizeValue (fun r => rustLinear (layer1Shape r)) :=
  fun _ _ h => linear_eq_ok h

theorem zero_linear_fits : LinearFits zeroRelation := by
  have := max_ge
  rw [U32.max_eq] at this
  refine ⟨by decide, ?_, ?_⟩
  · simp [layer1Shape, zeroRelation, CompressionLean.Layer1.Shape.rows]
    omega
  · simp [rustLinear, layer1Shape, zeroRelation, CompressionLean.Layer1.Shape.rows,
      CompressionLean.Layer1.Shape.cubeVars]
    omega

theorem linear_complete :
    Complete LinearFits degrees.linear usizeValue (fun r => rustLinear (layer1Shape r)) :=
  complete_of linear_sound ⟨zeroRelation, zero_linear_fits⟩ linear_succeeds

/-! ### `late` -/

theorem lateTerms_sum (s : CompressionLean.Layer1.Shape) : (lateTerms s).sum = rustLate s := by
  simp only [lateTerms, rustLate, List.sum_cons, List.sum_nil, Nat.add_zero]

theorem late_slice (r : degrees.Relation) (blocks : Usize) (hb : blocks.val = r.slot_variables.val + 1) :
    rustGkr (gkrShape (Array.to_slice (Aeneas.Std.Array.make 2#usize [blocks, r.block_variables]))) =
      rustGkr (lateShape (layer1Shape r)) := by
  rw [rustGkr_slice _ _ (lateShape (layer1Shape r)) rfl]
  intro t ht
  simp only [List.length_cons, List.length_nil] at ht
  interval_cases t <;> simp [lateShape, hb, layer1Shape]

theorem late_terms_eq_ok {r : degrees.Relation} {a : Aeneas.Std.Array Usize 2#usize}
    (h : degrees.late_terms r = ok a) : usizeValues a = lateTerms (layer1Shape r) := by
  unfold degrees.late_terms at h
  simp only [lift_ok, bind_ok] at h
  obtain ⟨blocks, hblocks, h⟩ := bind_eq_ok h
  obtain ⟨i, hi, h⟩ := bind_eq_ok h
  obtain ⟨i1, hi1, h⟩ := bind_eq_ok h
  obtain ⟨i2, hi2, h⟩ := bind_eq_ok h
  obtain ⟨i3, hi3, h⟩ := bind_eq_ok h
  rw [Result.ok.injEq] at h
  subst h
  have h1 : (1#usize).val = 1 := rfl
  have hb := dadd_eq_ok hblocks
  rw [h1] at hb
  simp only [values_make, List.map_cons, List.map_nil, lateTerms, List.cons.injEq, and_true]
  refine ⟨?_, ?_⟩
  · rw [dadd_eq_ok hi2, (exp2_eq_ok hi).1, (exp2_eq_ok hi1).1, hb]
    rfl
  · rw [gkr_eq_ok hi3, late_slice r blocks hb]

theorem late_terms_succeeds (r : degrees.Relation) (h : rustLate (layer1Shape r) ≤ Usize.max) :
    ∃ a, degrees.late_terms r = ok a := by
  obtain ⟨B, hB⟩ : ∃ B, 2 ^ (r.slot_variables.val + 1) = B := ⟨_, rfl⟩
  obtain ⟨K, hK⟩ : ∃ K, 2 ^ r.block_variables.val = K := ⟨_, rfl⟩
  obtain ⟨G, hG⟩ : ∃ G, rustGkr (lateShape (layer1Shape r)) = G := ⟨_, rfl⟩
  have hfit : B + K + G ≤ Usize.max := by
    rw [← hB, ← hK, ← hG]
    exact h
  have h1 : (1#usize).val = 1 := rfl
  have pb := pow_fits (v := r.slot_variables.val + 1) (by rw [hB]; omega)
  have pk := pow_fits (v := r.block_variables.val) (by rw [hK]; omega)
  obtain ⟨blocks, hblocks, hbv⟩ := dadd_ok r.slot_variables 1#usize (by rw [h1]; exact pb.2)
  rw [h1] at hbv
  obtain ⟨i, hi, hiv⟩ := exp2_ok blocks (by rw [hbv]; exact pb.1)
  obtain ⟨i1, hi1, hi1v⟩ := exp2_ok r.block_variables pk.1
  rw [hbv, hB] at hiv
  rw [hK] at hi1v
  obtain ⟨i2, hi2, -⟩ := dadd_ok i i1 (by rw [hiv, hi1v]; omega)
  obtain ⟨i3, hi3⟩ := gkr_succeeds (Array.to_slice (Aeneas.Std.Array.make 2#usize [blocks, r.block_variables]))
    (by rw [late_slice r blocks hbv, hG]; omega)
  exact ⟨Aeneas.Std.Array.make 2#usize [i2, i3], by
    simp only [degrees.late_terms, lift_ok, bind_ok, hblocks, hi, hi1, hi2, hi3]⟩

theorem late_eq_ok {r : degrees.Relation} {y : Usize} (h : degrees.late r = ok y) :
    y.val = rustLate (layer1Shape r) := by
  unfold degrees.late at h
  obtain ⟨a, ha, h⟩ := bind_eq_ok h
  have hv := late_terms_eq_ok ha
  have hl := array_list a
  rw [← lateTerms_sum, ← hv, usizeValues]
  match hval : a.val, hl with
  | [y0, y1], _ =>
    simp only [index_list hval 0#usize (by simp), index_list hval 1#usize (by simp), bind_ok] at h
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    simp at h
    rw [dadd_eq_ok h]
    omega

theorem late_succeeds (r : degrees.Relation) (h : rustLate (layer1Shape r) ≤ Usize.max) :
    ∃ y, degrees.late r = ok y := by
  obtain ⟨a, ha⟩ := late_terms_succeeds r h
  have hv := late_terms_eq_ok ha
  have hl := array_list a
  rw [← lateTerms_sum, ← hv, usizeValues] at h
  match hval : a.val, hl with
  | [y0, y1], _ =>
    simp only [hval, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil] at h
    obtain ⟨b, hb, -⟩ := dadd_ok y0 y1 (by omega)
    refine ⟨b, ?_⟩
    simp only [degrees.late, ha, bind_ok, index_list hval 0#usize (by simp), index_list hval 1#usize (by simp)]
    simp [hb]

theorem late_sound : Sound degrees.late usizeValue (fun r => rustLate (layer1Shape r)) :=
  fun _ _ h => late_eq_ok h

theorem zero_rustGkr_late : rustGkr (lateShape (layer1Shape zeroRelation)) = 5 := by
  rw [rustGkr_eq]
  simp [lateShape, layer1Shape, zeroRelation, CompressionLean.Layer1.GkrShape.layers, List.range_succ]

theorem late_complete :
    Complete (fun r => rustLate (layer1Shape r) ≤ Usize.max) degrees.late usizeValue
      (fun r => rustLate (layer1Shape r)) := by
  refine complete_of late_sound ⟨zeroRelation, ?_⟩ late_succeeds
  have := max_ge
  rw [U32.max_eq] at this
  simp only [rustLate, zero_rustGkr_late]
  simp [layer1Shape, zeroRelation]
  omega

/-! ### `layer1` -/

theorem fixed_eq : ∃ (f : Aeneas.Std.Array Usize 2#usize) (x y : Usize), degrees.FIXED = ok f ∧ f.val = [x, y] ∧
    x.val = 3 ∧ y.val = 6 := by
  have hmax := max_ge
  rw [U32.max_eq] at hmax
  obtain ⟨x, hx, hxv⟩ : ∃ z, (4#usize : Usize) - 1#usize = ok z ∧ z.val = 3 := by
    have := UScalar.sub_equiv (4#usize : Usize) 1#usize
    split at this
    · rename_i z hz
      exact ⟨z, Result.match.isOk.mp hz, by simp at this; omega⟩
    · simp at this
    · exact this.elim
  obtain ⟨y, hy, hyv⟩ : ∃ z, (2#usize : Usize) * 3#usize = ok z ∧ z.val = 6 := by
    have := UScalar.mul_equiv (2#usize : Usize) 3#usize
    rw [UScalar.max_USize_eq] at this
    split at this
    · rename_i z hz
      exact ⟨z, Result.match.isOk.mp hz, by simp at this; omega⟩
    · simp at this
      omega
    · exact this.elim
  exact ⟨Aeneas.Std.Array.make 2#usize [x, y], x, y,
    by simp only [degrees.FIXED, degrees.RUN_ORACLES, degrees.FOLD, hx, hy, bind_ok], Array.make_val _ _ _, hxv, hyv⟩

/-- `D ≥ 1`, and the ring rows and every layer-1 degree fit in a `usize`. -/
def Layer1Fits (r : degrees.Relation) : Prop :=
  1 ≤ r.d.val ∧ (layer1Shape r).rows ≤ Usize.max ∧ rustDegrees (layer1Shape r) ≤ Usize.max

theorem layer1_eq_ok {r : degrees.Relation} {y : Usize} (h : degrees.layer1 r = ok y) :
    y.val = rustDegrees (layer1Shape r) := by
  obtain ⟨f, x3, x6, hf, hfv, hx3, hx6⟩ := fixed_eq
  unfold degrees.layer1 at h
  simp only [hf, bind_ok, index_list hfv 0#usize (by simp), index_list hfv 1#usize (by simp)] at h
  obtain ⟨e, he, h⟩ := bind_eq_ok h
  obtain ⟨l, hl, h⟩ := bind_eq_ok h
  obtain ⟨a, ha, h⟩ := bind_eq_ok h
  obtain ⟨b, hb, h⟩ := bind_eq_ok h
  obtain ⟨c, hc, h⟩ := bind_eq_ok h
  obtain ⟨t, ht, h⟩ := bind_eq_ok h
  rw [dadd_eq_ok h, dadd_eq_ok hc, dadd_eq_ok hb, dadd_eq_ok ha, early_eq_ok he, linear_eq_ok hl, late_eq_ok ht,
    rustDegrees]
  have e0 : ([x3, x6][(0#usize).val]'(by simp)).val = 3 := hx3
  have e1 : ([x3, x6][(1#usize).val]'(by simp)).val = 6 := hx6
  rw [e0, e1]

theorem layer1_succeeds (r : degrees.Relation) (h : Layer1Fits r) : ∃ y, degrees.layer1 r = ok y := by
  obtain ⟨f, x3, x6, hf, hfv, hx3, hx6⟩ := fixed_eq
  obtain ⟨hd, hrows, hfit⟩ := h
  have hsum := hfit
  simp only [rustDegrees] at hsum
  obtain ⟨e, he⟩ := early_succeeds r (by omega)
  obtain ⟨l, hl⟩ := linear_succeeds r ⟨hd, hrows, by omega⟩
  obtain ⟨t, ht⟩ := late_succeeds r (by omega)
  have hev := early_eq_ok he
  have hlv := linear_eq_ok hl
  have htv := late_eq_ok ht
  obtain ⟨a, ha, hav⟩ := dadd_ok e l (by omega)
  obtain ⟨b, hb, hbv⟩ := dadd_ok a x3 (by omega)
  obtain ⟨c, hc, hcv⟩ := dadd_ok b x6 (by omega)
  obtain ⟨y, hy, -⟩ := dadd_ok c t (by omega)
  refine ⟨y, ?_⟩
  simp only [degrees.layer1, hf, bind_ok, index_list hfv 0#usize (by simp), index_list hfv 1#usize (by simp)]
  simp [he, hl, ht, ha, hb, hc, hy]

theorem layer1_sound : Sound degrees.layer1 usizeValue (fun r => rustDegrees (layer1Shape r)) :=
  fun _ _ h => layer1_eq_ok h

theorem layer1_complete :
    Complete Layer1Fits degrees.layer1 usizeValue (fun r => rustDegrees (layer1Shape r)) := by
  refine complete_of layer1_sound ⟨zeroRelation, ?_⟩ layer1_succeeds
  have := max_ge
  rw [U32.max_eq] at this
  refine ⟨by decide, zero_linear_fits.2.1, ?_⟩
  simp only [rustDegrees, rustEarly, rustLate, rustLinear, zero_rustGkr_early, zero_rustGkr_late]
  simp [layer1Shape, zeroRelation, CompressionLean.Layer1.Shape.rows, CompressionLean.Layer1.Shape.cubeVars]
  omega

/-! ## The shrink charges -/

theorem degree_eq : degrees.DEGREE = 8#usize := by unfold degrees.DEGREE; rfl

theorem matrices_eq : circuit.MATRICES = 8#usize := by unfold circuit.MATRICES; rfl

theorem shrink_terms_eq_ok {s : layout.Shape} {a : Aeneas.Std.Array Usize 4#usize}
    (h : degrees.shrink_terms s = ok a) :
    ∃ m n, usizeValues a = [m, 8 * m, 7, 2 * (n + 1)] ∧ m = rowVars poseidonTemplate (counts s) ∧
      n = cellVars poseidonTemplate (counts s) ∧ n ≤ UScalarTy.Usize.numBits := by
  unfold degrees.shrink_terms at h
  obtain ⟨rows, hrows, h⟩ := bind_eq_ok h
  obtain ⟨cells, hcells, h⟩ := bind_eq_ok h
  obtain ⟨i, hi, h⟩ := bind_eq_ok h
  obtain ⟨i1, hi1, h⟩ := bind_eq_ok h
  obtain ⟨i2, hi2, h⟩ := bind_eq_ok h
  obtain ⟨i3, hi3, h⟩ := bind_eq_ok h
  rw [Result.ok.injEq] at h
  subst h
  have h1 : (1#usize).val = 1 := rfl
  have h2 : (2#usize).val = 2 := rfl
  have h8 : (8#usize).val = 8 := rfl
  obtain ⟨hn, hnb⟩ := cell_variables_eq_ok hcells
  refine ⟨rows.val, cells.val, ?_, row_variables_sound s rows hrows, hn, hnb⟩
  simp only [values_make, List.map_cons, List.map_nil, List.cons.injEq, and_true]
  refine ⟨trivial, ?_, ?_, ?_⟩
  · rw [dmul_eq_ok hi, degree_eq, h8]
  · rw [(dsub_eq_ok hi1).1, matrices_eq, h1, h8]
  · rw [dmul_eq_ok hi3, dadd_eq_ok hi2, h1, h2]

theorem shrink_eq_ok {s : layout.Shape} {y : Usize} (h : degrees.shrink s = ok y) :
    y.val = CompressionLean.Shrink.rustDegrees (rowVars poseidonTemplate (counts s))
      (cellVars poseidonTemplate (counts s)) := by
  unfold degrees.shrink at h
  obtain ⟨a, ha, h⟩ := bind_eq_ok h
  obtain ⟨m, n, hv, hm, hn, -⟩ := shrink_terms_eq_ok ha
  have hl := array_list a
  match hval : a.val, hl with
  | [y0, y1, y2, y3], _ =>
    rw [usizeValues, hval] at hv
    simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true] at hv
    simp only [index_list hval 0#usize (by simp), index_list hval 1#usize (by simp),
      index_list hval 2#usize (by simp), index_list hval 3#usize (by simp), bind_ok] at h
    obtain ⟨b, hb, h⟩ := bind_eq_ok h
    obtain ⟨c, hc, h⟩ := bind_eq_ok h
    simp at hb hc h
    rw [dadd_eq_ok h, dadd_eq_ok hc, dadd_eq_ok hb, CompressionLean.Shrink.rustDegrees, ← hm, ← hn]
    omega

theorem variablesFor_le {x : ℕ} (h : x ≤ Usize.max) :
    CompressionLean.Shrink.variablesFor x ≤ UScalarTy.Usize.numBits := by
  have hp : 0 < 2 ^ UScalarTy.Usize.numBits := Nat.two_pow_pos _
  have := Nat.clog_le_of_le_pow (show x ≤ 2 ^ UScalarTy.Usize.numBits by rw [max_eq] at h; omega)
  have := numBits_pos
  unfold CompressionLean.Shrink.variablesFor
  omega

theorem shrink_succeeds (s : layout.Shape) (h : RowsFit s ∧ CellsFit s) : ∃ y, degrees.shrink s = ok y := by
  have hmax := max_ge
  rw [U32.max_eq] at hmax
  have hb := numBits_le
  obtain ⟨rows, hrows, hrv⟩ := row_variables_complete.2 s h.1
  obtain ⟨cells, hcells, -⟩ := cell_variables_complete.2 s h.2
  have hm : rows.val ≤ UScalarTy.Usize.numBits := by
    have hfit := h.1
    unfold RowsFit at hfit
    rw [usizeValue] at hrv
    rw [hrv]
    simp only [CompressionLean.Shrink.Layout.rowVars]
    exact variablesFor_le hfit
  have hn := (cell_variables_eq_ok hcells).2
  have h1 : (1#usize).val = 1 := rfl
  have h2 : (2#usize).val = 2 := rfl
  have h8 : (8#usize).val = 8 := rfl
  obtain ⟨i, hi, hiv⟩ := dmul_ok degrees.DEGREE rows (by rw [degree_eq, h8]; omega)
  obtain ⟨i1, hi1, hi1v⟩ := dsub_ok circuit.MATRICES 1#usize (by rw [matrices_eq, h1, h8]; omega)
  obtain ⟨i2, hi2, hi2v⟩ := dadd_ok cells 1#usize (by rw [h1]; omega)
  obtain ⟨i3, hi3, hi3v⟩ := dmul_ok 2#usize i2 (by rw [hi2v, h1, h2]; omega)
  have ha : degrees.shrink_terms s = ok (Aeneas.Std.Array.make 4#usize [rows, i, i1, i3]) := by
    simp only [degrees.shrink_terms, hrows, hcells, hi, hi1, hi2, hi3, bind_ok]
  have hval : (Aeneas.Std.Array.make 4#usize [rows, i, i1, i3]).val = [rows, i, i1, i3] := Array.make_val _ _ _
  rw [degree_eq, h8] at hiv
  rw [matrices_eq, h1, h8] at hi1v
  rw [hi2v, h1, h2] at hi3v
  obtain ⟨b, hb', hbv⟩ := dadd_ok rows i (by omega)
  obtain ⟨c, hc, hcv⟩ := dadd_ok b i1 (by omega)
  obtain ⟨y, hy, -⟩ := dadd_ok c i3 (by omega)
  refine ⟨y, ?_⟩
  simp only [degrees.shrink, ha, bind_ok, index_list hval 0#usize (by simp), index_list hval 1#usize (by simp),
    index_list hval 2#usize (by simp), index_list hval 3#usize (by simp)]
  simp [hb', hc, hy]

theorem shrink_sound :
    Sound degrees.shrink usizeValue (fun s => CompressionLean.Shrink.rustDegrees (rowVars poseidonTemplate (counts s))
      (cellVars poseidonTemplate (counts s))) :=
  fun _ _ h => shrink_eq_ok h

theorem shrink_complete :
    Complete (fun s => RowsFit s ∧ CellsFit s) degrees.shrink usizeValue
      (fun s => CompressionLean.Shrink.rustDegrees (rowVars poseidonTemplate (counts s))
        (cellVars poseidonTemplate (counts s))) :=
  complete_of shrink_sound ⟨emptyShape, empty_rows_fit, empty_cells_fit⟩ shrink_succeeds

end CompressionLeanRust
