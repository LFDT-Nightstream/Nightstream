import CompressionLean.Sim.Gkr
import CompressionLean.Sim.Views
import CompressionLean.Bridge.Early
import CompressionLean.Fields

/-!
Owns the simulation of the layer-1 verifier's early stage (`earlyStage` of
`Verifier.lean`, with the helpers of `Early.lean`) against the replay's
(`Ports/Replay/Stages.lean`, `Ports/Replay/Formulas.lean`).

- `earlyStage_tracks`: once the circuit's checks hold, the circuit's early
  stage tracks the replay's, and `IsEarlyOut` gives what later stages read.
- `Mle.lessThan_ensures`, `Mle.step_ensures`, `Mle.identity_ensures`,
  `Mle.chi_ensures`, `checkPair_ensures`, `observePairs_ensures`: the parts
  of `mle.rs` and `matrix.rs` that the late stage reads too.
- `gkrVerify_sizes`, `IsGkr.entries` and their helpers: the sizes of the
  replay's GKR claims and the entries of a GKR view, which the late stage
  reads too.
- The circuit and the replay may add or multiply in different orders. Their
  values agree in the challenge field `Cubic`, where `Ext.toK` is injective
  (`Ext.eq_of_toK`).
-/

namespace CompressionLean.Sim

open CompressionLean.Spec
open CompressionLean.Circuit
open CompressionLean.Lib
open CompressionLean.Replay (Word Ext)
open NightstreamFPrime.Spec (goldilocksModulus)
open Polynomial

variable {F : Type} [CommRing F]

/-! ## Exact arithmetic -/

/-- `K ⊗ Ext` values with one denotation are equal (`algebra.rs`). -/
private theorem IsKx.congr {c : Ctx F} {n : ℕ} {a : KxVar} {v v' : Replay.Kx} (h : IsKx c n a v)
    (e : v.toQ (K := Cubic) = v'.toQ) : IsKx c n a v' := by
  have re : v.re = v'.re := Ext.eq_of_toK (K := Cubic) (by simpa using congrArg QuadraticAlgebra.re e)
  have im : v.im = v'.im := Ext.eq_of_toK (K := Cubic) (by simpa using congrArg QuadraticAlgebra.im e)
  exact ⟨re ▸ h.1, im ▸ h.2⟩

/-! ## `mle.rs` in the replay -/

/-- One step of `less_than` at coordinate `t` (`mle.rs:36-55`). -/
private def lessThanStep (point : Array Ext) (bound t : ℕ) (acc : Ext × Ext) : Ext × Ext :=
  if (bound / 2 ^ t) % 2 = 1 then (acc.1 + acc.2 * Replay.oneMinus point[t]!, acc.2 * point[t]!)
  else (acc.1, acc.2 * Replay.oneMinus point[t]!)

/-- The scan of `less_than` (`mle.rs:36-55`) is a fold from the top coordinate. -/
private theorem lessThanScan_eq (point : Array Ext) (bound : ℕ) : ∀ s acc,
    Replay.lessThanScan point bound s acc =
      (List.range s).reverse.foldl (fun acc t => lessThanStep point bound t acc) acc
  | 0, _ => rfl
  | s + 1, (total, pre) => by
      rw [Replay.lessThanScan, List.range_succ, List.reverse_append, List.reverse_singleton, List.singleton_append,
        List.foldl_cons, ← lessThanScan_eq point bound s]
      unfold lessThanStep
      split_ifs <;> rfl

/-- Term `s` of `step`: `values[s]` times its interval (`mle.rs:57-72`). -/
private def stepTerm (gpoint : Array Ext) (starts : Array ℕ) (gvalues : Array Ext) (stop s : ℕ) : Ext :=
  gvalues[s]! * (Replay.lessThan gpoint (starts[s + 1]?.getD stop) - Replay.lessThan gpoint (starts[s]?.getD stop))

/-- `step` (`mle.rs:57-72`) is the sum of its terms; a value past the starts
has an empty interval. -/
private theorem step_eq (gpoint : Array Ext) (starts : Array ℕ) (gvalues : Array Ext) (stop : ℕ)
    (size : starts.size ≤ gvalues.size) :
    Replay.step gpoint (starts.toList.zip gvalues.toList) stop =
      Replay.sumRange gvalues.size (stepTerm gpoint starts gvalues stop) := by
  set segs := starts.toList.zip gvalues.toList
  have len : segs.length = starts.size := by simp [segs, List.length_zip]; omega
  have next : ∀ j, Replay.nextStart stop (segs.drop j) = starts[j]?.getD stop := by
    intro j
    by_cases hj : j < starts.size
    · rw [List.drop_eq_getElem_cons (by omega)]
      simp [Replay.nextStart, segs, Array.getElem?_eq_getElem hj]
    · rw [List.drop_eq_nil_of_le (by omega)]
      simp [Replay.nextStart, Array.getElem?_eq_none (by omega : starts.size ≤ j)]
  have tail : ∀ d j, starts.size - j = d → j ≤ starts.size →
      (Replay.stepSum gpoint stop (segs.drop j) (Replay.lessThan gpoint (starts[j]?.getD stop))).toK
        (K := Cubic) = ∑ s ∈ Finset.Ico j starts.size, (stepTerm gpoint starts gvalues stop s).toK := by
    intro d
    induction d with
    | zero =>
        intro j hd hj
        rw [List.drop_eq_nil_of_le (by omega), show j = starts.size by omega, Finset.Ico_self, Finset.sum_empty]
        exact Replay.Ext.toK_zero
    | succ d ih =>
        intro j hd hj
        have lt : j < segs.length := by omega
        rw [List.drop_eq_getElem_cons lt, List.getElem_zip, Replay.stepSum, next, Replay.Ext.toK_add,
          ih (j + 1) (by omega) (by omega), Finset.sum_eq_sum_Ico_succ_bot (show j < starts.size by omega),
          add_left_inj, stepTerm, Array.getElem_toList, getElem!_pos gvalues j (by omega)]
  have zero : ∀ s, starts.size ≤ s → (stepTerm gpoint starts gvalues stop s).toK (K := Cubic) = 0 := by
    intro s hs
    simp only [stepTerm, Array.getElem?_eq_none (by omega : starts.size ≤ s + 1),
      Array.getElem?_eq_none (by omega : starts.size ≤ s), Option.getD_none, Replay.Ext.toK_mul, Replay.Ext.toK_sub,
      sub_self, mul_zero]
  apply Ext.eq_of_toK (K := Cubic)
  rw [Replay.step, ← List.drop_zero (l := segs), next, tail _ 0 rfl (Nat.zero_le _), Bridge.toK_sumRange,
    ← Finset.sum_range_add_sum_Ico _ size, Finset.sum_eq_zero (fun s hs => zero s (Finset.mem_Ico.mp hs).1),
    add_zero, Finset.range_eq_Ico]

/-- `χ_r` (`mle.rs:84-107`) is the product of its factors. -/
private theorem toQ_chiList : ∀ (l : List (Word × Word)) (xs : List Ext),
    (Replay.chiList l xs).toQ (K := Cubic) = ∏ t ∈ Finset.range l.length, (Replay.chiFactor l[t]! xs[t]?).toQ
  | [], _ => by simp [Replay.chiList]
  | r :: rs, [] => by
      simp only [Replay.chiList, Replay.Kx.toQ_mul, toQ_chiList rs [], List.length_cons, Finset.prod_range_succ',
        List.getElem!_cons_zero, List.getElem!_cons_succ, List.getElem?_nil]
      rw [mul_comm]
  | r :: rs, x :: xs => by
      simp only [Replay.chiList, Replay.Kx.toQ_mul, toQ_chiList rs xs, List.length_cons, Finset.prod_range_succ',
        List.getElem!_cons_zero, List.getElem!_cons_succ, List.getElem?_cons_zero, List.getElem?_cons_succ]
      rw [mul_comm]

/-- A running product from one (`mle.rs:84-107`). -/
private theorem toQ_foldl_mul (f : ℕ → Replay.Kx) (k : ℕ) :
    ((List.range k).foldl (fun v t => v.mul (f t)) Replay.Kx.one).toQ (K := Cubic) =
      ∏ t ∈ Finset.range k, (f t).toQ := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Replay.Kx.toQ_mul, ih,
        Finset.prod_range_succ]

/-! ## Instruction shape -/

instance Mle.lessThan_appends (point : Array ExtVar) (bound : ℕ) : Appends (Mle.lessThan point bound) := by
  unfold Mle.lessThan; appends
instance Mle.step_appends (point : Array ExtVar) (starts : Array ℕ) (values : Array ExtVar) (stop : ℕ) :
    Appends (Mle.step point starts values stop) := by
  unfold Mle.step; appends
instance Mle.identity_appends (point : Array ExtVar) : Appends (Mle.identity point) := by
  unfold Mle.identity; appends
instance Mle.chi_appends (r : Array KVar) (point : Array ExtVar) : Appends (Mle.chi r point) := by
  unfold Mle.chi; appends
instance tableSum_appends (counts : Array Var) (bound : ℕ) (beta : ExtVar) : Appends (tableSum counts bound beta) := by
  unfold tableSum; appends
instance tuple_appends (index gamma re im : ExtVar) : Appends (tuple index gamma re im) := by unfold tuple; appends
instance checkPair_appends (x y : TreeClaimView) : Appends (checkPair x y) := by unfold checkPair; appends
instance matrixStep_appends (layout : Replay.Layout) (point : Array ExtVar) : Appends (matrixStep layout point) := by
  unfold matrixStep; appends
instance checkEarly_appends (layout : Replay.Layout) (r : Array KVar) (ch : EarlyChallengesView)
    (claims : Array TreeClaimView) (values : EarlyValuesView) : Appends (checkEarly layout r ch claims values) := by
  unfold checkEarly; appends
instance observePairs_appends (pairs : Array (ExtVar × ExtVar)) (d : Duplex) :
    Appends ((Circuit.observePairs pairs).run d) := by
  unfold Circuit.observePairs; appends
instance observeEarly_appends (values : EarlyValuesView) (d : Duplex) : Appends ((observeEarly values).run d) := by
  unfold observeEarly; appends

/-! ## `mle.rs` in the circuit -/

section Mle

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

/-- `less_than`: `Σ_{x < bound} eq(point, x)`, high bit first (`mle.rs:36-55`). -/
theorem Mle.lessThan_ensures {point : Array ExtVar} {gpoint : Array Ext} (bound : ℕ) (hp : IsExts c n point gpoint) :
    Ensures c n (Mle.lessThan point bound) fun r m => IsExt c m r (Replay.lessThan gpoint bound) := by
  unfold Mle.lessThan
  split
  · rename_i big
    refine extOne_ensures.post fun _ _ e => ?_
    rw [Replay.lessThan, ← hp.1, ← Nat.shiftRight_eq_div_pow, if_pos big]
    exact e
  · rename_i small
    set xs := (List.range point.size).reverse with hxs
    set G : ℕ → Ext × Ext := fun k => (xs.take k).foldl (fun acc t => lessThanStep gpoint bound t acc) (0, 1)
    refine extZero_ensures.bind fun total _ _ hzero => extOne_ensures.bind fun head _ _ hone => ?_
    refine (Ensures.loop (fun k (acc : ExtVar × ExtVar) m => IsExt c m acc.1 (G k).1 ∧ IsExt c m acc.2 (G k).2)
      ⟨hzero.mono, hone⟩ fun k hk acc m _ hacc => ?_).bind fun r _ _ hr => Ensures.pure ?_
    · obtain ⟨htotal, hhead⟩ := hacc
      have ht : xs[k] < point.size := by
        have mem : xs[k] ∈ (List.range point.size).reverse := List.getElem_mem hk
        rw [List.mem_reverse, List.mem_range] at mem
        exact mem
      have hx : IsExt c n point[xs[k]]! gpoint[xs[k]]! := hp.2 _ ht
      have G_succ : G (k + 1) = lessThanStep gpoint bound xs[k] (G k) := by
        simp only [G]
        rw [List.take_succ_eq_append_getElem hk, List.foldl_append, List.foldl_cons, List.foldl_nil]
      dsimp only
      refine (oneMinus_ensures hx.mono).bind fun low _ _ hlow => ?_
      split
      · rename_i bit
        have cond : (bound / 2 ^ xs[k]) % 2 = 1 := by simpa [Nat.testBit_eq_decide_div_mod_eq] using bit
        refine (extMul_ensures hhead.mono hlow).bind fun term _ _ hterm => ?_
        refine (extAdd_ensures htotal.mono hterm).bind fun total' _ _ htotal' => ?_
        refine (extMul_ensures hhead.mono hx.mono).bind fun head' _ _ hhead' => Ensures.pure ⟨rfl, ?_, ?_⟩ <;>
          rw [G_succ, lessThanStep, if_pos cond]
        · exact htotal'.mono
        · exact hhead'
      · rename_i bit
        have cond : ¬(bound / 2 ^ xs[k]) % 2 = 1 := by simpa [Nat.testBit_eq_decide_div_mod_eq] using bit
        refine (extMul_ensures hhead.mono hlow).bind fun head' _ _ hhead' => Ensures.pure ⟨rfl, ?_, ?_⟩ <;>
          rw [G_succ, lessThanStep, if_neg cond]
        · exact htotal.mono
        · exact hhead'
    · have e := hr.1
      simp only [G, List.take_length] at e
      rw [Replay.lessThan, ← hp.1, if_neg (by rw [← Nat.shiftRight_eq_div_pow]; exact small), lessThanScan_eq]
      exact e

/-- `step`: `values[s]` on `[starts[s], starts[s + 1])`, the last segment
ending at `stop` (`mle.rs:57-72`). A value past the starts adds nothing. -/
theorem Mle.step_ensures {point : Array ExtVar} {gpoint : Array Ext} {starts : Array ℕ} {values : Array ExtVar}
    {gvalues : Array Ext} (stop : ℕ) (hp : IsExts c n point gpoint) (hv : IsExts c n values gvalues)
    (size : starts.size ≤ values.size) :
    Ensures c n (Mle.step point starts values stop)
      fun r m => IsExt c m r (Replay.step gpoint (starts.toList.zip gvalues.toList) stop) := by
  set L : ℕ → Ext := fun s => Replay.lessThan gpoint (starts[s]?.getD stop)
  unfold Mle.step
  refine extZero_ensures.bind fun total _ _ hzero => ?_
  refine (Mle.lessThan_ensures _ hp.mono).bind fun below _ _ hbelow => ?_
  refine (Ensures.loopArray (fun k (acc : ExtVar × ExtVar) m =>
      IsExt c m acc.1 (Replay.sumRange k (stepTerm gpoint starts gvalues stop)) ∧ IsExt c m acc.2 (L k))
    ⟨hzero.mono, hbelow⟩ fun k hk acc m _ hacc => ?_).bind fun r _ _ hr => Ensures.pure ?_
  · obtain ⟨htotal, hlow⟩ := hacc
    have hk' : k < values.size := by simpa using hk
    rw [show values.zipIdx[k] = (values[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine (Mle.lessThan_ensures _ hp.mono).bind fun upper _ _ hupper => ?_
    refine (extSub_ensures hupper hlow.mono).bind fun interval _ _ hinterval => ?_
    refine (extMul_ensures (hv.2 k hk').mono hinterval).bind fun term _ _ hterm => ?_
    refine (extAdd_ensures htotal.mono hterm).bind fun total' _ _ htotal' => ?_
    exact Ensures.pure ⟨rfl, by rw [sumRange_succ]; exact htotal', hupper.mono⟩
  · have e := hr.1
    rw [Array.size_zipIdx, hv.1] at e
    rw [step_eq gpoint starts gvalues stop (by rw [← hv.1]; exact size)]
    exact e

/-- `identity`: `Σ_t 2^t point_t` (`mle.rs:74-82`). -/
theorem Mle.identity_ensures {point : Array ExtVar} {gpoint : Array Ext} (hp : IsExts c n point gpoint) :
    Ensures c n (Mle.identity point) fun r m => IsExt c m r (Replay.identityValue gpoint) := by
  unfold Mle.identity
  refine extZero_ensures.bind fun total _ _ hzero => ?_
  refine (Ensures.loopArray
      (fun k acc m => IsExt c m acc ((gpoint.reverse.toList.take k).foldl (fun t x => t + t + x) 0)) hzero
    fun k hk acc m _ hacc => ?_).bind fun r _ _ hr => Ensures.pure ?_
  · have hx : IsExt c n point.reverse[k] gpoint.reverse[k]! := by
      have := hp.reverse.2 k hk
      rwa [getElem!_pos point.reverse k hk] at this
    refine (extAdd_ensures hacc hacc).bind fun doubled _ _ hdoubled => ?_
    refine (extAdd_ensures hdoubled hx.mono).bind fun total' _ _ htotal' => Ensures.pure ⟨rfl, ?_⟩
    have hk' : k < gpoint.reverse.toList.length := by simpa [← hp.1] using hk
    rw [List.take_succ_eq_append_getElem hk', List.foldl_append, List.foldl_cons, List.foldl_nil,
      Array.getElem_toList, ← getElem!_pos]
    exact htotal'
  · rw [List.take_of_length_le (by simp [hp.1])] at hr
    rw [Replay.identityValue, Array.foldr_eq_foldl_reverse, ← Array.foldl_toList]
    exact hr

/-- `chi`: `Π_{t<n} eq(r_t, point_t) · Π_{t≥n} (1 - r_t)` in `K ⊗ Ext`
(`mle.rs:84-107`). -/
theorem Mle.chi_ensures {r : Array KVar} {gr : Array (Word × Word)} {point : Array ExtVar} {gpoint : Array Ext}
    (hrs : r.size = gr.size) (hr : ∀ i < r.size, IsWord c n r[i]!.1 gr[i]!.1 ∧ IsWord c n r[i]!.2 gr[i]!.2)
    (hp : IsExts c n point gpoint) :
    Ensures c n (Mle.chi r point) fun res m => IsKx c m res (Replay.chi gr gpoint) := by
  set V : ℕ → Replay.Kx := fun k =>
    (List.range k).foldl (fun v t => v.mul (Replay.chiFactor gr[t]! gpoint[t]?)) Replay.Kx.one
  unfold Mle.chi
  refine KxVar.one_ensures.bind fun total _ _ hone => ?_
  refine (Ensures.loopArray (fun k acc m => IsKx c m acc (V k)) hone fun k hk acc m _ hacc => ?_).bind
    fun res _ _ hres => Ensures.pure ?_
  · have hk' : k < r.size := by simpa using hk
    have V_succ : V (k + 1) = (V k).mul (Replay.chiFactor gr[k]! gpoint[k]?) := by
      simp only [V]
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    rw [show r.zipIdx[k] = (r[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine (KxVar.fromK_ensures (hr k hk').1.mono (hr k hk').2.mono).bind fun rt _ _ hrt => ?_
    split
    · rename_i x hpt
      have inside : k < point.size := by
        rw [Array.getElem?_eq_some_iff] at hpt
        exact hpt.1
      have hx : IsExt c n x gpoint[k]! := by
        have := hp.2 k inside
        rwa [getElem!_pos point k inside, (Array.getElem?_eq_some_iff.mp hpt).2] at this
      have some : gpoint[k]? = some gpoint[k]! := by
        rw [getElem!_pos gpoint k (hp.1 ▸ inside), Array.getElem?_eq_getElem]
      refine (extAdd_ensures hx.mono hx.mono).bind fun twice _ _ htwice => ?_
      refine extOne_ensures.bind fun one _ _ hone' => ?_
      refine (extSub_ensures htwice.mono hone').bind fun slope _ _ hslope => ?_
      refine (KxVar.scale_ensures hrt.mono hslope).bind fun scaled _ _ hscaled => ?_
      refine (oneMinus_ensures hx.mono).bind fun low _ _ hlow => ?_
      refine (extAdd_ensures hscaled.1.mono hlow).bind fun re _ _ hre => ?_
      refine (Ensures.pure (Q := fun f m' => IsKx c m' f (Replay.chiFactor gr[k]! gpoint[k]?)) ?_).bind
        fun factor _ _ hfactor => ?_
      · rw [some]
        exact ⟨hre, hscaled.2.mono⟩
      refine (KxVar.mul_ensures hacc.mono hfactor).bind fun total' _ _ htotal' => Ensures.pure ⟨rfl, ?_⟩
      rw [V_succ]
      exact htotal'
    · rename_i hpt
      have none : gpoint[k]? = none := by
        rw [Array.getElem?_eq_none_iff, ← hp.1]
        exact Array.getElem?_eq_none_iff.mp hpt
      refine (KxVar.oneMinus_ensures hrt).bind fun factor _ _ hfactor => ?_
      refine (KxVar.mul_ensures (gb := Replay.chiFactor gr[k]! gpoint[k]?) hacc.mono ?_).bind
        fun total' _ _ htotal' => Ensures.pure ⟨rfl, ?_⟩
      · rw [none]
        exact hfactor
      rw [V_succ]
      exact htotal'
  · refine hres.congr ?_
    rw [Array.size_zipIdx, hrs, toQ_foldl_mul, Replay.chi, toQ_chiList, Array.length_toList]
    refine Finset.prod_congr rfl fun t _ => ?_
    rw [Array.getElem!_toList, Array.getElem?_toList]

end Mle


/-! ## The early stage in the replay -/

open CompressionLean.Replay (Relation TreeShape)

/-- Equal values pass the replay's equality test (`field.rs`). -/
theorem ext_beq_of_eq {a b : Ext} (e : a = b) : (a == b) = true := e ▸ Ext.beq_self a

/-- A nonzero value passes the replay's zero test (`field.rs`). -/
private theorem ext_not_isZero {a : Ext} (ne : a ≠ 0) : (!a.isZero) = true := by
  cases h : a.isZero
  · rfl
  · exact absurd (Replay.Ext.eq_of_beq h) ne

/-- Term `t` of `table_sum`: `m_t/(β - (t - H))` (`norm.rs:36-49`). -/
private def tableTerm (bound : ℕ) (counts : Array Word) (beta : Ext) (slot : ℕ) : Ext :=
  (beta - Ext.ofGl (Word.ofNat slot - Word.ofNat bound)).inv.scale counts[slot]!

/-- `table_sum` (`norm.rs:36-49`) accepts when no slot is a pole. -/
private theorem tableSum_ok {bound : ℕ} {counts : Array Word} {beta : Ext}
    (poles : ∀ s < counts.size, beta - Ext.ofGl (Word.ofNat s - Word.ofNat bound) ≠ 0) :
    Replay.tableSum bound counts beta = .ok (Replay.sumRange counts.size (tableTerm bound counts beta)) := by
  unfold Replay.tableSum
  suffices h : ∀ k ≤ counts.size, (List.range k).foldlM (Replay.tableSumStep bound counts beta) 0 =
      .ok (Replay.sumRange k (tableTerm bound counts beta)) from h _ le_rfl
  intro k hk
  induction k with
  | zero => rfl
  | succ k ih =>
      rw [List.range_succ, List.foldlM_append, ih (by omega), sumRange_succ]
      simp [Replay.tableSumStep, Replay.ensure, ext_not_isZero (poles k (by omega)), tableTerm]

/-- `check_early` (`matrix.rs:261-317`) accepts when its pair checks pass and
its leaf claims hold. -/
private theorem checkEarly_ok {layout : Replay.Layout} {r : Array (Word × Word)} {ch : Replay.EarlyChallenges}
    {claims : Replay.GkrClaims} {v : Replay.EarlyValues}
    (rows : Replay.pairCheck claims.roots[1]! claims.roots[2]! "row lookup" = .ok ())
    (slots : Replay.pairCheck claims.roots[3]! claims.roots[4]! "slot scatter" = .ok ())
    (runLeaf : claims.leafValue 1 0 = ch.lookup.1 - Replay.tuple ch.lookup.2 v.row v.eRe v.eIm)
    (rowLeaf : claims.leafValue 2 1 = ch.lookup.1 - Replay.tuple ch.lookup.2
      (Replay.identityValue claims.leafPoint[2]!) (Replay.chi r claims.leafPoint[2]!).re
      (Replay.chi r claims.leafPoint[2]!).im)
    (scatterLeaf : claims.leafValue 3 1 = v.eRe + ch.scatter.2 * v.eIm)
    (pairLeaf : claims.leafValue 4 0 = Replay.sumRange (min v.pairs.size
      (Replay.eqTable ((claims.leafPoint[4]!).extract layout.slotVariables (claims.leafPoint[4]!).size)).size)
      fun j => (Replay.eqTable ((claims.leafPoint[4]!).extract layout.slotVariables
        (claims.leafPoint[4]!).size))[j]! * ((v.pairs[j]!).1 + ch.scatter.2 * (v.pairs[j]!).2))
    (indexLeaf : claims.leafValue 4 1 = ch.scatter.1 - Replay.identityValue claims.leafPoint[4]!) :
    Replay.checkEarly layout r ch claims v = .ok (Bridge.earlyOpeningsOf layout ch claims v) := by
  unfold Replay.checkEarly
  rw [rows, slots]
  simp [Replay.ensure, ext_beq_of_eq runLeaf, ext_beq_of_eq rowLeaf, ext_beq_of_eq scatterLeaf,
    ext_beq_of_eq pairLeaf, ext_beq_of_eq indexLeaf]
  rfl

/-- The trees of `GkrView::read` (`gkr.rs:608-656`): none is empty, and each
ends by the last layer. -/
theorem depth_bounds {proof : Replay.GkrProof} {shapes : Array TreeShape}
    (read : Replay.readGkr proof shapes = .ok ()) {u : ℕ} (inside : u < shapes.size) :
    1 ≤ shapes[u]!.depth ∧ shapes[u]!.depth ≤ Replay.layerCount shapes := by
  refine ⟨?_, ?_⟩
  · unfold Replay.readGkr at read
    split_ifs at read with bad
    · obtain ⟨_, thrown, _⟩ := Bridge.Except.bind_ok read
      cases thrown
    · simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or] at bad
      have := bad.1.1
      simp only [Array.any_eq_true, not_exists, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or] at this
      rw [getElem!_pos shapes u inside]
      have := (this u inside).1
      omega
  · unfold Replay.layerCount
    suffices h : ∀ l : List ℕ, u ∈ l → shapes[u]!.depth ≤ l.foldr (fun t d => max d shapes[t]!.depth) 0 from
      h _ (List.mem_range.mpr inside)
    intro l member
    induction l with
    | nil => cases member
    | cons t rest ih =>
        rw [List.foldr_cons]
        rcases List.mem_cons.mp member with rfl | inside
        · exact le_max_right _ _
        · exact (ih inside).trans (le_max_left _ _)

/-- A tree that ends at layer `k` gets one leaf value per part
(`gkr.rs:727-740`). -/
theorem reducedOf_size {shapes : Array TreeShape} {k u : ℕ} {layer : Replay.GkrLayer} {tau : Ext}
    (ok : Replay.layerShapeOk shapes k layer = true) (inside : u < shapes.size) (ends : shapes[u]!.depth = k) :
    (Replay.reducedOf ((Replay.activeTrees shapes k).zip layer.children).toList tau u).size =
      shapes[u]!.factors + 1 := by
  simp only [Replay.layerShapeOk, Bool.and_eq_true, beq_iff_eq] at ok
  obtain ⟨⟨⟨-, -⟩, childrenSize⟩, childSizes⟩ := ok
  have active : u ∈ (Replay.activeTrees shapes k).toList := by
    have ends' : shapes[u].depth = k := by rw [← getElem!_pos shapes u inside]; exact ends
    unfold Replay.activeTrees
    simp [inside, ends']
  obtain ⟨s, hs, hsu⟩ := List.mem_iff_getElem.mp active
  have member : ((Replay.activeTrees shapes k)[s]'(by simpa using hs), layer.children[s]'(by
      rw [childrenSize]; simpa using hs)) ∈ ((Replay.activeTrees shapes k).zip layer.children).toList := by
    rw [Array.toList_zip]
    exact List.mem_iff_getElem.mpr ⟨s, by simp [childrenSize]; simpa using hs, by simp⟩
  unfold Replay.reducedOf
  split
  · rename_i pair found
    have key : pair.1 = u := by simpa using List.find?_some found
    have size := Array.all_eq_true'.mp childSizes pair (Array.mem_toList_iff.mp (List.mem_of_find?_eq_some found))
    simp only [beq_iff_eq] at size
    rw [Array.size_map, Array.size_range, size, key, Replay.partFactors, if_pos ends]
    omega
  · rename_i missing
    exact absurd (List.find?_eq_none.mp missing _ member) (by simp [← hsu])

/-- The roots, leaf values and leaf points after the GKR layers from layer `k`
(`gkr.rs:694-746`). -/
theorem gkrLayers_sizes {shapes : Array TreeShape} :
    ∀ (layers : List Replay.GkrLayer) (k : ℕ) (cl g : Replay.GkrClaims) (t t' : Replay.Transcript),
      Runs (Replay.gkrLayers shapes k layers cl) t g t' → 1 ≤ k →
      (∀ i (h : i < layers.length), Replay.layerShapeOk shapes (k + i) layers[i] = true) →
      cl.leafValues.size = shapes.size → cl.leafPoint.size = shapes.size →
      g.roots = cl.roots ∧ g.leafValues.size = shapes.size ∧ g.leafPoint.size = shapes.size ∧
        ∀ u < shapes.size, ((g.leafValues[u]!).size, (g.leafPoint[u]!).size) =
          if k ≤ shapes[u]!.depth ∧ shapes[u]!.depth < k + layers.length then
            (shapes[u]!.factors + 1, shapes[u]!.depth) else ((cl.leafValues[u]!).size, (cl.leafPoint[u]!).size)
  | [], k, cl, g, t, t', run, _, _, vsize, psize => by
      rw [Replay.gkrLayers] at run
      obtain ⟨rfl, rfl⟩ := Bridge.run_pure run
      exact ⟨rfl, vsize, psize, fun u _ => by rw [if_neg (by simp)]⟩
  | layer :: rest, k, cl, g, t, t', run, one, ok, vsize, psize => by
      rw [Replay.gkrLayers] at run
      obtain ⟨_, t1, -, run⟩ := Bridge.run_bind run
      obtain ⟨_, t2, -, run⟩ := Bridge.run_bind run
      obtain ⟨result, t3, sum, run⟩ := Bridge.run_bind run
      obtain ⟨_, t4, -, run⟩ := Bridge.run_bind run
      obtain ⟨_, t5, -, run⟩ := Bridge.run_bind run
      obtain ⟨tau, t6, -, run⟩ := Bridge.run_bind run
      have ok₀ : Replay.layerShapeOk shapes k layer = true := ok 0 (by simp)
      have rounds : result.1.size = k - 1 := by
        have := Bridge.sumcheckRounds_size _ _ _ _ _ _ sum
        have shape := ok₀
        simp only [Replay.layerShapeOk, Bool.and_eq_true, beq_iff_eq] at shape
        rw [this, Array.length_toList, shape.1.1.1]
        simp
      obtain ⟨roots, vsize', psize', values⟩ := gkrLayers_sizes rest (k + 1) _ g t6 t' run (by omega)
        (fun i h => by
          have := ok (i + 1) (by simp; omega)
          simpa [Nat.add_assoc, Nat.add_comm 1 i] using this) (by simp [Replay.advance, vsize])
        (by simp [Replay.advance, psize])
      refine ⟨roots, vsize', psize', fun u hu => ?_⟩
      rw [values u hu]
      have adv : (Replay.advance shapes k cl ((Replay.activeTrees shapes k).zip layer.children).toList result.1
          tau).leafValues[u]! = if shapes[u]!.depth = k then
            Replay.reducedOf ((Replay.activeTrees shapes k).zip layer.children).toList tau u
          else cl.leafValues[u]! := by
        simp only [Replay.advance]
        exact getElem!_map_range _ _ (by omega)
      have advPoint : (Replay.advance shapes k cl ((Replay.activeTrees shapes k).zip layer.children).toList result.1
          tau).leafPoint[u]! = if shapes[u]!.depth = k then result.1.push tau else cl.leafPoint[u]! := by
        simp only [Replay.advance]
        exact getElem!_map_range _ _ (by omega)
      rw [adv, advPoint]
      by_cases e : shapes[u]!.depth = k
      · rw [if_neg (by omega), if_pos e, if_pos e, if_pos (by simp; omega), reducedOf_size ok₀ hu e, Array.size_push,
          rounds, e, Nat.sub_add_cancel one]
      · rw [if_neg e, if_neg e]
        simp only [List.length_cons]
        split_ifs <;> first | rfl | omega

/-- The roots of `verify` (`gkr.rs:679-751`) are the proof's, and each tree
has one leaf value per part and a leaf point of its depth. -/
theorem gkrVerify_sizes {proof : Replay.GkrProof} {shapes : Array TreeShape} {t t' : Replay.Transcript}
    {g : Replay.GkrClaims} (read : Replay.readGkr proof shapes = .ok ())
    (run : Runs (Replay.gkrVerify proof shapes) t g t') :
    g.roots = proof.roots ∧ ∀ u < shapes.size,
      (g.leafValues[u]!).size = shapes[u]!.factors + 1 ∧ (g.leafPoint[u]!).size = shapes[u]!.depth := by
  obtain ⟨-, layersSize, layersOk⟩ := Bridge.readGkr_ok read
  unfold Replay.gkrVerify at run
  obtain ⟨_, t1, -, run⟩ := Bridge.run_bind run
  obtain ⟨roots, -, -, values⟩ := gkrLayers_sizes proof.layers.toList 1 _ g t1 t' run le_rfl (fun i h => by
    rw [Nat.add_comm]
    simpa using layersOk i (by simpa using h)) (by simp) (by simp)
  refine ⟨roots, fun u hu => ?_⟩
  have bounds := depth_bounds read hu
  have e := values u hu
  rw [if_pos ⟨bounds.1, by rw [Array.length_toList, layersSize]; omega⟩] at e
  exact Prod.mk.inj e


/-- Tree `t`'s claim is the replay's (`TreeClaim`, `gkr.rs:70-78`). -/
private def IsTree (c : Ctx F) (n : ℕ) (v : TreeClaimView) (g : Replay.GkrClaims) (t : ℕ) : Prop :=
  IsExt c n v.root.1 (g.roots[t]!).1 ∧ IsExt c n v.root.2 (g.roots[t]!).2 ∧ IsExts c n v.point g.leafPoint[t]! ∧
    IsExts c n v.values g.leafValues[t]!

/-- `IsTree` survives later values (`gkr.rs:70-78`). -/
private theorem IsTree.mono {c : Ctx F} {n m : ℕ} {v : TreeClaimView} {g : Replay.GkrClaims} {t : ℕ}
    (h : IsTree c n v g t) (le : n ≤ m := by omega) : IsTree c m v g t :=
  ⟨h.1.mono le, h.2.1.mono le, h.2.2.1.mono le, h.2.2.2.mono le⟩

/-- Leaf value `j` of tree `t` (`gkr.rs:70-78`). -/
private theorem IsTree.value {c : Ctx F} {n : ℕ} {v : TreeClaimView} {g : Replay.GkrClaims} {t j : ℕ}
    (h : IsTree c n v g t) (hj : j < (g.leafValues[t]!).size) : IsExt c n v.values[j]! (g.leafValue t j) :=
  h.2.2.2.2 j (h.2.2.2.1 ▸ hj)

/-- What `check_early` leaves to openings (`matrix.rs:115-123`). -/
private def IsOpenings (c : Ctx F) (m : ℕ) (v : EarlyOpeningsView) (o : Replay.EarlyOpenings) : Prop :=
  IsExts c m v.runPoint o.runPoint ∧ IsExts c m v.pairPoint o.pairPoint ∧ IsExts c m v.rowPoint o.rowPoint ∧
    IsExt c m v.mult o.mult ∧ IsExt c m v.setup.1 o.rowClaim ∧ IsExt c m v.setup.2.1 o.slotClaim ∧
    IsExt c m v.setup.2.2 o.coefClaim

/-- The claims after the norm tree (`verifier.rs:166`). -/
private theorem extract_one {α : Type} [Inhabited α] (xs : Array α) (i : ℕ) :
    (xs.extract 1 xs.size)[i]! = xs[i + 1]! := by
  by_cases hi : i + 1 < xs.size
  · rw [getElem!_pos _ i (by simp; omega), Array.getElem_extract, getElem!_pos xs (i + 1) hi]
    simp [Nat.add_comm]
  · rw [getElem!_neg _ i (by simp; omega), getElem!_neg xs (i + 1) hi]

/-- The entries of two related arrays (`verifier.rs:43-122`). -/
theorem forall₂_entries {α β : Type} [Inhabited β] {R : α → β → Prop} {xs : Array α} {ys : Array β}
    (h : List.Forall₂ R xs.toList ys.toList) : xs.size = ys.size ∧ ∀ i (hi : i < xs.size), R xs[i] ys[i]! := by
  obtain ⟨len, get⟩ := List.forall₂_iff_get.mp h
  simp only [Array.length_toList] at len
  refine ⟨len, fun i hi => ?_⟩
  have := get i (by simpa using hi) (by simpa [← len] using hi)
  rw [getElem!_pos ys i (by omega)]
  simpa using this

/-- A GKR view's roots and layers, entry by entry (`gkr.rs:591-656`). -/
theorem IsGkr.entries {c : Ctx F} {n : ℕ} {v : GkrView} {gp : Replay.GkrProof} (h : IsGkr c n v gp) :
    (v.roots.size = gp.roots.size ∧ ∀ i < v.roots.size,
      IsExt c n (v.roots[i]!).1 (gp.roots[i]!).1 ∧ IsExt c n (v.roots[i]!).2 (gp.roots[i]!).2) ∧
    (v.layers.size = gp.layers.size ∧ ∀ i (hi : i < v.layers.size),
      (v.layers[i].rounds.size = (gp.layers[i]!).rounds.size ∧
        ∀ r < v.layers[i].rounds.size, IsExts c n v.layers[i].rounds[r]! (gp.layers[i]!).rounds[r]!) ∧
      (v.layers[i].children.size = (gp.layers[i]!).children.size ∧
        ∀ s < v.layers[i].children.size, IsExts c n v.layers[i].children[s]! (gp.layers[i]!).children[s]!)) := by
  obtain ⟨roots, layers⟩ := h
  obtain ⟨rs, re⟩ := forall₂_entries roots
  obtain ⟨ls, le⟩ := forall₂_entries layers
  refine ⟨⟨rs, fun i hi => ?_⟩, ls, fun i hi => ?_⟩
  · rw [getElem!_pos v.roots i hi]
    exact re i hi
  · obtain ⟨rounds, children⟩ := le i hi
    obtain ⟨rs', re'⟩ := forall₂_entries rounds
    obtain ⟨cs', ce'⟩ := forall₂_entries children
    exact ⟨⟨rs', fun r hr => by rw [getElem!_pos _ r hr]; exact re' r hr⟩, cs',
      fun s hs => by rw [getElem!_pos _ s hs]; exact ce' s hs⟩

/-! ## The root binding -/

instance observeRoot_appends (seed : Array Word) (root : DigestVar) (d : Duplex) :
    Appends ((Circuit.observeRoot seed root).run d) := by
  unfold Circuit.observeRoot; appends
instance earlyStage_appends (rel : Replay.Relation) (statement : ClaimView) (proof : ProofView) (d : Duplex) :
    Appends ((Circuit.earlyStage rel statement proof).run d) := by
  unfold Circuit.earlyStage; appends

section Root

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

/-- `observe_root` (`whir/mod.rs:221-232`): the separator words as constants,
then the root, as the replay's two absorptions. -/
theorem observeRoot_tracks {seed : Array Word} {root : DigestVar} {groot : Array Word} (hp : Permutes c.perm)
    (hr : IsDigest c n root groot) :
    Tracks c n (Circuit.observeRoot seed root) (Replay.observeAll seed >>= fun _ => Replay.observeAll groot)
      fun _ _ _ => True := by
  unfold Circuit.observeRoot
  exact Tracks.bind (Duplex.observeConsts_tracks hp seed) fun _ _ _ _ _ =>
    Duplex.observeSlice_tracks hp (IsWords.ofFn hr.1 fun lane => (hr.2 lane).mono)

end Root

/-- What later stages read of the early stage (`verifier.rs:153-168`): the
norm tree's leaf point, `z~` at it (`β - q~`), and the matrix openings, with
the run columns `(i, σ, m)` at the run point. -/
def IsEarlyOut (c : Ctx F) (m : ℕ) (v : EarlyOut) (o : Replay.EarlyOut) : Prop :=
  IsExts c m v.normPoint o.gkr.leafPoint[0]! ∧ IsExt c m v.zLeaf o.zLeaf ∧
    IsExts c m v.openings.runPoint o.openings.runPoint ∧ IsExts c m v.openings.pairPoint o.openings.pairPoint ∧
    IsExts c m v.openings.rowPoint o.openings.rowPoint ∧ IsExt c m v.openings.mult o.openings.mult ∧
    IsExt c m v.openings.setup.1 o.openings.rowClaim ∧ IsExt c m v.openings.setup.2.1 o.openings.slotClaim ∧
    IsExt c m v.openings.setup.2.2 o.openings.coefClaim

/-- `IsEarlyOut` survives later values (`verifier.rs:153-168`). -/
theorem IsEarlyOut.mono {c : Ctx F} {m m' : ℕ} {v : EarlyOut} {o : Replay.EarlyOut} (h : IsEarlyOut c m v o)
    (le : m ≤ m' := by omega) : IsEarlyOut c m' v o := by
  obtain ⟨norm, z, run, pair, row, mult, rowClaim, slotClaim, coefClaim⟩ := h
  exact ⟨norm.mono le, z.mono le, run.mono le, pair.mono le, row.mono le, mult.mono le, rowClaim.mono le,
    slotClaim.mono le, coefClaim.mono le⟩

/-- `forM` over a list in a transcript (`verifier.rs:318-331`): entry `x`
absorbs the words `f g` of its replay value `g`. -/
private theorem forMList {c : Ctx F} {α β : Type} {body : α → Transcript PUnit} {R : ℕ → α → β → Prop}
    (f : β → List Word) (mono : ∀ m m' x g, R m x g → m ≤ m' → R m' x g)
    (hb : ∀ {n : ℕ} {d : Duplex} {r : Replay.Duplex} x g, IsDuplex c n d r → R n x g →
      Ensures c n ((body x).run d) fun res m => IsDuplex c m res.2 ((f g).foldl Replay.Duplex.observe r))
    (hbody : ∀ x d, Appends ((body x).run d)) :
    ∀ (l : List α) (gl : List β) {n : ℕ} {d : Duplex} {r : Replay.Duplex}, IsDuplex c n d r →
      List.Forall₂ (R n) l gl →
      Ensures c n ((l.foldlM (fun _ x => body x) PUnit.unit).run d)
        fun res m => IsDuplex c m res.2 ((gl.flatMap f).foldl Replay.Duplex.observe r)
  | [], [], _, _, _, hd, _ => by
      rw [List.foldlM_nil, StateT.run_pure]
      exact Ensures.pure hd
  | x :: l, g :: gl, n, d, r, hd, hl => by
      obtain ⟨hxg, hrest⟩ := List.forall₂_cons.mp hl
      rw [List.foldlM_cons]
      refine (hb x g hd hxg).tbind (fun _ e m le he => ?_) (hbody x d)
        fun s e => Appends.tfoldlM_list (fun _ x d => hbody x d) l s e
      refine (forMList f mono hb hbody l gl he (hrest.imp fun _ _ h => mono _ _ _ _ h le)).post fun _ _ q => ?_
      rw [List.flatMap_cons, List.foldl_append]
      exact q
  | [], _ :: _, _, _, _, _, hl => nomatch hl
  | _ :: _, [], _, _, _, _, hl => nomatch hl

/-- The entries of an `IsExts` (`verifier.rs:318-324`). -/
private theorem isExts_forall₂ {c : Ctx F} {n : ℕ} {xs : Array ExtVar} {gs : Array Ext} (h : IsExts c n xs gs) :
    List.Forall₂ (IsExt c n) xs.toList gs.toList := by
  rw [List.forall₂_iff_get]
  refine ⟨by simp [h.1], fun i h₁ h₂ => ?_⟩
  have := h.2 i (by simpa using h₁)
  rw [getElem!_pos xs i (by simpa using h₁), getElem!_pos gs i (by simpa using h₂)] at this
  simpa using this

section Stage

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

/-- `check_pair` (`matrix.rs:402-415`): once its checks hold, the replay's
`pairCheck` accepts. -/
theorem checkPair_ensures {x y : TreeClaimView} {gx gy : Ext × Ext} (name : String) (hx : IsPair c n x.root gx)
    (hy : IsPair c n y.root gy) :
    Ensures c n (checkPair x y) fun _ _ => Replay.pairCheck gx gy name = .ok () := by
  obtain ⟨⟨px, qx⟩, xp, xv⟩ := x
  obtain ⟨⟨py, qy⟩, yp, yv⟩ := y
  obtain ⟨hpx, hqx⟩ := hx
  obtain ⟨hpy, hqy⟩ := hy
  unfold checkPair
  dsimp only at hpx hqx hpy hqy ⊢
  refine (extInv_ensures hqx).bind fun _ _ _ hix => ?_
  refine (extInv_ensures hqy.mono).bind fun _ _ _ hiy => ?_
  refine (extMul_ensures hpx.mono hqy.mono).bind fun left _ _ hleft => ?_
  refine (extMul_ensures hpy.mono hqx.mono).bind fun right _ _ hright => ?_
  refine (assertExtEqual_ensures hleft.mono hright).post fun _ _ q => ?_
  unfold Replay.pairCheck
  simp [Replay.ensure, ext_not_isZero hix.1, ext_not_isZero hiy.1, ext_beq_of_eq q]
  rfl

/-- `table_sum` (`norm.rs:36-49`): once the inverse checks hold, no slot is a
pole, and the sum is the replay's. -/
private theorem tableSum_ensures {counts : Array Var} {gcounts : Array Word} {beta : ExtVar} {gbeta : Ext}
    (bound : ℕ) (hc : IsWords c n counts gcounts) (hb : IsExt c n beta gbeta) :
    Ensures c n (tableSum counts bound beta) fun r m =>
      Replay.tableSum bound gcounts gbeta = .ok (Replay.sumRange gcounts.size (tableTerm bound gcounts gbeta)) ∧
        IsExt c m r (Replay.sumRange gcounts.size (tableTerm bound gcounts gbeta)) := by
  have size := hc.1
  unfold tableSum
  refine extZero_ensures.bind fun total _ _ hzero => ?_
  rw [← Array.forIn_toList]
  refine (Ensures.loop (fun k acc m => (∀ s < k, gbeta - Ext.ofGl (Word.ofNat s - Word.ofNat bound) ≠ 0) ∧
      IsExt c m acc (Replay.sumRange k (tableTerm bound gcounts gbeta))) ⟨fun s hs => absurd hs (by omega), hzero⟩
    fun k hk acc m _ hacc => ?_).bind fun r _ _ hr => Ensures.pure ?_
  · obtain ⟨poles, htotal⟩ := hacc
    have hk' : k < counts.size := by simpa using hk
    rw [show counts.zipIdx.toList[k] = (counts[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine (extConstant_ensures (Ext.ofGl (Word.ofNat k - Word.ofNat bound))).bind fun value _ _ hvalue => ?_
    refine (extSub_ensures hb.mono hvalue).bind fun den _ _ hden => ?_
    refine (extInv_ensures hden).bind fun inv _ _ hinv => ?_
    refine (extScale_ensures hinv.2 (hc.get (size ▸ hk')).mono).bind fun term _ _ hterm => ?_
    refine (extAdd_ensures htotal.mono hterm).bind fun total' _ _ htotal' =>
      Ensures.pure ⟨rfl, fun s hs => ?_, by rw [sumRange_succ]; exact htotal'⟩
    rcases Nat.lt_succ_iff_lt_or_eq.mp hs with lt | rfl
    · exact poles s lt
    · exact hinv.1
  · have e := hr
    rw [Array.length_toList, Array.size_zipIdx, size] at e
    exact ⟨tableSum_ok e.1, e.2⟩

/-- `observe_pairs` (`verifier.rs:326-331`): the replay's `observeExts` of the
flattened pairs. -/
theorem observePairs_ensures {d : Duplex} {r : Replay.Duplex} {pairs : Array (ExtVar × ExtVar)}
    {gpairs : Array (Ext × Ext)} (hp : Permutes c.perm) (hd : IsDuplex c n d r)
    (hv : List.Forall₂ (IsPair c n) pairs.toList gpairs.toList) :
    Ensures c n ((Circuit.observePairs pairs).run d) fun res m => IsDuplex c m res.2
      (((Replay.flatPairs gpairs).flatMap fun g => #[g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe r) := by
  unfold Circuit.observePairs Array.forM
  rw [← Array.foldlM_toList]
  refine (forMList (R := fun m => IsPair c m) (fun g => [g.1.c0, g.1.c1, g.1.c2, g.2.c0, g.2.c1, g.2.c2])
    (fun _ _ _ _ h le => h.mono le) (fun x g hd hx => ?_) (fun x _ => ?_) pairs.toList gpairs.toList hd hv).post
    fun _ _ q => ?_
  · obtain ⟨a, b⟩ := x
    dsimp only
    rw [StateT.run_bind]
    simp only [List.foldl_cons, List.foldl_nil]
    exact (Duplex.observeExt_ensures hp hd hx.1).bind fun _ _ _ hd' => Duplex.observeExt_ensures hp hd' hx.2.mono
  · obtain ⟨a, b⟩ := x
    dsimp only
    appends
  · have e : ((Replay.flatPairs gpairs).flatMap fun g => #[g.c0, g.c1, g.c2]).toList =
        gpairs.toList.flatMap fun g => [g.1.c0, g.1.c1, g.1.c2, g.2.c0, g.2.c1, g.2.c2] := by
      rw [Array.toList_flatMap, Replay.flatPairs, Array.toList_flatMap, List.flatMap_assoc]
      simp
    rw [← Array.foldl_toList, e]
    exact q

/-- `observe_pairs` tracks the replay's `observeExts` of the flattened pairs. -/
theorem observePairs_tracks {pairs : Array (ExtVar × ExtVar)} {gpairs : Array (Ext × Ext)} (hp : Permutes c.perm)
    (hv : List.Forall₂ (IsPair c n) pairs.toList gpairs.toList) :
    Tracks c n (Circuit.observePairs pairs) (Replay.observeExts (Replay.flatPairs gpairs)) fun _ _ _ => True :=
  fun _ t hd => (observePairs_ensures hp hd hv).post fun _ _ q => ⟨(), _, Bridge.runs_observeExts _ t, q, trivial⟩

/-- `observe_early` (`verifier.rs:318-324`): the replay's `observeExts` of the
run values, the flattened pairs and the block. -/
private theorem observeEarly_tracks {values : EarlyValuesView} {gv : Replay.EarlyValues} (hp : Permutes c.perm)
    (hruns : IsExts c n values.runs #[gv.eRe, gv.eIm, gv.row])
    (hpairs : List.Forall₂ (IsPair c n) values.pairs.toList gv.pairs.toList)
    (hblock : IsExt c n values.block gv.block) :
    Tracks c n (observeEarly values)
      (Replay.observeExts (#[gv.eRe, gv.eIm, gv.row] ++ Replay.flatPairs gv.pairs ++ #[gv.block]))
      fun _ _ _ => True := by
  intro d t hd
  unfold observeEarly
  refine Ensures.tbind (Q := fun res m => IsDuplex c m res.2
    (([gv.eRe, gv.eIm, gv.row].flatMap fun g => [g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe t.duplex)) ?_
    fun _ e₁ _ _ hd₁ => ?_
  · unfold Array.forM
    rw [← Array.foldlM_toList]
    exact forMList (R := fun m => IsExt c m) (fun g => [g.c0, g.c1, g.c2]) (fun _ _ _ _ h le => h.mono le)
      (fun x g hd hx => by simp only [List.foldl_cons, List.foldl_nil]; exact Duplex.observeExt_ensures hp hd hx)
      (fun _ _ => inferInstance) values.runs.toList [gv.eRe, gv.eIm, gv.row] hd (isExts_forall₂ hruns)
  refine Ensures.tbind (observePairs_ensures hp hd₁ (hpairs.imp fun _ _ q => q.mono)) fun _ e₂ _ _ hd₂ => ?_
  refine (Duplex.observeExt_ensures hp hd₂ hblock.mono).post fun _ _ q =>
    ⟨(), _, Bridge.runs_observeExts _ t, ?_, trivial⟩
  have e : ((#[gv.eRe, gv.eIm, gv.row] ++ Replay.flatPairs gv.pairs ++ #[gv.block]).flatMap
      fun g => #[g.c0, g.c1, g.c2]).toList = ([gv.eRe, gv.eIm, gv.row].flatMap fun g => [g.c0, g.c1, g.c2]) ++
        ((Replay.flatPairs gv.pairs).flatMap fun g => #[g.c0, g.c1, g.c2]).toList ++
          [gv.block.c0, gv.block.c1, gv.block.c2] := by
    simp [Array.toList_flatMap]
  dsimp only
  rw [← Array.foldl_toList, e, List.foldl_append, List.foldl_append]
  rw [← Array.foldl_toList] at q
  simp only [List.foldl_cons, List.foldl_nil]
  exact q

/-- `tuple`: `i + γ·(re + γ·im)` (`matrix.rs:394-400`). -/
private theorem tuple_ensures {index gamma re im : ExtVar} {gi gg gr gm : Ext} (hi : IsExt c n index gi)
    (hg : IsExt c n gamma gg) (hr : IsExt c n re gr) (hm : IsExt c n im gm) :
    Ensures c n (tuple index gamma re im) fun r m => IsExt c m r (Replay.tuple gg gi gr gm) :=
  (extMul_ensures hg hm).bind fun _ _ _ h₁ => (extAdd_ensures hr.mono h₁).bind fun _ _ _ h₂ =>
    (extMul_ensures hg.mono h₂).bind fun _ _ _ h₃ => extAdd_ensures hi.mono h₃

/-- `matrix_step` (`matrix.rs:368-380`): the replay's `matrixStep`. -/
private theorem matrixStep_ensures {layout : Replay.Layout} {point : Array ExtVar} {gpoint : Array Ext}
    (hp : IsExts c n point gpoint) :
    Ensures c n (matrixStep layout point) fun r m => IsExt c m r (Replay.matrixStep layout gpoint) := by
  unfold matrixStep
  refine (Ensures.rangeMapM layout.matrices (fun j a m => IsExt c m a (Ext.ofGl (Word.ofNat j)))
    (fun _ _ _ _ h le => h.mono le) fun j _ m _ => extConstant_ensures _).bind fun values _ _ hvalues => ?_
  obtain ⟨vsize, hv⟩ := hvalues
  refine (Mle.step_ensures (gvalues := (Array.range layout.matrices).map fun j => Ext.ofGl (Word.ofNat j)) _
    hp.mono ⟨by simp [vsize], fun i hi => ?_⟩ ?_).post fun _ _ e => ?_
  · have hi' : i < layout.matrices := vsize ▸ hi
    rw [getElem!_pos values i hi, getElem!_pos _ i (by simpa using hi'), Array.getElem_map, Array.getElem_range]
    exact hv i hi
  · rw [vsize, Array.size_extract]
    omega
  · rw [Replay.matrixStep]
    simpa [Array.toList_map, Array.toList_range] using e

/-- `Structure::check_early` (`matrix.rs:261-317`): once its checks hold, the
replay's `checkEarly` accepts, and the openings are the replay's. -/
private theorem checkEarly_ensures {layout : Replay.Layout} {r : Array KVar} {gr : Array (Word × Word)}
    {lb gamma sb mu : ExtVar} {ch : Replay.EarlyChallenges} {claims : Array TreeClaimView}
    {g : Replay.GkrClaims} {values : EarlyValuesView} {gv : Replay.EarlyValues}
    (hrs : r.size = gr.size) (hr : ∀ i < r.size, IsWord c n r[i]!.1 gr[i]!.1 ∧ IsWord c n r[i]!.2 gr[i]!.2)
    (hlb : IsExt c n lb ch.lookup.1) (hgamma : IsExt c n gamma ch.lookup.2) (hsb : IsExt c n sb ch.scatter.1)
    (hmu : IsExt c n mu ch.scatter.2) (htrees : ∀ t < 4, IsTree c n claims[t]! g (t + 1))
    (s₁ : 1 ≤ (g.leafValues[1]!).size) (s₂ : 2 ≤ (g.leafValues[2]!).size) (s₃ : 3 ≤ (g.leafValues[3]!).size)
    (s₄ : 2 ≤ (g.leafValues[4]!).size) (hruns : IsExts c n values.runs #[gv.eRe, gv.eIm, gv.row])
    (hps : values.pairs.size = gv.pairs.size)
    (hpairs : ∀ j < values.pairs.size, IsPair c n values.pairs[j]! gv.pairs[j]!) :
    Ensures c n (checkEarly layout r { lookup := (lb, gamma), scatter := (sb, mu) } claims values)
      fun res m => Replay.checkEarly layout gr ch g gv = .ok (Bridge.earlyOpeningsOf layout ch g gv) ∧
        IsOpenings c m res (Bridge.earlyOpeningsOf layout ch g gv) := by
  have tree₁ : IsTree c n claims[0]! g 1 := htrees 0 (by omega)
  have tree₂ : IsTree c n claims[1]! g 2 := htrees 1 (by omega)
  have tree₃ : IsTree c n claims[2]! g 3 := htrees 2 (by omega)
  have tree₄ : IsTree c n claims[3]! g 4 := htrees 3 (by omega)
  have runRe : IsExt c n values.runs[0]! gv.eRe := hruns.2 0 (by rw [hruns.1]; simp)
  have runIm : IsExt c n values.runs[1]! gv.eIm := hruns.2 1 (by rw [hruns.1]; simp)
  have runRow : IsExt c n values.runs[2]! gv.row := hruns.2 2 (by rw [hruns.1]; simp)
  unfold checkEarly
  dsimp only
  refine (checkPair_ensures (gx := g.roots[1]!) (gy := g.roots[2]!) "row lookup" ⟨tree₁.1, tree₁.2.1⟩
    ⟨tree₂.1, tree₂.2.1⟩).bind fun _ _ _ rows => ?_
  refine (checkPair_ensures (gx := g.roots[3]!) (gy := g.roots[4]!) "slot scatter" ⟨tree₃.1.mono, tree₃.2.1.mono⟩
    ⟨tree₄.1.mono, tree₄.2.1.mono⟩).bind fun _ _ _ slots => ?_
  refine (tuple_ensures runRow.mono hgamma.mono runRe.mono runIm.mono).bind fun combined _ _ hcombined => ?_
  refine (extSub_ensures hlb.mono hcombined).bind fun expected _ _ hexpected => ?_
  refine (assertExtEqual_ensures (tree₁.mono.value (by omega)) hexpected).bind fun _ _ _ runLeaf => ?_
  refine (Mle.chi_ensures hrs (fun i hi => ⟨(hr i hi).1.mono, (hr i hi).2.mono⟩) tree₂.2.2.1.mono).bind
    fun chi _ _ hchi => ?_
  refine (Mle.identity_ensures tree₂.2.2.1.mono).bind fun index _ _ hindex => ?_
  refine (tuple_ensures hindex hgamma.mono hchi.1.mono hchi.2.mono).bind fun combined' _ _ hcombined' => ?_
  refine (extSub_ensures hlb.mono hcombined').bind fun expected' _ _ hexpected' => ?_
  refine (assertExtEqual_ensures (tree₂.mono.value (by omega)) hexpected').bind fun _ _ _ rowLeaf => ?_
  refine (extMul_ensures hmu.mono runIm.mono).bind fun mixed _ _ hmixed => ?_
  refine (extAdd_ensures runRe.mono hmixed).bind fun e _ _ he => ?_
  refine (assertExtEqual_ensures (tree₃.mono.value (by omega)) he).bind fun _ _ _ scatterLeaf => ?_
  refine (matrixStep_ensures (layout := layout) tree₃.2.2.1.mono).bind fun matrix _ _ hmatrix => ?_
  refine (extScaleConstant_ensures (Word.ofNat (2 ^ layout.slotVariables)) hmatrix).bind fun tag _ _ htag => ?_
  refine (extSub_ensures hsb.mono (tree₃.mono.value (by omega))).bind fun rest _ _ hrest => ?_
  refine (extSub_ensures hrest htag.mono).bind fun slot _ _ hslot => ?_
  set geq := Replay.eqTable ((g.leafPoint[4]!).extract layout.slotVariables (g.leafPoint[4]!).size)
  refine (eqTable_ensures (tree₄.2.2.1.extract layout.slotVariables).mono).bind fun eq _ _ heq => ?_
  refine extZero_ensures.bind fun combined₀ _ _ hzero => ?_
  refine (Ensures.loopArray (fun k acc m => IsExt c m acc (Replay.sumRange k fun j => geq[j]! *
      ((gv.pairs[j]!).1 + ch.scatter.2 * (gv.pairs[j]!).2))) hzero fun k hk acc m _ hacc => ?_).bind
    fun combined _ _ hcombined'' => ?_
  · have hk' : k < values.pairs.size ∧ k < eq.size := by simpa using hk
    rw [show (values.pairs.zip eq)[k] = (values.pairs[k]!, eq[k]!) by simp [getElem!_pos, hk'.1, hk'.2]]
    have hpair : IsPair c m values.pairs[k]! gv.pairs[k]! := (hpairs k hk'.1).mono
    refine (extMul_ensures hmu.mono hpair.2).bind fun mixed' _ _ hmixed' => ?_
    refine (extAdd_ensures hpair.1.mono hmixed').bind fun value _ _ hvalue => ?_
    refine (extMul_ensures (heq.2 k hk'.2).mono hvalue).bind fun term _ _ hterm => ?_
    exact (extAdd_ensures hacc.mono hterm).bind fun total _ _ htotal =>
      Ensures.pure ⟨rfl, by rw [sumRange_succ]; exact htotal⟩
  have size : (values.pairs.zip eq).size = min gv.pairs.size geq.size := by
    rw [Array.size_zip, hps, heq.1]
  rw [size] at hcombined''
  refine (assertExtEqual_ensures (tree₄.mono.value (by omega)) hcombined'').bind fun _ _ _ pairLeaf => ?_
  refine (Mle.identity_ensures tree₄.2.2.1.mono).bind fun index' _ _ hindex' => ?_
  refine (extSub_ensures hsb.mono hindex').bind fun expected'' _ _ hexpected'' => ?_
  refine (assertExtEqual_ensures (tree₄.mono.value (by omega)) hexpected'').bind fun _ _ _ indexLeaf => ?_
  exact Ensures.pure ⟨checkEarly_ok rows slots runLeaf rowLeaf scatterLeaf pairLeaf indexLeaf, tree₃.2.2.1.mono,
    (tree₄.2.2.1.slice 0 layout.slotVariables).mono, tree₂.2.2.1.mono, tree₂.mono.value (by omega), runRow.mono,
    hslot.mono, tree₃.mono.value (by omega)⟩

/-- Two `sample_ext` calls draw the replay's pair (`samplePair`), then the
rest. -/
private theorem samplePair_tracks {γ δ : Type} (hp : Permutes c.perm) {g : ExtVar → ExtVar → Transcript γ}
    {y : Ext × Ext → Replay.ReplayM δ} {R : γ → δ → ℕ → Prop}
    (hg : ∀ a b ga gb m, n ≤ m → IsExt c m a ga → IsExt c m b gb → Tracks c m (g a b) (y (ga, gb)) R)
    (hg' : ∀ a b d, Appends ((g a b).run d) := by appends) :
    Tracks c n (Duplex.sampleExt >>= fun a => Duplex.sampleExt >>= fun b => g a b) (Replay.samplePair >>= y) R := by
  intro d t hd
  refine Ensures.tbind (Duplex.sampleExt_ensures hp hd) fun a e₁ _ _ h₁ => ?_
  refine Ensures.tbind (Duplex.sampleExt_ensures hp h₁.2) fun b e₂ _ _ h₂ => ?_
  refine (hg a b _ _ _ (by omega) h₁.1.mono h₂.1 e₂ _ ?_).post fun _ _ ⟨v, t', run, hdx, hr⟩ =>
    ⟨v, t', Bridge.runs_bind (Bridge.runs_samplePair t) run, hdx, hr⟩
  exact h₂.2

/-- **The early stage** (`verifier.rs:153-168`). Once the circuit's checks
hold, the circuit's early stage tracks the replay's, and the circuit's
outputs are the replay's. -/
theorem earlyStage_tracks {rel : Relation} {statement : ClaimView} {st : Replay.Statement} {proof : ProofView}
    {w : Var → Word} {g : Replay.Proof} (hp : Permutes c.perm) (hps : statement.point.size = st.point.size)
    (hpoint : ∀ i < statement.point.size,
      IsWord c n statement.point[i]!.1 st.point[i]!.1 ∧ IsWord c n statement.point[i]!.2 st.point[i]!.2)
    (hproof : IsProof c w n proof g) (read : Replay.readProof rel g = .ok ()) :
    Tracks c n (Circuit.earlyStage rel statement proof) (Replay.earlyStage rel st g)
      fun v o m => IsEarlyOut c m v o := by
  have shape := Bridge.readProof_ok read
  obtain ⟨rootsSize, -, -⟩ := Bridge.readGkr_ok shape.early
  obtain ⟨hp0, hhist, hgkr, hruns, hpairs, hblock, -⟩ := hproof
  unfold Circuit.earlyStage Replay.earlyStage
  rw [← bind_assoc (Replay.observeAll rel.commitmentSeed)]
  refine Tracks.bind (observeRoot_tracks hp hp0) fun _ _ _ _ _ => ?_
  refine Tracks.bind (Duplex.observeSlice_tracks hp hhist.mono) fun _ _ _ _ _ => ?_
  refine Tracks.bind (Duplex.sampleExt_tracks hp) fun beta gbeta _ _ hbeta => ?_
  refine samplePair_tracks hp fun lookupBeta gamma glb gg _ _ hlb hg => ?_
  refine samplePair_tracks hp fun scatterBeta mu gsb gmu _ _ hsb hmu => ?_
  refine Tracks.lift (tableSum_ensures rel.normBound hhist.mono hbeta.mono) fun total m₈ _ htotal => ?_
  refine Tracks.check htotal.1 ?_
  obtain ⟨⟨hrs, hroots⟩, hls, hlayers⟩ := (hgkr.mono (m := m₈)).entries
  refine Tracks.bind (gkrVerify_tracks hp hrs hroots hls hlayers shape.early).runs fun claims gc _ _ hc => ?_
  obtain ⟨hc, _, _, run⟩ := hc
  have five : claims.size = 5 := by
    rw [hc.1, (gkrVerify_sizes shape.early run).1, rootsSize, Bridge.earlyShapes_size]
  have leaves : ∀ u < 5, (gc.leafValues[u]!).size = (rel.earlyShapes[u]!).factors + 1 :=
    fun u hu => ((gkrVerify_sizes shape.early run).2 u (by rw [Bridge.earlyShapes_size]; exact hu)).1
  have tree : ∀ u < 5, IsTree c _ claims[u]! gc u := fun u hu =>
    ⟨(hc.2 u (by omega)).1, (hc.2 u (by omega)).2.1, (hc.2 u (by omega)).2.2.1, (hc.2 u (by omega)).2.2.2⟩
  have norm := tree 0 (by omega)
  refine Tracks.lift (extInv_ensures norm.2.1) fun _ _ _ hinv => ?_
  refine Tracks.lift (extMul_ensures norm.2.1.mono htotal.2.mono) fun expected _ _ hexpected => ?_
  refine Tracks.lift (assertExtEqual_ensures norm.1.mono hexpected) fun _ _ _ rootEq => ?_
  refine Tracks.lift (extSub_ensures hbeta.mono (norm.mono.value (j := 0) (by rw [leaves 0 (by omega)]; omega)))
    fun zLeaf _ _ hz => ?_
  refine Tracks.ensure (ext_not_isZero hinv.1) (Tracks.ensure (ext_beq_of_eq rootEq) ?_)
  have factors : (rel.earlyShapes[1]!).factors = 0 ∧ (rel.earlyShapes[2]!).factors = 1 ∧
      (rel.earlyShapes[3]!).factors = 2 ∧ (rel.earlyShapes[4]!).factors = 1 := ⟨rfl, rfl, rfl, rfl⟩
  obtain ⟨pairsSize, pairsOk⟩ := forall₂_entries hpairs
  refine Tracks.lift (checkEarly_ensures (gr := st.point) (g := gc) (gv := g.earlyValues)
    (ch := { beta := gbeta, lookup := (glb, gg), scatter := (gsb, gmu) }) hps
    (fun i hi => ⟨(hpoint i hi).1.mono, (hpoint i hi).2.mono⟩) hlb.mono hg.mono hsb.mono hmu.mono
    (fun t ht => by rw [extract_one]; exact (tree (t + 1) (by omega)).mono)
    (by rw [leaves 1 (by omega), factors.1]) (by rw [leaves 2 (by omega), factors.2.1])
    (by rw [leaves 3 (by omega), factors.2.2.1]) (by rw [leaves 4 (by omega), factors.2.2.2]) hruns.mono pairsSize
    (fun j hj => by rw [getElem!_pos _ j hj]; exact (pairsOk j hj).mono)) fun openings _ _ hopen => ?_
  obtain ⟨checkOk, hrun, hpair, hrow, hmult, hrowClaim, hslotClaim, hcoef⟩ := hopen
  refine Tracks.check checkOk ?_
  refine Tracks.bind (observeEarly_tracks hp hruns.mono (hpairs.imp fun _ _ q => q.mono) hblock.mono)
    fun _ _ _ _ _ => ?_
  exact Tracks.pure ⟨norm.2.2.1.mono, hz.mono, hrun.mono, hpair.mono, hrow.mono, hmult.mono, hrowClaim.mono,
    hslotClaim.mono, hcoef.mono⟩

end Stage

end CompressionLean.Sim
