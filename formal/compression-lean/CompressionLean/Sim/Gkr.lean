import CompressionLean.Sim.Sumcheck
import CompressionLean.Ports.Circuit.Gkr
import CompressionLean.Bridge.Gkr

namespace CompressionLean.Sim

open CompressionLean.Spec
open CompressionLean.Circuit
open CompressionLean.Lib
open CompressionLean.Replay (Ext TreeShape)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

def IsGkrClaims (c : Ctx F) (m : ℕ) (claims : Array TreeClaimView) (g : Replay.GkrClaims) : Prop :=
  claims.size = g.roots.size ∧ ∀ t < claims.size,
    IsExt c m claims[t]!.root.1 (g.roots[t]!).1 ∧ IsExt c m claims[t]!.root.2 (g.roots[t]!).2 ∧
      IsExts c m claims[t]!.point g.leafPoint[t]! ∧ IsExts c m claims[t]!.values g.leafValues[t]!

theorem IsGkrClaims.mono {c : Ctx F} {m m' : ℕ} {claims : Array TreeClaimView} {g : Replay.GkrClaims}
    (h : IsGkrClaims c m claims g) (le : m ≤ m' := by omega) : IsGkrClaims c m' claims g :=
  ⟨h.1, fun t ht => let ⟨p, q, point, values⟩ := h.2 t ht; ⟨p.mono le, q.mono le, point.mono le, values.mono le⟩⟩

private theorem horner_step (A W a nu R : Ext) : A + W * (a + nu * R) = A + W * a + W * nu * R :=
  coords_injective (F := Gl) (by
    simp only [coords_add, coords_mul]
    funext k
    fin_cases k <;> simp [extProduct, Matrix.vecHead, Matrix.vecTail] <;> ring)

private theorem add_mul_zero (A W : Ext) : A + W * 0 = A :=
  coords_injective (F := Gl) (by
    simp only [coords_add, coords_mul, coords_zero]
    funext k
    fin_cases k <;> simp [extProduct])

private theorem zero_add_one_mul (R : Ext) : R = 0 + 1 * R :=
  coords_injective (F := Gl) (by
    simp only [coords_add, coords_mul, coords_zero, coords_one]
    funext k
    fin_cases k <;> simp [extProduct])

private theorem flatMap_flatten {α β : Type} (f : α → List β) :
    ∀ L : List (List α), L.flatten.flatMap f = L.flatMap fun l => l.flatMap f
  | [] => rfl
  | l :: L => by rw [List.flatten_cons, List.flatMap_append, flatMap_flatten f L, List.flatMap_cons]

private theorem isExts_of_forall₂ {c : Ctx F} {n N : ℕ} {g : ℕ → Ext} {as : List ExtVar}
    (h : List.Forall₂ (fun j a => IsExt c n a (g j)) (List.range N) as) :
    IsExts c n as.toArray ((Array.range N).map g) := by
  obtain ⟨len, get⟩ := List.forall₂_iff_get.mp h
  simp only [List.length_range] at len
  refine ⟨by simp [← len], fun i hi => ?_⟩
  have hi' : i < N := by simpa [← len] using hi
  have := get i (by simpa using hi') (by simpa [← len] using hi')
  rw [getElem!_pos _ i (by simpa using hi), getElem!_pos _ i (by simpa using hi')]
  simpa using this

private def IsTreeClaim (c : Ctx F) (m : ℕ) (v : TreeClaimView) (g : Replay.GkrClaims) (t : ℕ) : Prop :=
  IsPair c m v.root g.roots[t]! ∧ IsExts c m v.point g.leafPoint[t]! ∧ IsExts c m v.values g.leafValues[t]!

private theorem IsTreeClaim.mono {c : Ctx F} {m m' : ℕ} {v : TreeClaimView} {g : Replay.GkrClaims} {t : ℕ}
    (h : IsTreeClaim c m v g t) (le : m ≤ m' := by omega) : IsTreeClaim c m' v g t :=
  let ⟨root, point, values⟩ := h
  ⟨root.mono le, point.mono le, values.mono le⟩

private def IsLayerState (c : Ctx F) (m : ℕ) (shapes : Array TreeShape) (j : ℕ) (cur : Array (ExtVar × ExtVar))
    (cl : Array TreeClaimView) (g : Replay.GkrClaims) : Prop :=
  cur.size = shapes.size ∧ cl.size = shapes.size ∧ g.current.size = shapes.size ∧
    g.leafPoint.size = shapes.size ∧ g.leafValues.size = shapes.size ∧
    ∀ t < shapes.size, IsPair c m cur[t]! g.current[t]! ∧ (shapes[t]!.depth ≤ j → IsTreeClaim c m cl[t]! g t)

private theorem IsLayerState.mono {c : Ctx F} {m m' : ℕ} {shapes : Array TreeShape} {j : ℕ}
    {cur : Array (ExtVar × ExtVar)} {cl : Array TreeClaimView} {g : Replay.GkrClaims}
    (h : IsLayerState c m shapes j cur cl g) (le : m ≤ m' := by omega) : IsLayerState c m' shapes j cur cl g :=
  let ⟨c1, c2, c3, c4, c5, trees⟩ := h
  ⟨c1, c2, c3, c4, c5, fun t ht => ⟨(trees t ht).1.mono le, fun hd => ((trees t ht).2 hd).mono le⟩⟩

section Advance

variable {shapes : Array TreeShape} {k t : ℕ} {g : Replay.GkrClaims} {pairs : List (ℕ × Array Ext)}
  {next : Array Ext} {tau : Ext}

private theorem advance_current (inside : t < g.current.size) :
    (Replay.advance shapes k g pairs next tau).current[t]! =
      if k < shapes[t]!.depth then ((Replay.reducedOf pairs tau t)[0]!, (Replay.reducedOf pairs tau t)[1]!)
      else g.current[t]! := by
  simp only [Replay.advance]
  exact getElem!_map_range _ _ inside

private theorem advance_leafPoint (inside : t < g.leafPoint.size) :
    (Replay.advance shapes k g pairs next tau).leafPoint[t]! =
      if shapes[t]!.depth = k then next.push tau else g.leafPoint[t]! := by
  simp only [Replay.advance]
  exact getElem!_map_range _ _ inside

private theorem advance_leafValues (inside : t < g.leafValues.size) :
    (Replay.advance shapes k g pairs next tau).leafValues[t]! =
      if shapes[t]!.depth = k then Replay.reducedOf pairs tau t else g.leafValues[t]! := by
  simp only [Replay.advance]
  exact getElem!_map_range _ _ inside

private theorem IsTreeClaim.advance {c : Ctx F} {m : ℕ} {v : TreeClaimView} (h : IsTreeClaim c m v g t)
    (ne : shapes[t]!.depth ≠ k) (hp : t < g.leafPoint.size) (hv : t < g.leafValues.size) :
    IsTreeClaim c m v (Replay.advance shapes k g pairs next tau) t := by
  refine ⟨h.1, ?_, ?_⟩
  · rw [advance_leafPoint hp, ite_eq_right ne]
    exact h.2.1
  · rw [advance_leafValues hv, ite_eq_right ne]
    exact h.2.2

end Advance

private theorem mem_activeTrees {shapes : Array TreeShape} {k t : ℕ} :
    t ∈ (Replay.activeTrees shapes k).toList ↔ t < shapes.size ∧ k ≤ shapes[t]!.depth := by
  unfold Replay.activeTrees
  simp

private theorem activeTrees_nodup (shapes : Array TreeShape) (k : ℕ) :
    (Replay.activeTrees shapes k).toList.Nodup := by
  unfold Replay.activeTrees
  rw [Array.toList_filter, Array.toList_range]
  exact List.nodup_range.filter _

private theorem mem_take_succ {l : List ℕ} (nodup : l.Nodup) {s : ℕ} (hs : s < l.length) :
    (∀ t, t ∈ l.take (s + 1) ↔ t ∈ l.take s ∨ t = l[s]) ∧ l[s] ∉ l.take s := by
  have e : l.take (s + 1) = l.take s ++ [l[s]] := by
    rw [List.take_add_one, List.getElem?_eq_getElem hs, Option.toList_some]
  refine ⟨fun t => by rw [e, List.mem_append, List.mem_singleton], fun member => ?_⟩
  have := (nodup.sublist (List.take_sublist (s + 1) l))
  rw [e, List.nodup_append] at this
  exact this.2.2 _ member _ (List.mem_singleton_self _) rfl

private theorem reducedOf_active {shapes : Array TreeShape} {k s : ℕ} {gchildren : Array (Array Ext)} {tau : Ext}
    (size : gchildren.size = (Replay.activeTrees shapes k).size) (hs : s < (Replay.activeTrees shapes k).size) :
    Replay.reducedOf ((Replay.activeTrees shapes k).zip gchildren).toList tau (Replay.activeTrees shapes k)[s] =
      (Array.range (gchildren[s]!.size / 2)).map fun j =>
        Replay.reduce ((gchildren[s]!)[2 * j]!, (gchildren[s]!)[2 * j + 1]!) tau := by
  have nodup : ((((Replay.activeTrees shapes k).zip gchildren).toList).map Prod.fst).Nodup := by
    rw [Array.toList_zip, List.map_fst_zip (by simp [size])]
    exact activeTrees_nodup shapes k
  have member : ((Replay.activeTrees shapes k)[s], gchildren[s]!) ∈
      ((Replay.activeTrees shapes k).zip gchildren).toList :=
    List.mem_iff_getElem.mpr ⟨s, by simp; omega, by simp [getElem!_pos gchildren s (by omega)]⟩
  unfold Replay.reducedOf
  rw [Bridge.find_of_mem nodup member]

private def drawnExt (t : Replay.Transcript) : Ext :=
  ⟨t.duplex.sample.1, t.duplex.sample.2.sample.1, t.duplex.sample.2.sample.2.sample.1⟩

private def afterExt (t : Replay.Transcript) : Replay.Transcript :=
  { duplex := t.duplex.sample.2.sample.2.sample.2, draws := t.draws.push (.ext (drawnExt t)) }

instance gkrCombine_appends (children : Array ExtVar) (mu : ExtVar) : Appends (Circuit.gkrCombine children mu) := by
  unfold Circuit.gkrCombine; appends
instance gkrVerify_appends (view : GkrView) (shapes : Array TreeShape) (d : Duplex) :
    Appends ((Circuit.gkrVerify view shapes).run d) := by
  unfold Circuit.gkrVerify; appends

private theorem listMapMT {c : Ctx F} {δ α : Type} {f : δ → Transcript α} (R : δ → α → ℕ → Prop)
    (mono : ∀ x a m m', R x a m → m ≤ m' → R x a m') (hf : ∀ x d, Appends ((f x).run d)) :
    ∀ (xs : List δ) {n : ℕ} {d : Duplex},
      (∀ x ∈ xs, ∀ m e, n ≤ m → Ensures c m ((f x).run e) fun r m => R x r.1 m ∧ r.2 = e) →
      Ensures c n ((xs.mapM f).run d) fun r m => List.Forall₂ (fun x a => R x a m) xs r.1 ∧ r.2 = d
  | [], n, d, _ => by
      rw [List.mapM_nil, StateT.run_pure]
      exact Ensures.pure ⟨List.Forall₂.nil, rfl⟩
  | x :: xs, n, d, h => by
      rw [List.mapM_cons]
      refine (h x List.mem_cons_self n d le_rfl).tbind (fun a e m le hx => ?_) (hf x d)
        fun _ _ => Appends.tbind (fun d => Appends.tmapM_list hf xs d) fun _ _ => Appends.tpure _ _
      obtain ⟨ha, he⟩ := hx
      refine (listMapMT R mono hf xs (d := e) fun y hy m' e' le' =>
        h y (List.mem_cons_of_mem _ hy) m' e' (le.trans le')).tbind (fun as e' m' le' hxs => ?_)
        (Appends.tmapM_list hf xs e) fun _ _ => Appends.tpure _ _
      obtain ⟨has, he'⟩ := hxs
      rw [StateT.run_pure]
      exact Ensures.pure ⟨List.Forall₂.cons (mono _ _ _ _ ha le') has, he'.trans he⟩

section Verify

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

private theorem gkrCombine_ensures {children : Array ExtVar} {gchildren : Array Ext} {mu : ExtVar} {gmu : Ext}
    (hc : IsExts c n children gchildren) (two : 2 ≤ gchildren.size) (hmu : IsExt c n mu gmu) :
    Ensures c n (Circuit.gkrCombine children mu) fun r m => IsExt c m r (Replay.combineVals gchildren gmu) := by
  unfold Circuit.gkrCombine
  dsimp only
  have numerator : ∀ side ≤ 1, ∀ {m : ℕ}, n ≤ m →
      Ensures c m (do
          let total ← Algebra.extOne
          let total ← forIn [:children.size / 2 - 1] total fun j total => do
            let total ← Build.extMul total children[2 * j + side]!
            Pure.pure (ForInStep.yield total)
          Pure.pure total)
        fun r m' => IsExt c m' r (Replay.prodRange (children.size / 2 - 1) fun j => gchildren[2 * j + side]!) := by
    intro side hside m le
    refine extOne_ensures.bind fun total _ _ hone => ?_
    rw [Std.Legacy.Range.forIn_eq_forIn_range', ← List.forIn_toArray]
    refine (Ensures.loopArray (fun j acc m' => IsExt c m' acc (Replay.prodRange j fun j => gchildren[2 * j + side]!))
      hone fun j hj acc m' _ hacc => ?_).post fun _ _ hr => by simpa using hr
    have hj' : j < children.size / 2 - 1 := by simpa using hj
    rw [show (List.range' [:children.size / 2 - 1].start [:children.size / 2 - 1].size
      [:children.size / 2 - 1].step).toArray[j] = j by simp]
    refine (extMul_ensures hacc (hc.2 (2 * j + side) (by omega)).mono).bind fun total' _ _ htotal =>
      Ensures.pure ⟨rfl, ?_⟩
    rw [prodRange_succ]
    exact htotal
  have hq : ∀ i ≤ 1,
      IsExt c n children[2 * (children.size / 2 - 1) + i]! gchildren[2 * (children.size / 2 - 1) + i]! :=
    fun i hi => hc.2 _ (by have := hc.1; omega)
  refine (numerator 0 (by omega) le_rfl).bind fun p0 _ _ hp0 => ?_
  refine (numerator 1 le_rfl (by omega)).bind fun p1 _ _ hp1 => ?_
  refine (extMul_ensures hp0.mono (hq 1 le_rfl).mono).bind fun a _ _ ha => ?_
  refine (extMul_ensures hp1.mono (hq 0 (by omega)).mono).bind fun b _ _ hb => ?_
  refine (extMul_ensures (hq 0 (by omega)).mono (hq 1 le_rfl).mono).bind fun qq _ _ hqq => ?_
  refine (extMul_ensures hmu.mono hqq).bind fun d _ _ hd => ?_
  refine (extAdd_ensures ha.mono hb.mono).bind fun sum _ _ hsum => ?_
  refine (extAdd_ensures hsum hd.mono).post fun _ _ e => ?_
  rw [hc.1] at e
  simp only [Replay.combineVals, ext_mul_assoc]
  exact e

private theorem reduceValues_ensures {values : Array ExtVar} {gvalues : Array Ext} {tau : ExtVar} {gtau : Ext}
    {d : Duplex} (hv : IsExts c n values gvalues) (ht : IsExt c n tau gtau) :
    Ensures c n (((Array.range (values.size / 2)).mapM (m := Transcript) fun j => do
        let difference ← liftM (Build.extSub values[2 * j + 1]! values[2 * j]!)
        let step ← liftM (Build.extMul tau difference)
        liftM (Build.extAdd values[2 * j]! step)).run d)
      fun res m => IsExts c m res.1
        ((Array.range (gvalues.size / 2)).map fun j => Replay.reduce (gvalues[2 * j]!, gvalues[2 * j + 1]!) gtau) ∧
        res.2 = d := by
  rw [Array.mapM_eq_mapM_toList, StateT.run_map]
  refine ((listMapMT (fun j a m => IsExt c m a (Replay.reduce (gvalues[2 * j]!, gvalues[2 * j + 1]!) gtau))
    (fun _ _ _ _ h le => h.mono le) (fun _ _ => by appends) _ fun j member m e _ => ?_).map
      (Q' := fun r m => List.Forall₂
        (fun j a => IsExt c m a (Replay.reduce (gvalues[2 * j]!, gvalues[2 * j + 1]!) gtau))
        (List.range (values.size / 2)) r.1.toList ∧ r.2 = d) fun r m h => by simpa using h).post fun r m q => ?_
  · have hj : 2 * j + 1 < values.size := by
      simp only [Array.toList_range, List.mem_range] at member
      omega
    refine Ensures.liftBind (extSub_ensures (hv.2 _ hj).mono (hv.2 _ (by omega)).mono) fun diff _ _ hdiff => ?_
    refine Ensures.liftBind (extMul_ensures ht.mono hdiff) fun step _ _ hstep => ?_
    exact Ensures.lift (extAdd_ensures (hv.2 _ (by omega)).mono hstep)
  · rw [hv.1] at q
    exact ⟨isExts_of_forall₂ q.1, q.2⟩

private theorem observePairs_ensures {d : Duplex} {r : Replay.Duplex} {vs : Array (ExtVar × ExtVar)}
    {gs : Array (Ext × Ext)} (hp : Permutes c.perm) (hd : IsDuplex c n d r) (hsize : vs.size = gs.size)
    (hv : ∀ i < vs.size, IsExt c n (vs[i]!).1 (gs[i]!).1 ∧ IsExt c n (vs[i]!).2 (gs[i]!).2) :
    Ensures c n ((forIn vs PUnit.unit fun (x : ExtVar × ExtVar) _ => do
        Duplex.observeExt x.1
        Duplex.observeExt x.2
        Pure.pure (ForInStep.yield PUnit.unit)).run d)
      fun res m => IsDuplex c m res.2
        (((Replay.flatPairs gs).flatMap fun g => #[g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe r) := by
  rw [← Array.forIn_toList]
  refine (Ensures.loopT (fun k _ e m => IsDuplex c m e
      (((gs.toList.take k).flatMap fun g => [g.1.c0, g.1.c1, g.1.c2, g.2.c0, g.2.c1, g.2.c2]).foldl
        Replay.Duplex.observe r)) (by simpa using hd) fun k hk _ e m _ he => ?_).post fun _ _ q => ?_
  · have hk' : k < vs.size := by simpa using hk
    have hx : IsExt c n (vs.toList[k]).1 (gs[k]!).1 ∧ IsExt c n (vs.toList[k]).2 (gs[k]!).2 := by
      rw [Array.getElem_toList, ← getElem!_pos vs k hk']
      exact hv k hk'
    rw [StateT.run_bind]
    refine (Duplex.observeExt_ensures hp he hx.1.mono).bind fun res _ _ h₁ => ?_
    rw [StateT.run_bind]
    refine (Duplex.observeExt_ensures hp h₁ hx.2.mono).bind fun res₂ _ _ h₂ => ?_
    rw [StateT.run_pure]
    refine Ensures.pure ⟨rfl, ?_⟩
    have hg : k < gs.size := hsize ▸ hk'
    rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using hg), Option.toList_some, List.flatMap_append,
      List.foldl_append, Array.getElem_toList, ← getElem!_pos gs k hg]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.foldl_cons, List.foldl_nil]
    exact h₂
  · rw [show vs.toList.length = gs.toList.length by simp [hsize], List.take_length] at q
    rw [← Array.foldl_toList, Array.toList_flatMap, Replay.flatPairs, Array.toList_flatMap, List.flatMap_assoc]
    simpa using q

private theorem observeChildren_ensures {d : Duplex} {r : Replay.Duplex} {vs : Array (Array ExtVar)}
    {gs : Array (Array Ext)} (hp : Permutes c.perm) (hd : IsDuplex c n d r) (hsize : vs.size = gs.size)
    (hv : ∀ i < vs.size, IsExts c n vs[i]! gs[i]!) :
    Ensures c n ((forIn vs PUnit.unit fun values _ => do
        forIn values PUnit.unit fun value _ => do
          Duplex.observeExt value
          Pure.pure (ForInStep.yield PUnit.unit)
        Pure.pure (ForInStep.yield PUnit.unit)).run d)
      fun res m => IsDuplex c m res.2 ((gs.flatten.flatMap fun g => #[g.c0, g.c1, g.c2]).foldl
        Replay.Duplex.observe r) := by
  rw [← Array.forIn_toList]
  refine (Ensures.loopT (fun k _ e m => IsDuplex c m e
      (((gs.toList.take k).flatMap fun g => g.toList.flatMap fun v => [v.c0, v.c1, v.c2]).foldl
        Replay.Duplex.observe r)) (by simpa using hd) fun k hk _ e m _ he => ?_).post fun _ _ q => ?_
  · have hk' : k < vs.size := by simpa using hk
    rw [StateT.run_bind]
    refine (observeExts_ensures (gs := gs[k]!) hp he ?_).bind fun res _ _ hres => ?_
    · rw [Array.getElem_toList, ← getElem!_pos vs k hk']
      exact (hv k hk').mono
    rw [StateT.run_pure]
    refine Ensures.pure ⟨rfl, ?_⟩
    have hg : k < gs.size := hsize ▸ hk'
    rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using hg), Option.toList_some, List.flatMap_append,
      List.foldl_append, Array.getElem_toList, ← getElem!_pos gs k hg]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
    rw [← Array.foldl_toList, Array.toList_flatMap] at hres
    exact hres
  · rw [show vs.toList.length = gs.toList.length by simp [hsize], List.take_length] at q
    rw [← Array.foldl_toList, Array.toList_flatMap, Array.toList_flatten, flatMap_flatten, List.flatMap_map]
    exact q

private theorem batchLoop_ensures {active : Array ℕ} {children : Array (Array ExtVar)}
    {gchildren : Array (Array Ext)} {cur : Array (ExtVar × ExtVar)} {gcur : Array (Ext × Ext)} {mu nu : ExtVar}
    {gmu gnu : Ext} {init : ExtVar × ExtVar × ExtVar} {d : Duplex} (hmu : IsExt c n mu gmu) (hnu : IsExt c n nu gnu)
    (hcur : ∀ i ∈ active, IsPair c n cur[i]! gcur[i]!) (hsize : children.size = gchildren.size)
    (hch : ∀ s < children.size, IsExts c n children[s]! gchildren[s]! ∧ 2 ≤ gchildren[s]!.size)
    (hclaim : IsExt c n init.1 0) (hexpected : IsExt c n init.2.1 0) (hweight : IsExt c n init.2.2 1) :
    Ensures c n ((forIn (m := Transcript) (active.zip children) init
        fun (x : ℕ × Array ExtVar) (acc : ExtVar × ExtVar × ExtVar) => do
          let mq ← liftM (Build.extMul mu cur[x.1]!.2)
          let pq ← liftM (Build.extAdd cur[x.1]!.1 mq)
          let term ← liftM (Build.extMul acc.2.2 pq)
          let claim ← liftM (Build.extAdd acc.1 term)
          let combined ← liftM (Circuit.gkrCombine x.2 mu)
          let term ← liftM (Build.extMul acc.2.2 combined)
          let expected ← liftM (Build.extAdd acc.2.1 term)
          let weight ← liftM (Build.extMul acc.2.2 nu)
          Pure.pure (ForInStep.yield (claim, expected, weight))).run d)
      fun res m => IsExt c m res.1.1 (Replay.batchClaim gcur gmu gnu (active.zip gchildren).toList) ∧
        IsExt c m res.1.2.1 (Replay.batchExpected gmu gnu (active.zip gchildren).toList) ∧ res.2 = d := by
  rw [← Array.forIn_toList]
  set pairs := (active.zip gchildren).toList with pairs_def
  refine (Ensures.loopT (fun s acc e m => e = d ∧ ∃ A E W : Ext, IsExt c m acc.1 A ∧ IsExt c m acc.2.1 E ∧
      IsExt c m acc.2.2 W ∧
      Replay.batchClaim gcur gmu gnu pairs = A + W * Replay.batchClaim gcur gmu gnu (pairs.drop s) ∧
      Replay.batchExpected gmu gnu pairs = E + W * Replay.batchExpected gmu gnu (pairs.drop s))
    ⟨rfl, 0, 0, 1, hclaim, hexpected, hweight, zero_add_one_mul _, zero_add_one_mul _⟩
    fun s hs acc e m _ hI => ?_).post fun r m q => ?_
  · obtain ⟨he, A, E, W, hA, hE, hW, eA, eE⟩ := hI
    have hsa : s < active.size := by simp at hs; omega
    have hsc : s < children.size := by simp at hs; omega
    rw [show (active.zip children).toList[s] = (active[s], children[s]) by simp]
    dsimp only
    have hi : IsPair c n cur[active[s]]! gcur[active[s]]! := hcur _ (Array.getElem_mem _)
    have hv : IsExts c n children[s] gchildren[s]! ∧ 2 ≤ gchildren[s]!.size := by
      have := hch s hsc
      rwa [getElem!_pos children s hsc] at this
    refine Ensures.liftBind (extMul_ensures hmu.mono hi.2.mono) fun mq _ _ hmq => ?_
    refine Ensures.liftBind (extAdd_ensures hi.1.mono hmq) fun pq _ _ hpq => ?_
    refine Ensures.liftBind (extMul_ensures hW.mono hpq) fun t₁ _ _ ht₁ => ?_
    refine Ensures.liftBind (extAdd_ensures hA.mono ht₁) fun cl _ _ hcl => ?_
    refine Ensures.liftBind (gkrCombine_ensures hv.1.mono hv.2 hmu.mono) fun cb _ _ hcb => ?_
    refine Ensures.liftBind (extMul_ensures hW.mono hcb) fun t₂ _ _ ht₂ => ?_
    refine Ensures.liftBind (extAdd_ensures hE.mono ht₂) fun ex _ _ hex => ?_
    refine Ensures.liftBind (extMul_ensures hW.mono hnu.mono) fun w _ _ hw => ?_
    rw [StateT.run_pure]
    have hs' : s < pairs.length := by simp [pairs_def, ← hsize, hsa, hsc]
    refine Ensures.pure ⟨rfl, he, _, _, _, hcl.mono, hex.mono, hw, ?_, ?_⟩
    · rw [eA, List.drop_eq_getElem_cons hs', Replay.batchClaim, horner_step]
      simp [pairs_def]
    · rw [eE, List.drop_eq_getElem_cons hs', Replay.batchExpected, horner_step]
      simp [pairs_def, getElem!_pos gchildren s (by omega)]
  · obtain ⟨e, A, E, W, hA, hE, -, eA, eE⟩ := q
    have len : (active.zip children).toList.length = pairs.length := by simp [pairs_def, hsize]
    rw [len, List.drop_length, Replay.batchClaim, add_mul_zero] at eA
    rw [len, List.drop_length, Replay.batchExpected, add_mul_zero] at eE
    exact ⟨eA ▸ hA, eE ▸ hE, e⟩

private theorem reduceLoop_ensures {shapes : Array TreeShape} {j : ℕ} {roots : Array (ExtVar × ExtVar)}
    {next : Array ExtVar} {tau : ExtVar} {children : Array (Array ExtVar)} {gc : Replay.GkrClaims}
    {gchildren : Array (Array Ext)} {gnext : Array Ext} {gtau : Ext} {cur : Array (ExtVar × ExtVar)}
    {cl : Array TreeClaimView} {d : Duplex}
    (hs : IsLayerState c n shapes j cur cl gc) (hroots : ∀ t < shapes.size, IsPair c n roots[t]! gc.roots[t]!)
    (hcs : children.size = gchildren.size) (hga : gchildren.size = (Replay.activeTrees shapes (j + 1)).size)
    (hch : ∀ s < children.size, IsExts c n children[s]! gchildren[s]! ∧
      gchildren[s]!.size = 2 * (Replay.partFactors shapes (j + 1) (Replay.activeTrees shapes (j + 1))[s]! + 1))
    (hnext : IsExts c n next (gnext.push gtau)) (htau : IsExt c n tau gtau) :
    Ensures c n ((forIn (m := Transcript) ((Replay.activeTrees shapes (j + 1)).zip children) (cur, cl)
        fun (x : ℕ × Array ExtVar) (acc : Array (ExtVar × ExtVar) × Array TreeClaimView) => do
          let reduced ← Array.mapM (fun i => do
              let difference ← liftM (Build.extSub x.2[2 * i + 1]! x.2[2 * i]!)
              let step ← liftM (Build.extMul tau difference)
              liftM (Build.extAdd x.2[2 * i]! step)) (Array.range (x.2.size / 2))
          if shapes[x.1]!.depth = j + 1 then
            Pure.pure (ForInStep.yield
              (acc.1, acc.2.set! x.1 { root := roots[x.1]!, point := next, values := reduced }))
          else Pure.pure (ForInStep.yield (acc.1.set! x.1 (reduced[0]!, reduced[1]!), acc.2))).run d)
      fun res m => res.2 = d ∧ IsLayerState c m shapes (j + 1) res.1.1 res.1.2
        (Replay.advance shapes (j + 1) gc ((Replay.activeTrees shapes (j + 1)).zip gchildren).toList gnext gtau) := by
  rw [← Array.forIn_toList]
  have nodup := activeTrees_nodup shapes (j + 1)
  obtain ⟨c1, c2, c3, c4, c5, c6⟩ := hs
  set act := Replay.activeTrees shapes (j + 1) with act_def
  set g := Replay.advance shapes (j + 1) gc (act.zip gchildren).toList gnext gtau with g_def
  refine (Ensures.loopT (fun s acc e m => e = d ∧ acc.1.size = shapes.size ∧ acc.2.size = shapes.size ∧
      ∀ t < shapes.size, IsPair c m acc.1[t]! (if t ∈ act.toList.take s then g.current[t]! else gc.current[t]!) ∧
        (shapes[t]!.depth ≤ j ∨ (t ∈ act.toList.take s ∧ shapes[t]!.depth = j + 1) → IsTreeClaim c m acc.2[t]! g t))
    ⟨rfl, c1, c2, fun t ht => ⟨by simpa using (c6 t ht).1, fun hd => ?_⟩⟩ fun s hs' acc e m _ hI => ?_).post
      fun r m q => ?_
  · simp only [List.take_zero, List.not_mem_nil, false_and, or_false] at hd
    exact ((c6 t ht).2 hd).advance (by omega) (by omega) (by omega)
  · obtain ⟨he, sz1, sz2, inv⟩ := hI
    have hsa : s < act.size := by simp at hs'; omega
    have hsc : s < children.size := by simp at hs'; omega
    have hsl : s < act.toList.length := by simpa using hsa
    rw [show (act.zip children).toList[s] = (act[s], children[s]) by simp]
    dsimp only
    have hmem : act.toList[s] ∈ act.toList := List.getElem_mem hsl
    rw [Array.getElem_toList] at hmem
    obtain ⟨hin, hdeep⟩ := mem_activeTrees.mp hmem
    obtain ⟨split, fresh⟩ := mem_take_succ nodup hsl
    rw [Array.getElem_toList] at split fresh
    have hv : IsExts c n children[s] gchildren[s]! ∧
        gchildren[s]!.size = 2 * (Replay.partFactors shapes (j + 1) act[s] + 1) := by
      have := hch s hsc
      rw [getElem!_pos children s hsc, getElem!_pos act s hsa] at this
      exact this
    have hro : Replay.reducedOf (act.zip gchildren).toList gtau act[s] =
        (Array.range (gchildren[s]!.size / 2)).map fun i =>
          Replay.reduce ((gchildren[s]!)[2 * i]!, (gchildren[s]!)[2 * i + 1]!) gtau :=
      reducedOf_active hga hsa
    refine (reduceValues_ensures hv.1.mono htau.mono).tbind fun reduced e₁ m₁ _ hred => ?_
    obtain ⟨hr, he₁⟩ := hred
    by_cases hk : shapes[act[s]]!.depth = j + 1
    · rw [ite_eq_left hk, StateT.run_pure]
      refine Ensures.pure ⟨rfl, ?_⟩
      dsimp only [ForInStep.value]
      refine ⟨he₁.trans he, sz1, by rw [Array.size_set!]; exact sz2, fun t ht => ⟨?_, fun hd => ?_⟩⟩
      · by_cases ts : t = act[s]
        · subst ts
          rw [ite_eq_left ((split _).mpr (Or.inr rfl)), advance_current (by omega), ite_eq_right (by omega)]
          have := (inv _ ht).1
          rw [ite_eq_right fresh] at this
          exact this.mono
        · have := (inv t ht).1
          simp only [(split t).trans (or_iff_left ts)]
          exact this.mono
      · rw [getElem!_set! _ _ _ _ (by omega)]
        by_cases ts : act[s] = t
        · subst ts
          rw [ite_eq_left rfl]
          refine ⟨(hroots _ ht).mono, ?_, ?_⟩
          · rw [advance_leafPoint (by omega), ite_eq_left hk]
            exact hnext.mono
          · rw [advance_leafValues (by omega), ite_eq_left hk, hro]
            exact hr
        · rw [ite_eq_right ts]
          exact ((inv t ht).2 (hd.imp_right fun hd' =>
            ⟨((split t).mp hd'.1).resolve_right (Ne.symm ts), hd'.2⟩)).mono
    · rw [ite_eq_right hk, StateT.run_pure]
      refine Ensures.pure ⟨rfl, ?_⟩
      dsimp only [ForInStep.value]
      have inner : j + 1 < shapes[act[s]]!.depth := by omega
      refine ⟨he₁.trans he, by rw [Array.size_set!]; exact sz1, sz2, fun t ht => ⟨?_, fun hd => ?_⟩⟩
      · rw [getElem!_set! _ _ _ _ (by omega)]
        by_cases ts : act[s] = t
        · subst ts
          rw [ite_eq_left rfl, ite_eq_left ((split _).mpr (Or.inr rfl)), advance_current (by omega), ite_eq_left inner, hro]
          have size : gchildren[s]!.size / 2 = 2 := by
            rw [hv.2, Replay.partFactors, ite_eq_right hk]
          exact ⟨hr.2 0 (by rw [hr.1]; simp [size]), hr.2 1 (by rw [hr.1]; simp [size])⟩
        · rw [ite_eq_right ts]
          have := (inv t ht).1
          simp only [(split t).trans (or_iff_left (Ne.symm ts))]
          exact this.mono
      · by_cases ts : t = act[s]
        · subst ts
          exfalso
          rcases hd with hd | hd
          · omega
          · exact hk hd.2
        · exact ((inv t ht).2 (hd.imp_right fun hd' => ⟨((split t).mp hd'.1).resolve_right ts, hd'.2⟩)).mono
  · obtain ⟨he, sz1, sz2, inv⟩ := q
    have len : (act.zip children).toList.length = act.toList.length := by simp [hcs, hga]
    rw [len, List.take_length] at inv
    refine ⟨he, sz1, sz2, by simp [g_def, Replay.advance, c3], by simp [g_def, Replay.advance, c4],
      by simp [g_def, Replay.advance, c5], fun t ht => ⟨?_, fun hd => ?_⟩⟩
    · have := (inv t ht).1
      split_ifs at this with member
      · exact this
      · rw [advance_current (by omega), ite_eq_right ?_]
        · exact this
        · intro deep
          exact member (mem_activeTrees.mpr ⟨ht, by omega⟩)
    · refine (inv t ht).2 ?_
      by_cases deep : shapes[t]!.depth ≤ j
      · exact Or.inl deep
      · exact Or.inr ⟨mem_activeTrees.mpr ⟨ht, by omega⟩, by omega⟩

theorem gkrVerify_tracks {view : GkrView} {proof : Replay.GkrProof} {shapes : Array TreeShape}
    (hp : Permutes c.perm) (hrs : view.roots.size = proof.roots.size)
    (hroots : ∀ i < view.roots.size, IsPair c n view.roots[i]! proof.roots[i]!)
    (hls : view.layers.size = proof.layers.size)
    (hlayers : ∀ i (hi : i < view.layers.size),
      (view.layers[i].rounds.size = (proof.layers[i]!).rounds.size ∧
        ∀ r < view.layers[i].rounds.size, IsExts c n view.layers[i].rounds[r]! (proof.layers[i]!).rounds[r]!) ∧
      (view.layers[i].children.size = (proof.layers[i]!).children.size ∧
        ∀ s < view.layers[i].children.size,
          IsExts c n view.layers[i].children[s]! (proof.layers[i]!).children[s]!))
    (read : Replay.readGkr proof shapes = .ok ()) :
    Tracks c n (Circuit.gkrVerify view shapes) (Replay.gkrVerify proof shapes)
      fun res g m => IsGkrClaims c m res g := by
  intro d t hd
  obtain ⟨rootsSize, layersSize, layersOk⟩ := Bridge.readGkr_ok read
  unfold Circuit.gkrVerify
  dsimp only
  set init : Replay.GkrClaims :=
    { roots := proof.roots, point := #[], current := proof.roots,
      leafPoint := shapes.map fun _ => #[], leafValues := shapes.map fun _ => #[] } with init_def
  set t₁ : Replay.Transcript := { t with duplex :=
    ((Replay.flatPairs proof.roots).flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe t.duplex }
    with t₁_def
  refine Ensures.tbind (observePairs_ensures hp hd hrs hroots) fun _ e₀ m₀ _ hd₀ => ?_
  refine Ensures.tbind (Q := fun r m => ∃ g tL, Runs (Replay.gkrLayers shapes 1 proof.layers.toList init) t₁ g tL ∧
      IsDuplex c m r.2 tL.duplex ∧ IsLayerState c m shapes proof.layers.size r.1.1 r.1.2.1 g ∧ g.roots = proof.roots)
    ?_ fun res e m _ hres => ?_
  · rw [← Array.forIn_toList]
    refine (Ensures.loopT (fun j acc e m => ∃ g tj,
        Runs (Replay.gkrLayers shapes 1 (proof.layers.toList.take j) init) t₁ g tj ∧ IsDuplex c m e tj.duplex ∧
        IsLayerState c m shapes j acc.1 acc.2.1 g ∧ IsExts c m acc.2.2 g.point ∧ g.roots = proof.roots)
      ⟨init, t₁, Bridge.runs_pure _ _, hd₀, ?_, IsExts.empty, rfl⟩ fun j hj acc e m _ hI => ?_).post
      fun r m q => ?_
    · refine ⟨by rw [hrs, rootsSize], by simp, by simp [init_def, rootsSize], by simp [init_def],
        by simp [init_def], fun u hu => ⟨(hroots u (by omega)).mono,
          fun deep => absurd deep (by have := Bridge.readGkr_depth read hu; omega)⟩⟩
    · obtain ⟨gc, tj, run₀, hdj, state, hpt, roots⟩ := hI
      have hj' : j < view.layers.size := by simpa using hj
      have hjp : j < proof.layers.size := hls ▸ hj'
      rw [show view.layers.zipIdx.toList[j] = (view.layers[j], j) by simp]
      dsimp only
      obtain ⟨⟨hrsize, hrounds⟩, ⟨hcsize, hchildren⟩⟩ := hlayers j hj'
      have hS : (proof.layers[j]!).rounds.size = j ∧ (∀ r ∈ (proof.layers[j]!).rounds, 3 ≤ r.size) ∧
          (proof.layers[j]!).children.size = (Replay.activeTrees shapes (j + 1)).size ∧
          ∀ s < (proof.layers[j]!).children.size, (proof.layers[j]!).children[s]!.size =
            2 * (Replay.partFactors shapes (j + 1) (Replay.activeTrees shapes (j + 1))[s]! + 1) := by
        have := Bridge.layerShapeOk_facts (layersOk j hjp)
        rw [← getElem!_pos proof.layers j hjp, Nat.add_sub_cancel] at this
        exact this
      obtain ⟨-, wide, hga, hcount⟩ := hS
      have state₀ := state
      obtain ⟨c1, c2, c3, c4, c5, c6⟩ := state
      refine Ensures.tbind (Duplex.sampleExt_ensures hp hdj) fun mu _ _ _ h₁ => ?_
      refine Ensures.tbind (Duplex.sampleExt_ensures hp h₁.2) fun nu e₂ _ _ h₂ => ?_
      refine Ensures.liftBind extZero_ensures fun z₁ _ _ hz₁ => ?_
      refine Ensures.liftBind extZero_ensures fun z₂ _ _ hz₂ => ?_
      refine Ensures.liftBind extOne_ensures fun o _ _ ho => ?_
      refine Ensures.tbind (batchLoop_ensures (active := Replay.activeTrees shapes (j + 1))
        (gchildren := (proof.layers[j]!).children) (gcur := gc.current) h₁.1.mono h₂.1.mono
        (fun i hi => (c6 i (mem_activeTrees.mp (Array.mem_toList_iff.mpr hi)).1).1.mono) hcsize
        (fun s hs => ⟨(hchildren s hs).mono, by have := hcount s (by rw [← hcsize]; exact hs); omega⟩)
        hz₁.mono hz₂.mono ho) fun acc' e₆ _ _ h₆ => ?_
      obtain ⟨hclaim, hexpected, he₆⟩ := h₆
      have he₆' : e₆ = e₂ := he₆
      refine Ensures.tbind (sumcheckReplay_tracks hp hrsize (fun r hr => (hrounds r hr).mono)
        (fun r hr => by have := wide r hr; omega) hclaim _ (afterExt (afterExt tj))
        (by rw [he₆']; exact h₂.2.mono)) fun sc e₇ _ _ h₇ => ?_
      obtain ⟨⟨gpoint, gfinal⟩, ts, sum, hd₇, hnext, hlast⟩ := h₇
      refine Ensures.liftBind (eqEval_ensures hpt.mono hnext) fun eq _ _ heq => ?_
      refine Ensures.liftBind (extMul_ensures heq hexpected.mono) fun target _ _ htarget => ?_
      refine Ensures.liftBind (assertExtEqual_ensures hlast.mono htarget) fun _ _ _ check => ?_
      refine Ensures.tbind (observeChildren_ensures (r := ts.duplex) hp hd₇.mono hcsize
        fun s hs => (hchildren s hs).mono) fun _ _ _ _ hd₁₁ => ?_
      refine Ensures.tbind (Duplex.sampleExt_ensures hp hd₁₁) fun tau e₁₂ _ _ h₁₂ => ?_
      refine Ensures.tbind (reduceLoop_ensures (gc := gc) (gchildren := (proof.layers[j]!).children)
        (gnext := gpoint) state₀.mono
        (fun u hu => by
          rw [roots]
          exact (hroots u (by rw [hrs, rootsSize]; exact hu)).mono)
        hcsize hga (fun s hs => ⟨(hchildren s hs).mono, hcount s (by rw [← hcsize]; exact hs)⟩)
        (hnext.mono.push h₁₂.1) h₁₂.1) fun st e₁₃ _ _ h₁₃ => ?_
      obtain ⟨he₁₃, state₁₃⟩ := h₁₃
      rw [StateT.run_pure]
      refine Ensures.pure ⟨rfl, _, _, Bridge.runs_gkrLayers_succ hjp run₀ (Bridge.runs_gkrLayer
        (Bridge.runs_sampleExt tj) (Bridge.runs_sampleExt _) sum check (Bridge.runs_observeExts _ _)
        (Bridge.runs_sampleExt _)), ?_, state₁₃, hnext.mono.push h₁₂.1.mono, roots⟩
      have he : e₁₃ = e₁₂ := he₁₃
      rw [he]
      exact h₁₂.2.mono
    · obtain ⟨g, tL, run, hdx, state, -, roots⟩ := q
      have len : view.layers.zipIdx.toList.length = proof.layers.toList.length := by simp [hls]
      rw [len, List.take_length] at run
      rw [len, Array.length_toList] at state
      exact ⟨g, tL, run, hdx, state, roots⟩
  · rw [StateT.run_pure]
    obtain ⟨g, tL, run, hdx, state, roots⟩ := hres
    obtain ⟨c1, c2, c3, c4, c5, c6⟩ := state
    refine Ensures.pure ⟨g, tL, Bridge.runs_bind (Bridge.runs_observeExts _ _) run, hdx,
      ⟨by rw [c2, roots, rootsSize], fun u hu => ?_⟩⟩
    have hu' : u < shapes.size := by rw [← c2]; exact hu
    obtain ⟨root, point, values⟩ := (c6 u hu').2 (by rw [layersSize]; exact Bridge.depth_le_layerCount shapes hu')
    exact ⟨root.1, root.2, point, values⟩

end Verify

end CompressionLean.Sim
