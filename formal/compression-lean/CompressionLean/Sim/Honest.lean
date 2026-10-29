import CompressionLean.Sim.SetupLeaf
import CompressionLean.Sim.Early

namespace CompressionLean.Sim

open CompressionLean.Spec
open CompressionLean.Circuit
open CompressionLean.Lib
open CompressionLean.Replay (Word Ext Relation foldVars runOracles generator twoAdicGenerator)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

private theorem foldl_take_succ {α β : Type} (f : β → α → β) (b : β) (l : List α) {k : ℕ}
    (hk : k < l.length) : (l.take (k + 1)).foldl f b = f ((l.take k).foldl f b) l[k] := by
  rw [List.take_add_one, List.getElem?_eq_getElem hk, Option.toList_some, List.foldl_append, List.foldl_cons,
    List.foldl_nil]

theorem foldl_coords (vs : Array Ext) (r : Replay.Duplex) :
    (vs.flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe r =
      vs.foldl (fun r v => ((r.observe v.c0).observe v.c1).observe v.c2) r := by
  rw [← Array.foldl_toList, Array.toList_flatMap, List.foldl_flatMap, ← Array.foldl_toList]
  rfl

theorem digit_mod (x b i : ℕ) (hi : i < b) : x % 2 ^ b / 2 ^ i % 2 = x / 2 ^ i % 2 := by
  have e := Nat.testBit_mod_two_pow x b i
  rw [decide_eq_true hi, Bool.true_and, Nat.testBit_eq_decide_div_mod_eq, Nat.testBit_eq_decide_div_mod_eq,
    decide_eq_decide] at e
  omega

private theorem mapM_sampleExt {β : Type} (g : Array ExtVar → Transcript β) :
    ((Array.range foldVars).mapM (fun _ => Duplex.sampleExt) >>= g) =
      (Duplex.sampleExt >>= fun a => Duplex.sampleExt >>= fun b => Duplex.sampleExt >>= fun e => g #[a, b, e]) := by
  simp [Array.mapM_eq_mapM_toList, foldVars, List.range_succ]

instance setupPoint_appends (point : Array ExtVar) (vars : ℕ) : Appends (Circuit.setupPoint point vars) := by
  unfold Circuit.setupPoint; appends
instance combinePartials_appends (partials point : Array ExtVar) :
    Appends (Circuit.combinePartials partials point) := by
  unfold Circuit.combinePartials; appends
instance queryPoint_appends (vars : ℕ) (bits : Array Var) : Appends (Circuit.queryPoint vars bits) := by
  unfold Circuit.queryPoint; appends
instance honestStage_appends (rel : Relation) (input : HonestInput) (d : Duplex) :
    Appends ((Circuit.honestStage rel input).run d) := by
  unfold Circuit.honestStage; appends

section Handles

variable {c : Ctx F} {n : ℕ}

theorem setupPoint_ensures {point : Array ExtVar} {gpoint : Array Ext} {vars : ℕ} (hp : IsExts c n point gpoint)
    (size : gpoint.size ≤ vars) :
    Ensures c n (Circuit.setupPoint point vars) fun r m => IsExts c m r (Replay.padded vars gpoint) := by
  unfold Circuit.setupPoint
  refine extZero_ensures.bind fun zero _ _ hzero => Ensures.pure ?_
  rw [Array.extract_eq_self_of_le (by rw [hp.1]; exact size), hp.1]
  exact hp.mono.append (IsExts.replicate hzero _)

private theorem rootConsts_tracks (words : Array Word) :
    Tracks c n (forIn words (#[] : Array Var) fun word (root : Array Var) => do
        let w ← liftM (Build.const word)
        Pure.pure (ForInStep.yield (root.push w)))
      (Pure.pure PUnit.unit) fun res _ m => IsWords c m res words := by
  intro d t hd
  rw [← Array.forIn_toList]
  refine (Ensures.loopT (fun k (acc : Array Var) e m => IsWords c m acc (words.toList.take k).toArray ∧ IsDuplex c m e t.duplex)
    ⟨IsWords.empty, hd⟩ fun k hk acc e m _ hI => ?_).post fun _ _ q => ?_
  · obtain ⟨hacc, he⟩ := hI
    refine Ensures.liftBind (const_ensures words.toList[k]) fun w _ _ hw => ?_
    rw [StateT.run_pure]
    refine Ensures.pure ⟨rfl, ?_, he.mono⟩
    rw [List.take_add_one, List.getElem?_eq_getElem hk, Option.toList_some, ← List.push_toArray]
    exact hacc.mono.push hw
  · obtain ⟨hres, he⟩ := q
    rw [List.take_of_length_le (by simp), Array.toArray_toList] at hres
    exact ⟨(), t, Bridge.runs_pure _ _, he, hres⟩

def IsHonestOut (c : Ctx F) (m : ℕ) (rel : Relation) (early : Replay.EarlyOut) (lin : Replay.LinearOut)
    (v : HonestOut) (out : Replay.HonestOut) : Prop :=
  IsExts c m v.alpha out.alpha ∧
    IsExts c m v.keyPoint (Replay.padded rel.setupVars (lin.lanePoint ++ lin.blockPoint)) ∧
    IsExts c m v.runPoint (Replay.padded rel.setupVars early.openings.runPoint) ∧
    v.indexBits.size = out.queries.size ∧
    (∀ q < v.indexBits.size, (v.indexBits[q]!).size = rel.setupVars + 1 - foldVars ∧
      ∀ i < rel.setupVars + 1 - foldVars, IsWord c m (v.indexBits[q]!)[i]! (Word.ofNat (out.queries[q]! / 2 ^ i % 2))) ∧
    IsExts c m v.queried out.queried

theorem IsHonestOut.mono {m m' : ℕ} {rel : Relation} {early : Replay.EarlyOut} {lin : Replay.LinearOut}
    {v : HonestOut} {out : Replay.HonestOut} (h : IsHonestOut c m rel early lin v out) (le : m ≤ m' := by omega) :
    IsHonestOut c m' rel early lin v out := by
  obtain ⟨ha, hk, hr, hs, hb, hq⟩ := h
  exact ⟨ha.mono le, hk.mono le, hr.mono le, hs, fun q lt => ⟨(hb q lt).1, fun i hi => ((hb q lt).2 i hi).mono le⟩,
    hq.mono le⟩

end Handles

section Stage

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

theorem combinePartials_ensures {partials point : Array ExtVar} {gpartials gpoint : Array Ext}
    (hpa : IsExts c n partials gpartials) (hpo : IsExts c n point gpoint) :
    Ensures c n (Circuit.combinePartials partials point)
      fun r m => IsExt c m r (Replay.combine gpartials gpoint) :=
  (eqTable_ensures hpo).bind fun _ _ _ ht => dot_ensures ht hpa.mono

theorem queryPoint_ensures {vars index : ℕ} {bits : Array Var} (hsize : bits.size = vars + 1 - foldVars)
    (hb : ∀ i < bits.size, IsWord c n bits[i]! (Word.ofNat (index / 2 ^ i % 2))) :
    Ensures c n (Circuit.queryPoint vars bits) fun r m => IsExts c m r (Replay.leafQueryPoint vars index) := by
  unfold Circuit.queryPoint Replay.leafQueryPoint
  dsimp only
  generalize hq : Replay.powerFromBits (twoAdicGenerator (vars + 1 - foldVars)) index (vars + 1 - foldVars) *
    generator.pow (2 ^ foldVars) = q
  refine (powerFromBits_ensures _ index hb).bind fun rotation _ _ hrotation => ?_
  refine (scale_ensures (generator.pow (2 ^ foldVars)) hrotation).bind fun start _ _ hstart => ?_
  rw [hsize, hq] at hstart
  refine (Ensures.loopRange (fun k (acc : Var × Array ExtVar) m => IsWord c m acc.1 (q.pow (2 ^ k)) ∧
      IsExts c m acc.2 ((Array.range k).map fun k => Ext.ofGl (q.pow (2 ^ k)))) ?_
    fun k _ acc m _ hacc => ?_).bind fun res _ _ hres => Ensures.pure ?_
  · have e : q = q.pow (2 ^ 0) := Word.eq_of_toF (by rw [Word.toF_pow, pow_zero, pow_one])
    exact ⟨e ▸ hstart, IsExts.range_zero _⟩
  · obtain ⟨hcur, hpoint⟩ := hacc
    refine (lift_ensures hcur).bind fun lifted _ _ hlifted => ?_
    refine (mul_ensures hcur.mono hcur.mono).bind fun next _ _ hnext => Ensures.pure ⟨rfl, ?_, ?_⟩
    · have e : q.pow (2 ^ k) * q.pow (2 ^ k) = q.pow (2 ^ (k + 1)) := Word.eq_of_toF (by
        rw [Word.toF_mul, Word.toF_pow, Word.toF_pow, ← pow_add, ← two_mul, ← pow_succ'])
      exact e ▸ hnext
    · rw [map_range_succ]
      exact hpoint.mono.push hlifted.mono
  · have e := hres.2
    rw [Nat.sub_zero] at e
    rw [show ((List.range (vars - foldVars)).map fun k => Ext.ofGl (q.pow (2 ^ k))).toArray =
      (Array.range (vars - foldVars)).map fun k => Ext.ofGl (q.pow (2 ^ k)) from Array.ext' (by simp)]
    exact e

private theorem observePartials_tracks {partials : Array (Array ExtVar)} {gpartials : Array (Array Ext)}
    (hp : Permutes c.perm) (hsize : partials.size = gpartials.size)
    (hpa : ∀ g < partials.size, IsExts c n partials[g]! gpartials[g]!) :
    Tracks c n (forIn partials PUnit.unit fun values _ => do
        forIn values PUnit.unit fun value _ => do
          Duplex.observeExt value
          Pure.pure (ForInStep.yield PUnit.unit)
        Pure.pure (ForInStep.yield PUnit.unit))
      (Replay.observeExts gpartials.flatten) fun _ _ _ => True := by
  intro d t hd
  rw [← Array.forIn_toList]
  refine (Ensures.loopT (fun k _ e m => IsDuplex c m e ((gpartials.toList.take k).foldl
      (fun r vs => vs.foldl (fun r v => ((r.observe v.c0).observe v.c1).observe v.c2) r) t.duplex)) hd
    fun k hk _ e m _ hI => ?_).post fun _ m q => ⟨(), _, Bridge.runs_observeExts _ t, ?_, trivial⟩
  · have hk' : k < partials.size := by simpa using hk
    rw [show partials.toList[k] = partials[k]! by simp [getElem!_pos partials k hk']]
    refine Ensures.tbind (observeExts_ensures hp hI (hpa k hk').mono) fun _ e' m' _ hd' => ?_
    rw [StateT.run_pure]
    refine Ensures.pure ⟨rfl, ?_⟩
    rw [foldl_coords] at hd'
    rwa [foldl_take_succ _ _ _ (by simp; omega), Array.getElem_toList, ← getElem!_pos]
  · rw [List.take_of_length_le (by simp [hsize])] at q
    dsimp only
    rwa [foldl_coords, Array.foldl_flatten, ← Array.foldl_toList]

private theorem sampleBitsList_ensures (bits : ℕ) (small : bits ≤ 64) (hp : Permutes c.perm) :
    ∀ (l : List ℕ) {n : ℕ} {d : Duplex} (t : Replay.Transcript) (init : Array Word), IsDuplex c n d t.duplex →
      Ensures c n ((l.mapM fun _ => Duplex.sampleBits bits : Transcript (List (Array Var))).run d)
        fun res m => ∃ ws : List Word, ∃ t' : Replay.Transcript,
          Runs (l.foldlM (fun values _ => do Pure.pure (values.push (← Replay.sampleWord))) init) t
            (init ++ ws.toArray) t' ∧
          ws.length = l.length ∧ res.1.length = l.length ∧
          (∀ q < l.length, (res.1[q]!).size = bits ∧
            ∀ i < bits, IsWord c m (res.1[q]!)[i]! (Word.ofNat ((ws[q]!).toNat / 2 ^ i % 2))) ∧
          IsDuplex c m res.2 t'.duplex
  | [], n, d, t, init, hd => by
      rw [List.mapM_nil, StateT.run_pure]
      exact Ensures.pure ⟨[], t, by simpa using Bridge.runs_pure init t, rfl, rfl,
        fun q hq => absurd hq (by simp), hd⟩
  | x :: l, n, d, t, init, hd => by
      rw [List.mapM_cons]
      refine Ensures.tbind (Duplex.sampleBits_ensures bits small hp hd) fun a e m _ ha => ?_
      obtain ⟨size, digits, hd₁⟩ := ha
      refine Ensures.tbind (sampleBitsList_ensures bits small hp l { t with duplex := t.duplex.sample.2 }
        (init.push t.duplex.sample.1) hd₁) fun as e' m' _ has => ?_
      rw [StateT.run_pure]
      obtain ⟨ws, t', runs, wlen, alen, adigits, hd'⟩ := has
      refine Ensures.pure ⟨t.duplex.sample.1 :: ws, t', ?_, by simp [wlen], by simp [alen], fun q hq => ?_, hd'⟩
      · rw [List.foldlM_cons]
        refine Bridge.runs_bind (Bridge.runs_bind (Bridge.runs_sampleWord t) (Bridge.runs_pure _ _)) ?_
        have e : init.push t.duplex.sample.1 ++ ws.toArray = init ++ (t.duplex.sample.1 :: ws).toArray :=
          Array.ext' (by simp)
        exact e ▸ runs
      · cases q with
        | zero => exact ⟨by simpa using size, fun i hi => by simpa using (digits i hi).mono (m := m')⟩
        | succ q =>
            simp only [List.getElem!_cons_succ]
            exact adigits q (by simpa using hq)

private theorem sampleBits_tracks (bits : ℕ) (small : bits ≤ 64) (hp : Permutes c.perm) (k : ℕ) :
    Tracks c n ((Array.range k).mapM fun _ => Duplex.sampleBits bits) (Replay.sampleWords k)
      fun res ws m => res.size = ws.size ∧ ∀ q < res.size, (res[q]!).size = bits ∧
        ∀ i < bits, IsWord c m (res[q]!)[i]! (Word.ofNat ((ws.map fun w => w.toNat % 2 ^ bits)[q]! / 2 ^ i % 2)) := by
  intro d t hd
  rw [Array.mapM_eq_mapM_toList, StateT.run_map, Array.toList_range]
  refine (sampleBitsList_ensures bits small hp (List.range k) t #[] hd).map fun res m h => ?_
  obtain ⟨ws, t', runs, wlen, rlen, digits, hd'⟩ := h
  refine ⟨ws.toArray, { t' with draws := t'.draws.push (.words ws.toArray) }, ?_, hd', by simp [wlen, rlen],
    fun q hq => ?_⟩
  · unfold Replay.sampleWords
    rw [Array.empty_append] at runs
    exact Bridge.runs_bind runs (Bridge.runs_bind (Bridge.runs_record _ _) (Bridge.runs_pure _ _))
  · have lt : q < (List.range k).length := by simpa [rlen] using hq
    have hq' : q < ws.toArray.size := by simpa [wlen] using lt
    obtain ⟨size, bitsOk⟩ : (res.1.toArray[q]!).size = bits ∧ ∀ i < bits,
        IsWord c m (res.1.toArray[q]!)[i]! (Word.ofNat ((ws.toArray[q]!).toNat / 2 ^ i % 2)) := by
      simpa using digits q lt
    refine ⟨size, fun i hi => ?_⟩
    rw [getElem!_pos (ws.toArray.map fun w => w.toNat % 2 ^ bits) _ (by simpa using hq'), Array.getElem_map,
      ← getElem!_pos ws.toArray _ hq', digit_mod _ _ _ hi]
    exact bitsOk i hi

private theorem runWeights_tracks {one mu : ExtVar} {gmu : Ext} (hone : IsExt c n one 1) (hmu : IsExt c n mu gmu) :
    Tracks c n (forIn [1:runOracles] #[one] fun _ (weights : Array ExtVar) => do
        let next ← liftM (Build.extMul weights.back! mu)
        Pure.pure (ForInStep.yield (weights.push next)))
      (Pure.pure PUnit.unit) fun res _ m => IsExts c m res ((Array.range runOracles).map gmu.pow) := by
  intro d t hd
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  refine (Ensures.loopT (fun k (acc : Array ExtVar) e m => IsExts c m acc ((Array.range (k + 1)).map gmu.pow) ∧
      IsDuplex c m e t.duplex) ⟨?_, hd⟩ fun k _ acc e m _ hI => ?_).post fun _ _ q => ?_
  · rw [show Array.range (0 + 1) = #[0] by decide, Array.map_singleton, ext_pow_zero]
    exact ⟨rfl, fun i hi => match i, hi with | 0, _ => hone⟩
  · obtain ⟨hacc, he⟩ := hI
    refine Ensures.liftBind (extMul_ensures (hacc.back (by simp)) hmu.mono) fun next _ _ hnext => ?_
    rw [StateT.run_pure]
    refine Ensures.pure ⟨rfl, ?_, he.mono⟩
    rw [show ((Array.range (k + 1)).map gmu.pow).back! = gmu.pow k by
      rw [Array.back!, getElem!_pos _ _ (by simp)]; simp, ← ext_pow_succ] at hnext
    rw [map_range_succ]
    exact hacc.mono.push hnext
  · obtain ⟨hres, he⟩ := q
    exact ⟨(), t, Bridge.runs_pure _ _, he, by simpa [runOracles, Std.Legacy.Range.size] using hres⟩

private theorem embed_tracks {xs : Array ExtVar} {gxs : Array Ext} {init : ExtVar} {ginit : Ext}
    (hx : IsExts c n xs gxs) (hi : IsExt c n init ginit) :
    Tracks c n (forIn xs init fun x (embedded : ExtVar) => do
        let low ← liftM (Algebra.oneMinus x)
        let embedded ← liftM (Build.extMul embedded low)
        Pure.pure (ForInStep.yield embedded))
      (Pure.pure PUnit.unit) fun res _ m => IsExt c m res (gxs.foldl (fun total x => total * Replay.oneMinus x) ginit) := by
  intro d t hd
  have size := hx.1
  rw [← Array.forIn_toList]
  refine (Ensures.loopT (fun k (acc : ExtVar) e m =>
      IsExt c m acc ((gxs.toList.take k).foldl (fun total x => total * Replay.oneMinus x) ginit) ∧
        IsDuplex c m e t.duplex) ⟨hi, hd⟩ fun k hk acc e m _ hI => ?_).post fun _ _ q => ?_
  · obtain ⟨hacc, he⟩ := hI
    have hk' : k < xs.size := by simpa using hk
    rw [show xs.toList[k] = xs[k]! by simp [getElem!_pos xs k hk']]
    refine Ensures.liftBind (oneMinus_ensures (hx.2 k hk').mono) fun low _ _ hlow => ?_
    refine Ensures.liftBind (extMul_ensures hacc.mono hlow) fun total _ _ htotal => ?_
    rw [StateT.run_pure]
    refine Ensures.pure ⟨rfl, ?_, he.mono⟩
    rw [foldl_take_succ _ _ _ (by simp; omega), Array.getElem_toList, ← getElem!_pos]
    exact htotal
  · obtain ⟨hres, he⟩ := q
    rw [List.take_of_length_le (by simp [size]), Array.foldl_toList] at hres
    exact ⟨(), t, Bridge.runs_pure _ _, he, hres⟩

private theorem runValue_tracks {values : Array ExtVar} {gvalues : Array Ext} {init mu : ExtVar} {ginit gmu : Ext}
    (hv : IsExts c n values gvalues) (hi : IsExt c n init ginit) (hmu : IsExt c n mu gmu) :
    Tracks c n (forIn values init fun value (runValue : ExtVar) => do
        let scaled ← liftM (Build.extMul mu runValue)
        let runValue ← liftM (Build.extAdd value scaled)
        Pure.pure (ForInStep.yield runValue))
      (Pure.pure PUnit.unit)
      fun res _ m => IsExt c m res (gvalues.foldl (fun total value => value + gmu * total) ginit) := by
  intro d t hd
  have size := hv.1
  rw [← Array.forIn_toList]
  refine (Ensures.loopT (fun k (acc : ExtVar) e m =>
      IsExt c m acc ((gvalues.toList.take k).foldl (fun total value => value + gmu * total) ginit) ∧
        IsDuplex c m e t.duplex) ⟨hi, hd⟩ fun k hk acc e m _ hI => ?_).post fun _ _ q => ?_
  · obtain ⟨hacc, he⟩ := hI
    have hk' : k < values.size := by simpa using hk
    rw [show values.toList[k] = values[k]! by simp [getElem!_pos values k hk']]
    refine Ensures.liftBind (extMul_ensures hmu.mono hacc) fun scaled _ _ hscaled => ?_
    refine Ensures.liftBind (extAdd_ensures (hv.2 k hk').mono hscaled) fun total _ _ htotal => ?_
    rw [StateT.run_pure]
    refine Ensures.pure ⟨rfl, ?_, he.mono⟩
    rw [foldl_take_succ _ _ _ (by simp; omega), Array.getElem_toList, ← getElem!_pos]
    exact htotal
  · obtain ⟨hres, he⟩ := q
    rw [List.take_of_length_le (by simp [size]), Array.foldl_toList] at hres
    exact ⟨(), t, Bridge.runs_pure _ _, he, hres⟩

private theorem queried_tracks {rootV : Array Var} {groot : Array Word} {vars : ℕ}
    {indexBits : Array (Array Var)} {queries : Array ℕ} {leaves : Array LeafView} {gleaves : Array Replay.Leaf}
    {w₀ w₁ alpha : Array ExtVar} {gw₀ gw₁ galpha : Array Ext} (hp : Permutes c.perm)
    (hroot : IsWords c n rootV groot) (root : groot.size = 4) (hbs : indexBits.size = queries.size)
    (hbits : ∀ q < indexBits.size, (indexBits[q]!).size = vars + 1 - foldVars ∧
      ∀ i < vars + 1 - foldVars, IsWord c n (indexBits[q]!)[i]! (Word.ofNat (queries[q]! / 2 ^ i % 2)))
    (hls : leaves.size = gleaves.size)
    (hleaves : ∀ q (hq : q < leaves.size), IsWords c n leaves[q].values (gleaves[q]!).values ∧
      leaves[q].path.size = (gleaves[q]!).path.size ∧ leaves[q].path.size = vars + 1 - foldVars ∧
      ∀ k < leaves[q].path.size, IsDigest c n leaves[q].path[k]! (gleaves[q]!).path[k]!)
    (hw₀ : IsExts c n w₀ gw₀) (hw₁ : IsExts c n w₁ gw₁) (halpha : IsExts c n alpha galpha)
    (hsize : galpha.size = foldVars) :
    Tracks c n (forIn (indexBits.zip leaves) (#[] : Array ExtVar) fun x (queried : Array ExtVar) => match x with
        | (bits, leaf) => do
          let r ← liftM (Circuit.setupCheck (fun lane => rootV[lane]!) vars bits leaf #[w₀, w₁] alpha)
          Pure.pure (ForInStep.yield (queried ++ r)))
      (Pure.pure PUnit.unit) fun res _ m => ∃ gq, (queries.zip gleaves).toList.foldlM (fun queried pair => do
          Pure.pure (queried ++ (← Replay.setupCheck groot vars pair.1 pair.2 #[gw₀, gw₁] galpha))) #[] = .ok gq ∧
        IsExts c m res gq := by
  intro d t hd
  have rootSize := hroot.1
  rw [← Array.forIn_toList]
  refine (Ensures.loopT (fun k (acc : Array ExtVar) e m =>
      (∃ gq, ((queries.zip gleaves).toList.take k).foldlM (fun queried pair => do
        Pure.pure (queried ++ (← Replay.setupCheck groot vars pair.1 pair.2 #[gw₀, gw₁] galpha))) #[] = .ok gq ∧
        IsExts c m acc gq) ∧ IsDuplex c m e t.duplex) ⟨⟨#[], rfl, IsExts.empty⟩, hd⟩ fun k hk acc e m _ hI => ?_).post
    fun _ _ q => ?_
  · obtain ⟨⟨gq, run, hq⟩, he⟩ := hI
    have hk' : k < indexBits.size ∧ k < leaves.size := by simpa using hk
    rw [show (indexBits.zip leaves).toList[k] = (indexBits[k]!, leaves[k]'hk'.2) by
      simp [getElem!_pos indexBits k hk'.1]]
    dsimp only
    obtain ⟨bitsSize, bitsOk⟩ := hbits k hk'.1
    obtain ⟨hvalues, hpsize, hpvars, hpath⟩ := hleaves k hk'.2
    refine Ensures.liftBind (setupCheck_ensures (groot := groot) (index := queries[k]!) hp
      ⟨root, fun lane => by rw [Fin.getElem!_fin]; exact (hroot.get (by omega)).mono⟩ bitsSize
      (fun i hi => (bitsOk i (bitsSize ▸ hi)).mono) hvalues.mono hpsize hpvars (fun j hj => (hpath j hj).mono)
      hw₀.mono hw₁.mono halpha.mono hsize) fun r _ _ hr => ?_
    rw [StateT.run_pure]
    obtain ⟨vals, check, hv⟩ := hr
    have lt : k < (queries.zip gleaves).toList.length := by simp; omega
    refine Ensures.pure ⟨rfl, ⟨gq ++ vals, ?_, hq.mono.append hv⟩, he.mono⟩
    rw [List.take_add_one, List.getElem?_eq_getElem lt, Option.toList_some, List.foldlM_append, run]
    simp only [Array.getElem_toList, Array.getElem_zip, List.foldlM_cons, List.foldlM_nil]
    rw [← getElem!_pos queries k (by simp at lt; omega), ← getElem!_pos gleaves k (by simp at lt; omega), check]
    rfl
  · obtain ⟨hres, he⟩ := q
    rw [List.take_of_length_le (by simp; omega)] at hres
    exact ⟨(), t, Bridge.runs_pure _ _, he, hres⟩

theorem honestStage_tracks {rel : Relation} {input : HonestInput} {proof : Replay.Proof} {early : Replay.EarlyOut}
    {lin : Replay.LinearOut} (hp : Permutes c.perm)
    (hlane : IsExts c n input.lanePoint lin.lanePoint) (hblock : IsExts c n input.blockPoint lin.blockPoint)
    (hrun : IsExts c n input.runPoint early.openings.runPoint)
    (hrow : IsExt c n input.setup.1 early.openings.rowClaim)
    (hslot : IsExt c n input.setup.2.1 early.openings.slotClaim)
    (hcoef : IsExt c n input.setup.2.2 early.openings.coefClaim)
    (hblockValue : IsExt c n input.block early.values.block)
    (hrows : IsExts c n input.commitmentRows ((Array.range rel.kappa).map lin.mixing.commitment))
    (hkey : IsExt c n input.keyValue lin.keyValue) (hps : input.partials.size = proof.partials.size)
    (hpartials : ∀ g < input.partials.size, IsExts c n input.partials[g]! proof.partials[g]!)
    (hp1 : IsDigest c n input.p1 proof.p1) (hls : input.leaves.size = proof.leaves.size)
    (hleaves : ∀ q (hq : q < input.leaves.size), IsWords c n input.leaves[q].values (proof.leaves[q]!).values ∧
      input.leaves[q].path.size = (proof.leaves[q]!).path.size ∧
      input.leaves[q].path.size = rel.setupVars + 1 - foldVars ∧
      ∀ k < input.leaves[q].path.size, IsDigest c n input.leaves[q].path[k]! (proof.leaves[q]!).path[k]!)
    (keySize : (lin.lanePoint ++ lin.blockPoint).size ≤ rel.setupVars)
    (runSize : early.openings.runPoint.size ≤ rel.setupVars) (root : rel.key.root.size = 4)
    (small : rel.setupVars + 1 - foldVars ≤ 64) :
    Tracks c n (Circuit.honestStage rel input) (Replay.honestStage rel proof early lin)
      fun v out m => IsHonestOut c m rel early lin v out := by
  obtain ⟨lanePoint, blockPoint, runPoint, ⟨row, slot, coefficient⟩, block, commitmentRows, keyValue, partials, p1,
    leaves⟩ := input
  dsimp only at hlane hblock hrun hrow hslot hcoef hblockValue hrows hkey hps hpartials hp1 hls hleaves
  unfold Circuit.honestStage Replay.honestStage
  refine Tracks.bind (Duplex.sampleExt_tracks hp) fun mu gmu _ _ hmu => ?_
  refine Tracks.bind (observePartials_tracks hp hps fun g hg => (hpartials g hg).mono) fun _ _ _ _ _ => ?_
  rw [mapM_sampleExt]
  refine Tracks.bind (Duplex.sampleExt_tracks hp) fun a₀ ga₀ _ _ ha₀ => ?_
  refine Tracks.bind (Duplex.sampleExt_tracks hp) fun a₁ ga₁ _ _ ha₁ => ?_
  refine Tracks.bind (Duplex.sampleExt_tracks hp) fun a₂ ga₂ _ _ ha₂ => ?_
  have halpha := IsExts.triple ha₀.mono ha₁.mono ha₂
  refine Tracks.lift (setupPoint_ensures (hlane.append hblock).mono keySize) fun keyPoint _ _ hkp => ?_
  refine Tracks.lift (setupPoint_ensures hrun.mono runSize) fun runPoint' _ _ hrp => ?_
  refine Tracks.quiet (embed_tracks (hrun.extract _).mono hblockValue.mono) fun embedded _ _ hem => ?_
  refine Tracks.quiet (runValue_tracks (IsExts.triple hcoef.mono hslot.mono hrow.mono) hem hmu.mono)
    fun runValue _ _ hrv => ?_
  refine Tracks.lift (combinePartials_ensures (IsExts.nested ⟨hps, hpartials⟩ 0).mono (hkp.slice 0 foldVars).mono)
    fun keyPartial _ _ hkpart => ?_
  refine Tracks.lift (assertExtEqual_ensures hkpart hkey.mono) fun _ _ _ keyCheck => ?_
  refine Tracks.lift (combinePartials_ensures (IsExts.nested ⟨hps, hpartials⟩ 1).mono (hrp.slice 0 foldVars).mono)
    fun runPartial _ _ hrpart => ?_
  refine Tracks.lift (assertExtEqual_ensures hrpart hrv.mono) fun _ _ _ runCheck => ?_
  refine Tracks.ensure (by rw [keyCheck]; exact Ext.beq_self _) (Tracks.ensure (by rw [runCheck]; exact Ext.beq_self _) ?_)
  rw [← bind_assoc (Replay.observeAll rel.commitmentSeed)]
  refine Tracks.bind (observeRoot_tracks hp hp1.mono) fun _ _ _ _ _ => ?_
  refine Tracks.bind (sampleBits_tracks (rel.setupVars + 1 - foldVars) small hp rel.queries)
    fun indexBits ws _ _ hbits => ?_
  obtain ⟨isize, ibits⟩ := hbits
  refine Tracks.lift extOne_ensures fun one _ _ hone => ?_
  refine Tracks.quiet (runWeights_tracks hone hmu.mono) fun runWeights _ _ hw => ?_
  refine Tracks.quiet (rootConsts_tracks rel.key.root) fun rootV _ _ hroot => ?_
  refine Tracks.quiet (queried_tracks (gleaves := proof.leaves) hp hroot root (by simp [isize])
    (fun q hq => ⟨(ibits q hq).1, fun i hi => ((ibits q hq).2 i hi).mono⟩) hls
    (fun q hq => ?_) hrows.mono hw.mono halpha.mono rfl) fun queried _ _ hq => ?_
  · obtain ⟨hvalues, hpsize, hpvars, hpath⟩ := hleaves q hq
    exact ⟨hvalues.mono, hpsize, hpvars, fun k hk => (hpath k hk).mono⟩
  obtain ⟨gq, qok, hq⟩ := hq
  exact Tracks.check qok (Tracks.pure ⟨halpha.mono, hkp.mono, hrp.mono, by simp [isize],
    fun q hq' => ⟨(ibits q hq').1, fun i hi => ((ibits q hq').2 i hi).mono⟩, hq⟩)

end Stage

end CompressionLean.Sim
