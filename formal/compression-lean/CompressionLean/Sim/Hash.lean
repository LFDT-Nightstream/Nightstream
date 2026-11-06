import CompressionLean.Sim.HashCursor

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lowering
open CompressionLean.Lib
open CompressionLean.Replay (Word Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

attribute [local irreducible] Replay.permute

instance Duplex.new_appends : Appends Duplex.new := by unfold Duplex.new; appends
instance Duplex.duplexing_appends (d : Duplex) : Appends (Duplex.duplexing.run d) := by
  unfold Duplex.duplexing; appends
instance Duplex.observe_appends (v : Var) (d : Duplex) : Appends ((Duplex.observe v).run d) := by
  unfold Duplex.observe; appends
instance Duplex.observeSlice_appends (vs : Array Var) (d : Duplex) : Appends ((Duplex.observeSlice vs).run d) := by
  unfold Duplex.observeSlice Array.forM; rw [← Array.foldlM_toList]; appends
instance Duplex.observeExt_appends (v : ExtVar) (d : Duplex) : Appends ((Duplex.observeExt v).run d) := by
  unfold Duplex.observeExt; appends
instance Duplex.sample_appends (d : Duplex) : Appends (Duplex.sample.run d) := by unfold Duplex.sample; appends
instance Duplex.sampleExt_appends (d : Duplex) : Appends (Duplex.sampleExt.run d) := by
  unfold Duplex.sampleExt; appends
instance Duplex.sampleBits_appends (bits : ℕ) (d : Duplex) : Appends ((Duplex.sampleBits bits).run d) := by
  unfold Duplex.sampleBits; appends

theorem Duplex.new_ensures {c : Ctx F} {n : ℕ} :
    Ensures c n Duplex.new fun d m => IsDuplex c m d Replay.Duplex.new :=
  (const_ensures 0).bind fun _ _ _ hz =>
    Ensures.pure ⟨IsWords.replicate hz 16, IsWords.empty, IsWords.empty, rfl, by decide, by simp⟩

section Duplex

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

private theorem duplexing_ready_ensures {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm)
    (hd : IsDuplex c n d r) :
    Ensures c n (Duplex.duplexing.run d) fun res m => IsDuplex c m res.2 r.duplexing ∧ res.2.cursor = none := by
  unfold Duplex.duplexing
  rw [StateT.run_bind, StateT.run_get, pure_bind]
  dsimp only
  rw [← Array.forIn_toList, Array.toList_zipIdx, forIn_pure (fun x s => s.set! x.2 x.1) (fun x s => by cases x; rfl),
    StateT.run_bind, StateT.run_pure, pure_bind]
  dsimp only
  rw [foldl_zipIdx_set]
  simp only [Array.getElem!_toList, Array.length_toList]
  generalize hs₁ : (List.range d.input.size).foldl (fun s i => s.set! i d.input[i]!) d.state = s₁
  let t₁ := (List.range d.input.size).foldl (fun s i => s.set! i r.input[i]!) r.state
  have ws₁ : IsWords c n s₁ t₁ := by
    rw [← hs₁]
    exact hd.state.setRange _ (fun i => i) fun i hi => hd.input.get (by rw [← hd.input.1]; exact hi)
  have wt₁ : t₁.size = 16 := by simp only [t₁, size_setRange, hd.width]
  by_cases hpos : d.input.size > 0
  · rw [ite_eq_left hpos]
    refine Ensures.liftBind (const_ensures 0) fun z m₁ _ hz => ?_
    rw [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, StateT.run_bind, StateT.run_pure,
      pure_bind]
    dsimp only
    rw [range_foldl d.input.size rate (fun i (s : Array Var) => s.set! i z)]
    generalize hs₂ : (List.range (rate - d.input.size)).foldl (fun s i => s.set! (d.input.size + i) z) s₁ = s₂
    let t₂ := (List.range (rate - d.input.size)).foldl (fun s i => s.set! (d.input.size + i) (0 : Word)) t₁
    let t₃ := t₂.set! rate (t₂[rate]! + Word.ofNat d.input.size)
    have wt₂ : t₂.size = 16 := by simp only [t₂, size_setRange, wt₁]
    have ws₂ : IsWords c m₁ s₂ t₂ := by
      rw [← hs₂]
      exact ws₁.mono.setRange _ (fun i => d.input.size + i) (f := fun _ => z) (g := fun _ => 0) fun _ _ => hz
    refine Ensures.liftBind (const_ensures (Word.ofNat d.input.size)) fun t _ _ ht => ?_
    refine Ensures.liftBind (add_ensures (ws₂.mono.get (by rw [wt₂]; decide)) ht) fun sum _ _ hsum => ?_
    refine Ensures.liftBind (permute_ensures (gs := t₃) (state := fun lane => (s₂.set! rate sum)[lane]!) hp
      fun k => (ws₂.mono.set hsum rate).get (by rw [Array.size_set!, wt₂]; exact k.isLt)) fun outs _ _ houts => ?_
    rw [StateT.run_set]
    refine Ensures.pure ?_
    have hr : r.duplexing = ⟨Replay.permute t₃, #[], (Replay.permute t₃).extract 0 Replay.rate⟩ := by
      unfold Replay.Duplex.duplexing
      simp only [← hd.input.1, ite_eq_left hpos]
      rfl
    rw [hr]
    exact ⟨IsDuplex.permuted t₃ outs houts, rfl⟩
  · rw [ite_eq_right hpos]
    refine Ensures.liftBind (permute_ensures (gs := t₁) (state := fun lane => s₁[lane]!) hp
      fun k => ws₁.get (by rw [wt₁]; exact k.isLt)) fun outs _ _ houts => ?_
    rw [StateT.run_set]
    refine Ensures.pure ?_
    have hr : r.duplexing = ⟨Replay.permute t₁, #[], (Replay.permute t₁).extract 0 Replay.rate⟩ := by
      unfold Replay.Duplex.duplexing
      simp only [← hd.input.1, ite_eq_right hpos]
      rfl
    rw [hr]
    exact ⟨IsDuplex.permuted t₁ outs houts, rfl⟩

theorem Duplex.duplexing_ensures {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : IsDuplex c n d r) :
    Ensures c n (Duplex.duplexing.run d) fun res m => IsDuplex c m res.2 r.duplexing :=
  (duplexing_ready_ensures hp hd).post fun _ _ h => h.1

theorem Duplex.observe_ensures {d : Duplex} {r : Replay.Duplex} {v : Var} {g : Word} (hp : Permutes c.perm)
    (hd : IsDuplex c n d r) (hv : IsWord c n v g) :
    Ensures c n ((Duplex.observe v).run d) fun res m => IsDuplex c m res.2 (r.observe g) := by
  unfold Duplex.observe
  rw [StateT.run_bind, StateT.run_modify, pure_bind, StateT.run_bind, StateT.run_get, pure_bind]
  dsimp only
  have hd' : IsDuplex c n ⟨d.state, d.input.push v, #[], none⟩ ⟨r.state, r.input.push g, #[]⟩ :=
    ⟨hd.state, hd.input.push hv, IsWords.empty, hd.width, by simp, by simp⟩
  by_cases full : (d.input.push v).size = rate
  · rw [ite_eq_left full]
    refine (Duplex.duplexing_ensures hp hd').post fun _ _ q => ?_
    have e : (r.input.push g).size = Replay.rate := by rw [← hd'.input.1]; exact full
    have hr : r.observe g = Replay.Duplex.duplexing ⟨r.state, r.input.push g, #[]⟩ := by
      simp only [Replay.Duplex.observe, e, beq_self_eq_true, ite_true]
    rw [hr]
    exact q
  · rw [ite_eq_right full, StateT.run_pure]
    refine Ensures.pure ?_
    have e : ¬(r.input.push g).size = Replay.rate := by rw [← hd'.input.1]; exact full
    have hr : r.observe g = ⟨r.state, r.input.push g, #[]⟩ := by
      simp only [Replay.Duplex.observe, beq_iff_eq, e, ite_false]
    rw [hr]
    exact hd'

private theorem observeList_ensures (hp : Permutes c.perm) :
    ∀ (l : List Var) (gl : List Word) {n : ℕ} {d : Duplex} {r : Replay.Duplex}, IsDuplex c n d r →
      l.length = gl.length → (∀ i < l.length, IsWord c n l[i]! gl[i]!) →
      Ensures c n ((l.foldlM (fun _ v => Duplex.observe v) ()).run d)
        fun res m => IsDuplex c m res.2 (gl.foldl Replay.Duplex.observe r)
  | [], gl, _, _, _, hd, hlen, _ => by
      rw [List.foldlM_nil, StateT.run_pure, List.eq_nil_of_length_eq_zero hlen.symm]
      exact Ensures.pure hd
  | _ :: _, [], _, _, _, _, hlen, _ => absurd hlen (by simp)
  | v :: l, g :: gl, _, _, _, hd, hlen, hl => by
      rw [List.foldlM_cons, StateT.run_bind]
      refine (Duplex.observe_ensures hp hd (g := g) (by simpa using hl 0 (by simp))).bind fun res _ _ hres => ?_
      exact observeList_ensures hp l gl hres (by simpa using hlen) fun i hi => by
        simpa using (hl (i + 1) (by simp; omega)).mono

theorem Duplex.observeSlice_ensures {d : Duplex} {r : Replay.Duplex} {vs : Array Var} {gs : Array Word}
    (hp : Permutes c.perm) (hd : IsDuplex c n d r) (hv : IsWords c n vs gs) :
    Ensures c n ((Duplex.observeSlice vs).run d)
      fun res m => IsDuplex c m res.2 (gs.foldl Replay.Duplex.observe r) := by
  unfold Duplex.observeSlice Array.forM
  rw [← Array.foldlM_toList, ← Array.foldl_toList]
  exact observeList_ensures hp vs.toList gs.toList hd (by simpa using hv.1) fun i hi => by
    rw [Array.getElem!_toList, Array.getElem!_toList]
    exact hv.2 i (by simpa using hi)

theorem Duplex.observeExt_ensures {d : Duplex} {r : Replay.Duplex} {a : ExtVar} {e : Ext} (hp : Permutes c.perm)
    (hd : IsDuplex c n d r) (ha : IsExt c n a e) :
    Ensures c n ((Duplex.observeExt a).run d)
      fun res m => IsDuplex c m res.2 (((r.observe e.c0).observe e.c1).observe e.c2) := by
  unfold Duplex.observeExt
  refine (Duplex.observe_ensures hp hd (ha 0)).tbind fun _ _ _ _ h₁ => ?_
  refine (Duplex.observe_ensures hp h₁ (ha 1).mono).tbind fun _ _ _ _ h₂ => ?_
  exact Duplex.observe_ensures hp h₂ (ha 2).mono

theorem Duplex.sample_ensures {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : IsDuplex c n d r) :
    Ensures c n (Duplex.sample.run d) fun res m => IsWord c m res.1 r.sample.1 ∧ IsDuplex c m res.2 r.sample.2 := by
  unfold Duplex.sample
  rw [StateT.run_bind, StateT.run_get, pure_bind]
  dsimp only
  by_cases dynamic : d.cursor.isSome = true
  · rw [ite_eq_left dynamic]
    refine Ensures.liftBind (const_ensures 1) fun one _ _ hone => ?_
    simpa using Duplex.sampleIf_ensures (take := true) hp hd.mono hone (hd.cursorInput dynamic)
  rw [ite_eq_right dynamic]
  have static : d.cursor = none := by cases hc : d.cursor <;> simp_all
  split
  · rename_i cond
    rw [StateT.run_bind]
    refine (duplexing_ready_ensures hp hd).bind fun res _ _ hres => ?_
    rw [StateT.run_bind, StateT.run_get, pure_bind, StateT.run_bind, StateT.run_set, pure_bind, StateT.run_pure]
    refine Ensures.pure ?_
    have e : (!r.input.isEmpty || r.output.isEmpty) = true := hd.cond static ▸ cond
    simp only [Replay.Duplex.sample, e, ite_true]
    exact hres.1.pop hres.2 (by rw [duplexing_output_size]; decide)
  · rename_i cond
    rw [StateT.run_bind, StateT.run_get, pure_bind, StateT.run_bind, StateT.run_set, pure_bind, StateT.run_pure]
    refine Ensures.pure ?_
    have e : ¬(!r.input.isEmpty || r.output.isEmpty) = true := hd.cond static ▸ cond
    simp only [Replay.Duplex.sample, e]
    refine hd.pop static ?_
    simp only [Bool.or_eq_true, Bool.not_eq_true', Array.isEmpty_iff_size_eq_zero, not_or] at e
    omega

theorem Duplex.sampleExt_ensures {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : IsDuplex c n d r) :
    Ensures c n (Duplex.sampleExt.run d)
      fun res m => IsExt c m res.1 ⟨r.sample.1, r.sample.2.sample.1, r.sample.2.sample.2.sample.1⟩ ∧
        IsDuplex c m res.2 r.sample.2.sample.2.sample.2 := by
  unfold Duplex.sampleExt
  refine (Duplex.sample_ensures hp hd).tbind fun _ _ _ _ hx => ?_
  refine (Duplex.sample_ensures hp hx.2).tbind fun _ _ _ _ hy => ?_
  refine (Duplex.sample_ensures hp hy.2).tbind fun _ _ _ _ hz => ?_
  rw [StateT.run_pure]
  refine Ensures.pure ⟨fun k => ?_, hz.2⟩
  fin_cases k
  · exact hx.1.mono
  · exact hy.1.mono
  · exact hz.1

theorem Duplex.sampleBits_ensures {d : Duplex} {r : Replay.Duplex} (bits : ℕ) (small : bits ≤ 64)
    (hp : Permutes c.perm) (hd : IsDuplex c n d r) :
    Ensures c n ((Duplex.sampleBits bits).run d)
      fun res m => res.1.size = bits ∧ (∀ i < bits, IsWord c m res.1[i]! (Word.ofNat (r.sample.1.toNat / 2 ^ i % 2))) ∧
        IsDuplex c m res.2 r.sample.2 := by
  unfold Duplex.sampleBits
  refine (Duplex.sample_ensures hp hd).tbind fun w e _ _ hw => ?_
  refine Ensures.liftBind (bits_ensures 64 hw.1) fun res _ _ hres => ?_
  obtain ⟨cells, check⟩ := hres
  rw [StateT.run_pure]
  refine Ensures.pure ⟨by simp [small], fun i hi => ?_, hw.2.mono⟩
  have get : ((Array.ofFn res).extract 0 bits)[i]! = res ⟨i, by omega⟩ := by
    rw [getElem!_pos _ _ (by simp; omega)]
    simp
  rw [get]
  obtain ⟨at_, lt_, val⟩ := cells ⟨i, by omega⟩
  refine ⟨lt_, ?_⟩
  rw [val, at_]
  exact bits_digits check i (by omega)

end Duplex

end CompressionLean.Sim
