import CompressionLean.Ports.Circuit.Linear
import CompressionLean.Sim.EvalK
import CompressionLean.Sim.Honest

/-!
Owns the simulation of the linear stage of `Linear.lean` (`verifier.rs::verify`) against
`Replay.linearStage`.

- `linearStage_tracks`: once the circuit's checks hold, the circuit's linear stage tracks the
  replay's, and `IsLinearOut` gives what the honest fold, the late stage and the openings read.
- The circuit runs one sum-check over every linear round and splits its point at the block bits; the
  replay runs one sum-check per part (`splitSumcheck_tracks`, from `runs_sumcheckReplay_split`).
- The circuit adds the terms of `Ω~` in Rust's order, and `omegaComputed` sums the public terms first.
  Field addition is associative, so the values agree.
-/

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Replay (Word Ext Relation D laneVars)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-! ## Instruction shape -/

instance laneValue_appends (zeta : ExtVar) (point : Array ExtVar) : Appends (Circuit.laneValue zeta point) := by
  unfold Circuit.laneValue; appends
instance linearStage_appends (rel : Relation) (statement : ClaimView) (quotient : Array ExtVar)
    (linear : Array (Array ExtVar)) (finals : Array ExtVar) (d : Duplex) :
    Appends ((Circuit.linearStage rel statement quotient linear finals).run d) := by
  unfold Circuit.linearStage; appends

/-- What the honest fold, the late stage and the openings read of the linear stage
(`verifier.rs::verify`): the mixing, `τ`, the block, lane and linear points, and the finals. -/
def IsLinearOut (c : Ctx F) (m : ℕ) (rel : Relation) (v : LinearView) (out : Replay.LinearOut) : Prop :=
  (IsMixing c m v.mixing out.mixing ∧ v.mixing.powers.size = rel.rowCount) ∧ IsExts c m v.tau out.tau ∧
    IsExts c m v.blockPoint out.blockPoint ∧ IsExts c m v.lanePoint out.lanePoint ∧
    IsExts c m v.linearPoint (out.lanePart ++ out.blockPoint) ∧ IsExt c m v.keyValue out.keyValue ∧
    IsExt c m v.omegaValue out.omegaValue ∧ IsExt c m v.zValue out.zValue

/-- `IsLinearOut` survives later values. -/
theorem IsLinearOut.mono {c : Ctx F} {n m : ℕ} {rel : Relation} {v : LinearView} {out : Replay.LinearOut}
    (h : IsLinearOut c n rel v out) (le : n ≤ m := by omega) : IsLinearOut c m rel v out := by
  obtain ⟨⟨mix, size⟩, tau, block, lane, linear, key, omega, z⟩ := h
  exact ⟨⟨mix.mono le, size⟩, tau.mono le, block.mono le, lane.mono le, linear.mono le, key.mono le, omega.mono le,
    z.mono le⟩

section Stage

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

/-- `lane_value` (`verifier.rs::lane_value`): `Σ_{l<D} eq(x, l)·ζ^l`. -/
private theorem laneValue_ensures {zeta : ExtVar} {point : Array ExtVar} {gz : Ext} {gpoint : Array Ext}
    (size : D ≤ 2 ^ gpoint.size) (hz : IsExt c n zeta gz) (hp : IsExts c n point gpoint) :
    Ensures c n (Circuit.laneValue zeta point) fun r m => IsExt c m r (Replay.laneValue gz gpoint) := by
  unfold Circuit.laneValue
  refine (eqTable_ensures hp).bind fun eq _ _ heq => extZero_ensures.bind fun total _ _ hzero =>
    extOne_ensures.bind fun power _ _ hone => ?_
  have eqSize : eq.size = 2 ^ gpoint.size := by rw [heq.1]; simp [Replay.eqTable]
  refine (Ensures.loopArray (fun k acc m => IsExt c m acc.1 (Replay.sumRange k fun l =>
      (Replay.eqTable gpoint)[l]! * gz.pow l) ∧ IsExt c m acc.2 (gz.pow k))
    ⟨hzero.mono, by rw [ext_pow_zero]; exact hone⟩ fun k hk acc m _ hacc => ?_).bind
    fun res _ _ hres => Ensures.pure ?_
  · obtain ⟨htotal, hpower⟩ := hacc
    have hk' : k < eq.size := by simp at hk; omega
    have weight : IsExt c m (eq.extract 0 D)[k] (Replay.eqTable gpoint)[k]! := by
      have e : (eq.extract 0 D)[k] = eq[k]! := by rw [getElem!_pos eq k hk']; simp
      rw [e]
      exact (heq.2 k hk').mono
    refine (extMul_ensures weight hpower).bind fun term _ _ hterm => ?_
    refine (extAdd_ensures htotal.mono hterm).bind fun total' _ _ htotal' => ?_
    refine (extMul_ensures hpower.mono hz.mono).bind fun power' _ _ hpower' => Ensures.pure ⟨rfl, ?_, ?_⟩
    · rw [sumRange_succ]
      exact htotal'.mono
    · rw [ext_pow_succ]
      exact hpower'
  · have e := hres.1
    rw [Array.size_extract, eqSize, Nat.min_eq_left size, Nat.sub_zero] at e
    exact e

/-- The Eval_K terms `Σ weight·value` added to `Ω~` (`verifier.rs::verify`), which the replay
computes without a transcript step. -/
private theorem pairs_tracks {xs : List (ExtVar × ExtVar)} {gxs : List (Ext × Ext)} {init : ExtVar}
    {ginit : Ext} (hlen : xs.length = gxs.length)
    (hx : ∀ k (hk : k < xs.length), IsExt c n xs[k].1 (gxs[k]!).1 ∧ IsExt c n xs[k].2 (gxs[k]!).2)
    (hi : IsExt c n init ginit) :
    Tracks c n (forIn xs init fun x (omega : ExtVar) => do
        let term ← liftM (Build.extMul x.1 x.2)
        let omega ← liftM (Build.extAdd omega term)
        Pure.pure (ForInStep.yield omega))
      (Pure.pure PUnit.unit) fun res _ m => IsExt c m res (gxs.foldl (fun total p => total + p.1 * p.2) ginit) := by
  intro d t hd
  refine (Ensures.loopT (fun k (acc : ExtVar) e m =>
      IsExt c m acc ((gxs.take k).foldl (fun total p => total + p.1 * p.2) ginit) ∧ IsDuplex c m e t.duplex)
    ⟨hi, hd⟩ fun k hk acc e m _ hI => ?_).post fun _ _ q => ?_
  · obtain ⟨hacc, he⟩ := hI
    refine Ensures.liftBind (extMul_ensures (hx k hk).1.mono (hx k hk).2.mono) fun term _ _ hterm => ?_
    refine Ensures.liftBind (extAdd_ensures hacc.mono hterm) fun total _ _ htotal => ?_
    rw [StateT.run_pure]
    refine Ensures.pure ⟨rfl, ?_, he.mono⟩
    have hk' : k < gxs.length := hlen ▸ hk
    rw [List.take_add_one, List.getElem?_eq_getElem hk', Option.toList_some, List.foldl_append, List.foldl_cons,
      List.foldl_nil, ← getElem!_pos gxs k hk']
    exact htotal
  · obtain ⟨hres, he⟩ := q
    rw [List.take_of_length_le (by omega)] at hres
    exact ⟨(), t, Bridge.runs_pure _ _, he, hres⟩

/-- The public terms `Σ_b eq(s_b, b)·λ^(public row b)` added to `Ω~` (`verifier.rs::verify`), which
the replay computes without a transcript step. -/
private theorem public_tracks {point : Array ExtVar} {gpoint : Array Ext} {mixing : MixingVar}
    {mix : Replay.Mixing} {blocks : ℕ} {init : ExtVar} {ginit : Ext} (hp : IsExts c n point gpoint)
    (hm : IsMixing c n mixing mix) (hsize : mix.kappa + 2 + 2 * mix.matrices + blocks ≤ mixing.powers.size)
    (hi : IsExt c n init ginit) :
    Tracks c n (forIn [:blocks] init fun block (omega : ExtVar) => do
        let eq ← liftM (Algebra.eqAt point block)
        let term ← liftM (Build.extMul eq (mixing.public block))
        let omega ← liftM (Build.extAdd omega term)
        Pure.pure (ForInStep.yield omega))
      (Pure.pure PUnit.unit) fun res _ m =>
        IsExt c m res (ginit + Replay.sumRange blocks fun b => Replay.eqAt gpoint b * mix.public b) := by
  intro d t hd
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  refine (Ensures.loopT (fun k (acc : ExtVar) e m =>
      IsExt c m acc (ginit + Replay.sumRange k fun b => Replay.eqAt gpoint b * mix.public b) ∧
        IsDuplex c m e t.duplex) ⟨?_, hd⟩ fun k hk acc e m _ hI => ?_).post fun _ _ q => ?_
  · show IsExt c n init (ginit + 0)
    rw [ext_add_zero]
    exact hi
  · obtain ⟨hacc, he⟩ := hI
    have hk' : k < blocks := by simpa [Std.Legacy.Range.size] using hk
    rw [List.getElem_range', Nat.zero_add, Nat.one_mul]
    refine Ensures.liftBind (eqAt_ensures k hp.mono) fun eq _ _ heq => ?_
    refine Ensures.liftBind (extMul_ensures heq (hm.public (by omega)).mono) fun term _ _ hterm => ?_
    refine Ensures.liftBind (extAdd_ensures hacc.mono hterm) fun total _ _ htotal => ?_
    rw [StateT.run_pure]
    refine Ensures.pure ⟨rfl, ?_, he.mono⟩
    rw [sumRange_succ, ← ext_add_assoc]
    exact htotal
  · obtain ⟨hres, he⟩ := q
    rw [show (List.range' 0 [:blocks].size 1).length = blocks by simp [Std.Legacy.Range.size]] at hres
    exact ⟨(), t, Bridge.runs_pure _ _, he, hres⟩

/-- The circuit's one sum-check over every round (`verifier.rs::verify`), then the rest, tracks the
replay's two, split at `k` (`runs_sumcheckReplay_split`). -/
private theorem splitSumcheck_tracks {γ δ : Type} (hp : Permutes c.perm) {rounds : Array (Array ExtVar)}
    {grounds : Array (Array Ext)} {claim : ExtVar} {gclaim : Ext} (k : ℕ) (hr : rounds.size = grounds.size)
    (hm : ∀ i < rounds.size, IsExts c n rounds[i]! grounds[i]!) (hne : ∀ g ∈ grounds, 0 < g.size)
    (hc : IsExt c n claim gclaim) {g : Array ExtVar × ExtVar → Transcript γ}
    {y : Array Ext × Ext → Array Ext × Ext → Replay.ReplayM δ} {R : γ → δ → ℕ → Prop}
    (hg : ∀ res first middle second last m, n ≤ m → first.size = min k grounds.size →
      second.size = grounds.size - k → IsExts c m res.1 (first ++ second) → IsExt c m res.2 last →
      Tracks c m (g res) (y (first, middle) (second, last)) R)
    (hg' : ∀ a d, Appends ((g a).run d) := by appends) :
    Tracks c n (Circuit.sumcheckReplay rounds claim >>= g) (Replay.sumcheckReplay (grounds.extract 0 k) gclaim >>=
      fun p => Replay.sumcheckReplay (grounds.extract k) p.2 >>= fun q => y p q) R := by
  intro d t hd
  refine Ensures.tbind (sumcheckReplay_tracks hp hr hm hne hc d t hd) fun res e m le h => ?_
  obtain ⟨out, t', run, hdx, hpt, hfin⟩ := h
  obtain ⟨first, middle, t₁, second, run₁, run₂, hpoint, size₁, size₂⟩ := Bridge.runs_sumcheckReplay_split k run
  refine (hg res first middle second out.2 m le size₁ size₂ (hpoint ▸ hpt) hfin e t' hdx).post
    fun _ _ ⟨b, t'', run', hdx', hr'⟩ => ⟨b, t'', Bridge.runs_bind run₁ (Bridge.runs_bind run₂ run'), hdx', hr'⟩

/-- **The linear stage** (`verifier.rs::verify`). Once the circuit's checks hold, the circuit's
linear stage tracks the replay's, and the circuit's outputs are the replay's. `long` is
`EvalKBits.long`: `eval_k`'s coordinate bits fit the statement point. -/
theorem linearStage_tracks {rel : Relation} {statement : ClaimView} {st : Replay.Statement}
    {quotient : Array ExtVar} {linear : Array (Array ExtVar)} {finals : Array ExtVar} {proof : Replay.Proof}
    (hp : Permutes c.perm) (fits : st.fits rel = true) (hst : IsClaim c n statement st)
    (hq : IsExts c n quotient proof.quotient) (hls : linear.size = proof.linear.size)
    (hl : ∀ i < linear.size, IsExts c n linear[i]! proof.linear[i]!) (hf : IsExts c n finals proof.finals)
    (linearSize : proof.linear.size = rel.cubeVars) (nonempty : ∀ g ∈ proof.linear, 0 < g.size)
    (finalsSize : proof.finals.size = 3) (long : Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo ≤ rel.pointVariables) :
    Tracks c n (Circuit.linearStage rel statement quotient linear finals) (Replay.linearStage rel st proof)
      fun v out m => IsLinearOut c m rel v out := by
  have pointSize : st.point.size = rel.pointVariables := by
    have sizes := fits
    simp only [Replay.Statement.fits, Bool.and_eq_true, beq_iff_eq] at sizes
    exact sizes.1.1.1.2
  have finalsAt : ∀ i < 3, IsExt c n finals[i]! proof.finals[i]! := fun i hi =>
    hf.2 i (by rw [hf.1, finalsSize]; exact hi)
  unfold Circuit.linearStage Replay.linearStage
  refine Tracks.bind (Duplex.sampleExt_tracks hp) fun lambda glam _ _ hlam => ?_
  refine Tracks.lift (MixingVar.new_ensures rel hlam) fun mixing _ _ hmix => ?_
  obtain ⟨hmix, mixSize⟩ := hmix
  refine Tracks.bind (observeExts_tracks hp hq.mono) fun _ _ _ _ _ => ?_
  refine Tracks.bind (Duplex.sampleExt_tracks hp) fun zeta gz _ _ hz => ?_
  refine Tracks.lift (Ring.targets_ensures fits hst.mono hmix.mono mixSize) fun targets _ _ htargets => ?_
  refine Tracks.lift (Ring.liftedTarget_ensures htargets hq.mono hz.mono) fun value _ _ hvalue => ?_
  refine splitSumcheck_tracks hp rel.blockVars hls (fun i hi => (hl i hi).mono) nonempty hvalue
    fun res first middle second last _ _ size₁ size₂ hpt hlast => ?_
  obtain ⟨point, lastV⟩ := res
  dsimp only at hpt hlast ⊢
  have bsize : first.size = rel.blockVars := by
    rw [size₁, linearSize, Relation.cubeVars]
    omega
  have lsize : second.size = laneVars := by
    rw [size₂, linearSize, Relation.cubeVars]
    omega
  have e₁ : (first ++ second).extract 0 rel.blockVars = first := by
    rw [← bsize, Array.extract_append_left, Array.extract_size]
  have e₂ : (first ++ second).extract rel.blockVars = second := by
    rw [← bsize, Array.size_append, Array.extract_append_right, Array.extract_size]
  have hb := hpt.slice 0 rel.blockVars
  have hlane := hpt.extract rel.blockVars
  rw [e₁] at hb
  rw [e₂] at hlane
  refine Tracks.bind (observeExts_tracks hp hf.mono) fun _ _ _ _ _ => ?_
  refine Tracks.lift (Mle.scaledEqPoint_ensures laneVars hz.mono) fun sc _ _ hsc => ?_
  obtain ⟨gscale, glanePoint, ok, hscale, hlp⟩ := hsc
  refine Tracks.check ok ?_
  obtain ⟨scale, lanePoint⟩ := sc
  dsimp only at hscale hlp ⊢
  refine Tracks.lift (Ring.tau_ensures hz.mono) fun tau _ _ htau => ?_
  refine Tracks.lift (Mle.evalK_ensures rel.key.blocks (by rw [bsize, pointSize]; exact long) (by simp [Replay.tau])
    hst.point.mono hb.mono htau) fun ek _ _ hek => ?_
  refine Tracks.lift (extMul_ensures hscale.mono (finalsAt 0 (by decide)).mono) fun o₀ _ _ ho₀ => ?_
  have hmk := hmix.evalK (by
    rw [mixSize]
    show rel.kappa + 1 < rel.kappa + 2 + 2 * rel.matrices + rel.publicBlocks
    omega)
  refine Tracks.quiet (pairs_tracks (gxs := [((⟨glam, rel.kappa, rel.matrices⟩ : Replay.Mixing).evalK.1,
      (Replay.evalK st.point first (Replay.tau gz) rel.key.blocks).re),
      ((⟨glam, rel.kappa, rel.matrices⟩ : Replay.Mixing).evalK.2,
      (Replay.evalK st.point first (Replay.tau gz) rel.key.blocks).im)]) rfl (fun k hk => ?_) ho₀)
    fun o₂ _ _ ho₂ => ?_
  · obtain _ | _ | k := k
    · exact ⟨hmk.1.mono, hek.1.mono⟩
    · exact ⟨hmk.2.mono, hek.2.mono⟩
    · simp at hk
  refine Tracks.lift (extAdd_ensures ho₂ (finalsAt 1 (by decide)).mono) fun o₃ _ _ ho₃ => ?_
  refine Tracks.quiet (public_tracks hb.mono hmix.mono (by rw [mixSize]; exact le_of_eq rfl) ho₃)
    fun o₄ _ _ ho₄ => ?_
  refine Tracks.lift (laneValue_ensures (by rw [lsize]; decide) hz.mono hlane.mono) fun lanes _ _ hlanes => ?_
  refine Tracks.lift (extMul_ensures ho₄.mono hlanes) fun weighted _ _ hweighted => ?_
  refine Tracks.lift (extMul_ensures hweighted (finalsAt 2 (by decide)).mono) fun expected _ _ hexpected => ?_
  refine Tracks.lift (assertExtEqual_ensures hlast.mono hexpected) fun _ _ _ check => ?_
  refine Tracks.ensure (by rw [check]; exact Replay.Ext.beq_self _) ?_
  exact Tracks.pure ⟨⟨hmix.mono, mixSize⟩, htau.mono, hb.mono, hlp.mono, hlane.mono.append hb.mono,
    (finalsAt 0 (by decide)).mono, (finalsAt 1 (by decide)).mono, (finalsAt 2 (by decide)).mono⟩

end Stage

end CompressionLean.Sim
