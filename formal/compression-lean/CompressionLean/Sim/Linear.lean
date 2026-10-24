import CompressionLean.Ports.Circuit.Linear
import CompressionLean.Sim.EvalK
import CompressionLean.Sim.Honest

/-!
Owns the simulation of the linear stage of `Linear.lean` (`verifier.rs:170-206`) against
`Replay.linearStage`.

- `linearStage_ensures`: once the circuit's checks hold, the replay's linear stage runs, and
  `IsLinearOut` gives what the honest fold, the late stage and the openings read.
- The circuit runs one sum-check over every linear round and splits its point at the block bits; the
  replay runs one sum-check per part (`runs_sumcheckReplay_split`).
- The circuit adds the terms of `Ω~` in Rust's order, and `omegaComputed` sums the public terms first.
  Field addition is associative, so the values agree.
-/

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Replay (Word Ext Relation D laneVars)
open CompressionLean.Bridge (Runs runs_bind runs_pure runs_sampleExt runs_observeExts runs_ensure
  runs_liftExcept)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-- What the honest fold, the late stage and the openings read of the linear stage
(`verifier.rs:170-206`): the mixing, `τ`, the block, lane and linear points, and the finals. -/
def IsLinearOut (c : Ctx F) (m : ℕ) (rel : Relation) (v : LinearView) (out : Replay.LinearOut) : Prop :=
  (IsMixing c m v.mixing out.mixing ∧ v.mixing.powers.size = rel.rowCount) ∧ IsExts c m v.tau out.tau ∧
    IsExts c m v.blockPoint out.blockPoint ∧ IsExts c m v.lanePoint out.lanePoint ∧
    IsExts c m v.linearPoint (out.lanePart ++ out.blockPoint) ∧ IsExt c m v.keyValue out.keyValue ∧
    IsExt c m v.omegaValue out.omegaValue ∧ IsExt c m v.zValue out.zValue

/-- `IsLinearOut` survives later values. -/
theorem IsLinearOut.mono {c : Ctx F} {n m : ℕ} {rel : Relation} {v : LinearView} {out : Replay.LinearOut}
    (h : IsLinearOut c n rel v out) (le : n ≤ m) : IsLinearOut c m rel v out :=
  ⟨⟨h.1.1.mono le, h.1.2⟩, h.2.1.mono le, h.2.2.1.mono le, h.2.2.2.1.mono le, h.2.2.2.2.1.mono le,
    h.2.2.2.2.2.1.mono le, h.2.2.2.2.2.2.1.mono le, h.2.2.2.2.2.2.2.mono le⟩

variable {c : Ctx F} {n : ℕ} {P : Prop}

/-- A builder call in a transcript keeps the duplex (`verifier.rs:187-206`). -/
private theorem OldEnsures.liftBind {α β : Type} {b : Build α} {g : α → Transcript β} {d : Duplex}
    {Q : α → ℕ → Prop} {R : β × Duplex → ℕ → Prop} (hb : OldEnsures c n P b Q)
    (hg : ∀ a m, n ≤ m → OldEnsures c m (P ∧ Q a m) ((g a).run d) R) :
    OldEnsures c n P ((monadLift b >>= g : Transcript β).run d) R := by
  rw [StateT.run_bind, StateT.run_monadLift, bind_assoc]
  exact hb.bind fun a m le => by rw [pure_bind]; exact hg a m le

variable [CharP F goldilocksModulus]

/-- `lane_value` (`verifier.rs:340-350`): `Σ_{l<D} eq(x, l)·ζ^l`. -/
private theorem Simulates.laneValue {zeta : ExtVar} {point : Array ExtVar} {gz : Ext} {gpoint : Array Ext}
    (size : P → D ≤ 2 ^ gpoint.size) (hz : P → IsExt c n zeta gz) (hp : P → IsExts c n point gpoint) :
    Simulates c n P (Circuit.laneValue zeta point) (fun r m => IsExt c m r (Replay.laneValue gz gpoint)) True := by
  unfold Circuit.laneValue
  refine (Simulates.eqTable hp).bind₀ fun eq m₁ l₁ => Simulates.extZero.bind₀ fun total m₂ l₂ =>
    Simulates.extOne.bind₀ fun power m₃ l₃ => ?_
  refine ((Simulates.loop₀ (P := ((P ∧ IsExts c m₁ eq (Replay.eqTable gpoint)) ∧ IsExt c m₂ total 0) ∧
      IsExt c m₃ power 1)
    (fun k acc m => IsExt c m acc.1 (Replay.sumRange k fun l => (Replay.eqTable gpoint)[l]! * gz.pow l) ∧
      IsExt c m acc.2 (gz.pow k)) fun k hk acc m lm => ?_).pre fun h => ⟨h, h.1.2.mono l₃, ?_⟩).bind₀
    fun res m₄ l₄ => Simulates.pure fun h => ?_
  · have hk' : k < eq.size := by simp at hk; omega
    have weight : P ∧ IsExts c m₁ eq (Replay.eqTable gpoint) → IsExt c m₁ (eq.extract 0 D)[k]
        (Replay.eqTable gpoint)[k]! := fun h => by
      have e : (eq.extract 0 D)[k] = eq[k]! := by rw [getElem!_pos eq k hk']; simp
      rw [e]
      exact h.2.2 k hk'
    refine (Simulates.extMul (fun h => (weight h.1.1.1).mono (l₂.trans (l₃.trans lm))) fun h => h.2.2).bind₀
      fun term m₅ l₅ => (Simulates.extAdd (fun h => h.1.2.1.mono l₅) fun h => h.2).bind₀ fun total' m₆ l₆ =>
      (Simulates.extMul (fun h => h.1.1.2.2.mono (l₅.trans l₆))
        fun h => (hz h.1.1.1.1.1.1).mono (l₁.trans (l₂.trans (l₃.trans (lm.trans (l₅.trans l₆)))))).bind₀
      fun power' m₇ l₇ => Simulates.pure fun h => ⟨rfl, ?_, ?_⟩
    · rw [sumRange_succ]
      exact h.1.2.mono l₇
    · rw [ext_pow_succ]
      exact h.2
  · rw [ext_pow_zero]
    exact h.2
  · have e := h.2.1
    have eqSize : eq.size = 2 ^ gpoint.size := by rw [h.1.1.1.2.1]; simp [Replay.eqTable]
    rw [Array.size_extract, eqSize, Nat.min_eq_left (size h.1.1.1.1), Nat.sub_zero] at e
    exact e

/-- `eval_k` (`mle.rs:146-204`) with the size of `r` known under `P`. -/
private theorem Simulates.evalK' {r : Array KVar} {sBlock tau : Array ExtVar} {gr : Array (Word × Word)} {gs gt : Array Ext}
    (blocks : ℕ) (rSize : P → Nat.log2 (D * 2 ^ gs.size).nextPowerOfTwo ≤ gr.size) (tauSize : D ≤ gt.size)
    (hr : P → IsKs c n r gr) (hs : P → IsExts c n sBlock gs) (ht : P → IsExts c n tau gt) :
    Simulates c n P (Mle.evalK r sBlock tau blocks) (fun res m => IsKx c m res (Replay.evalK gr gs gt blocks)) True := by
  by_cases h : Nat.log2 (D * 2 ^ gs.size).nextPowerOfTwo ≤ gr.size
  · exact Simulates.evalK blocks h tauSize hr hs ht
  · exact (Simulates.evalK (gr := Array.replicate (Nat.log2 (D * 2 ^ gs.size).nextPowerOfTwo) (0, 0)) blocks (by simp)
      tauSize (fun hP => absurd (rSize hP) h) hs ht).post (fun hP => absurd (rSize hP) h) fun _ => Iff.rfl

/-- The Eval_K terms `Σ weight·value` added to `Ω~` (`verifier.rs:192-196`). -/
private theorem pairsLoop {d : Duplex} {xs : List (ExtVar × ExtVar)} {gxs : List (Ext × Ext)} {init : ExtVar}
    {ginit : Ext} (hlen : xs.length = gxs.length)
    (hx : P → ∀ k (hk : k < xs.length), IsExt c n xs[k].1 (gxs[k]!).1 ∧ IsExt c n xs[k].2 (gxs[k]!).2)
    (hi : P → IsExt c n init ginit) :
    Simulates c n P ((forIn xs init fun x (omega : ExtVar) => do
        let term ← liftM (Build.extMul x.1 x.2)
        let omega ← liftM (Build.extAdd omega term)
        Pure.pure (ForInStep.yield omega) : Transcript ExtVar).run d)
      (fun res m => IsExt c m res.1 (gxs.foldl (fun total p => total + p.1 * p.2) ginit) ∧ res.2 = d) True := by
  refine ((Simulates.loopT (P := P) (d := d) (I := fun k (acc : ExtVar) e m =>
      IsExt c m acc ((gxs.take k).foldl (fun total p => total + p.1 * p.2) ginit) ∧ e = d) (D := fun _ => True)
      fun k hk acc e m lm => ?_).pre fun h => by exact ⟨h, hi h, rfl⟩).post (fun h _ _ q => ?_) fun _ => by simp
  · refine Simulates.tbind₀ (Simulates.lift (Simulates.extMul (fun h => ((hx h.1) k hk).1.mono lm) fun h => ((hx h.1) k hk).2.mono lm))
      fun term e₁ m₁ l₁ => ?_
    refine Simulates.tbind₀ (Simulates.lift (Simulates.extAdd (fun h => h.1.2.1.mono l₁) fun h => h.2.1)) fun total e₂ m₂ l₂ => ?_
    rw [StateT.run_pure]
    refine Simulates.pure fun h => ⟨rfl, ?_, h.2.2.trans (h.1.2.2.trans h.1.1.2.2)⟩
    have hk' : k < gxs.length := hlen ▸ hk
    rw [List.take_add_one, List.getElem?_eq_getElem hk', Option.toList_some, List.foldl_append, List.foldl_cons,
      List.foldl_nil, ← getElem!_pos gxs k hk']
    exact h.2.1
  · rw [List.take_of_length_le (by omega)] at q
    exact q

/-- The public terms `Σ_b eq(s_b, b)·λ^(public row b)` added to `Ω~` (`verifier.rs:198-202`). -/
private theorem publicLoop {d : Duplex} {point : Array ExtVar} {gpoint : Array Ext} {mixing : MixingVar}
    {mix : Replay.Mixing} {blocks : ℕ} {init : ExtVar} {ginit : Ext} (hp : P → IsExts c n point gpoint)
    (hm : P → IsMixing c n mixing mix) (hsize : P → mix.kappa + 2 + 2 * mix.matrices + blocks ≤ mixing.powers.size)
    (hi : P → IsExt c n init ginit) :
    Simulates c n P ((forIn [:blocks] init fun block (omega : ExtVar) => do
        let eq ← liftM (Algebra.eqAt point block)
        let term ← liftM (Build.extMul eq (mixing.public block))
        let omega ← liftM (Build.extAdd omega term)
        Pure.pure (ForInStep.yield omega) : Transcript ExtVar).run d)
      (fun res m => IsExt c m res.1 (ginit + Replay.sumRange blocks fun b => Replay.eqAt gpoint b * mix.public b) ∧
        res.2 = d) True := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  refine ((Simulates.loopT (P := P) (d := d) (I := fun k (acc : ExtVar) e m =>
      IsExt c m acc (ginit + Replay.sumRange k fun b => Replay.eqAt gpoint b * mix.public b) ∧ e = d)
      (D := fun _ => True) fun k hk acc e m lm => ?_).pre fun h => ⟨h, ?_, rfl⟩).post (fun h _ _ q => ?_)
      fun _ => by simp
  · have hk' : k < blocks := by simpa [Std.Legacy.Range.size] using hk
    rw [List.getElem_range', Nat.zero_add, Nat.one_mul]
    refine Simulates.tbind₀ (Simulates.lift (Simulates.eqAt k fun h => (hp h.1).mono lm)) fun eq e₁ m₁ l₁ => ?_
    refine Simulates.tbind₀ (Simulates.lift (Simulates.extMul (fun h => h.2.1)
      fun h => ((hm h.1.1).public (by have := hsize h.1.1; omega)).mono (lm.trans l₁))) fun term e₂ m₂ l₂ => ?_
    refine Simulates.tbind₀ (Simulates.lift (Simulates.extAdd (fun h => h.1.1.2.1.mono (l₁.trans l₂)) fun h => h.2.1))
      fun total e₃ m₃ l₃ => ?_
    rw [StateT.run_pure]
    refine Simulates.pure fun h => ⟨rfl, ?_, h.2.2.trans (h.1.2.2.trans (h.1.1.2.2.trans h.1.1.1.2.2))⟩
    rw [sumRange_succ, ← ext_add_assoc]
    exact h.2.1
  · show IsExt c n init (ginit + 0)
    rw [ext_add_zero]
    exact hi h
  · rw [show (List.range' 0 [:blocks].size 1).length = blocks by simp [Std.Legacy.Range.size]] at q
    exact q

/-- **The linear stage** (`verifier.rs:170-206`). Once the circuit's checks hold, the replay's linear
stage runs from the matching transcript, the circuit's outputs are the replay's, and the duplexes still
match. `long` is `EvalKBits.long`: `eval_k`'s coordinate bits fit the statement point. -/
theorem linearStage_ensures {rel : Relation} {statement : ClaimView} {st : Replay.Statement}
    {quotient : Array ExtVar} {linear : Array (Array ExtVar)} {finals : Array ExtVar} {proof : Replay.Proof}
    {d : Duplex} {t : Replay.Transcript} (hp : Permutes c.perm) (hd : P → IsDuplex c n d t.duplex)
    (fits : st.fits rel = true) (hst : P → IsClaim c n statement st)
    (hq : P → IsExts c n quotient proof.quotient)
    (hl : P → linear.size = proof.linear.size ∧ ∀ i < linear.size, IsExts c n linear[i]! proof.linear[i]!)
    (hf : P → IsExts c n finals proof.finals) (linearSize : P → proof.linear.size = rel.cubeVars)
    (nonempty : P → ∀ g ∈ proof.linear, 0 < g.size) (finalsSize : P → proof.finals.size = 3)
    (long : Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo ≤ rel.pointVariables) :
    OldEnsures c n P ((Circuit.linearStage rel statement quotient linear finals).run d)
      (fun res m => ∃ out t', Runs (Replay.linearStage rel st proof) t out t' ∧
        IsLinearOut c m rel res.1 out ∧ IsDuplex c m res.2 t'.duplex) := by
  have pointSize : st.point.size = rel.pointVariables := by
    have sizes := fits
    simp only [Replay.Statement.fits, Bool.and_eq_true, beq_iff_eq] at sizes
    exact sizes.1.1.1.2
  have finalsAt : P → ∀ i < 3, IsExt c n finals[i]! proof.finals[i]! := fun h i hi =>
    (hf h).2 i (by rw [(hf h).1, finalsSize h]; exact hi)
  unfold Circuit.linearStage
  refine OldEnsures.tbind (Simulates.sampleExt hp hd).ensures fun lambda d₁ m₁ l₁ => ?_
  refine OldEnsures.tbind (OldEnsures.lift (Simulates.mixing rel fun h => h.2.1.1).ensures) fun mixing d₂ m₂ l₂ => ?_
  refine OldEnsures.tbind (Simulates.observeExts hp (fun h => h.2.2 ▸ h.1.2.1.2.mono l₂)
    fun h => (hq h.1.1).mono (l₁.trans l₂)).ensures fun _ d₃ m₃ l₃ => ?_
  refine OldEnsures.tbind (Simulates.sampleExt hp fun h => h.2.1).ensures fun zeta d₄ m₄ l₄ => ?_
  set glam : Ext := ⟨t.duplex.sample.1, t.duplex.sample.2.sample.1, t.duplex.sample.2.sample.2.sample.1⟩
  set r₁ := t.duplex.sample.2.sample.2.sample.2
  set r₃ := (proof.quotient.flatMap fun g => #[g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe r₁
  set gz : Ext := ⟨r₃.sample.1, r₃.sample.2.sample.1, r₃.sample.2.sample.2.sample.1⟩
  set r₄ := r₃.sample.2.sample.2.sample.2
  set mix : Replay.Mixing := ⟨glam, rel.kappa, rel.matrices⟩
  have L₄ : n ≤ m₄ := l₁.trans (l₂.trans (l₃.trans l₄))
  refine OldEnsures.pre (P' := P ∧ IsMixing c m₄ mixing mix ∧ mixing.powers.size = rel.rowCount ∧
    IsExt c m₄ zeta gz ∧ IsDuplex c m₄ d₄ r₄) ?_ fun h => ⟨h.1.1.1.1, h.1.1.2.1.1.1.mono (l₃.trans l₄),
      h.1.1.2.1.1.2, h.2.1.1, h.2.1.2⟩
  refine OldEnsures.tbind (OldEnsures.lift (Simulates.targets fits (fun h => (hst h.1).mono L₄)
    fun h => ⟨h.2.1, h.2.2.1⟩).ensures) fun targets d₅ m₅ l₅ => ?_
  refine OldEnsures.tbind (OldEnsures.lift (Simulates.liftedTarget (fun h => h.2.1.1) (fun h => (hq h.1.1).mono (L₄.trans l₅))
    fun h => h.1.2.2.2.1.mono l₅).ensures) fun value d₆ m₆ l₆ => ?_
  set gvalue := Replay.liftedTarget (Replay.targets st mix) proof.quotient gz
  set t₃ : Replay.Transcript := { duplex := r₄, draws := (t.draws.push (.ext glam)).push (.ext gz) }
  have L₆ : m₄ ≤ m₆ := l₅.trans l₆
  refine OldEnsures.pre (P' := P ∧ IsMixing c m₆ mixing mix ∧ mixing.powers.size = rel.rowCount ∧
    IsExt c m₆ zeta gz ∧ IsExt c m₆ value gvalue ∧ IsDuplex c m₆ d₆ t₃.duplex) ?_ fun h => ?_
  rotate_left
  · obtain ⟨⟨⟨hP, hm, hs, hz, hd₄⟩, -, e₅⟩, ⟨hv, -⟩, e₆⟩ := h
    dsimp only at e₅ e₆
    exact ⟨hP, hm.mono L₆, hs, hz.mono L₆, hv, by rw [e₆, e₅]; exact hd₄.mono L₆⟩
  refine OldEnsures.tbind ((Simulates.sumcheckReplay hp (t := t₃) (fun h => h.2.2.2.2.2)
    (fun h => ⟨(hl h.1).1, fun i hi => ((hl h.1).2 i hi).mono (L₄.trans L₆)⟩) (fun h => nonempty h.1)
    fun h => h.2.2.2.2.1).ensures.post (Q' := fun res m => ∃ gb gmiddle t₅ glane glast t₇,
      Runs (Replay.sumcheckReplay (proof.linear.extract 0 rel.blockVars) gvalue) t₃ (gb, gmiddle) t₅ ∧
      Runs (Replay.sumcheckReplay (proof.linear.extract rel.blockVars) gmiddle) t₅ (glane, glast) t₇ ∧
      gb.size = rel.blockVars ∧ glane.size = laneVars ∧ IsExts c m (res.1.1.extract 0 rel.blockVars) gb ∧
      IsExts c m (res.1.1.extract rel.blockVars) glane ∧ IsExt c m res.1.2 glast ∧ IsDuplex c m res.2 t₇.duplex)
    fun h r m q => ?_) fun res d₇ m₇ l₇ => ?_
  · obtain ⟨⟨gpoint, gfinal, t', run, hpt, hfin, hdup⟩, -⟩ := q
    obtain ⟨gb, gmiddle, t₅, glane, run₁, run₂, rfl, size₁, size₂⟩ :=
      Bridge.runs_sumcheckReplay_split rel.blockVars run
    have bsize : gb.size = rel.blockVars := by
      rw [size₁, linearSize h.1, Relation.cubeVars]
      omega
    have lsize : glane.size = laneVars := by
      rw [size₂, linearSize h.1, Relation.cubeVars]
      omega
    have e₁ : (gb ++ glane).extract 0 rel.blockVars = gb := by
      rw [← bsize, Array.extract_append_left, Array.extract_size]
    have e₂ : (gb ++ glane).extract rel.blockVars = glane := by
      rw [← bsize, Array.size_append, Array.extract_append_right, Array.extract_size]
    have b₁ := hpt.slice 0 rel.blockVars
    have b₂ := hpt.extract rel.blockVars
    rw [e₁] at b₁
    rw [e₂] at b₂
    exact ⟨gb, gmiddle, t₅, glane, gfinal, t', run₁, run₂, bsize, lsize, b₁, b₂, hfin, hdup⟩
  obtain ⟨point, last⟩ := res
  refine OldEnsures.exists_and (#[] : Array Ext) fun gb => OldEnsures.exists_and (0 : Ext) fun gmiddle =>
    OldEnsures.exists_and t fun t₅ => OldEnsures.exists_and (#[] : Array Ext) fun glane =>
    OldEnsures.exists_and (0 : Ext) fun glast => OldEnsures.exists_and t fun t₇ => ?_
  dsimp only
  have L₇ : n ≤ m₇ := L₄.trans (L₆.trans l₇)
  refine OldEnsures.tbind (Simulates.observeExts hp (fun h => h.2.2.2.2.2.2.2.2) fun h => (hf h.1.1).mono L₇).ensures
    fun _ d₈ m₈ l₈ => ?_
  refine OldEnsures.liftBind (OldEnsures.scaledEqPoint laneVars fun h => h.1.1.2.2.2.1.mono (l₇.trans l₈))
    fun sc m₉ l₉ => ?_
  obtain ⟨scale, lanePoint⟩ := sc
  refine OldEnsures.exists_and (0 : Ext) fun gscale => OldEnsures.exists_and (#[] : Array Ext) fun glanePoint => ?_
  dsimp only
  set r₈ := (proof.finals.flatMap fun g => #[g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe t₇.duplex
  have M₉ : m₇ ≤ m₉ := l₈.trans l₉
  refine OldEnsures.pre (P' := P ∧
    (Runs (Replay.sumcheckReplay (proof.linear.extract 0 rel.blockVars) gvalue) t₃ (gb, gmiddle) t₅ ∧
      Runs (Replay.sumcheckReplay (proof.linear.extract rel.blockVars) gmiddle) t₅ (glane, glast) t₇ ∧
      gb.size = rel.blockVars ∧ glane.size = laneVars ∧
      Replay.scaledEqPoint gz laneVars = .ok (gscale, glanePoint)) ∧
    IsMixing c m₉ mixing mix ∧ mixing.powers.size = rel.rowCount ∧ IsExt c m₉ zeta gz ∧
    IsExts c m₉ (point.extract 0 rel.blockVars) gb ∧ IsExts c m₉ (point.extract rel.blockVars) glane ∧
    IsExt c m₉ last glast ∧ IsDuplex c m₉ d₈ r₈ ∧ IsExt c m₉ scale gscale ∧ IsExts c m₉ lanePoint glanePoint) ?_
    fun h => ?_
  rotate_left
  · obtain ⟨⟨⟨⟨hP, hm, hs, hz, -, -⟩, run₁, run₂, bsize, lsize, hb, hlane, hlast, -⟩, hd₈, -⟩, ok, hsc, hlp⟩ := h
    exact ⟨hP, ⟨run₁, run₂, bsize, lsize, ok⟩, hm.mono (l₇.trans M₉), hs, hz.mono (l₇.trans M₉), hb.mono M₉,
      hlane.mono M₉, hlast.mono M₉, hd₈.mono l₉, hsc, hlp⟩
  refine OldEnsures.liftBind (Simulates.tau fun h => h.2.2.2.2.1).ensures fun tau m₁₀ l₁₀ => ?_
  refine OldEnsures.liftBind (Simulates.evalK' rel.key.blocks (fun h => by rw [h.1.2.1.2.2.1, pointSize]; exact long)
    (by simp [Replay.tau]) (fun h => (hst h.1.1).point.mono (L₇.trans (M₉.trans l₁₀)))
    (fun h => h.1.2.2.2.2.2.1.mono l₁₀) fun h => h.2.1).ensures fun ek m₁₁ l₁₁ => ?_
  refine OldEnsures.liftBind (Simulates.extMul (fun h => h.1.1.2.2.2.2.2.2.2.2.2.1.mono (l₁₀.trans l₁₁))
    (fun h => (finalsAt h.1.1.1 0 (by decide)).mono (L₇.trans (M₉.trans (l₁₀.trans l₁₁))))).ensures
    fun o₀ m₁₂ l₁₂ => ?_
  set gek := Replay.evalK st.point gb (Replay.tau gz) rel.key.blocks
  refine OldEnsures.tbind (pairsLoop (gxs := [(mix.evalK.1, gek.re), (mix.evalK.2, gek.im)]) rfl (fun h k hk => ?_)
    fun h => h.2.1).ensures fun o₂ d₁₃ m₁₃ l₁₃ => ?_
  · have hmk := (h.1.1.1.2.2.1).evalK (by
      rw [h.1.1.1.2.2.2.1]
      show rel.kappa + 1 < rel.kappa + 2 + 2 * rel.matrices + rel.publicBlocks
      omega)
    have hek := h.1.2.1
    obtain _ | _ | k := k
    · exact ⟨hmk.1.mono (l₁₀.trans (l₁₁.trans l₁₂)), hek.1.mono l₁₂⟩
    · exact ⟨hmk.2.mono (l₁₀.trans (l₁₁.trans l₁₂)), hek.2.mono l₁₂⟩
    · simp at hk
  refine OldEnsures.liftBind (Simulates.extAdd (fun h => h.2.1.1) fun h => (finalsAt h.1.1.1.1.1 1 (by decide)).mono
    (L₇.trans (M₉.trans (l₁₀.trans (l₁₁.trans (l₁₂.trans l₁₃)))))).ensures fun o₃ m₁₄ l₁₄ => ?_
  have M₁₄ : m₉ ≤ m₁₄ := l₁₀.trans (l₁₁.trans (l₁₂.trans (l₁₃.trans l₁₄)))
  refine OldEnsures.tbind (publicLoop (fun h => h.1.1.1.1.1.2.2.2.2.2.1.mono M₁₄) (fun h => h.1.1.1.1.1.2.2.1.mono M₁₄)
    (fun h => by rw [h.1.1.1.1.1.2.2.2.1]; exact le_of_eq rfl) fun h => h.2.1).ensures fun o₄ d₁₄ m₁₅ l₁₅ => ?_
  set gomega := Replay.omegaComputed rel st mix (Replay.tau gz) gb gscale proof.finals[0]! proof.finals[1]!
  have M₁₅ : m₉ ≤ m₁₅ := M₁₄.trans l₁₅
  refine OldEnsures.pre (P' := P ∧
    (Runs (Replay.sumcheckReplay (proof.linear.extract 0 rel.blockVars) gvalue) t₃ (gb, gmiddle) t₅ ∧
      Runs (Replay.sumcheckReplay (proof.linear.extract rel.blockVars) gmiddle) t₅ (glane, glast) t₇ ∧
      gb.size = rel.blockVars ∧ glane.size = laneVars ∧
      Replay.scaledEqPoint gz laneVars = .ok (gscale, glanePoint)) ∧
    IsMixing c m₁₅ mixing mix ∧ mixing.powers.size = rel.rowCount ∧ IsExt c m₁₅ zeta gz ∧
    IsExts c m₁₅ (point.extract 0 rel.blockVars) gb ∧ IsExts c m₁₅ (point.extract rel.blockVars) glane ∧
    IsExt c m₁₅ last glast ∧ IsDuplex c m₁₅ d₁₄ r₈ ∧ IsExts c m₁₅ lanePoint glanePoint ∧
    IsExts c m₁₅ tau (Replay.tau gz) ∧ IsExt c m₁₅ o₄ gomega) ?_ fun h => ?_
  rotate_left
  · obtain ⟨⟨⟨⟨⟨⟨⟨hP, hR, hm, hs, hz, hb, hlane, hlast, hd₈, -, hlp⟩, htau, -⟩, -, -⟩, -, -⟩, ⟨-, e₁₃⟩, -⟩, -, -⟩,
      ⟨ho, e₁₄⟩, -⟩ := h
    dsimp only at e₁₃ e₁₄
    exact ⟨hP, hR, hm.mono M₁₅, hs, hz.mono M₁₅, hb.mono M₁₅, hlane.mono M₁₅, hlast.mono M₁₅,
      by rw [e₁₄, e₁₃]; exact hd₈.mono M₁₅, hlp.mono M₁₅,
      htau.mono (l₁₁.trans (l₁₂.trans (l₁₃.trans (l₁₄.trans l₁₅)))), ho⟩
  refine OldEnsures.liftBind (Simulates.laneValue (fun h => by rw [h.2.1.2.2.2.1]; decide) (fun h => h.2.2.2.2.1)
    fun h => h.2.2.2.2.2.2.1).ensures fun lanes m₁₆ l₁₆ => ?_
  refine OldEnsures.liftBind (Simulates.extMul (fun h => h.1.2.2.2.2.2.2.2.2.2.2.2.mono l₁₆) fun h => h.2.1).ensures
    fun weighted m₁₇ l₁₇ => ?_
  refine OldEnsures.liftBind (Simulates.extMul (fun h => h.2.1) fun h => (finalsAt h.1.1.1 2 (by decide)).mono
    (L₇.trans (M₉.trans (M₁₅.trans (l₁₆.trans l₁₇))))).ensures fun expected m₁₈ l₁₈ => ?_
  refine OldEnsures.liftBind (Simulates.assertExtEqual (fun h => h.1.1.1.2.2.2.2.2.2.2.1.mono (l₁₆.trans (l₁₇.trans l₁₈)))
    fun h => h.2.1).ensures fun _ m₁₉ l₁₉ => ?_
  rw [StateT.run_pure]
  refine OldEnsures.pure fun h => ?_
  obtain ⟨⟨⟨⟨⟨hP, ⟨run₁, run₂, -, -, ok⟩, hm, hs, -, hb, hlane, -, hdup, hlp, htau, -⟩, -, -⟩, -, -⟩, -, -⟩, -,
    check⟩ := h
  have N : m₁₅ ≤ m₁₉ := l₁₆.trans (l₁₇.trans (l₁₈.trans l₁₉))
  have hfin : ∀ i < 3, IsExt c m₁₉ finals[i]! proof.finals[i]! := fun i hi =>
    (finalsAt hP i hi).mono (L₇.trans (M₉.trans (M₁₅.trans N)))
  have checkOk : (glast == gomega * Replay.laneValue gz glane * proof.finals[2]!) = true := by
    rw [check]
    exact Replay.Ext.beq_self _
  refine ⟨{ lam := glam, zeta := gz, blockPoint := gb, middle := gmiddle, lanePart := glane, last := glast,
            keyValue := proof.finals[0]!, omegaValue := proof.finals[1]!, zValue := proof.finals[2]!,
            lanePoint := glanePoint, mixing := mix, tau := Replay.tau gz }, { t₇ with duplex := r₈ }, ?_,
    ⟨⟨hm.mono N, hs⟩, htau.mono N, hb.mono N, hlp.mono N, (hlane.mono N).append (hb.mono N), hfin 0 (by decide),
      hfin 1 (by decide), hfin 2 (by decide)⟩, hdup.mono N⟩
  unfold Replay.linearStage
  exact runs_bind (runs_sampleExt t) (runs_bind (runs_observeExts _ _) (runs_bind (runs_sampleExt _)
    (runs_bind run₁ (runs_bind run₂ (runs_bind (runs_observeExts _ _) (runs_bind (runs_liftExcept ok _)
    (runs_bind (runs_ensure checkOk _ _) (runs_pure _ _))))))))

end CompressionLean.Sim
