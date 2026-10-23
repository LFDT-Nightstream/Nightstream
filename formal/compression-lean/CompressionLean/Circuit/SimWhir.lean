import CompressionLean.Circuit.SimViews
import CompressionLean.Circuit.SimSumcheck

/-!
Owns the simulation of our WHIR verifier (`whirVerify`, `whir/mod.rs:506-661`)
against the replay's transcript of one opening (`Replay.whirTranscript`): when
the checks hold, the duplex is the replay's and the opened values are the
opening's (`whirVerify_ensures`).

- `Appends`, `Quiet`: a builder program appends instructions and one value per
  output; a transcript program also keeps the duplex. The `appends` tactic
  proves both for the arithmetic, the Merkle paths and the STIR checks, which
  the replay does not take.
- `Tracks c n H t x Q`: from a duplex that is the replay's, when the checks of
  `t` hold, a run of `x` ends at its duplex. Its combinators match a program to
  the replay step by step.
- A query index's inverse check keeps the sample from `p - 1`
  (`Ensures.extInv_ne`), as the replay's `whirQueries` checks.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl Ext WhirShape SumcheckShape Runs ReplayM)
open NightstreamFPrime.Spec (goldilocksModulus)

/-! ## Programs that the replay does not take -/

/-- `b` appends instructions and one value per output. -/
private def Appends {α : Type} (b : Build α) : Prop :=
  ∀ p : Builder, ∃ new : List Instr, (b.run p).2.instrs = p.instrs ++ new.toArray ∧
    (b.run p).2.values = p.values + (new.map Instr.outputs).sum

/-- `t` keeps the duplex and appends instructions. -/
private def Quiet {α : Type} (t : Transcript α) : Prop :=
  ∀ d, Appends (t.run d) ∧ ∀ p, ((t.run d).run p).1.2 = d

namespace Appends

variable {α β δ γ : Type}

private theorem pure (a : α) : Appends (Pure.pure a : Build α) := fun p =>
  ⟨[], by show p.instrs = p.instrs ++ #[]; simp, by show p.values = p.values + 0; rfl⟩

private theorem bind {b : Build α} {f : α → Build β} (hb : Appends b) (hf : ∀ a, Appends (f a)) : Appends (b >>= f) :=
  fun p => by
    obtain ⟨new₁, hi₁, hv₁⟩ := hb p
    obtain ⟨new₂, hi₂, hv₂⟩ := hf (b.run p).1 (b.run p).2
    refine ⟨new₁ ++ new₂, ?_, ?_⟩
    · show ((f (b.run p).1).run (b.run p).2).2.instrs = _
      rw [hi₂, hi₁]
      simp
    · show ((f (b.run p).1).run (b.run p).2).2.values = _
      rw [hv₂, hv₁]
      simp only [List.map_append, List.sum_append]
      omega

private theorem map {b : Build α} (f : α → β) (hb : Appends b) : Appends (f <$> b) := fun p => hb p

private theorem emit (i : Instr) : Appends (Build.emit i) := fun p =>
  ⟨[i], by show p.instrs.push i = _; simp, by show p.values + i.outputs = _; simp⟩

private theorem forIn_list (l : List δ) (init : γ) {body : δ → γ → Build (ForInStep γ)}
    (h : ∀ x acc, Appends (body x acc)) : Appends (forIn l init body) := by
  induction l generalizing init with
  | nil => exact pure _
  | cons x l ih =>
      rw [List.forIn_cons]
      refine bind (h x init) fun r => ?_
      cases r with
      | done b => exact pure _
      | yield b => exact ih b

private theorem forIn_array (xs : Array δ) (init : γ) {body : δ → γ → Build (ForInStep γ)}
    (h : ∀ x acc, Appends (body x acc)) : Appends (forIn xs init body) := by
  rw [← Array.forIn_toList]
  exact forIn_list _ _ h

private theorem forIn_range (r : Std.Legacy.Range) (init : γ) {body : ℕ → γ → Build (ForInStep γ)}
    (h : ∀ x acc, Appends (body x acc)) : Appends (forIn r init body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact forIn_list _ _ h

private theorem mapM_list {f : δ → Build α} (h : ∀ x, Appends (f x)) : ∀ l : List δ, Appends (l.mapM f)
  | [] => pure _
  | x :: l => by
      rw [List.mapM_cons]
      exact bind (h x) fun _ => bind (mapM_list h l) fun _ => pure _

private theorem mapM_array {f : δ → Build α} (h : ∀ x, Appends (f x)) (xs : Array δ) : Appends (xs.mapM f) := by
  rw [Array.mapM_eq_mapM_toList]
  exact map _ (mapM_list h _)

private theorem const (g : Gl) : Appends (Build.const g) := emit _
private theorem add (a b : Var) : Appends (Build.add a b) := emit _
private theorem sub (a b : Var) : Appends (Build.sub a b) := emit _
private theorem mul (a b : Var) : Appends (Build.mul a b) := emit _
private theorem scale (a : Var) (g : Gl) : Appends (Build.scale a g) := emit _
private theorem extMul (a b : ExtVar) : Appends (Build.extMul a b) := map _ (emit _)
private theorem extInv (a : ExtVar) : Appends (Build.extInv a) := map _ (emit _)
private theorem bits (a : Var) (count : ℕ) : Appends (Build.bits a count) := map _ (emit _)
private theorem hint (count : ℕ) : Appends (Build.hint count) := map _ (emit _)
private theorem permute (s : Fin 16 → Var) : Appends (Build.permute s) := map _ (emit _)
private theorem assertZero (a : Var) : Appends (Build.assertZero a) := bind (emit _) fun _ => pure _

private theorem coordinatewise (f : Var → Var → Build Var) (hf : ∀ a b, Appends (f a b)) (a b : ExtVar) :
    Appends (Build.coordinatewise f a b) :=
  bind (hf _ _) fun _ => bind (hf _ _) fun _ => bind (hf _ _) fun _ => pure _

private theorem extAdd (a b : ExtVar) : Appends (Build.extAdd a b) := coordinatewise _ add a b
private theorem extSub (a b : ExtVar) : Appends (Build.extSub a b) := coordinatewise _ sub a b
private theorem extScale (a : ExtVar) (s : Var) : Appends (Build.extScale a s) := coordinatewise _ (fun _ _ => mul _ _) _ _
private theorem select (b w₀ w₁ : Var) : Appends (Build.select b w₀ w₁) :=
  bind (sub _ _) fun _ => bind (mul _ _) fun _ => add _ _
private theorem assertEqual (a b : Var) : Appends (Build.assertEqual a b) := bind (sub _ _) fun _ => assertZero _
private theorem assertExtEqual (a b : ExtVar) : Appends (Build.assertExtEqual a b) :=
  bind (extSub _ _) fun _ => bind (assertZero _) fun _ => bind (assertZero _) fun _ => assertZero _
private theorem extConstant (e : Ext) : Appends (Algebra.extConstant e) :=
  bind (const _) fun _ => bind (const _) fun _ => bind (const _) fun _ => pure _
private theorem extZero : Appends Algebra.extZero := extConstant _
private theorem extOne : Appends Algebra.extOne := extConstant _

end Appends

namespace Quiet

variable {α β δ γ : Type}

private theorem pure (a : α) : Quiet (Pure.pure a : Transcript α) := fun _ => ⟨Appends.pure _, fun _ => rfl⟩

private theorem bind {t : Transcript α} {f : α → Transcript β} (ht : Quiet t) (hf : ∀ a, Quiet (f a)) : Quiet (t >>= f) :=
  fun d => by
    rw [StateT.run_bind]
    refine ⟨Appends.bind (ht d).1 fun r => (hf r.1 r.2).1, fun p => ?_⟩
    show (((f ((t.run d).run p).1.1).run ((t.run d).run p).1.2).run ((t.run d).run p).2).1.2 = d
    rw [(hf _ _).2, (ht d).2]

private theorem lift {b : Build α} (hb : Appends b) : Quiet (liftM b : Transcript α) := fun d => by
  show Appends ((monadLift b : Transcript α).run d) ∧ _
  rw [StateT.run_monadLift]
  exact ⟨Appends.bind hb fun _ => Appends.pure _, fun _ => rfl⟩

private theorem map {t : Transcript α} (f : α → β) (ht : Quiet t) : Quiet (f <$> t) := by
  rw [map_eq_pure_bind]
  exact bind ht fun _ => pure _

private theorem forIn_list (l : List δ) (init : γ) {body : δ → γ → Transcript (ForInStep γ)}
    (h : ∀ x acc, Quiet (body x acc)) : Quiet (forIn l init body) := by
  induction l generalizing init with
  | nil => exact pure _
  | cons x l ih =>
      rw [List.forIn_cons]
      refine bind (h x init) fun r => ?_
      cases r with
      | done b => exact pure _
      | yield b => exact ih b

private theorem forIn_array (xs : Array δ) (init : γ) {body : δ → γ → Transcript (ForInStep γ)}
    (h : ∀ x acc, Quiet (body x acc)) : Quiet (forIn xs init body) := by
  rw [← Array.forIn_toList]
  exact forIn_list _ _ h

private theorem forIn_range (r : Std.Legacy.Range) (init : γ) {body : ℕ → γ → Transcript (ForInStep γ)}
    (h : ∀ x acc, Quiet (body x acc)) : Quiet (forIn r init body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact forIn_list _ _ h

private theorem mapM_list {f : δ → Transcript α} (h : ∀ x, Quiet (f x)) : ∀ l : List δ, Quiet (l.mapM f)
  | [] => pure _
  | x :: l => by
      rw [List.mapM_cons]
      exact bind (h x) fun _ => bind (mapM_list h l) fun _ => pure _

private theorem mapM_array {f : δ → Transcript α} (h : ∀ x, Quiet (f x)) (xs : Array δ) : Quiet (xs.mapM f) := by
  rw [Array.mapM_eq_mapM_toList]
  exact map _ (mapM_list h _)

end Quiet

/-- One step of `appends`; the WHIR helpers add a rule below. -/
local syntax "appends_step" : tactic

local macro_rules
  | `(tactic| appends_step) => `(tactic| first
    | with_reducible exact Appends.pure _ | with_reducible exact Quiet.pure _
    | with_reducible exact Appends.const _ | with_reducible exact Appends.add _ _
    | with_reducible exact Appends.sub _ _ | with_reducible exact Appends.mul _ _
    | with_reducible exact Appends.scale _ _ | with_reducible exact Appends.extMul _ _
    | with_reducible exact Appends.extInv _ | with_reducible exact Appends.bits _ _
    | with_reducible exact Appends.hint _ | with_reducible exact Appends.permute _
    | with_reducible exact Appends.assertZero _ | with_reducible exact Appends.extAdd _ _
    | with_reducible exact Appends.extSub _ _ | with_reducible exact Appends.extScale _ _
    | with_reducible exact Appends.select _ _ _ | with_reducible exact Appends.assertEqual _ _
    | with_reducible exact Appends.assertExtEqual _ _ | with_reducible exact Appends.extConstant _
    | with_reducible exact Appends.extZero | with_reducible exact Appends.extOne
    | with_reducible apply Appends.bind | with_reducible apply Quiet.bind
    | with_reducible apply Appends.map | with_reducible apply Quiet.lift
    | with_reducible apply Appends.forIn_list | with_reducible apply Appends.forIn_array
    | with_reducible apply Appends.forIn_range | with_reducible apply Quiet.forIn_list
    | with_reducible apply Quiet.forIn_array | with_reducible apply Quiet.forIn_range
    | with_reducible apply Appends.mapM_array | with_reducible apply Quiet.mapM_array
    | split | with_reducible intro)

/-- Prove `Appends` and `Quiet` goals along the builder's combinators. -/
local macro "appends" : tactic => `(tactic| repeat appends_step)

namespace Appends

private theorem expand (x : ExtVar) (k : ℕ) : Appends (Whir.expand x k) := by
  unfold Whir.expand
  appends

private theorem mle (values r : Array ExtVar) : Appends (Whir.mle values r) := by
  unfold Whir.mle
  appends

private theorem horner (values : Array ExtVar) (v : Var) : Appends (Whir.horner values v) := by
  unfold Whir.horner
  appends

private theorem powers (c : Whir.Constraint) : Appends c.powers := by
  unfold Whir.Constraint.powers
  appends

private theorem claimed (c : Whir.Constraint) (powers : Array ExtVar) : Appends (c.claimed powers) := by
  unfold Whir.Constraint.claimed
  appends

private theorem weight (c : Whir.Constraint) (powers r : Array ExtVar) : Appends (c.weight powers r) := by
  unfold Whir.Constraint.weight Whir.evalEq Whir.evalSelect
  appends

private theorem hashLeaf (values : Array Var) : Appends (Circuit.hashLeaf values) := by
  unfold Circuit.hashLeaf
  appends

private theorem merkleRoot (leaf : DigestVar) (bits : Array Var) (path : Array DigestVar) :
    Appends (Circuit.merkleRoot leaf bits path) := by
  unfold Circuit.merkleRoot Circuit.compress
  appends

private theorem powerFromBits (g : Gl) (bits : Array Var) : Appends (Algebra.powerFromBits g bits) := by
  unfold Algebra.powerFromBits
  appends

end Appends

local macro_rules
  | `(tactic| appends_step) => `(tactic| first
    | with_reducible exact Appends.expand _ _ | with_reducible exact Appends.mle _ _
    | with_reducible exact Appends.horner _ _ | with_reducible exact Appends.powers _
    | with_reducible exact Appends.claimed _ _ | with_reducible exact Appends.weight _ _ _
    | with_reducible exact Appends.hashLeaf _ | with_reducible exact Appends.merkleRoot _ _ _
    | with_reducible exact Appends.powerFromBits _ _)

/-! ## Following the replay -/

section Tracks

variable {F : Type} [CommRing F] {c : Ctx F} {n : ℕ} {P : Prop} {α β γ δ ε : Type}

private theorem Appends.ensures {b : Build α} (h : Appends b) : Ensures c n P b fun _ m => n ≤ m := by
  intro p E hp hE
  obtain ⟨new, hi, hv⟩ := h p
  obtain ⟨more, hm, hl⟩ := c.steps_values new E
  refine ⟨new, hi, by rw [hm, List.length_append, hE, hl, hv, hp], fun _ _ _ => ?_⟩
  rw [hv, hp]
  exact Nat.le_add_right _ _

private theorem Quiet.ensures {t : Transcript α} (h : Quiet t) (d : Duplex) :
    Ensures c n P (t.run d) fun r m => r.2 = d ∧ n ≤ m := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := (h d).1.ensures (c := c) (n := n) (P := P) p E hp hE
  exact ⟨new, hi, hl, fun hP hF hh => ⟨(h d).2 p, hs hP hF hh⟩⟩

/-- Fix the witness of an existential precondition. Any `x₀` gives the
instructions, which do not depend on it. -/
private theorem Ensures.elim {ι : Type} {b : Build α} {Q : α → ℕ → Prop} {R : ι → Prop} (x₀ : ι)
    (h : ∀ x, Ensures c n (R x) b Q) (hp : P → ∃ x, R x) : Ensures c n P b Q := by
  intro p E hp' hE
  obtain ⟨new, hi, hl, -⟩ := h x₀ p E hp' hE
  refine ⟨new, hi, hl, fun hP hF hh => ?_⟩
  obtain ⟨x, hx⟩ := hp hP
  obtain ⟨new', hi', -, hs⟩ := h x p E hp' hE
  obtain rfl : new = new' := List.toArray_inj ((Array.append_right_inj _).mp (hi.symm.trans hi'))
  exact hs hx hF hh

/-- **A transcript program tracks a replay action.** From a duplex that is the
replay's, when the checks of `t` hold, its result meets `Q`, and a run of `x`
ends at its duplex. `H` holds the facts about the inputs. -/
private def Tracks (c : Ctx F) (n : ℕ) (H : Prop) (t : Transcript α) (x : ReplayM β) (Q : α → ℕ → Prop) : Prop :=
  ∀ d (tr : Replay.Transcript), Ensures c n (H ∧ IsDuplex c n d tr.duplex) (t.run d)
    fun res m => Q res.1 m ∧ ∃ b tr', Runs x tr b tr' ∧ IsDuplex c m res.2 tr'.duplex

namespace Tracks

variable {H : Prop}

private theorem pure {a : α} {b : β} {Q : α → ℕ → Prop} (h : H → Q a n) : Tracks c n H (Pure.pure a) (Pure.pure b) Q :=
  fun d tr => by
    rw [StateT.run_pure]
    exact Ensures.pure fun hP => ⟨h hP.1, b, tr, Replay.runs_pure _ _, hP.2⟩

/-- A step that the replay takes, then the rest. -/
private theorem bind [Inhabited β] {t : Transcript α} {g : α → Transcript γ} {x : ReplayM β} {y : ReplayM δ}
    {R : γ → ℕ → Prop} (ht : Tracks c n H t x fun _ _ => True) (hg : ∀ a m, n ≤ m → Tracks c m H (g a) y R) :
    Tracks c n H (t >>= g) (x >>= fun _ => y) R := fun d tr => by
  refine Ensures.tbind (ht d tr) fun a e m le => ?_
  refine Ensures.elim ((default : β), tr) (R := fun w : β × Replay.Transcript =>
    (H ∧ IsDuplex c n d tr.duplex) ∧ Runs x tr w.1 w.2 ∧ IsDuplex c m e w.2.duplex) (fun w => ?_) ?_
  · refine ((hg a m le e w.2).pre fun h => ⟨h.1.1, h.2.2⟩).post fun h _ _ q => ⟨q.1, ?_⟩
    obtain ⟨b', tr'', hrun, hd⟩ := q.2
    exact ⟨b', tr'', Replay.runs_bind h.2.1 hrun, hd⟩
  · rintro ⟨hP, -, b, tr', hrun, hd⟩
    exact ⟨(b, tr'), hP, hrun, hd⟩

/-- The replay's last step, then a rest that the replay does not take. -/
private theorem bindLast {t : Transcript α} {g : α → Transcript γ} {x : ReplayM PUnit} {R : γ → ℕ → Prop}
    (ht : Tracks c n H t x fun _ _ => True) (hg : ∀ a m, n ≤ m → Tracks c m H (g a) (Pure.pure ()) R) :
    Tracks c n H (t >>= g) x R := by
  have h := bind ht hg
  rwa [bind_pure_unit] at h

/-- A step that keeps the duplex and that the replay does not take. -/
private theorem quiet {t : Transcript α} {g : α → Transcript γ} {y : ReplayM δ} {R : γ → ℕ → Prop} (ht : Quiet t)
    (hg : ∀ a m, n ≤ m → Tracks c m H (g a) y R) : Tracks c n H (t >>= g) y R := fun d tr =>
  Ensures.tbind (ht.ensures d) fun a e m le =>
    (hg a m le e tr).pre fun h => ⟨h.1.1, by rw [show e = d from h.2.1]; exact h.1.2.mono le⟩

/-- A builder call that the replay does not take. -/
private theorem lift {b : Build α} {g : α → Transcript γ} {y : ReplayM δ} {R : γ → ℕ → Prop} (hb : Appends b)
    (hg : ∀ a m, n ≤ m → Tracks c m H (g a) y R) : Tracks c n H ((liftM b : Transcript α) >>= g) y R :=
  quiet (Quiet.lift hb) hg

/-- A program that keeps the duplex tracks the empty replay. -/
private theorem ofQuiet {t : Transcript α} (h : Quiet t) : Tracks c n H t (Pure.pure ()) fun _ _ => True := fun d tr =>
  (h.ensures d).post fun hP _ _ q => ⟨trivial, (), tr, Replay.runs_pure _ _, by rw [q.1]; exact hP.2.mono q.2⟩

/-- Change the replay action under `H`. -/
private theorem replay {t : Transcript α} {x x' : ReplayM β} {Q : α → ℕ → Prop} (h : Tracks c n H t x Q)
    (same : H → ∀ tr b tr', Runs x tr b tr' → Runs x' tr b tr') : Tracks c n H t x' Q := fun d tr =>
  (h d tr).post fun hP _ _ q => ⟨q.1, by
    obtain ⟨b, tr', hrun, hd⟩ := q.2
    exact ⟨b, tr', same hP.1 _ _ _ hrun, hd⟩⟩

/-- Rewrite the replay action under `H`. -/
private theorem replay_eq {t : Transcript α} {x x' : ReplayM β} {Q : α → ℕ → Prop} (h : Tracks c n H t x Q)
    (e : H → x = x') : Tracks c n H t x' Q :=
  h.replay fun hH _ _ _ run => e hH ▸ run

/-- A loop and the replay's loop over lists of one length under `H`. -/
private theorem loop [Inhabited ε] {xs : List δ} {ys : List ε} {init : γ} {body : δ → γ → Transcript (ForInStep γ)}
    {f : ε → ReplayM PUnit} (len : H → xs.length = ys.length)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m →
      Tracks c m H (body xs[k] acc) (f ys[k]!) fun r _ => r = .yield r.value) :
    Tracks c n H (forIn xs init body) (ys.forM f) fun _ _ => True := fun d tr => by
  refine ((Ensures.loopT (P := H ∧ IsDuplex c n d tr.duplex)
    (fun k _ e m => ∃ tr', Runs ((ys.take k).forM f) tr () tr' ∧ IsDuplex c m e tr'.duplex)
    fun k hk acc e m le => ?_).pre fun h => ⟨h, tr, Replay.runs_pure _ _, h.2⟩).post fun h r m q => ?_
  · refine Ensures.elim tr (R := fun tr' => (H ∧ IsDuplex c n d tr.duplex) ∧
        Runs ((ys.take k).forM f) tr () tr' ∧ IsDuplex c m e tr'.duplex) (fun tr' => ?_)
      fun ⟨hP, tr', hrun, hd⟩ => ⟨tr', hP, hrun, hd⟩
    refine ((step k hk acc m le e tr').pre fun h => ⟨h.1.1, h.2.2⟩).post fun h r m' q => ⟨q.1, ?_⟩
    obtain ⟨b, tr'', hrun, hd⟩ := q.2
    have hk' : k < ys.length := len h.1.1 ▸ hk
    refine ⟨tr'', ?_, hd⟩
    rw [List.take_add_one, List.getElem?_eq_getElem hk', Option.toList_some]
    show Runs (forM (ys.take k ++ [ys[k]]) f) tr () tr''
    rw [List.forM_append, List.forM_cons, List.forM_nil]
    rw [getElem!_pos ys k hk'] at hrun
    exact Replay.runs_bind h.2.1 (Replay.runs_bind hrun (Replay.runs_pure _ _))
  · obtain ⟨tr', hrun, hd⟩ := q
    rw [len h.1, List.take_length] at hrun
    exact ⟨trivial, (), tr', hrun, hd⟩

end Tracks

end Tracks

/-- Pass the builder calls that the replay does not take. -/
local macro "lift_steps" : tactic =>
  `(tactic| repeat (with_reducible refine Tracks.lift (by appends) fun _ _ _ => ?_))

/-! ## The duplex steps of WHIR -/

section Steps

variable {F : Type} [CommRing F] [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P H : Prop} {α β : Type}

private theorem forall₂_getElem! {α β : Type} [Inhabited β] {R : α → β → Prop} {xs : List α} {ys : List β}
    (h : List.Forall₂ R xs ys) {k : ℕ} (hk : k < xs.length) : R xs[k] ys[k]! := by
  have len := h.length_eq
  rw [getElem!_pos ys k (len ▸ hk)]
  simpa using h.get (i := k) hk (len ▸ hk)

private theorem foldl_observeExts (r : Replay.Duplex) (a : Ext) :
    (#[a].flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe r =
      ((r.observe a.c0).observe a.c1).observe a.c2 := by
  rw [← Array.foldl_toList, Array.toList_flatMap]
  simp only [List.flatMap_cons, List.flatMap_nil, List.foldl_cons, List.foldl_nil, List.append_nil]

private theorem foldl_observePair (r : Replay.Duplex) (a b : Ext) :
    (#[a, b].flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe r =
      (#[b].flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe
        ((#[a].flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe r) := by
  rw [foldl_observeExts, foldl_observeExts, ← Array.foldl_toList, Array.toList_flatMap]
  simp only [List.flatMap_cons, List.flatMap_nil, List.foldl_cons, List.foldl_nil, List.append_nil,
    List.cons_append, List.nil_append]

private theorem runs_observeAll_eq {vs : Array Gl} {t t' : Replay.Transcript} {u : Unit}
    (h : Runs (Replay.observeAll vs) t u t') : t' = { t with duplex := vs.foldl Replay.Duplex.observe t.duplex } := by
  cases h
  rfl

/-- Two `observe_ext` calls are the replay's `observeExts` of the pair. -/
private theorem runs_observePair {a b : Ext} {rest : ReplayM β} {tr tr' : Replay.Transcript} {v : β}
    (h : Runs (Replay.observeExt a >>= fun _ => Replay.observeExt b >>= fun _ => rest) tr v tr') :
    Runs (Replay.observeExts #[a, b] >>= fun _ => rest) tr v tr' := by
  obtain ⟨_, t1, h1, h⟩ := Replay.run_bind h
  obtain ⟨_, t2, h2, h3⟩ := Replay.run_bind h
  rw [runs_observeAll_eq h1] at h2
  rw [runs_observeAll_eq h2] at h3
  refine Replay.runs_bind (Replay.runs_observeExts _ _) ?_
  rw [foldl_observePair]
  exact h3

namespace Tracks

/-- `absorb` (`whir/mod.rs:221-226`): the replay's `observeAll`. -/
private theorem absorb (hp : Permutes c.perm) (words : Array Gl) :
    Tracks c n H (Whir.absorb words) (Replay.observeAll words) fun _ _ => True := fun d tr => by
  unfold Whir.absorb
  rw [StateT.run_bind]
  refine Ensures.bind (Q := fun r m => IsDuplex c m r.2 (words.foldl Replay.Duplex.observe tr.duplex)) ?_
    fun r m _ => ?_
  · rw [← Array.forIn_toList]
    refine ((Sim.loopT (P := H ∧ IsDuplex c n d tr.duplex)
      (fun k _ e m => IsDuplex c m e ((words.toList.take k).foldl Replay.Duplex.observe tr.duplex)) (fun _ => True)
      fun k hk _ e m le => ?_).pre fun h => ⟨h, h.2⟩).ensures.post fun _ r m q => ?_
    · refine Sim.tbind₀ (Sim.lift (Sim.const (c := c) _)) fun v e₁ m₁ l₁ => ?_
      refine Sim.tbind₀ (Sim.observe hp (fun h => h.2.2 ▸ h.1.2.mono l₁) fun h => h.2.1) fun _ e₂ m₂ l₂ => ?_
      rw [StateT.run_pure]
      refine Sim.pure fun h => ⟨rfl, ?_⟩
      rw [List.take_add_one, List.getElem?_eq_getElem hk, Option.toList_some, List.foldl_append]
      exact h.2
    · have e := q.1
      rwa [List.take_length, Array.foldl_toList] at e
  · rw [StateT.run_pure]
    exact Ensures.pure fun h => ⟨trivial, (), _, Replay.runs_observeAll words tr, h.2⟩

private theorem sampleExt (hp : Permutes c.perm) : Tracks c n H Duplex.sampleExt Replay.sampleExt fun _ _ => True :=
  fun _ tr => (Sim.sampleExt hp fun h => h.2).ensures.post fun _ _ _ q =>
    ⟨trivial, _, _, Replay.runs_sampleExt tr, q.1.2⟩

private theorem observeExt (hp : Permutes c.perm) {v : ExtVar} {g : Ext} (hv : H → IsExt c n v g) :
    Tracks c n H (Duplex.observeExt v) (Replay.observeExt g) fun _ _ => True := fun _ tr =>
  (Sim.observeExt hp (fun h => h.2) fun h => hv h.1).ensures.post fun _ _ _ q =>
    ⟨trivial, (), _, Replay.runs_observeExts #[g] tr, by rw [foldl_observeExts]; exact q.1⟩

/-- `for value in values { observe_ext }`: the replay's `observeExts`. -/
private theorem observeExts (hp : Permutes c.perm) {vs : Array ExtVar} {gs : Array Ext} (hv : H → IsExts c n vs gs) :
    Tracks c n H (forIn vs PUnit.unit fun value _ => do
        Duplex.observeExt value
        Pure.pure (ForInStep.yield PUnit.unit))
      (Replay.observeExts gs) fun _ _ => True := fun _ tr =>
  (Sim.observeExts hp (fun h => h.2) fun h => hv h.1).ensures.post fun _ _ _ q =>
    ⟨trivial, (), _, Replay.runs_observeExts gs tr, q.1⟩

private theorem observeSlice (hp : Permutes c.perm) {vs : Array Var} {gs : Array Gl} (hv : H → IsWords c n vs gs) :
    Tracks c n H (Duplex.observeSlice vs) (Replay.observeAll gs) fun _ _ => True := fun _ tr =>
  (Sim.observeSlice hp (fun h => h.2) fun h => hv h.1).ensures.post fun _ _ _ q =>
    ⟨trivial, (), _, Replay.runs_observeAll gs tr, q.1⟩

end Tracks

/-- An inverse that passes its check has a nonzero input (`ext_inverse`). -/
private theorem Ensures.extInv_ne {a : ExtVar} {ga : Ext} (ha : P → IsExt c n a ga) :
    Ensures c n P (Build.extInv a) fun _ _ => ga ≠ 0 := by
  have one_ne : (1 : F) ≠ 0 := by
    intro e
    have := (CharP.cast_eq_zero_iff F goldilocksModulus 1).mp (by simpa using e)
    norm_num [goldilocksModulus] at this
  refine ((Sim.emit (c := c) (n := n) (P := P) (Q := fun _ _ => True)
    (C := extProduct (coords (F := F) ga) (fun k : Fin 3 => c.tape (n + k)) = ![1, 0, 0]) (.extInv a)
    fun hP E hl hF => ⟨trivial, ?_⟩).map (f := Build.handles) (Q' := fun _ _ => True)
      fun _ _ _ => trivial).ensures.post fun _ _ _ q => ?_
  · show E.holds ∧ extProduct (fun i => E.value (a i)) (fun k => c.tape (E.values.length + k)) = ![1, 0, 0] ↔ _
    have ra : (fun i => E.value (a i)) = coords ga := funext fun i => (ha hP i).read (Ctx.follows_of_step hF) hl
    rw [ra, hl]
  · intro hz
    have e := congrFun q.2 0
    rw [hz, coords_zero] at e
    simp [extProduct] at e
    exact one_ne e.symm

namespace Tracks

/-- `query_index` (`whir/mod.rs:399-416`): a sample that its inverse check
keeps from `p - 1`, as the replay draws it. -/
private theorem queryIndex (hp : Permutes c.perm) (width : ℕ) :
    Tracks c n H (Whir.queryIndex width) (do
        let word ← Replay.sampleWord
        Replay.ensure (!(word + 1 == 0)) "WHIR query sample p - 1"
        Replay.record (.sample word)) fun _ _ => True := fun d tr => by
  unfold Whir.queryIndex
  refine Ensures.tbind (Sim.sample hp fun h => h.2).ensures fun s e₁ m₁ l₁ => ?_
  refine Ensures.tbind (Ensures.lift (Sim.const (c := c) 1).ensures) fun one e₂ m₂ l₂ => ?_
  refine Ensures.tbind (Ensures.lift (Sim.add (fun h => h.1.2.1.1.mono l₂) fun h => h.2.1.1).ensures)
    fun shifted e₃ m₃ l₃ => ?_
  refine Ensures.tbind (Ensures.lift (Sim.const (c := c) 0).ensures) fun z e₄ m₄ l₄ => ?_
  refine Ensures.tbind (Ensures.lift (Ensures.extInv_ne (ga := ⟨tr.duplex.sample.1 + 1, 0, 0⟩) fun h k => ?_))
    fun _ e₅ m₅ l₅ => ?_
  · fin_cases k
    · exact h.1.2.1.1.mono l₄
    · exact h.2.1.1
    · exact h.2.1.1
  refine (Quiet.ensures (by appends) e₅).post fun h r m q =>
    ⟨trivial, (), { duplex := tr.duplex.sample.2, draws := tr.draws.push (.sample tr.duplex.sample.1) }, ?_, ?_⟩
  · have hne : tr.duplex.sample.1 + 1 ≠ 0 := fun e => h.2.1 (by rw [e]; rfl)
    refine Replay.runs_bind (Replay.runs_sampleWord tr) (Replay.runs_bind (Replay.runs_ensure ?_ _ _)
      (Replay.runs_record _ _))
    rw [Bool.not_eq_true', Bool.eq_false_iff]
    exact fun hb => hne (Replay.Gl.eq_of_beq hb)
  · have e : r.2 = e₁ := q.1.trans (h.2.2.trans (h.1.2.2.trans (h.1.1.2.2.trans h.1.1.1.2.2)))
    rw [e]
    exact h.1.1.1.1.2.1.2.mono (l₂.trans (l₃.trans (l₄.trans (l₅.trans q.2))))

/-- `stir`'s draws (`whir/mod.rs:418-504`): each query index, or none when
the draw count is zero. The rest checks without the duplex. -/
private theorem stir (hp : Permutes c.perm) (queries : Array (Array Var)) (draws indexBits logFoldedDomain : ℕ)
    (root : DigestVar) (folding : Array ExtVar) (base : Bool) :
    Tracks c n H (Whir.stir queries draws indexBits logFoldedDomain root folding base) (Replay.whirQueries draws)
      fun _ _ => True := by
  unfold Whir.stir Replay.whirQueries
  dsimp only
  split
  · rename_i h0
    subst h0
    exact (ofQuiet (by appends)).replay fun _ _ _ _ run => run
  · refine bindLast ?_ fun _ _ _ => ofQuiet (by appends)
    rw [Std.Legacy.Range.forIn_eq_forIn_range']
    refine loop (fun _ => by simp [Std.Legacy.Range.size]) fun k hk acc m le => ?_
    exact bindLast (queryIndex hp _) fun _ _ _ => pure fun _ => rfl

/-- `sumcheck` (`whir/mod.rs:306-332`): the replay's `whirSumcheck` of the
round pairs. -/
private theorem sumcheck (hp : Permutes c.perm) {shape : SumcheckShape} {rounds : Array (ExtVar × ExtVar)}
    {pairs : Array (Ext × Ext)} {claim : ExtVar} (hr : H → List.Forall₂ (IsPair c n) rounds.toList pairs.toList) :
    Tracks c n H (Whir.sumcheck shape rounds claim) (Replay.whirSumcheck shape pairs) fun _ _ => True := by
  have len : H → rounds.size = pairs.size := fun h => by simpa using (hr h).length_eq
  unfold Whir.sumcheck Replay.whirSumcheck
  by_cases e : rounds.isEmpty = true
  · rw [if_pos e]
    refine (pure (b := ()) fun _ => trivial).replay fun h _ _ _ run => ?_
    rw [if_pos (by rw [Array.isEmpty_iff_size_eq_zero] at e ⊢; rw [← len h]; exact e)]
    exact run
  · rw [if_neg e]
    refine replay (x := Replay.observeAll shape.seed >>= fun _ => pairs.toList.forM fun pair => do
        Replay.observeExts #[pair.1, pair.2]
        let _ ← Replay.sampleExt
        Pure.pure ()) ?_ fun h _ _ _ run => ?_
    · refine bind (absorb hp _) fun _ m₁ l₁ => ?_
      refine lift Appends.extOne fun one m₂ l₂ => ?_
      dsimp only
      refine bindLast ?_ fun _ _ _ => pure fun _ => trivial
      rw [← Array.forIn_toList]
      refine loop (fun h => by simpa using (hr h).length_eq) fun k hk acc m le => ?_
      have hpair : H → IsPair c m rounds.toList[k] pairs.toList[k]! := fun h =>
        (forall₂_getElem! (hr h) hk).mono (l₁.trans (l₂.trans le))
      generalize pairs.toList[k]! = g at hpair ⊢
      rcases hx : rounds.toList[k] with ⟨a₀, a₁⟩
      rw [hx] at hpair
      dsimp only
      refine replay (x := Replay.observeExt g.1 >>= fun _ => Replay.observeExt g.2 >>= fun _ =>
          Replay.sampleExt >>= fun _ => Pure.pure ()) ?_ fun _ _ _ _ run => runs_observePair run
      refine bind (observeExt hp fun h => (hpair h).1) fun _ m₃ l₃ => ?_
      refine bind (observeExt hp fun h => (hpair h).2.mono l₃) fun _ m₄ l₄ => ?_
      refine bind (sampleExt hp) fun r m₅ l₅ => ?_
      lift_steps
      exact pure fun _ => rfl
    · rw [if_neg (by rw [Array.isEmpty_iff_size_eq_zero] at e ⊢; rw [← len h]; exact e)]
      exact run

end Tracks

end Steps

/-! ## The verifier -/

section Verify

variable {F : Type} [CommRing F] [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P H : Prop}

private theorem readOpening_evals {shape : WhirShape} {g : Replay.Opening} (h : Replay.readOpening shape g = .ok ()) :
    g.evals.size = shape.batches.size := by
  unfold Replay.readOpening at h
  obtain ⟨_, h1, -⟩ := Replay.Except.bind_ok h
  have h1 := Replay.Except.ensure_ok h1
  simp only [Bool.and_eq_true, beq_iff_eq] at h1
  exact h1.1.1.2

private theorem forall₂_getElem!_array {α β : Type} [Inhabited β] {R : α → β → Prop} {xs : Array α} {ys : Array β}
    (h : List.Forall₂ R xs.toList ys.toList) {k : ℕ} (hk : k < xs.size) : R xs[k] ys[k]! := by
  have := forall₂_getElem! h (k := k) (by simpa using hk)
  rwa [Array.getElem_toList, array_toList_getElem!] at this

omit [CharP F goldilocksModulus] in
private theorem IsExts.getElem! {xs : Array (Array ExtVar)} {gs : Array (Array Ext)}
    (h : List.Forall₂ (IsExts c n) xs.toList gs.toList) (k : ℕ) : IsExts c n xs[k]! gs[k]! := by
  by_cases hk : k < xs.size
  · rw [getElem!_pos xs k hk]
    exact forall₂_getElem!_array h hk
  · have len : xs.size = gs.size := by simpa using h.length_eq
    rw [getElem!_neg xs k hk, getElem!_neg gs k (len ▸ hk)]
    exact IsExts.empty

/-- `whir::verify` (`whir/mod.rs:506-661`) tracks the replay's transcript of
the opening, and returns the opened values. -/
private theorem Tracks.whirVerify (hp : Permutes c.perm) {shape : WhirShape} {key : WhirKey} {root : DigestVar}
    {view : OpeningView} {points : Array (Array ExtVar)} {g : Replay.Opening} {w : Var → Gl}
    (ho : H → IsOpening c w n view g) (hread : H → Replay.readOpening shape g = .ok ())
    (hbatches : key.batches.size = shape.batches.size) :
    Tracks c n H (Circuit.whirVerify shape key root view points) (Replay.whirTranscript shape g)
      fun r m => List.Forall₂ (IsExts c m) r.toList g.evals.toList := by
  unfold Circuit.whirVerify Replay.whirTranscript
  dsimp only
  refine bind ?_ fun _ m₁ l₁ => ?_
  · rw [← Array.forIn_toList]
    refine loop (fun h => by simpa using (ho h).1.1) fun k hk acc m le => ?_
    have ha : H → IsExt c n view.initialOod.toList[k] g.initialOod.toList[k]! := fun h => by
      have := (ho h).1.2 k (by simpa using hk)
      rwa [getElem!_pos view.initialOod k (by simpa using hk), ← Array.getElem_toList,
        ← array_toList_getElem!] at this
    refine bind (absorb hp _) fun _ _ _ => ?_
    refine bind (sampleExt hp) fun _ _ _ => ?_
    lift_steps
    exact bindLast (observeExt hp fun h => (ha h).mono (by omega)) fun _ _ _ => pure fun _ => rfl
  refine bind ?_ fun _ m₂ l₂ => ?_
  · rw [← Array.forIn_toList]
    refine loop (ys := (shape.batches.zip g.evals).toList)
      (fun h => by simp [hbatches, readOpening_evals (hread h)]) fun k hk acc m le => ?_
    have hk' : k < key.batches.size := by simpa using hk
    simp only [Array.getElem_toList, Array.getElem_zipIdx, Nat.zero_add]
    refine replay_eq (x := do Replay.observeAll shape.batches[k]!.seed; Replay.observeExts g.evals[k]!) ?_
      fun h => ?_
    · refine bind (absorb hp _) fun _ _ _ => ?_
      refine bindLast (observeExts hp fun h => ((IsExts.getElem! (ho h).2.1 k).mono (by omega))) fun _ _ _ => ?_
      lift_steps
      refine quiet (by appends) fun _ _ _ => ?_
      exact pure fun _ => rfl
    · have hs : k < shape.batches.size := hbatches ▸ hk'
      have he : k < g.evals.size := (readOpening_evals (hread h)).symm ▸ hs
      rw [array_toList_getElem!, getElem!_pos (shape.batches.zip g.evals) k (by simp [hs, he]),
        Array.getElem_zip, getElem!_pos shape.batches k hs, getElem!_pos g.evals k he]
  refine bind (absorb hp _) fun _ m₃ l₃ => ?_
  refine bind (absorb hp _) fun _ m₄ l₄ => ?_
  refine bind (sampleExt hp) fun _ m₅ l₅ => ?_
  refine quiet (by appends) fun _ m₆ l₆ => ?_
  lift_steps
  refine bind (sumcheck hp fun h => (ho h).2.2.1.imp fun _ _ x => x.mono (by omega)) fun x m₇ l₇ => ?_
  obtain ⟨initial, next⟩ := x
  dsimp only
  refine bind ?_ fun _ m₈ l₈ => ?_
  · rw [← Array.forIn_toList]
    refine loop (ys := (shape.rounds.zip g.rounds).toList) (fun h => by
      have := (ho h).2.2.2.1.length_eq
      simp only [Array.length_toList] at this
      simp [this]) fun k hk acc m le => ?_
    have hk' : k < shape.rounds.size ∧ k < view.rounds.size := by simpa using hk
    have hround : H → IsRound c w n view.rounds[k] g.rounds[k]! := fun h =>
      forall₂_getElem!_array (ho h).2.2.2.1 hk'.2
    simp only [Array.getElem_toList, Array.getElem_zipIdx, Nat.zero_add, Array.getElem_zip]
    refine replay_eq (x := do
        Replay.observeAll g.rounds[k]!.root
        g.rounds[k]!.ood.toList.forM fun answer => do
          let _ ← Replay.sampleExt
          Replay.observeExt answer
        Replay.whirQueries shape.rounds[k].queryDraws
        let _ ← Replay.sampleExt
        Replay.whirSumcheck shape.rounds[k].sumcheck g.rounds[k]!.sumcheck) ?_ fun h => ?_
    · refine bind (observeSlice hp fun h => IsWords.ofFn (hround h).1.1 fun i => ((hround h).1.2 i).mono (by omega))
        fun _ m₉ l₉ => ?_
      refine bind ?_ fun _ m₁₀ l₁₀ => ?_
      · rw [← Array.forIn_toList]
        refine loop (fun h => by simpa using (hround h).2.1.1) fun j hj acc' m' le' => ?_
        have ha : H → IsExt c n view.rounds[k].ood.toList[j] g.rounds[k]!.ood.toList[j]! := fun h => by
          have := (hround h).2.1.2 j (by simpa using hj)
          rwa [getElem!_pos view.rounds[k].ood j (by simpa using hj), ← Array.getElem_toList,
            ← array_toList_getElem!] at this
        refine bind (sampleExt hp) fun _ _ _ => ?_
        lift_steps
        exact bindLast (observeExt hp fun h => (ha h).mono (by omega)) fun _ _ _ => pure fun _ => rfl
      refine bind (stir hp _ _ _ _ _ _ _) fun _ m₁₁ l₁₁ => ?_
      refine bind (sampleExt hp) fun _ m₁₂ l₁₂ => ?_
      lift_steps
      refine bindLast (sumcheck hp fun h => (hround h).2.2.1.imp fun _ _ x => x.mono (by omega)) fun x _ _ => ?_
      obtain ⟨folding, next⟩ := x
      exact pure fun _ => rfl
    · have hg : k < g.rounds.size := by
        have := (ho h).2.2.2.1.length_eq
        simp only [Array.length_toList] at this
        omega
      rw [array_toList_getElem!, getElem!_pos (shape.rounds.zip g.rounds) k (by simp [hk'.1, hg]),
        Array.getElem_zip, getElem!_pos g.rounds k hg]
  refine bind (observeExts hp fun h => (ho h).2.2.2.2.1.mono (by omega)) fun _ m₉ l₉ => ?_
  refine bind (stir hp _ _ _ _ _ _ _) fun _ m₁₀ l₁₀ => ?_
  refine quiet (by appends) fun _ m₁₁ l₁₁ => ?_
  refine bindLast (sumcheck hp fun h => (ho h).2.2.2.2.2.2.imp fun _ _ x => x.mono (by omega)) fun x m₁₂ l₁₂ => ?_
  obtain ⟨finalRandomness, next⟩ := x
  dsimp only
  lift_steps
  refine quiet (by appends) fun _ m₁₃ l₁₃ => ?_
  lift_steps
  exact pure fun h => (ho h).2.1.imp fun _ _ x => x.mono (by omega)

/-- **`whir::verify`** (`whir/mod.rs:506-661`): when its checks hold, the
duplex follows the replay's transcript of the opening `g`, and the opened
values are `g`'s. The key has the shape's batches, and `g` has the shape's
counts (`readOpening`, which `OpeningView::read` gives). -/
theorem whirVerify_ensures (hp : Permutes c.perm) {shape : WhirShape} {key : WhirKey} {root : DigestVar}
    {view : OpeningView} {points : Array (Array ExtVar)} {g : Replay.Opening} {w : Var → Gl} {d : Duplex}
    {t : Replay.Transcript} (hd : P → IsDuplex c n d t.duplex) (ho : P → IsOpening c w n view g)
    (hread : P → Replay.readOpening shape g = .ok ()) (hbatches : key.batches.size = shape.batches.size) :
    Ensures c n P ((whirVerify shape key root view points).run d) fun res m =>
      ∃ t', Runs (Replay.whirTranscript shape g) t () t' ∧ List.Forall₂ (IsExts c m) res.1.toList g.evals.toList ∧
        IsDuplex c m res.2 t'.duplex :=
  ((Tracks.whirVerify hp ho hread hbatches d t).pre fun h => ⟨h, hd h⟩).post fun _ _ _ q => by
    obtain ⟨opened, b, t', run, hd'⟩ := q
    exact ⟨t', run, opened, hd'⟩

end Verify

end CompressionLean.Circuit
