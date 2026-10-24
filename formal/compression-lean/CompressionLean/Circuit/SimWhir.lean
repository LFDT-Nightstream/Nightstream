import CompressionLean.Circuit.SimViews
import CompressionLean.Circuit.SimSumcheck

/-!
Owns the simulation of our WHIR verifier (`whirVerify`, `whir/mod.rs:506-661`)
and of the openings (`openingsStage`, `verifier.rs:266-315`) against the
replay. `whirVerify_ensures`: when the checks hold, the duplex follows
`Replay.whirTranscript` and the opened values are the opening's.
`openingsStage_ensures`: the tail of `Replay.verify` runs, both WHIR
transcripts and both `openingsAgree` checks.

- `Appends`, `Quiet`: a builder program appends instructions and one value per
  output; a transcript program also keeps the duplex. The `appends` tactic
  proves both for the arithmetic, the Merkle paths and the STIR checks, which
  the replay does not take.
- `Tracks c n H t x Q`: from a duplex that is the replay's, when the checks of
  `t` hold, a run of `x` ends at its duplex. Its combinators match a program to
  the replay step by step.
- A query index's inverse check keeps the sample from `p - 1`, as the replay's
  `whirQueries` checks.
- The circuit compares opened and expected values pair by pair. The replay
  also compares their sizes, which Rust asserts, so the batch sizes are
  premises.
-/
namespace CompressionLean.Circuit

open CompressionLean.Lowering
open CompressionLean.Replay (Word Ext WhirShape SumcheckShape Runs ReplayM)
open NightstreamFPrime.Spec (goldilocksModulus)

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

private theorem const (g : Word) : Appends (Build.const g) := emit _
private theorem add (a b : Var) : Appends (Build.add a b) := emit _
private theorem sub (a b : Var) : Appends (Build.sub a b) := emit _
private theorem mul (a b : Var) : Appends (Build.mul a b) := emit _
private theorem scale (a : Var) (g : Word) : Appends (Build.scale a g) := emit _
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

private theorem powerFromBits (g : Word) (bits : Array Var) : Appends (Algebra.powerFromBits g bits) := by
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

private theorem pre {t : Transcript α} {x : ReplayM β} {Q : α → ℕ → Prop} {H' : Prop} (h : Tracks c n H' t x Q)
    (hH : H → H') : Tracks c n H t x Q := fun d tr => (h d tr).pre fun hP => ⟨hH hP.1, hP.2⟩

/-- A step that the replay takes, then the rest, which knows `t`'s result by `Q`. -/
private theorem bindQ [Inhabited β] {t : Transcript α} {g : α → Transcript γ} {x : ReplayM β} {y : ReplayM δ}
    {Q : α → ℕ → Prop} {R : γ → ℕ → Prop} (ht : Tracks c n H t x Q)
    (hg : ∀ a m, n ≤ m → Tracks c m (H ∧ Q a m) (g a) y R) : Tracks c n H (t >>= g) (x >>= fun _ => y) R :=
  fun d tr => by
    refine Ensures.tbind (ht d tr) fun a e m le => ?_
    refine Ensures.elim ((default : β), tr) (R := fun w : β × Replay.Transcript =>
      (H ∧ IsDuplex c n d tr.duplex) ∧ Q a m ∧ Runs x tr w.1 w.2 ∧ IsDuplex c m e w.2.duplex) (fun w => ?_) ?_
    · refine ((hg a m le e w.2).pre fun h => ⟨⟨h.1.1, h.2.1⟩, h.2.2.2⟩).post fun h _ _ q => ⟨q.1, ?_⟩
      obtain ⟨b', tr'', hrun, hd⟩ := q.2
      exact ⟨b', tr'', Replay.runs_bind h.2.2.1 hrun, hd⟩
    · rintro ⟨hP, hq, b, tr', hrun, hd⟩
      exact ⟨(b, tr'), hP, hq, hrun, hd⟩

/-- A step that the replay takes, then the rest. -/
private theorem bind [Inhabited β] {t : Transcript α} {g : α → Transcript γ} {x : ReplayM β} {y : ReplayM δ}
    {R : γ → ℕ → Prop} (ht : Tracks c n H t x fun _ _ => True) (hg : ∀ a m, n ≤ m → Tracks c m H (g a) y R) :
    Tracks c n H (t >>= g) (x >>= fun _ => y) R :=
  bindQ ht fun a m le => (hg a m le).pre fun h => h.1

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

/-- A step that keeps the duplex and that the replay does not take; the rest
knows its result by `Q` and its checks by `C`. -/
private theorem keep {t : Transcript α} {g : α → Transcript γ} {y : ReplayM δ} {Q : α → ℕ → Prop} {C : Prop}
    {R : γ → ℕ → Prop} (ht : ∀ d, Simulates c n H (t.run d) (fun r m => Q r.1 m ∧ r.2 = d) C)
    (hg : ∀ a m, n ≤ m → Tracks c m (H ∧ Q a m ∧ C) (g a) y R) : Tracks c n H (t >>= g) y R := fun d tr =>
  Ensures.tbind ((ht d).pre fun h => h.1).ensures fun a e m le =>
    (hg a m le e tr).pre fun h => ⟨⟨h.1.1, h.2.1.1, h.2.2⟩, by rw [show e = d from h.2.1.2]; exact h.1.2.mono le⟩

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

/-- A check that the replay makes and the program does not. -/
private theorem check {t : Transcript α} {e : Except String PUnit} {y : ReplayM δ} {R : α → ℕ → Prop}
    (ok : H → e = .ok ()) (h : Tracks c n H t y R) : Tracks c n H t (liftExcept e >>= fun _ => y) R :=
  h.replay fun hH tr _ _ run => Replay.runs_bind (Replay.runs_liftExcept (ok hH) tr) run

/-- The replay's last check. -/
private theorem checkLast {t : Transcript α} {e : Except String PUnit} {R : α → ℕ → Prop}
    (ok : H → e = .ok ()) (h : Tracks c n H t (Pure.pure ()) R) : Tracks c n H t (liftExcept e) R := by
  have h := check ok h
  rwa [bind_pure_unit] at h

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

private theorem runs_observeAll_eq {vs : Array Word} {t t' : Replay.Transcript} {u : Unit}
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
private theorem absorb (hp : Permutes c.perm) (words : Array Word) :
    Tracks c n H (Whir.absorb words) (Replay.observeAll words) fun _ _ => True := fun d tr => by
  unfold Whir.absorb
  rw [StateT.run_bind]
  refine Ensures.bind (Q := fun r m => IsDuplex c m r.2 (words.foldl Replay.Duplex.observe tr.duplex)) ?_
    fun r m _ => ?_
  · rw [← Array.forIn_toList]
    refine ((Simulates.loopT (P := H ∧ IsDuplex c n d tr.duplex)
      (fun k _ e m => IsDuplex c m e ((words.toList.take k).foldl Replay.Duplex.observe tr.duplex)) (fun _ => True)
      fun k hk _ e m le => ?_).pre fun h => ⟨h, h.2⟩).ensures.post fun _ r m q => ?_
    · refine Simulates.tbind₀ (Simulates.lift (Simulates.const (c := c) _)) fun v e₁ m₁ l₁ => ?_
      refine Simulates.tbind₀ (Simulates.observe hp (fun h => h.2.2 ▸ h.1.2.mono l₁) fun h => h.2.1) fun _ e₂ m₂ l₂ => ?_
      rw [StateT.run_pure]
      refine Simulates.pure fun h => ⟨rfl, ?_⟩
      rw [List.take_add_one, List.getElem?_eq_getElem hk, Option.toList_some, List.foldl_append]
      exact h.2
    · have e := q.1
      rwa [List.take_length, Array.foldl_toList] at e
  · rw [StateT.run_pure]
    exact Ensures.pure fun h => ⟨trivial, (), _, Replay.runs_observeAll words tr, h.2⟩

private theorem sampleExt (hp : Permutes c.perm) : Tracks c n H Duplex.sampleExt Replay.sampleExt fun _ _ => True :=
  fun _ tr => (Simulates.sampleExt hp fun h => h.2).ensures.post fun _ _ _ q =>
    ⟨trivial, _, _, Replay.runs_sampleExt tr, q.1.2⟩

private theorem observeExt (hp : Permutes c.perm) {v : ExtVar} {g : Ext} (hv : H → IsExt c n v g) :
    Tracks c n H (Duplex.observeExt v) (Replay.observeExt g) fun _ _ => True := fun _ tr =>
  (Simulates.observeExt hp (fun h => h.2) fun h => hv h.1).ensures.post fun _ _ _ q =>
    ⟨trivial, (), _, Replay.runs_observeExts #[g] tr, by rw [foldl_observeExts]; exact q.1⟩

/-- `for value in values { observe_ext }`: the replay's `observeExts`. -/
private theorem observeExts (hp : Permutes c.perm) {vs : Array ExtVar} {gs : Array Ext} (hv : H → IsExts c n vs gs) :
    Tracks c n H (forIn vs PUnit.unit fun value _ => do
        Duplex.observeExt value
        Pure.pure (ForInStep.yield PUnit.unit))
      (Replay.observeExts gs) fun _ _ => True := fun _ tr =>
  (Simulates.observeExts hp (fun h => h.2) fun h => hv h.1).ensures.post fun _ _ _ q =>
    ⟨trivial, (), _, Replay.runs_observeExts gs tr, q.1⟩

private theorem observeSlice (hp : Permutes c.perm) {vs : Array Var} {gs : Array Word} (hv : H → IsWords c n vs gs) :
    Tracks c n H (Duplex.observeSlice vs) (Replay.observeAll gs) fun _ _ => True := fun _ tr =>
  (Simulates.observeSlice hp (fun h => h.2) fun h => hv h.1).ensures.post fun _ _ _ q =>
    ⟨trivial, (), _, Replay.runs_observeAll gs tr, q.1⟩

end Tracks

/-- An inverse that passes its check has a nonzero input (`ext_inverse`). -/
private theorem Ensures.extInv_ne {a : ExtVar} {ga : Ext} (ha : P → IsExt c n a ga) :
    Ensures c n P (Build.extInv a) fun _ _ => ga ≠ 0 := by
  have one_ne : (1 : F) ≠ 0 := by
    intro e
    have := (CharP.cast_eq_zero_iff F goldilocksModulus 1).mp (by simpa using e)
    norm_num [goldilocksModulus] at this
  refine ((Simulates.emit (c := c) (n := n) (P := P) (Q := fun _ _ => True)
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
  refine Ensures.tbind (Simulates.sample hp fun h => h.2).ensures fun s e₁ m₁ l₁ => ?_
  refine Ensures.tbind (Ensures.lift (Simulates.const (c := c) 1).ensures) fun one e₂ m₂ l₂ => ?_
  refine Ensures.tbind (Ensures.lift (Simulates.add (fun h => h.1.2.1.1.mono l₂) fun h => h.2.1.1).ensures)
    fun shifted e₃ m₃ l₃ => ?_
  refine Ensures.tbind (Ensures.lift (Simulates.const (c := c) 0).ensures) fun z e₄ m₄ l₄ => ?_
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
    exact fun hb => hne (Replay.Word.eq_of_beq hb)
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

section Verify

variable {F : Type} [CommRing F] [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P H : Prop}

private theorem map_eq_of_zip {α β γ : Type} {f : α → γ} {g : β → γ} :
    ∀ {l₁ : List α} {l₂ : List β}, l₁.length = l₂.length → (∀ p ∈ l₁.zip l₂, f p.1 = g p.2) →
      l₁.map f = l₂.map g
  | [], [], _, _ => rfl
  | [], _ :: _, len, _ => by simp at len
  | _ :: _, [], len, _ => by simp at len
  | a :: l₁, b :: l₂, len, h => by
      rw [List.map_cons, List.map_cons, h (a, b) (by simp),
        map_eq_of_zip (by simpa using len) fun p hp => h p (by simp [hp])]

/-- `readOpening`'s batch sizes are the shape's. -/
private theorem readOpening_sizes {shape : WhirShape} {g : Replay.Opening} (h : Replay.readOpening shape g = .ok ()) :
    g.evals.toList.map (·.size) = shape.batches.toList.map (·.size) := by
  unfold Replay.readOpening at h
  obtain ⟨_, h1, -⟩ := Replay.Except.bind_ok h
  have h1 := Replay.Except.ensure_ok h1
  simp only [Bool.and_eq_true, beq_iff_eq, ← Array.all_toList, List.all_eq_true, Array.toList_zip] at h1
  refine (map_eq_of_zip (by simpa using h1.1.1.2.symm) fun p hp => ?_).symm
  exact (h1.2 p hp).symm

private theorem readOpening_evals {shape : WhirShape} {g : Replay.Opening} (h : Replay.readOpening shape g = .ok ()) :
    g.evals.size = shape.batches.size := by
  simpa using congrArg List.length (readOpening_sizes h)

private theorem forall₂_getElem!_array {α β : Type} [Inhabited β] {R : α → β → Prop} {xs : Array α} {ys : Array β}
    (h : List.Forall₂ R xs.toList ys.toList) {k : ℕ} (hk : k < xs.size) : R xs[k] ys[k]! := by
  have := forall₂_getElem! h (k := k) (by simpa using hk)
  rwa [Array.getElem_toList, Array.getElem!_toList] at this

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
    {view : OpeningView} {points : Array (Array ExtVar)} {g : Replay.Opening} {w : Var → Word}
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
        ← Array.getElem!_toList] at this
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
      rw [Array.getElem!_toList, getElem!_pos (shape.batches.zip g.evals) k (by simp [hs, he]),
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
            ← Array.getElem!_toList] at this
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
      rw [Array.getElem!_toList, getElem!_pos (shape.rounds.zip g.rounds) k (by simp [hk'.1, hg]),
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
    {view : OpeningView} {points : Array (Array ExtVar)} {g : Replay.Opening} {w : Var → Word} {d : Duplex}
    {t : Replay.Transcript} (hd : P → IsDuplex c n d t.duplex) (ho : P → IsOpening c w n view g)
    (hread : P → Replay.readOpening shape g = .ok ()) (hbatches : key.batches.size = shape.batches.size) :
    Ensures c n P ((whirVerify shape key root view points).run d) fun res m =>
      ∃ t', Runs (Replay.whirTranscript shape g) t () t' ∧ List.Forall₂ (IsExts c m) res.1.toList g.evals.toList ∧
        IsDuplex c m res.2 t'.duplex :=
  ((Tracks.whirVerify hp ho hread hbatches d t).pre fun h => ⟨h, hd h⟩).post fun _ _ _ q => by
    obtain ⟨opened, b, t', run, hd'⟩ := q
    exact ⟨t', run, opened, hd'⟩

end Verify

section Openings

variable {F : Type} [CommRing F] {c : Ctx F} {n : ℕ} {P H : Prop}

private theorem isExts_iff {xs : Array ExtVar} {gs : Array Ext} :
    IsExts c n xs gs ↔ List.Forall₂ (IsExt c n) xs.toList gs.toList := by
  rw [List.forall₂_iff_get]
  constructor
  · intro h
    refine ⟨by simp [h.1], fun i h1 h2 => ?_⟩
    have := h.2 i (by simpa using h1)
    rw [getElem!_pos xs i (by simpa using h1), getElem!_pos gs i (by simpa using h2)] at this
    simpa using this
  · intro h
    have size : xs.size = gs.size := by simpa using h.1
    refine ⟨size, fun i hi => ?_⟩
    rw [getElem!_pos xs i hi, getElem!_pos gs i (size ▸ hi)]
    simpa using h.2 i (by simpa using hi) (by simpa [← size] using hi)

private theorem forall₂_zip {xs ys : List ExtVar} {gs hs : List Ext} (hx : List.Forall₂ (IsExt c n) xs gs)
    (hy : List.Forall₂ (IsExt c n) ys hs) : List.Forall₂ (IsPair c n) (xs.zip ys) (gs.zip hs) := by
  induction hx generalizing ys hs with
  | nil => exact .nil
  | cons a _ ih =>
      cases hy with
      | nil => exact .nil
      | cons b rest => exact .cons ⟨a, b⟩ (ih rest)

private theorem isExts_slice {xs : Array ExtVar} {gs : Array Ext} (h : IsExts c n xs gs) (a b : ℕ) :
    IsExts c n (xs.extract a b) (gs.extract a b) := by
  rw [isExts_iff] at h ⊢
  rw [Array.toList_extract, Array.toList_extract, List.extract_eq_take_drop, List.extract_eq_take_drop]
  exact List.forall₂_take _ (List.forall₂_drop _ h)

private theorem isExts_pair {a b : ExtVar} {ga gb : Ext} (ha : IsExt c n a ga) (hb : IsExt c n b gb) :
    IsExts c n #[a, b] #[ga, gb] :=
  (IsExts.empty.push ha).push hb

/-- The coordinate groups of opened batches whose sizes are multiples of
three (`batch.chunks(3)`, `verifier.rs:276`). -/
private theorem groups_rel {opened : Array (Array ExtVar)} {evals : Array (Array Ext)}
    (h : List.Forall₂ (IsExts c n) opened.toList evals.toList) (div : ∀ b ∈ evals.toList, b.size % 3 = 0) :
    List.Forall₂ (IsExts c n)
      (opened.flatMap fun batch => (Array.range ((batch.size + 2) / 3)).map fun i =>
        batch.extract (3 * i) (3 * i + 3)).toList
      (evals.flatMap fun batch => (Array.range (batch.size / 3)).map fun i =>
        batch.extract (3 * i) (3 * i + 3)).toList := by
  rw [Array.toList_flatMap, Array.toList_flatMap]
  generalize opened.toList = xs at h
  generalize evals.toList = ys at h div
  induction h with
  | nil => exact .nil
  | @cons x y xs ys hxy _ ih =>
      rw [List.flatMap_cons, List.flatMap_cons]
      refine List.rel_append ?_ (ih fun b hb => div b (List.mem_cons_of_mem _ hb))
      have hs : (x.size + 2) / 3 = y.size / 3 := by
        have := div y List.mem_cons_self
        rw [hxy.1]
        omega
      rw [hs, Array.toList_map, Array.toList_map, List.forall₂_map_left_iff, List.forall₂_map_right_iff,
        List.forall₂_same]
      exact fun i _ => isExts_slice hxy _ _

/-- `flat` (`verifier.rs:295`) against the replay's `flatPairs`. -/
private theorem flat_rel {pairs : Array (ExtVar × ExtVar)} {gpairs : Array (Ext × Ext)}
    {f : ExtVar × ExtVar → Array ExtVar} (hf : ∀ a b, f (a, b) = #[a, b])
    (h : List.Forall₂ (IsPair c n) pairs.toList gpairs.toList) :
    IsExts c n (pairs.flatMap f) (Replay.flatPairs gpairs) := by
  rw [isExts_iff, Replay.flatPairs, Array.toList_flatMap, Array.toList_flatMap]
  generalize pairs.toList = xs at h
  generalize gpairs.toList = ys at h
  induction h with
  | nil => exact .nil
  | @cons x y xs ys hxy _ ih =>
      obtain ⟨a, b⟩ := x
      rw [List.flatMap_cons, List.flatMap_cons, hf]
      exact List.rel_append (.cons hxy.1 (.cons hxy.2 .nil)) ih

private theorem flatPairs_size (pairs : Array (Ext × Ext)) : (Replay.flatPairs pairs).size = 2 * pairs.size := by
  rw [Replay.flatPairs, Array.size_flatMap, ← Array.sum_toList, Array.toList_map]
  simp [List.sum_replicate, Nat.mul_comm]

private theorem ext_beq_refl (a : Ext) : (a == a) = true := by simp [BEq.beq]

/-- `openingsAgree` passes on batches of the expected sizes and equal values. -/
private theorem agree_of {opened expected : Array (Array Ext)} {what : String}
    (sizes : opened.map (·.size) = expected.map (·.size))
    (same : ∀ p ∈ (opened.flatten.zip expected.flatten).toList, p.1 = p.2) :
    Replay.openingsAgree opened expected what = .ok () := by
  have all : ((opened.flatten.zip expected.flatten).all fun pair => pair.1 == pair.2) = true := by
    rw [← Array.all_toList, List.all_eq_true]
    intro p hp
    rw [same p hp]
    exact ext_beq_refl p.2
  unfold Replay.openingsAgree
  rw [show (opened.map (·.size) == expected.map (·.size)) = true from beq_iff_eq.mpr sizes, all]
  rfl

variable [CharP F goldilocksModulus]

namespace Simulates

/-- `from_coordinates` (`verifier.rs:352-362`). -/
private theorem fromCoordinates {values : Array ExtVar} {gvalues : Array Ext} (hv : P → IsExts c n values gvalues) :
    Simulates c n P (Circuit.fromCoordinates values) (fun r m => IsExt c m r (Replay.fromCoordinates gvalues)) True := by
  unfold Circuit.fromCoordinates Replay.fromCoordinates
  refine extZero.bind₀ fun total m₁ l₁ => ?_
  refine ((loop₀ (P := P ∧ IsExt c m₁ total 0)
    (fun k acc m => IsExt c m acc (Replay.sumRange k fun i => Ext.basis i * gvalues[i]!))
    fun k hk acc m lm => ?_).pre fun h => ⟨h, h.2⟩).bind₀ fun r m _ => Simulates.pure fun h => ?_
  · have hk' : k < values.size := by simpa using hk
    rw [show values.zipIdx[k] = (values[k]!, k) by simp [getElem!_pos, hk']]
    exact (extConstant (Ext.basis k)).bind₀ fun basis m₂ l₂ =>
      (Simulates.extMul (fun h => h.2) fun h => ((hv h.1.1.1).2 k hk').mono (l₁.trans (lm.trans l₂))).bind₀
        fun term m₃ l₃ => (extAdd (fun h => h.1.1.2.mono (l₂.trans l₃)) fun h => h.2).bind₀
        fun total' m₄ l₄ => Simulates.pure fun h => ⟨rfl, by rw [sumRange_succ]; exact h.2⟩
  · have e := h.2
    rw [Array.size_zipIdx, (hv h.1.1).1] at e
    exact e

/-- `combine` (`verifier.rs:334-337`). -/
private theorem combineValue {partials point : Array ExtVar} {gpartials gpoint : Array Ext}
    (hp : P → IsExts c n partials gpartials) (hq : P → IsExts c n point gpoint) :
    Simulates c n P (Circuit.combinePartials partials point) (fun r m => IsExt c m r (Replay.combine gpartials gpoint))
      True :=
  (eqTable hq).bind₀ fun _ _ l => dot (fun h => h.2) fun h => (hp h.1).mono l

omit [CharP F goldilocksModulus] in
/-- `mapM` over a list that `R` relates to `ys`: result `i` meets `S` at `ys[i]`. -/
private theorem listMapM₂ {δ ε α : Type} [Inhabited ε] {f : δ → Build α} {R : δ → ε → ℕ → Prop}
    {S : α → ε → ℕ → Prop} (monoR : ∀ x y m m', R x y m → m ≤ m' → R x y m')
    (monoS : ∀ a y m m', S a y m → m ≤ m' → S a y m')
    (h : ∀ x y m (Q : Prop), (Q → R x y m) → Simulates c m Q (f x) (S · y) True) :
    ∀ (xs : List δ) (ys : List ε) {n : ℕ} {P : Prop}, (P → List.Forall₂ (R · · n) xs ys) →
      Simulates c n P (xs.mapM f) (fun as m => List.Forall₂ (S · · m) as ys) True
  | [], ys, n, P, hr => by
      rw [List.mapM_nil]
      exact Simulates.pure fun hP => by cases hr hP; exact .nil
  | x :: xs, [], n, P, hr => by
      rw [List.mapM_cons]
      refine (h x default n P fun hP => absurd (hr hP) (by simp)).bind₀ fun a m _ => ?_
      exact (listMapM₂ monoR monoS h xs [] (P := P ∧ S a default m) fun hP => absurd (hr hP.1) (by simp)).bind₀
        fun _ _ _ => Simulates.pure fun hP => absurd (hr hP.1.1) (by simp)
  | x :: xs, y :: ys, n, P, hr => by
      rw [List.mapM_cons]
      refine (h x y n P fun hP => (List.forall₂_cons.mp (hr hP)).1).bind₀ fun a m le => ?_
      refine (listMapM₂ monoR monoS h xs ys (P := P ∧ S a y m) fun hP =>
        (List.forall₂_cons.mp (hr hP.1)).2.imp fun _ _ r => monoR _ _ _ _ r le).bind₀ fun as m' le' => ?_
      exact Simulates.pure fun hP => .cons (monoS _ _ _ _ hP.1.2 le') hP.2

/-- The equality checks of `verifier.rs:280-282` and `verifier.rs:312-314`:
once they hold, each pair's values are equal. -/
private theorem assertPairs {xs : Array (ExtVar × ExtVar)} {gs : Array (Ext × Ext)}
    {body : ExtVar × ExtVar → PUnit → Transcript (ForInStep PUnit)}
    (hb : ∀ a b u, body (a, b) u = (do liftM (Build.assertExtEqual a b); Pure.pure (ForInStep.yield PUnit.unit)))
    (hl : P → List.Forall₂ (IsPair c n) xs.toList gs.toList) (d : Duplex) :
    Simulates c n P ((forIn xs PUnit.unit body).run d) (fun r _ => True ∧ r.2 = d) (∀ p ∈ gs.toList, p.1 = p.2) := by
  rw [← Array.forIn_toList]
  generalize xs.toList = l at hl
  generalize gs.toList = gl at hl ⊢
  refine ((loopT (fun _ _ e _ => e = d) (fun k => (gl[k]!).1 = (gl[k]!).2) fun k hk _ e m le => ?_).pre
    fun h => ⟨h, rfl⟩).post (fun _ _ _ h => ⟨trivial, h⟩) fun hP => ?_
  · have hpair : P → IsPair c m l[k] gl[k]! := fun h => (forall₂_getElem! (hl h) hk).mono le
    rcases hx : l[k] with ⟨a, b⟩
    rw [hx] at hpair
    rw [hb]
    refine ((Simulates.tbind (Simulates.lift (assertExtEqual (fun h => (hpair h.1).1) fun h => (hpair h.1).2)) fun _ e' m' _ =>
      Simulates.pure fun h => ⟨rfl, h.2.2.trans h.1.2⟩).post (fun _ _ _ q => q) fun _ => Iff.of_eq (and_true _))
  · have len := (hl hP).length_eq
    constructor
    · intro hall p hp
      obtain ⟨k, hk, rfl⟩ := List.mem_iff_getElem.mp hp
      have := hall k (len ▸ hk)
      rwa [getElem!_pos gl k hk] at this
    · intro hall k hk
      rw [getElem!_pos gl k (len ▸ hk)]
      exact hall _ (List.getElem_mem _)

end Simulates

end Openings

section Stage

variable {F : Type} [CommRing F] [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P H : Prop}

/-- The openings (`verifier.rs:266-315`) track the tail of `Replay.verify`. -/
private theorem Tracks.openingsStage (hp : Permutes c.perm) {rel : Replay.Relation} {p1Key p0Key : WhirKey}
    {o : OpeningsInput} {proof : Replay.Proof} {early : Replay.EarlyOut} {lin : Replay.LinearOut}
    {honest : Replay.HonestOut} {late : Replay.LateOut} {w : Var → Word}
    (h1 : H → IsOpening c w n o.p1Opening proof.p1Opening) (h0 : H → IsOpening c w n o.p0Opening proof.p0Opening)
    (hpartials : H → List.Forall₂ (IsExts c n) o.partials.toList proof.partials.toList)
    (halpha : H → IsExts c n o.alpha honest.alpha) (hqueried : H → IsExts c n o.queried honest.queried)
    (homega : H → IsExt c n o.omegaValue lin.omegaValue) (hlateOmega : H → IsExt c n o.lateOmega late.openings.omega)
    (hzLeaf : H → IsExt c n o.zLeaf early.zLeaf) (hzValue : H → IsExt c n o.zValue lin.zValue)
    (hruns : H → IsExts c n o.earlyValues.runs #[early.values.eRe, early.values.eIm, early.values.row])
    (hpairs : H → List.Forall₂ (IsPair c n) o.earlyValues.pairs.toList early.values.pairs.toList)
    (hblock : H → IsExt c n o.earlyValues.block early.values.block)
    (hlateValues : H → List.Forall₂ (IsPair c n) o.lateValues.toList late.pairs.toList)
    (hlateBlock : H → IsExt c n o.lateBlock late.openings.block) (hmult : H → IsExt c n o.mult early.openings.mult)
    (r1 : H → Replay.readOpening rel.p1Whir proof.p1Opening = .ok ())
    (r0 : H → Replay.readOpening rel.p0Whir proof.p0Opening = .ok ())
    (hqsize : H → honest.queried.size = 2 * rel.queries) (hesize : H → early.values.pairs.size = rel.matrices)
    (hlsize : H → late.pairs.size = rel.matrices)
    (k1 : p1Key.batches.size = rel.p1Whir.batches.size) (k0 : p0Key.batches.size = rel.p0Whir.batches.size)
    (hp1 : ∀ b ∈ rel.p1Whir.batches, b.size % 3 = 0)
    (hp1count : (rel.p1Whir.batches.toList.map fun b => b.size / 3).sum = 2 * rel.queries + 4)
    (hp0 : rel.p0Whir.batches.toList.map (·.size) = [1, 1, 2, 2 * rel.matrices, 1, 2 * rel.matrices + 1, 1]) :
    Tracks c n H (Circuit.openingsStage rel p1Key p0Key o) (do
        Replay.whirTranscript rel.p1Whir proof.p1Opening
        liftExcept (Replay.openingsAgree #[Replay.p1Values proof.p1Opening.evals]
          #[Replay.p1Expected proof honest lin late] "P1 openings")
        Replay.whirTranscript rel.p0Whir proof.p0Opening
        liftExcept (Replay.openingsAgree proof.p0Opening.evals (Replay.p0Expected early lin late) "P0 openings"))
      fun _ _ => True := by
  have div : H → ∀ b ∈ proof.p1Opening.evals.toList, b.size % 3 = 0 := fun h b hb => by
    have hm : b.size ∈ rel.p1Whir.batches.toList.map (·.size) := by
      rw [← readOpening_sizes (r1 h)]
      exact List.mem_map_of_mem hb
    obtain ⟨b', hb', e⟩ := List.mem_map.mp hm
    rw [← e]
    exact hp1 b' (Array.mem_toList_iff.mp hb')
  unfold Circuit.openingsStage
  dsimp only
  refine bindQ (Tracks.whirVerify hp h1 r1 k1) fun opened m₁ l₁ => ?_
  refine keep (fun _ => Simulates.lift (Simulates.combineValue (fun h => (IsExts.getElem! (hpartials h.1) 0).mono l₁)
    fun h => (halpha h.1).mono l₁)) fun keyGroup m₂ l₂ => ?_
  refine keep (fun _ => Simulates.lift (Simulates.combineValue (fun h => (IsExts.getElem! (hpartials h.1.1) 1).mono
    (l₁.trans l₂)) fun h => (halpha h.1.1).mono (l₁.trans l₂))) fun runGroup m₃ l₃ => ?_
  refine keep (Q := fun a m => IsExts c m a (Replay.p1Values proof.p1Opening.evals)) (C := True)
    (fun _ => Simulates.lift (Q := fun a m => IsExts c m a (Replay.p1Values proof.p1Opening.evals)) ?_)
    fun values m₄ l₄ => ?_
  · rw [Array.mapM_eq_mapM_toList]
    refine (Simulates.listMapM₂ (R := fun x y m => IsExts c m x y)
      (S := fun a y m => IsExt c m a (Replay.fromCoordinates y)) (fun _ _ _ _ h le => h.mono le)
      (fun _ _ _ _ h le => h.mono le) (fun _ _ _ _ hq => Simulates.fromCoordinates hq) _
      (proof.p1Opening.evals.flatMap fun batch => (Array.range (batch.size / 3)).map fun i =>
        batch.extract (3 * i) (3 * i + 3)).toList fun h => ?_).map fun as m hf => ?_
    · exact groups_rel (h.1.1.2.imp fun _ _ x => x.mono (l₂.trans l₃)) (div h.1.1.1)
    · have e : Replay.p1Values proof.p1Opening.evals = (proof.p1Opening.evals.flatMap fun batch =>
          (Array.range (batch.size / 3)).map fun i => batch.extract (3 * i) (3 * i + 3)).map
            Replay.fromCoordinates := by
        rw [Array.map_flatMap]
        simp only [Array.map_map]
        rfl
      rw [isExts_iff, List.toList_toArray, e, Array.toList_map, List.forall₂_map_right_iff]
      exact hf
  refine keep (Q := fun _ _ => True) (fun e => Simulates.assertPairs (gs := (Replay.p1Values proof.p1Opening.evals).zip
      (Replay.p1Expected proof honest lin late)) (fun _ _ _ => rfl) (fun h => ?_) e) fun _ m₅ l₅ => ?_
  · rw [Array.toList_zip, Array.toList_zip]
    refine forall₂_zip (isExts_iff.mp h.2.1) (isExts_iff.mp ?_)
    exact ((isExts_pair (h.1.1.2.1.mono (l₃.trans l₄)) (h.1.2.1.mono l₄)).append
      ((hqueried h.1.1.1.1).mono (by omega))).append
      (isExts_pair ((homega h.1.1.1.1).mono (by omega)) ((hlateOmega h.1.1.1.1).mono (by omega)))
  refine check (fun h => ?_) (pre ?_ fun h => h.1.1.1.1.1)
  · refine agree_of ?_ ?_
    · have e1 : (Replay.p1Values proof.p1Opening.evals).size = 2 * rel.queries + 4 := by
        rw [Replay.p1Values, Array.size_flatMap, ← Array.sum_toList, Array.toList_map]
        simp only [Array.size_map, Array.size_range]
        rw [show (proof.p1Opening.evals.toList.map fun a => a.size / 3) =
            (proof.p1Opening.evals.toList.map (·.size)).map (· / 3) by simp [Function.comp_def],
          readOpening_sizes (r1 h.1.1.1.1.1), List.map_map]
        exact hp1count
      have e2 : (Replay.p1Expected proof honest lin late).size = 2 * rel.queries + 4 := by
        simp [Replay.p1Expected, hqsize h.1.1.1.1.1]
        omega
      simp [e1, e2]
    · rw [Array.flatten_singleton, Array.flatten_singleton]
      exact h.2.2
  refine bindQ (Tracks.whirVerify hp (fun h => (h0 h).mono (by omega)) r0 k0) fun opened m₆ l₆ => ?_
  refine keep (Q := fun _ _ => True) (fun e => Simulates.assertPairs (gs := proof.p0Opening.evals.flatten.zip
      (Replay.p0Expected early lin late).flatten) (fun _ _ _ => rfl) (fun h => ?_) e) fun _ m₇ l₇ => ?_
  · have hH := h.1
    rw [Array.toList_zip, Array.toList_zip, Array.toList_flatten, Array.toList_flatten, Array.toList_flatten,
      Array.toList_flatten]
    refine forall₂_zip (List.rel_flatten ?_) (List.rel_flatten ?_) <;>
      rw [List.forall₂_map_left_iff, List.forall₂_map_right_iff]
    · exact h.2.imp fun _ _ x => isExts_iff.mp x
    · refine .cons ?_ (.cons ?_ (.cons ?_ (.cons ?_ (.cons ?_ (.cons ?_ (.cons ?_ .nil)))))) <;>
        refine isExts_iff.mp ?_
      · exact IsExts.empty.push ((hzLeaf hH).mono (by omega))
      · exact IsExts.empty.push ((hzValue hH).mono (by omega))
      · exact isExts_pair (((hruns hH).2 0 (by simp [(hruns hH).1])).mono (by omega))
          (((hruns hH).2 1 (by simp [(hruns hH).1])).mono (by omega))
      · exact flat_rel (fun _ _ => rfl) ((hpairs hH).imp fun _ _ x => x.mono (by omega))
      · exact IsExts.empty.push ((hblock hH).mono (by omega))
      · exact (flat_rel (fun _ _ => rfl) ((hlateValues hH).imp fun _ _ x => x.mono (by omega))).append
          (IsExts.empty.push ((hlateBlock hH).mono (by omega)))
      · exact IsExts.empty.push ((hmult hH).mono (by omega))
  refine checkLast (fun h => ?_) (pure fun _ => trivial)
  refine agree_of ?_ h.2.2
  apply Array.ext'
  rw [Array.toList_map, Array.toList_map, readOpening_sizes (r0 h.1.1), hp0]
  simp [Replay.p0Expected, flatPairs_size, hesize h.1.1, hlsize h.1.1]

/-- **The openings** (`verifier.rs:266-315`): when the checks hold, the
replay's two WHIR transcripts and its two opening checks (`openingsAgree`),
the tail of `Replay.verify`, run from the matching transcript and end at the
circuit's duplex.

The handles hold the replay's values field by field. Both openings pass
`readOpening`, and the honest, early and late outputs have the proof's
sizes. The keys have their shapes' batch counts. The relation's P1 batches
are triples of coordinates, one per expected value, and its P0 batches have
the sizes of the expected values; Rust asserts both (`verifier.rs:279`,
`verifier.rs:308-311`). -/
theorem openingsStage_ensures (hp : Permutes c.perm) {rel : Replay.Relation} {p1Key p0Key : WhirKey}
    {o : OpeningsInput} {proof : Replay.Proof} {early : Replay.EarlyOut} {lin : Replay.LinearOut}
    {honest : Replay.HonestOut} {late : Replay.LateOut} {w : Var → Word} {d : Duplex} {t : Replay.Transcript}
    (hd : P → IsDuplex c n d t.duplex)
    (h1 : P → IsOpening c w n o.p1Opening proof.p1Opening) (h0 : P → IsOpening c w n o.p0Opening proof.p0Opening)
    (hpartials : P → List.Forall₂ (IsExts c n) o.partials.toList proof.partials.toList)
    (halpha : P → IsExts c n o.alpha honest.alpha) (hqueried : P → IsExts c n o.queried honest.queried)
    (homega : P → IsExt c n o.omegaValue lin.omegaValue) (hlateOmega : P → IsExt c n o.lateOmega late.openings.omega)
    (hzLeaf : P → IsExt c n o.zLeaf early.zLeaf) (hzValue : P → IsExt c n o.zValue lin.zValue)
    (hruns : P → IsExts c n o.earlyValues.runs #[early.values.eRe, early.values.eIm, early.values.row])
    (hpairs : P → List.Forall₂ (IsPair c n) o.earlyValues.pairs.toList early.values.pairs.toList)
    (hblock : P → IsExt c n o.earlyValues.block early.values.block)
    (hlateValues : P → List.Forall₂ (IsPair c n) o.lateValues.toList late.pairs.toList)
    (hlateBlock : P → IsExt c n o.lateBlock late.openings.block) (hmult : P → IsExt c n o.mult early.openings.mult)
    (r1 : P → Replay.readOpening rel.p1Whir proof.p1Opening = .ok ())
    (r0 : P → Replay.readOpening rel.p0Whir proof.p0Opening = .ok ())
    (hqsize : P → honest.queried.size = 2 * rel.queries) (hesize : P → early.values.pairs.size = rel.matrices)
    (hlsize : P → late.pairs.size = rel.matrices)
    (k1 : p1Key.batches.size = rel.p1Whir.batches.size) (k0 : p0Key.batches.size = rel.p0Whir.batches.size)
    (hp1 : ∀ b ∈ rel.p1Whir.batches, b.size % 3 = 0)
    (hp1count : (rel.p1Whir.batches.toList.map fun b => b.size / 3).sum = 2 * rel.queries + 4)
    (hp0 : rel.p0Whir.batches.toList.map (·.size) = [1, 1, 2, 2 * rel.matrices, 1, 2 * rel.matrices + 1, 1]) :
    Ensures c n P ((openingsStage rel p1Key p0Key o).run d) fun res m =>
      ∃ t', Runs (do
          Replay.whirTranscript rel.p1Whir proof.p1Opening
          liftExcept (Replay.openingsAgree #[Replay.p1Values proof.p1Opening.evals]
            #[Replay.p1Expected proof honest lin late] "P1 openings")
          Replay.whirTranscript rel.p0Whir proof.p0Opening
          liftExcept (Replay.openingsAgree proof.p0Opening.evals (Replay.p0Expected early lin late) "P0 openings"))
        t () t' ∧ IsDuplex c m res.2 t'.duplex :=
  ((Tracks.openingsStage hp h1 h0 hpartials halpha hqueried homega hlateOmega hzLeaf hzValue hruns hpairs hblock
    hlateValues hlateBlock hmult r1 r0 hqsize hesize hlsize k1 k0 hp1 hp1count hp0 d t).pre
      fun h => ⟨h, hd h⟩).post fun _ _ _ q => by
    obtain ⟨-, b, t', run, hd'⟩ := q
    exact ⟨t', run, hd'⟩

end Stage

end CompressionLean.Circuit
