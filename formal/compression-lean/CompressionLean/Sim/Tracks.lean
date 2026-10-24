import CompressionLean.Sim.Hash
import CompressionLean.Bridge.Run

/-!
Owns transcript programs that track the replay. `Tracks c n t x Q`: from a
duplex that is the replay transcript's, when the checks of `t` hold, a run
of the replay action `x` ends at the program's duplex, and `Q` relates the
program's result, the replay's result and the new count.

- `Tracks.bind` steps the program and the replay together, and the rest
  receives both results; `Tracks.bindLast` ends the replay.
- `Tracks.lift` and `Tracks.quiet` pass the program's steps that the replay
  does not take: a builder call, or a transcript step that tracks the
  replay's `pure ()`, as one that keeps the duplex does (`Tracks.ofKeeps`).
- `Tracks.check` and `Tracks.ensure` pass a check that the replay makes and
  the program does not; `Tracks.replay` changes the replay action to one
  that runs alike.
- `Tracks.loop` runs a program loop beside a replay loop of one length.
- `Duplex.X_tracks`: the duplex steps that the replay takes.
-/

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Replay (Word Ext ReplayM)
open CompressionLean.Bridge (Runs)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F] {c : Ctx F} {n : ℕ} {α β γ δ ε : Type}

/-- **A transcript program tracks a replay action.** -/
def Tracks (c : Ctx F) (n : ℕ) (t : Transcript α) (x : ReplayM β) (Q : α → β → ℕ → Prop) : Prop :=
  ∀ d (tr : Replay.Transcript), IsDuplex c n d tr.duplex →
    Ensures c n (t.run d) fun r m => ∃ b tr', Runs x tr b tr' ∧ IsDuplex c m r.2 tr'.duplex ∧ Q r.1 b m

namespace Tracks

theorem pure {a : α} {b : β} {Q : α → β → ℕ → Prop} (h : Q a b n) :
    Tracks c n (Pure.pure a) (Pure.pure b) Q := fun d tr hd => by
  rw [StateT.run_pure]
  exact Ensures.pure ⟨b, tr, Bridge.runs_pure _ _, hd, h⟩

theorem post {t : Transcript α} {x : ReplayM β} {Q Q' : α → β → ℕ → Prop} (h : Tracks c n t x Q)
    (hq : ∀ a b m, Q a b m → Q' a b m) : Tracks c n t x Q' := fun d tr hd =>
  (h d tr hd).post fun _ _ ⟨b, tr', hrun, hd', hb⟩ => ⟨b, tr', hrun, hd', hq _ _ _ hb⟩

/-- **A step that the replay takes, then the rest**, which receives both
results and what `Q` says of them. -/
theorem bind {t : Transcript α} {g : α → Transcript γ} {x : ReplayM β} {y : β → ReplayM δ}
    {Q : α → β → ℕ → Prop} {R : γ → δ → ℕ → Prop} (ht : Tracks c n t x Q)
    (hg : ∀ a b m, n ≤ m → Q a b m → Tracks c m (g a) (y b) R)
    (ht' : ∀ d, Appends (t.run d) := by appends) (hg' : ∀ a d, Appends ((g a).run d) := by appends) :
    Tracks c n (t >>= g) (x >>= y) R := fun d tr hd =>
  (ht d tr hd).tbind (fun a _ m le ⟨b, tr', hrun, hd', hq⟩ => (hg a b m le hq _ tr' hd').post
    fun _ _ ⟨b', tr'', hrun', hd'', hr⟩ => ⟨b', tr'', Bridge.runs_bind hrun hrun', hd'', hr⟩) (ht' d) hg'

/-- The replay's last step `x`, then a rest that passes the replay's result
on. -/
theorem bindLast {t : Transcript α} {g : α → Transcript γ} {x : ReplayM β} {Q : α → β → ℕ → Prop}
    {R : γ → β → ℕ → Prop} (ht : Tracks c n t x Q)
    (hg : ∀ a b m, n ≤ m → Q a b m → Tracks c m (g a) (Pure.pure b) R)
    (ht' : ∀ d, Appends (t.run d) := by appends) (hg' : ∀ a d, Appends ((g a).run d) := by appends) :
    Tracks c n (t >>= g) x R := by
  have h := bind ht hg ht' hg'
  rwa [bind_pure] at h

/-- A builder call that the replay does not take. -/
theorem lift {b : Build α} {g : α → Transcript γ} {y : ReplayM δ} {Q : α → ℕ → Prop} {R : γ → δ → ℕ → Prop}
    (hb : Ensures c n b Q) (hg : ∀ a m, n ≤ m → Q a m → Tracks c m (g a) y R)
    (hb' : Appends b := by appends) (hg' : ∀ a d, Appends ((g a).run d) := by appends) :
    Tracks c n ((liftM b : Transcript α) >>= g) y R := fun d tr hd =>
  Ensures.liftBind hb (fun a m le hq => hg a m le hq d tr hd.mono) hb' hg'

/-- A transcript step that the replay does not take: it tracks the
replay's `pure ()`. -/
theorem quiet {t : Transcript α} {g : α → Transcript γ} {y : ReplayM δ} {Q : α → ℕ → Prop}
    {R : γ → δ → ℕ → Prop} (ht : Tracks c n t (Pure.pure PUnit.unit) fun a _ m => Q a m)
    (hg : ∀ a m, n ≤ m → Q a m → Tracks c m (g a) y R)
    (ht' : ∀ d, Appends (t.run d) := by appends) (hg' : ∀ a d, Appends ((g a).run d) := by appends) :
    Tracks c n (t >>= g) y R := by
  have h := bind ht (fun a _ m le hq => hg a m le hq) ht' hg'
  rwa [pure_bind] at h

/-- A transcript step that keeps the duplex tracks the replay's `pure ()`. -/
theorem ofKeeps {t : Transcript α} (hk : Keeps t := by keeps) (ht : ∀ d, Appends (t.run d) := by appends) :
    Tracks c n t (Pure.pure PUnit.unit) fun _ _ _ => True := fun d tr hd =>
  (Ensures.keeps hk (ht d)).post fun _ _ ⟨le, e⟩ => ⟨(), tr, Bridge.runs_pure _ _, by rw [e]; exact hd.mono le, trivial⟩

/-- Change the replay action to one that runs alike. -/
theorem replay {t : Transcript α} {x x' : ReplayM β} {Q : α → β → ℕ → Prop} (h : Tracks c n t x Q)
    (same : ∀ tr b tr', Runs x tr b tr' → Runs x' tr b tr') : Tracks c n t x' Q := fun d tr hd =>
  (h d tr hd).post fun _ _ ⟨b, tr', hrun, hd', hq⟩ => ⟨b, tr', same _ _ _ hrun, hd', hq⟩

/-- A check that the replay makes and the program does not, with the value
`a` that it returns. -/
theorem check {t : Transcript α} {e : Except String ε} {a : ε} {y : ε → ReplayM δ} {R : α → δ → ℕ → Prop}
    (ok : e = .ok a) (h : Tracks c n t (y a) R) : Tracks c n t (liftExcept e >>= y) R :=
  h.replay fun tr _ _ run => Bridge.runs_bind (Bridge.runs_liftExcept ok tr) run

/-- The replay's `ensure` of a condition that holds. -/
theorem ensure {t : Transcript α} {b : Bool} {what : String} {y : ReplayM δ} {R : α → δ → ℕ → Prop}
    (ok : b = true) (h : Tracks c n t y R) : Tracks c n t (Replay.ensure b what >>= fun _ => y) R :=
  h.replay fun tr _ _ run => Bridge.runs_bind (Bridge.runs_ensure ok what tr) run

/-- **A program loop beside a replay loop of one length.** `I k acc m`: after
`k` steps the accumulator is `acc`, with `m` values. -/
theorem loop [Inhabited ε] {xs : List δ} {ys : List ε} {init : γ} {body : δ → γ → Transcript (ForInStep γ)}
    {f : ε → ReplayM PUnit} (I : ℕ → γ → ℕ → Prop) (len : xs.length = ys.length) (h0 : I 0 init n)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → I k acc m →
      Tracks c m (body xs[k] acc) (f ys[k]!) fun r _ m' => r = .yield r.value ∧ I (k + 1) r.value m')
    (hb : ∀ x acc d, Appends ((body x acc).run d) := by appends) :
    Tracks c n (forIn xs init body) (ys.forM f) fun r _ m => I xs.length r m := fun d tr hd => by
  refine (Ensures.loopT (fun k acc e m => ∃ tr', Runs ((ys.take k).forM f) tr () tr' ∧
      IsDuplex c m e tr'.duplex ∧ I k acc m) ⟨tr, Bridge.runs_pure _ _, hd, h0⟩
    (fun k hk acc e m le hI => ?_) hb).post fun r m q => ?_
  · obtain ⟨tr', hrun, hd', hI⟩ := hI
    refine (step k hk acc m le hI e tr' hd').post fun r m' ⟨_, tr'', hrun', hd'', hy, hI'⟩ =>
      ⟨hy, tr'', ?_, hd'', hI'⟩
    have hk' : k < ys.length := len ▸ hk
    rw [List.take_add_one, List.getElem?_eq_getElem hk', Option.toList_some]
    show Runs (forM (ys.take k ++ [ys[k]]) f) tr () tr''
    rw [List.forM_append, List.forM_cons, List.forM_nil]
    rw [getElem!_pos ys k hk'] at hrun'
    exact Bridge.runs_bind hrun (Bridge.runs_bind hrun' (Bridge.runs_pure _ _))
  · obtain ⟨tr', hrun, hd', hI⟩ := q
    rw [len, List.take_length] at hrun
    exact ⟨(), tr', hrun, hd', hI⟩

end Tracks

/-! ## The duplex steps that the replay takes -/

theorem foldl_observeExt (r : Replay.Duplex) (a : Ext) :
    (#[a].flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe r =
      ((r.observe a.c0).observe a.c1).observe a.c2 := by
  rw [← Array.foldl_toList, Array.toList_flatMap]
  simp only [List.flatMap_cons, List.flatMap_nil, List.foldl_cons, List.foldl_nil, List.append_nil]

section Steps

variable [CharP F goldilocksModulus]

theorem Duplex.observeExt_tracks (hp : Permutes c.perm) {v : ExtVar} {g : Ext} (hv : IsExt c n v g) :
    Tracks c n (Duplex.observeExt v) (Replay.observeExt g) fun _ _ _ => True := fun _ tr hd =>
  (Duplex.observeExt_ensures hp hd hv).post fun _ _ q =>
    ⟨(), _, Bridge.runs_observeExts #[g] tr, by rw [foldl_observeExt]; exact q, trivial⟩

theorem Duplex.observeSlice_tracks (hp : Permutes c.perm) {vs : Array Var} {gs : Array Word}
    (hv : IsWords c n vs gs) :
    Tracks c n (Duplex.observeSlice vs) (Replay.observeAll gs) fun _ _ _ => True := fun _ tr hd =>
  (Duplex.observeSlice_ensures hp hd hv).post fun _ _ q => ⟨(), _, Bridge.runs_observeAll gs tr, q, trivial⟩

/-- `sample_ext`: the program's extension value is the replay's sample. -/
theorem Duplex.sampleExt_tracks (hp : Permutes c.perm) :
    Tracks c n Duplex.sampleExt Replay.sampleExt fun r g m => IsExt c m r g := fun _ tr hd =>
  (Duplex.sampleExt_ensures hp hd).post fun _ _ q => ⟨_, _, Bridge.runs_sampleExt tr, q.2, q.1⟩

end Steps

end CompressionLean.Sim
