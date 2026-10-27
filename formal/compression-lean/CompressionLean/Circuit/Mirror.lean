import CompressionLean.Circuit.Grows
import CompressionLean.Final.Fields

/-!
Owns `Mirrors`: a transcript program of the verifier and a replay program
that run in step. From a builder state whose duplex is the replay's, on a
run whose checks hold, the replay succeeds, ends with the builder's duplex,
and its result meets a relation with the program's result.

- `Ext` identities: an `Ext` value denotes an element of `Cubic`
  (`toC`), injectively, and the replay's operations denote the field's, so
  two `Ext` expressions are equal when their denotations are (`toC_inj`).
- `Mirrors.bind` runs two mirrored steps in order; `Mirrors.left` adds a
  builder step that the replay does not make, `Mirrors.right` a replay step
  that changes no duplex (a check, a pure value, a record).
- The duplex steps: observing words and extension values
  (`observeAll`, `observeExts`) and drawing extension values
  (`sampleExt`, `samplePair`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl Ext)
open NightstreamFPrime.Spec (goldilocksModulus)
open Polynomial

/-! ## `Ext` values in `Cubic` -/

/-- `Cubic` with its root `x`, `x³ = x + 1`. -/
@[reducible] noncomputable def cubicField : Bridge.ChallengeField Final.Cubic where
  root := AdjoinRoot.root Final.cubic
  cubic := by
    have root : AdjoinRoot.mk Final.cubic (X ^ 3 - X - 1) = 0 := AdjoinRoot.mk_self
    rw [map_sub, map_sub, map_pow, AdjoinRoot.mk_X, map_one] at root
    linear_combination root
  card := Final.card_cubic

/-- The element of `Cubic` that an `Ext` value denotes. -/
noncomputable def toC (a : Ext) : Final.Cubic := @Replay.Ext.toK Final.Cubic _ _ _ cubicField a

section
attribute [local instance] cubicField

theorem toC_add (a b : Ext) : toC (a + b) = toC a + toC b := Replay.Ext.toK_add a b
theorem toC_sub (a b : Ext) : toC (a - b) = toC a - toC b := Replay.Ext.toK_sub a b
theorem toC_mul (a b : Ext) : toC (a * b) = toC a * toC b := Replay.Ext.toK_mul a b
theorem toC_zero : toC 0 = 0 := Replay.Ext.toK_zero
theorem toC_one : toC 1 = 1 := Replay.Ext.toK_one
theorem toC_ofGl (a : Gl) : toC (Ext.ofGl a) = algebraMap _ _ a.toF := Replay.Ext.toK_ofGl a
theorem toC_scale (a : Ext) (g : Gl) : toC (a.scale g) = toC a * algebraMap _ _ g.toF := Replay.Ext.toK_scale a g
theorem toC_pow (a : Ext) (e : ℕ) : toC (a.pow e) = toC a ^ e := Replay.Ext.toK_pow a e
theorem toC_inv (a : Ext) : toC a.inv = (toC a)⁻¹ := Replay.Ext.toK_inv a
theorem toC_oneMinus (a : Ext) : toC (Replay.oneMinus a) = 1 - toC a := by
  rw [Replay.oneMinus, toC_sub, toC_one]
theorem toC_sumRange (n : ℕ) (f : ℕ → Ext) : toC (Replay.sumRange n f) = ∑ i ∈ Finset.range n, toC (f i) :=
  Replay.toK_sumRange n f
theorem toC_prodRange (n : ℕ) (f : ℕ → Ext) : toC (Replay.prodRange n f) = ∏ i ∈ Finset.range n, toC (f i) :=
  Replay.toK_prodRange n f

/-- Two `Ext` values are equal when they denote one element of `Cubic`. -/
theorem toC_inj {a b : Ext} (h : toC a = toC b) : a = b := by
  have zero : toC (a - b) = 0 := by rw [toC_sub, h, sub_self]
  exact Sim.ext_sub_eq_zero.mp (Replay.Ext.eq_zero_of_toK zero)

end

/-! ## Mirrored runs -/

instance : Nonempty Replay.Transcript := ⟨⟨Replay.Duplex.new, #[]⟩⟩

variable {F : Type} [CommRing F]

/-- **A transcript program mirrors a replay program.** From a builder state
with `n` values whose duplex is the replay's, on a run whose checks hold,
the replay of `x` from `tr` succeeds, ends with the builder's duplex, and
its result meets `R` with the program's. -/
def Mirrors {α β : Type} (c : Ctx F) (n : ℕ) (P : Prop) (t : Transcript α) (x : Replay.ReplayM β)
    (R : α → β → ℕ → Prop) : Prop :=
  ∀ d tr, Accepts c n (P ∧ IsDuplex c n d tr.duplex) (t.run d) fun res m =>
    ∃ b tr', Replay.Runs x tr b tr' ∧ IsDuplex c m res.2 tr'.duplex ∧ R res.1 b m

namespace Mirrors

variable {c : Ctx F} {α β γ δ : Type} {n : ℕ} {P : Prop}

/-- Two mirrored steps in order. -/
theorem bind [Nonempty β] {t : Transcript α} {g : α → Transcript γ} {x : Replay.ReplayM β}
    {f : β → Replay.ReplayM δ} {R : α → β → ℕ → Prop} {S : γ → δ → ℕ → Prop}
    (h : Mirrors c n P t x R)
    (k : ∀ a b m, n ≤ m → Mirrors c m (P ∧ R a b m) (g a) (f b) S) : Mirrors c n P (t >>= g) (x >>= f) S := by
  intro d tr
  refine (h d tr).tbind fun a e m le => ?_
  refine (Accepts.exists_pre (ι := β × Replay.Transcript)
    (A := fun bt => ((P ∧ IsDuplex c n d tr.duplex) ∧ Replay.Runs x tr bt.1 bt.2) ∧ (P ∧ R a bt.1 m) ∧
      IsDuplex c m e bt.2.duplex) fun bt => ?_).pre fun h => ?_
  · refine ((k a bt.1 m le e bt.2).pre fun h => h.2).post fun h _ _ q => ?_
    obtain ⟨b', tr'', run, hd, hr⟩ := q
    exact ⟨b', tr'', Replay.runs_bind h.1.2 run, hd, hr⟩
  · obtain ⟨hP, b, tr', run, hd, hr⟩ := h
    exact ⟨(b, tr'), ⟨hP, run⟩, ⟨hP.1, hr⟩, hd⟩

/-- A builder step that the replay does not make. -/
theorem left {b : Build α} {g : α → Transcript γ} {x : Replay.ReplayM δ} {Q : α → ℕ → Prop}
    {S : γ → δ → ℕ → Prop} (h : Accepts c n P b Q)
    (k : ∀ a m, n ≤ m → Mirrors c m (P ∧ Q a m) (g a) x S) :
    Mirrors c n P ((monadLift b : Transcript α) >>= g) x S := by
  intro d tr
  refine (Accepts.lift (d := d) (h.pre fun h => h.1)).tbind fun a e m le => ?_
  refine ((k a m le e tr).pre fun h => ?_)
  obtain ⟨⟨hP, hd⟩, hq, rfl⟩ := h
  exact ⟨⟨hP, hq⟩, hd.mono le⟩

/-- A replay step that changes no duplex: under `P` it succeeds with a
result that meets `S0`. -/
theorem right [Nonempty β] {t : Transcript γ} {y : Replay.ReplayM β} {f : β → Replay.ReplayM δ}
    {S0 : β → Prop} {S : γ → δ → ℕ → Prop}
    (hy : P → ∀ tr, ∃ b tr', Replay.Runs y tr b tr' ∧ tr'.duplex = tr.duplex ∧ S0 b)
    (k : ∀ b, Mirrors c n (P ∧ S0 b) t (f b) S) : Mirrors c n P t (y >>= f) S := by
  intro d tr
  refine (Accepts.exists_pre (ι := β × Replay.Transcript)
    (A := fun bt => (Replay.Runs y tr bt.1 bt.2 ∧ (P ∧ S0 bt.1)) ∧ IsDuplex c n d bt.2.duplex) fun bt => ?_).pre
      fun h => ?_
  · refine ((k bt.1 d bt.2).pre fun h => ⟨h.1.2, h.2⟩).post fun h _ _ q => ?_
    obtain ⟨b', tr'', run, hd, hr⟩ := q
    exact ⟨b', tr'', Replay.runs_bind h.1.1 run, hd, hr⟩
  · obtain ⟨hP, hd⟩ := h
    obtain ⟨b, tr', run, same, hs⟩ := hy hP tr
    exact ⟨(b, tr'), ⟨run, hP, hs⟩, by rw [same]; exact hd⟩

theorem pure {a : α} {b : β} {R : α → β → ℕ → Prop} (h : P → R a b n) :
    Mirrors c n P (Pure.pure a) (Pure.pure b) R := fun d tr => by
  rw [StateT.run_pure]
  exact Accepts.pure fun h' => ⟨b, tr, Replay.runs_pure b tr, h'.2, h h'.1⟩

theorem post {t : Transcript α} {x : Replay.ReplayM β} {R R' : α → β → ℕ → Prop} (h : Mirrors c n P t x R)
    (hr : P → ∀ a b m, R a b m → R' a b m) : Mirrors c n P t x R' := fun d tr =>
  (h d tr).post fun h a m q => by
    obtain ⟨b, tr', run, hd, hq⟩ := q
    exact ⟨b, tr', run, hd, hr h.1 _ _ _ hq⟩

theorem pre {t : Transcript α} {x : Replay.ReplayM β} {R : α → β → ℕ → Prop} {P' : Prop}
    (h : Mirrors c n P' t x R) (hp : P → P') : Mirrors c n P t x R := fun d tr =>
  (h d tr).pre fun h => ⟨hp h.1, h.2⟩

end Mirrors

/-! ## Duplex steps -/

/-- `sample_ext`: three samples, recorded as one draw. -/
theorem _root_.CompressionLean.Replay.runs_sampleExt (tr : Replay.Transcript) :
    Replay.Runs Replay.sampleExt tr
      ⟨tr.duplex.sample.1, tr.duplex.sample.2.sample.1, tr.duplex.sample.2.sample.2.sample.1⟩
      { duplex := tr.duplex.sample.2.sample.2.sample.2,
        draws := tr.draws.push (.ext ⟨tr.duplex.sample.1, tr.duplex.sample.2.sample.1,
          tr.duplex.sample.2.sample.2.sample.1⟩) } := rfl

namespace Mirrors

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

theorem observeAll (hp : Permutes c.perm) {vs : Array Var} {gs : Array Gl} (hv : P → IsWords c n vs gs) :
    Mirrors c n P (Duplex.observeSlice vs) (Replay.observeAll gs) fun _ _ _ => True := fun _ tr =>
  ((Sim.observeSlice hp (fun h => h.2) fun h => hv h.1).accepts).post fun _ _ _ q =>
    ⟨(), _, Replay.runs_observeAll gs tr, q.1, trivial⟩

/-- `sample_ext`, recorded as one draw. -/
theorem sampleExt (hp : Permutes c.perm) :
    Mirrors c n P Duplex.sampleExt Replay.sampleExt fun a e m => IsExt c m a e := fun _ tr =>
  ((Sim.sampleExt hp (fun h => h.2)).accepts).post fun _ _ _ q => ⟨_, _, Replay.runs_sampleExt tr, q.1.2, q.1.1⟩

end Mirrors

end CompressionLean.Circuit
