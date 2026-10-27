import CompressionLean.Circuit.SimHash
import CompressionLean.Bridge.Run

/-!
Owns the one-way reading of simulated programs (`Accepts`): what a builder
program's result means on a run whose checks hold. `Sim` states the checks
of a run exactly; a proof that the circuit's acceptance gives the replay's
needs only that the checks give the result, so a check can feed a later
precondition.

- `Accepts c n P b Q`: after any builder state with `n` values, `b` appends
  instructions, and their run from any executor state with `n` values has
  one value per handle. If `P` holds, the run follows `val` and its checks
  hold, then `b`'s result meets `Q` at the new count.
- `Sim.accepts`: a simulated program accepts with its checks in the result.
- `Accepts.exec`: for a program that `Accepts` describes from the empty
  builder, with `val` the values of its run, an accepting executor run
  gives the result.
- Replay runs (`Replay.Runs`) compose forward (`runs_bind`), so a
  postcondition can say that the replay of a stage succeeds.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

namespace Ctx

variable (c : Ctx F)

/-- A step's checks include the earlier ones. -/
theorem holds_of_step (E : Exec.State F) (i : Instr) (h : (c.step E i).holds) : E.holds := by
  cases i <;> first | exact h.1 | exact h

theorem holds_of_steps (L : List Instr) (E : Exec.State F) (h : (c.steps L E).holds) : E.holds := by
  induction L generalizing E with
  | nil => exact h
  | cons i L ih => exact c.holds_of_step E i (ih (c.step E i) h)

end Ctx

/-- **A builder program on accepting runs.** After any builder state with
`n` values, `b` appends instructions `new`; their run from any executor
state with `n` values has one value per handle. If `P` holds, the run
follows `val` and its checks hold, then `b`'s result meets `Q` at the new
count. -/
def Accepts {α : Type} (c : Ctx F) (n : ℕ) (P : Prop) (b : Build α) (Q : α → ℕ → Prop) : Prop :=
  ∀ (p : Builder) (E : Exec.State F), p.values = n → E.values.length = n →
    ∃ new : List Instr, (b.run p).2.instrs = p.instrs ++ new.toArray ∧
      (c.steps new E).values.length = (b.run p).2.values ∧
      (P → c.Follows (c.steps new E) → (c.steps new E).holds → Q (b.run p).1 (b.run p).2.values)

/-- A simulated program accepts with its checks in the result. -/
theorem Sim.accepts {c : Ctx F} {α : Type} {n : ℕ} {P : Prop} {b : Build α} {Q : α → ℕ → Prop} {C : Prop}
    (h : Sim c n P b Q C) : Accepts c n P b fun a m => Q a m ∧ C := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF hh => ⟨(hs hP hF).1, ((hs hP hF).2.mp hh).2⟩⟩

namespace Accepts

variable {c : Ctx F} {α β γ δ : Type} {n : ℕ} {P : Prop} {Q : α → ℕ → Prop}

theorem pure {a : α} (h : P → Q a n) : Accepts c n P (Pure.pure a) Q := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := Sim.pure (c := c) h p E hp hE
  exact ⟨new, hi, hl, fun hP hF _ => (hs hP hF).1⟩

theorem bind {b : Build α} {f : α → Build β} {R : β → ℕ → Prop} (hb : Accepts c n P b Q)
    (hf : ∀ a m, n ≤ m → Accepts c m (P ∧ Q a m) (f a) R) : Accepts c n P (b >>= f) R := by
  intro p E hp hE
  obtain ⟨new₁, hi₁, hl₁, hs₁⟩ := hb p E hp hE
  have hle : n ≤ (b.run p).2.values := by
    obtain ⟨more, hm, -⟩ := c.steps_values new₁ E
    rw [← hl₁, hm, List.length_append, hE]
    omega
  obtain ⟨new₂, hi₂, hl₂, hs₂⟩ := hf (b.run p).1 _ hle (b.run p).2 (c.steps new₁ E) rfl hl₁
  refine ⟨new₁ ++ new₂, ?_, ?_, ?_⟩
  · show ((f (b.run p).1).run (b.run p).2).2.instrs = _
    rw [hi₂, hi₁]
    simp
  · rw [Ctx.steps_append]
    exact hl₂
  · intro hP hF hh
    rw [Ctx.steps_append] at hF hh
    exact hs₂ ⟨hP, hs₁ hP (Ctx.follows_of_steps hF) (c.holds_of_steps new₂ _ hh)⟩ hF hh

/-- Weaken the result under `P`. -/
theorem post {b : Build α} {Q' : α → ℕ → Prop} (h : Accepts c n P b Q) (hq : P → ∀ a m, Q a m → Q' a m) :
    Accepts c n P b Q' := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF hh => hq hP _ _ (hs hP hF hh)⟩

/-- Strengthen the precondition. -/
theorem pre {b : Build α} {P' : Prop} (h : Accepts c n P' b Q) (hp : P → P') : Accepts c n P b Q := by
  intro p E hp' hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp' hE
  exact ⟨new, hi, hl, fun hP hF hh => hs (hp hP) hF hh⟩

/-- A precondition that names a witness: one program for every witness. -/
theorem exists_pre {ι : Type} [Nonempty ι] {b : Build α} {A : ι → Prop} (h : ∀ i, Accepts c n (A i) b Q) :
    Accepts c n (∃ i, A i) b Q := by
  intro p E hp hE
  by_cases some : ∃ i, A i
  · obtain ⟨i, hi⟩ := some
    obtain ⟨new, eq, hl, hs⟩ := h i p E hp hE
    exact ⟨new, eq, hl, fun _ hF hh => hs hi hF hh⟩
  · obtain ⟨new, eq, hl, -⟩ := h (Classical.arbitrary ι) p E hp hE
    exact ⟨new, eq, hl, fun hP => absurd hP some⟩

theorem map {b : Build α} {f : α → β} {Q' : β → ℕ → Prop} (h : Accepts c n P b Q)
    (hq : ∀ a m, Q a m → Q' (f a) m) : Accepts c n P (f <$> b) Q' := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF hh => hq _ _ (hs hP hF hh)⟩

/-- `t >>= g` in a transcript. -/
theorem tbind {t : Transcript α} {g : α → Transcript β} {d : Duplex} {R : β × Duplex → ℕ → Prop}
    {Q : α × Duplex → ℕ → Prop} (ht : Accepts c n P (t.run d) Q)
    (hg : ∀ a e m, n ≤ m → Accepts c m (P ∧ Q (a, e) m) ((g a).run e) R) : Accepts c n P ((t >>= g).run d) R := by
  rw [StateT.run_bind]
  exact ht.bind fun r m le => hg r.1 r.2 m le

/-- A builder call in a transcript. -/
theorem lift {b : Build α} {d : Duplex} (h : Accepts c n P b Q) :
    Accepts c n P ((monadLift b : Transcript α).run d) (fun r m => Q r.1 m ∧ r.2 = d) := by
  rw [StateT.run_monadLift]
  exact h.bind fun a m _ => Accepts.pure fun h => ⟨h.2, rfl⟩

end Accepts

/-- **The executor's acceptance gives the result.** If `Accepts` describes
`b` from the empty builder under `P`, `val` is the values of the executor's
run of `b`'s program and the executor accepts it, then the result meets
`Q`. -/
theorem Accepts.exec {c : Ctx F} {α : Type} {P : Prop} {b : Build α} {Q : α → ℕ → Prop}
    (h : Accepts c 0 P b Q) (hP : P) (hval : c.val = (Exec.run c.perm c.stmt c.tape b.program).value)
    (accepts : Exec c.perm b.program c.stmt c.tape) : Q (b.run {}).1 (b.run {}).2.values := by
  obtain ⟨new, hi, -, hs⟩ := h {} Exec.start rfl rfl
  have run : Exec.run c.perm c.stmt c.tape b.program = c.steps new Exec.start := by
    rw [Exec.run, Build.program, hi]
    simp only [Array.empty_append, List.toList_toArray]
    rfl
  exact hs hP (fun v _ => by rw [hval, run]) (by rw [← run]; exact accepts)

end CompressionLean.Circuit

namespace CompressionLean.Replay

/-! ## Replay runs, forward -/

theorem runs_bind {α β : Type} {x : ReplayM α} {f : α → ReplayM β} {t t' t'' : Transcript} {a : α} {b : β}
    (hx : Runs x t a t') (hf : Runs (f a) t' b t'') : Runs (x >>= f) t b t'' := by
  unfold Runs at *
  rw [eval_bind, hx]
  exact hf

theorem runs_pure {α : Type} (a : α) (t : Transcript) : Runs (Pure.pure a) t a t := rfl

theorem runs_map {α β : Type} {f : α → β} {x : ReplayM α} {t t' : Transcript} {a : α} (h : Runs x t a t') :
    Runs (f <$> x) t (f a) t' := by
  unfold Runs at *
  rw [eval_map, h]

theorem runs_observeAll (vs : Array Gl) (t : Transcript) :
    Runs (observeAll vs) t () { t with duplex := vs.foldl Duplex.observe t.duplex } := rfl

theorem runs_ensure {ok : Bool} (what : String) (t : Transcript) (h : ok = true) : Runs (ensure ok what) t () t := by
  subst h
  rfl

theorem runs_liftExcept {α : Type} {e : Except String α} {a : α} (t : Transcript) (h : e = .ok a) :
    Runs (liftExcept e) t a t := by
  subst h
  rfl

theorem runs_sampleWord (t : Transcript) :
    Runs sampleWord t t.duplex.sample.1 { t with duplex := t.duplex.sample.2 } := rfl

/-- Three samples, not recorded. -/
theorem runs_drawExt (t : Transcript) :
    Runs drawExt t ⟨t.duplex.sample.1, t.duplex.sample.2.sample.1, t.duplex.sample.2.sample.2.sample.1⟩
      { t with duplex := t.duplex.sample.2.sample.2.sample.2 } := rfl

end CompressionLean.Replay
