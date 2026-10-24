import CompressionLean.Circuit.Exec
import CompressionLean.Circuit.Hash
import CompressionLean.Bridge.Field
import Mathlib.Data.List.GetD

/-!
Owns the simulation of builder programs (`Sim`): what the executor computes
on the instructions that a program writes, read as replay words.

- `Ctx`: what the executor runs with (the permutation, the statement words,
  the tape), and `val`, the values of the last state of the run.
- Handle `h` is the word `g` among `n` values (`IsWord`) when `h < n` and
  `val h = word g`. `IsExt` and `IsWords` lift it to extension values and
  arrays.
- `Sim c n P b Q C`: after any builder state with `n` values, `b` appends
  instructions, and their run from any executor state with `n` values has
  one value per handle. If `P` holds and the run follows `val`, the result
  of `b` meets `Q` at the new count, and the checks of the run hold iff the
  earlier checks and `C` hold.
- `Sim.exec`: for a program that `Sim` describes from the empty builder,
  with `val` the values of its run, the executor accepts it iff `C`.
- `Ensures c n P b Q`: `Sim` in one direction, for soundness. When the run's
  checks hold, the result meets `Q`, which may use them. `Ensures.exec`
  reads `Q` off an accepting run.
- `EnsuresAt c p P b Q`: `Ensures` from one builder state `p`. Its `bind`
  hands the next program the handles that `b` returns, so a statement about
  a whole program can name the handles of its parts.

`P` and `C` are propositions about `val`. Only the meaning of the run
depends on `P`, never which instructions the program writes, so `bind` can
pass `P` on, and one run at any precondition fixes the instructions
(`Sim.elim`, `Ensures.under`). The field has characteristic `p`, so `word` is a ring map on
words (`word_add`, `word_mul`, `word_sub`) and is injective.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-! ## Words in a field of characteristic `p` -/

section Words

variable [CharP F goldilocksModulus]

theorem word_eq_cast (g : Gl) : (word g : F) = ZMod.castHom (dvd_refl _) F g.toF := by
  simp [word, Gl.toF, Gl.toNat]

theorem word_add (a b : Gl) : (word (a + b) : F) = word a + word b := by
  rw [word_eq_cast, word_eq_cast, word_eq_cast, Gl.toF_add, map_add]

theorem word_mul (a b : Gl) : (word (a * b) : F) = word a * word b := by
  rw [word_eq_cast, word_eq_cast, word_eq_cast, Gl.toF_mul, map_mul]

theorem word_sub (a b : Gl) : (word (a - b) : F) = word a - word b := by
  rw [word_eq_cast, word_eq_cast, word_eq_cast, Gl.toF_sub, map_sub]

theorem word_ofNat (k : ℕ) : (word (Gl.ofNat k) : F) = k := by
  rw [word_eq_cast, Gl.toF_ofNat, map_natCast]

@[simp] theorem word_zero : (word 0 : F) = 0 := by
  rw [word_eq_cast, Gl.toF_zero, map_zero]

@[simp] theorem word_one : (word 1 : F) = 1 := by
  rw [word_eq_cast, Gl.toF_one, map_one]

theorem word_neg (a : Gl) : (word (-a) : F) = -word a := by
  rw [word_eq_cast, word_eq_cast, Gl.toF_neg, map_neg]

theorem word_injective : Function.Injective (word : Gl → F) := fun a b h => by
  rw [word_eq_cast, word_eq_cast] at h
  exact Gl.eq_of_toF (ZMod.castHom_injective F h)

end Words

/-! ## Runs -/

/-- What the executor runs with, and `val`, the values of the last state of
the run. -/
structure Ctx (F : Type) where
  perm : List F → List F
  stmt : List F
  tape : ℕ → F
  val : Var → F

namespace Ctx

variable (c : Ctx F)

/-- One instruction. -/
def step (E : Exec.State F) (i : Instr) : Exec.State F := Exec.step c.perm c.stmt c.tape E i

/-- The state after `L` from `E`. -/
def steps (L : List Instr) (E : Exec.State F) : Exec.State F := L.foldl c.step E

/-- `E` has `val`'s value at each of its handles. -/
def Follows (E : Exec.State F) : Prop := ∀ h < E.values.length, E.value h = c.val h

theorem steps_append (L L' : List Instr) (E : Exec.State F) : c.steps (L ++ L') E = c.steps L' (c.steps L E) :=
  List.foldl_append

/-- An instruction appends its outputs. -/
theorem step_values (E : Exec.State F) (i : Instr) :
    ∃ more : List F, (c.step E i).values = E.values ++ more ∧ more.length = i.outputs := by
  cases i <;> exact ⟨_, rfl, by simp [Instr.outputs]⟩

theorem steps_values (L : List Instr) (E : Exec.State F) :
    ∃ more : List F, (c.steps L E).values = E.values ++ more ∧ more.length = (L.map Instr.outputs).sum := by
  induction L generalizing E with
  | nil => exact ⟨[], by simp [steps]⟩
  | cons i L ih =>
      obtain ⟨m, hm, hl⟩ := c.step_values E i
      obtain ⟨m', hm', hl'⟩ := ih (c.step E i)
      exact ⟨m ++ m', by rw [show c.steps (i :: L) E = c.steps L (c.step E i) from rfl, hm', hm, List.append_assoc],
        by simp [hl, hl']⟩

theorem follows_of_steps {c : Ctx F} {L : List Instr} {E : Exec.State F} (h : c.Follows (c.steps L E)) :
    c.Follows E := by
  obtain ⟨more, hm, -⟩ := c.steps_values L E
  intro v hv
  have := h v (by rw [hm]; simp; omega)
  rwa [Exec.State.value, hm, List.getD_append _ _ _ _ hv] at this

/-- Checks only accumulate: a step that holds started from a state that held. -/
theorem holds_of_step {c : Ctx F} {E : Exec.State F} {i : Instr} (h : (c.step E i).holds) : E.holds := by
  cases i <;> simp only [Ctx.step, Exec.step, Exec.State.next] at h <;> first | exact h | exact h.1

theorem holds_of_steps {c : Ctx F} {L : List Instr} {E : Exec.State F} (h : (c.steps L E).holds) : E.holds := by
  induction L generalizing E with
  | nil => exact h
  | cons i L ih => exact holds_of_step (ih h)

/-- **The statement words.** A program that starts with `k` `public`
instructions (`record.rs:297-303`): the first `k` values of its run are the
first `k` statement words. -/
theorem run_pubs {L : Array Instr} {k : ℕ} {rest : List Instr} (pre : L.toList = List.replicate k .pub ++ rest)
    {j : ℕ} (hj : j < k) : (Exec.run c.perm c.stmt c.tape L).value j = c.stmt.getD j 0 := by
  have pubs : ∀ k, (c.steps (List.replicate k .pub) Exec.start).values = (List.range k).map (c.stmt.getD · 0) ∧
      (c.steps (List.replicate k .pub) Exec.start).reads = k := by
    intro k
    induction k with
    | zero => exact ⟨rfl, rfl⟩
    | succ k ih =>
        rw [List.replicate_succ', c.steps_append]
        show (c.step _ .pub).values = _ ∧ (c.step _ .pub).reads = _
        simp only [Ctx.step, Exec.step, ih.1, ih.2, List.range_succ, List.map_append]
        exact ⟨rfl, trivial⟩
  have run : Exec.run c.perm c.stmt c.tape L = c.steps rest (c.steps (List.replicate k .pub) Exec.start) := by
    rw [← c.steps_append, ← pre]
    rfl
  obtain ⟨more, hm, -⟩ := c.steps_values rest (c.steps (List.replicate k .pub) Exec.start)
  rw [run, Exec.State.value, hm, (pubs k).1, List.getD_append _ _ _ _ (by simpa using hj)]
  simp [hj]

end Ctx

/-- Handle `h` is one of the first `n` values, and `val` gives it the word
`g`. -/
def IsWord (c : Ctx F) (n : ℕ) (h : Var) (g : Gl) : Prop := h < n ∧ c.val h = word g

theorem IsWord.mono {c : Ctx F} {n m : ℕ} {h : Var} {g : Gl} (w : IsWord c n h g) (le : n ≤ m) : IsWord c m h g :=
  ⟨lt_of_lt_of_le w.1 le, w.2⟩

theorem IsWord.congr {c : Ctx F} {n : ℕ} {h : Var} {g g' : Gl} (w : IsWord c n h g) (e : (word g : F) = word g') :
    IsWord c n h g' :=
  ⟨w.1, w.2.trans e⟩

/-- An executor state that follows `val` reads a word's value. -/
theorem IsWord.read {c : Ctx F} {n : ℕ} {h : Var} {g : Gl} (w : IsWord c n h g) {E : Exec.State F}
    (hF : c.Follows E) (hl : E.values.length = n) : E.value h = word g :=
  (hF h (hl ▸ w.1)).trans w.2

/-- **A builder program, simulated.** After any builder state with `n`
values, `b` appends instructions `new`; their run from any executor state
with `n` values has one value per handle. If `P` holds and the run follows
`val`, `b`'s result meets `Q` at the new count, and the run's checks hold
iff the earlier ones and `C` hold. -/
def Sim {α : Type} (c : Ctx F) (n : ℕ) (P : Prop) (b : Build α) (Q : α → ℕ → Prop) (C : Prop) : Prop :=
  ∀ (p : Builder) (E : Exec.State F), p.values = n → E.values.length = n →
    ∃ new : List Instr, (b.run p).2.instrs = p.instrs ++ new.toArray ∧
      (c.steps new E).values.length = (b.run p).2.values ∧
      (P → c.Follows (c.steps new E) →
        Q (b.run p).1 (b.run p).2.values ∧ ((c.steps new E).holds ↔ E.holds ∧ C))

namespace Sim

variable {c : Ctx F} {α β : Type} {n : ℕ} {P : Prop} {Q : α → ℕ → Prop} {C : Prop}

theorem pure {a : α} (h : P → Q a n) : Sim c n P (Pure.pure a) Q True := by
  intro p E hp hE
  refine ⟨[], by show p.instrs = p.instrs ++ #[]; simp, by show E.values.length = p.values; rw [hE, hp],
    fun hP _ => ⟨?_, by simp [Ctx.steps]⟩⟩
  exact hp ▸ h hP

theorem bind {b : Build α} {f : α → Build β} {R : β → ℕ → Prop} {C' : Prop} (hb : Sim c n P b Q C)
    (hf : ∀ a m, n ≤ m → Sim c m (P ∧ Q a m) (f a) R C') : Sim c n P (b >>= f) R (C ∧ C') := by
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
  · intro hP hF
    rw [Ctx.steps_append] at hF ⊢
    obtain ⟨hq, hh₁⟩ := hs₁ hP (Ctx.follows_of_steps hF)
    obtain ⟨hr, hh₂⟩ := hs₂ ⟨hP, hq⟩ hF
    exact ⟨hr, by rw [hh₂, hh₁, and_assoc]⟩

/-- `bind` when `b` checks nothing. -/
theorem bind₀ {b : Build α} {f : α → Build β} {R : β → ℕ → Prop} {C' : Prop} (hb : Sim c n P b Q True)
    (hf : ∀ a m, n ≤ m → Sim c m (P ∧ Q a m) (f a) R C') : Sim c n P (b >>= f) R C' := by
  have h := hb.bind hf
  rwa [true_and] at h

/-- Weaken the result and change the checks, both under `P`. -/
theorem post {b : Build α} {Q' : α → ℕ → Prop} {C' : Prop} (h : Sim c n P b Q C)
    (hq : P → ∀ a m, Q a m → Q' a m) (hc : P → (C ↔ C')) : Sim c n P b Q' C' := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF => ⟨hq hP _ _ (hs hP hF).1, (hs hP hF).2.trans (and_congr_right fun _ => hc hP)⟩⟩

/-- Strengthen the precondition. -/
theorem pre {b : Build α} {P' : Prop} (h : Sim c n P' b Q C) (hp : P → P') : Sim c n P b Q C := by
  intro p E hp' hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp' hE
  exact ⟨new, hi, hl, fun hP hF => hs (hp hP) hF⟩

/-- One instruction. -/
theorem emit (i : Instr) {Q : Var → ℕ → Prop}
    (h : P → ∀ E : Exec.State F, E.values.length = n → c.Follows (c.step E i) →
      Q n (n + i.outputs) ∧ ((c.step E i).holds ↔ E.holds ∧ C)) :
    Sim c n P (Build.emit i) Q C := by
  intro p E hp hE
  refine ⟨[i], by show p.instrs.push i = p.instrs ++ #[i]; simp, ?_, fun hP hF => ?_⟩
  · obtain ⟨more, hm, hl⟩ := c.step_values E i
    show (c.step E i).values.length = p.values + i.outputs
    rw [hm, List.length_append, hl, hE, hp]
  · show Q p.values (p.values + i.outputs) ∧ _
    rw [hp]
    exact h hP E hE hF

/-- `f <$> b`. -/
theorem map {b : Build α} {f : α → β} {Q' : β → ℕ → Prop} (h : Sim c n P b Q C)
    (hq : ∀ a m, Q a m → Q' (f a) m) : Sim c n P (f <$> b) Q' C := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF => ⟨hq _ _ (hs hP hF).1, (hs hP hF).2⟩⟩

/-- Fix the witness of an existential precondition. Any `x₀` gives the
instructions, which do not depend on it. -/
theorem elim {ι : Type} {b : Build α} {R : ι → Prop} (x₀ : ι) (h : ∀ x, Sim c n (R x) b Q C)
    (hp : P → ∃ x, R x) : Sim c n P b Q C := by
  intro p E hp' hE
  obtain ⟨new, hi, hl, -⟩ := h x₀ p E hp' hE
  refine ⟨new, hi, hl, fun hP hF => ?_⟩
  obtain ⟨x, hx⟩ := hp hP
  obtain ⟨new', hi', -, hs⟩ := h x p E hp' hE
  obtain rfl : new = new' := List.toArray_inj ((Array.append_right_inj _).mp (hi.symm.trans hi'))
  exact hs hx hF

end Sim

/-- **The executor accepts what a simulated program writes.** If `Sim`
describes `b` from the empty builder under `P`, and `val` is the values of
the executor's run of `b`'s program, then the result meets `Q` and the
executor accepts the program iff `C`. -/
theorem Sim.exec {c : Ctx F} {α : Type} {P : Prop} {b : Build α} {Q : α → ℕ → Prop} {C : Prop}
    (h : Sim c 0 P b Q C) (hP : P) (hval : c.val = (Exec.run c.perm c.stmt c.tape b.program).value) :
    Q (b.run {}).1 (b.run {}).2.values ∧ (Exec c.perm b.program c.stmt c.tape ↔ C) := by
  obtain ⟨new, hi, -, hs⟩ := h {} Exec.start rfl rfl
  have run : Exec.run c.perm c.stmt c.tape b.program = c.steps new Exec.start := by
    rw [Exec.run, Build.program, hi]
    simp only [Array.empty_append, List.toList_toArray]
    rfl
  obtain ⟨hq, hh⟩ := hs hP (fun v _ => by rw [hval, run])
  refine ⟨hq, ?_⟩
  rw [Exec, run, hh]
  exact ⟨fun h => h.2, fun h => ⟨trivial, h⟩⟩

/-! ## Instructions -/

/-- The coordinates of an extension value. -/
def _root_.CompressionLean.Replay.Ext.coord (e : Ext) : Fin 3 → Gl := ![e.c0, e.c1, e.c2]

/-- The three handles of `a` are the coordinates of `e`. -/
def IsExt (c : Ctx F) (n : ℕ) (a : ExtVar) (e : Ext) : Prop := ∀ k, IsWord c n (a k) (e.coord k)

theorem IsExt.mono {c : Ctx F} {n m : ℕ} {a : ExtVar} {e : Ext} (h : IsExt c n a e) (le : n ≤ m) : IsExt c m a e :=
  fun k => (h k).mono le

/-- Both values of a pair. -/
def IsPair (c : Ctx F) (n : ℕ) (a : ExtVar × ExtVar) (g : Ext × Ext) : Prop := IsExt c n a.1 g.1 ∧ IsExt c n a.2 g.2

theorem IsPair.mono {c : Ctx F} {n m : ℕ} {a : ExtVar × ExtVar} {g : Ext × Ext} (h : IsPair c n a g) (le : n ≤ m) :
    IsPair c m a g :=
  ⟨h.1.mono le, h.2.mono le⟩

/-- `Permutes perm`: on words, `perm` is the replay's Poseidon2 permutation. -/
def Permutes (perm : List F → List F) : Prop :=
  ∀ (s : Array Gl) (k : Fin 16), (perm (List.ofFn fun i : Fin 16 => word s[i]!)).getD k 0 = word (Replay.permute s)[k]!

namespace Ctx

variable {c : Ctx F}

theorem follows_of_step {E : Exec.State F} {i : Instr} (h : c.Follows (c.step E i)) : c.Follows E :=
  follows_of_steps (L := [i]) h

/-- Output `k` of a step that appends `outs`. -/
theorem val_next {E : Exec.State F} {outs : List F} {check : Prop} (hF : c.Follows (E.next outs check)) {k : ℕ}
    (hk : k < outs.length) : c.val (E.values.length + k) = outs.getD k 0 := by
  rw [← hF _ (by simp [Exec.State.next]; omega), Exec.State.value]
  simp only [Exec.State.next]
  rw [List.getD_append_right _ _ _ _ (by omega), Nat.add_sub_cancel_left]

end Ctx

section Instructions

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

theorem coord_mul (a b : Ext) :
    (fun k => (word ((a * b).coord k) : F)) = extProduct (fun i => word (a.coord i)) (fun i => word (b.coord i)) := by
  obtain ⟨a0, a1, a2⟩ := a
  obtain ⟨b0, b1, b2⟩ := b
  funext k
  fin_cases k
  · show (word (a0 * b0 + (a1 * b2 + a2 * b1)) : F) = _
    simp [extProduct, Ext.coord, word_add, word_mul]
    ring
  · show (word (a0 * b1 + a1 * b0 + (a1 * b2 + a2 * b1) + a2 * b2) : F) = _
    simp [extProduct, Ext.coord, word_add, word_mul]
    ring
  · show (word (a0 * b2 + a1 * b1 + a2 * b0 + a2 * b2) : F) = _
    simp [extProduct, Ext.coord, word_add, word_mul]
    ring

theorem Ext.eq_of_coord {a b : Ext} (h : ∀ k, (word (a.coord k) : F) = word (b.coord k)) : a = b := by
  obtain ⟨a0, a1, a2⟩ := a
  obtain ⟨b0, b1, b2⟩ := b
  have e0 : a0 = b0 := word_injective (h 0)
  have e1 : a1 = b1 := word_injective (h 1)
  have e2 : a2 = b2 := word_injective (h 2)
  subst e0 e1 e2
  rfl

@[simp] theorem Ext.coord_one : (1 : Ext).coord = ![1, 0, 0] := rfl

@[simp] theorem Ext.coord_zero : (0 : Ext).coord = ![0, 0, 0] := rfl

omit [CharP F goldilocksModulus] in
theorem gl_sub_eq_zero {a b : Gl} : a - b = 0 ↔ a = b := by
  refine ⟨fun h => Gl.eq_of_toF ?_, fun h => ?_⟩
  · have := congrArg Gl.toF h
    rw [Gl.toF_sub, Gl.toF_zero] at this
    exact sub_eq_zero.mp this
  · subst h
    exact Gl.eq_of_toF (by rw [Gl.toF_sub, sub_self, Gl.toF_zero])

namespace Sim

omit [CharP F goldilocksModulus] in
theorem const (g : Gl) : Sim c n P (Build.const g) (fun h m => IsWord c m h g) True :=
  emit _ fun _ E hl hF => by
    refine ⟨⟨by simp [Instr.outputs], ?_⟩, Iff.rfl⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    rfl

omit [CharP F goldilocksModulus] in
theorem priv : Sim c n P Build.priv (fun h m => h < m ∧ c.val h = c.tape h) True :=
  emit _ fun _ E hl hF => by
    refine ⟨⟨by simp [Instr.outputs], ?_⟩, Iff.rfl⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    rfl

theorem add {a b : Var} {ga gb : Gl} (ha : P → IsWord c n a ga) (hb : P → IsWord c n b gb) :
    Sim c n P (Build.add a b) (fun h m => IsWord c m h (ga + gb)) True :=
  emit _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    refine ⟨⟨by simp [Instr.outputs], ?_⟩, Iff.rfl⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp [(ha hP).read hE hl, (hb hP).read hE hl, word_add]

theorem sub {a b : Var} {ga gb : Gl} (ha : P → IsWord c n a ga) (hb : P → IsWord c n b gb) :
    Sim c n P (Build.sub a b) (fun h m => IsWord c m h (ga - gb)) True :=
  emit _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    refine ⟨⟨by simp [Instr.outputs], ?_⟩, Iff.rfl⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp [(ha hP).read hE hl, (hb hP).read hE hl, word_sub, word_neg]
    ring

theorem scale {a : Var} {ga : Gl} (by_ : Gl) (ha : P → IsWord c n a ga) :
    Sim c n P (Build.scale a by_) (fun h m => IsWord c m h (ga * by_)) True :=
  emit _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    refine ⟨⟨by simp [Instr.outputs], ?_⟩, Iff.rfl⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp [(ha hP).read hE hl, word_mul]

theorem mul {a b : Var} {ga gb : Gl} (ha : P → IsWord c n a ga) (hb : P → IsWord c n b gb) :
    Sim c n P (Build.mul a b) (fun h m => IsWord c m h (ga * gb)) True :=
  emit _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    refine ⟨⟨by simp [Instr.outputs], ?_⟩, Iff.rfl⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp [(ha hP).read hE hl, (hb hP).read hE hl, word_mul]

theorem assertZero {a : Var} {ga : Gl} (ha : P → IsWord c n a ga) :
    Sim c n P (Build.assertZero a) (fun _ _ => True) (ga = 0) := by
  refine ((emit (Q := fun _ _ => True) (C := ga = 0) (.assertZero a) fun hP E hl hF => ⟨trivial, ?_⟩).bind
    fun _ _ _ => pure fun _ => trivial).post (fun _ _ _ h => h) (fun _ => Iff.of_eq (and_true _))
  show E.holds ∧ E.value a = 0 ↔ E.holds ∧ ga = 0
  rw [(ha hP).read (Ctx.follows_of_step hF) hl]
  exact and_congr_right fun _ => ⟨fun h => word_injective (h.trans word_zero.symm), fun h => h ▸ word_zero⟩

theorem assertEqual {a b : Var} {ga gb : Gl} (ha : P → IsWord c n a ga) (hb : P → IsWord c n b gb) :
    Sim c n P (Build.assertEqual a b) (fun _ _ => True) (ga = gb) :=
  ((sub ha hb).bind fun _ _ _ => assertZero fun h => h.2).post (fun _ _ _ h => h)
    (fun _ => by rw [true_and, gl_sub_eq_zero])

omit [CharP F goldilocksModulus] in
/-- `r1cs` (`record.rs:450-460`): the rows hold on the values of the column
handles. -/
theorem r1cs {rows : List (Shrink.R1csRow Gl)} {columns : List Var} (hc : P → ∀ h ∈ columns, h < n) :
    Sim c n P (Build.r1cs rows columns) (fun _ _ => True)
      (∀ r ∈ rows, R1csRow.Holds (fun j => (columns.map c.val).getD j 0) r) := by
  refine ((emit (Q := fun _ _ => True) (C := ∀ r ∈ rows, R1csRow.Holds (fun j => (columns.map c.val).getD j 0) r)
    (.r1cs rows columns) fun hP E hl hF => ⟨trivial, ?_⟩).bind fun _ _ _ => pure fun _ => trivial).post
    (fun _ _ _ h => h) (fun _ => Iff.of_eq (and_true _))
  show E.holds ∧ (∀ r ∈ rows, R1csRow.Holds (fun j => (columns.map E.value).getD j 0) r) ↔ _
  rw [List.map_congr_left fun h mem => Ctx.follows_of_step hF h (hl ▸ hc hP h mem)]

theorem extMul {a b : ExtVar} {ga gb : Ext} (ha : P → IsExt c n a ga) (hb : P → IsExt c n b gb) :
    Sim c n P (Build.extMul a b) (fun r m => IsExt c m r (ga * gb)) True :=
  (emit (Q := fun h m => ∀ k : Fin 3, IsWord c m (h + k) ((ga * gb).coord k)) _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    refine ⟨fun k => ⟨Nat.add_lt_add_left k.isLt n, ?_⟩, Iff.rfl⟩
    rw [← hl, Ctx.val_next hF (by simp)]
    have ra : (fun i => E.value (a i)) = fun i => word (ga.coord i) := funext fun i => (ha hP i).read hE hl
    have rb : (fun i => E.value (b i)) = fun i => word (gb.coord i) := funext fun i => (hb hP i).read hE hl
    simp only [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
    simp [ra, rb, ← coord_mul]).map fun _ _ h k => h k

theorem extInv {a : ExtVar} {ga gi : Ext} (ha : P → IsExt c n a ga)
    (hi : P → ∀ k : Fin 3, c.tape (n + k) = word (gi.coord k)) :
    Sim c n P (Build.extInv a) (fun r m => IsExt c m r gi) (ga * gi = 1) :=
  (emit (Q := fun h m => ∀ k : Fin 3, IsWord c m (h + k) (gi.coord k)) _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    have ra : (fun i => E.value (a i)) = fun i => word (ga.coord i) := funext fun i => (ha hP i).read hE hl
    have ri : (fun k : Fin 3 => c.tape (E.values.length + k)) = fun k => word (gi.coord k) :=
      funext fun k => hl ▸ hi hP k
    refine ⟨fun k => ⟨Nat.add_lt_add_left k.isLt n, ?_⟩, ?_⟩
    · rw [← hl, Ctx.val_next hF (by simp)]
      simp only [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
      rw [dif_pos k.isLt, Option.getD_some]
      exact congrFun ri k
    · show E.holds ∧ extProduct (fun i => E.value (a i)) (fun k => c.tape (E.values.length + k)) = ![1, 0, 0] ↔ _
      rw [ra, ri, ← coord_mul]
      have one : (fun k => (word ((1 : Ext).coord k) : F)) = ![1, 0, 0] := by
        funext k
        fin_cases k <;> simp
      rw [← one]
      exact and_congr_right fun _ => ⟨fun h => Ext.eq_of_coord (F := F) (congrFun h), fun h => by rw [h]⟩).map
    fun _ _ h k => h k

omit [CharP F goldilocksModulus] in
theorem bits {a : Var} {ga : Gl} (count : ℕ) (ha : P → IsWord c n a ga) :
    Sim c n P (Build.bits a count) (fun bs m => ∀ i, bs i = n + i ∧ bs i < m ∧ c.val (bs i) = c.tape (bs i))
      (Bits (word ga) count fun i => c.tape (n + i)) :=
  (emit (Q := fun h m => h = n ∧ ∀ i : Fin count, h + i < m ∧ c.val (h + i) = c.tape (h + i)) _ fun hP E hl hF => by
    refine ⟨⟨rfl, fun i => ⟨by simp [Instr.outputs], ?_⟩⟩, ?_⟩
    · rw [← hl, Ctx.val_next hF (by simp)]
      simp
    · show E.holds ∧ Bits (E.value a) count (fun i => c.tape (E.values.length + i)) ↔ _
      rw [(ha hP).read (Ctx.follows_of_step hF) hl, hl]).map fun _ _ h i => ⟨by rw [← h.1]; rfl, h.2 i⟩

omit [CharP F goldilocksModulus] in
theorem hint (count : ℕ) :
    Sim c n P (Build.hint count) (fun hs m => ∀ i, hs i = n + i ∧ hs i < m ∧ c.val (hs i) = c.tape (hs i)) True :=
  (emit (Q := fun h m => h = n ∧ ∀ i : Fin count, h + i < m ∧ c.val (h + i) = c.tape (h + i)) _ fun _ E hl hF =>
    ⟨⟨rfl, fun i => ⟨by simp [Instr.outputs], by rw [← hl, Ctx.val_next hF (by simp)]; simp⟩⟩, Iff.rfl⟩).map
    fun _ _ h i => ⟨by rw [← h.1]; rfl, h.2 i⟩

omit [CharP F goldilocksModulus] in
theorem permute {state : Fin 16 → Var} {gs : Array Gl} (hp : Permutes c.perm)
    (hs : P → ∀ k, IsWord c n (state k) gs[k]!) :
    Sim c n P (Build.permute state) (fun r m => ∀ k, IsWord c m (r k) (Replay.permute gs)[k]!) True :=
  (emit (Q := fun h m => ∀ k : Fin 16, IsWord c m (h + k) (Replay.permute gs)[k]!) _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    have rs : (fun k => E.value (state k)) = fun k : Fin 16 => word gs[k]! := funext fun k => (hs hP k).read hE hl
    refine ⟨fun k => ⟨Nat.add_lt_add_left k.isLt n, ?_⟩, Iff.rfl⟩
    rw [← hl, Ctx.val_next hF (by simp)]
    simp only [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
    rw [dif_pos k.isLt, Option.getD_some, ← hp gs k, rs, List.getD_eq_getElem?_getD]).map fun _ _ h k => h k

end Sim

end Instructions

/-! ## Loops and transcripts -/

namespace Sim

variable {c : Ctx F} {α β γ δ : Type} {n : ℕ} {P : Prop} {Q : α → ℕ → Prop} {C : Prop}

/-- A loop from step `j` on. -/
theorem loop_from {xs : List δ} {body : δ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop) (D : ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → Sim c m (P ∧ I k acc m) (body xs[k] acc)
      (fun r m' => r = .yield r.value ∧ I (k + 1) r.value m') (D k)) :
    ∀ t j, xs.length - j = t → j ≤ xs.length → ∀ acc m, n ≤ m →
      Sim c m (P ∧ I j acc m) (forIn (xs.drop j) acc body) (fun r m' => I xs.length r m')
        (∀ k, j ≤ k → k < xs.length → D k)
  | 0, j, ht, hj, acc, m, _ => by
      have e : j = xs.length := by omega
      subst e
      rw [List.drop_length, List.forIn_nil]
      exact (Sim.pure fun h => h.2).post (fun _ _ _ h => h) fun _ => ⟨fun _ k lo hi => absurd hi (by omega),
        fun _ => trivial⟩
  | t + 1, j, ht, hj, acc, m, hm => by
      have hj' : j < xs.length := by omega
      rw [List.drop_eq_getElem_cons hj', List.forIn_cons]
      refine (Sim.bind (C' := ∀ k, j + 1 ≤ k → k < xs.length → D k) (step j hj' acc m hm) fun r m' le => ?_).post (fun _ _ _ h => h) fun _ =>
        ⟨fun h k lo hi => (Nat.eq_or_lt_of_le lo).elim (fun e => e ▸ h.1) fun lt => h.2 k lt hi,
          fun h => ⟨h j le_rfl hj', fun k lo hi => h k (by omega) hi⟩⟩
      cases r with
      | done b =>
          exact (Sim.pure (Q := fun _ _ => True) fun _ => trivial).post (fun h => absurd h.2.1 (by simp))
            fun h => absurd h.2.1 (by simp)
      | yield b =>
          exact (loop_from I D step t (j + 1) (by omega) hj' b m' (hm.trans le)).pre fun h => ⟨h.1.1, h.2.2⟩

/-- **A loop over a list.** `I k acc m`: after `k` steps the accumulator is
`acc` with `m` values. Step `k` checks `D k`. -/
theorem loop {xs : List δ} {init : γ} {body : δ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop)
    (D : ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → Sim c m (P ∧ I k acc m) (body xs[k] acc)
      (fun r m' => r = .yield r.value ∧ I (k + 1) r.value m') (D k)) :
    Sim c n (P ∧ I 0 init n) (forIn xs init body) (fun r m => I xs.length r m) (∀ k < xs.length, D k) :=
  (loop_from I D step _ 0 rfl (Nat.zero_le _) init n le_rfl).post (fun _ _ _ h => h)
    fun _ => ⟨fun h k hk => h k (Nat.zero_le _) hk, fun h k _ hk => h k hk⟩

/-- `t >>= g` in a transcript. -/
theorem tbind {t : Transcript α} {g : α → Transcript β} {d : Duplex} {R : β × Duplex → ℕ → Prop} {C' : Prop}
    {Q : α × Duplex → ℕ → Prop} (ht : Sim c n P (t.run d) Q C)
    (hg : ∀ a e m, n ≤ m → Sim c m (P ∧ Q (a, e) m) ((g a).run e) R C') : Sim c n P ((t >>= g).run d) R (C ∧ C') := by
  rw [StateT.run_bind]
  exact ht.bind fun r m le => hg r.1 r.2 m le

/-- `tbind` when `t` checks nothing. -/
theorem tbind₀ {t : Transcript α} {g : α → Transcript β} {d : Duplex} {R : β × Duplex → ℕ → Prop} {C' : Prop}
    {Q : α × Duplex → ℕ → Prop} (ht : Sim c n P (t.run d) Q True)
    (hg : ∀ a e m, n ≤ m → Sim c m (P ∧ Q (a, e) m) ((g a).run e) R C') : Sim c n P ((t >>= g).run d) R C' := by
  have h := ht.tbind hg
  rwa [true_and] at h

/-- A builder call in a transcript. -/
theorem lift {b : Build α} {d : Duplex} (h : Sim c n P b Q C) :
    Sim c n P ((monadLift b : Transcript α).run d) (fun r m => Q r.1 m ∧ r.2 = d) C := by
  rw [StateT.run_monadLift]
  exact (h.bind fun a m _ => Sim.pure fun h => ⟨h.2, rfl⟩).post (fun _ _ _ h => h) fun _ => Iff.of_eq (and_true _)

/-- A loop over a list in a transcript; `I k acc e m` also names the duplex. -/
theorem loopT_from {xs : List δ} {body : δ → γ → Transcript (ForInStep γ)} (I : ℕ → γ → Duplex → ℕ → Prop)
    (D : ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc e m, n ≤ m → Sim c m (P ∧ I k acc e m) ((body xs[k] acc).run e)
      (fun r m' => r.1 = .yield r.1.value ∧ I (k + 1) r.1.value r.2 m') (D k)) :
    ∀ t j, xs.length - j = t → j ≤ xs.length → ∀ acc e m, n ≤ m →
      Sim c m (P ∧ I j acc e m) ((forIn (xs.drop j) acc body).run e) (fun r m' => I xs.length r.1 r.2 m')
        (∀ k, j ≤ k → k < xs.length → D k)
  | 0, j, ht, hj, acc, e, m, _ => by
      have h : j = xs.length := by omega
      subst h
      rw [List.drop_length, List.forIn_nil, StateT.run_pure]
      exact (Sim.pure fun h => h.2).post (fun _ _ _ h => h) fun _ => ⟨fun _ k lo hi => absurd hi (by omega),
        fun _ => trivial⟩
  | t + 1, j, ht, hj, acc, e, m, hm => by
      have hj' : j < xs.length := by omega
      rw [List.drop_eq_getElem_cons hj', List.forIn_cons, StateT.run_bind]
      refine (Sim.bind (C' := ∀ k, j + 1 ≤ k → k < xs.length → D k) (step j hj' acc e m hm) fun r m' le => ?_).post (fun _ _ _ h => h) fun _ =>
        ⟨fun h k lo hi => (Nat.eq_or_lt_of_le lo).elim (fun e => e ▸ h.1) fun lt => h.2 k lt hi,
          fun h => ⟨h j le_rfl hj', fun k lo hi => h k (by omega) hi⟩⟩
      obtain ⟨r, e'⟩ := r
      cases r with
      | done b =>
          show Sim c m' _ (Pure.pure (b, e') : Build (γ × Duplex)) _ _
          exact (Sim.pure (Q := fun _ _ => True) fun _ => trivial).post (fun h => absurd h.2.1 (by simp))
            fun h => absurd h.2.1 (by simp)
      | yield b =>
          exact (loopT_from I D step t (j + 1) (by omega) hj' b e' m' (hm.trans le)).pre fun h => ⟨h.1.1, h.2.2⟩

/-- **A loop over a list in a transcript.** -/
theorem loopT {xs : List δ} {init : γ} {body : δ → γ → Transcript (ForInStep γ)} {d : Duplex}
    (I : ℕ → γ → Duplex → ℕ → Prop) (D : ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc e m, n ≤ m → Sim c m (P ∧ I k acc e m) ((body xs[k] acc).run e)
      (fun r m' => r.1 = .yield r.1.value ∧ I (k + 1) r.1.value r.2 m') (D k)) :
    Sim c n (P ∧ I 0 init d n) ((forIn xs init body).run d) (fun r m => I xs.length r.1 r.2 m)
      (∀ k < xs.length, D k) :=
  (loopT_from I D step _ 0 rfl (Nat.zero_le _) init d n le_rfl).post (fun _ _ _ h => h)
    fun _ => ⟨fun h k hk => h k (Nat.zero_le _) hk, fun h k _ hk => h k hk⟩

/-- `mapM` over a list: result `i` meets `R` at entry `i`, which survives
later values. -/
theorem listMapM {f : δ → Build α} (R : δ → α → ℕ → Prop) (mono : ∀ x a m m', R x a m → m ≤ m' → R x a m') :
    ∀ (xs : List δ) {n : ℕ} {P : Prop}, (∀ x ∈ xs, ∀ m, n ≤ m → Sim c m P (f x) (R x) True) →
      Sim c n P (xs.mapM f) (fun as m => List.Forall₂ (fun x a => R x a m) xs as) True
  | [], n, P, _ => by
      rw [List.mapM_nil]
      exact Sim.pure fun _ => List.Forall₂.nil
  | x :: xs, n, P, h => by
      rw [List.mapM_cons]
      refine (h x List.mem_cons_self n le_rfl).bind₀ fun a m le => ?_
      refine (listMapM R mono xs (P := P ∧ R x a m) fun y hy m' le' =>
        (h y (List.mem_cons_of_mem _ hy) m' (le.trans le')).pre fun h => h.1).bind₀ fun as m' le' => ?_
      exact Sim.pure fun hh => List.Forall₂.cons (mono _ _ _ _ hh.1.2 le') hh.2

/-- `mapM` over an array. -/
theorem mapM {n : ℕ} {P : Prop} {xs : Array δ} {f : δ → Build α} (R : δ → α → ℕ → Prop)
    (mono : ∀ x a m m', R x a m → m ≤ m' → R x a m') (h : ∀ x ∈ xs, ∀ m, n ≤ m → Sim c m P (f x) (R x) True) :
    Sim c n P (xs.mapM f) (fun as m => List.Forall₂ (fun x a => R x a m) xs.toList as.toList) True := by
  rw [Array.mapM_eq_mapM_toList]
  exact (listMapM R mono xs.toList fun x hx => h x (Array.mem_toList_iff.mp hx)).map fun as m h => by simpa using h

end Sim

/-! ## Soundness only -/

/-- **A builder program, simulated for soundness.** `Sim` in one direction:
when the run's checks hold, `b`'s result meets `Q`, and `Q` may use those
checks. `ProgramEncodes` needs only this direction. A result that means
something only once a check passes (a bit, a Merkle path) then feeds the
rest of the program, and a check may name the tape cells of its own run. -/
def Ensures {α : Type} (c : Ctx F) (n : ℕ) (P : Prop) (b : Build α) (Q : α → ℕ → Prop) : Prop :=
  ∀ (p : Builder) (E : Exec.State F), p.values = n → E.values.length = n →
    ∃ new : List Instr, (b.run p).2.instrs = p.instrs ++ new.toArray ∧
      (c.steps new E).values.length = (b.run p).2.values ∧
      (P → c.Follows (c.steps new E) → (c.steps new E).holds → Q (b.run p).1 (b.run p).2.values)

variable {c : Ctx F} {α β γ δ : Type} {n : ℕ} {P : Prop} {Q : α → ℕ → Prop}

/-- A simulated program ensures its result and its checks. -/
theorem Sim.ensures {b : Build α} {C : Prop} (h : Sim c n P b Q C) : Ensures c n P b fun a m => Q a m ∧ C := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF hh => ⟨(hs hP hF).1, ((hs hP hF).2.mp hh).2⟩⟩

/-- A simulated program that checks nothing ensures its result. -/
theorem Sim.ensures₀ {b : Build α} (h : Sim c n P b Q True) : Ensures c n P b Q := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF _ => (hs hP hF).1⟩

namespace Ensures

theorem pure {a : α} (h : P → Q a n) : Ensures c n P (Pure.pure a) Q := by
  intro p E hp hE
  refine ⟨[], by show p.instrs = p.instrs ++ #[]; simp, by show E.values.length = p.values; rw [hE, hp],
    fun hP _ _ => ?_⟩
  exact hp ▸ h hP

theorem bind {b : Build α} {f : α → Build β} {R : β → ℕ → Prop} (hb : Ensures c n P b Q)
    (hf : ∀ a m, n ≤ m → Ensures c m (P ∧ Q a m) (f a) R) : Ensures c n P (b >>= f) R := by
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
    exact hs₂ ⟨hP, hs₁ hP (Ctx.follows_of_steps hF) (Ctx.holds_of_steps hh)⟩ hF hh

theorem post {b : Build α} {Q' : α → ℕ → Prop} (h : Ensures c n P b Q) (hq : P → ∀ a m, Q a m → Q' a m) :
    Ensures c n P b Q' := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF hh => hq hP _ _ (hs hP hF hh)⟩

theorem pre {b : Build α} {P' : Prop} (h : Ensures c n P' b Q) (hp : P → P') : Ensures c n P b Q := by
  intro p E hp' hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp' hE
  exact ⟨new, hi, hl, fun hP hF hh => hs (hp hP) hF hh⟩

theorem map {b : Build α} {f : α → β} {Q' : β → ℕ → Prop} (h : Ensures c n P b Q)
    (hq : ∀ a m, Q a m → Q' (f a) m) : Ensures c n P (f <$> b) Q' := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF hh => hq _ _ (hs hP hF hh)⟩

/-- Use a fact `A` that `P` gives. The instructions do not depend on the
precondition, so `h₀`, at any precondition, fixes them. -/
theorem under {P₀ A : Prop} {b : Build α} {Q₀ : α → ℕ → Prop} (h₀ : Ensures c n P₀ b Q₀) (hA : P → A)
    (h : A → Ensures c n P b Q) : Ensures c n P b Q := by
  intro p E hp hE
  obtain ⟨new, hi, hl, -⟩ := h₀ p E hp hE
  refine ⟨new, hi, hl, fun hP hF hh => ?_⟩
  obtain ⟨new', hi', -, hs⟩ := h (hA hP) p E hp hE
  obtain rfl : new = new' := List.toArray_inj ((Array.append_right_inj _).mp (hi.symm.trans hi'))
  exact hs hP hF hh

/-- Fix the witness of an existential precondition. Any `x₀` gives the
instructions. -/
theorem elim {ι : Type} {b : Build α} {R : ι → Prop} (x₀ : ι) (h : ∀ x, Ensures c n (R x) b Q)
    (hp : P → ∃ x, R x) : Ensures c n P b Q :=
  under (h x₀) hp fun ⟨x, hx⟩ => (h x).pre fun _ => hx

/-- Name the witness of an existential precondition. -/
theorem exists_and {ι : Type} {b : Build α} {X : ι → Prop} (i₀ : ι) (h : ∀ i, Ensures c n (P ∧ X i) b Q) :
    Ensures c n (P ∧ ∃ i, X i) b Q :=
  elim i₀ h fun ⟨hP, i, hx⟩ => ⟨i, hP, hx⟩

/-- Name the witness, using a fact `A` of each witness; `i₀` has it and fixes
the instructions. -/
theorem witness {ι : Type} {b : Build α} {X A : ι → Prop} (i₀ : ι) (h₀ : A i₀) (hA : ∀ i, X i → A i)
    (h : ∀ i, A i → Ensures c n (P ∧ X i) b Q) : Ensures c n (P ∧ ∃ i, X i) b Q :=
  under (A := ∃ i, X i ∧ A i) (h i₀ h₀) (fun H => let ⟨i, hx⟩ := H.2; ⟨i, hx, hA i hx⟩) fun ⟨i, hx, ha⟩ =>
    (h i ha).pre fun H => ⟨H.1, hx⟩

/-- `bind` after a step that checks nothing. -/
theorem bindSim {b : Build α} {f : α → Build β} {R : β → ℕ → Prop} (hb : Sim c n P b Q True)
    (hf : ∀ a m, n ≤ m → Ensures c m (P ∧ Q a m) (f a) R) : Ensures c n P (b >>= f) R :=
  hb.ensures₀.bind hf

/-- `bind` after an assertion. -/
theorem bindCheck {b : Build α} {f : α → Build β} {C : Prop} {R : β → ℕ → Prop}
    (hb : Sim c n P b (fun _ _ => True) C) (hf : ∀ a m, n ≤ m → Ensures c m (P ∧ C) (f a) R) :
    Ensures c n P (b >>= f) R :=
  (hb.ensures.post fun _ _ _ h => h.2).bind hf

/-- `t >>= g` in a transcript. -/
theorem tbind {t : Transcript α} {g : α → Transcript β} {d : Duplex} {R : β × Duplex → ℕ → Prop}
    {Q : α × Duplex → ℕ → Prop} (ht : Ensures c n P (t.run d) Q)
    (hg : ∀ a e m, n ≤ m → Ensures c m (P ∧ Q (a, e) m) ((g a).run e) R) : Ensures c n P ((t >>= g).run d) R := by
  rw [StateT.run_bind]
  exact ht.bind fun r m le => hg r.1 r.2 m le

/-- `tbind` after a step that checks nothing. -/
theorem tbindSim {t : Transcript α} {g : α → Transcript β} {d : Duplex} {R : β × Duplex → ℕ → Prop}
    {Q : α × Duplex → ℕ → Prop} (ht : Sim c n P (t.run d) Q True)
    (hg : ∀ a e m, n ≤ m → Ensures c m (P ∧ Q (a, e) m) ((g a).run e) R) : Ensures c n P ((t >>= g).run d) R :=
  ht.ensures₀.tbind hg

/-- A builder call in a transcript. -/
theorem lift {b : Build α} {d : Duplex} (h : Ensures c n P b Q) :
    Ensures c n P ((monadLift b : Transcript α).run d) (fun r m => Q r.1 m ∧ r.2 = d) := by
  rw [StateT.run_monadLift]
  exact h.bind fun a m _ => Ensures.pure fun h => ⟨h.2, rfl⟩

/-- A loop from step `j` on. -/
theorem loop_from {xs : List δ} {body : δ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → Ensures c m (P ∧ I k acc m) (body xs[k] acc)
      (fun r m' => r = .yield r.value ∧ I (k + 1) r.value m')) :
    ∀ t j, xs.length - j = t → j ≤ xs.length → ∀ acc m, n ≤ m →
      Ensures c m (P ∧ I j acc m) (forIn (xs.drop j) acc body) (fun r m' => I xs.length r m')
  | 0, j, ht, hj, acc, m, _ => by
      have e : j = xs.length := by omega
      subst e
      rw [List.drop_length, List.forIn_nil]
      exact Ensures.pure fun h => h.2
  | t + 1, j, ht, hj, acc, m, hm => by
      have hj' : j < xs.length := by omega
      rw [List.drop_eq_getElem_cons hj', List.forIn_cons]
      refine (step j hj' acc m hm).bind fun r m' le => ?_
      cases r with
      | done b => exact (Ensures.pure (Q := fun _ _ => True) fun _ => trivial).post fun h => absurd h.2.1 (by simp)
      | yield b => exact (loop_from I step t (j + 1) (by omega) hj' b m' (hm.trans le)).pre fun h => ⟨h.1.1, h.2.2⟩

/-- **A loop over a list.** `I k acc m`: after `k` steps the accumulator is
`acc` with `m` values. -/
theorem loop {xs : List δ} {init : γ} {body : δ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → Ensures c m (P ∧ I k acc m) (body xs[k] acc)
      (fun r m' => r = .yield r.value ∧ I (k + 1) r.value m')) :
    Ensures c n (P ∧ I 0 init n) (forIn xs init body) (fun r m => I xs.length r m) :=
  loop_from I step _ 0 rfl (Nat.zero_le _) init n le_rfl

/-- A loop over a list in a transcript; `I k acc e m` also names the duplex. -/
theorem loopT_from {xs : List δ} {body : δ → γ → Transcript (ForInStep γ)} (I : ℕ → γ → Duplex → ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc e m, n ≤ m → Ensures c m (P ∧ I k acc e m) ((body xs[k] acc).run e)
      (fun r m' => r.1 = .yield r.1.value ∧ I (k + 1) r.1.value r.2 m')) :
    ∀ t j, xs.length - j = t → j ≤ xs.length → ∀ acc e m, n ≤ m →
      Ensures c m (P ∧ I j acc e m) ((forIn (xs.drop j) acc body).run e) (fun r m' => I xs.length r.1 r.2 m')
  | 0, j, ht, hj, acc, e, m, _ => by
      have h : j = xs.length := by omega
      subst h
      rw [List.drop_length, List.forIn_nil, StateT.run_pure]
      exact Ensures.pure fun h => h.2
  | t + 1, j, ht, hj, acc, e, m, hm => by
      have hj' : j < xs.length := by omega
      rw [List.drop_eq_getElem_cons hj', List.forIn_cons, StateT.run_bind]
      refine (step j hj' acc e m hm).bind fun r m' le => ?_
      obtain ⟨r, e'⟩ := r
      cases r with
      | done b =>
          show Ensures c m' _ (Pure.pure (b, e') : Build (γ × Duplex)) _
          exact (Ensures.pure (Q := fun _ _ => True) fun _ => trivial).post fun h => absurd h.2.1 (by simp)
      | yield b =>
          exact (loopT_from I step t (j + 1) (by omega) hj' b e' m' (hm.trans le)).pre fun h => ⟨h.1.1, h.2.2⟩

/-- **A loop over a list in a transcript.** -/
theorem loopT {xs : List δ} {init : γ} {body : δ → γ → Transcript (ForInStep γ)} {d : Duplex}
    (I : ℕ → γ → Duplex → ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc e m, n ≤ m → Ensures c m (P ∧ I k acc e m) ((body xs[k] acc).run e)
      (fun r m' => r.1 = .yield r.1.value ∧ I (k + 1) r.1.value r.2 m')) :
    Ensures c n (P ∧ I 0 init d n) ((forIn xs init body).run d) (fun r m => I xs.length r.1 r.2 m) :=
  loopT_from I step _ 0 rfl (Nat.zero_le _) init d n le_rfl

/-- **An accepting run meets what the program ensures.** -/
theorem exec {b : Build α} (h : Ensures c 0 P b Q) (hP : P)
    (hval : c.val = (Exec.run c.perm c.stmt c.tape b.program).value)
    (accepts : Exec c.perm b.program c.stmt c.tape) : Q (b.run {}).1 (b.run {}).2.values := by
  obtain ⟨new, hi, -, hs⟩ := h {} Exec.start rfl rfl
  have run : Exec.run c.perm c.stmt c.tape b.program = c.steps new Exec.start := by
    rw [Exec.run, Build.program, hi]
    simp only [Array.empty_append, List.toList_toArray]
    rfl
  exact hs hP (fun v _ => by rw [hval, run]) (by rw [Exec, run] at accepts; exact accepts)

end Ensures

/-- `Ensures` from one builder state `p`. The program's result is then a
known term, `(b.run p).1`, so the next program can name the handles that the
run writes. -/
def EnsuresAt (c : Ctx F) (p : Builder) (P : Prop) (b : Build α) (Q : α → ℕ → Prop) : Prop :=
  ∀ (E : Exec.State F), E.values.length = p.values →
    ∃ new : List Instr, (b.run p).2.instrs = p.instrs ++ new.toArray ∧
      (c.steps new E).values.length = (b.run p).2.values ∧
      (P → c.Follows (c.steps new E) → (c.steps new E).holds → Q (b.run p).1 (b.run p).2.values)

namespace EnsuresAt

variable {α β : Type} {P : Prop}

theorem of_ensures {p : Builder} {b : Build α} {Q : α → ℕ → Prop} (h : Ensures c p.values P b Q) :
    EnsuresAt c p P b Q := fun E hE => h p E rfl hE

/-- `bind` at the state that `b` leaves, with the result that `b` returns. -/
theorem bind {p : Builder} {b : Build α} {f : α → Build β} {Q : α → ℕ → Prop} {R : β → ℕ → Prop}
    (hb : EnsuresAt c p P b Q)
    (hf : p.values ≤ (b.run p).2.values →
      EnsuresAt c (b.run p).2 (P ∧ Q (b.run p).1 (b.run p).2.values) (f (b.run p).1) R) :
    EnsuresAt c p P (b >>= f) R := by
  intro E hE
  obtain ⟨new₁, hi₁, hl₁, hs₁⟩ := hb E hE
  have hle : p.values ≤ (b.run p).2.values := by
    obtain ⟨more, hm, -⟩ := c.steps_values new₁ E
    rw [← hl₁, hm, List.length_append, hE]
    omega
  obtain ⟨new₂, hi₂, hl₂, hs₂⟩ := hf hle (c.steps new₁ E) hl₁
  refine ⟨new₁ ++ new₂, ?_, ?_, ?_⟩
  · show ((f (b.run p).1).run (b.run p).2).2.instrs = _
    rw [hi₂, hi₁]
    simp
  · rw [Ctx.steps_append]
    exact hl₂
  · intro hP hF hh
    rw [Ctx.steps_append] at hF hh
    exact hs₂ ⟨hP, hs₁ hP (Ctx.follows_of_steps hF) (Ctx.holds_of_steps hh)⟩ hF hh

theorem pre {P' : Prop} {p : Builder} {b : Build α} {Q : α → ℕ → Prop} (h : EnsuresAt c p P b Q) (hp : P' → P) :
    EnsuresAt c p P' b Q := by
  intro E hE
  obtain ⟨new, hi, hl, hs⟩ := h E hE
  exact ⟨new, hi, hl, fun h' => hs (hp h')⟩

theorem post {p : Builder} {b : Build α} {Q Q' : α → ℕ → Prop} (h : EnsuresAt c p P b Q)
    (hq : P → Q (b.run p).1 (b.run p).2.values → Q' (b.run p).1 (b.run p).2.values) : EnsuresAt c p P b Q' := by
  intro E hE
  obtain ⟨new, hi, hl, hs⟩ := h E hE
  exact ⟨new, hi, hl, fun hP hF hh => hq hP (hs hP hF hh)⟩

/-- **An accepting run meets what the program ensures from the empty builder.** -/
theorem exec {b : Build α} {Q : α → ℕ → Prop} (h : EnsuresAt c {} P b Q) (hP : P)
    (hval : c.val = (Exec.run c.perm c.stmt c.tape b.program).value)
    (accepts : Exec c.perm b.program c.stmt c.tape) : Q (b.run {}).1 (b.run {}).2.values := by
  obtain ⟨new, hi, -, hs⟩ := h Exec.start rfl
  have run : Exec.run c.perm c.stmt c.tape b.program = c.steps new Exec.start := by
    rw [Exec.run, Build.program, hi]
    simp only [Array.empty_append, List.toList_toArray]
    rfl
  exact hs hP (fun v _ => by rw [hval, run]) (by rw [Exec, run] at accepts; exact accepts)

end EnsuresAt

end CompressionLean.Circuit
