import CompressionLean.Lowering.Exec
import CompressionLean.Ports.Circuit.Hash
import CompressionLean.Bridge.Field
import Mathlib.Data.List.GetD

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lowering
open CompressionLean.Replay (Word Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

section Words

variable [CharP F goldilocksModulus]

theorem word_eq_cast (g : Word) : (Word.cast g : F) = ZMod.castHom (dvd_refl _) F g.toF := by
  simp [Word.cast, Word.toF, Word.toNat]

theorem word_add (a b : Word) : (Word.cast (a + b) : F) = Word.cast a + Word.cast b := by
  rw [word_eq_cast, word_eq_cast, word_eq_cast, Word.toF_add, map_add]

theorem word_mul (a b : Word) : (Word.cast (a * b) : F) = Word.cast a * Word.cast b := by
  rw [word_eq_cast, word_eq_cast, word_eq_cast, Word.toF_mul, map_mul]

theorem word_sub (a b : Word) : (Word.cast (a - b) : F) = Word.cast a - Word.cast b := by
  rw [word_eq_cast, word_eq_cast, word_eq_cast, Word.toF_sub, map_sub]

theorem word_ofNat (k : ℕ) : (Word.cast (Word.ofNat k) : F) = k := by
  rw [word_eq_cast, Word.toF_ofNat, map_natCast]

@[simp] theorem word_zero : (Word.cast 0 : F) = 0 := by
  rw [word_eq_cast, Word.toF_zero, map_zero]

@[simp] theorem word_one : (Word.cast 1 : F) = 1 := by
  rw [word_eq_cast, Word.toF_one, map_one]

theorem word_neg (a : Word) : (Word.cast (-a) : F) = -Word.cast a := by
  rw [word_eq_cast, word_eq_cast, Word.toF_neg, map_neg]

theorem word_injective : Function.Injective (Word.cast : Word → F) := fun a b h => by
  rw [word_eq_cast, word_eq_cast] at h
  exact Word.eq_of_toF (ZMod.castHom_injective F h)

end Words


structure Ctx (F : Type) where
  perm : List F → List F
  stmt : List F
  tape : ℕ → F
  val : Var → F

namespace Ctx

variable (c : Ctx F)

def step (E : Exec.State F) (i : Instr) : Exec.State F := Exec.step c.perm c.stmt c.tape E i

def steps (L : List Instr) (E : Exec.State F) : Exec.State F := L.foldl c.step E

def Follows (E : Exec.State F) : Prop := ∀ h < E.values.length, E.value h = c.val h

theorem steps_append (L L' : List Instr) (E : Exec.State F) : c.steps (L ++ L') E = c.steps L' (c.steps L E) :=
  List.foldl_append

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

theorem holds_of_step {c : Ctx F} {E : Exec.State F} {i : Instr} (h : (c.step E i).holds) : E.holds := by
  cases i <;> simp only [Ctx.step, Exec.step, Exec.State.next] at h <;> first | exact h | exact h.1

theorem holds_of_steps {c : Ctx F} {L : List Instr} {E : Exec.State F} (h : (c.steps L E).holds) : E.holds := by
  induction L generalizing E with
  | nil => exact h
  | cons i L ih => exact holds_of_step (ih h)

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

def IsWord (c : Ctx F) (n : ℕ) (h : Var) (g : Word) : Prop := h < n ∧ c.val h = Word.cast g

theorem IsWord.mono {c : Ctx F} {n m : ℕ} {h : Var} {g : Word} (w : IsWord c n h g) (le : n ≤ m := by omega) :
    IsWord c m h g :=
  ⟨lt_of_lt_of_le w.1 le, w.2⟩

theorem IsWord.congr {c : Ctx F} {n : ℕ} {h : Var} {g g' : Word} (w : IsWord c n h g) (e : (Word.cast g : F) = Word.cast g') :
    IsWord c n h g' :=
  ⟨w.1, w.2.trans e⟩

theorem IsWord.read {c : Ctx F} {n : ℕ} {h : Var} {g : Word} (w : IsWord c n h g) {E : Exec.State F}
    (hF : c.Follows E) (hl : E.values.length = n) : E.value h = Word.cast g :=
  (hF h (hl ▸ w.1)).trans w.2


def _root_.CompressionLean.Replay.Ext.coord (e : Ext) : Fin 3 → Word := ![e.c0, e.c1, e.c2]

def IsExt (c : Ctx F) (n : ℕ) (a : ExtVar) (e : Ext) : Prop := ∀ k, IsWord c n (a k) (e.coord k)

theorem IsExt.mono {c : Ctx F} {n m : ℕ} {a : ExtVar} {e : Ext} (h : IsExt c n a e) (le : n ≤ m := by omega) :
    IsExt c m a e :=
  fun k => (h k).mono le

def IsPair (c : Ctx F) (n : ℕ) (a : ExtVar × ExtVar) (g : Ext × Ext) : Prop := IsExt c n a.1 g.1 ∧ IsExt c n a.2 g.2

theorem IsPair.mono {c : Ctx F} {n m : ℕ} {a : ExtVar × ExtVar} {g : Ext × Ext} (h : IsPair c n a g)
    (le : n ≤ m := by omega) :
    IsPair c m a g :=
  ⟨h.1.mono le, h.2.mono le⟩

def Permutes (perm : List F → List F) : Prop :=
  ∀ (s : Array Word) (k : Fin 16), (perm (List.ofFn fun i : Fin 16 => Word.cast s[i]!)).getD k 0 = Word.cast (Replay.permute s)[k]!

namespace Ctx

variable {c : Ctx F}

theorem follows_of_step {E : Exec.State F} {i : Instr} (h : c.Follows (c.step E i)) : c.Follows E :=
  follows_of_steps (L := [i]) h

theorem val_next {E : Exec.State F} {outs : List F} {check : Prop} (hF : c.Follows (E.next outs check)) {k : ℕ}
    (hk : k < outs.length) : c.val (E.values.length + k) = outs.getD k 0 := by
  rw [← hF _ (by simp [Exec.State.next]; omega), Exec.State.value]
  simp only [Exec.State.next]
  rw [List.getD_append_right _ _ _ _ (by omega), Nat.add_sub_cancel_left]

end Ctx

@[simp] theorem Ext.coord_one : (1 : Ext).coord = ![1, 0, 0] := rfl

@[simp] theorem Ext.coord_zero : (0 : Ext).coord = ![0, 0, 0] := rfl

theorem gl_sub_eq_zero {a b : Word} : a - b = 0 ↔ a = b := by
  refine ⟨fun h => Word.eq_of_toF ?_, fun h => ?_⟩
  · have := congrArg Word.toF h
    rw [Word.toF_sub, Word.toF_zero] at this
    exact sub_eq_zero.mp this
  · subst h
    exact Word.eq_of_toF (by rw [Word.toF_sub, sub_self, Word.toF_zero])

section Coordinates

variable [CharP F goldilocksModulus]

theorem coord_mul (a b : Ext) :
    (fun k => (Word.cast ((a * b).coord k) : F)) = extProduct (fun i => Word.cast (a.coord i)) (fun i => Word.cast (b.coord i)) := by
  obtain ⟨a0, a1, a2⟩ := a
  obtain ⟨b0, b1, b2⟩ := b
  funext k
  fin_cases k
  · show (Word.cast (a0 * b0 + (a1 * b2 + a2 * b1)) : F) = _
    simp [extProduct, Ext.coord, word_add, word_mul]
    ring
  · show (Word.cast (a0 * b1 + a1 * b0 + (a1 * b2 + a2 * b1) + a2 * b2) : F) = _
    simp [extProduct, Ext.coord, word_add, word_mul]
    ring
  · show (Word.cast (a0 * b2 + a1 * b1 + a2 * b0 + a2 * b2) : F) = _
    simp [extProduct, Ext.coord, word_add, word_mul]
    ring

theorem Ext.eq_of_coord {a b : Ext} (h : ∀ k, (Word.cast (a.coord k) : F) = Word.cast (b.coord k)) : a = b := by
  obtain ⟨a0, a1, a2⟩ := a
  obtain ⟨b0, b1, b2⟩ := b
  have e0 : a0 = b0 := word_injective (h 0)
  have e1 : a1 = b1 := word_injective (h 1)
  have e2 : a2 = b2 := word_injective (h 2)
  subst e0 e1 e2
  rfl

end Coordinates

end CompressionLean.Sim
