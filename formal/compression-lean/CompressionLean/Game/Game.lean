import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

namespace CompressionLean

open scoped BigOperators

inductive Game (α : Type) : Type 1 where
  | pure : α → Game α
  | draw (C : Type) [Fintype C] [Nonempty C] (bad : Finset C) (next : C → Game α) : Game α
  | recv (M : Type) [Nonempty M] (next : M → Game α) : Game α

namespace Game

variable {α β : Type}

def bind : Game α → (α → Game β) → Game β
  | pure a, f => f a
  | draw C bad next, f => draw C bad (fun c => bind (next c) f)
  | recv M next, f => recv M (fun m => bind (next m) f)

def Avoids : Game α → (α → Prop) → Prop
  | pure a, Q => Q a
  | draw _ bad next, Q => ∀ c, c ∉ bad → Avoids (next c) Q
  | recv _ next, Q => ∀ m, Avoids (next m) Q

def CostLe : Game α → ℝ → Prop
  | pure _, ε => 0 ≤ ε
  | draw C bad next, ε =>
      ∃ rest : ℝ, (bad.card : ℝ) / Fintype.card C + rest ≤ ε ∧ ∀ c, CostLe (next c) rest
  | recv _ next, ε => ∀ m, CostLe (next m) ε

def Strategy (α : Type) : Type 1 := ∀ (M : Type) [Nonempty M], (M → Game α) → M

noncomputable def prob (s : Strategy α) : Game α → (α → Prop) → ℝ
  | pure a, E => by classical exact if E a then 1 else 0
  | draw _ _ next, E => 𝔼 c, prob s (next c) E
  | recv M next, E => prob s (next (s M next)) E

theorem prob_le_one (s : Strategy α) (g : Game α) (E : α → Prop) : prob s g E ≤ 1 := by
  induction g with
  | pure a => simp only [prob]; split <;> norm_num
  | draw C bad next ih => exact Finset.expect_le Finset.univ_nonempty (fun c _ => ih c)
  | recv M next ih => exact ih _

theorem CostLe.nonneg {g : Game α} {ε : ℝ} (h : g.CostLe ε) : 0 ≤ ε := by
  induction g generalizing ε with
  | pure a => exact h
  | draw C bad next ih =>
      obtain ⟨rest, bound, nexts⟩ := h
      have density : 0 ≤ (bad.card : ℝ) / Fintype.card C := by positivity
      linarith [ih (Classical.arbitrary C) (nexts (Classical.arbitrary C))]
  | recv M next ih => exact ih (Classical.arbitrary M) (h (Classical.arbitrary M))

theorem CostLe.mono {g : Game α} {ε δ : ℝ} (h : g.CostLe ε) (le : ε ≤ δ) : g.CostLe δ := by
  induction g generalizing ε δ with
  | pure a => exact h.trans le
  | draw C bad next ih =>
      obtain ⟨rest, bound, nexts⟩ := h
      exact ⟨rest, bound.trans le, nexts⟩
  | recv M next ih => exact fun m => ih m (h m) le

open Classical in
private theorem expect_mem (C : Type) [Fintype C] [Nonempty C] (bad : Finset C) :
    (𝔼 c, (if c ∈ bad then (1 : ℝ) else 0)) = (bad.card : ℝ) / Fintype.card C := by
  classical
  rw [Finset.expect_eq_sum_div_card, Finset.sum_boole, Finset.card_univ]
  congr 2
  simp

theorem prob_not_le (s : Strategy α) (g : Game α) (Q : α → Prop) (ε : ℝ)
    (avoids : g.Avoids Q) (cost : g.CostLe ε) : prob s g (fun a => ¬ Q a) ≤ ε := by
  classical
  induction g generalizing ε with
  | pure a =>
      have holds : Q a := avoids
      simp only [prob]
      rw [ite_eq_right (not_not.mpr holds)]
      exact cost
  | draw C bad next ih =>
      obtain ⟨rest, bound, nexts⟩ := cost
      have restNonneg : 0 ≤ rest := (nexts (Classical.arbitrary C)).nonneg
      have each (c : C) :
          prob s (next c) (fun a => ¬ Q a) ≤ (if c ∈ bad then (1 : ℝ) else 0) + rest := by
        by_cases inside : c ∈ bad
        · rw [ite_eq_left inside]
          linarith [prob_le_one s (next c) (fun a => ¬ Q a)]
        · rw [ite_eq_right inside, zero_add]
          exact ih c rest (avoids c inside) (nexts c)
      calc prob s (draw C bad next) (fun a => ¬ Q a)
          = 𝔼 c, prob s (next c) (fun a => ¬ Q a) := rfl
        _ ≤ 𝔼 c, ((if c ∈ bad then (1 : ℝ) else 0) + rest) :=
            Finset.expect_le_expect (fun c _ => each c)
        _ = (bad.card : ℝ) / Fintype.card C + rest := by
            rw [Finset.expect_add_distrib, expect_mem, Finset.expect_const Finset.univ_nonempty]
        _ ≤ ε := bound
  | recv M next ih => exact ih _ ε (avoids _) (cost _)

theorem Avoids.mono {g : Game α} {Q R : α → Prop} (h : g.Avoids Q) (imp : ∀ a, Q a → R a) :
    g.Avoids R := by
  induction g with
  | pure a => exact imp a h
  | draw C bad next ih => exact fun c out => ih c (h c out)
  | recv M next ih => exact fun m => ih m (h m)

theorem avoids_bind (g : Game α) (f : α → Game β) (Q : β → Prop) :
    (bind g f).Avoids Q ↔ g.Avoids (fun a => (f a).Avoids Q) := by
  induction g with
  | pure a => exact Iff.rfl
  | draw C bad next ih => exact forall_congr' fun c => imp_congr_right fun _ => ih c
  | recv M next ih => exact forall_congr' fun m => ih m

def Ends : Game α → (α → Prop) → Prop
  | pure a, P => P a
  | draw _ _ next, P => ∀ c, Ends (next c) P
  | recv _ next, P => ∀ m, Ends (next m) P

theorem Ends.of_forall {P : α → Prop} (all : ∀ a, P a) (g : Game α) : g.Ends P := by
  induction g with
  | pure a => exact all a
  | draw C bad next ih => exact ih
  | recv M next ih => exact ih

theorem ends_bind (g : Game α) (f : α → Game β) (P : β → Prop) :
    (bind g f).Ends P ↔ g.Ends (fun a => (f a).Ends P) := by
  induction g with
  | pure a => exact Iff.rfl
  | draw C bad next ih => exact forall_congr' fun c => ih c
  | recv M next ih => exact forall_congr' fun m => ih m

theorem costLe_bind_of {g : Game α} {f : α → Game β} {P : α → Prop} {ε δ : ℝ} (h : g.CostLe ε)
    (ends : g.Ends P) (hf : ∀ a, P a → (f a).CostLe δ) : (bind g f).CostLe (ε + δ) := by
  induction g generalizing ε with
  | pure a =>
      have := (hf a ends).nonneg
      exact (hf a ends).mono (by linarith [show (0 : ℝ) ≤ ε from h])
  | draw C bad next ih =>
      obtain ⟨rest, bound, nexts⟩ := h
      exact ⟨rest + δ, by linarith, fun c => ih c (nexts c) (ends c)⟩
  | recv M next ih => exact fun m => ih m (h m) (ends m)

theorem costLe_bind {g : Game α} {f : α → Game β} {ε δ : ℝ} (h : g.CostLe ε)
    (hf : ∀ a, (f a).CostLe δ) : (bind g f).CostLe (ε + δ) :=
  costLe_bind_of h (Ends.of_forall (P := fun _ => True) (fun _ => trivial) g) fun a _ => hf a

theorem density_le {C : Type} [Fintype C] (bad : Finset C) (n : ℕ) (card : bad.card ≤ n) :
    (bad.card : ℝ) / Fintype.card C ≤ n / Fintype.card C :=
  div_le_div_of_nonneg_right (by exact_mod_cast card) (by positivity)

theorem draw_step {α C D : Type} [Fintype C] [Nonempty C] [Fintype D] (bad : Finset C)
    (next : C → Game α) (a rest : ℕ) (density : (bad.card : ℝ) / Fintype.card C ≤ a / Fintype.card D)
    (nexts : ∀ c, (next c).CostLe (rest / Fintype.card D)) :
    (Game.draw C bad next).CostLe (((a + rest : ℕ) : ℝ) / Fintype.card D) :=
  ⟨_, by push_cast; rw [add_div]; linarith, nexts⟩

end Game

end CompressionLean
