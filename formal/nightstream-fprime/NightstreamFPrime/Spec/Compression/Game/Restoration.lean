import NightstreamFPrime.Spec.Compression.Game.Game

/-! The state-restoration experiment of a game (Fiat–Shamir with a random oracle). The prover
asks for the challenge of any node, at most `q` times, and then names a path (`restore`). The path
counts only when the oracle answered each of its draws (`walk`). A break has probability at most
`q` times the largest bad-set density of a node (`restore_sound`). -/

namespace NightstreamFPrime.Spec.Compression.Game

variable {α : Type}

def Pos : Game α → Type
  | pure _ => Unit
  | draw C _ next => Option (Σ c : C, Pos (next c))
  | recv M next => Option (Σ m : M, Pos (next m))

instance Pos.nonempty (g : Game α) : Nonempty (Pos g) := by
  cases g with
  | pure a => exact ⟨()⟩
  | draw C bad next => exact ⟨(none : Option (Σ c : C, Pos (next c)))⟩
  | recv M next => exact ⟨(none : Option (Σ m : M, Pos (next m)))⟩

def Pos.node : (g : Game α) → Pos g → Game α
  | pure a, _ => pure a
  | g@(draw _ _ _), none => g
  | draw _ _ next, some ⟨c, p⟩ => Pos.node (next c) p
  | g@(recv _ _), none => g
  | recv _ next, some ⟨m, p⟩ => Pos.node (next m) p

abbrev Log (α : Type) := List (Game α × Game α)

def answer {β : Type} (k : Log α → Game β) (log : Log α) : Game α → Game β
  | n@(draw C bad next) => .draw C bad fun c => k ((n, next c) :: log)
  | _ => k log

open Classical in
noncomputable def walk (log : Log α) : (g : Game α) → Pos g → Option α
  | pure a, _ => some a
  | draw _ _ _, none => none
  | g@(draw _ _ next), some ⟨c, p⟩ => if (g, next c) ∈ log then walk log (next c) p else none
  | recv _ _, none => none
  | recv _ next, some ⟨m, p⟩ => walk log (next m) p

def Keeps (Q : α → Prop) (log : Log α) : Prop :=
  ∀ e ∈ log, e.1.Avoids Q → e.2.Avoids Q

theorem walk_sound {Q : α → Prop} {log : Log α} (keeps : Keeps Q log) (g : Game α) :
    ∀ (p : Pos g) (a : α), g.Avoids Q → walk log g p = some a → Q a := by
  induction g with
  | pure b =>
      intro p a avoids equal
      simp only [walk, Option.some.injEq] at equal
      exact equal ▸ avoids
  | draw C bad next ih =>
      intro p a avoids equal
      match p with
      | none => simp [walk] at equal
      | some ⟨c, p⟩ =>
          simp only [walk] at equal
          split_ifs at equal with logged
          exact ih c p a (keeps _ logged avoids) equal
  | recv M next ih =>
      intro p a avoids equal
      match p with
      | none => simp [walk] at equal
      | some ⟨m, p⟩ => exact ih m p a (avoids m) equal

def DensityLe : Game α → ℝ → Prop
  | pure _, _ => True
  | draw C bad next, ε => (bad.card : ℝ) / Fintype.card C ≤ ε ∧ ∀ c, DensityLe (next c) ε
  | recv _ next, ε => ∀ m, DensityLe (next m) ε

theorem DensityLe.mono {g : Game α} {ε δ : ℝ} (h : g.DensityLe ε) (le : ε ≤ δ) : g.DensityLe δ := by
  induction g with
  | pure a => trivial
  | draw C bad next ih => exact ⟨h.1.trans le, fun c => ih c (h.2 c)⟩
  | recv M next ih => exact fun m => ih m (h m)

theorem CostLe.densityLe {g : Game α} {ε : ℝ} (h : g.CostLe ε) : g.DensityLe ε := by
  induction g generalizing ε with
  | pure a => trivial
  | draw C bad next ih =>
      obtain ⟨rest, bound, nexts⟩ := h
      have restNonneg : 0 ≤ rest := (nexts (Classical.arbitrary C)).nonneg
      have density : 0 ≤ (bad.card : ℝ) / Fintype.card C := by positivity
      exact ⟨by linarith, fun c => (ih c (nexts c)).mono (by linarith)⟩
  | recv M next ih => exact fun m => ih m (h m)

theorem DensityLe.node {g : Game α} {ε : ℝ} (h : g.DensityLe ε) (p : Pos g) : (Pos.node g p).DensityLe ε := by
  induction g with
  | pure a => exact h
  | draw C bad next ih =>
      match p with
      | none => exact h
      | some ⟨c, p⟩ => exact ih c (h.2 c) p
  | recv M next ih =>
      match p with
      | none => exact h
      | some ⟨m, p⟩ => exact ih m (h m) p

theorem densityLe_bind {β : Type} {g : Game α} {f : α → Game β} {P : α → Prop} {ε : ℝ}
    (h : g.DensityLe ε) (ends : g.Ends P) (hf : ∀ a, P a → (f a).DensityLe ε) :
    (bind g f).DensityLe ε := by
  induction g with
  | pure a => exact hf a ends
  | draw C bad next ih => exact ⟨h.1, fun c => ih c (h.2 c) (ends c)⟩
  | recv M next ih => exact fun m => ih m (h m) (ends m)

theorem prob_mono (s : Strategy α) (g : Game α) {E E' : α → Prop} (imp : ∀ a, E a → E' a) :
    prob s g E ≤ prob s g E' := by
  classical
  induction g with
  | pure a =>
      simp only [prob]
      by_cases h : E a
      · rw [ite_eq_left h, ite_eq_left (imp a h)]
      · rw [ite_eq_right h]
        split_ifs <;> norm_num
  | draw C bad next ih => exact Finset.expect_le_expect fun c _ => ih c
  | recv M next ih => exact ih _

/-- The experiment: `q` oracle queries at chosen nodes, then a path. -/
noncomputable def restore (g : Game α) : ℕ → Log α → Game (Option α)
  | 0, log => .recv (Pos g) fun p => .pure (walk log g p)
  | q + 1, log => .recv (Pos g) fun p => answer (fun log' => restore g q log') log (Pos.node g p)

theorem restore_avoids (g : Game α) (Q : α → Prop) (avoids : g.Avoids Q) :
    ∀ (q : ℕ) (log : Log α), Keeps Q log →
      (restore g q log).Avoids fun r => ∀ a, r = some a → Q a := by
  intro q
  induction q with
  | zero =>
      intro log keeps p
      exact fun a equal => walk_sound keeps g p a avoids equal
  | succ q ih =>
      intro log keeps p
      show (answer (fun log' => restore g q log') log (Pos.node g p)).Avoids _
      generalize Pos.node g p = n
      cases n with
      | draw C bad next =>
          intro c out
          refine ih _ fun e member => ?_
          rcases List.mem_cons.mp member with rfl | inside
          · exact fun node => node c out
          · exact keeps e inside
      | pure b => exact ih log keeps
      | recv M next => exact ih log keeps

theorem restore_cost (g : Game α) (ε : ℝ) (density : g.DensityLe ε) (nonneg : 0 ≤ ε) :
    ∀ (q : ℕ) (log : Log α), (restore g q log).CostLe (q * ε) := by
  intro q
  induction q with
  | zero => intro log p; show (0 : ℝ) ≤ _; simp
  | succ q ih =>
      intro log p
      show (answer (fun log' => restore g q log') log (Pos.node g p)).CostLe _
      have node := density.node p
      generalize Pos.node g p = n at node
      cases n with
      | draw C bad next =>
          refine ⟨q * ε, ?_, fun c => ih _⟩
          push_cast
          linarith [node.1]
      | pure b => exact (ih log).mono (by push_cast; linarith)
      | recv M next => exact (ih log).mono (by push_cast; linarith)

/-- A prover with `q` oracle queries names an answered path that ends outside `Q` with probability
at most `q · ε`, where `ε` bounds the bad-set density of each node. -/
theorem restore_sound (g : Game α) (s : Strategy (Option α)) (Q : α → Prop) (ε : ℝ)
    (avoids : g.Avoids Q) (density : g.DensityLe ε) (nonneg : 0 ≤ ε) (q : ℕ) :
    prob s (restore g q []) (fun r => ¬ ∀ a, r = some a → Q a) ≤ q * ε :=
  prob_not_le s _ _ _ (restore_avoids g Q avoids q [] (by simp [Keeps]))
    (restore_cost g ε density nonneg q [])

end NightstreamFPrime.Spec.Compression.Game
