import CompressionLean.Game.Game

namespace CompressionLean.Game

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

end CompressionLean.Game
