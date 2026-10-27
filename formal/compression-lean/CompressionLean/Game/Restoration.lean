import CompressionLean.Game.Game

/-!
Owns the parts of the state-restoration experiment over a game: the
positions of a game (`Pos`), the log of answers (`Log`), one query
(`answer`), the verifier's walk of an output path (`walk`, `walk_sound`) and
the density bound of the bad sets (`DensityLe`). `Compose.lean` builds the
experiment over two games from them.

The experiment models a Fiat–Shamir prover. The prover names any node of the
game, as often as it wants; a draw node answers with a fresh uniform
challenge, and the log records the node and the child that the answer leads
to. At the end the prover outputs one path. The verifier walks the path and
accepts only if each draw step is a logged answer at that node. The log is
keyed by node, not by transcript prefix, so the prover may also reuse an
answer at two prefixes that reach the same node: this gives it more power,
not less.

If the game's error is at most `ε` (`CostLe`), every bad set has density at
most `ε`, so `q` queries leave the claim with probability at most `q·ε`
(`restorePair_sound`).

Does not own: the step from a hash to this experiment. When the transcript
encoding is injective on prefixes, a random-oracle prover with `Q` calls that
give at most `k` challenges each is a prover here with at most `k·Q + R`
queries (`R`, the draws on one path).
-/

namespace CompressionLean.Game

variable {α : Type}

/-- A position in `g`: a path from the root that stops at some node. -/
def Pos : Game α → Type
  | pure _ => Unit
  | draw C _ next => Option (Σ c : C, Pos (next c))
  | recv M next => Option (Σ m : M, Pos (next m))

instance Pos.nonempty (g : Game α) : Nonempty (Pos g) := by
  cases g with
  | pure a => exact ⟨()⟩
  | draw C bad next => exact ⟨(none : Option (Σ c : C, Pos (next c)))⟩
  | recv M next => exact ⟨(none : Option (Σ m : M, Pos (next m)))⟩

/-- The node at a position. -/
def Pos.node : (g : Game α) → Pos g → Game α
  | pure a, _ => pure a
  | g@(draw _ _ _), none => g
  | draw _ _ next, some ⟨c, p⟩ => Pos.node (next c) p
  | g@(recv _ _), none => g
  | recv _ next, some ⟨m, p⟩ => Pos.node (next m) p

/-- The log of answers: a node and the child its answer leads to. -/
abbrev Log (α : Type) := List (Game α × Game α)

/-- One query at node `n`: a draw node answers with a fresh challenge, and
the log records the step. Other nodes answer nothing. -/
def answer {β : Type} (k : Log α → Game β) (log : Log α) : Game α → Game β
  | n@(draw C bad next) => .draw C bad fun c => k ((n, next c) :: log)
  | _ => k log

open Classical in
/-- The verifier's walk of an output path: each draw step must be logged. -/
noncomputable def walk (log : Log α) : (g : Game α) → Pos g → Option α
  | pure a, _ => some a
  | draw _ _ _, none => none
  | g@(draw _ _ next), some ⟨c, p⟩ => if (g, next c) ∈ log then walk log (next c) p else none
  | recv _ _, none => none
  | recv _ next, some ⟨m, p⟩ => walk log (next m) p

/-- Every logged step keeps a node that avoids its bad sets. -/
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

/-- Every bad set of `g`, at every node, has density at most `ε`. -/
def DensityLe : Game α → ℝ → Prop
  | pure _, _ => True
  | draw C bad next, ε => (bad.card : ℝ) / Fintype.card C ≤ ε ∧ ∀ c, DensityLe (next c) ε
  | recv _ next, ε => ∀ m, DensityLe (next m) ε

theorem DensityLe.mono {g : Game α} {ε δ : ℝ} (h : g.DensityLe ε) (le : ε ≤ δ) : g.DensityLe δ := by
  induction g with
  | pure a => trivial
  | draw C bad next ih => exact ⟨h.1.trans le, fun c => ih c (h.2 c)⟩
  | recv M next ih => exact fun m => ih m (h m)

/-- An error bound bounds every bad set's density. -/
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
