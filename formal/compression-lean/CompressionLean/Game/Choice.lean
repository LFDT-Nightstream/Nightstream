import CompressionLean.Game.Compose

/-! Two games behind one choice (CW-6.5). The prover names a side, then plays that side's game.
Option B's final program runs one layer-1 verifier per slot, each on its own relation; one
restoration game over the shrink game and this choice restores both (`restorePair`). The choice
draws nothing, so it avoids what both sides avoid (`choice_avoids`) at the larger of their
densities (`choice_density`). A position of a side is a position of the choice that ends where it
ends (`node_left`, `node_right`). -/

namespace CompressionLean.Game

variable {α β γ : Type}

/-- The prover names a side, then plays that side's game. -/
def choice (g : Game α) (h : Game β) : Game (α ⊕ β) :=
  .recv Bool fun right => cond right (bind h (pure ∘ Sum.inr)) (bind g (pure ∘ Sum.inl))

/-- The position before any move. -/
def Pos.root : (g : Game α) → Pos g
  | pure _ => ()
  | draw _ _ _ => none
  | recv _ _ => none

theorem Pos.node_root (g : Game α) : Pos.node g (Pos.root g) = g := by
  cases g <;> rfl

/-- A position of `g` in `bind g f`: after a result of `g`, the root of `f`. -/
def Pos.bind (f : α → Game β) : (g : Game α) → Pos g → Pos (Game.bind g f)
  | pure a, _ => Pos.root (f a)
  | draw _ _ _, none => none
  | draw _ _ next, some ⟨c, p⟩ => some ⟨c, Pos.bind f (next c) p⟩
  | recv _ _, none => none
  | recv _ next, some ⟨m, p⟩ => some ⟨m, Pos.bind f (next m) p⟩

theorem Pos.node_bind (f : α → Game β) : ∀ (g : Game α) (p : Pos g),
    Pos.node (Game.bind g f) (Pos.bind f g p) = Game.bind (Pos.node g p) f
  | pure a, _ => Pos.node_root (f a)
  | draw _ _ _, none => rfl
  | draw _ _ next, some ⟨c, p⟩ => Pos.node_bind f (next c) p
  | recv _ _, none => rfl
  | recv _ next, some ⟨m, p⟩ => Pos.node_bind f (next m) p

/-- Position `p` of the left side, in the choice. -/
def Pos.left (h : Game β) {g : Game α} (p : Pos g) : Pos (choice g h) :=
  some ⟨false, Pos.bind (pure ∘ Sum.inl) g p⟩

/-- Position `p` of the right side, in the choice. -/
def Pos.right (g : Game α) {h : Game β} (p : Pos h) : Pos (choice g h) :=
  some ⟨true, Pos.bind (pure ∘ Sum.inr) h p⟩

theorem node_left (h : Game β) {g : Game α} {p : Pos g} {a : α} (leaf : Pos.node g p = .pure a) :
    Pos.node (choice g h) (Pos.left h p) = .pure (.inl a) := by
  show Pos.node (Game.bind g (pure ∘ Sum.inl)) (Pos.bind _ g p) = _
  rw [Pos.node_bind, leaf]
  rfl

theorem node_right (g : Game α) {h : Game β} {p : Pos h} {b : β} (leaf : Pos.node h p = .pure b) :
    Pos.node (choice g h) (Pos.right g p) = .pure (.inr b) := by
  show Pos.node (Game.bind h (pure ∘ Sum.inr)) (Pos.bind _ h p) = _
  rw [Pos.node_bind, leaf]
  rfl

theorem choice_avoids {g : Game α} {h : Game β} {Q : α → Prop} {R : β → Prop}
    (avoidsG : g.Avoids Q) (avoidsH : h.Avoids R) : (choice g h).Avoids (Sum.elim Q R) := by
  intro right
  cases right
  · exact (avoids_bind g _ _).mpr (avoidsG.mono fun _ holds => holds)
  · exact (avoids_bind h _ _).mpr (avoidsH.mono fun _ holds => holds)

theorem DensityLe.bind_pure {g : Game α} {ε : ℝ} (f : α → β) (density : g.DensityLe ε) :
    (Game.bind g (pure ∘ f)).DensityLe ε := by
  induction g with
  | pure a => trivial
  | draw C bad next ih => exact ⟨density.1, fun c => ih c (density.2 c)⟩
  | recv M next ih => exact fun m => ih m (density m)

theorem choice_density {g : Game α} {h : Game β} {ε : ℝ} (densityG : g.DensityLe ε)
    (densityH : h.DensityLe ε) : (choice g h).DensityLe ε := by
  intro right
  cases right
  · exact densityG.bind_pure _
  · exact densityH.bind_pure _

end CompressionLean.Game
