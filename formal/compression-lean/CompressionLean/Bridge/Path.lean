import CompressionLean.Layer1.Restoration

/-!
Owns the paths of a game. A path is a position (`Pos`, the output type of
the state-restoration experiment) that ends in a result. `Pos.draws` lists
the challenges it drew, in order and with their types; `Pos.Avoids` says
that each of them was off its bad set. `Reaches g ds a`: some path of `g`
draws `ds` and ends in `a`.

An avoiding path of a game whose avoiding paths end in `Q` ends in `Q`
(`Pos.sound`). So a path breaks the claim only by a draw in a bad set: the
event that `prob_not_le` and `restore_sound` bound.
-/

namespace CompressionLean.Layer1.Game

variable {α β : Type}

/-- The challenges a position drew, in order, each with its type. -/
def Pos.draws : (g : Game α) → Pos g → List (Σ C : Type, C)
  | pure _, _ => []
  | draw _ _ _, none => []
  | draw C _ next, some ⟨c, p⟩ => ⟨C, c⟩ :: Pos.draws (next c) p
  | recv _ _, none => []
  | recv _ next, some ⟨m, p⟩ => Pos.draws (next m) p

/-- Every draw of a position is off its bad set. -/
def Pos.Avoids : (g : Game α) → Pos g → Prop
  | pure _, _ => True
  | draw _ _ _, none => True
  | draw _ bad next, some ⟨c, p⟩ => c ∉ bad ∧ Pos.Avoids (next c) p
  | recv _ _, none => True
  | recv _ next, some ⟨m, p⟩ => Pos.Avoids (next m) p

/-- Some path of `g` draws `ds` and ends in `a`. -/
def Reaches (g : Game α) (ds : List (Σ C : Type, C)) (a : α) : Prop :=
  ∃ p : Pos g, Pos.node g p = .pure a ∧ Pos.draws g p = ds

theorem Reaches.pure (a : α) : Reaches (.pure a) [] a := ⟨(), rfl, rfl⟩

theorem Reaches.draw {C : Type} [Fintype C] [Nonempty C] {bad : Finset C} {next : C → Game α} (c : C)
    {ds : List (Σ C : Type, C)} {a : α} (h : Reaches (next c) ds a) :
    Reaches (.draw C bad next) (⟨C, c⟩ :: ds) a := by
  obtain ⟨p, node, draws⟩ := h
  exact ⟨some ⟨c, p⟩, node, by rw [Pos.draws, draws]⟩

theorem Reaches.recv {M : Type} [Nonempty M] {next : M → Game α} (m : M) {ds : List (Σ C : Type, C)} {a : α}
    (h : Reaches (next m) ds a) : Reaches (.recv M next) ds a := by
  obtain ⟨p, node, draws⟩ := h
  exact ⟨some ⟨m, p⟩, node, draws⟩

theorem Reaches.bind {g : Game α} {f : α → Game β} {ds₁ ds₂ : List (Σ C : Type, C)} {a : α} {b : β}
    (first : Reaches g ds₁ a) (second : Reaches (f a) ds₂ b) : Reaches (Game.bind g f) (ds₁ ++ ds₂) b := by
  induction g generalizing ds₁ with
  | pure a' =>
      obtain ⟨p, node, draws⟩ := first
      cases node
      cases draws
      exact second
  | draw C bad next ih =>
      obtain ⟨p, node, draws⟩ := first
      match p, node, draws with
      | some ⟨c, p'⟩, node, draws =>
          rw [← draws, Pos.draws, List.cons_append]
          exact Reaches.draw c (ih c ⟨p', node, rfl⟩)
  | recv M next ih =>
      obtain ⟨p, node, draws⟩ := first
      match p, node, draws with
      | some ⟨m, p'⟩, node, draws => exact Reaches.recv m (ih m ⟨p', node, draws⟩)

theorem Reaches.congr {g : Game α} {ds ds' : List (Σ C : Type, C)} {a : α} (h : Reaches g ds a) (same : ds = ds') :
    Reaches g ds' a := same ▸ h

/-- An avoiding path of a game that avoids `Q` ends in `Q`. -/
theorem Pos.sound {g : Game α} {Q : α → Prop} (avoids : g.Avoids Q) :
    ∀ (p : Pos g) (a : α), Pos.Avoids g p → Pos.node g p = .pure a → Q a := by
  induction g with
  | pure b =>
      intro p a _ node
      cases node
      exact avoids
  | draw C bad next ih =>
      intro p a clean node
      match p, clean, node with
      | some ⟨c, p'⟩, clean, node => exact ih c (avoids c clean.1) p' a clean.2 node
  | recv M next ih =>
      intro p a clean node
      match p, clean, node with
      | some ⟨m, p'⟩, clean, node => exact ih m (avoids m) p' a clean node

end CompressionLean.Layer1.Game
