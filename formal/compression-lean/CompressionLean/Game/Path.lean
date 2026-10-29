import CompressionLean.Game.Restoration

namespace CompressionLean.Game

variable {α β : Type}

def Pos.draws : (g : Game α) → Pos g → List (Σ C : Type, C)
  | pure _, _ => []
  | draw _ _ _, none => []
  | draw C _ next, some ⟨c, p⟩ => ⟨C, c⟩ :: Pos.draws (next c) p
  | recv _ _, none => []
  | recv _ next, some ⟨m, p⟩ => Pos.draws (next m) p

def Pos.msgs : (g : Game α) → Pos g → List (Σ M : Type, M)
  | pure _, _ => []
  | draw _ _ _, none => []
  | draw _ _ next, some ⟨c, p⟩ => Pos.msgs (next c) p
  | recv _ _, none => []
  | recv M next, some ⟨m, p⟩ => ⟨M, m⟩ :: Pos.msgs (next m) p

theorem Pos.eq_of_msgs_draws {g : Game α} {p p' : Pos g} (msgs : Pos.msgs g p = Pos.msgs g p')
    (draws : Pos.draws g p = Pos.draws g p') : p = p' := by
  induction g with
  | pure a => rfl
  | draw C bad next ih =>
      match p, p', msgs, draws with
      | none, none, _, _ => rfl
      | none, some _, _, draws => cases draws
      | some _, none, _, draws => cases draws
      | some ⟨c, q⟩, some ⟨c', q'⟩, msgs, draws =>
          obtain ⟨head, tail⟩ := List.cons.inj draws
          cases eq_of_heq (Sigma.mk.inj head).2
          rw [ih c msgs tail]
  | recv M next ih =>
      match p, p', msgs, draws with
      | none, none, _, _ => rfl
      | none, some _, msgs, _ => cases msgs
      | some _, none, msgs, _ => cases msgs
      | some ⟨m, q⟩, some ⟨m', q'⟩, msgs, draws =>
          obtain ⟨head, tail⟩ := List.cons.inj msgs
          cases eq_of_heq (Sigma.mk.inj head).2
          rw [ih m tail draws]

def Reaches (g : Game α) (ms ds : List (Σ C : Type, C)) (a : α) : Prop :=
  ∃ p : Pos g, Pos.node g p = .pure a ∧ Pos.draws g p = ds ∧ Pos.msgs g p = ms

theorem Reaches.pure (a : α) : Reaches (.pure a) [] [] a := ⟨(), rfl, rfl, rfl⟩

theorem Reaches.draw {C : Type} [Fintype C] [Nonempty C] {bad : Finset C} {next : C → Game α} (c : C)
    {ms ds : List (Σ C : Type, C)} {a : α} (h : Reaches (next c) ms ds a) :
    Reaches (.draw C bad next) ms (⟨C, c⟩ :: ds) a := by
  obtain ⟨p, node, draws, msgs⟩ := h
  exact ⟨some ⟨c, p⟩, node, by rw [Pos.draws, draws], msgs⟩

theorem Reaches.recv {M : Type} [Nonempty M] {next : M → Game α} (m : M) {ms ds : List (Σ C : Type, C)} {a : α}
    (h : Reaches (next m) ms ds a) : Reaches (.recv M next) (⟨M, m⟩ :: ms) ds a := by
  obtain ⟨p, node, draws, msgs⟩ := h
  exact ⟨some ⟨m, p⟩, node, draws, by rw [Pos.msgs, msgs]⟩

theorem Reaches.bind {g : Game α} {f : α → Game β} {ms₁ ms₂ ds₁ ds₂ : List (Σ C : Type, C)} {a : α} {b : β}
    (first : Reaches g ms₁ ds₁ a) (second : Reaches (f a) ms₂ ds₂ b) :
    Reaches (Game.bind g f) (ms₁ ++ ms₂) (ds₁ ++ ds₂) b := by
  induction g generalizing ms₁ ds₁ with
  | pure a' =>
      obtain ⟨p, node, draws, msgs⟩ := first
      cases node
      cases draws
      cases msgs
      exact second
  | draw C bad next ih =>
      obtain ⟨p, node, draws, msgs⟩ := first
      match p, node, draws, msgs with
      | some ⟨c, p'⟩, node, draws, msgs =>
          rw [← draws, Pos.draws, List.cons_append]
          exact Reaches.draw c (ih c ⟨p', node, rfl, msgs⟩)
  | recv M next ih =>
      obtain ⟨p, node, draws, msgs⟩ := first
      match p, node, draws, msgs with
      | some ⟨m, p'⟩, node, draws, msgs =>
          rw [← msgs, Pos.msgs, List.cons_append]
          exact Reaches.recv m (ih m ⟨p', node, draws, rfl⟩)

theorem Reaches.congr {g : Game α} {ms ms' ds ds' : List (Σ C : Type, C)} {a : α} (h : Reaches g ms ds a)
    (sameMs : ms = ms') (sameDs : ds = ds') : Reaches g ms' ds' a := sameMs ▸ sameDs ▸ h

end CompressionLean.Game
