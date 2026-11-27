import NightstreamFPrime.Spec.Compression.Game.Game

/-! One run of a game (`Game.Runs`): the game ends
in a result when each draw takes the next challenge of a path and each receipt takes the next
message. `Runs` is a function of the path (`runs_unique`), and `runs_bind` joins the run of a
game with the run of its continuation. -/

namespace NightstreamFPrime.Spec.Compression

namespace Game

/-- One step of a path: the challenge of a draw or the message of a receipt, with its type. -/
inductive Move : Type 1
  | draw {C : Type} (challenge : C)
  | recv {M : Type} (message : M)

variable {α : Type}

/-- The game ends in `result` when each draw takes the next challenge of `path`, each receipt
takes the next message, and no step of `path` is left at the end. -/
def Runs : Game α → List Move → α → Prop
  | pure a, [], result => a = result
  | draw C _ next, Move.draw (C := C') c :: path, result =>
      ∃ same : C' = C, Runs (next (cast same c)) path result
  | recv M next, Move.recv (M := M') m :: path, result =>
      ∃ same : M' = M, Runs (next (cast same m)) path result
  | _, _, _ => False

theorem runs_unique {g : Game α} {path : List Move} {a b : α} (first : g.Runs path a)
    (second : g.Runs path b) : a = b := by
  induction g generalizing path with
  | pure x =>
      cases path with
      | nil => exact first.symm.trans second
      | cons m path => cases m <;> exact first.elim
  | draw C bad next step =>
      cases path with
      | nil => exact first.elim
      | cons m path =>
          cases m with
          | draw c =>
              obtain ⟨same, first⟩ := first
              obtain ⟨_, second⟩ := second
              exact step _ first second
          | recv m => exact first.elim
  | recv M next step =>
      cases path with
      | nil => exact first.elim
      | cons m path =>
          cases m with
          | draw c => exact first.elim
          | recv m =>
              obtain ⟨same, first⟩ := first
              obtain ⟨_, second⟩ := second
              exact step _ first second

/-- A run of `g` to `a` and then a run of `f a` is a run of `g.bind f`. -/
theorem runs_bind {β : Type} {g : Game α} {f : α → Game β} {path rest : List Move} {a : α}
    {b : β} (first : g.Runs path a) (second : (f a).Runs rest b) :
    (g.bind f).Runs (path ++ rest) b := by
  induction g generalizing path with
  | pure x =>
      cases path with
      | nil =>
          have equal : x = a := first
          subst equal
          exact second
      | cons m path => cases m <;> exact first.elim
  | draw C bad next step =>
      cases path with
      | nil => exact first.elim
      | cons m path =>
          cases m with
          | draw c =>
              obtain ⟨same, first⟩ := first
              exact ⟨same, step _ first⟩
          | recv m => exact first.elim
  | recv M next step =>
      cases path with
      | nil => exact first.elim
      | cons m path =>
          cases m with
          | draw c => exact first.elim
          | recv m =>
              obtain ⟨same, first⟩ := first
              exact ⟨same, step _ first⟩

theorem runs_bind_end {β : Type} {g : Game α} {f : α → Game β} {path : List Move} {a : α}
    {b : β} (first : g.Runs path a) (second : (f a).Runs [] b) : (g.bind f).Runs path b := by
  simpa using runs_bind first second

end Game

end NightstreamFPrime.Spec.Compression
