import Nightstream.SuperNeo.CheckPlan

/-!
Focused regressions for the generic verifier-check plan calculus.

| Assurance phase | Family | Regression |
|---|---|---|
| exactness | two independent checks | full acceptance is equivalent to the target conjunction |
| necessity | remove either check | a concrete one-coordinate forgery is accepted |
| plan hygiene | duplicate names | `without` removes every occurrence of one family |
-/

namespace tests.CheckPlan

open Nightstream.SuperNeo.CheckPlan

inductive Family where
  | left
  | right
deriving DecidableEq

structure Input where
  left : Bool
  right : Bool

def semantics : Family -> Input -> Prop
  | .left, input => input.left = true
  | .right, input => input.right = true

def target (input : Input) : Prop :=
  input.left = true /\ input.right = true

def fullPlan : List Family := [.left, .right]

theorem fullExact : Exact semantics target fullPlan := by
  intro input
  constructor
  · intro accepted
    exact ⟨accepted .left (by simp [fullPlan]),
      accepted .right (by simp [fullPlan])⟩
  · rintro ⟨left, right⟩ check member
    cases check
    · exact left
    · exact right

theorem leftNecessary :
    NecessaryForSoundness semantics target fullPlan .left := by
  refine ⟨⟨false, true⟩, ?_, ?_⟩
  · intro check member
    cases check <;> simp [fullPlan, without, semantics] at member ⊢
  · simp [target]

theorem rightNecessary :
    NecessaryForSoundness semantics target fullPlan .right := by
  refine ⟨⟨true, false⟩, ?_, ?_⟩
  · intro check member
    cases check <;> simp [fullPlan, without, semantics] at member ⊢
  · simp [target]

theorem fullInclusionMinimal :
    InclusionMinimalSound semantics target fullPlan := by
  apply inclusionMinimalSound_of_witnesses
  · exact (exact_iff_sound_and_complete.mp fullExact).1
  · intro check _member
    cases check
    · exact leftNecessary
    · exact rightNecessary

example :
    without [Family.left, Family.right, Family.left] Family.left =
      [Family.right] := by
  decide

end tests.CheckPlan
