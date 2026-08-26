import NightstreamFPrime.Circuit.StraightLine
import NightstreamFPrime.Gadgets.Sampling.First54Step

/-!
Owns one value transition of the fixed first-54 selector.

For slot `j`, the child keeps the prior value and adds the current symbol only
when the prior one-hot position is `j` and the candidate is accepted. It does
not own the position transition, candidate decoding, or shortfall assertion.
-/

namespace NightstreamFPrime.Gadgets.Sampling.First54ValueStep

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

def outputCount : Nat := 54

structure Interface where
  accepted : Nat → Expr
  symbol : Nat → Expr
  priorPosition : Nat → Fin First54Step.slotCount → Expr
  priorOutput : Nat → Fin outputCount → Expr

def positionSlot (slot : Fin outputCount) : Fin First54Step.slotCount :=
  ⟨slot.val, lt_trans slot.isLt (by decide)⟩

def update (accepted symbol : F)
    (priorPosition : Fin First54Step.slotCount → F)
    (priorOutput : Fin outputCount → F)
    (slot : Fin outputCount) : F :=
  priorOutput slot +
    priorPosition (positionSlot slot) * accepted * symbol

def recipe (interface : Interface) (offset : Nat)
    (slot : Fin outputCount) : Expr :=
  interface.priorOutput offset slot +
    interface.priorPosition offset (positionSlot slot) *
      interface.accepted offset * interface.symbol offset

def recipes (interface : Interface) (offset : Nat) : List Expr :=
  List.ofFn (recipe interface offset)

def output (offset : Nat) (slot : Fin outputCount) : Expr :=
  Expr.var (offset + slot.val)

def operations (interface : Interface) (offset : Nat) : List Op :=
  [.witness (WitnessBatch.arithmetic offset (recipes interface offset))]

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + outputCount, operations interface offset)

structure Assumptions (interface : Interface) (offset : Nat)
    (_env : Env) : Prop where
  acceptedBelow : (interface.accepted offset).VarsBelow offset
  symbolBelow : (interface.symbol offset).VarsBelow offset
  priorPositionBelow : ∀ slot,
    (interface.priorPosition offset slot).VarsBelow offset
  priorOutputBelow : ∀ slot,
    (interface.priorOutput offset slot).VarsBelow offset

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  ∀ slot,
    (output offset slot).eval env =
      update ((interface.accepted offset).eval env)
        ((interface.symbol offset).eval env)
        (fun current => (interface.priorPosition offset current).eval env)
        (fun current => (interface.priorOutput offset current).eval env) slot

private theorem recipe_eval (interface : Interface) (offset : Nat)
    (env : Env) (slot : Fin outputCount) :
    (recipe interface offset slot).eval env =
      update ((interface.accepted offset).eval env)
        ((interface.symbol offset).eval env)
        (fun current => (interface.priorPosition offset current).eval env)
        (fun current => (interface.priorOutput offset current).eval env) slot := by
  rfl

@[simp] theorem recipes_length (interface : Interface) (offset : Nat) :
    (recipes interface offset).length = outputCount := by
  simp [recipes]

private theorem recipe_varsBelow (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env)
    (slot : Fin outputCount) :
    (recipe interface offset slot).VarsBelow offset := by
  unfold recipe
  exact Expr.VarsBelow.add _ _ _ (assumptions.priorOutputBelow slot)
    (Expr.VarsBelow.mul _ _ _
      (Expr.VarsBelow.mul _ _ _
        (assumptions.priorPositionBelow (positionSlot slot))
        assumptions.acceptedBelow)
      assumptions.symbolBelow)

theorem recipes_causal (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    RecipesCausal offset (recipes interface offset) := by
  apply recipesCausal_of_all_below
  intro expression member
  rcases List.mem_ofFn.mp member with ⟨slot, rfl⟩
  exact recipe_varsBelow interface offset assumptions slot

theorem flatConstraints_operations (interface : Interface) (offset : Nat) :
    flatConstraints (operations interface offset) =
      recipeConstraints offset (recipes interface offset) := by
  rfl

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (operations interface offset) = outputCount := by
  simp [operations, localLength, Op.localLength, recipes_length]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (operations interface offset)).length = outputCount := by
  rw [flatConstraints_operations, recipeConstraints_length, recipes_length]

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsBelow (offset + outputCount) := by
  rw [flatConstraints_operations]
  have scope := recipeConstraints_varsBelow_of_causal offset
    (recipes interface offset) (recipes_causal interface offset env assumptions)
  rw [recipes_length] at scope
  exact scope

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (operations interface offset)) :
    SpecHolds interface offset env := by
  have recipeRows := rows
    (.witness (WitnessBatch.arithmetic offset (recipes interface offset)))
    (by simp [operations])
  intro slot
  have value := recipeConstraints_value env offset (recipes interface offset)
    recipeRows slot.val (by simpa [recipes_length] using slot.isLt)
  rw [show (recipes interface offset).get
      ⟨slot.val, by simpa [recipes_length] using slot.isLt⟩ =
        recipe interface offset slot by simp [recipes]] at value
  simpa [output, recipe_eval] using value

theorem holdsFlat_of_spec (interface : Interface) (env : Env) (offset : Nat)
    (specification : SpecHolds interface offset env) :
    holdsFlat env (operations interface offset) := by
  unfold holdsFlat
  rw [flatConstraints_operations]
  apply recipeConstraints_hold_of_values
  intro index bounded
  let slot : Fin outputCount := ⟨index, by
    simpa [recipes_length] using bounded⟩
  have value := specification slot
  rw [show (recipes interface offset).get ⟨index, bounded⟩ =
      recipe interface offset slot by simp [recipes, slot]]
  simpa [output, recipe_eval, slot] using value

theorem complete (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) := by
  let completed := executeRecipes env offset (recipes interface offset)
  refine ⟨completed, ?_, ?_⟩
  · have agrees := executeRecipes_agreesOutside env offset
      (recipes interface offset)
    rw [localLength_eq]
    simpa [recipes_length] using agrees
  · unfold holdsFlat
    rw [flatConstraints_operations]
    exact executeRecipes_holds_recipeConstraints env offset
      (recipes interface offset) (recipes_causal interface offset env assumptions)

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (_specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) :=
  complete interface env offset assumptions

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  privateCount := fun _ => outputCount
  rowCount := fun _ => outputCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := flatConstraints_length interface
  soundness := soundness interface
  completeness := completeness interface

end NightstreamFPrime.Gadgets.Sampling.First54ValueStep
