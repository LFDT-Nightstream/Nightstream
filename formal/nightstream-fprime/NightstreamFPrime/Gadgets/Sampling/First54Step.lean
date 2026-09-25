import NightstreamFPrime.Circuit.StraightLine

/-!
Owns one transition of the fixed first-54 selection position machine.

The caller supplies one Boolean accepted bit and one one-hot position vector
with slots `0..54`. The child owns the next vector. An accepted candidate
moves the active slot right; a rejected candidate keeps it; slot 54 is
absorbing. This child does not own candidate decoding or output values.
-/

namespace NightstreamFPrime.Gadgets.Sampling.First54Step

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

def slotCount : Nat := 55
def fullSlot : Nat := 54

structure Interface where
  accepted : Nat → Expr
  prior : Nat → Fin slotCount → Expr

def output (offset : Nat) (slot : Fin slotCount) : Expr :=
  Expr.var (offset + slot.val)

def previousSlot (slot : Fin slotCount) (positive : 0 < slot.val) :
    Fin slotCount :=
  ⟨slot.val - 1, by
    have bounded := slot.isLt
    omega⟩

def update (accepted : F) (prior : Fin slotCount → F)
    (slot : Fin slotCount) : F :=
  if first : slot.val = 0 then
    prior slot * (1 - accepted)
  else if full : slot.val = fullSlot then
    prior slot + prior (previousSlot slot (by omega)) * accepted
  else
    prior slot * (1 - accepted) +
      prior (previousSlot slot (by omega)) * accepted

def recipe (interface : Interface) (offset : Nat)
    (slot : Fin slotCount) : Expr :=
  let accepted := interface.accepted offset
  let prior := interface.prior offset
  if first : slot.val = 0 then
    prior slot * (1 - accepted)
  else if full : slot.val = fullSlot then
    prior slot + prior (previousSlot slot (by omega)) * accepted
  else
    prior slot * (1 - accepted) +
      prior (previousSlot slot (by omega)) * accepted

def recipes (interface : Interface) (offset : Nat) : List Expr :=
  List.ofFn (recipe interface offset)

def operations (interface : Interface) (offset : Nat) : List Op :=
  [.witness (WitnessBatch.arithmetic offset (recipes interface offset))]

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + slotCount, operations interface offset)

structure Assumptions (interface : Interface) (offset : Nat)
    (env : Env) : Prop where
  acceptedBelow : (interface.accepted offset).VarsBelow offset
  priorBelow : ∀ slot, (interface.prior offset slot).VarsBelow offset
  acceptedBoolean : (interface.accepted offset).eval env = 0 ∨
    (interface.accepted offset).eval env = 1

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  ∀ slot,
    (output offset slot).eval env =
      update ((interface.accepted offset).eval env)
        (fun current => (interface.prior offset current).eval env) slot

private theorem recipe_eval (interface : Interface) (offset : Nat)
    (env : Env) (slot : Fin slotCount) :
    (recipe interface offset slot).eval env =
      update ((interface.accepted offset).eval env)
        (fun current => (interface.prior offset current).eval env) slot := by
  have evalOne : (1 : Expr).eval env = (1 : F) := rfl
  unfold recipe update
  by_cases first : slot.val = 0
  · simp only [dif_pos first, Expr.eval_hmul, Expr.eval_sub, evalOne]
  · by_cases full : slot.val = fullSlot
    · simp only [dif_neg first, dif_pos full, Expr.eval_hadd,
        Expr.eval_hmul]
    · simp only [dif_neg first, dif_neg full, Expr.eval_hadd,
        Expr.eval_hmul, Expr.eval_sub, evalOne]

@[simp] theorem recipes_length (interface : Interface) (offset : Nat) :
    (recipes interface offset).length = slotCount := by
  simp [recipes]

private theorem recipe_varsBelow (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env)
    (slot : Fin slotCount) :
    (recipe interface offset slot).VarsBelow offset := by
  unfold recipe
  split
  · exact Expr.VarsBelow.mul _ _ _ (assumptions.priorBelow slot)
      (Expr.VarsBelow.sub _ _ _ trivial assumptions.acceptedBelow)
  · split
    · exact Expr.VarsBelow.add _ _ _ (assumptions.priorBelow slot)
        (Expr.VarsBelow.mul _ _ _
          (assumptions.priorBelow (previousSlot slot (by omega)))
          assumptions.acceptedBelow)
    · exact Expr.VarsBelow.add _ _ _
        (Expr.VarsBelow.mul _ _ _ (assumptions.priorBelow slot)
          (Expr.VarsBelow.sub _ _ _ trivial assumptions.acceptedBelow))
        (Expr.VarsBelow.mul _ _ _
          (assumptions.priorBelow (previousSlot slot (by omega)))
          assumptions.acceptedBelow)

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
    localLength (operations interface offset) = slotCount := by
  simp [operations, localLength, Op.localLength, recipes_length]

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (operations interface offset)).length = slotCount := by
  rw [flatConstraints_operations, recipeConstraints_length, recipes_length]

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsBelow (offset + slotCount) := by
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
  let slot : Fin slotCount := ⟨index, by
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
  privateCount := fun _ => slotCount
  rowCount := fun _ => slotCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := flatConstraints_length interface
  soundness := soundness interface
  completeness := completeness interface

end NightstreamFPrime.Gadgets.Sampling.First54Step
