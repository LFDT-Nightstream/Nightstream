import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Gadgets.Poseidon2.Permutation

/-!
Owns one quadratic-extension squeeze for the Poseidon2 duplex gadget. It
reads lane zero, permutes, reads lane zero again, then permutes a second time.
It defines no protocol labels or transcript schedule.
-/

namespace NightstreamFPrime.Gadgets.Poseidon2.Duplex.Squeeze

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2

abbrev EState := Layer.EState
abbrev FState := Layer.FState

abbrev firstPermutation (start : Nat) (state : EState) :=
  Permutation.compile start state Permutation.schedule

abbrev secondPermutation (start : Nat) (state : EState) :=
  Permutation.compile
    (start + (firstPermutation start state).recipes.length)
    (firstPermutation start state).output Permutation.schedule

structure Program where
  recipes : List Expr
  sample : KExpr
  output : EState

def compile (start : Nat) (state : EState) : Program where
  recipes := (firstPermutation start state).recipes ++
    (secondPermutation start state).recipes
  sample :=
    ⟨state 0, (firstPermutation start state).output 0⟩
  output := (secondPermutation start state).output

theorem compile_recipes_eq (start : Nat) (state : EState) :
    (compile start state).recipes =
      (firstPermutation start state).recipes ++
        (secondPermutation start state).recipes := rfl

theorem compile_sample_eq (start : Nat) (state : EState) :
    (compile start state).sample =
      ⟨state 0, (firstPermutation start state).output 0⟩ := rfl

theorem compile_output_apply (start : Nat) (state : EState)
    (lane : Fin 8) :
    (compile start state).output lane =
      (secondPermutation start state).output lane := rfl

theorem first_recipes_length (start : Nat) (state : EState) :
    (firstPermutation start state).recipes.length = 592 :=
  Permutation.compile_schedule_recipe_count start state

theorem second_recipes_length (start : Nat) (state : EState) :
    (secondPermutation start state).recipes.length = 592 :=
  Permutation.compile_schedule_recipe_count _ _

theorem compile_recipes_length (start : Nat) (state : EState) :
    (compile start state).recipes.length = 1184 := by
  rw [compile_recipes_eq, List.length_append, first_recipes_length,
    second_recipes_length]

def referenceSample (state : Spec.Poseidon2.State) : K :=
  ⟨state.getD 0 0, (Spec.Poseidon2.permute state).getD 0 0⟩

def referenceState (state : Spec.Poseidon2.State) :
    Spec.Poseidon2.State :=
  Spec.Poseidon2.permute (Spec.Poseidon2.permute state)

theorem compile_sound (env : Env) (start : Nat) (state : EState)
    (rows : ConstraintsHold env
      (recipeConstraints start (compile start state).recipes)) :
    (compile start state).sample.eval env =
        referenceSample (List.ofFn (Layer.evalState env state)) ∧
      List.ofFn (Layer.evalState env (compile start state).output) =
        referenceState (List.ofFn (Layer.evalState env state)) := by
  have splitRows :
      ConstraintsHold env
          (recipeConstraints start (firstPermutation start state).recipes) ∧
        ConstraintsHold env
          (recipeConstraints
            (start + (firstPermutation start state).recipes.length)
            (secondPermutation start state).recipes) := by
    change ConstraintsHold env (recipeConstraints start
      ((firstPermutation start state).recipes ++
        (secondPermutation start state).recipes)) at rows
    rw [Permutation.recipeConstraints_append] at rows
    exact (Permutation.constraintsHold_append env _ _).mp rows
  have firstSound := Permutation.compile_schedule_sound env start state
    splitRows.1
  have secondSound := Permutation.compile_schedule_sound env
    (start + (firstPermutation start state).recipes.length)
    (firstPermutation start state).output splitRows.2
  constructor
  · have secondCoordinate := congrArg
      (fun values : List F => values.getD 0 0) firstSound
    rw [compile_sample_eq]
    unfold KExpr.eval referenceSample
    apply congrArg₂ K.mk
    · simp [Layer.evalState, List.ofFn_succ]
    · simpa [Layer.evalState, List.ofFn_succ] using secondCoordinate
  · calc
      List.ofFn (Layer.evalState env (compile start state).output) =
          List.ofFn (Layer.evalState env
            (secondPermutation start state).output) := by
        apply congrArg List.ofFn
        funext lane
        exact congrArg (Expr.eval env)
          (compile_output_apply start state lane)
      _ = Spec.Poseidon2.permute
            (List.ofFn (Layer.evalState env
              (firstPermutation start state).output)) := secondSound
      _ = referenceState (List.ofFn (Layer.evalState env state)) := by
        unfold referenceState
        exact congrArg Spec.Poseidon2.permute firstSound

theorem compile_causal (start : Nat) (state : EState)
    (stateBelow : ∀ lane, (state lane).VarsBelow start) :
    RecipesCausal start (compile start state).recipes := by
  have firstCausal := Permutation.compile_schedule_causal start state
    stateBelow
  have firstOutputBelow : ∀ lane,
      ((firstPermutation start state).output lane).VarsBelow
        (start + (firstPermutation start state).recipes.length) := by
    intro lane
    exact Permutation.compile_output_varsBelow start state
      Permutation.schedule stateBelow lane
  have secondCausal := Permutation.compile_schedule_causal
    (start + (firstPermutation start state).recipes.length)
    (firstPermutation start state).output firstOutputBelow
  change RecipesCausal start
    ((firstPermutation start state).recipes ++
      (secondPermutation start state).recipes)
  exact Permutation.recipesCausal_append_causal start _ _ firstCausal
    secondCausal

theorem compile_output_below (start : Nat) (state : EState)
    (stateBelow : ∀ lane, (state lane).VarsBelow start) (lane : Fin 8) :
    ((compile start state).output lane).VarsBelow
      (start + (compile start state).recipes.length) := by
  have firstOutputBelow : ∀ current,
      ((firstPermutation start state).output current).VarsBelow
        (start + (firstPermutation start state).recipes.length) := by
    intro current
    exact Permutation.compile_output_varsBelow start state
      Permutation.schedule stateBelow current
  have secondOutputBelow := Permutation.compile_output_varsBelow
    (start + 592) (firstPermutation start state).output
    Permutation.schedule (by
      intro current
      simpa only [first_recipes_length] using firstOutputBelow current) lane
  have secondLength := Permutation.compile_schedule_recipe_count
    (start + 592) (firstPermutation start state).output
  change ((secondPermutation start state).output lane).VarsBelow _
  rw [compile_recipes_length]
  unfold secondPermutation
  rw [first_recipes_length]
  rw [secondLength] at secondOutputBelow
  exact secondOutputBelow

theorem compile_sample_below (start : Nat) (state : EState)
    (stateBelow : ∀ lane, (state lane).VarsBelow start) :
    ((compile start state).sample.c0.VarsBelow
        (start + (compile start state).recipes.length)) ∧
      ((compile start state).sample.c1.VarsBelow
        (start + (compile start state).recipes.length)) := by
  constructor
  · rw [compile_sample_eq, compile_recipes_length]
    exact Expr.VarsBelow.mono _ (stateBelow 0) (by omega)
  · have firstOutput := Permutation.compile_output_varsBelow start state
      Permutation.schedule stateBelow (0 : Fin 8)
    rw [compile_sample_eq, compile_recipes_length]
    rw [first_recipes_length] at firstOutput
    exact Expr.VarsBelow.mono _ firstOutput (by omega)

end NightstreamFPrime.Gadgets.Poseidon2.Duplex.Squeeze
