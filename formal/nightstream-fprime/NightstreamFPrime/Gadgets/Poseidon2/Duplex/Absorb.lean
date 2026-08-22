import NightstreamFPrime.Gadgets.Poseidon2.Hash

/-!
Owns the absorb-only primitive used by the Poseidon2 duplex gadget. It reuses
the hash compiler and proves equality to the existing Poseidon2 block fold.
It does not define a transcript schedule.
-/

namespace NightstreamFPrime.Gadgets.Poseidon2.Duplex.Absorb

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2

abbrev EState := Layer.EState
abbrev FState := Layer.FState

def reference (state : Spec.Poseidon2.State) (input : List F) :
    Spec.Poseidon2.State :=
  (Hash.inputChunks input).foldl Spec.Poseidon2.absorbBlock state

/-- The existing absorption compiler evaluates to the existing Poseidon2
reference fold. -/
theorem compile_sound (env : Env) (start : Nat) (state : EState)
    (input : List Expr)
    (rows : ConstraintsHold env
      (recipeConstraints start
        (Hash.compileAbsorptions start state
          (Hash.inputChunks input)).recipes)) :
    List.ofFn (Layer.evalState env
      (Hash.compileAbsorptions start state (Hash.inputChunks input)).output) =
      reference (List.ofFn (Layer.evalState env state))
        (Hash.evalList env input) := by
  have compiled := Hash.compileAbsorptions_sound env start state
    (Hash.inputChunks input) rows
  calc
    List.ofFn (Layer.evalState env
        (Hash.compileAbsorptions start state (Hash.inputChunks input)).output) =
        List.ofFn (Hash.absorbManyF (Layer.evalState env state)
          ((Hash.inputChunks input).map (Hash.evalList env))) :=
      congrArg List.ofFn compiled
    _ = ((Hash.inputChunks input).map (Hash.evalList env)).foldl
          Spec.Poseidon2.absorbBlock
          (List.ofFn (Layer.evalState env state)) :=
      Hash.absorbManyF_eq_reference _ _
    _ = reference (List.ofFn (Layer.evalState env state))
          (Hash.evalList env input) := by
      rw [Hash.inputChunks_eval]
      rfl

theorem compile_causal (start : Nat) (state : EState) (input : List Expr)
    (stateBelow : ∀ lane, (state lane).VarsBelow start)
    (inputBelow : ∀ expression ∈ input, expression.VarsBelow start) :
    RecipesCausal start
      (Hash.compileAbsorptions start state
        (Hash.inputChunks input)).recipes := by
  exact Hash.compileAbsorptions_causal start state (Hash.inputChunks input)
    stateBelow (Hash.inputChunks_below input start inputBelow)

theorem compile_output_below (start : Nat) (state : EState)
    (input : List Expr)
    (stateBelow : ∀ lane, (state lane).VarsBelow start)
    (inputBelow : ∀ expression ∈ input, expression.VarsBelow start)
    (lane : Fin 8) :
    ((Hash.compileAbsorptions start state
      (Hash.inputChunks input)).output lane).VarsBelow
        (start + (Hash.compileAbsorptions start state
          (Hash.inputChunks input)).recipes.length) := by
  exact Hash.compileAbsorptions_output_varsBelow start state
    (Hash.inputChunks input) stateBelow
    (Hash.inputChunks_below input start inputBelow) lane

end NightstreamFPrime.Gadgets.Poseidon2.Duplex.Absorb
