import NightstreamFPrime.Circuit.StraightLine
import NightstreamFPrime.Gadgets.Poseidon2.Permutation

/-!
Owns one verifier-driven Poseidon2 permutation with no caller-supplied expected
output.

The caller supplies only an input state below the call offset. The child
allocates the canonical 592-recipe permutation program and exposes its final
eight expressions. Its specification identifies those expressions with the
executable Poseidon2 permutation of the input state.
-/

namespace NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

abbrev EState := Layer.EState

structure Interface where
  initialState : Nat → EState

def program (interface : Interface) (offset : Nat) : Permutation.Program :=
  Permutation.compile offset (interface.initialState offset)
    Permutation.schedule

set_option maxRecDepth 100000 in -- fixed-size: one Poseidon2 permutation, not artifact data
set_option maxHeartbeats 2000000 in -- fixed-size: one Poseidon2 permutation, not artifact data
theorem program_recipes_length (interface : Interface) (offset : Nat) :
    (program interface offset).recipes.length = 592 := by
  unfold program
  exact Permutation.compile_schedule_recipe_count offset
    (interface.initialState offset)

def output (_interface : Interface) (offset : Nat) : EState :=
  Permutation.scheduleOutput offset

def operations (interface : Interface) (offset : Nat) : List Op :=
  [.witness (WitnessBatch.arithmetic offset (program interface offset).recipes)]

set_option maxRecDepth 100000 in -- fixed-size: one Poseidon2 permutation, not artifact data
theorem flatConstraints_operations (interface : Interface) (offset : Nat) :
    flatConstraints (operations interface offset) =
      recipeConstraints offset (program interface offset).recipes := by
  rfl

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + (program interface offset).recipes.length,
    operations interface offset)

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  ∀ lane, (interface.initialState offset lane).VarsBelow offset

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  List.ofFn (Layer.evalState env (output interface offset)) =
    Spec.Poseidon2.permute
      (List.ofFn (Layer.evalState env (interface.initialState offset)))

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (operations interface offset)) :
    SpecHolds interface offset env := by
  have recipeRows := rows
    (.witness (WitnessBatch.arithmetic offset
      (program interface offset).recipes)) (by simp [operations])
  have computed := Permutation.compile_schedule_sound env offset
    (interface.initialState offset) recipeRows
  unfold SpecHolds
  rw [output, Permutation.scheduleOutput_eq_compile]
  exact computed

set_option maxRecDepth 100000 in -- fixed-size: one Poseidon2 permutation, not artifact data
theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (operations interface offset) = 592 := by
  unfold operations localLength
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    Op.localLength, WitnessBatch.arithmetic_outputLength, Nat.add_zero]
  exact program_recipes_length interface offset

set_option maxRecDepth 100000 in -- fixed-size: one Poseidon2 permutation, not artifact data
theorem flatConstraints_length_eq (interface : Interface) (offset : Nat) :
    (flatConstraints (operations interface offset)).length = 592 := by
  rw [flatConstraints_operations]
  rw [recipeConstraints_length]
  exact program_recipes_length interface offset

set_option maxRecDepth 100000 in -- fixed-size: one Poseidon2 permutation, not artifact data
theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (assumptions : ∀ lane,
      (interface.initialState offset lane).VarsBelow offset) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsBelow (offset + 592) := by
  rw [flatConstraints_operations]
  have causal := Permutation.compile_schedule_causal offset
    (interface.initialState offset) assumptions
  have scope := recipeConstraints_varsBelow_of_causal offset
    (program interface offset).recipes causal
  rw [show (program interface offset).recipes.length = 592 by
    exact Permutation.compile_schedule_recipe_count offset
      (interface.initialState offset)] at scope
  exact scope

set_option maxRecDepth 100000 in -- fixed-size: one Poseidon2 permutation, not artifact data
theorem complete (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) := by
  let completed := executeRecipes env offset (program interface offset).recipes
  have causal := Permutation.compile_schedule_causal offset
    (interface.initialState offset) assumptions
  refine ⟨completed, ?_, ?_⟩
  · have agrees := executeRecipes_agreesOutside env offset
      (program interface offset).recipes
    rw [program_recipes_length] at agrees
    rw [localLength_eq]
    exact agrees
  · unfold holdsFlat
    rw [flatConstraints_operations]
    exact executeRecipes_holds_recipeConstraints env offset
      (program interface offset).recipes causal

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (_specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) :=
  complete interface env offset assumptions

set_option maxRecDepth 100000 in -- fixed-size: one Poseidon2 permutation, not artifact data
theorem output_varsBelow (interface : Interface) (offset : Nat)
    (_assumptions : ∀ lane,
      (interface.initialState offset lane).VarsBelow offset) :
    ∀ lane, (output interface offset lane).VarsBelow (offset + 592) := by
  intro lane
  simp [output, Permutation.scheduleOutput, Permutation.freshState,
    Expr.VarsBelow]
  omega

def circuit (interface : Interface) : FormalCircuit :=
  { main := main interface
    assumptions := Assumptions interface
    spec := SpecHolds interface
    privateCount := fun _ => 592
    rowCount := fun _ => 592
    privateCount_eq := by
      intro offset
      exact localLength_eq interface offset
    rowCount_eq := by
      intro offset
      exact flatConstraints_length_eq interface offset
    soundness := by
      intro env offset assumptions rows
      exact soundness interface env offset assumptions rows
    completeness := by
      intro env offset assumptions specification
      exact completeness interface env offset assumptions specification }

end NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned
