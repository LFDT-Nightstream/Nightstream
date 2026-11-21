import NightstreamFPrime.Gadgets.Poseidon2.LiveRows
import NightstreamFPrime.Circuit.Execution
import NightstreamFPrime.Gadgets.Poseidon2.Permutation

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
    (program interface offset).recipes.length = 1096 := by
  unfold program
  exact Permutation.compile_schedule_recipe_count offset
    (interface.initialState offset)

def output (_interface : Interface) (offset : Nat) : EState :=
  Permutation.scheduleOutput offset

def operations (interface : Interface) (offset : Nat) : List Op :=
  [.witness (WitnessBatch.arithmetic offset (program interface offset).recipes),
    .mark [Permutation.markOf offset (interface.initialState offset)]]

theorem flatConstraints_operations (interface : Interface) (offset : Nat) :
    flatConstraints (operations interface offset) =
      recipeConstraints offset (program interface offset).recipes :=
  flatConstraints_singleton _

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
    localLength (operations interface offset) = 1096 := by
  unfold operations localLength
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    Op.localLength, WitnessBatch.arithmetic_outputLength, Nat.add_zero]
  exact program_recipes_length interface offset

set_option maxRecDepth 100000 in -- fixed-size: one Poseidon2 permutation, not artifact data
theorem flatConstraints_length_eq (interface : Interface) (offset : Nat) :
    (flatConstraints (operations interface offset)).length = 1096 := by
  rw [flatConstraints_operations]
  rw [recipeConstraints_length]
  exact program_recipes_length interface offset

set_option maxRecDepth 100000 in -- fixed-size: one Poseidon2 permutation, not artifact data
theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (assumptions : ∀ lane,
      (interface.initialState offset lane).VarsBelow offset) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsBelow (offset + 1096) := by
  rw [flatConstraints_operations]
  have causal := Permutation.compile_schedule_causal offset
    (interface.initialState offset) assumptions
  have scope := recipeConstraints_varsBelow_of_causal offset
    (program interface offset).recipes causal
  rw [show (program interface offset).recipes.length = 1096 by
    exact Permutation.compile_schedule_recipe_count offset
      (interface.initialState offset)] at scope
  exact scope

theorem ordered (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    Witness.Ordered offset (witnesses (operations interface offset))
      (offset + localLength (operations interface offset)) :=
  Witness.ordered_batch rfl (localLength_batch offset _ _)
    (Permutation.compile_schedule_causal offset (interface.initialState offset) assumptions)

theorem computed_holdsFlat (interface : Interface) (env : Env) (offset : Nat)
    (computed : Witness.Computed env (witnesses (operations interface offset))) :
    holdsFlat env (operations interface offset) :=
  Witness.computed_holdsFlat (by simp [operations, Op.RecipesOnly]) computed

theorem complete (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) :=
  Witness.complete (ordered interface offset env assumptions) fun completed _ =>
    computed_holdsFlat interface completed offset

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
    ∀ lane, (output interface offset lane).VarsBelow (offset + 1096) := by
  intro lane
  have laneBound : lane.val < 16 := lane.isLt
  simp [output, Permutation.scheduleOutput, Permutation.freshState,
    Expr.VarsBelow]
  omega

def circuit (interface : Interface) : FormalCircuit :=
  { main := main interface
    assumptions := Assumptions interface
    spec := SpecHolds interface
    privateCount := fun _ => 1096
    rowCount := fun _ => 1096
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

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

theorem localLength_operations (interface : Interface) (offset : Nat) :
    localLength (operations interface offset) = 1096 :=
  (localLength_batch offset _ _).trans (program_recipes_length interface offset)

theorem placed (interface : Interface) (offset : Nat) :
    Placed offset (localLength (operations interface offset)) (operations interface offset) := by
  rw [localLength_operations]
  refine ⟨fun mark member => ?_, by simp [operations, marks, Op.marks]⟩
  simp only [operations, marks, List.flatMap_cons, List.flatMap_nil, Op.marks, List.nil_append,
    List.append_nil, List.mem_singleton] at member
  subst member
  simp

theorem liveRows (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (affine : ∀ lane, (interface.initialState offset lane).Affine)
    (reads : ∀ lane, (interface.initialState offset lane).VarsSatisfy
      fun column => column < offset ∧ live column) :
    LiveRows live (operations interface offset) :=
  .batch (by simp [Permutation.markWindow_markOf, program])
    (fun mark member => by
      simp only [List.mem_singleton] at member
      subst member
      exact Permutation.markOf_lanes_facts offset _ affine reads)

/-- The permutation's outputs are live. -/
theorem output_live (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (own : OwnLive live offset (localLength (operations interface offset))
      (operations interface offset)) (lane : Fin 16) :
    (output interface offset lane).VarsSatisfy live := by
  rw [localLength_operations] at own
  have small := lane.isLt
  refine own (offset + 1080 + lane.val) (by omega) (by omega) ?_
  rintro ⟨mark, member, lower, upper⟩
  simp only [operations, marks, List.flatMap_cons, List.flatMap_nil, Op.marks, List.nil_append,
    List.append_nil, List.mem_singleton] at member
  subst member
  simp only [Permutation.markOf_start] at upper
  omega

end NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned
