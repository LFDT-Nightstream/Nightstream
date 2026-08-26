import NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned
import NightstreamFPrime.Layout.Poseidon2

/-!
Owns physical lowering for one verifier-owned Poseidon2 permutation.

Every permutation recipe is affine or one rank-one product of affine values.
The optimized lowering therefore adds no column and emits one physical row per
logical recipe.
-/

namespace NightstreamFPrime.Layout.Poseidon2.PermutationOwned

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout.Poseidon2

namespace Owned

abbrev Interface :=
  NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.Interface
abbrev program :=
  NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.program
abbrev operations :=
  NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.operations
abbrev circuit :=
  NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.circuit
abbrev Assumptions :=
  NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.Assumptions
abbrev SpecHolds :=
  NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.SpecHolds
abbrev soundness :=
  NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.soundness
abbrev completeness :=
  NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.completeness
abbrev localLength_eq :=
  NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.localLength_eq
abbrev flatConstraints_operations :=
  NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.flatConstraints_operations

end Owned

structure InputsAffine (interface : Owned.Interface) (offset : Nat) : Prop where
  initialState : StateAffine (interface.initialState offset)

private theorem recipesDirect (interface : Owned.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.RecipesDirect offset (Owned.program interface offset).recipes := by
  unfold Owned.program
  exact compile_schedule_direct offset (interface.initialState offset)
    inputs.initialState

def logicalConstraints (interface : Owned.Interface) (offset : Nat) : List Expr :=
  flatConstraints (Owned.operations interface offset)

theorem totalFreshCount_eq (interface : Owned.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalFreshCount (logicalConstraints interface offset) = 0 := by
  unfold logicalConstraints
  rw [Owned.flatConstraints_operations]
  exact R1CS.recipeConstraints_totalFreshCount offset _
    (recipesDirect interface offset inputs)

theorem totalRowCount_eq (interface : Owned.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalRowCount (logicalConstraints interface offset) = 592 := by
  unfold logicalConstraints
  rw [Owned.flatConstraints_operations]
  rw [R1CS.recipeConstraints_totalRowCount]
  · exact NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.program_recipes_length
      interface offset
  · exact recipesDirect interface offset inputs

def footprint (interface : Owned.Interface)
    (inputs : ∀ offset, InputsAffine interface offset) :
    R1CS.CircuitFootprint (Owned.circuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 592
  freshColumnCount_eq := by
    intro offset
    change R1CS.totalFreshCount (logicalConstraints interface offset) = 0
    exact totalFreshCount_eq interface offset (inputs offset)
  physicalRowCount_eq := by
    intro offset
    change R1CS.totalRowCount (logicalConstraints interface offset) = 592
    exact totalRowCount_eq interface offset (inputs offset)

def plan (interface : Owned.Interface) (offset : Nat) : R1CS.LoweringPlan where
  constraints := logicalConstraints interface offset
  firstFresh := offset + 592

def PhysicalHolds (interface : Owned.Interface) (offset : Nat)
    (env : Env) : Prop :=
  R1CS.RowsHold env (plan interface offset).rows

theorem physical_implies_spec (interface : Owned.Interface) (offset : Nat)
    (env : Env) (assumptions : Owned.Assumptions interface offset env)
    (physical : PhysicalHolds interface offset env) :
    Owned.SpecHolds interface offset env := by
  apply Owned.soundness interface env offset assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold env (logicalConstraints interface offset)
  exact R1CS.LoweringPlan.sound (plan interface offset) env physical

theorem physical_complete (interface : Owned.Interface) (offset : Nat)
    (env : Env) (inputs : InputsAffine interface offset)
    (assumptions : Owned.Assumptions interface offset env)
    (specification : Owned.SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset 592 ∧
      PhysicalHolds interface offset completed := by
  rcases Owned.completeness interface env offset assumptions specification with
    ⟨completed, agrees, logical⟩
  refine ⟨completed, ?_, ?_⟩
  · rw [Owned.localLength_eq] at agrees
    exact agrees
  · apply R1CS.LoweringPlan.complete_of_noFresh
    · unfold plan logicalConstraints
      rw [Owned.flatConstraints_operations]
      exact R1CS.recipeConstraints_noFresh offset _
        (recipesDirect interface offset inputs)
    · exact logical

end NightstreamFPrime.Layout.Poseidon2.PermutationOwned
