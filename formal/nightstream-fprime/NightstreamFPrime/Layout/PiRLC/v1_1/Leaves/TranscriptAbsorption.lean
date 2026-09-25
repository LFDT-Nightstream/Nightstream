import NightstreamFPrime.Layout.Poseidon2.Duplex
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption

/-!
Owns the physical lowering of one Π_RLC scalar-domain entry.

The lifecycle leaf absorbs `[4, coordinate]` with one Poseidon2 permutation.
All 592 recipes lower directly, so this owner adds no lowering column and no
boundary-copy row. Physical and logical satisfaction are proved in both
directions through the generic R1CS lowering boundary.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.TranscriptAbsorption

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout.Poseidon2
open NightstreamFPrime.Layout.Poseidon2.Duplex

namespace Leaf

abbrev Interface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.Interface
abbrev constantWords :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.constantWords
abbrev actions :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.actions
abbrev ownedInterface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.ownedInterface
abbrev circuit :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.circuit
abbrev Assumptions :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.Assumptions
abbrev SpecHolds :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.SpecHolds
abbrev soundness :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.soundness
abbrev completeness :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.completeness
abbrev localLength_eq :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.localLength_eq

end Leaf

structure InputsAffine (interface : Leaf.Interface) (offset : Nat) : Prop where
  initialState : StateAffine (interface.initialState offset)

private theorem constantWords_affine (words : List NightstreamFPrime.Spec.F) :
    ListAffine (Leaf.constantWords words) := by
  intro expression member
  rcases List.mem_map.mp member with ⟨word, _, rfl⟩
  exact R1CS.isAffine_const word

theorem actions_affine (coordinate : Nat) :
    ActionsAffine (Leaf.actions coordinate) := by
  apply ActionsAffine.cons
  · exact constantWords_affine _
  · intro action member
    simp at member

private theorem recipesDirect (interface : Leaf.Interface)
    (coordinate offset : Nat) (inputs : InputsAffine interface offset) :
    R1CS.RecipesDirect offset
      (Formal.Owned.program
        (Leaf.ownedInterface interface coordinate) offset).recipes := by
  exact compile_recipes_direct offset (interface.initialState offset)
    (Leaf.actions coordinate) inputs.initialState
    (actions_affine coordinate)

private theorem allAssertions_eq_nil (interface : Leaf.Interface)
    (coordinate offset : Nat) :
    Formal.Owned.allAssertions
      (Leaf.ownedInterface interface coordinate) offset = [] := by
  rfl

private theorem flatConstraints_eq (interface : Leaf.Interface)
    (coordinate offset : Nat) :
    flatConstraints
        (Circuit.ops (Leaf.circuit interface coordinate).main offset) =
      recipeConstraints offset
        (Formal.Owned.program
          (Leaf.ownedInterface interface coordinate) offset).recipes := by
  change flatConstraints
      (Formal.Owned.opsAt
        (Leaf.ownedInterface interface coordinate) offset) = _
  rw [Formal.Owned.flatConstraints_opsAt,
    allAssertions_eq_nil, List.append_nil]

private theorem noFresh (interface : Leaf.Interface)
    (coordinate offset : Nat) (inputs : InputsAffine interface offset) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (Leaf.circuit interface coordinate).main offset),
      R1CS.constraintFreshCount expression = 0 := by
  rw [flatConstraints_eq]
  exact R1CS.recipeConstraints_noFresh offset _
    (recipesDirect interface coordinate offset inputs)

def footprint (interface : Leaf.Interface) (coordinate : Nat)
    (inputs : ∀ offset, InputsAffine interface offset) :
    R1CS.CircuitFootprint (Leaf.circuit interface coordinate) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 592
  freshColumnCount_eq := by
    intro offset
    rw [flatConstraints_eq]
    exact R1CS.recipeConstraints_totalFreshCount offset _
      (recipesDirect interface coordinate offset (inputs offset))
  physicalRowCount_eq := by
    intro offset
    rw [flatConstraints_eq]
    rw [R1CS.recipeConstraints_totalRowCount]
    · have logical := Leaf.localLength_eq interface coordinate offset
      change localLength
        (Formal.Owned.opsAt
          (Leaf.ownedInterface interface coordinate) offset) = 592 at logical
      rw [Formal.Owned.opsAt_localLength] at logical
      exact logical
    · exact recipesDirect interface coordinate offset (inputs offset)

theorem freshColumnCount_eq (interface : Leaf.Interface)
    (coordinate : Nat) (inputs : ∀ offset, InputsAffine interface offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints
      (Circuit.ops (Leaf.circuit interface coordinate).main offset)) = 0 :=
  (footprint interface coordinate inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq (interface : Leaf.Interface)
    (coordinate : Nat) (inputs : ∀ offset, InputsAffine interface offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints
      (Circuit.ops (Leaf.circuit interface coordinate).main offset)) = 592 :=
  (footprint interface coordinate inputs).physicalRowCount_eq offset

def loweringPlan (interface : Leaf.Interface) (coordinate offset firstFresh : Nat) :
    R1CS.LoweringPlan where
  constraints := flatConstraints
    (Circuit.ops (Leaf.circuit interface coordinate).main offset)
  firstFresh := firstFresh

/-- Physical rows imply the exact verifier-owned transcript entry. -/
theorem physical_implies_spec (interface : Leaf.Interface)
    (coordinate offset firstFresh : Nat) (env : Env)
    (assumptions : Leaf.Assumptions interface offset env)
    (rows : R1CS.RowsHold env
      (loweringPlan interface coordinate offset firstFresh).rows) :
    Leaf.SpecHolds interface coordinate offset env := by
  have logical := R1CS.lowerConstraints_sound env
    (flatConstraints
      (Circuit.ops (Leaf.circuit interface coordinate).main offset))
    firstFresh rows
  apply Leaf.soundness interface coordinate env offset assumptions
  exact holdsFlat_implies_holds env _ logical

/-- Honest logical execution also satisfies the exact physical rows. -/
theorem physical_complete (interface : Leaf.Interface)
    (coordinate offset firstFresh : Nat) (env : Env)
    (inputs : InputsAffine interface offset)
    (assumptions : Leaf.Assumptions interface offset env)
    (specification : Leaf.SpecHolds interface coordinate offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength
          (Circuit.ops (Leaf.circuit interface coordinate).main offset)) ∧
      R1CS.RowsHold completed
        (loweringPlan interface coordinate offset firstFresh).rows := by
  rcases Leaf.completeness interface coordinate env offset assumptions
      specification with ⟨completed, agrees, logical⟩
  refine ⟨completed, agrees, ?_⟩
  exact R1CS.lowerConstraints_complete_of_noFresh completed
    (flatConstraints
      (Circuit.ops (Leaf.circuit interface coordinate).main offset))
    firstFresh (noFresh interface coordinate offset inputs) logical

end NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.TranscriptAbsorption
