import NightstreamFPrime.Export.Stage1.PermutationCompilerTransport

/-!
Owns the source-row projection from the existing Duplex action compiler to
its compact permutation invocations. The action and block lists retain their
existing compiler inputs, states, and witness starts.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PermutationCompilerTransport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1.Invocations
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1

private theorem compileBlocks_complete_of_sourceConstraints
    (phase rowStart witnessStart : Nat) (state : EState)
    (blocks : List (List Expr)) (env : Env)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (stateAffine : Poseidon2.StateAffine state)
    (blocksAffine : Poseidon2.BlocksAffine blocks)
    (rows : ConstraintsHold (Spartan.pullback env)
      (recipeConstraints witnessStart
        (Hash.compileAbsorptions witnessStart state blocks).recipes)) :
    ∀ current ∈ (compileBlocks phase rowStart witnessStart state blocks).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env := by
  induction blocks generalizing rowStart witnessStart state with
  | nil => simp only [compileBlocks, List.not_mem_nil, false_implies, implies_true]
  | cons block rest induction =>
      have blockAffine : Poseidon2.ListAffine block := blocksAffine block (by simp)
      have restAffine : Poseidon2.BlocksAffine rest := by
        intro current member
        exact blocksAffine current (List.mem_cons_of_mem _ member)
      have separated := Formal.splitRecipeRows (Spartan.pullback env) witnessStart
        (Permutation.compile witnessStart (Hash.absorbE state block)
          Permutation.schedule).recipes
        (Hash.compileAbsorptions (witnessStart + 592)
          (Permutation.compile witnessStart (Hash.absorbE state block)
            Permutation.schedule).output rest).recipes rows
      have headRows : ConstraintsHold (Spartan.pullback env)
          (recipeConstraints witnessStart
            (Permutation.compile witnessStart (Hash.absorbE state block)
              Permutation.schedule).recipes) := separated.1
      have tailRows : ConstraintsHold (Spartan.pullback env)
          (recipeConstraints (witnessStart + 592)
            (Hash.compileAbsorptions (witnessStart + 592)
              (permutationOutput witnessStart) rest).recipes) := by
        rw [permutationOutput_eq_compile]
        simpa only [Permutation.compile_schedule_recipe_count] using separated.2
      intro current member
      simp only [compileBlocks, List.mem_cons] at member
      rcases member with rfl | member
      · exact invocation_complete_of_sourceConstraints phase rowStart witnessStart
          (Hash.absorbE state block) env witnessLocal
          (Poseidon2.absorbE_affine state block stateAffine blockAffine) headRows
      · exact induction (rowStart + 592) (witnessStart + 592)
          (permutationOutput witnessStart) (by omega)
          (permutationOutput_affine witnessStart) restAffine tailRows current member

/-- The actual source recipe rows of a Duplex action list supply every
compact permutation invocation in that same list. Squeeze expectation checks
remain in the source assertion rows; no expected-value equality is assumed. -/
theorem compileActions_complete_of_sourceConstraints
    (phase rowStart witnessStart : Nat) (state : EState)
    (actions : List Action) (env : Env)
    (witnessLocal : Spartan.piCcsPhaseOffset ≤ witnessStart)
    (stateAffine : Poseidon2.StateAffine state)
    (actionsAffine : ActionsInvocationInputsAffine actions)
    (rows : ConstraintsHold (Spartan.pullback env)
      (recipeConstraints witnessStart
        (Formal.compile witnessStart state actions).recipes)) :
    ∀ current ∈ (compileActions phase rowStart witnessStart state actions).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env := by
  induction actions generalizing rowStart witnessStart state with
  | nil => simp only [compileActions, List.not_mem_nil, false_implies, implies_true]
  | cons action actions induction =>
      have tailAffine : ActionsInvocationInputsAffine actions := by
        intro shape member
        exact actionsAffine shape (by simpa only [List.map_cons, List.mem_cons] using Or.inr member)
      cases action with
      | absorb input =>
          let blocks := Hash.inputChunks input
          let traced := compileBlocks phase rowStart witnessStart state blocks
          let absorbed := Hash.compileAbsorptions witnessStart state blocks
          have inputAffine : Poseidon2.ListAffine input :=
            actionsAffine (.absorb input) (by simp only [List.map_cons, Formal.Action.shape, List.mem_cons, true_or])
          have blocksAffine := Poseidon2.inputChunks_affine input inputAffine
          have nextEq : traced.witnessNext = witnessStart + absorbed.recipes.length := by
            rw [compileBlocks_witnessNext, Hash.compileAbsorptions_recipes_length]
          have stateEq : traced.state = absorbed.output :=
            compileBlocks_state_eq phase rowStart witnessStart state blocks
          have separated := Formal.splitRecipeRows (Spartan.pullback env) witnessStart
            absorbed.recipes
            (Formal.compile (witnessStart + absorbed.recipes.length) absorbed.output actions).recipes rows
          have nextLocal : Spartan.piCcsPhaseOffset ≤ traced.witnessNext := by
            rw [nextEq]
            omega
          have nextAffine : Poseidon2.StateAffine traced.state := by
            rw [stateEq]
            exact Poseidon2.compileAbsorptions_output_affine witnessStart state blocks stateAffine blocksAffine
          have tailRows : ConstraintsHold (Spartan.pullback env)
              (recipeConstraints traced.witnessNext
                (Formal.compile traced.witnessNext traced.state actions).recipes) := by
            rw [nextEq, stateEq]
            exact separated.2
          intro current member
          change current ∈ traced.invocations ++
            (compileActions phase traced.rowNext traced.witnessNext traced.state actions).invocations at member
          rcases List.mem_append.mp member with member | member
          · exact compileBlocks_complete_of_sourceConstraints phase rowStart witnessStart state blocks env
              witnessLocal stateAffine blocksAffine separated.1 current member
          · exact induction traced.rowNext traced.witnessNext traced.state nextLocal nextAffine
              tailAffine tailRows current member
      | squeezeK expected =>
          let squeezed := Squeeze.compile witnessStart state
          have separated := Formal.splitRecipeRows (Spartan.pullback env) witnessStart
            squeezed.recipes
            (Formal.compile (witnessStart + squeezed.recipes.length) squeezed.output actions).recipes rows
          have splitSqueeze := Formal.splitRecipeRows (Spartan.pullback env) witnessStart
            (Squeeze.firstPermutation witnessStart state).recipes
            (Squeeze.secondPermutation witnessStart state).recipes separated.1
          have firstRows : ConstraintsHold (Spartan.pullback env)
              (recipeConstraints witnessStart
                (Permutation.compile witnessStart state Permutation.schedule).recipes) := splitSqueeze.1
          have secondRows : ConstraintsHold (Spartan.pullback env)
              (recipeConstraints (witnessStart + 592)
                (Permutation.compile (witnessStart + 592)
                  (permutationOutput witnessStart) Permutation.schedule).recipes) := by
            rw [permutationOutput_eq_compile]
            simpa only [Squeeze.secondPermutation, Squeeze.first_recipes_length] using splitSqueeze.2
          have tailRows : ConstraintsHold (Spartan.pullback env)
              (recipeConstraints (witnessStart + 1184)
                (Formal.compile (witnessStart + 1184)
                  (permutationOutput (witnessStart + 592)) actions).recipes) := by
            rw [squeezeOutput_eq_compile]
            simpa only [squeezed, Squeeze.compile_recipes_length] using separated.2
          intro current member
          simp only [compileActions, List.mem_cons] at member
          rcases member with rfl | rfl | member
          · exact invocation_complete_of_sourceConstraints phase rowStart witnessStart state env
              witnessLocal stateAffine firstRows
          · exact invocation_complete_of_sourceConstraints phase (rowStart + 592) (witnessStart + 592)
              (permutationOutput witnessStart) env (by omega)
              (permutationOutput_affine witnessStart) secondRows
          · exact induction (rowStart + 1184) (witnessStart + 1184)
              (permutationOutput (witnessStart + 592)) (by omega)
              (permutationOutput_affine (witnessStart + 592)) tailAffine tailRows current member

end NightstreamFPrime.Export.Stage1.PermutationCompilerTransport
