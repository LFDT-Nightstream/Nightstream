import NightstreamFPrime.Gadgets.Poseidon2.Hash
import NightstreamFPrime.Gadgets.Poseidon2.Permutation.Mark

/-! The marks of the hash compiler's permutations (SPEC.md section 4.4). `compileAbsorptions`
places one permutation per block, each from the last one's output, and `compile` adds the
padded last permutation. `absorptionMarks` and `marks` name these permutations, and the batch's
rows are their windows, in order (`compileAbsorptions_windows`, `compile_windows`). They are built
without the recipes, so the emitter pays no second compile. -/

set_option autoImplicit false

namespace NightstreamFPrime.Gadgets.Poseidon2.Hash

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

/-- The state after the absorptions, without the recipes. -/
def absorbedState : Nat → EState → List (List Expr) → EState
  | _, state, [] => state
  | start, _, _ :: rest => absorbedState (start + 1096) (Permutation.scheduleOutput start) rest

/-- The marks of the permutations of `compileAbsorptions`. -/
def absorptionMarks : Nat → EState → List (List Expr) → List PermutationMark
  | _, _, [] => []
  | start, state, block :: rest =>
      Permutation.markOf start (absorbE state block) ::
        absorptionMarks (start + 1096) (Permutation.scheduleOutput start) rest

/-- The marks of the permutations of `compile`. -/
def marks (start : Nat) (input : List Expr) : List PermutationMark :=
  let blocks := Spec.Poseidon2.fullBlocks input
  absorptionMarks start zeroE blocks ++
    [Permutation.markOf (start + blocks.length * 1096)
      (padE (Spec.Poseidon2.lastWords input) (absorbedState start zeroE blocks))]

theorem compileAbsorptions_output (start : Nat) (state : EState) (blocks : List (List Expr)) :
    (compileAbsorptions start state blocks).output = absorbedState start state blocks := by
  induction blocks generalizing start state with
  | nil => rfl
  | cons block rest ih =>
      simp only [compileAbsorptions, absorbedState]
      rw [ih, ← Permutation.scheduleOutput_eq_compile]

/-- The rows of the absorptions are the windows of their marks. -/
theorem compileAbsorptions_windows (start : Nat) (state : EState) (blocks : List (List Expr)) :
    recipeConstraints start (compileAbsorptions start state blocks).recipes =
      (absorptionMarks start state blocks).flatMap Permutation.markWindow := by
  induction blocks generalizing start state with
  | nil => rfl
  | cons block rest ih =>
      simp only [compileAbsorptions, absorptionMarks, List.flatMap_cons,
        Permutation.recipeConstraints_append, Permutation.markWindow_markOf,
        Permutation.compile_schedule_recipe_count]
      rw [← Permutation.scheduleOutput_eq_compile, ih]

/-- The rows of the hash are the windows of its marks. -/
theorem compile_windows (start : Nat) (input : List Expr) :
    recipeConstraints start (compile start input).recipes =
      (marks start input).flatMap Permutation.markWindow := by
  unfold compile marks
  dsimp only
  rw [Permutation.recipeConstraints_append, compileAbsorptions_windows, List.flatMap_append,
    compileAbsorptions_recipes_length, compileAbsorptions_output, List.flatMap_singleton,
    Permutation.markWindow_markOf]

end NightstreamFPrime.Gadgets.Poseidon2.Hash
