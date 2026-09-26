import NightstreamFPrime.Layout.PiRlcWideSampler.Challenges
import NightstreamFPrime.Lifecycle.PiRLC.Wide.ProjectedBatch

/-! Temporary digit words keep the existing R1CS convolution shape. Their
binding equations are reconstructed from the retained bits and add no row
or coordinate to BatchPlan. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.DigitProjection

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Lifecycle.PiRLC.Wide

/-- Inlining the bit expression adds three multiplication nodes to every
use in the existing generic ring-convolution lowering. -/
theorem challenge_mulCounts (offset : Nat) (source : Fin Batch.sourceCount) (position : Fin ringDegree) :
    R1CS.mulCount (Batch.outputChallenge offset source position) = 4 ∧
      R1CS.mulCount (ProjectedBatch.outputChallenge offset source position) = 1 := by
  constructor <;> rfl

theorem challenge_variable (offset : Nat) (source : Fin Batch.sourceCount) (position : Fin ringDegree) :
    ∃ column, ProjectedBatch.outputChallenge offset source position = Expr.var column - 2 := by
  exact ⟨ProjectedBatch.wordsOffset offset + (Fin.encodeProd (source, position)).val, rfl⟩

/-- All temporary words are exact expression views of the checked bits. -/
theorem reconstructs (interface : Batch.Interface) (offset : Nat) (env : Env)
    (spec : ProjectedBatch.SpecHolds interface offset env) (index : Fin DigitWords.count) :
    env (ProjectedBatch.wordsOffset offset + index.val) = (DigitWords.recipe offset index).eval env := spec.words index

theorem reconstructed_rows (offset : Nat) (env : Env)
    (words : ∀ index : Fin DigitWords.count,
      env (ProjectedBatch.wordsOffset offset + index.val) = (DigitWords.recipe offset index).eval env) :
    holdsFlat env (DigitWords.operations offset (ProjectedBatch.wordsOffset offset)) := by
  change ConstraintsHold env (flatConstraints _)
  rw [DigitWords.constraints_eq]
  apply recipeConstraints_hold_of_values
  intro index below
  have bound : index < DigitWords.count := by simpa only [DigitWords.recipes_length] using below
  simpa only [DigitWords.recipes, List.get_ofFn] using! words ⟨index, bound⟩

theorem recipe_affine (offset : Nat) (index : Fin DigitWords.count) :
    R1CS.IsAffine (DigitWords.recipe offset index) := by
  unfold DigitWords.recipe WideReduction.Program.outputWord WideReduction.linearExpr
  apply R1CS.IsAffine.add (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var _))
  apply R1CS.IsAffine.add (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var _))
  exact R1CS.IsAffine.add (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var _)) (R1CS.isAffine_const _)

private theorem affine_direct (recipes : List Expr) (affine : ∀ recipe ∈ recipes, R1CS.IsAffine recipe)
    (offset : Nat) : R1CS.RecipesDirect offset recipes := by
  induction recipes generalizing offset with
  | nil => trivial
  | cons recipe rest ih =>
      exact ⟨R1CS.IsDirectRecipe.of_affine offset (affine recipe (by simp)),
        ih (fun recipe member => affine recipe (by simp [member])) (offset + 1)⟩

private theorem affine_counts {n : Nat} (recipes : Fin n → Expr) (offset : Nat)
    (affine : ∀ index, R1CS.IsAffine (recipes index)) :
    R1CS.totalFreshCount (recipeConstraints offset (List.ofFn recipes)) = 0 ∧
      R1CS.totalRowCount (recipeConstraints offset (List.ofFn recipes)) = n := by
  have direct := affine_direct (List.ofFn recipes) (by
    intro recipe member; obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member; exact affine index) offset
  exact ⟨R1CS.recipeConstraints_totalFreshCount offset _ direct,
    by simpa only [List.length_ofFn] using R1CS.recipeConstraints_totalRowCount offset _ direct⟩

theorem r1cs_counts (samplerOffset offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (DigitWords.operations samplerOffset offset)) = 0 ∧
      R1CS.totalRowCount (flatConstraints (DigitWords.operations samplerOffset offset)) = 918 := by
  rw [DigitWords.constraints_eq, ← DigitWords.count_eq]
  exact affine_counts (DigitWords.recipe samplerOffset) offset (recipe_affine samplerOffset)

theorem projection_cost : DigitWords.count = 918 := DigitWords.count_eq

end NightstreamFPrime.Layout.PiRlcWideSampler.DigitProjection
