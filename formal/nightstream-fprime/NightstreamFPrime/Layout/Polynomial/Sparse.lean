import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Gadgets.Polynomial.Sparse

/-!
Owns the physical footprint of the owned sparse quadratic-extension polynomial
evaluator. Every stored product cell and both result cells are one rank-one
row, so the evaluator needs no lowering cell.

The proof follows the symbolic recipe constructors. It does not inspect a
physical column number or evaluate an emitted circuit package.
-/

namespace NightstreamFPrime.Layout.Polynomial.Sparse

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Gadgets.Polynomial.Sparse
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

private theorem storeProducts_direct :
    ∀ (start : Nat) (accumulated : KExpr) (rest : List KExpr),
      KAffine accumulated → (∀ factor ∈ rest, KAffine factor) →
      R1CS.RecipesDirect start
          (Owned.storeProducts start accumulated rest).recipes ∧
        KAffine (Owned.storeProducts start accumulated rest).output
  | _, _, [], accumulatedAffine, _ => ⟨trivial, accumulatedAffine⟩
  | start, accumulated, factor :: rest, accumulatedAffine, restAffine => by
      have tail := storeProducts_direct (start + 3)
        (NightstreamFPrime.Gadgets.Polynomial.Horner.productAt start) rest
        ⟨R1CS.isAffine_var _, R1CS.isAffine_var _⟩
        (fun current member => restAffine current (by simp [member]))
      refine ⟨?_, tail.2⟩
      exact R1CS.recipesDirect_append start _ _
        (mulRecipes_direct start accumulated factor accumulatedAffine
          (restAffine factor (by simp)))
        (by simpa using tail.1)

private theorem compileMonomial_direct {matrixCount : Nat} (start : Nat)
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr)
    (pointAffine : ∀ index, KAffine (point index)) :
    R1CS.RecipesDirect start (Owned.compileMonomial start monomial point).recipes ∧
      KAffine (Owned.compileMonomial start monomial point).output := by
  have factorsAffine : ∀ factor ∈ Owned.factors monomial point, KAffine factor := by
    intro factor member
    simp only [Owned.factors, List.mem_flatMap, List.mem_replicate] at member
    rcases member with ⟨index, _, _, rfl⟩
    exact pointAffine index
  unfold Owned.compileMonomial
  cases equals : Owned.factors monomial point with
  | nil => exact ⟨trivial, R1CS.isAffine_const _, R1CS.isAffine_const _⟩
  | cons first rest =>
      rw [equals] at factorsAffine
      have firstAffine := factorsAffine first (by simp)
      exact storeProducts_direct start _ rest
        ⟨R1CS.IsAffine.add (R1CS.IsAffine.const_mul _ firstAffine.1)
            (R1CS.IsAffine.const_mul _ firstAffine.2),
          R1CS.IsAffine.add (R1CS.IsAffine.const_mul _ firstAffine.2)
            (R1CS.IsAffine.const_mul _ firstAffine.1)⟩
        (fun factor member => factorsAffine factor (by simp [member]))

private theorem compileTerms_direct {matrixCount : Nat}
    (point : Fin matrixCount → KExpr)
    (pointAffine : ∀ index, KAffine (point index)) :
    ∀ (start : Nat) (sum : KExpr) (terms : List (Monomial K matrixCount)),
      KAffine sum →
      R1CS.RecipesDirect start (Owned.compileTerms point start sum terms).recipes ∧
        KAffine (Owned.compileTerms point start sum terms).output
  | _, _, [], sumAffine => ⟨trivial, sumAffine⟩
  | start, sum, monomial :: rest, sumAffine => by
      let term := Owned.compileMonomial start monomial point
      have termDirect := compileMonomial_direct start monomial point
        pointAffine
      have tail := compileTerms_direct point pointAffine
        (start + term.recipes.length) (KExpr.add sum term.output) rest
        ⟨R1CS.IsAffine.add sumAffine.1 termDirect.2.1,
          R1CS.IsAffine.add sumAffine.2 termDirect.2.2⟩
      exact ⟨R1CS.recipesDirect_append start _ _ termDirect.1 tail.1, tail.2⟩

/-- Every stored product and both result cells are one rank-one row. -/
theorem recipes_direct {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Owned.Interface matrixCount) (offset : Nat)
    (pointAffine : ∀ index, KAffine (interface.point offset index)) :
    R1CS.RecipesDirect offset (Owned.recipes polynomial interface offset) := by
  have terms := compileTerms_direct (interface.point offset) pointAffine
    offset KExpr.zero polynomial.terms
    ⟨R1CS.isAffine_const _, R1CS.isAffine_const _⟩
  exact R1CS.recipesDirect_append offset _ _ terms.1
    ⟨R1CS.IsDirectRecipe.of_affine _ terms.2.1,
      R1CS.IsDirectRecipe.of_affine _ terms.2.2, trivial⟩

theorem ownedCircuit_totalFreshCount {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Owned.Interface matrixCount) (offset : Nat)
    (pointAffine : ∀ index, KAffine (interface.point offset index)) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Owned.circuit polynomial interface).main offset)) = 0 := by
  change R1CS.totalFreshCount
    (flatConstraints (Owned.opsAt polynomial interface offset)) = 0
  rw [Owned.flatConstraints_opsAt]
  exact R1CS.recipeConstraints_totalFreshCount offset _
    (recipes_direct polynomial interface offset pointAffine)

theorem ownedCircuit_totalRowCount {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Owned.Interface matrixCount) (offset : Nat)
    (pointAffine : ∀ index, KAffine (interface.point offset index)) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Owned.circuit polynomial interface).main offset)) =
      Owned.productCount polynomial + 2 := by
  change R1CS.totalRowCount
    (flatConstraints (Owned.opsAt polynomial interface offset)) = _
  rw [Owned.flatConstraints_opsAt,
    R1CS.recipeConstraints_totalRowCount offset _
      (recipes_direct polynomial interface offset pointAffine),
    Owned.recipes_length]

end NightstreamFPrime.Layout.Polynomial.Sparse
