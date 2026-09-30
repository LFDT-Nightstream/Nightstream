import NightstreamFPrime.Gadgets.Polynomial.HornerSupport
import NightstreamFPrime.Gadgets.Polynomial.Sparse

/-!
Owns variable-support propagation for the owned sparse constraint-polynomial
evaluator. The polynomial and evaluation order remain unchanged.
-/

namespace NightstreamFPrime.Gadgets.Polynomial.Sparse

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

namespace Owned

private theorem storeProducts_supported (allowed : Nat → Prop) :
    ∀ (start : Nat) (accumulated : KExpr) (rest : List KExpr),
      Horner.KSupported accumulated allowed →
      (∀ factor ∈ rest, Horner.KSupported factor allowed) →
      (∀ index, start ≤ index → index < start + 3 * rest.length →
        allowed index) →
      (∀ recipe ∈ (storeProducts start accumulated rest).recipes,
        recipe.VarsSatisfy allowed) ∧
        Horner.KSupported (storeProducts start accumulated rest).output allowed
  | _, _, [], accumulatedSupport, _, _ =>
      ⟨by simp [storeProducts], accumulatedSupport⟩
  | start, accumulated, factor :: rest, accumulatedSupport, restSupport,
      localCells => by
      have cell : ∀ index, start ≤ index → index < start + 3 →
          allowed index := fun index lower upper =>
        localCells index lower (by simp only [List.length_cons]; omega)
      have productSupport : Horner.KSupported (Horner.productAt start)
          allowed :=
        ⟨cell (start + 1) (by omega) (by omega),
          cell (start + 2) (by omega) (by omega)⟩
      have tail := storeProducts_supported allowed (start + 3)
        (Horner.productAt start) rest productSupport
        (fun current member => restSupport current (by simp [member]))
        (fun index lower upper => localCells index (by omega)
          (by simp only [List.length_cons]; omega))
      refine ⟨?_, tail.2⟩
      intro recipe member
      simp only [storeProducts, List.mem_append] at member
      rcases member with product | rest
      · exact Horner.mulRecipes_supported start accumulated factor allowed
          accumulatedSupport (restSupport factor (by simp))
          (cell start (by omega) (by omega))
          (cell (start + 1) (by omega) (by omega)) recipe product
      · exact tail.1 recipe rest

private theorem compileMonomial_supported {matrixCount : Nat}
    (allowed : Nat → Prop) (start : Nat)
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr)
    (pointSupport : ∀ index, Horner.KSupported (point index) allowed)
    (localCells : ∀ index, start ≤ index →
      index < start + (compileMonomial start monomial point).recipes.length →
      allowed index) :
    (∀ recipe ∈ (compileMonomial start monomial point).recipes,
      recipe.VarsSatisfy allowed) ∧
      Horner.KSupported (compileMonomial start monomial point).output
        allowed := by
  have factorsSupport : ∀ factor ∈ factors monomial point,
      Horner.KSupported factor allowed := by
    intro factor member
    simp only [factors, List.mem_flatMap, List.mem_replicate] at member
    rcases member with ⟨index, _, _, rfl⟩
    exact pointSupport index
  unfold compileMonomial at localCells ⊢
  cases equals : factors monomial point with
  | nil => exact ⟨by simp, trivial, trivial⟩
  | cons first rest =>
      rw [equals] at localCells factorsSupport
      have firstSupport := factorsSupport first (by simp)
      apply storeProducts_supported allowed start
        (scale monomial.coefficient first) rest
        ⟨⟨⟨trivial, firstSupport.1⟩, ⟨trivial, firstSupport.2⟩⟩,
          ⟨⟨trivial, firstSupport.2⟩, ⟨trivial, firstSupport.1⟩⟩⟩
        (fun factor member => factorsSupport factor (by simp [member]))
      intro index lower upper
      apply localCells index lower
      rw [storeProducts_length]
      exact upper

private theorem compileTerms_supported {matrixCount : Nat}
    (allowed : Nat → Prop) (point : Fin matrixCount → KExpr)
    (pointSupport : ∀ index, Horner.KSupported (point index) allowed) :
    ∀ (start : Nat) (sum : KExpr) (terms : List (Monomial K matrixCount)),
      Horner.KSupported sum allowed →
      (∀ index, start ≤ index →
        index < start + (compileTerms point start sum terms).recipes.length →
        allowed index) →
      (∀ recipe ∈ (compileTerms point start sum terms).recipes,
        recipe.VarsSatisfy allowed) ∧
        Horner.KSupported (compileTerms point start sum terms).output allowed
  | _, _, [], sumSupport, _ => ⟨by simp [compileTerms], sumSupport⟩
  | start, sum, monomial :: rest, sumSupport, localCells => by
      have termSupport := compileMonomial_supported allowed start monomial
        point pointSupport (fun index lower upper => localCells index lower (by
          simp only [compileTerms, List.length_append]
          exact Nat.lt_of_lt_of_le upper (by omega)))
      have tail := compileTerms_supported allowed point pointSupport
        (start + (compileMonomial start monomial point).recipes.length)
        (KExpr.add sum (compileMonomial start monomial point).output) rest
        (Horner.KSupported.add sumSupport termSupport.2)
        (fun index lower upper => localCells index (by omega) (by
          simp only [compileTerms, List.length_append]
          omega))
      refine ⟨?_, tail.2⟩
      intro recipe member
      simp only [compileTerms, List.mem_append] at member
      rcases member with first | second
      · exact termSupport.1 recipe first
      · exact tail.1 recipe second

/-- Exact support propagation through the owned sparse evaluator. -/
theorem flatConstraints_varsSatisfy {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat)
    (allowed : Nat → Prop)
    (pointSupport : ∀ index,
      Horner.KSupported (interface.point offset index) allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + localLength
        (Circuit.ops (circuit polynomial interface).main offset) →
      allowed index) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (circuit polynomial interface).main offset),
      expression.VarsSatisfy allowed := by
  rw [localLength_eq] at localSupport
  have terms := compileTerms_supported allowed (interface.point offset)
    pointSupport offset KExpr.zero polynomial.terms
    (Horner.KSupported.zero allowed)
    (fun index lower upper => localSupport index lower (by
      change index < offset + (program polynomial interface offset).recipes.length
        at upper
      rw [program_length] at upper
      omega))
  have recipesSupported : ∀ recipe ∈ recipes polynomial interface offset,
      recipe.VarsSatisfy allowed := by
    intro recipe member
    simp only [recipes, List.mem_append, List.mem_cons, List.not_mem_nil,
      or_false] at member
    rcases member with stored | rfl | rfl
    · exact terms.1 recipe stored
    · exact terms.2.1
    · exact terms.2.2
  change ∀ expression ∈ flatConstraints (opsAt polynomial interface offset),
    expression.VarsSatisfy allowed
  rw [flatConstraints_opsAt]
  apply Horner.recipeConstraints_varsSatisfy offset
    (recipes polynomial interface offset) allowed recipesSupported
  intro index indexBound
  apply localSupport (offset + index)
  · omega
  · rw [recipes_length] at indexBound
    omega

/-- The owned sparse result is the exact two-cell localCells output. -/
theorem output_varsSatisfy {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (interface : Interface matrixCount) (offset : Nat)
    (allowed : Nat → Prop)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + localLength
        (Circuit.ops (circuit polynomial interface).main offset) →
      allowed index) :
    Horner.KSupported (output polynomial interface offset) allowed := by
  rw [localLength_eq] at localSupport
  exact ⟨localSupport _ (by omega) (by omega),
    localSupport _ (by omega) (by omega)⟩

end Owned

end NightstreamFPrime.Gadgets.Polynomial.Sparse
