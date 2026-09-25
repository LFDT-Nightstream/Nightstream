import NightstreamFPrime.Gadgets.Polynomial.HornerSupport
import NightstreamFPrime.Gadgets.Polynomial.Sparse

/-!
Owns variable-support propagation for the sparse constraint-polynomial
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

theorem pow_supported (value : KExpr) (allowed : Nat → Prop)
    (support : Horner.KSupported value allowed) : ∀ exponent,
    Horner.KSupported (pow value exponent) allowed
  | 0 => Horner.KSupported.one allowed
  | exponent + 1 =>
      Horner.KSupported.mul (pow_supported value allowed support exponent)
        support

theorem multiplyPower_supported (accumulated value : KExpr)
    (exponent : Nat) (allowed : Nat → Prop)
    (accumulatedSupport : Horner.KSupported accumulated allowed)
    (valueSupport : Horner.KSupported value allowed) :
    Horner.KSupported (multiplyPower accumulated value exponent) allowed := by
  by_cases zero : exponent = 0
  · simp [multiplyPower, zero, accumulatedSupport]
  · simp only [multiplyPower, zero, if_false]
    exact Horner.KSupported.mul accumulatedSupport
      (pow_supported value allowed valueSupport exponent)

private theorem monomialFold_supported {matrixCount : Nat}
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) (allowed : Nat → Prop)
    (pointSupport : ∀ index, Horner.KSupported (point index) allowed) :
    ∀ (indices : List (Fin matrixCount)) (initial : KExpr),
      Horner.KSupported initial allowed →
      Horner.KSupported
        (indices.foldl
          (fun accumulated index => multiplyPower accumulated (point index)
            (monomial.exponents index)) initial) allowed
  | [], _, initialSupport => initialSupport
  | index :: indices, initial, initialSupport => by
      apply monomialFold_supported monomial point allowed pointSupport indices
      exact multiplyPower_supported initial (point index)
        (monomial.exponents index) allowed initialSupport (pointSupport index)

theorem evaluateMonomial_supported {matrixCount : Nat}
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) (allowed : Nat → Prop)
    (pointSupport : ∀ index, Horner.KSupported (point index) allowed) :
    Horner.KSupported (evaluateMonomial monomial point) allowed := by
  apply monomialFold_supported monomial point allowed pointSupport
  exact ⟨trivial, trivial⟩

private theorem polynomialFold_supported {matrixCount : Nat}
    (point : Fin matrixCount → KExpr) (allowed : Nat → Prop)
    (pointSupport : ∀ index, Horner.KSupported (point index) allowed) :
    ∀ (terms : List (Monomial K matrixCount)) (initial : KExpr),
      Horner.KSupported initial allowed →
      Horner.KSupported
        (terms.foldl
          (fun accumulated monomial =>
            KExpr.add accumulated (evaluateMonomial monomial point))
          initial) allowed
  | [], _, initialSupport => initialSupport
  | monomial :: terms, initial, initialSupport => by
      apply polynomialFold_supported point allowed pointSupport terms
      exact Horner.KSupported.add initialSupport
        (evaluateMonomial_supported monomial point allowed pointSupport)

theorem evaluate_supported {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (point : Fin matrixCount → KExpr) (allowed : Nat → Prop)
    (pointSupport : ∀ index, Horner.KSupported (point index) allowed) :
    Horner.KSupported (evaluate polynomial point) allowed := by
  apply polynomialFold_supported point allowed pointSupport
  exact Horner.KSupported.zero allowed

namespace Owned

/-- Exact support propagation through the owned two-row sparse evaluator. -/
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
  have expressionSupport := Sparse.evaluate_supported polynomial
    (interface.point offset) allowed pointSupport
  have recipesSupported : ∀ recipe ∈ recipes polynomial interface offset,
      recipe.VarsSatisfy allowed := by
    intro recipe member
    simp only [recipes, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact expressionSupport.1
    · exact expressionSupport.2
  change ∀ expression ∈ recipeConstraints offset
      (recipes polynomial interface offset),
    expression.VarsSatisfy allowed
  apply Horner.recipeConstraints_varsSatisfy offset
    (recipes polynomial interface offset) allowed recipesSupported
  intro index indexBound
  apply localSupport (offset + index)
  · omega
  · rw [localLength_eq polynomial interface offset]
    simp [recipes] at indexBound
    omega

/-- The owned sparse result is the exact two-variable local output. -/
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
  unfold output Horner.KSupported
  simp only [Expr.VarsSatisfy]
  constructor
  · apply localSupport offset (by omega)
    rw [localLength_eq polynomial interface offset]
    omega
  · apply localSupport (offset + 1) (by omega)
    rw [localLength_eq polynomial interface offset]
    omega

end Owned

end NightstreamFPrime.Gadgets.Polynomial.Sparse
