import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Gadgets.Polynomial.Sparse

/-!
Owns the physical multiplication-count model for the reusable sparse
quadratic-extension polynomial evaluator.

The count model follows the symbolic expression constructors. It does not
inspect a physical column number or evaluate an emitted circuit package.
-/

namespace NightstreamFPrime.Layout.Polynomial.Sparse

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

/-- Multiplication-node counts for the two base-field components. -/
structure Counts where
  c0 : Nat
  c1 : Nat
deriving DecidableEq

namespace Counts

def zero : Counts := ⟨0, 0⟩

def add (left right : Counts) : Counts :=
  ⟨left.c0 + right.c0, left.c1 + right.c1⟩

/-- Cost of one symbolic quadratic-extension multiplication. -/
def mul (left right : Counts) : Counts :=
  ⟨left.c0 + right.c0 + left.c1 + right.c1 + 3,
    left.c0 + right.c1 + left.c1 + right.c0 + 2⟩

end Counts

def expressionCounts (value : KExpr) : Counts :=
  ⟨R1CS.mulCount value.c0, R1CS.mulCount value.c1⟩

@[simp] theorem expressionCounts_zero :
    expressionCounts KExpr.zero = Counts.zero := by
  rfl

@[simp] theorem expressionCounts_one :
    expressionCounts KExpr.one = Counts.zero := by
  rfl

@[simp] theorem expressionCounts_constant (value : K) :
    expressionCounts
      (NightstreamFPrime.Gadgets.Polynomial.Sparse.constant value) =
        Counts.zero := by
  rfl

@[simp] theorem expressionCounts_add (left right : KExpr) :
    expressionCounts (KExpr.add left right) =
      Counts.add (expressionCounts left) (expressionCounts right) := by
  cases left
  cases right
  rfl

@[simp] theorem expressionCounts_mul (left right : KExpr) :
    expressionCounts (KExpr.mul left right) =
      Counts.mul (expressionCounts left) (expressionCounts right) := by
  cases left
  cases right
  simp [expressionCounts, Counts.mul, KExpr.mul, R1CS.mulCount]
  constructor <;> omega

def powCounts (value : Counts) : Nat → Counts
  | 0 => Counts.zero
  | exponent + 1 => Counts.mul (powCounts value exponent) value

theorem expressionCounts_pow (value : KExpr) : ∀ exponent,
    expressionCounts
        (NightstreamFPrime.Gadgets.Polynomial.Sparse.pow value exponent) =
      powCounts (expressionCounts value) exponent
  | 0 => rfl
  | exponent + 1 => by
      simp only [NightstreamFPrime.Gadgets.Polynomial.Sparse.pow,
        powCounts, expressionCounts_mul]
      rw [expressionCounts_pow value exponent]

def multiplyPowerCounts
    (accumulated value : Counts) (exponent : Nat) : Counts :=
  if exponent = 0 then accumulated
  else Counts.mul accumulated (powCounts value exponent)

theorem expressionCounts_multiplyPower
    (accumulated value : KExpr) (exponent : Nat) :
    expressionCounts
        (NightstreamFPrime.Gadgets.Polynomial.Sparse.multiplyPower
          accumulated value exponent) =
      multiplyPowerCounts (expressionCounts accumulated)
        (expressionCounts value) exponent := by
  by_cases zero : exponent = 0
  · simp [NightstreamFPrime.Gadgets.Polynomial.Sparse.multiplyPower,
      multiplyPowerCounts, zero]
  · simp only [NightstreamFPrime.Gadgets.Polynomial.Sparse.multiplyPower,
      multiplyPowerCounts, zero, if_false, expressionCounts_mul]
    rw [expressionCounts_pow]

def monomialCounts {Field : Type} {matrixCount : Nat}
    (monomial : Monomial Field matrixCount)
    (point : Fin matrixCount → Counts) : Counts :=
  (canonicalFinIndices matrixCount).foldl
    (fun accumulated index =>
      multiplyPowerCounts accumulated (point index)
        (monomial.exponents index))
    Counts.zero

private theorem expressionCounts_monomialFold {matrixCount : Nat}
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) :
    ∀ (indices : List (Fin matrixCount)) (initial : KExpr),
      expressionCounts
          (indices.foldl
            (fun accumulated index =>
              NightstreamFPrime.Gadgets.Polynomial.Sparse.multiplyPower
                accumulated (point index) (monomial.exponents index))
            initial) =
        indices.foldl
          (fun accumulated index =>
            multiplyPowerCounts accumulated (expressionCounts (point index))
              (monomial.exponents index))
          (expressionCounts initial)
  | [], _ => rfl
  | index :: indices, initial => by
      simp only [List.foldl_cons]
      rw [expressionCounts_monomialFold monomial point indices]
      rw [expressionCounts_multiplyPower]

theorem expressionCounts_evaluateMonomial {matrixCount : Nat}
    (monomial : Monomial K matrixCount)
    (point : Fin matrixCount → KExpr) :
    expressionCounts
        (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
          monomial point) =
      monomialCounts monomial (fun index => expressionCounts (point index)) := by
  unfold NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
    monomialCounts
  rw [expressionCounts_monomialFold]
  rfl

def polynomialCounts {Field : Type} {matrixCount : Nat}
    (polynomial : ConstraintPolynomial Field matrixCount)
    (point : Fin matrixCount → Counts) : Counts :=
  polynomial.terms.foldl
    (fun accumulated monomial =>
      Counts.add accumulated (monomialCounts monomial point))
    Counts.zero

private theorem expressionCounts_polynomialFold {matrixCount : Nat}
    (point : Fin matrixCount → KExpr) :
    ∀ (terms : List (Monomial K matrixCount)) (initial : KExpr),
      expressionCounts
          (terms.foldl
            (fun accumulated monomial =>
              KExpr.add accumulated
                (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluateMonomial
                  monomial point))
            initial) =
        terms.foldl
          (fun accumulated monomial =>
            Counts.add accumulated
              (monomialCounts monomial
                (fun index => expressionCounts (point index))))
          (expressionCounts initial)
  | [], _ => rfl
  | monomial :: terms, initial => by
      simp only [List.foldl_cons]
      rw [expressionCounts_polynomialFold point terms]
      rw [expressionCounts_add, expressionCounts_evaluateMonomial]

theorem expressionCounts_evaluate {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (point : Fin matrixCount → KExpr) :
    expressionCounts
        (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluate polynomial point) =
      polynomialCounts polynomial
        (fun index => expressionCounts (point index)) := by
  unfold NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluate polynomialCounts
  rw [expressionCounts_polynomialFold]
  rfl

def linearPolynomialCounts {Field : Type} {matrixCount : Nat}
    (polynomial : ConstraintPolynomial Field matrixCount) : Counts :=
  polynomialCounts polynomial (fun _ => Counts.zero)

theorem expressionCounts_evaluate_of_linear {matrixCount : Nat}
    (polynomial : ConstraintPolynomial K matrixCount)
    (point : Fin matrixCount → KExpr)
    (linear : ∀ index, KExprLinear (point index)) :
    expressionCounts
        (NightstreamFPrime.Gadgets.Polynomial.Sparse.evaluate polynomial point) =
      linearPolynomialCounts polynomial := by
  rw [expressionCounts_evaluate]
  unfold linearPolynomialCounts
  congr 2
  funext index
  simp [expressionCounts, Counts.zero, (linear index).c0_mulCount,
    (linear index).c1_mulCount]

theorem monomialCounts_liftMonomial
    {Base : Type}
    {matrixCount : Nat}
    (lift : Base → K)
    (monomial : Monomial Base matrixCount)
    (point : Fin matrixCount → Counts) :
    monomialCounts
        (ConstraintPolynomialLift.liftMonomial lift monomial) point =
      monomialCounts monomial point := by
  rfl

private theorem polynomialCounts_map_liftMonomial
    {Base : Type}
    {matrixCount : Nat}
    (lift : Base → K)
    (point : Fin matrixCount → Counts) :
    ∀ (terms : List (Monomial Base matrixCount)) (initial : Counts),
      (terms.map (ConstraintPolynomialLift.liftMonomial lift)).foldl
          (fun accumulated monomial =>
            Counts.add accumulated (monomialCounts monomial point)) initial =
        terms.foldl
          (fun accumulated monomial =>
            Counts.add accumulated (monomialCounts monomial point)) initial
  | [], _ => rfl
  | monomial :: terms, initial => by
      simp only [List.map_cons, List.foldl_cons,
        monomialCounts_liftMonomial]
      exact polynomialCounts_map_liftMonomial lift point terms _

theorem linearPolynomialCounts_liftConstraintPolynomial
    {Base : Type}
    {matrixCount : Nat}
    (lift : Base → K)
    (polynomial : ConstraintPolynomial Base matrixCount) :
    linearPolynomialCounts
        (ConstraintPolynomialLift.liftConstraintPolynomial lift polynomial) =
      linearPolynomialCounts polynomial := by
  unfold linearPolynomialCounts polynomialCounts
    ConstraintPolynomialLift.liftConstraintPolynomial
  exact polynomialCounts_map_liftMonomial lift (fun _ => Counts.zero)
    polynomial.terms Counts.zero

end NightstreamFPrime.Layout.Polynomial.Sparse
