import NightstreamFPrime.Spec.ProductionRelation.SelectivePolynomial

/-!
Owns the one production CCS polynomial for the Nightstream F-prime relation.

The relation uses the Lean-owned selective low-norm compiler gate. Its first
13 matrix slots are named selective ports. Slot 13 is a canonical zero matrix.
SuperNeo v1.1 Pad is not a CCS matrix and remains the separate `Eval_K`
family.
-/

namespace NightstreamFPrime.Spec.ProductionRelation

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

/-- Fixed SuperNeo v1.1 `Eval_A` arity. -/
def matrixCount : Nat := SelectivePolynomial.matrixCount

/-- Number of matrix slots used by the selective compiler. -/
def meaningfulPortCount : Nat := SelectivePolynomial.meaningfulPortCount

/-- Final canonical-zero matrix slot. -/
def zeroPort : Fin matrixCount := SelectivePolynomial.zeroPort

/-- The sole production CCS polynomial. -/
def polynomial : ConstraintPolynomial F matrixCount :=
  SelectivePolynomial.polynomial

@[simp] theorem matrixCount_eq : matrixCount = 14 := by
  rfl

@[simp] theorem meaningfulPortCount_eq : meaningfulPortCount = 13 := by
  rfl

@[simp] theorem polynomial_terms :
    polynomial.terms = SelectivePolynomial.terms := by
  rfl

@[simp] theorem polynomial_degreeBound : polynomial.degreeBound = 9 := by
  rfl

theorem polynomial_canonicalEqualityGatedDegreeBound :
    polynomial.canonicalEqualityGatedDegreeBound = 9 :=
  SelectivePolynomial.polynomial_canonicalEqualityGatedDegreeBound

theorem polynomial_zeroPort
    (candidate : Monomial F matrixCount)
    (member : candidate ∈ polynomial.terms) :
    candidate.exponents zeroPort = 0 :=
  SelectivePolynomial.polynomial_zeroPort candidate member

end NightstreamFPrime.Spec.ProductionRelation
