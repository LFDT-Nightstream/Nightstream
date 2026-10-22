import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

/-!
Structural carrier lift for an explicit sparse CCS polynomial.

Protocol: shared CCS syntax infrastructure.
Phase: carrier placement before polynomial evaluation.
Constraint family: none; this file emits no rows.

Owns: coefficient-only lifting of sparse monomials and complete constraint
polynomials, with exact preservation of exponents, term order, declared
metadata, and syntax-derived degree.

Does not own: algebraic homomorphism laws, evaluation, norm residuals,
protocol source data, SumCheck, Rust, R1CS, or constraint counts.

Emits constraints: no.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift

universe uBase uExtension

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def liftMonomial
    {Base : Type uBase}
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (lift : Base -> Extension)
    (monomial : CCSResidualTable.Monomial Base matrixCount) :
    CCSResidualTable.Monomial Extension matrixCount where
  coefficient := lift monomial.coefficient
  exponents := monomial.exponents

theorem liftMonomial_totalDegree
    {Base : Type uBase}
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (lift : Base -> Extension)
    (monomial : CCSResidualTable.Monomial Base matrixCount) :
    (liftMonomial lift monomial).totalDegree = monomial.totalDegree := by
  rfl

def liftConstraintPolynomial
    {Base : Type uBase}
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (lift : Base -> Extension)
    (polynomial : CCSResidualTable.ConstraintPolynomial Base matrixCount) :
    CCSResidualTable.ConstraintPolynomial Extension matrixCount where
  degreeBound := polynomial.degreeBound
  terms := polynomial.terms.map (liftMonomial lift)
  termsBelowDegree := by
    intro term member
    rcases List.mem_map.mp member with ⟨baseTerm, baseMember, rfl⟩
    simpa only [liftMonomial_totalDegree] using
      polynomial.termsBelowDegree baseTerm baseMember

private theorem maxDegreeFold_map_liftMonomial
    {Base : Type uBase}
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (lift : Base -> Extension)
    (terms : List (CCSResidualTable.Monomial Base matrixCount))
    (initial : Nat) :
    (terms.map (liftMonomial lift)).foldl
        (fun current term => current.max (term.totalDegree + 1)) initial =
      terms.foldl
        (fun current term => current.max (term.totalDegree + 1)) initial := by
  induction terms generalizing initial with
  | nil => rfl
  | cons term terms inductionHypothesis =>
      simp only [List.map_cons, List.foldl_cons, liftMonomial_totalDegree]
      exact inductionHypothesis _

theorem liftConstraintPolynomial_canonicalEqualityGatedDegreeBound
    {Base : Type uBase}
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (lift : Base -> Extension)
    (polynomial : CCSResidualTable.ConstraintPolynomial Base matrixCount) :
    (liftConstraintPolynomial lift polynomial).canonicalEqualityGatedDegreeBound =
      polynomial.canonicalEqualityGatedDegreeBound := by
  unfold CCSResidualTable.ConstraintPolynomial.canonicalEqualityGatedDegreeBound
    liftConstraintPolynomial
  exact maxDegreeFold_map_liftMonomial lift polynomial.terms 0

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift
