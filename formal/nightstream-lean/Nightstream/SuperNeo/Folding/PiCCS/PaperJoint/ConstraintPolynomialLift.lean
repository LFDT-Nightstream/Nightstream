import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable

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

Authority boundary: lifting changes coefficients through one explicit
function and cannot alter exponent vectors or add/remove/reorder terms.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.ccs.syntax.lift.monomial` | lift coefficient; preserve exponent vector | computed | `liftMonomial` |
| `pi_ccs.ccs.syntax.lift.degree` | total degree is unchanged | derived | `liftMonomial_totalDegree` |
| `pi_ccs.ccs.syntax.lift.polynomial` | preserve ordered sparse syntax and degree proof | computed | `liftConstraintPolynomial` |
| `pi_ccs.ccs.syntax.lift.canonical_degree` | preserve the syntax-derived equality-gated ceiling | derived | `liftConstraintPolynomial_canonicalEqualityGatedDegreeBound` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift

universe uBase uExtension

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Lift only a monomial's coefficient. -/
def liftMonomial
    {Base : Type uBase}
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (lift : Base -> Extension)
    (monomial : CCSResidualTable.Monomial Base matrixCount) :
    CCSResidualTable.Monomial Extension matrixCount where
  coefficient := lift monomial.coefficient
  exponents := monomial.exponents

/-- A coefficient lift cannot change a sparse monomial's total degree. -/
theorem liftMonomial_totalDegree
    {Base : Type uBase}
    {Extension : Type uExtension}
    {matrixCount : Nat}
    (lift : Base -> Extension)
    (monomial : CCSResidualTable.Monomial Base matrixCount) :
    (liftMonomial lift monomial).totalDegree = monomial.totalDegree := by
  rfl

/-- Lift an explicit sparse CCS polynomial without changing its term order,
exponents, declared degree metadata, or proof that terms obey that metadata. -/
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

/-- Coefficient lifting preserves the canonical verifier degree because it
cannot change the ordered exponent vectors. -/
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

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift
