import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable

/-!
Add one ignored leading matrix variable to the paper CCS polynomial.

Protocol: SuperNeo `Pi_CCS`, Section 7.3 / Appendix D.4.
Phase: relation normalization before the identity-first paper verifier.
Constraint family: sparse CCS polynomial syntax only; this file emits no rows.

Owns: the exact arity change `t -> t + 1` that reserves matrix index zero for
the paper's identity matrix while moving every original matrix variable to its
successor index.  The added exponent is definitionally zero, so term order,
total degree, the canonical SumCheck ceiling, and evaluation are preserved.

Does not own: square-domain padding, matrix entries, assignments, carried
evaluation coordinates, transcript messages, Rust, R1CS, or costs.

Emits constraints: no.

Authority boundary: this is a syntax-directed transform of the independently
stated sparse polynomial.  No evaluator or equivalence proposition is supplied
by a caller.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.ccs.syntax.prepend.monomial` | insert exponent zero at matrix index zero and shift every original exponent by one | computed | `prependIgnoredMonomial` |
| `pi_ccs.ccs.syntax.prepend.degree` | preserve monomial total degree and the canonical equality-gated ceiling | derived | `prependIgnoredMonomial_totalDegree`, `prependIgnoredVariable_canonicalEqualityGatedDegreeBound` |
| `pi_ccs.ccs.syntax.prepend.polynomial` | preserve sparse term order and declared degree while changing arity from `t` to `t + 1` | computed | `prependIgnoredVariable` |
| `pi_ccs.ccs.syntax.prepend.evaluation` | evaluation ignores the new head value and equals evaluation of the original polynomial on the tail | derived | `evaluateMonomial_prependIgnoredMonomial`, `evaluatePolynomial_prependIgnoredVariable` |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialPrepend

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uField

private theorem canonicalFinIndices_succ (count : Nat) :
    canonicalFinIndices (count + 1) =
      0 :: (canonicalFinIndices count).map Fin.succ := by
  unfold canonicalFinIndices
  rw [List.ofFn_succ]
  congr 1
  simp [Function.comp_def]

/-- Reserve variable zero and move every original exponent to its successor
index.  The new variable cannot affect the monomial because its exponent is
zero. -/
def prependIgnoredMonomial
    {Field : Type uField}
    {matrixCount : Nat}
    (monomial : CCSResidualTable.Monomial Field matrixCount) :
    CCSResidualTable.Monomial Field (matrixCount + 1) where
  coefficient := monomial.coefficient
  exponents := Fin.cases 0 monomial.exponents

/-- Adding the ignored head variable preserves the syntax-derived total
degree exactly. -/
theorem prependIgnoredMonomial_totalDegree
    {Field : Type uField}
    {matrixCount : Nat}
    (monomial : CCSResidualTable.Monomial Field matrixCount) :
    (prependIgnoredMonomial monomial).totalDegree = monomial.totalDegree := by
  unfold CCSResidualTable.Monomial.totalDegree prependIgnoredMonomial
  rw [canonicalFinIndices_succ]
  simp [Function.comp_def]

/-- Add the ignored leading variable to every sparse term without changing
term order or declared degree metadata. -/
def prependIgnoredVariable
    {Field : Type uField}
    {matrixCount : Nat}
    (polynomial : CCSResidualTable.ConstraintPolynomial Field matrixCount) :
    CCSResidualTable.ConstraintPolynomial Field (matrixCount + 1) where
  degreeBound := polynomial.degreeBound
  terms := polynomial.terms.map prependIgnoredMonomial
  termsBelowDegree := by
    intro term member
    rcases List.mem_map.mp member with ⟨sourceTerm, sourceMember, rfl⟩
    simpa only [prependIgnoredMonomial_totalDegree] using
      polynomial.termsBelowDegree sourceTerm sourceMember

private theorem maxDegreeFold_map_prependIgnoredMonomial
    {Field : Type uField}
    {matrixCount : Nat}
    (terms : List (CCSResidualTable.Monomial Field matrixCount))
    (initial : Nat) :
    (terms.map prependIgnoredMonomial).foldl
        (fun current term => current.max (term.totalDegree + 1)) initial =
      terms.foldl
        (fun current term => current.max (term.totalDegree + 1)) initial := by
  induction terms generalizing initial with
  | nil => rfl
  | cons term terms inductionHypothesis =>
      simp only [List.map_cons, List.foldl_cons,
        prependIgnoredMonomial_totalDegree]
      exact inductionHypothesis _

/-- Reserving the identity variable does not change the canonical verifier
degree ceiling. -/
theorem prependIgnoredVariable_canonicalEqualityGatedDegreeBound
    {Field : Type uField}
    {matrixCount : Nat}
    (polynomial : CCSResidualTable.ConstraintPolynomial Field matrixCount) :
    (prependIgnoredVariable polynomial).canonicalEqualityGatedDegreeBound =
      polynomial.canonicalEqualityGatedDegreeBound := by
  unfold CCSResidualTable.ConstraintPolynomial.canonicalEqualityGatedDegreeBound
    prependIgnoredVariable
  exact maxDegreeFold_map_prependIgnoredMonomial polynomial.terms 0

/-- Evaluation ignores the new head coordinate and reads every original
coordinate at its successor index. -/
theorem evaluateMonomial_prependIgnoredMonomial
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {matrixCount : Nat}
    (monomial : CCSResidualTable.Monomial Field matrixCount)
    (point : Fin (matrixCount + 1) -> Field) :
    CCSResidualTable.evaluateMonomial ops
        (prependIgnoredMonomial monomial) point =
      CCSResidualTable.evaluateMonomial ops monomial
        (fun index => point index.succ) := by
  unfold CCSResidualTable.evaluateMonomial prependIgnoredMonomial
  rw [canonicalFinIndices_succ]
  simp only [List.foldl_cons, Fin.cases_zero, CCSResidualTable.pow]
  rw [laws.mul_one]
  rw [List.foldl_map]
  rfl

private theorem polynomialFold_prependIgnoredVariable
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {matrixCount : Nat}
    (point : Fin (matrixCount + 1) -> Field)
    (terms : List (CCSResidualTable.Monomial Field matrixCount))
    (accumulated : Field) :
    (terms.map prependIgnoredMonomial).foldl
        (fun value monomial =>
          ops.add value (CCSResidualTable.evaluateMonomial ops monomial point))
        accumulated =
      terms.foldl
        (fun value monomial =>
          ops.add value (CCSResidualTable.evaluateMonomial ops monomial
            (fun index => point index.succ)))
        accumulated := by
  induction terms generalizing accumulated with
  | nil => rfl
  | cons monomial terms inductionHypothesis =>
      simp only [List.map_cons, List.foldl_cons]
      rw [evaluateMonomial_prependIgnoredMonomial ops laws]
      exact inductionHypothesis _

/-- Complete sparse-polynomial evaluation is unchanged after the identity
port is prepended. -/
theorem evaluatePolynomial_prependIgnoredVariable
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {matrixCount : Nat}
    (polynomial : CCSResidualTable.ConstraintPolynomial Field matrixCount)
    (point : Fin (matrixCount + 1) -> Field) :
    CCSResidualTable.evaluatePolynomial ops
        (prependIgnoredVariable polynomial) point =
      CCSResidualTable.evaluatePolynomial ops polynomial
        (fun index => point index.succ) := by
  unfold CCSResidualTable.evaluatePolynomial prependIgnoredVariable
  exact polynomialFold_prependIgnoredVariable ops laws point polynomial.terms
    ops.zero

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialPrepend
