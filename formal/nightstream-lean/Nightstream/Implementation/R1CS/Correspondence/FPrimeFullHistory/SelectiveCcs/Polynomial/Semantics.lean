import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Ports

/-!
Contract: exact model-level syntax and evaluation of the production selective
CCS polynomial.

Owns: all 66 sparse monomials, their order, Goldilocks coefficients, the exact
13-port arity, and the syntax-derived degree ceiling. This is an independent
mathematical transcription of the currently emitted gate polynomial.

Does not own: any emitted matrix row, source-column coefficient, row order,
multiplicity, Rust artifact, assignment decoder, or permission to remove rows.
Those remain fail-closed until raw production data are decoded and compared to
this syntax.

Emits constraints: no.

| Stage path | Mathematical obligation | Terms | Maximum degree | Rust owner |
|---|---|---:|---:|---|
| `f_prime.selective_ccs.polynomial.boolean` | `g * (bit^2 - bit)` | 2 | 3 | `selective_polynomial` |
| `f_prime.selective_ccs.polynomial.product` | `g * (a*b - c)` | 2 | 3 | `selective_polynomial` |
| `f_prime.selective_ccs.polynomial.sbox` | `g * sbox^7` | 1 | 8 | `selective_polynomial` |
| `f_prime.selective_ccs.polynomial.centered` | `g * (u^3-u)` | 2 | 4 | `selective_polynomial` |
| `f_prime.selective_ccs.polynomial.evaluation` | `e * (sum five products - c)` | 6 | 3 | `selective_polynomial` |
| `f_prime.selective_ccs.polynomial.canonical` | selected two-trit borrow transition | 53 | 7 | `selective_polynomial` |
| **total** | exact sparse polynomial | **66** | **8** | `selective_polynomial` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports

/-- Multiplicative inverse of two in the production Goldilocks field. -/
def half : F := ⟨9223372034707292161, by decide⟩

/-- Multiplicative inverse of four in the production Goldilocks field. -/
def quarter : F := ⟨13835058052060938241, by decide⟩

theorem half_exact : (2 : F) * half = 1 := by decide

theorem quarter_exact : quarter = half * half := by decide

/-- Rust-compatible exponent-vector construction: listed roles overwrite the
zero vector at their exact numeric ports. Every production term below lists a
role at most once. -/
def exponentVector (powers : List (Role × Nat)) : Fin 13 -> Nat :=
  fun index => powers.foldl
    (fun current power => if index = power.1.index then power.2 else current) 0

def monomial (coefficient : F) (powers : List (Role × Nat)) :
    Monomial F 13 where
  coefficient := coefficient
  exponents := exponentVector powers

/-- The thirteen terms shared by ordinary and evaluation rows. -/
def baseTerms : List (Monomial F 13) := [
  monomial 1 [(Role.generalSelector, 1), (Role.bit, 2)],
  monomial (-1) [(Role.generalSelector, 1), (Role.bit, 1)],
  monomial 1 [(Role.generalSelector, 1), (Role.a, 1), (Role.b, 1)],
  monomial (-1) [(Role.generalSelector, 1), (Role.c, 1)],
  monomial 1 [(Role.generalSelector, 1), (Role.sboxInput, 7)],
  monomial 1 [(Role.generalSelector, 1), (Role.centeredUnit, 3)],
  monomial (-1) [(Role.generalSelector, 1), (Role.centeredUnit, 1)],
  monomial (-1) [(Role.evalSelector, 1), (Role.c, 1)],
  monomial 1 [(Role.evalSelector, 1), (Role.bit, 1), (Role.a, 1)],
  monomial 1 [(Role.evalSelector, 1), (Role.b, 1), (Role.sboxInput, 1)],
  monomial 1 [(Role.evalSelector, 1), (Role.centeredUnit, 1),
    (Role.canonicalDigit, 1)],
  monomial 1 [(Role.evalSelector, 1), (Role.canonicalBorrow, 1),
    (Role.canonicalNextBorrow, 1)],
  monomial 1 [(Role.evalSelector, 1), (Role.canonicalBoundDigit, 1),
    (Role.evalTailRight, 1)]
]

/-- Exact expansion of the five normalized two-trit borrow classes. Ports
8--12 are the mutually exclusive class selectors on canonical rows. -/
def canonicalTerms : List (Monomial F 13) := [
  monomial (-1) [(Role.generalSelector, 1), (Role.canonicalDigit, 1)],
  monomial 1 [(Role.generalSelector, 1), (Role.c, 1), (Role.canonicalDigit, 1)],
  monomial quarter [(Role.generalSelector, 1), (Role.a, 1),
    (Role.centeredUnit, 1), (Role.canonicalDigit, 1)],
  monomial (-quarter) [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.a, 1), (Role.centeredUnit, 1), (Role.canonicalDigit, 1)],
  monomial (-quarter) [(Role.generalSelector, 1), (Role.a, 2),
    (Role.centeredUnit, 1), (Role.canonicalDigit, 1)],
  monomial quarter [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.a, 2), (Role.centeredUnit, 1), (Role.canonicalDigit, 1)],
  monomial (-quarter) [(Role.generalSelector, 1), (Role.a, 1),
    (Role.centeredUnit, 2), (Role.canonicalDigit, 1)],
  monomial quarter [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.a, 1), (Role.centeredUnit, 2), (Role.canonicalDigit, 1)],
  monomial quarter [(Role.generalSelector, 1), (Role.a, 2),
    (Role.centeredUnit, 2), (Role.canonicalDigit, 1)],
  monomial (-quarter) [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.a, 2), (Role.centeredUnit, 2), (Role.canonicalDigit, 1)],
  monomial (-1) [(Role.generalSelector, 1), (Role.canonicalBorrow, 1)],
  monomial 1 [(Role.generalSelector, 1), (Role.c, 1),
    (Role.canonicalBorrow, 1)],
  monomial (-half) [(Role.generalSelector, 1), (Role.a, 1),
    (Role.canonicalBorrow, 1)],
  monomial half [(Role.bit, 1), (Role.generalSelector, 1), (Role.a, 1),
    (Role.canonicalBorrow, 1)],
  monomial half [(Role.generalSelector, 1), (Role.a, 2),
    (Role.canonicalBorrow, 1)],
  monomial (-half) [(Role.bit, 1), (Role.generalSelector, 1), (Role.a, 2),
    (Role.canonicalBorrow, 1)],
  monomial quarter [(Role.generalSelector, 1), (Role.a, 1),
    (Role.centeredUnit, 1), (Role.canonicalBorrow, 1)],
  monomial (-quarter) [(Role.generalSelector, 1), (Role.a, 2),
    (Role.centeredUnit, 1), (Role.canonicalBorrow, 1)],
  monomial quarter [(Role.generalSelector, 1), (Role.a, 1),
    (Role.centeredUnit, 2), (Role.canonicalBorrow, 1)],
  monomial (-half) [(Role.bit, 1), (Role.generalSelector, 1), (Role.a, 1),
    (Role.centeredUnit, 2), (Role.canonicalBorrow, 1)],
  monomial (-quarter) [(Role.generalSelector, 1), (Role.a, 2),
    (Role.centeredUnit, 2), (Role.canonicalBorrow, 1)],
  monomial half [(Role.bit, 1), (Role.generalSelector, 1), (Role.a, 2),
    (Role.centeredUnit, 2), (Role.canonicalBorrow, 1)],
  monomial (-1) [(Role.generalSelector, 1), (Role.canonicalNextBorrow, 1)],
  monomial 1 [(Role.generalSelector, 1), (Role.c, 1),
    (Role.canonicalNextBorrow, 1)],
  monomial (-half) [(Role.generalSelector, 1), (Role.a, 1),
    (Role.canonicalNextBorrow, 1)],
  monomial half [(Role.generalSelector, 1), (Role.a, 2),
    (Role.canonicalNextBorrow, 1)],
  monomial quarter [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.a, 1), (Role.centeredUnit, 1), (Role.canonicalNextBorrow, 1)],
  monomial (-quarter) [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.a, 2), (Role.centeredUnit, 1), (Role.canonicalNextBorrow, 1)],
  monomial quarter [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.a, 1), (Role.centeredUnit, 2), (Role.canonicalNextBorrow, 1)],
  monomial (-quarter) [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.a, 2), (Role.centeredUnit, 2), (Role.canonicalNextBorrow, 1)],
  monomial (-1) [(Role.generalSelector, 1), (Role.canonicalBoundDigit, 1)],
  monomial 1 [(Role.generalSelector, 1), (Role.c, 1),
    (Role.canonicalBoundDigit, 1)],
  monomial (-half) [(Role.generalSelector, 1), (Role.a, 1),
    (Role.canonicalBoundDigit, 1)],
  monomial half [(Role.generalSelector, 1), (Role.a, 2),
    (Role.canonicalBoundDigit, 1)],
  monomial (-half) [(Role.generalSelector, 1), (Role.centeredUnit, 1),
    (Role.canonicalBoundDigit, 1)],
  monomial half [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.centeredUnit, 1), (Role.canonicalBoundDigit, 1)],
  monomial half [(Role.generalSelector, 1), (Role.a, 2),
    (Role.centeredUnit, 1), (Role.canonicalBoundDigit, 1)],
  monomial (-half) [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.a, 2), (Role.centeredUnit, 1), (Role.canonicalBoundDigit, 1)],
  monomial half [(Role.generalSelector, 1), (Role.centeredUnit, 2),
    (Role.canonicalBoundDigit, 1)],
  monomial (-half) [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.centeredUnit, 2), (Role.canonicalBoundDigit, 1)],
  monomial (-half) [(Role.generalSelector, 1), (Role.a, 2),
    (Role.centeredUnit, 2), (Role.canonicalBoundDigit, 1)],
  monomial half [(Role.bit, 1), (Role.generalSelector, 1), (Role.a, 2),
    (Role.centeredUnit, 2), (Role.canonicalBoundDigit, 1)],
  monomial 1 [(Role.generalSelector, 1), (Role.c, 1),
    (Role.evalTailRight, 1)],
  monomial (-1) [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.evalTailRight, 1)],
  monomial (-half) [(Role.generalSelector, 1), (Role.a, 1),
    (Role.evalTailRight, 1)],
  monomial (-half) [(Role.generalSelector, 1), (Role.a, 2),
    (Role.evalTailRight, 1)],
  monomial 1 [(Role.bit, 1), (Role.generalSelector, 1), (Role.a, 2),
    (Role.evalTailRight, 1)],
  monomial (-half) [(Role.generalSelector, 1), (Role.centeredUnit, 1),
    (Role.evalTailRight, 1)],
  monomial half [(Role.generalSelector, 1), (Role.a, 2),
    (Role.centeredUnit, 1), (Role.evalTailRight, 1)],
  monomial (-half) [(Role.generalSelector, 1), (Role.centeredUnit, 2),
    (Role.evalTailRight, 1)],
  monomial 1 [(Role.bit, 1), (Role.generalSelector, 1),
    (Role.centeredUnit, 2), (Role.evalTailRight, 1)],
  monomial half [(Role.generalSelector, 1), (Role.a, 2),
    (Role.centeredUnit, 2), (Role.evalTailRight, 1)],
  monomial (-1) [(Role.bit, 1), (Role.generalSelector, 1), (Role.a, 2),
    (Role.centeredUnit, 2), (Role.evalTailRight, 1)]
]

/-- Exact ordered 66-term sparse syntax produced by Rust's
`selective_polynomial`. -/
def terms : List (Monomial F 13) :=
  baseTerms ++ canonicalTerms

theorem base_term_count_exact : baseTerms.length = 13 := by decide

theorem canonical_term_count_exact : canonicalTerms.length = 53 := by decide

theorem term_count_exact : terms.length = 66 := by decide

private theorem every_term_degree_checked :
    terms.all (fun term => decide (term.totalDegree < 9)) = true := by
  decide

/-- The declared strict bound nine is derived from every explicit term rather
than accepted as caller metadata. -/
def polynomial : ConstraintPolynomial F 13 where
  degreeBound := 9
  terms := terms
  termsBelowDegree := by
    intro term member
    exact of_decide_eq_true
      ((List.all_eq_true.mp every_term_degree_checked) term member)

set_option maxRecDepth 10000 in
/-- The exact sparse syntax has maximum degree eight, hence equality-gating it
for SumCheck requires the canonical ceiling nine. -/
theorem canonicalEqualityGatedDegreeBound_exact :
    polynomial.canonicalEqualityGatedDegreeBound = 9 := by
  decide

/-- Direct evaluation of the exact sparse syntax over the production
Goldilocks operations. -/
def evaluate (point : Fin 13 -> F) : F :=
  evaluatePolynomial baseOps polynomial point

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
