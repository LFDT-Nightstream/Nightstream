import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Semantics

/-!
Contract: named mathematical components of the selective CCS polynomial.

Owns: the six top-level residual families obtained by grouping the exact 66
sparse terms, and their exact recomposition into `Semantics.evaluate` for every
13-port point.

Does not own: a claim that these are the protocol-minimal obligations, any
production matrix row, row multiplicity, Rust conformance, or permission to
remove constraints.

Emits constraints: no.

| Stage path | Mathematical residual | Sparse terms | Authority class |
|---|---|---:|---|
| `f_prime.selective_ccs.polynomial.boolean` | `g * (bit^2 - bit)` | 2 | checked component |
| `f_prime.selective_ccs.polynomial.product` | `g * (a*b - c)` | 2 | checked component |
| `f_prime.selective_ccs.polynomial.sbox` | `g * s^7` | 1 | checked component |
| `f_prime.selective_ccs.polynomial.centered` | `g * (u^3 - u)` | 2 | checked component |
| `f_prime.selective_ccs.polynomial.evaluation` | `e * (bit*a + b*s + u*d + borrow*next + bound*tail - c)` | 6 | checked component |
| `f_prime.selective_ccs.polynomial.canonical` | five selected two-trit borrow relations | 53 | checked component |
| `f_prime.selective_ccs.polynomial.total` | sum of the six named residuals | 66 | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics

private theorem fadd_assoc (left middle right : F) :
    (left + middle) + right = left + (middle + right) :=
  Lean.Grind.Fin.add_assoc _ _ _

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨fadd_assoc⟩

/-- The six independently named residual families in the current polynomial.
This is an obligation vocabulary, not a claim of protocol-level minimality. -/
inductive Family where
  | boolean
  | product
  | sbox
  | centered
  | evaluation
  | canonical
deriving DecidableEq, Repr

/-- Boolean residual, written in the exact monomial order of the sparse
evaluator. -/
def booleanResidual (point : Fin 13 -> F) : F :=
  point Role.bit.index * point Role.bit.index *
      point Role.generalSelector.index +
    -(point Role.bit.index * point Role.generalSelector.index)

/-- Product residual, written in the exact monomial order of the sparse
evaluator. -/
def productResidual (point : Fin 13 -> F) : F :=
  point Role.generalSelector.index * point Role.a.index * point Role.b.index +
    -(point Role.generalSelector.index * point Role.c.index)

/-- Degree-seven S-box residual. -/
def sboxResidual (point : Fin 13 -> F) : F :=
  point Role.generalSelector.index *
    (point Role.sboxInput.index * point Role.sboxInput.index *
      point Role.sboxInput.index * point Role.sboxInput.index *
      point Role.sboxInput.index * point Role.sboxInput.index *
      point Role.sboxInput.index)

/-- Centered-unit residual. -/
def centeredResidual (point : Fin 13 -> F) : F :=
  point Role.generalSelector.index *
      (point Role.centeredUnit.index * point Role.centeredUnit.index *
        point Role.centeredUnit.index) +
    -(point Role.generalSelector.index * point Role.centeredUnit.index)

/-- Five-pair evaluation residual. The addends follow the exact sparse
syntax; commutative factoring is a derived presentation, not its definition. -/
def evaluationResidual (point : Fin 13 -> F) : F :=
  -(point Role.c.index * point Role.evalSelector.index) +
    point Role.bit.index * point Role.a.index * point Role.evalSelector.index +
    point Role.b.index * point Role.sboxInput.index * point Role.evalSelector.index +
    point Role.centeredUnit.index * point Role.evalSelector.index *
      point Role.canonicalDigit.index +
    point Role.evalSelector.index * point Role.canonicalBorrow.index *
      point Role.canonicalNextBorrow.index +
    point Role.evalSelector.index * point Role.canonicalBoundDigit.index *
      point Role.evalTailRight.index

/-- Recursive sparse sum, kept separate from the production fold evaluator so
termwise selector proofs do not expand all 53 canonical monomials at once. -/
def evaluateTerms (point : Fin 13 -> F) :
    List (Monomial F 13) -> F
  | [] => 0
  | term :: tail =>
      evaluateMonomial baseOps term point + evaluateTerms point tail

/-- Exact 53-term expansion of the five selected two-trit transitions. -/
def canonicalResidual (point : Fin 13 -> F) : F :=
  evaluateTerms point canonicalTerms

set_option maxRecDepth 10000 in
/-- The canonical family is inactive when the general selector is zero. -/
theorem canonicalResidual_zero_of_generalSelector_zero
    (point : Fin 13 → F)
    (generalZero : point Role.generalSelector.index = 0) :
    canonicalResidual point = 0 := by
  change point 1 = 0 at generalZero
  simp [canonicalResidual, evaluateTerms, canonicalTerms, monomial,
    exponentVector,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluateMonomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
    Role.index, generalZero, Fin.zero_mul, Fin.mul_zero, Fin.zero_add,
    Fin.add_zero, Lean.Grind.AddCommGroup.neg_zero]
  grind

set_option maxRecDepth 10000 in
/-- The canonical family is zero when all five selected class ports are zero. -/
theorem canonicalResidual_zero_of_classPorts_zero
    (point : Fin 13 → F)
    (digitZero : point Role.canonicalDigit.index = 0)
    (borrowZero : point Role.canonicalBorrow.index = 0)
    (nextBorrowZero : point Role.canonicalNextBorrow.index = 0)
    (boundDigitZero : point Role.canonicalBoundDigit.index = 0)
    (tailZero : point Role.evalTailRight.index = 0) :
    canonicalResidual point = 0 := by
  change point 8 = 0 at digitZero
  change point 9 = 0 at borrowZero
  change point 10 = 0 at nextBorrowZero
  change point 11 = 0 at boundDigitZero
  change point 12 = 0 at tailZero
  simp [canonicalResidual, evaluateTerms, canonicalTerms, monomial,
    exponentVector,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluateMonomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
    Role.index, digitZero, borrowZero, nextBorrowZero, boundDigitZero,
    tailZero, Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

/-- Select one named component without changing its equation. -/
def residual : Family -> (Fin 13 -> F) -> F
  | .boolean => booleanResidual
  | .product => productResidual
  | .sbox => sboxResidual
  | .centered => centeredResidual
  | .evaluation => evaluationResidual
  | .canonical => canonicalResidual

/-- Human-auditable grouping of the complete sparse polynomial. -/
def combinedResidual (point : Fin 13 -> F) : F :=
  booleanResidual point + productResidual point + sboxResidual point +
    centeredResidual point + evaluationResidual point + canonicalResidual point

/-- The six named equations are neither an approximation nor a parallel
specification: for every possible matrix-image vector, their sum is exactly
the existing 66-term sparse evaluator. -/
theorem evaluate_eq_combinedResidual (point : Fin 13 -> F) :
    evaluate point = combinedResidual point := by
  simp [evaluate, polynomial, terms, baseTerms, canonicalTerms,
    monomial, exponentVector,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluatePolynomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluateMonomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
    combinedResidual, booleanResidual, productResidual, sboxResidual,
    centeredResidual, evaluationResidual, canonicalResidual, evaluateTerms,
    Role.index,
    Fin.one_mul, Fin.mul_one, Fin.zero_add, Lean.Grind.Fin.neg_mul]
  simp only [fadd_assoc]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components
