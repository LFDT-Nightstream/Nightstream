import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Semantics

/-!
Contract: named mathematical components of the selective CCS polynomial.

Owns: the six top-level residual families obtained by grouping the exact 27
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
| `f_prime.selective_ccs.polynomial.canonical` | shifted-base-three borrow interpolation | 14 | checked component |
| `f_prime.selective_ccs.polynomial.total` | sum of the six named residuals | 27 | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components

open Nightstream.SuperNeo.Concrete
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

/-- Five-pair evaluation residual. The addends follow the exact 27-term sparse
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

/-- Exact fourteen-term shifted-base-three borrow interpolation residual. -/
def canonicalResidual (point : Fin 13 -> F) : F :=
  half * point Role.generalSelector.index * point Role.canonicalBoundDigit.index +
    point Role.generalSelector.index * point Role.canonicalNextBorrow.index +
    -(point Role.generalSelector.index * point Role.canonicalBorrow.index) +
    -(half * point Role.generalSelector.index * point Role.canonicalDigit.index) +
    -(half * point Role.generalSelector.index *
      (point Role.canonicalBoundDigit.index *
        point Role.canonicalBoundDigit.index)) +
    quarter * point Role.generalSelector.index * point Role.canonicalDigit.index *
      point Role.canonicalBoundDigit.index +
    -(half * point Role.generalSelector.index *
      (point Role.canonicalDigit.index * point Role.canonicalDigit.index)) +
    point Role.generalSelector.index * point Role.canonicalBorrow.index *
      (point Role.canonicalBoundDigit.index *
        point Role.canonicalBoundDigit.index) +
    quarter * point Role.generalSelector.index * point Role.canonicalDigit.index *
      (point Role.canonicalBoundDigit.index *
        point Role.canonicalBoundDigit.index) +
    -(half * point Role.generalSelector.index * point Role.canonicalDigit.index *
      point Role.canonicalBorrow.index * point Role.canonicalBoundDigit.index) +
    -(quarter * point Role.generalSelector.index *
      (point Role.canonicalDigit.index * point Role.canonicalDigit.index) *
      point Role.canonicalBoundDigit.index) +
    point Role.generalSelector.index *
      (point Role.canonicalDigit.index * point Role.canonicalDigit.index) *
      point Role.canonicalBorrow.index +
    3 * quarter * point Role.generalSelector.index *
      (point Role.canonicalDigit.index * point Role.canonicalDigit.index) *
      (point Role.canonicalBoundDigit.index *
        point Role.canonicalBoundDigit.index) +
    -(3 * half * point Role.generalSelector.index *
      (point Role.canonicalDigit.index * point Role.canonicalDigit.index) *
      point Role.canonicalBorrow.index *
      (point Role.canonicalBoundDigit.index *
        point Role.canonicalBoundDigit.index))

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
the existing 27-term sparse evaluator. -/
theorem evaluate_eq_combinedResidual (point : Fin 13 -> F) :
    evaluate point = combinedResidual point := by
  simp [evaluate, polynomial, terms, monomial, exponentVector,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluatePolynomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluateMonomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
    combinedResidual, booleanResidual, productResidual, sboxResidual,
    centeredResidual, evaluationResidual, canonicalResidual, Role.index,
    Fin.one_mul, Fin.mul_one, Fin.zero_add, Lean.Grind.Fin.neg_mul]
  simp only [fadd_assoc]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components
