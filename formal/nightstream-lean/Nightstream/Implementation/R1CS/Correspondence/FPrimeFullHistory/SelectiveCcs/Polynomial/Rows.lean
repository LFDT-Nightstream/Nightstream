import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Components

/-!
Contract: exact model-level row specializations and component-isolation points
of the selective polynomial.

Owns: sparse 13-port points for ordinary emitted row shapes, one canonical
family isolation point, and proofs of which named polynomial components remain
active at each point.

Does not own: concrete matrix coefficients, source-column substitution,
selectors being Boolean/one-hot, row multiplicity, Rust conformance, or
permission to remove rows.

Emits constraints: no.

| Emitted row shape | Nonzero port families | Exact active component(s) |
|---|---|---|
| Boolean/domain | `g`, `bit` | Boolean |
| product | `g`, `a`, `b`, `c` | product |
| Poseidon2 S-box | `g`, `s`, `c` | product output term plus S-box |
| centered unit | `g`, `u` | centered |
| product-sum/evaluation | `e`, five factor pairs, `c` | evaluation |
| canonical-family isolation | `g`, class-selector ports | canonical |

The S-box row intentionally activates two named term groups: `g*s^7` comes
from the S-box component while `-g*c` comes from the product component. The
component vocabulary is therefore an algebraic partition, not a claim that
every physical row belongs to exactly one component.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components

set_option maxRecDepth 10000

/-- Sparse port-image constructor used only by the closed row shapes below.
Later entries overwrite earlier entries, matching matrix contribution
materialization; every exported constructor uses distinct roles. -/
def sparsePoint (entries : List (Role × F)) : Fin 13 -> F := fun port =>
  entries.foldl
    (fun current entry => if port = entry.1.index then entry.2 else current) 0

def booleanPoint (selector bit : F) : Fin 13 -> F :=
  sparsePoint [(.generalSelector, selector), (.bit, bit)]

def productPoint (selector left right output : F) : Fin 13 -> F :=
  sparsePoint [(.generalSelector, selector), (.a, left), (.b, right),
    (.c, output)]

def sboxPoint (selector input output : F) : Fin 13 -> F :=
  sparsePoint [(.generalSelector, selector), (.sboxInput, input),
    (.c, output)]

def centeredPoint (selector unit : F) : Fin 13 -> F :=
  sparsePoint [(.generalSelector, selector), (.centeredUnit, unit)]

def evaluationPoint (selector bit a b sbox unit digit borrow nextBorrow
    boundDigit tail output : F) : Fin 13 -> F :=
  sparsePoint [(.evalSelector, selector), (.bit, bit), (.a, a), (.b, b),
    (.sboxInput, sbox), (.centeredUnit, unit), (.canonicalDigit, digit),
    (.canonicalBorrow, borrow), (.canonicalNextBorrow, nextBorrow),
    (.canonicalBoundDigit, boundDigit), (.evalTailRight, tail),
    (.c, output)]

def canonicalPoint
    (selector digit borrow nextBorrow boundDigit : F) : Fin 13 -> F :=
  sparsePoint [(.generalSelector, selector), (.canonicalDigit, digit),
    (.canonicalBorrow, borrow), (.canonicalNextBorrow, nextBorrow),
    (.canonicalBoundDigit, boundDigit)]

theorem evaluate_booleanPoint (selector bit : F) :
    evaluate (booleanPoint selector bit) =
      booleanResidual (booleanPoint selector bit) := by
  rw [evaluate_eq_combinedResidual]
  simp [combinedResidual, productResidual, sboxResidual, centeredResidual,
    evaluationResidual, canonicalResidual, evaluateTerms, canonicalTerms, monomial,
    exponentVector,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluateMonomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
    booleanPoint, sparsePoint,
    Role.index, Fin.mul_zero, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]
  apply Fin.ext
  simp [Fin.val_add]
  exact Nat.mod_eq_of_lt
    (booleanResidual (booleanPoint selector bit)).isLt

theorem evaluate_productPoint (selector left right output : F) :
    evaluate (productPoint selector left right output) =
      productResidual (productPoint selector left right output) := by
  rw [evaluate_eq_combinedResidual]
  simp [combinedResidual, booleanResidual, sboxResidual, centeredResidual,
    evaluationResidual, canonicalResidual, evaluateTerms, canonicalTerms, monomial,
    exponentVector,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluateMonomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
    productPoint, sparsePoint,
    Role.index, Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

theorem evaluate_sboxPoint (selector input output : F) :
    evaluate (sboxPoint selector input output) =
      productResidual (sboxPoint selector input output) +
        sboxResidual (sboxPoint selector input output) := by
  rw [evaluate_eq_combinedResidual]
  simp [combinedResidual, booleanResidual, centeredResidual,
    evaluationResidual, canonicalResidual, evaluateTerms, canonicalTerms, monomial,
    exponentVector,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluateMonomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
    sboxPoint, sparsePoint,
    Role.index, Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

theorem evaluate_centeredPoint (selector unit : F) :
    evaluate (centeredPoint selector unit) =
      centeredResidual (centeredPoint selector unit) := by
  rw [evaluate_eq_combinedResidual]
  simp [combinedResidual, booleanResidual, productResidual, sboxResidual,
    evaluationResidual, canonicalResidual, evaluateTerms, canonicalTerms, monomial,
    exponentVector,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluateMonomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
    centeredPoint, sparsePoint,
    Role.index, Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

theorem evaluate_evaluationPoint (selector bit a b sbox unit digit borrow
    nextBorrow boundDigit tail output : F) :
    evaluate (evaluationPoint selector bit a b sbox unit digit borrow
      nextBorrow boundDigit tail output) =
        evaluationResidual (evaluationPoint selector bit a b sbox unit digit
          borrow nextBorrow boundDigit tail output) := by
  rw [evaluate_eq_combinedResidual]
  simp [combinedResidual, booleanResidual, productResidual, sboxResidual,
    centeredResidual, canonicalResidual, evaluateTerms, canonicalTerms, monomial,
    exponentVector,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluateMonomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
    evaluationPoint, sparsePoint,
    Role.index, Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]
  apply Fin.ext
  simp [Fin.val_add, Fin.val_mul]
  exact Nat.mod_eq_of_lt
    (evaluationResidual (evaluationPoint selector bit a b sbox unit digit
      borrow nextBorrow boundDigit tail output)).isLt

theorem evaluate_canonicalPoint
    (selector digit borrow nextBorrow boundDigit : F) :
    evaluate (canonicalPoint selector digit borrow nextBorrow boundDigit) =
      canonicalResidual
        (canonicalPoint selector digit borrow nextBorrow boundDigit) := by
  rw [evaluate_eq_combinedResidual]
  simp [combinedResidual, booleanResidual, productResidual, sboxResidual,
    centeredResidual, evaluationResidual, canonicalPoint, sparsePoint,
    Role.index, Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows
