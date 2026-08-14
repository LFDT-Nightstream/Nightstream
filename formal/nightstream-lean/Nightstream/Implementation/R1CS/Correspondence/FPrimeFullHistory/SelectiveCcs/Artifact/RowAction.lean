import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Interpreter
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Rows

/-!
Contract: semantic action of one decoded selective-CCS matrix row.

Owns: the canonical finite matrix-vector product for each of the thirteen
artifact ports, its equality with the paper Boolean-row adapter, evaluation of
the interpreted relation's fixed polynomial, and reduction to the six named
model-level row shapes.

Does not own: a concrete artifact, Rust serialization, row-family labels,
proof that an emitted row has one of these shapes, row multiplicity, protocol
minimality, or permission to remove constraints.

Emits constraints: no.

| Stage path | Mathematical obligation | Required evidence | Result |
|---|---|---|---|
| `f_prime.selective_ccs.artifact.row.matrix_action` | one exact finite dot product per port | decoded matrix plus typed assignment | `matrixImageAt` |
| `f_prime.selective_ccs.artifact.row.paper_bridge` | numeric row action equals the paper's padded Boolean row action | row-domain coverage | `matrixImageAt_eq_paddedMatrixVectorAt` |
| `f_prime.selective_ccs.artifact.row.residual` | evaluate the relation-owned 74-term polynomial | thirteen matrix images | `residualAt_eq_evaluate` |
| `f_prime.selective_ccs.artifact.row.shape.*` | exact port image activates only the stated residual components | extensional equality with a `Polynomial.Rows` point | six `residualAt_*Point` theorems |

The shape hypotheses are deliberately extensional matrix-image equalities.
A Rust family tag, rewrite ID, or stage label cannot discharge them.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.SelectiveCcs.RelationProfile

/-- One decoded numeric matrix row applied to a typed assignment. The dummy
zero-dimensional Boolean vertex lets this definition reuse the paper's sole
canonical finite matrix-vector implementation without choosing a row cube. -/
def matrixImageAt
    {rows columns : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns)
    (row : Fin rows) (port : Fin 13) : F :=
  matrixVectorAt baseOps
    (fun (_ : BooleanVertex 0) column =>
      relation.matrixAt port row column)
    assignment .nil

/-- The direct numeric action is exactly the action of the independently
defined zero-padded paper matrix at the corresponding Boolean row. This is
the model-level row-index bridge; it does not select production row variables. -/
theorem matrixImageAt_eq_paddedMatrixVectorAt
    {rows columns rowVariables : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns)
    (covers : rows ≤ 2 ^ rowVariables)
    (row : Fin rows) (port : Fin 13) :
    matrixImageAt relation assignment row port =
      matrixVectorAt baseOps
        (RowPadding.padRows (relation.matrixAt port)) assignment
        (RowPadding.numericRowVertex covers row) := by
  simp [matrixImageAt, matrixVectorAt]

/-- The thirteen matrix images consumed by the fixed selective polynomial. -/
def rowPoint
    {rows columns : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns)
    (row : Fin rows) : Fin 13 → F :=
  fun port => matrixImageAt relation assignment row port

/-- Residual of one decoded row under the polynomial fixed by the interpreted
relation. No polynomial or acceptance predicate is supplied by the caller. -/
def residualAt
    {rows columns : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns)
    (row : Fin rows) : F :=
  CCSResidualTable.evaluatePolynomial baseOps
    (InterpretedRelation.constraintPolynomial relation)
    (rowPoint relation assignment row)

/-- The relation-owned polynomial is definitionally the independent exact
selective polynomial. -/
theorem residualAt_eq_evaluate
    {rows columns : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns)
    (row : Fin rows) :
    residualAt relation assignment row =
      evaluate (rowPoint relation assignment row) := by
  rfl

theorem residualAt_booleanPoint
    {rows columns : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns) (row : Fin rows)
    (selector bit : F)
    (images : rowPoint relation assignment row =
      booleanPoint selector bit) :
    residualAt relation assignment row =
      booleanResidual (booleanPoint selector bit) := by
  rw [residualAt_eq_evaluate, images, evaluate_booleanPoint]

theorem residualAt_productPoint
    {rows columns : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns) (row : Fin rows)
    (selector left right output : F)
    (images : rowPoint relation assignment row =
      productPoint selector left right output) :
    residualAt relation assignment row =
      productResidual (productPoint selector left right output) := by
  rw [residualAt_eq_evaluate, images, evaluate_productPoint]

theorem residualAt_sboxPoint
    {rows columns : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns) (row : Fin rows)
    (selector input output : F)
    (images : rowPoint relation assignment row =
      sboxPoint selector input output) :
    residualAt relation assignment row =
      productResidual (sboxPoint selector input output) +
        sboxResidual (sboxPoint selector input output) := by
  rw [residualAt_eq_evaluate, images, evaluate_sboxPoint]

theorem residualAt_centeredPoint
    {rows columns : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns) (row : Fin rows)
    (selector unit : F)
    (images : rowPoint relation assignment row =
      centeredPoint selector unit) :
    residualAt relation assignment row =
      centeredResidual (centeredPoint selector unit) := by
  rw [residualAt_eq_evaluate, images, evaluate_centeredPoint]

theorem residualAt_evaluationPoint
    {rows columns : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns) (row : Fin rows)
    (selector bit a b sbox unit digit borrow nextBorrow boundDigit tail
      output : F)
    (images : rowPoint relation assignment row =
      evaluationPoint selector bit a b sbox unit digit borrow nextBorrow
        boundDigit tail output) :
    residualAt relation assignment row =
      evaluationResidual
        (evaluationPoint selector bit a b sbox unit digit borrow nextBorrow
          boundDigit tail output) := by
  rw [residualAt_eq_evaluate, images, evaluate_evaluationPoint]

theorem residualAt_canonicalPoint
    {rows columns : Nat}
    (relation : InterpretedRelation rows columns)
    (assignment : Assignment F columns) (row : Fin rows)
    (selector digit borrow nextBorrow boundDigit : F)
    (images : rowPoint relation assignment row =
      canonicalPoint selector digit borrow nextBorrow boundDigit) :
    residualAt relation assignment row =
      canonicalResidual
        (canonicalPoint selector digit borrow nextBorrow boundDigit) := by
  rw [residualAt_eq_evaluate, images, evaluate_canonicalPoint]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction
