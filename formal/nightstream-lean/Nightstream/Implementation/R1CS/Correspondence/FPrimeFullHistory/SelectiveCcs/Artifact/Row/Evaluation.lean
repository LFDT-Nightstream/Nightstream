import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Boolean
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.GroupedProduct

/-!
Contract: coefficient-derived semantics of one decoded evaluation row.

Owns: exact evaluation-selector classification, equality with the independent
five-product point, and the active-row equivalence between zero residual and
the five-product equation.

Does not own: a Rust family label, executable source provenance,
source-to-final assignment encoding, production multiplicity, selector
dispatch, constraint necessity, or row removal.

Emits constraints: no.

| Stage path | Checked property | Result |
|---|---|---|
| evaluation gate | general selector empty, evaluation selector unit | `ValidatedEvaluationRow` |
| exact row point | thirteen decoded actions equal the five-product point | `rowPoint_eq_evaluationPoint` |
| active residual | selector one and zero residual iff five-product equality | `residual_zero_iff_fiveProduct` |
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Evaluation

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct

/-- Exact coefficient shape of an evaluation-gated row. Other port actions
remain arbitrary and become the five products plus the output image. -/
def IsEvaluationAt (row : DecodedRow)
    (selectorColumn : Fin row.columns) : Prop :=
  (row.port Role.generalSelector.index).terms = [] ∧
    (row.port Role.evalSelector.index).terms =
      [{ column := selectorColumn, coefficient := 1 }]

instance (row : DecodedRow) (selectorColumn : Fin row.columns) :
    Decidable (IsEvaluationAt row selectorColumn) := by
  unfold IsEvaluationAt
  infer_instance

structure ValidatedEvaluationRow (row : DecodedRow) where
  selectorColumn : Fin row.columns
  shape : IsEvaluationAt row selectorColumn

def validateEvaluationAt (row : DecodedRow)
    (selectorColumn : Fin row.columns) :
    Option (ValidatedEvaluationRow row) :=
  if shape : IsEvaluationAt row selectorColumn then
    some ⟨selectorColumn, shape⟩
  else
    none

theorem rowPoint_eq_evaluationPoint
    (row : DecodedRow) (validated : ValidatedEvaluationRow row)
    (assignment : Fin row.columns → F) :
    rowPoint row assignment =
      evaluationPoint
        (assignment validated.selectorColumn)
        (action (row.port Role.bit.index) assignment)
        (action (row.port Role.a.index) assignment)
        (action (row.port Role.b.index) assignment)
        (action (row.port Role.sboxInput.index) assignment)
        (action (row.port Role.centeredUnit.index) assignment)
        (action (row.port Role.canonicalDigit.index) assignment)
        (action (row.port Role.canonicalBorrow.index) assignment)
        (action (row.port Role.canonicalNextBorrow.index) assignment)
        (action (row.port Role.canonicalBoundDigit.index) assignment)
        (action (row.port Role.evalTailRight.index) assignment)
        (action (row.port Role.c.index) assignment) := by
  have shape := validated.shape
  simp only [IsEvaluationAt] at shape
  rcases shape with ⟨generalEmpty, evaluationUnit⟩
  funext port
  let role := Role.ofIndex port
  have indexEq : role.index = port := Role.index_ofIndex port
  rw [← indexEq]
  cases role
  case generalSelector =>
    simp only [rowPoint, action]
    rw [generalEmpty]
    simp [evaluationPoint, sparsePoint, Role.index]
  case evalSelector =>
    simp only [rowPoint, action]
    rw [evaluationUnit]
    simp [evaluationPoint, sparsePoint, Role.index, Fin.one_mul]
  all_goals
    simp [rowPoint, evaluationPoint, sparsePoint, Role.index,
      action]

/-- With the branch selector fixed to one, the decoded row has no weaker or
stronger algebraic meaning than its five-product equation. -/
theorem residual_zero_iff_fiveProduct
    (row : DecodedRow) (validated : ValidatedEvaluationRow row)
    (assignment : Fin row.columns → F)
    (selectorOne : assignment validated.selectorColumn = 1) :
    residual row assignment = 0 ↔
      action (row.port Role.c.index) assignment =
        fiveProductSum
          (action (row.port Role.bit.index) assignment)
          (action (row.port Role.a.index) assignment)
          (action (row.port Role.b.index) assignment)
          (action (row.port Role.sboxInput.index) assignment)
          (action (row.port Role.centeredUnit.index) assignment)
          (action (row.port Role.canonicalDigit.index) assignment)
          (action (row.port Role.canonicalBorrow.index) assignment)
          (action (row.port Role.canonicalNextBorrow.index) assignment)
          (action (row.port Role.canonicalBoundDigit.index) assignment)
          (action (row.port Role.evalTailRight.index) assignment) := by
  change
    Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.evaluate
      (rowPoint row assignment) = 0 ↔ _
  rw [rowPoint_eq_evaluationPoint row validated assignment, selectorOne]
  apply evaluationPoint_zero_iff_fiveProduct

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Evaluation
