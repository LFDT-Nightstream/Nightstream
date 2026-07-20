import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.PolynomialGating

/-!
Coefficient-derived selector gating for the compact `y_zcol` rows.

Owns: classification of a decoded compact row by its two selector-port
coefficient streams and transport from active Nat-row satisfaction to the
ungated selective-polynomial equation.

Does not own: generated-row truth, family labels, the steady selector value,
source-column meaning, rewrite semantics, protocol authority, or permission
to remove rows.

Emits constraints: no.

The active-selector premise is deliberately external. A gated row at selector
zero is vacuous and cannot establish any source obligation.

| Gating leaf | Mathematical obligation | Authority class |
|---|---|---|
| selector ports | coefficient streams identify one gate class | checked |
| active gate | selector one exposes the ungated residual | derived |
| inactive gate | no semantic conclusion is exported | excluded boundary |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Gating

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.PolynomialGating
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics

/-- Exact selector-port contribution streams required for one gate class.
This predicate inspects decoded coefficients and does not consult the row's
generated family label. -/
def IsGateAt (row : DecodedRow) (gate : GatePort)
    (selectorColumn : Fin row.columns) : Prop :=
  match gate with
  | .general =>
      expandedFieldTerms (row.port Role.generalSelector.index) =
          [(selectorColumn, 1)] /\
        expandedFieldTerms (row.port Role.evalSelector.index) = []
  | .evaluation =>
      expandedFieldTerms (row.port Role.generalSelector.index) = [] /\
        expandedFieldTerms (row.port Role.evalSelector.index) =
          [(selectorColumn, 1)]

instance (row : DecodedRow) (gate : GatePort)
    (selectorColumn : Fin row.columns) :
    Decidable (IsGateAt row gate selectorColumn) := by
  cases gate <;> unfold IsGateAt <;> infer_instance

/-- Proof-carrying gate classification for one compact row. -/
structure ValidatedGateRow (row : DecodedRow) where
  gate : GatePort
  selectorColumn : Fin row.columns
  shape : IsGateAt row gate selectorColumn

def validateGateAt (row : DecodedRow) (gate : GatePort)
    (selectorColumn : Fin row.columns) : Option (ValidatedGateRow row) :=
  if shape : IsGateAt row gate selectorColumn then
    some ⟨gate, selectorColumn, shape⟩
  else
    none

private theorem action_unit (row : DecodedRow) (port : Fin 13)
    (selectorColumn : Fin row.columns)
    (shape : expandedFieldTerms (row.port port) = [(selectorColumn, 1)])
    (assignment : Fin row.columns → F) :
    action (row.port port) assignment = assignment selectorColumn := by
  simp only [action, shape, List.foldl_cons, List.foldl_nil,
    Fin.zero_add]
  exact Fin.one_mul _

private theorem action_empty (row : DecodedRow) (port : Fin 13)
    (shape : expandedFieldTerms (row.port port) = [])
    (assignment : Fin row.columns → F) :
    action (row.port port) assignment = 0 := by
  simp [action, shape]

/-- Selector factorization follows from the decoded contribution streams,
not from the generated family tag. -/
theorem residual_eq_selector_mul_ungated
    (row : DecodedRow) (validated : ValidatedGateRow row)
    (assignment : Fin row.columns → F) :
    residual row assignment =
      assignment validated.selectorColumn *
        evaluate (ungate validated.gate (rowPoint row assignment)) := by
  rcases validated with ⟨gate, selectorColumn, shape⟩
  cases gate with
  | general =>
      simp only [IsGateAt] at shape
      rw [residual]
      apply evaluate_general_gated
      · exact action_unit row Role.generalSelector.index selectorColumn
          shape.1 assignment
      · exact action_empty row Role.evalSelector.index shape.2 assignment
  | evaluation =>
      simp only [IsGateAt] at shape
      rw [residual]
      apply evaluate_evaluation_gated
      · exact action_empty row Role.generalSelector.index shape.1 assignment
      · exact action_unit row Role.evalSelector.index selectorColumn
          shape.2 assignment

/-- Active compact-row satisfaction is exactly the ungated arithmetic
obligation. -/
theorem residual_zero_iff_ungated_zero_of_selector_one
    (row : DecodedRow) (validated : ValidatedGateRow row)
    (assignment : Fin row.columns → F)
    (selectorOne : assignment validated.selectorColumn = 1) :
    residual row assignment = 0 ↔
      evaluate (ungate validated.gate (rowPoint row assignment)) = 0 := by
  rw [residual_eq_selector_mul_ungated row validated assignment,
    selectorOne, Fin.one_mul]

/-- Nat-carrier form used by the generated compact artifact. -/
theorem natResidual_zero_iff_ungated_zero_of_selector_one
    (row : DecodedRow) (validated : ValidatedGateRow row)
    (assignment : Nat → Nat)
    (selectorOne : assignment validated.selectorColumn.val = 1) :
    natResidual row assignment = 0 ↔
      evaluate
        (ungate validated.gate
          (rowPoint row (fieldAssignment assignment))) = 0 := by
  have fieldSelectorOne :
      fieldAssignment assignment validated.selectorColumn = 1 := by
    apply Fin.ext
    simp only [fieldAssignment, fieldResidue, selectorOne]
    decide
  rw [← residual_fieldAssignment_eq_natResidual]
  exact residual_zero_iff_ungated_zero_of_selector_one row validated
    (fieldAssignment assignment) fieldSelectorOne

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Gating
