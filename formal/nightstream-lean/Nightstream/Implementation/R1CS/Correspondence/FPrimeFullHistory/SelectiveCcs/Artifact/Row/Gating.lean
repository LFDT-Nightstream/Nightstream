import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.RowAction
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Boolean
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.PolynomialGating

/-!
Contract: coefficient-only selector-gate classification for one decoded row
and its explicit action bridge to an interpreted compact relation.

Owns: exact active/inactive selector-port shapes, factorization of the decoded
row residual, an extensional thirteen-port action-equality boundary, and
transport of selector factorization to `Artifact.RowAction.residualAt`.

Does not own: Rust family labels, generated coverage intervals, a concrete
matrix bundle, proof of `ExactRowAction`, arithmetic-family semantics,
constraint necessity, or row removal.

Emits constraints: no.

Authority boundary: `IsGateAt` inspects sparse coefficients and ignores
`DecodedRow.family`. `ExactRowAction` is an explicit premise over all
assignments and all thirteen ports; neither a row label nor a selector-support
summary can manufacture it.

| Stage path | Mathematical obligation | Result |
|---|---|---|
| `f_prime.selective_ccs.artifact.row.gate.shape` | active selector port is one unit term and inactive selector port is empty | `ValidatedGateRow` |
| `f_prime.selective_ccs.artifact.row.gate.factor` | decoded residual is selector times its ungated residual | `residual_eq_selector_mul_ungated` |
| `f_prime.selective_ccs.artifact.row.gate.action` | compact relation and decoded row have equal actions on every port | `ExactRowAction` |
| `f_prime.selective_ccs.artifact.row.gate.refinement` | physical interpreted residual inherits exact selector factorization | `ExactRowAction.residualAt_eq_selector_mul_ungated` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Gating

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.PolynomialGating

/-- Exact coefficient shape required for one of the two selector gate classes.
The other eleven matrix ports remain arbitrary. -/
def IsGateAt (row : DecodedRow) (gate : GatePort)
    (selectorColumn : Fin row.columns) : Prop :=
  match gate with
  | .general =>
      (row.port Role.generalSelector.index).terms =
          [{ column := selectorColumn, coefficient := 1 }] ∧
        (row.port Role.evalSelector.index).terms = []
  | .evaluation =>
      (row.port Role.generalSelector.index).terms = [] ∧
        (row.port Role.evalSelector.index).terms =
          [{ column := selectorColumn, coefficient := 1 }]

instance (row : DecodedRow) (gate : GatePort)
    (selectorColumn : Fin row.columns) :
    Decidable (IsGateAt row gate selectorColumn) := by
  cases gate <;> unfold IsGateAt <;> infer_instance

/-- Proof-carrying coefficient classifier independent of family metadata. -/
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

/-- Selector factorization derived only from decoded sparse coefficients. -/
theorem residual_eq_selector_mul_ungated
    (row : DecodedRow) (validated : ValidatedGateRow row)
    (assignment : Fin row.columns → F) :
    Boolean.residual row assignment =
      assignment validated.selectorColumn *
        evaluate
          (ungate validated.gate (Boolean.rowPoint row assignment)) := by
  rcases validated with ⟨gate, selectorColumn, shape⟩
  cases gate with
  | general =>
      simp only [IsGateAt] at shape
      rw [Boolean.residual]
      apply evaluate_general_gated
      · simp only [Boolean.rowPoint, Boolean.action]
        rw [shape.1]
        simp [Fin.one_mul]
      · simp only [Boolean.rowPoint, Boolean.action]
        rw [shape.2]
        rfl
  | evaluation =>
      simp only [IsGateAt] at shape
      rw [Boolean.residual]
      apply evaluate_evaluation_gated
      · simp only [Boolean.rowPoint, Boolean.action]
        rw [shape.1]
        rfl
      · simp only [Boolean.rowPoint, Boolean.action]
        rw [shape.2]
        simp [Fin.one_mul]

/-- Extensional equality between one decoded sparse row and the same physical
row of an interpreted compact relation. This is the required non-label bridge. -/
structure ExactRowAction
    (row : DecodedRow)
    (relation : InterpretedRelation row.rows row.columns) : Prop where
  matrixImage_eq_action : ∀ assignment port,
    RowAction.matrixImageAt relation assignment row.emittedRow port =
      Boolean.action (row.port port) assignment

theorem ExactRowAction.rowPoint_eq
    {row : DecodedRow}
    {relation : InterpretedRelation row.rows row.columns}
    (exact : ExactRowAction row relation)
    (assignment : Assignment F row.columns) :
    RowAction.rowPoint relation assignment row.emittedRow =
      Boolean.rowPoint row assignment := by
  funext port
  exact exact.matrixImage_eq_action assignment port

theorem ExactRowAction.residualAt_eq_decoded
    {row : DecodedRow}
    {relation : InterpretedRelation row.rows row.columns}
    (exact : ExactRowAction row relation)
    (assignment : Assignment F row.columns) :
    RowAction.residualAt relation assignment row.emittedRow =
      Boolean.residual row assignment := by
  rw [RowAction.residualAt_eq_evaluate, exact.rowPoint_eq assignment]
  rfl

/-- Exact matrix-action refinement transports coefficient-derived selector
factorization to the interpreted relation residual. -/
theorem ExactRowAction.residualAt_eq_selector_mul_ungated
    {row : DecodedRow}
    {relation : InterpretedRelation row.rows row.columns}
    (exact : ExactRowAction row relation)
    (validated : ValidatedGateRow row)
    (assignment : Assignment F row.columns) :
    RowAction.residualAt relation assignment row.emittedRow =
      assignment validated.selectorColumn *
        evaluate
          (ungate validated.gate
            (RowAction.rowPoint relation assignment row.emittedRow)) := by
  calc
    RowAction.residualAt relation assignment row.emittedRow =
        Boolean.residual row assignment :=
      exact.residualAt_eq_decoded assignment
    _ = assignment validated.selectorColumn *
          evaluate
            (ungate validated.gate (Boolean.rowPoint row assignment)) :=
      residual_eq_selector_mul_ungated row validated assignment
    _ = assignment validated.selectorColumn *
          evaluate
            (ungate validated.gate
              (RowAction.rowPoint relation assignment row.emittedRow)) := by
      rw [exact.rowPoint_eq assignment]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Gating
