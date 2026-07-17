import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Decoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Rows

/-!
Contract: semantic classification of one decoded Boolean/domain row.

Owns: sparse-row action on an arbitrary typed assignment, an exact
coefficient-based Boolean shape, and reduction of the full selective
polynomial to its Boolean residual.

Does not own: the Rust family label, the constant-one assignment invariant,
artifact generation, equality with a production matrix row, multiplicity,
protocol minimality, or permission to remove rows.

Emits constraints: no.

| Stage path | Mathematical obligation | Lean result |
|---|---|---|
| `f_prime.selective_ccs.artifact.row.action` | `sum coefficient * assignment[column]` per port | `action` |
| `f_prime.selective_ccs.artifact.row.boolean.shape` | ports 0/1 are exact unit terms; ports 2--12 empty | `ValidatedBooleanRow` |
| `f_prime.selective_ccs.artifact.row.boolean.point` | all thirteen actions equal the independent Boolean point | `rowPoint_eq_booleanPoint` |
| `f_prime.selective_ccs.artifact.row.boolean.residual` | full polynomial reduces to `g*b*(b-1)` | `residual_eq_booleanResidual` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics

def action {columns : Nat} (port : DecodedPort columns)
    (assignment : Fin columns → F) : F :=
  port.terms.foldl
    (fun total term => total + term.coefficient * assignment term.column) 0

def rowPoint (row : DecodedRow)
    (assignment : Fin row.columns → F) : Fin 13 → F :=
  fun port => action (row.port port) assignment

def residual (row : DecodedRow)
    (assignment : Fin row.columns → F) : F :=
  evaluate (rowPoint row assignment)

def IsBooleanAt (row : DecodedRow)
    (bitColumn selectorColumn : Fin row.columns) : Prop :=
  (row.port Role.bit.index).terms =
      [{ column := bitColumn, coefficient := 1 }] ∧
    (row.port Role.generalSelector.index).terms =
      [{ column := selectorColumn, coefficient := 1 }] ∧
    ∀ port : Fin 13,
      port ≠ Role.bit.index →
      port ≠ Role.generalSelector.index →
      (row.port port).terms = []

instance (row : DecodedRow) (bitColumn selectorColumn : Fin row.columns) :
    Decidable (IsBooleanAt row bitColumn selectorColumn) := by
  unfold IsBooleanAt
  infer_instance

/-- Proof-carrying Boolean classification. The candidate columns are accepted
only when every decoded port has the exact coefficient shape. -/
structure ValidatedBooleanRow (row : DecodedRow) where
  bitColumn : Fin row.columns
  selectorColumn : Fin row.columns
  shape : IsBooleanAt row bitColumn selectorColumn

def validateBooleanAt (row : DecodedRow)
    (bitColumn selectorColumn : Fin row.columns) :
    Option (ValidatedBooleanRow row) :=
  if shape : IsBooleanAt row bitColumn selectorColumn then
    some ⟨bitColumn, selectorColumn, shape⟩
  else
    none

theorem rowPoint_eq_booleanPoint
    (row : DecodedRow) (validated : ValidatedBooleanRow row)
    (assignment : Fin row.columns → F) :
    rowPoint row assignment =
      booleanPoint
        (assignment validated.selectorColumn)
        (assignment validated.bitColumn) := by
  funext port
  by_cases bitPort : port = Role.bit.index
  · subst port
    simp only [rowPoint, action]
    rw [validated.shape.1]
    simp [booleanPoint, sparsePoint,
      Role.index, Fin.one_mul, Fin.zero_add]
  · by_cases selectorPort : port = Role.generalSelector.index
    · subst port
      simp only [rowPoint, action]
      rw [validated.shape.2.1]
      simp [booleanPoint, sparsePoint,
        Role.index, Fin.one_mul, Fin.zero_add]
    · simp only [rowPoint, action]
      rw [validated.shape.2.2 port bitPort selectorPort]
      have portNeZero : port ≠ (0 : Fin 13) := by
        simpa only [Role.index] using bitPort
      have portNeOne : port ≠ (1 : Fin 13) := by
        simpa only [Role.index] using selectorPort
      simp [booleanPoint, sparsePoint, portNeZero, portNeOne]

theorem residual_eq_booleanResidual
    (row : DecodedRow) (validated : ValidatedBooleanRow row)
    (assignment : Fin row.columns → F) :
    residual row assignment =
      booleanResidual
        (booleanPoint
          (assignment validated.selectorColumn)
          (assignment validated.bitColumn)) := by
  rw [residual, rowPoint_eq_booleanPoint row validated assignment,
    evaluate_booleanPoint]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
