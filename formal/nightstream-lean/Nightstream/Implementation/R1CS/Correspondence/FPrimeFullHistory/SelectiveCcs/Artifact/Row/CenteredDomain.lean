import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Boolean
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.PackedRows

/-!
Contract: coefficient-derived semantics of one packed centered-domain row.

Owns: the exact G/E/U/A sparse-port classifier, its optional fixed-zero right
coordinate, and equality of all thirteen matrix actions with the independent
centered-pair point.

Does not own: a generated row, a Rust family label, selector activation,
production multiplicity, the Goldilocks nonresidue result, source-column
meaning, constraint necessity, or row removal.

Emits constraints: no.

Assurance tier: model-level. A concrete artifact consumer must supply the
decoded coefficient shape and the production security reduction.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.CenteredDomain

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows

def optionalUnitTerms {columns : Nat} :
    Option (Fin columns) → List (DecodedTerm columns)
  | none => []
  | some column => [{ column, coefficient := 1 }]

def optionalValue {columns : Nat} (assignment : Fin columns → F) :
    Option (Fin columns) → F
  | none => 0
  | some column => assignment column

/-- Exact coefficient shape shared by a two-coordinate row and its odd tail.
`right = none` means that the A port is empty, so its matrix image is zero. -/
def IsCenteredDomainAt (row : DecodedRow)
    (selectorColumn leftColumn : Fin row.columns)
    (rightColumn : Option (Fin row.columns)) : Prop :=
  (row.port Role.generalSelector.index).terms =
      [{ column := selectorColumn, coefficient := 1 }] ∧
    (row.port Role.evalSelector.index).terms =
      [{ column := selectorColumn, coefficient := 1 }] ∧
    (row.port Role.centeredUnit.index).terms =
      [{ column := leftColumn, coefficient := 1 }] ∧
    (row.port Role.a.index).terms = optionalUnitTerms rightColumn ∧
    ∀ port : Fin 13,
      port ≠ Role.generalSelector.index →
      port ≠ Role.evalSelector.index →
      port ≠ Role.centeredUnit.index →
      port ≠ Role.a.index →
      (row.port port).terms = []

instance (row : DecodedRow)
    (selectorColumn leftColumn : Fin row.columns)
    (rightColumn : Option (Fin row.columns)) :
    Decidable (IsCenteredDomainAt row selectorColumn leftColumn rightColumn) := by
  unfold IsCenteredDomainAt
  infer_instance

structure ValidatedCenteredDomainRow (row : DecodedRow) where
  selectorColumn : Fin row.columns
  leftColumn : Fin row.columns
  rightColumn : Option (Fin row.columns)
  shape : IsCenteredDomainAt row selectorColumn leftColumn rightColumn

def validateCenteredDomainAt (row : DecodedRow)
    (selectorColumn leftColumn : Fin row.columns)
    (rightColumn : Option (Fin row.columns)) :
    Option (ValidatedCenteredDomainRow row) :=
  if shape : IsCenteredDomainAt row selectorColumn leftColumn rightColumn then
    some ⟨selectorColumn, leftColumn, rightColumn, shape⟩
  else
    none

theorem rowPoint_eq_centeredPairPoint
    (row : DecodedRow) (validated : ValidatedCenteredDomainRow row)
    (assignment : Fin row.columns → F) :
    rowPoint row assignment =
      centeredPairPoint
        (assignment validated.selectorColumn)
        (assignment validated.leftColumn)
        (optionalValue assignment validated.rightColumn) := by
  funext port
  by_cases generalPort : port = Role.generalSelector.index
  · subst port
    simp only [rowPoint, action]
    rw [validated.shape.1]
    simp [centeredPairPoint, sparsePoint, Role.index]
  · by_cases evaluationPort : port = Role.evalSelector.index
    · subst port
      simp only [rowPoint, action]
      rw [validated.shape.2.1]
      simp [centeredPairPoint, sparsePoint, Role.index]
    · by_cases unitPort : port = Role.centeredUnit.index
      · subst port
        simp only [rowPoint, action]
        rw [validated.shape.2.2.1]
        simp [centeredPairPoint, sparsePoint, Role.index]
      · by_cases rightPort : port = Role.a.index
        · subst port
          simp only [rowPoint, action]
          rw [validated.shape.2.2.2.1]
          cases validated.rightColumn <;>
            simp [optionalUnitTerms, optionalValue, centeredPairPoint,
              sparsePoint, Role.index]
        · simp only [rowPoint, action]
          rw [validated.shape.2.2.2.2 port generalPort evaluationPort
            unitPort rightPort]
          have generalPortValue : port ≠ (1 : Fin 13) := by
            simpa only [Role.index] using generalPort
          have evaluationPortValue : port ≠ (7 : Fin 13) := by
            simpa only [Role.index] using evaluationPort
          have unitPortValue : port ≠ (6 : Fin 13) := by
            simpa only [Role.index] using unitPort
          have rightPortValue : port ≠ (2 : Fin 13) := by
            simpa only [Role.index] using rightPort
          simp [centeredPairPoint, sparsePoint, Role.index,
            generalPortValue, evaluationPortValue, unitPortValue,
            rightPortValue]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.CenteredDomain
