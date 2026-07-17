import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsRowSchema
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Decoder

/-!
Contract: fail-closed decoder for one materialized selective-CCS row.

Owns: schema and arity checks, positive dimensions, typed row/column indices,
canonical nonzero Goldilocks coefficients, and strictly increasing columns in
each of thirteen ports.

Does not own: the Rust exporter, truth of family metadata, matrix semantics,
row classification, protocol soundness, constraint necessity, or row removal.

Emits constraints: no.

| Stage path | Checked property | Successful type |
|---|---|---|
| `f_prime.selective_ccs.artifact.row.term` | bounded column and canonical nonzero coefficient | `DecodedTerm` |
| `f_prime.selective_ccs.artifact.row.port` | strictly increasing unique columns | `DecodedPort` |
| `f_prime.selective_ccs.artifact.row.decode` | version, dimensions, row bound, thirteen ports | `DecodedRow` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire

/-- The sole materialized-row wire version accepted here. -/
def supportedSchemaVersion : Nat := 1

structure DecodedTerm (columns : Nat) where
  column : Fin columns
  coefficient : F
deriving DecidableEq, Repr

structure DecodedPort (columns : Nat) where
  terms : List (DecodedTerm columns)
  columnsStrict : terms.Pairwise
    (fun left right => left.column.val < right.column.val)
  coefficientsNonzero : ∀ term ∈ terms, term.coefficient ≠ 0

structure DecodedRow where
  rows : Nat
  columns : Nat
  rowsPositive : 0 < rows
  columnsPositive : 0 < columns
  emittedRow : Fin rows
  runIndex : Nat
  family : RawFamily
  arm : Option Nat
  ports : Fin 13 → DecodedPort columns

def DecodedRow.port (row : DecodedRow) (port : Fin 13) :
    DecodedPort row.columns :=
  row.ports port

private def decodeTerm (columns : Nat) (raw : RawTerm) :
    Option (DecodedTerm columns) :=
  if columnInRange : raw.column < columns then do
    let coefficient ←
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField
        raw.coefficient
    if coefficient ≠ 0 then
      pure
        { column := ⟨raw.column, columnInRange⟩
          coefficient }
    else
      none
  else
    none

private def DecodedPort.Valid {columns : Nat}
    (terms : List (DecodedTerm columns)) : Prop :=
  terms.Pairwise (fun left right => left.column.val < right.column.val) ∧
    ∀ term ∈ terms, term.coefficient ≠ 0

private instance {columns : Nat} (terms : List (DecodedTerm columns)) :
    Decidable (DecodedPort.Valid terms) := by
  unfold DecodedPort.Valid
  infer_instance

def decodePort (columns : Nat) (raw : RawPort) :
    Option (DecodedPort columns) := do
  let terms ← raw.terms.mapM (decodeTerm columns)
  if valid : DecodedPort.Valid terms then
    pure
      { terms
        columnsStrict := valid.1
        coefficientsNonzero := valid.2 }
  else
    none

/-- Decode without trusting or repairing any row or coefficient metadata. -/
def decodeRow (raw : RawRow) : Option DecodedRow :=
  if version : raw.schemaVersion = supportedSchemaVersion then
    if rowsPositive : 0 < raw.rows then
      if columnsPositive : 0 < raw.columns then
        if rowInRange : raw.emittedRow < raw.rows then do
          let ports ← raw.ports.mapM (decodePort raw.columns)
          if portCount : ports.length = 13 then
            pure
              { rows := raw.rows
                columns := raw.columns
                rowsPositive
                columnsPositive
                emittedRow := ⟨raw.emittedRow, rowInRange⟩
                runIndex := raw.runIndex
                family := raw.family
                arm := raw.arm
                ports := fun port => ports.get ⟨port.val, by
                  rw [portCount]
                  exact port.isLt⟩ }
          else
            none
        else
          none
      else
        none
    else
      none
  else
    none

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
