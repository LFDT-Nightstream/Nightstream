import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Schema
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Decoder

/-!
Fail-closed decoder for one compact selective row in the fixed-point `y_zcol`
projection artifact.

Owns: schema and thirteen-port checks, positive dimensions, typed emitted-row
and explicit-term coordinates, canonical nonzero Goldilocks coefficients, and
positive in-bounds geometric runs with canonical nonzero initial and ratio.

Does not own: generated-row truth, fragment-family truth, selector truth,
source-column meaning, rewrite semantics, protocol authority, or row removal.

Emits constraints: no.

Unlike the full materialized-row decoder, this decoder preserves geometric
runs. It neither trusts modular aliases nor silently truncates a run at the
relation boundary.

| Decoder leaf | Mathematical obligation | Authority class |
|---|---|---|
| sparse term | bounded column and canonical nonzero coefficient | checked |
| geometric run | positive length and exact in-bounds interval | checked |
| compact row | fixed schema and thirteen ordered ports | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized

private abbrev decodeField :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField

def supportedSchemaVersion : Nat := 1

structure DecodedTerm (columns : Nat) where
  column : Fin columns
  coefficient : F
  coefficientNonzero : coefficient ≠ 0

structure DecodedGeometricRun (columns : Nat) where
  columnStart : Nat
  length : Nat
  lengthPositive : 0 < length
  endBound : columnStart + length ≤ columns
  initial : F
  ratio : F
  initialNonzero : initial ≠ 0
  ratioNonzero : ratio ≠ 0

def DecodedGeometricRun.column {columns : Nat}
    (run : DecodedGeometricRun columns) (offset : Fin run.length) :
    Fin columns :=
  ⟨run.columnStart + offset.val,
    Nat.lt_of_lt_of_le (Nat.add_lt_add_left offset.isLt run.columnStart)
      run.endBound⟩

structure DecodedPort (columns : Nat) where
  explicit : List (DecodedTerm columns)
  geometric : List (DecodedGeometricRun columns)

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
    let coefficient ← decodeField raw.coefficient
    if coefficientNonzero : coefficient ≠ 0 then
      pure
        { column := ⟨raw.column, columnInRange⟩
          coefficient
          coefficientNonzero }
    else
      none
  else
    none

private def decodeGeometricRun (columns : Nat) (raw : RawGeometricRun) :
    Option (DecodedGeometricRun columns) :=
  if lengthPositive : 0 < raw.length then
    if endBound : raw.columnStart + raw.length ≤ columns then do
      let initial ← decodeField raw.initial
      let ratio ← decodeField raw.ratio
      if initialNonzero : initial ≠ 0 then
        if ratioNonzero : ratio ≠ 0 then
          pure
            { columnStart := raw.columnStart
              length := raw.length
              lengthPositive
              endBound
              initial
              ratio
              initialNonzero
              ratioNonzero }
        else
          none
      else
        none
    else
      none
  else
    none

def decodePort (columns : Nat) (raw : RawPort) :
    Option (DecodedPort columns) := do
  let explicit ← raw.explicit.mapM (decodeTerm columns)
  let geometric ← raw.geometric.mapM (decodeGeometricRun columns)
  pure { explicit, geometric }

/-! Small kernel lemmas for structural rows. These keep downstream artifact
certificates on proof-free raw records; callers never need to evaluate a list
of proof-carrying decoded ports with `native_decide`. -/

def emptyDecodedPort (columns : Nat) : DecodedPort columns :=
  { explicit := [], geometric := [] }

def unitDecodedTerm (columns column : Nat) (columnInRange : column < columns) :
    DecodedTerm columns :=
  { column := ⟨column, columnInRange⟩
    coefficient := 1
    coefficientNonzero := by decide }

def unitDecodedPort (columns column : Nat) (columnInRange : column < columns) :
    DecodedPort columns :=
  { explicit := [unitDecodedTerm columns column columnInRange]
    geometric := [] }

def canonicalDecodedTerm (columns column coefficient : Nat)
    (columnInRange : column < columns)
    (coefficientCanonical : coefficient < goldilocksModulus)
    (coefficientNonzero :
      (⟨coefficient, coefficientCanonical⟩ : F) ≠ 0) :
    DecodedTerm columns :=
  { column := ⟨column, columnInRange⟩
    coefficient := ⟨coefficient, coefficientCanonical⟩
    coefficientNonzero }

/-- Generic kernel fact for one canonical nonzero explicit term. Downstream
artifact proofs use this theorem instead of asking `native_decide` to evaluate
proof-carrying decoded rows. -/
theorem decodeTerm_canonical (columns column coefficient : Nat)
    (columnInRange : column < columns)
    (coefficientCanonical : coefficient < goldilocksModulus)
    (coefficientNonzero :
      (⟨coefficient, coefficientCanonical⟩ : F) ≠ 0) :
    decodeTerm columns { column, coefficient } =
      some (canonicalDecodedTerm columns column coefficient columnInRange
        coefficientCanonical coefficientNonzero) := by
  unfold decodeTerm
  simp only
  rw [dif_pos columnInRange]
  unfold decodeField
  rw [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField,
    dif_pos coefficientCanonical]
  change
    (if h : (⟨coefficient, coefficientCanonical⟩ : F) ≠ 0 then
        some (canonicalDecodedTerm columns column coefficient columnInRange
          coefficientCanonical h)
      else none) =
      some (canonicalDecodedTerm columns column coefficient columnInRange
        coefficientCanonical coefficientNonzero)
  rw [dif_pos coefficientNonzero]

@[simp] theorem decodePort_empty (columns : Nat) :
    decodePort columns { explicit := [], geometric := [] } =
      some (emptyDecodedPort columns) := by
  rfl

@[simp] theorem decodePort_unit (columns column : Nat)
    (columnInRange : column < columns) :
    decodePort columns
        { explicit := [{ column, coefficient := 1 }], geometric := [] } =
      some (unitDecodedPort columns column columnInRange) := by
  have oneCanonical : 1 < goldilocksModulus := by decide
  have modulusNeOne : goldilocksModulus ≠ 1 := by decide
  have decodedOne :
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField 1 =
        some (1 : F) := by
    rw [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField,
      dif_pos oneCanonical]
    apply congrArg some
    apply Fin.ext
    rfl
  simp [decodePort, decodeTerm, decodeField, columnInRange,
    decodedOne, modulusNeOne, unitDecodedPort, unitDecodedTerm]

/-- Decode one compact row without reducing, dropping, or repairing any
artifact field. Duplicate contributions remain additive stream entries. -/
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

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Decoder
