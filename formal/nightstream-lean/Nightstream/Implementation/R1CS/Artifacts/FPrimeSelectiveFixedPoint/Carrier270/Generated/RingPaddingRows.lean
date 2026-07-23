import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Schema

/-! Generated file: exact fixed-point final ring-alignment rows.

Owns: all 28 proof-free thirteen-port rows emitted after the final selective
column allocation to align the relation width to `D = 54`.

Does not own: the earlier private-layout padding, decoding, row semantics,
constant-one authority, CCS/CE membership, commitment alignment, or row
removal. Do not hand-edit.

Emits constraints: no.

| Artifact field | Exact source | Equation ownership |
|---|---|---|
| `firstEmittedRow` | final emitter row cursor | first final alignment row |
| `runIndex` | compiler ownership ledger | unique ring-padding run |
| `rawRows` | final thirteen-port emitter | `-(z[0] * z[firstPaddingColumn+i])` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.RingPaddingRows

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized

def relationRows : Nat := 14944219

def relationColumns : Nat := 11437038

def firstPaddingColumn : Nat := 11437010

def paddingWidth : Nat := 28

def firstEmittedRow : Nat := 14944191

def runIndex : Nat := 248374

set_option maxRecDepth 100000 in
def rawRows : List RawRow := [
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944191
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437010, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944192
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437011, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944193
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437012, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944194
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437013, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944195
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437014, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944196
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437015, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944197
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437016, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944198
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437017, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944199
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437018, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944200
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437019, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944201
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437020, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944202
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437021, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944203
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437022, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944204
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437023, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944205
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437024, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944206
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437025, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944207
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437026, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944208
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437027, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944209
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437028, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944210
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437029, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944211
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437030, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944212
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437031, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944213
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437032, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944214
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437033, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944215
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437034, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944216
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437035, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944217
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437036, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
,
  { schemaVersion := 1
    rows := 14944219
    columns := 11437038
    emittedRow := 14944218
    runIndex := 248374
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11437037, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      ] }
]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.RingPaddingRows
