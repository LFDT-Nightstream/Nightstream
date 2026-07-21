import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Schema

/-! Generated file: exact fixed-point final ring-alignment rows.

Owns: all 52 proof-free thirteen-port rows emitted after the final selective
column allocation to align the relation width to `D = 64`.

Does not own: the earlier private-layout padding, decoding, row semantics,
constant-one authority, CCS/CE membership, commitment alignment, or row
removal. Do not hand-edit.

Emits constraints: no.

| Artifact field | Exact source | Equation ownership |
|---|---|---|
| `firstEmittedRow` | final emitter row cursor | first final alignment row |
| `runIndex` | compiler ownership ledger | unique ring-padding run |
| `rawRows` | final thirteen-port emitter | `-(z[0] * z[11725454+i])` for `i < 52` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.RingPaddingRows

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized

def firstEmittedRow : Nat := 14946859

def runIndex : Nat := 258519

set_option maxRecDepth 100000 in
def rawRows : List RawRow := [
  { schemaVersion := 1
    rows := 14946911
    columns := 11725506
    emittedRow := 14946859
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725454, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946860
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725455, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946861
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725456, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946862
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725457, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946863
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725458, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946864
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725459, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946865
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725460, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946866
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725461, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946867
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725462, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946868
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725463, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946869
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725464, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946870
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725465, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946871
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725466, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946872
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725467, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946873
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725468, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946874
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725469, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946875
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725470, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946876
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725471, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946877
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725472, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946878
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725473, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946879
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725474, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946880
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725475, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946881
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725476, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946882
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725477, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946883
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725478, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946884
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725479, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946885
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725480, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946886
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725481, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946887
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725482, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946888
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725483, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946889
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725484, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946890
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725485, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946891
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725486, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946892
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725487, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946893
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725488, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946894
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725489, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946895
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725490, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946896
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725491, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946897
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725492, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946898
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725493, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946899
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725494, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946900
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725495, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946901
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725496, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946902
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725497, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946903
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725498, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946904
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725499, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946905
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725500, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946906
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725501, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946907
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725502, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946908
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725503, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946909
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725504, coefficient := 1 }], geometric := [] }
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
    rows := 14946911
    columns := 11725506
    emittedRow := 14946910
    runIndex := 258519
    family := .ringPadding
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 11725505, coefficient := 1 }], geometric := [] }
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
