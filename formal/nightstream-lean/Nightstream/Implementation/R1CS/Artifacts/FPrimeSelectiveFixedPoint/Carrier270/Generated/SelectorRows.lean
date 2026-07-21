import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Schema

/-! Generated file: exact fixed-point selector rows.

Owns: three proof-free selector-domain rows and one proof-free selector-total
row projected from the prepared selective emitter.

Does not own: decoding, selector values, retained-row coverage, branch
semantics, CCS/CE membership, or row removal. Do not hand-edit.

Emits constraints: no.

| Artifact field | Exact source | Equation ownership |
|---|---|---|
| `rawRows[0..3]` | selector-domain owner | Boolean selector residuals |
| `rawRows[3]` | one-hot owner | selector sum equals constant one |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.SelectorRows

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized

def rawRows : List RawRow := [
  { schemaVersion := 1
    rows := 14946911
    columns := 11725506
    emittedRow := 0
    runIndex := 0
    family := .selectorDomain
    arm := none
    ports := [
        { explicit := [{ column := 270, coefficient := 1 }], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
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
    emittedRow := 1
    runIndex := 0
    family := .selectorDomain
    arm := none
    ports := [
        { explicit := [{ column := 271, coefficient := 1 }], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
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
    emittedRow := 2
    runIndex := 0
    family := .selectorDomain
    arm := none
    ports := [
        { explicit := [{ column := 272, coefficient := 1 }], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
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
    emittedRow := 4729579
    runIndex := 5
    family := .oneHot
    arm := none
    ports := [
        { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 1 }], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [], geometric := [] }
      , { explicit := [{ column := 0, coefficient := 18446744069414584320 }, { column := 270, coefficient := 1 }, { column := 271, coefficient := 1 }, { column := 272, coefficient := 1 }], geometric := [] }
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

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.SelectorRows
