import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsRowSchema

/-!
Wire schema for one lossless compact row from the selective emitter.

Owns: explicit sparse terms and geometric term runs for each of thirteen
ordered ports, final relation dimensions, emitted-row identity, and diagnostic
compiler-owner metadata.

Does not own: field decoding, sparse-term normalization, row semantics,
selector truth, source-expression correspondence, protocol authority,
constraint necessity, or row removal.

Emits constraints: no.

Authority boundary: Rust projects these records from the same `MatrixTerms`
stream consumed by full CSC construction. Lean must decode every field word,
check bounds, and interpret a run as
`initial * ratio^(column-columnStart)` over its exact half-open interval.

| Wire leaf | Mathematical obligation | Authority class |
|---|---|---|
| explicit term | one sparse `(column, coefficient)` contribution | computed |
| geometric run | one exact half-open coefficient progression | computed |
| raw row | thirteen ordered ports at one emitted row | checked after decoding |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire

structure RawGeometricRun where
  columnStart : Nat
  length : Nat
  initial : Nat
  ratio : Nat
deriving DecidableEq, Repr

structure RawPort where
  explicit : List RawTerm
  geometric : List RawGeometricRun
deriving DecidableEq, Repr

structure RawRow where
  schemaVersion : Nat
  rows : Nat
  columns : Nat
  emittedRow : Nat
  runIndex : Nat
  family : RawFamily
  arm : Option Nat
  ports : List RawPort
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
