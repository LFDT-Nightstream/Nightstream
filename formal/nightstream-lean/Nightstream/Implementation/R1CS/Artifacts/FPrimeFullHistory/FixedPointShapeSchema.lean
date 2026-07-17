import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveSelectorCoverageSchema

/-!
Artifact-owned wire schema for one stabilized selective-relation header.

Owns: untrusted fixed-point input and shape-output headers; the materialized
header plus its actual emitted matrix count; and exact sparse-polynomial syntax
fields.

Does not own: validity, fixed-point convergence, semantic matrix count,
polynomial meaning, row-domain policy, Rust conformance, matrix payloads,
assignment ordering, relation acceptance, costs, or row removal.

Emits constraints: no.

Authority boundary: every field is untrusted data. Handwritten correspondence
must compare it with independent semantics before constructing a relation
shape. A header equality or digest cannot establish matrix contents.

| Wire field | Intended Rust source | Preserved data |
|---|---|---|
| `terminalInput` | final fixed-point round input | rows, columns, public width, complete polynomial |
| `selectiveOutput` | final shape-only selective result | same complete relation header |
| `materialized.verifier` | emitted selective relation | same complete verifier header |
| `materialized.matrixCount` | emitted selective matrices | actual matrix-vector length, independent of polynomial arity |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Wire

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.Wire

/-- Raw relation information whose meaning must be checked independently. -/
structure RawHeader where
  rows : Nat
  columns : Nat
  publicInputLength : Nat
  polynomialArity : Nat
  polynomialTerms : List RawPolynomialTerm
deriving DecidableEq, Repr

/-- Raw emitted structure header. Matrix count is separate because agreement
with polynomial arity is a checked obligation, not a representation identity. -/
structure RawMaterializedHeader where
  verifier : RawHeader
  matrixCount : Nat
deriving DecidableEq, Repr

/-- Compact final-round snapshot. It deliberately excludes profiler totals,
digests, stage labels, and caller-declared stabilization flags. -/
structure RawSnapshot where
  schemaVersion : Nat
  terminalInput : RawHeader
  selectiveOutput : RawHeader
  materialized : RawMaterializedHeader
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Wire
