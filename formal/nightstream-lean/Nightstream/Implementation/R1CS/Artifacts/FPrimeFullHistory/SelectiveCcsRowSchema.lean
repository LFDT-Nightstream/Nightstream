/-!
Artifact-owned wire schema for one materialized selective-CCS row.

Owns: untrusted row dimensions, exclusive-run metadata, the diagnostic family
tag, optional arm, and thirteen ordered sparse port rows.

Does not own: field semantics, row-family truth, validity, matrix action,
Rust conformance, protocol necessity, or production values.

Emits constraints: no.

Authority boundary: family and run fields are provenance metadata only. A
semantic consumer must classify the actual port terms after canonical decoding.

| Wire field | Rust source | Semantic status |
|---|---|---|
| dimensions and row | final selective `Structure` | untrusted until bounded |
| run/family/arm | exclusive emitted-row ledger | diagnostic metadata |
| port terms | final `CcsMatrix.materialize_row` | untrusted until canonical decoding |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire

/-- Exact emitted-family vocabulary. This tag never proves the row shape. -/
inductive RawFamily where
  | selectorDomain
  | sharedDomain
  | armDomain
  | oneHot
  | publicPadding
  | privatePadding
  | retained
  | poseidon2
  | centeredUnit
  | shiftedTernaryCanonical
  | polynomialEvaluation
  | productSum
  | ringPadding
deriving DecidableEq, Repr

/-- One untrusted canonical-field word at one untrusted column. -/
structure RawTerm where
  column : Nat
  coefficient : Nat
deriving DecidableEq, Repr

/-- One ordered sparse port row. -/
structure RawPort where
  terms : List RawTerm
deriving DecidableEq, Repr

/-- Literal row artifact emitted from a checked selective snapshot. -/
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

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire
