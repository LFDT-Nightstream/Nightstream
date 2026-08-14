import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsRowSchema

/-!
Wire schema for run-compressed selector coverage of one selective CCS
relation.

Owns: literal dimensions, selector columns, ordered sparse-polynomial syntax,
exclusive owner intervals, and final selector-matrix support intervals.

Does not own: validity, field decoding, family truth, Rust conformance,
production values, branch semantics, constraint necessity, or row removal.

Emits constraints: no.

Authority boundary: every field is untrusted data until the handwritten
decoder checks interval partitions, owner/gate reconciliation, canonical
polynomial syntax, and selector-column separation.

| Wire branch | Rust source | Preserved data |
|---|---|---|
| RawPolynomialTerm | final Structure.f | canonical coefficient and 13 exponents |
| RawOwnerRun | exclusive emitter ledger | interval, family, optional arm |
| RawGateRun | final selector CSC matrices | interval, port, column, coefficient |
| RawCoverage | checked snapshot | dimensions and the three compact lists |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.Wire

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire

inductive RawGatePort where
  | general
  | evaluation
  | generalEvaluation
deriving DecidableEq, Repr

structure RawOwnerRun where
  start : Nat
  stop : Nat
  family : RawFamily
  arm : Option Nat
deriving DecidableEq, Repr

structure RawGateRun where
  start : Nat
  stop : Nat
  port : RawGatePort
  column : Nat
  coefficient : Nat
deriving DecidableEq, Repr

structure RawPolynomialTerm where
  coefficient : Nat
  exponents : List Nat
deriving DecidableEq, Repr

structure RawCoverage where
  schemaVersion : Nat
  rows : Nat
  columns : Nat
  selectorColumns : List Nat
  polynomialArity : Nat
  polynomialTerms : List RawPolynomialTerm
  ownerRuns : List RawOwnerRun
  gateRuns : List RawGateRun
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.Wire
