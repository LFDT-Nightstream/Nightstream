import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsRowSchema

/-! Generated file: one deterministic selective-compiler row fixture.

Owns: exact final-matrix coefficients and diagnostic row-ledger provenance for
the first selector-domain row of the two-arm snapshot test fixture.

Does not own: a production F-prime profile, row-family truth, semantic
soundness, constraint necessity, or permission to remove rows.

Emits constraints: no. Rust materializes the final compact matrices before
rendering; Lean independently decodes and classifies their coefficients.

| Artifact branch | Exact source | Lean consumer |
|---|---|---|
| dimensions and row | final selective structure | fail-closed row decoder |
| thirteen sparse ports | final materialized matrices | coefficient semantics |
| run/family/arm | exclusive emitted-row ledger | diagnostic only |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveCcsSelectorDomainRow

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire

def rawRow : RawRow where
  schemaVersion := 1
  rows := 542
  columns := 1458
  emittedRow := 0
  runIndex := 0
  family := .selectorDomain
  arm := none
  ports := [
    { terms := [{ column := 54, coefficient := 1 }] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveCcsSelectorDomainRow
