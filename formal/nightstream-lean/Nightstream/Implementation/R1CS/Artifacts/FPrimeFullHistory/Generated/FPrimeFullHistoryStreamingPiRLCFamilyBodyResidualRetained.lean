import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetainedSchema

/-! Generated file: compact receipt for the exhaustive normalized production
PiRLC residual retained-row scan.

Owns: dimensions, the source row interval, source and final slot starts,
low-norm widths and radices, retained row starts, selector columns, and exact
nonzero censuses observed by the Rust scan.

Does not own: semantic truth, matrix authority, assignment values, selector
authority, the local commitment output, recursive orchestration, or lifecycle
soundness. Lean recomputes the arithmetic properties of this inert receipt.

Emits constraints: no. Rust checks every selected source and final matrix row
before it renders this data.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetainedSchema

def audit : RawAudit where
  schemaVersion := 1
  sourceRowStart := 144277
  sourceRows := 108
  localColumns := 146224
  sourceColumnShift := 640
  finalRows := 282459
  finalColumns := 2521314
  selectorColumns := [648, 649]
  emittedStarts := [78133, 202109]
  sourceStarts := [144918, 145026, 145134]
  finalStarts := [1076091, 1078575, 1081059]
  widths := [23, 23, 23]
  radices := [7, 7, 7]
  sourceNnz := [324, 108, 0]
  finalPortNnz := [0, 216, 14904, 216, 0, 0, 0, 0, 0, 0, 0, 0, 0]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained
