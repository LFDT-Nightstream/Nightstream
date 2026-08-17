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
  sourceRowStart := 163501
  sourceRows := 108
  localColumns := 165664
  sourceColumnShift := 640
  finalRows := 491046
  finalColumns := 8858862
  selectorColumns := [648, 649]
  emittedStarts := [69499, 305010]
  sourceStarts := [164142, 164250, 164358]
  finalStarts := [2129127, 2133555, 2137983]
  widths := [41, 41, 41]
  radices := [3, 3, 3]
  sourceNnz := [324, 108, 0]
  finalPortNnz := [0, 216, 26568, 216, 0, 0, 0, 0, 0, 0, 0, 0, 0]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained
