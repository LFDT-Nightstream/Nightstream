import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyCarryRetainedSchema

/-! Generated file: compact receipt for the exhaustive normalized production
PiRLC carry retained-row scan.

Owns: dimensions, the source row interval, source and final slot starts,
low-norm widths and radices, retained row starts, selector columns, and exact
nonzero censuses observed by the Rust scan.

Does not own: semantic truth, matrix authority, assignment values, selector
authority, challenge range, recursive orchestration, or lifecycle soundness.
Lean recomputes the arithmetic properties of this inert receipt.

Emits constraints: no. Rust checks every selected source and final matrix row
before it renders this data.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyCarryRetainedSchema

def audit : RawAudit where
  schemaVersion := 1
  sourceRowStart := 163609
  sourceRows := 1837
  localColumns := 165664
  sourceColumnShift := 640
  finalRows := 491046
  finalColumns := 8858862
  selectorColumns := [648, 649]
  emittedStarts := [69607, 305118]
  sourceStarts := [641, 164466, 165384, 166302, 166303]
  finalStarts := [702, 2142411, 2180049, 2217687, 2217728]
  widths := [41, 41, 41, 41, 41]
  radices := [3, 3, 3, 3, 3]
  sourceNnz := [4593, 1837, 0]
  finalPortNnz := [0, 3674, 303106, 3674, 0, 0, 0, 0, 0, 0, 0, 0, 0]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained
