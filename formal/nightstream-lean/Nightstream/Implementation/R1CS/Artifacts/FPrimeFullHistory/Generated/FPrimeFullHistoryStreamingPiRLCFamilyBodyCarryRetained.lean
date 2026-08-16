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
  sourceRowStart := 144385
  sourceRows := 1621
  localColumns := 146224
  sourceColumnShift := 640
  finalRows := 282459
  finalColumns := 2521314
  selectorColumns := [648, 649]
  emittedStarts := [78241, 202217]
  sourceStarts := [641, 145242, 146052, 146862, 146863]
  finalStarts := [702, 1083543, 1102173, 1120803, 1120826]
  widths := [23, 23, 23, 23, 23]
  radices := [7, 7, 7, 7, 7]
  sourceNnz := [4053, 1621, 0]
  finalPortNnz := [0, 3242, 150754, 3242, 0, 0, 0, 0, 0, 0, 0, 0, 0]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained
