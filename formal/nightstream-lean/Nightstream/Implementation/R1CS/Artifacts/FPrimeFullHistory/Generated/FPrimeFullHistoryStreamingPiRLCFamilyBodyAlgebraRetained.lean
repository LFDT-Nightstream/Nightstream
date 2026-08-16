import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetainedSchema

/-! Generated file: compact receipt for the exhaustive normalized production
PiRLC algebra retained-row scan.

Owns: dimensions, source and final slot starts, low-norm widths and radices,
retained row starts, selector columns, and exact nonzero censuses observed by
the Rust scan.

Does not own: semantic truth, matrix authority, assignment values, selector
authority, recursive orchestration, or lifecycle soundness. Lean recomputes
the arithmetic properties of this inert receipt.

Emits constraints: no. Rust checks every selected source and final matrix row
before it renders this data.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetainedSchema

def audit : RawAudit where
  schemaVersion := 1
  sourceRows := 43794
  localColumns := 45415
  sourceColumnShift := 640
  finalRows := 282459
  finalColumns := 2521314
  selectorColumns := [648, 649]
  emittedStarts := [34296, 158272]
  sourceStarts := [641, 1451, 2261, 2315]
  finalStarts := [702, 19332, 52542, 53784]
  widths := [23, 41, 23, 23]
  radices := [7, 3, 7, 7]
  sourceNnz := [87534, 103680, 43794]
  finalPortNnz := [0, 87588, 2099628, 6343920, 2014524, 0, 0, 0, 0, 0, 0, 0, 0]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained
