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
  sourceRows := 49626
  localColumns := 51463
  sourceColumnShift := 640
  finalRows := 491046
  finalColumns := 8858862
  selectorColumns := [648, 649]
  emittedStarts := [19830, 255341]
  sourceStarts := [641, 1559, 2477, 2531]
  finalStarts := [702, 38340, 75978, 78192]
  widths := [41, 41, 41, 41]
  radices := [3, 3, 3, 3]
  sourceNnz := [99198, 117504, 49626]
  finalPortNnz := [0, 99252, 4164156, 9635328, 4069332, 0, 0, 0, 0, 0, 0, 0, 0]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyAlgebraRetained
