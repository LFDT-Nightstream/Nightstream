import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRowsSchema

/-! Generated file: compact receipt for the exhaustive normalized production
PiRLC opening-row scan.

Owns: exact source trace geometry, active digit-domain rows, zero-word rows,
two-trit canonical rows, final opening slots, chunk classes, and nonzero
censuses for both parity arms.

Does not own: assignment values, outer norm authority, semantic canonicality,
recursive orchestration, or lifecycle soundness. Lean checks the arithmetic
properties of this inert receipt.

Emits constraints: no. Rust checks every selected source and final matrix row
before it renders this data.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRowsSchema

def audit : RawAudit where
  schemaVersion := 1
  armCount := 2
  openingCount := 810
  digitCount := 41
  borrowCount := 20
  chunkCount := 21
  sourceZeroRowStart := 43794
  sourceZeroDigitStart := 46055
  sourceFieldStart := 1451
  sourceDigitStart := 46096
  sourceDigitStride := 122
  sourceCanonicalRowStart := 43835
  sourceCanonicalRowStride := 124
  centeredRowStart := 2
  centeredRowCount := 16605
  zeroEmittedStarts := [78090, 202066]
  canonicalEmittedStarts := [141262, 265410]
  selectorColumns := [648, 649]
  finalDigitStart := 19332
  finalDigitStride := 41
  finalZeroStart := 1059804
  finalBorrowStart := 1059845
  finalBorrowStride := 20
  finalRows := 282459
  finalColumns := 2521314
  normalizedChunkBounds := [3, 0, 3, 3, 3, 0, 1, 3, 1, 2, 4, 3, 2, 1, 3, 0, 0, 0, 3, 4, 1]
  complementedChunks := [false, false, false, true, true, false, true, false, false, false, false, false, true, true, true, true, false, true, true, false, false]
  sourceZeroNnz := [41, 41, 0]
  finalPortNnz := [46980, 50707, 50707, 82, 46980, 46980, 50625, 16605, 8100, 6480, 3240, 12960, 3240]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows
