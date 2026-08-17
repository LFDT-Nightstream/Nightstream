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
  openingCount := 918
  digitCount := 41
  borrowCount := 20
  chunkCount := 21
  sourceZeroRowStart := 49626
  sourceZeroDigitStart := 52103
  sourceFieldStart := 1559
  sourceDigitStart := 52144
  sourceDigitStride := 122
  sourceCanonicalRowStart := 49667
  sourceCanonicalRowStride := 124
  centeredRowStart := 2
  centeredRowCount := 0
  zeroEmittedStarts := [69456, 304967]
  canonicalEmittedStarts := [236063, 471746]
  selectorColumns := [648, 649]
  finalDigitStart := 38340
  finalDigitStride := 41
  finalZeroStart := 2110644
  finalBorrowStart := 2110685
  finalBorrowStride := 20
  finalRows := 491046
  finalColumns := 8858862
  normalizedChunkBounds := [3, 0, 3, 3, 3, 0, 1, 3, 1, 2, 4, 3, 2, 1, 3, 0, 0, 0, 3, 4, 1]
  complementedChunks := [false, false, false, true, true, false, true, false, false, false, false, false, true, true, true, true, false, true, true, false, false]
  sourceZeroNnz := [41, 41, 0]
  finalPortNnz := [53244, 38638, 38638, 82, 53244, 53244, 38556, 0, 9180, 7344, 3672, 14688, 3672]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows
