/-!
Schema for the compact normalized PiRLC opening-row scan receipt.

This file owns inert artifact data only. It does not validate matrix content,
row semantics, assignments, outer norm authority, or lifecycle state.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRowsSchema

def supportedSchemaVersion : Nat := 1

structure RawAudit where
  schemaVersion : Nat
  armCount : Nat
  openingCount : Nat
  digitCount : Nat
  borrowCount : Nat
  chunkCount : Nat
  sourceZeroRowStart : Nat
  sourceZeroDigitStart : Nat
  sourceFieldStart : Nat
  sourceDigitStart : Nat
  sourceDigitStride : Nat
  sourceCanonicalRowStart : Nat
  sourceCanonicalRowStride : Nat
  centeredRowStart : Nat
  centeredRowCount : Nat
  zeroEmittedStarts : List Nat
  canonicalEmittedStarts : List Nat
  selectorColumns : List Nat
  finalDigitStart : Nat
  finalDigitStride : Nat
  finalZeroStart : Nat
  finalBorrowStart : Nat
  finalBorrowStride : Nat
  finalRows : Nat
  finalColumns : Nat
  normalizedChunkBounds : List Nat
  complementedChunks : List Bool
  sourceZeroNnz : List Nat
  finalPortNnz : List Nat
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRowsSchema
