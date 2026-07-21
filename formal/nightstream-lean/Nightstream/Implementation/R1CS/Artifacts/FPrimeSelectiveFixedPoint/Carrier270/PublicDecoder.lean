import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.Generated.PublicDecoderChunk0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.Generated.PublicDecoderChunk1

/-!
Artifact-checked public-coordinate decoder for the bounded fixed-point profile.

Owns: fail-closed equality between the two Rust-generated proof-free chunks
and the exact 270-coordinate owner schedule. The chunks contain 256 and 14
records, respectively.

Does not own: assignment values, private coordinates, matrix semantics,
CCS/CE membership, commitment-key alignment, or row removal.

Emits constraints: no.

| Artifact leaf | Obligation | Lean owner |
|---|---|---|
| chunk zero | exact owners for columns `0..256` | `generated_chunk0_exact` |
| chunk one | exact owners for columns `256..270` | `generated_chunk1_exact` |
| joined lookup | every public column has its canonical owner | `generatedCoordinate_exact` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicDecoder

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Wire

def rawChunk0 : List RawCoordinate :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.Chunk0.rawCoordinates

def rawChunk1 : List RawCoordinate :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.Chunk1.rawCoordinates

def generatedTotalColumns0 : Nat :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.Chunk0.totalColumns

def generatedTotalColumns1 : Nat :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.Chunk1.totalColumns

def schemaVersion : Nat := 1
def logicalPublicWidth : Nat := 257
def alignedPublicWidth : Nat := 270
def firstChunkWidth : Nat := 256
def secondChunkWidth : Nat := 14

def expectedCoordinate (column : Nat) : RawCoordinate :=
  if column = 0 then
    { schemaVersion, column, source := .constantOne }
  else if column < logicalPublicWidth then
    { schemaVersion, column, source := .sourceField column }
  else
    { schemaVersion, column, source := .fixedZero }

def expectedChunk0 : List RawCoordinate :=
  (List.range firstChunkWidth).map expectedCoordinate

def expectedChunk1 : List RawCoordinate :=
  (List.range secondChunkWidth).map fun offset =>
    expectedCoordinate (firstChunkWidth + offset)

/-- `native_decide` compares exactly 256 proof-free `RawCoordinate` records. -/
theorem generated_chunk0_exact :
    rawChunk0 = expectedChunk0 := by
  native_decide

/-- `native_decide` compares exactly 14 proof-free `RawCoordinate` records. -/
theorem generated_chunk1_exact :
    rawChunk1 = expectedChunk1 := by
  native_decide

theorem generated_chunk_lengths :
    rawChunk0.length = firstChunkWidth ∧
      rawChunk1.length = secondChunkWidth ∧
      firstChunkWidth + secondChunkWidth = alignedPublicWidth := by
  constructor
  · rw [generated_chunk0_exact]
    simp [expectedChunk0, firstChunkWidth]
  constructor
  · rw [generated_chunk1_exact]
    simp [expectedChunk1, secondChunkWidth]
  · decide

theorem generated_totalColumns_exact :
    generatedTotalColumns0 = 11725506 ∧
      generatedTotalColumns1 = generatedTotalColumns0 := by
  native_decide

def generatedCoordinate (column : Fin alignedPublicWidth) : RawCoordinate :=
  if first : column.val < firstChunkWidth then
    rawChunk0[column.val]'(by
      rw [generated_chunk_lengths.1]
      exact first)
  else
    rawChunk1[column.val - firstChunkWidth]'(by
      rw [generated_chunk_lengths.2.1]
      have columnBound := column.isLt
      simp only [alignedPublicWidth, firstChunkWidth, secondChunkWidth] at columnBound ⊢
      omega)

theorem generatedCoordinate_exact
    (column : Fin alignedPublicWidth) :
    generatedCoordinate column = expectedCoordinate column.val := by
  by_cases first : column.val < firstChunkWidth
  · simp [generatedCoordinate, first, generated_chunk0_exact, expectedChunk0]
  · simp only [generatedCoordinate, dif_neg first]
    have columnBound := column.isLt
    have offsetBound : column.val - firstChunkWidth < secondChunkWidth := by
      simp only [alignedPublicWidth, firstChunkWidth, secondChunkWidth] at columnBound ⊢
      omega
    have restored :
        firstChunkWidth + (column.val - firstChunkWidth) = column.val := by
      simp only [firstChunkWidth] at first ⊢
      omega
    simp only [generated_chunk1_exact, expectedChunk1, List.getElem_map,
      List.getElem_range]
    rw [restored]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicDecoder
