import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact

namespace NightstreamTests.FPrimeFullHistorySelectiveCcsArtifactDecoder

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder

private def rows : Nat := 54
private def columns : Nat := 5
private def seed : List Nat := List.range 32

private def plainCsc : RawCsc where
  rows := rows
  columns := columns
  colPtr := [0, 1, 1, 1, 1, 1]
  rowIdx := [0]
  vals := [7]

private def emptyCsc : RawCsc where
  rows := rows
  columns := columns
  colPtr := List.replicate (columns + 1) 0
  rowIdx := []
  vals := []

private def seededBlock : RawSeededBlock where
  rowStart := 0
  wordStarts := [1, 3]
  wordWidth := 2
  kappa := 1
  messageCols := 1
  chunkSize := 1
  chunkSeedsByRow := [[seed]]
  superneoTransformedColumns := false

private def geometricRun : RawGeometricRun where
  row := rows - 1
  columnStart := 0
  length := 2
  initial := 2
  ratio := 3

private def compactMatrix : RawMatrix :=
  .cscWithSeededPhi81 emptyCsc [seededBlock] [geometricRun]

/-- One accepted sentinel exercises both supported Rust tags and both compact
payload families without claiming that the values came from production. -/
private def mixedBundle : RawBundle where
  schemaVersion := 1
  rows := rows
  columns := columns
  matrices := compactMatrix :: List.replicate 12 (.csc plainCsc)

example : (decodeProductionBundle 4 mixedBundle).isSome = true := by
  native_decide

private def identityBundle : RawBundle :=
  { mixedBundle with
    matrices := .identity rows :: List.replicate 12 (.csc plainCsc) }

/-- Identity is preserved by the wire type and rejected by the compact
selective decoder rather than silently expanded or retagged. -/
example : (decodeProductionBundle 4 identityBundle).isNone = true := by
  native_decide

private def emptyCompactBundle : RawBundle :=
  { mixedBundle with
    matrices :=
      .cscWithSeededPhi81 emptyCsc [] [] ::
        List.replicate 12 (.csc plainCsc) }

/-- A compact tag with no compact payload cannot alias the ordinary CSC tag. -/
example : (decodeProductionBundle 4 emptyCompactBundle).isNone = true := by
  native_decide

private def malformedSeedBlock : RawSeededBlock :=
  { seededBlock with chunkSeedsByRow := [[List.range 31]] }

private def malformedSeedBundle : RawBundle :=
  { mixedBundle with
    matrices :=
      .cscWithSeededPhi81 emptyCsc [malformedSeedBlock] [geometricRun] ::
        List.replicate 12 (.csc plainCsc) }

example : (decodeProductionBundle 4 malformedSeedBundle).isNone = true := by
  native_decide

private def noncanonicalCsc : RawCsc :=
  { plainCsc with vals := [goldilocksModulus] }

private def noncanonicalFieldBundle : RawBundle :=
  { mixedBundle with
    matrices := .csc noncanonicalCsc :: List.replicate 12 (.csc plainCsc) }

/-- Canonical decoding rejects the modulus itself instead of reducing it to
the same field element as zero. -/
example : (decodeProductionBundle 4 noncanonicalFieldBundle).isNone = true := by
  native_decide

private def wrongArityBundle : RawBundle :=
  { mixedBundle with matrices := List.replicate 12 (.csc plainCsc) }

example : (decodeProductionBundle 4 wrongArityBundle).isNone = true := by
  native_decide

end NightstreamTests.FPrimeFullHistorySelectiveCcsArtifactDecoder
