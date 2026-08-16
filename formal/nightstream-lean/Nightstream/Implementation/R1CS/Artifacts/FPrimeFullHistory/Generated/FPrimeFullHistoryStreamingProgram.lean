import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingFPrimeProgramSchema

/-! Generated file: exact verifier-owned work schedule for the bounded-width
Nebula F-prime relation.

Owns: the Rust phase codes, compact phase runs, and production stream counts.

Does not own: phase-local constraints, relation dimensions, recursive proof
integration, or security reduction.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingProgram

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgram.Artifact

def profileId : String := "nebula-fprime-streaming-program"
def rawProgram : RawProgram where
  schemaVersion := 8
  stateChunkFields := 1024
  priorStateFrameFields := 83874
  priorStateChunks := 82
  claimFrameFields := 88023
  claimChunkFields := 1024
  claimChunks := 86
  piCcsRounds := 26
  piRlcFamilies := 110
  firstPiRlcFamilyProgramCursor := 199
  successorPrefixFrameFields := 83756
  successorPrefixChunks := 82
  workItemCount := 400
  lifecycleGroupCount := 2
  circuitKindCount := 23
  claimCoordinateOverlayKindCount := 26
  combinedOverlayKindCount := 136
  piRlcFamilyFirstOverlayKind := 26
  piRlcFamilyEvenPhaseKind := 10
  piRlcFamilyOddPhaseKind := 11
  piRlcFamilyBodySourceRows := 146006
  piRlcFamilyBodyEvenSourceRows := 275006
  piRlcFamilyBodyOddSourceRows := 276206
  piRlcFamilyBodyEvenRows := 558932
  piRlcFamilyBodyOddRows := 560132
  piRlcFamilyBodyEvenColumns := 559136
  piRlcFamilyBodyOddColumns := 560336
  piRlcFamilyOverlayRows := 108
  piRlcFamilyOverlayColumns := 33360
  piRlcFamilyLinkFieldCount := 33359
  piRlcFamilyTotalLinkFieldCount := 3669490
  phasePublicLogicalColumns := 641
  phasePublicColumns := 648
  afterStateDigestStart := 1
  afterStateDigestEnd := 257
  beforeStateDigestStart := 257
  beforeStateDigestEnd := 513
  beforeCursorStart := 513
  beforeCursorEnd := 577
  afterCursorStart := 577
  afterCursorEnd := 641
  phasePublicPaddingStart := 641
  phasePublicPaddingEnd := 648
  lifecycleGroupMap := [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  ]
  circuitKindMap := [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  , 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  , 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  , 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
  , 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
  , 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 9, 10
  , 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10
  , 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10
  , 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10
  , 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10
  , 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10
  , 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 14, 15, 16, 17, 17, 17, 17, 17, 17
  , 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17
  , 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17
  , 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17
  , 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 19, 20, 21, 22
  ]
  claimCoordinateOverlayKindMap := [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  , 1, 1, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
  , 21, 22, 23, 24, 25, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]
  piRlcFamilyOverlayKindMap := [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26
  , 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46
  , 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66
  , 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86
  , 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106
  , 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126
  , 127, 128, 129, 130, 131, 132, 133, 134, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]
  claimCoordinateOverlayLinkRuns := [
    { overlayKind := 1, phaseKind := 3, chunkIndex := 1, activeOffsetStart := 0, activeFieldCount := 0 }
  , { overlayKind := 2, phaseKind := 4, chunkIndex := 85, activeOffsetStart := 0, activeFieldCount := 0 }
  , { overlayKind := 3, phaseKind := 3, chunkIndex := 0, activeOffsetStart := 383, activeFieldCount := 52 }
  , { overlayKind := 4, phaseKind := 3, chunkIndex := 60, activeOffsetStart := 987, activeFieldCount := 37 }
  , { overlayKind := 5, phaseKind := 3, chunkIndex := 61, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 6, phaseKind := 3, chunkIndex := 62, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 7, phaseKind := 3, chunkIndex := 63, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 8, phaseKind := 3, chunkIndex := 64, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 9, phaseKind := 3, chunkIndex := 65, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 10, phaseKind := 3, chunkIndex := 66, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 11, phaseKind := 3, chunkIndex := 67, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 12, phaseKind := 3, chunkIndex := 68, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 13, phaseKind := 3, chunkIndex := 69, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 14, phaseKind := 3, chunkIndex := 70, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 15, phaseKind := 3, chunkIndex := 71, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 16, phaseKind := 3, chunkIndex := 72, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 17, phaseKind := 3, chunkIndex := 73, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 18, phaseKind := 3, chunkIndex := 74, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 19, phaseKind := 3, chunkIndex := 75, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 20, phaseKind := 3, chunkIndex := 76, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 21, phaseKind := 3, chunkIndex := 77, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 22, phaseKind := 3, chunkIndex := 78, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 23, phaseKind := 3, chunkIndex := 79, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 24, phaseKind := 3, chunkIndex := 80, activeOffsetStart := 0, activeFieldCount := 1024 }
  , { overlayKind := 25, phaseKind := 3, chunkIndex := 81, activeOffsetStart := 0, activeFieldCount := 651 }
  ]
  piRlcFamilyOverlayLinkRuns := [
    { phaseFieldStart := 46055, overlayFieldStart := 1, outerCount := 1, phaseStride := 41, overlayStride := 41, fieldCount := 41 }
  , { phaseFieldStart := 46096, overlayFieldStart := 42, outerCount := 810, phaseStride := 122, overlayStride := 41, fieldCount := 41 }
  , { phaseFieldStart := 144918, overlayFieldStart := 33252, outerCount := 1, phaseStride := 108, overlayStride := 108, fieldCount := 108 }
  ]
  runs := [
    { phaseCode := 0, firstIndex := 0, count := 1 }
  , { phaseCode := 11, firstIndex := 0, count := 82 }
  , { phaseCode := 1, firstIndex := 0, count := 86 }
  , { phaseCode := 2, firstIndex := 0, count := 1 }
  , { phaseCode := 3, firstIndex := 0, count := 26 }
  , { phaseCode := 4, firstIndex := 0, count := 1 }
  , { phaseCode := 5, firstIndex := 0, count := 1 }
  , { phaseCode := 6, firstIndex := 0, count := 1 }
  , { phaseCode := 7, firstIndex := 0, count := 110 }
  , { phaseCode := 8, firstIndex := 0, count := 1 }
  , { phaseCode := 9, firstIndex := 0, count := 1 }
  , { phaseCode := 10, firstIndex := 0, count := 1 }
  , { phaseCode := 16, firstIndex := 0, count := 1 }
  , { phaseCode := 14, firstIndex := 0, count := 1 }
  , { phaseCode := 18, firstIndex := 0, count := 82 }
  , { phaseCode := 12, firstIndex := 0, count := 1 }
  , { phaseCode := 13, firstIndex := 0, count := 1 }
  , { phaseCode := 15, firstIndex := 0, count := 1 }
  , { phaseCode := 17, firstIndex := 0, count := 1 }
  ]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingProgram
