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
  schemaVersion := 2
  stateChunkFields := 1024
  priorStateFrameFields := 83874
  priorStateChunks := 82
  claimFrameFields := 88023
  claimChunkFields := 1024
  claimChunks := 86
  piCcsRounds := 26
  piRlcFamilies := 110
  successorPrefixFrameFields := 83756
  successorPrefixChunks := 82
  workItemCount := 400
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
