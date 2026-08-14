import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SourceStageCoverageSchema

/-! Generated file: exact top-level source-stage census for the production-width
WASM Nebula radix-four recursive arm.

Owns: the five exclusive source-field dispositions and the exact aggregation
of every nonempty physical stage into fourteen reviewed path prefixes.

Does not own: arithmetic-family semantics, path-label authority, individual
source-to-final decoder rules, relation soundness, constraint necessity, or
permission to remove rows or columns.

Emits constraints: no. Rust emits this file after the live compiler checks the
exclusive physical-stage partition and decoder dispositions.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourSourceStageCoverage

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SourceStageCoverage

def profileId : String := "wasm-nebula-radix-four-candidate-v1"
def rawCoverage : RawCoverage where
  schemaVersion := 1
  physicalStages := 6578
  unownedEmptyStages := 3
  sourceFields := 16181176
  direct := 2838233
  decompositionAlias := 5006998
  equalityAlias := 445
  linearDefinition := 86880
  traceEliminated := 8248620
  allocatedCoordinates := 11502388
  owners := [
    { owner := .application, stages := 1, sourceFields := 49789,
       direct := 31500, decompositionAlias := 0, equalityAlias := 0,
       linearDefinition := 1085, traceEliminated := 17204, allocatedCoordinates := 169110 }
  , { owner := .prelude, stages := 1, sourceFields := 25945,
       direct := 4150, decompositionAlias := 0, equalityAlias := 1,
       linearDefinition := 542, traceEliminated := 21252, allocatedCoordinates := 85673 }
  , { owner := .transcript, stages := 1, sourceFields := 16880,
       direct := 2408, decompositionAlias := 0, equalityAlias := 0,
       linearDefinition := 304, traceEliminated := 14168, allocatedCoordinates := 55384 }
  , { owner := .piCcs, stages := 134, sourceFields := 8672348,
       direct := 1499577, decompositionAlias := 2701039, equalityAlias := 444,
       linearDefinition := 57822, traceEliminated := 4413466, allocatedCoordinates := 6687893 }
  , { owner := .runningParentPiDec, stages := 1, sourceFields := 38880,
       direct := 36450, decompositionAlias := 0, equalityAlias := 0,
       linearDefinition := 0, traceEliminated := 2430, allocatedCoordinates := 36450 }
  , { owner := .piRlc, stages := 6420, sourceFields := 2548608,
       direct := 444370, decompositionAlias := 733228, equalityAlias := 0,
       linearDefinition := 24623, traceEliminated := 1346387, allocatedCoordinates := 3275198 }
  , { owner := .piDec, stages := 2, sourceFields := 38880,
       direct := 36450, decompositionAlias := 0, equalityAlias := 0,
       linearDefinition := 0, traceEliminated := 2430, allocatedCoordinates := 36450 }
  , { owner := .pointBinding, stages := 1, sourceFields := 0,
       direct := 0, decompositionAlias := 0, equalityAlias := 0,
       linearDefinition := 0, traceEliminated := 0, allocatedCoordinates := 0 }
  , { owner := .priorLink, stages := 4, sourceFields := 5904,
       direct := 792, decompositionAlias := 448, equalityAlias := 0,
       linearDefinition := 110, traceEliminated := 4554, allocatedCoordinates := 18226 }
  , { owner := .nebula, stages := 1, sourceFields := 110520,
       direct := 17142, decompositionAlias := 14391, equalityAlias := 0,
       linearDefinition := 1404, traceEliminated := 77583, allocatedCoordinates := 246100 }
  , { owner := .accumulator, stages := 6, sourceFields := 4648044,
       direct := 761846, decompositionAlias := 1557508, equalityAlias := 0,
       linearDefinition := 290, traceEliminated := 2328400, allocatedCoordinates := 810186 }
  , { owner := .counters, stages := 1, sourceFields := 324,
       direct := 6, decompositionAlias := 128, equalityAlias := 0,
       linearDefinition := 190, traceEliminated := 0, allocatedCoordinates := 176 }
  , { owner := .output, stages := 1, sourceFields := 25054,
       direct := 3542, decompositionAlias := 256, equalityAlias := 0,
       linearDefinition := 510, traceEliminated := 20746, allocatedCoordinates := 81542 }
  , { owner := .semanticLinks, stages := 1, sourceFields := 0,
       direct := 0, decompositionAlias := 0, equalityAlias := 0,
       linearDefinition := 0, traceEliminated := 0, allocatedCoordinates := 0 }
  ]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourSourceStageCoverage
