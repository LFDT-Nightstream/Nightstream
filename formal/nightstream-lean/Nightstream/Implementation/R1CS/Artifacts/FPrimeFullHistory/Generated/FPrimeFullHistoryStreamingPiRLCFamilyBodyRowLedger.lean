import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema

/-! Generated file: exact compact row-owner ledger for the normalized
production PiRLC parity bodies.

Owns: fixed emitted intervals, retained source-to-emitted intervals, and
affine rewrite batches copied from the production selective compiler audit.

Does not own: row semantics, port images, matrix actions, assignment values,
selector authority, or lifecycle soundness.

Emits constraints: no. Rust expands this data and checks exact equality with
the compiler audit before rendering it. Lean checks all ownership covers.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema

def ledger : RawLedger where
  schemaVersion := 1
  rows := 491046
  columns := 8858862
  evenSourceRows := 1300897
  oddSourceRows := 1302097
  rewriteCount := 14638
  evenLinearDefinitionCount := 4520
  oddLinearDefinitionCount := 4520
  fixedRuns :=
    [
      { start := 0, length := 2, family := .selectorDomain, arm := none }
    , { start := 2, length := 18360, family := .sharedDomain, arm := none }
    , { start := 18362, length := 704, family := .armDomain, arm := some 0 }
    , { start := 19066, length := 704, family := .armDomain, arm := some 1 }
    , { start := 19770, length := 1, family := .oneHot, arm := none }
    , { start := 19771, length := 7, family := .publicPadding, arm := none }
    , { start := 19778, length := 52, family := .privatePadding, arm := none }
    , { start := 491024, length := 22, family := .ringPadding, arm := none }
    ]
  retainedRuns :=
    [
      { arm := 0, sourceStart := 0, length := 49626, emittedStart := 19830 }
    , { arm := 0, sourceStart := 49626, length := 41, emittedStart := 69456 }
    , { arm := 0, sourceStart := 163499, length := 2, emittedStart := 69497 }
    , { arm := 0, sourceStart := 163501, length := 108, emittedStart := 69499 }
    , { arm := 0, sourceStart := 163609, length := 1837, emittedStart := 69607 }
    , { arm := 0, sourceStart := 310646, length := 140, emittedStart := 71444 }
    , { arm := 0, sourceStart := 626424, length := 2169, emittedStart := 71584 }
    , { arm := 0, sourceStart := 1289391, length := 1, emittedStart := 73753 }
    , { arm := 0, sourceStart := 1289397, length := 69, emittedStart := 73754 }
    , { arm := 0, sourceStart := 1294903, length := 276, emittedStart := 73823 }
    , { arm := 0, sourceStart := 1300621, length := 276, emittedStart := 74099 }
    , { arm := 1, sourceStart := 0, length := 49626, emittedStart := 255341 }
    , { arm := 1, sourceStart := 49626, length := 41, emittedStart := 304967 }
    , { arm := 1, sourceStart := 163499, length := 2, emittedStart := 305008 }
    , { arm := 1, sourceStart := 163501, length := 108, emittedStart := 305010 }
    , { arm := 1, sourceStart := 163609, length := 1837, emittedStart := 305118 }
    , { arm := 1, sourceStart := 311846, length := 140, emittedStart := 306955 }
    , { arm := 1, sourceStart := 627624, length := 2169, emittedStart := 307095 }
    , { arm := 1, sourceStart := 1290591, length := 1, emittedStart := 309264 }
    , { arm := 1, sourceStart := 1290597, length := 69, emittedStart := 309265 }
    , { arm := 1, sourceStart := 1296103, length := 276, emittedStart := 309334 }
    , { arm := 1, sourceStart := 1301821, length := 276, emittedStart := 309610 }
    ]
  rewriteBatches :=
    [
      { rewriteStart := 0, count := 242, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 165446, sourceStride := 600, sourceWidth := 600, emittedStart := 74375, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 242, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 310801, sourceStride := 0, sourceWidth := 600, emittedStart := 95187, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 243, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 311402, sourceStride := 0, sourceWidth := 600, emittedStart := 95273, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 244, count := 260, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 312002, sourceStride := 600, sourceWidth := 600, emittedStart := 95359, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 504, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 468003, sourceStride := 0, sourceWidth := 600, emittedStart := 117719, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 505, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 468616, sourceStride := 0, sourceWidth := 600, emittedStart := 117805, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 506, count := 261, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 469217, sourceStride := 600, sourceWidth := 600, emittedStart := 117891, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 767, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 625818, sourceStride := 0, sourceWidth := 600, emittedStart := 140337, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 768, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 628613, sourceStride := 0, sourceWidth := 600, emittedStart := 140423, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 769, count := 545, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 629217, sourceStride := 604, sourceWidth := 600, emittedStart := 140509, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 1314, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 958394, sourceStride := 0, sourceWidth := 600, emittedStart := 187379, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 1315, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 959010, sourceStride := 0, sourceWidth := 600, emittedStart := 187465, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 1316, count := 545, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 959614, sourceStride := 604, sourceWidth := 600, emittedStart := 187551, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 1861, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 1288791, sourceStride := 0, sourceWidth := 600, emittedStart := 234421, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 1862, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 1289474, sourceStride := 0, sourceWidth := 600, emittedStart := 234507, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 1863, count := 7, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 1290078, sourceStride := 604, sourceWidth := 600, emittedStart := 234593, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 1870, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 1294303, sourceStride := 0, sourceWidth := 600, emittedStart := 235195, emittedStride := 0, emittedWidth := 90 }
    , { rewriteStart := 1871, count := 8, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 1295192, sourceStride := 604, sourceWidth := 600, emittedStart := 235285, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 1879, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 1300021, sourceStride := 0, sourceWidth := 600, emittedStart := 235973, emittedStride := 0, emittedWidth := 90 }
    , { rewriteStart := 1880, count := 918, rewriteStride := 1, arm := 0, kind := .shiftedTernaryCanonical, sourceStart := 49667, sourceStride := 124, sourceWidth := 124, emittedStart := 236063, emittedStride := 21, emittedWidth := 21 }
    , { rewriteStart := 7318, count := 244, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 165446, sourceStride := 600, sourceWidth := 600, emittedStart := 309886, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 7562, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 312001, sourceStride := 0, sourceWidth := 600, emittedStart := 330870, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 7563, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 312602, sourceStride := 0, sourceWidth := 600, emittedStart := 330956, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 7564, count := 260, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 313202, sourceStride := 600, sourceWidth := 600, emittedStart := 331042, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 7824, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 469203, sourceStride := 0, sourceWidth := 600, emittedStart := 353402, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 7825, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 469816, sourceStride := 0, sourceWidth := 600, emittedStart := 353488, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 7826, count := 261, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 470417, sourceStride := 600, sourceWidth := 600, emittedStart := 353574, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 8087, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 627018, sourceStride := 0, sourceWidth := 600, emittedStart := 376020, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 8088, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 629813, sourceStride := 0, sourceWidth := 600, emittedStart := 376106, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 8089, count := 545, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 630417, sourceStride := 604, sourceWidth := 600, emittedStart := 376192, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 8634, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 959594, sourceStride := 0, sourceWidth := 600, emittedStart := 423062, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 8635, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 960210, sourceStride := 0, sourceWidth := 600, emittedStart := 423148, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 8636, count := 545, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 960814, sourceStride := 604, sourceWidth := 600, emittedStart := 423234, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 9181, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 1289991, sourceStride := 0, sourceWidth := 600, emittedStart := 470104, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 9182, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 1290674, sourceStride := 0, sourceWidth := 600, emittedStart := 470190, emittedStride := 0, emittedWidth := 86 }
    , { rewriteStart := 9183, count := 7, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 1291278, sourceStride := 604, sourceWidth := 600, emittedStart := 470276, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 9190, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 1295503, sourceStride := 0, sourceWidth := 600, emittedStart := 470878, emittedStride := 0, emittedWidth := 90 }
    , { rewriteStart := 9191, count := 8, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 1296392, sourceStride := 604, sourceWidth := 600, emittedStart := 470968, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 9199, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 1301221, sourceStride := 0, sourceWidth := 600, emittedStart := 471656, emittedStride := 0, emittedWidth := 90 }
    , { rewriteStart := 9200, count := 918, rewriteStride := 1, arm := 1, kind := .shiftedTernaryCanonical, sourceStart := 49667, sourceStride := 124, sourceWidth := 124, emittedStart := 471746, emittedStride := 21, emittedWidth := 21 }
    ]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger
