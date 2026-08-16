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
  rows := 282459
  columns := 2521314
  evenSourceRows := 569886
  oddSourceRows := 571086
  rewriteCount := 3268
  fixedRuns :=
    [
      { start := 0, length := 2, family := .selectorDomain, arm := none }
    , { start := 2, length := 32826, family := .sharedDomain, arm := none }
    , { start := 32828, length := 704, family := .armDomain, arm := some 0 }
    , { start := 33532, length := 704, family := .armDomain, arm := some 1 }
    , { start := 34236, length := 1, family := .oneHot, arm := none }
    , { start := 34237, length := 7, family := .publicPadding, arm := none }
    , { start := 34244, length := 52, family := .privatePadding, arm := none }
    , { start := 282420, length := 39, family := .ringPadding, arm := none }
    ]
  retainedRuns :=
    [
      { arm := 0, sourceStart := 0, length := 43794, emittedStart := 34296 }
    , { arm := 0, sourceStart := 43794, length := 41, emittedStart := 78090 }
    , { arm := 0, sourceStart := 144275, length := 2, emittedStart := 78131 }
    , { arm := 0, sourceStart := 144277, length := 108, emittedStart := 78133 }
    , { arm := 0, sourceStart := 144385, length := 1621, emittedStart := 78241 }
    , { arm := 0, sourceStart := 275006, length := 140, emittedStart := 79862 }
    , { arm := 0, sourceStart := 558380, length := 1, emittedStart := 80002 }
    , { arm := 0, sourceStart := 558386, length := 69, emittedStart := 80003 }
    , { arm := 0, sourceStart := 563892, length := 276, emittedStart := 80072 }
    , { arm := 0, sourceStart := 569610, length := 276, emittedStart := 80348 }
    , { arm := 1, sourceStart := 0, length := 43794, emittedStart := 158272 }
    , { arm := 1, sourceStart := 43794, length := 41, emittedStart := 202066 }
    , { arm := 1, sourceStart := 144275, length := 2, emittedStart := 202107 }
    , { arm := 1, sourceStart := 144277, length := 108, emittedStart := 202109 }
    , { arm := 1, sourceStart := 144385, length := 1621, emittedStart := 202217 }
    , { arm := 1, sourceStart := 276206, length := 140, emittedStart := 203838 }
    , { arm := 1, sourceStart := 559580, length := 1, emittedStart := 203978 }
    , { arm := 1, sourceStart := 559586, length := 69, emittedStart := 203979 }
    , { arm := 1, sourceStart := 565092, length := 276, emittedStart := 204048 }
    , { arm := 1, sourceStart := 570810, length := 276, emittedStart := 204324 }
    ]
  rewriteBatches :=
    [
      { rewriteStart := 0, count := 215, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 146006, sourceStride := 600, sourceWidth := 600, emittedStart := 80624, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 215, count := 2, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 275161, sourceStride := 601, sourceWidth := 600, emittedStart := 99114, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 217, count := 233, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 276362, sourceStride := 600, sourceWidth := 600, emittedStart := 99286, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 450, count := 2, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 416163, sourceStride := 613, sourceWidth := 600, emittedStart := 119324, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 452, count := 234, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 417377, sourceStride := 600, sourceWidth := 600, emittedStart := 119496, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 686, count := 2, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 557778, sourceStride := 685, sourceWidth := 600, emittedStart := 139620, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 688, count := 7, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 559067, sourceStride := 604, sourceWidth := 600, emittedStart := 139792, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 695, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 563292, sourceStride := 0, sourceWidth := 600, emittedStart := 140394, emittedStride := 0, emittedWidth := 90 }
    , { rewriteStart := 696, count := 8, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 564181, sourceStride := 604, sourceWidth := 600, emittedStart := 140484, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 704, count := 1, rewriteStride := 1, arm := 0, kind := .poseidon2, sourceStart := 569010, sourceStride := 0, sourceWidth := 600, emittedStart := 141172, emittedStride := 0, emittedWidth := 90 }
    , { rewriteStart := 705, count := 810, rewriteStride := 1, arm := 0, kind := .shiftedTernaryCanonical, sourceStart := 43835, sourceStride := 124, sourceWidth := 124, emittedStart := 141262, emittedStride := 21, emittedWidth := 21 }
    , { rewriteStart := 1515, count := 15, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 275146, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1530, count := 2, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 275761, sourceStride := 140401, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1532, count := 13, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 416763, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1545, count := 2, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 417376, sourceStride := 140401, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1547, count := 2, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 558378, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1549, count := 5, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 558381, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1554, count := 8, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 558455, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1562, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 559063, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1566, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 559667, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1570, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 560271, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1574, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 560875, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1578, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 561479, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1582, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 562083, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1586, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 562687, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1590, count := 2, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 563291, sourceStride := 877, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1592, count := 12, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 564169, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1604, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 564781, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1608, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 565385, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1612, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 565989, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1616, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 566593, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1620, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 567197, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1624, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 567801, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1628, count := 4, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 568405, sourceStride := 1, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1632, count := 1, rewriteStride := 1, arm := 0, kind := .linearDefinition, sourceStart := 569009, sourceStride := 0, sourceWidth := 1, emittedStart := 158272, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 1633, count := 217, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 146006, sourceStride := 600, sourceWidth := 600, emittedStart := 204600, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 1850, count := 2, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 276361, sourceStride := 601, sourceWidth := 600, emittedStart := 223262, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 1852, count := 233, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 277562, sourceStride := 600, sourceWidth := 600, emittedStart := 223434, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 2085, count := 2, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 417363, sourceStride := 613, sourceWidth := 600, emittedStart := 243472, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 2087, count := 234, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 418577, sourceStride := 600, sourceWidth := 600, emittedStart := 243644, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 2321, count := 2, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 558978, sourceStride := 685, sourceWidth := 600, emittedStart := 263768, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 2323, count := 7, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 560267, sourceStride := 604, sourceWidth := 600, emittedStart := 263940, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 2330, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 564492, sourceStride := 0, sourceWidth := 600, emittedStart := 264542, emittedStride := 0, emittedWidth := 90 }
    , { rewriteStart := 2331, count := 8, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 565381, sourceStride := 604, sourceWidth := 600, emittedStart := 264632, emittedStride := 86, emittedWidth := 86 }
    , { rewriteStart := 2339, count := 1, rewriteStride := 1, arm := 1, kind := .poseidon2, sourceStart := 570210, sourceStride := 0, sourceWidth := 600, emittedStart := 265320, emittedStride := 0, emittedWidth := 90 }
    , { rewriteStart := 2340, count := 810, rewriteStride := 1, arm := 1, kind := .shiftedTernaryCanonical, sourceStart := 43835, sourceStride := 124, sourceWidth := 124, emittedStart := 265410, emittedStride := 21, emittedWidth := 21 }
    , { rewriteStart := 3150, count := 15, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 276346, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3165, count := 2, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 276961, sourceStride := 140401, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3167, count := 13, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 417963, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3180, count := 2, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 418576, sourceStride := 140401, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3182, count := 2, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 559578, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3184, count := 5, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 559581, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3189, count := 8, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 559655, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3197, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 560263, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3201, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 560867, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3205, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 561471, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3209, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 562075, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3213, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 562679, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3217, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 563283, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3221, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 563887, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3225, count := 2, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 564491, sourceStride := 877, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3227, count := 12, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 565369, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3239, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 565981, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3243, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 566585, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3247, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 567189, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3251, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 567793, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3255, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 568397, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3259, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 569001, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3263, count := 4, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 569605, sourceStride := 1, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    , { rewriteStart := 3267, count := 1, rewriteStride := 1, arm := 1, kind := .linearDefinition, sourceStart := 570209, sourceStride := 0, sourceWidth := 1, emittedStart := 282420, emittedStride := 0, emittedWidth := 0 }
    ]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger
