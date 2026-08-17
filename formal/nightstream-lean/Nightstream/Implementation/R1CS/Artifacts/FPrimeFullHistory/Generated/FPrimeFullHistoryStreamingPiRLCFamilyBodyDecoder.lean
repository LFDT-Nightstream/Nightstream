import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema

/-! Generated file: exact compact source-to-final decoder for both production
PiRLC parity bodies.

Owns: the two source ranges, final normalized column bound, three shared
decoder templates, exact affine template instances, and residual strided
rule batches emitted from the supported b = 2 production selective layout.

Does not own: source-row semantics, matrix soundness, selector authority,
assignment values, or lifecycle soundness.

Emits constraints: no. Rust checks every expanded rule against the prepared
layout before it renders this inert data. Lean validates the compact cover.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema

def templateRules00 : List RawRun :=
  [
    { sourceStart := 0, length := 41, resolution := .decompositionAlias 0 0 0 1 0 1 true }
  , { sourceStart := 41, length := 42, resolution := .traceEliminated }
  , { sourceStart := 83, length := 1, resolution := .direct 0 0 1 false }
  , { sourceStart := 84, length := 1, resolution := .traceEliminated }
  , { sourceStart := 85, length := 1, resolution := .direct 1 0 1 false }
  , { sourceStart := 86, length := 1, resolution := .traceEliminated }
  , { sourceStart := 87, length := 1, resolution := .direct 2 0 1 false }
  , { sourceStart := 88, length := 1, resolution := .traceEliminated }
  , { sourceStart := 89, length := 1, resolution := .direct 3 0 1 false }
  , { sourceStart := 90, length := 1, resolution := .traceEliminated }
  , { sourceStart := 91, length := 1, resolution := .direct 4 0 1 false }
  , { sourceStart := 92, length := 1, resolution := .traceEliminated }
  , { sourceStart := 93, length := 1, resolution := .direct 5 0 1 false }
  , { sourceStart := 94, length := 1, resolution := .traceEliminated }
  , { sourceStart := 95, length := 1, resolution := .direct 6 0 1 false }
  , { sourceStart := 96, length := 1, resolution := .traceEliminated }
  , { sourceStart := 97, length := 1, resolution := .direct 7 0 1 false }
  , { sourceStart := 98, length := 1, resolution := .traceEliminated }
  , { sourceStart := 99, length := 1, resolution := .direct 8 0 1 false }
  , { sourceStart := 100, length := 1, resolution := .traceEliminated }
  , { sourceStart := 101, length := 1, resolution := .direct 9 0 1 false }
  , { sourceStart := 102, length := 1, resolution := .traceEliminated }
  , { sourceStart := 103, length := 1, resolution := .direct 10 0 1 false }
  , { sourceStart := 104, length := 1, resolution := .traceEliminated }
  , { sourceStart := 105, length := 1, resolution := .direct 11 0 1 false }
  , { sourceStart := 106, length := 1, resolution := .traceEliminated }
  , { sourceStart := 107, length := 1, resolution := .direct 12 0 1 false }
  , { sourceStart := 108, length := 1, resolution := .traceEliminated }
  , { sourceStart := 109, length := 1, resolution := .direct 13 0 1 false }
  , { sourceStart := 110, length := 1, resolution := .traceEliminated }
  , { sourceStart := 111, length := 1, resolution := .direct 14 0 1 false }
  , { sourceStart := 112, length := 1, resolution := .traceEliminated }
  , { sourceStart := 113, length := 1, resolution := .direct 15 0 1 false }
  , { sourceStart := 114, length := 1, resolution := .traceEliminated }
  , { sourceStart := 115, length := 1, resolution := .direct 16 0 1 false }
  , { sourceStart := 116, length := 1, resolution := .traceEliminated }
  , { sourceStart := 117, length := 1, resolution := .direct 17 0 1 false }
  , { sourceStart := 118, length := 1, resolution := .traceEliminated }
  , { sourceStart := 119, length := 1, resolution := .direct 18 0 1 false }
  , { sourceStart := 120, length := 1, resolution := .traceEliminated }
  , { sourceStart := 121, length := 1, resolution := .direct 19 0 1 false }
  ]

def templateRules01 : List RawRun :=
  [
    { sourceStart := 0, length := 11, resolution := .traceEliminated }
  , { sourceStart := 11, length := 1, resolution := .direct 0 0 41 false }
  , { sourceStart := 12, length := 3, resolution := .traceEliminated }
  , { sourceStart := 15, length := 1, resolution := .direct 41 0 41 false }
  , { sourceStart := 16, length := 3, resolution := .traceEliminated }
  , { sourceStart := 19, length := 1, resolution := .direct 82 0 41 false }
  , { sourceStart := 20, length := 3, resolution := .traceEliminated }
  , { sourceStart := 23, length := 1, resolution := .direct 123 0 41 false }
  , { sourceStart := 24, length := 3, resolution := .traceEliminated }
  , { sourceStart := 27, length := 1, resolution := .direct 164 0 41 false }
  , { sourceStart := 28, length := 3, resolution := .traceEliminated }
  , { sourceStart := 31, length := 1, resolution := .direct 205 0 41 false }
  , { sourceStart := 32, length := 3, resolution := .traceEliminated }
  , { sourceStart := 35, length := 1, resolution := .direct 246 0 41 false }
  , { sourceStart := 36, length := 3, resolution := .traceEliminated }
  , { sourceStart := 39, length := 1, resolution := .direct 287 0 41 false }
  , { sourceStart := 40, length := 11, resolution := .traceEliminated }
  , { sourceStart := 51, length := 1, resolution := .direct 328 0 41 false }
  , { sourceStart := 52, length := 3, resolution := .traceEliminated }
  , { sourceStart := 55, length := 1, resolution := .direct 369 0 41 false }
  , { sourceStart := 56, length := 3, resolution := .traceEliminated }
  , { sourceStart := 59, length := 1, resolution := .direct 410 0 41 false }
  , { sourceStart := 60, length := 3, resolution := .traceEliminated }
  , { sourceStart := 63, length := 1, resolution := .direct 451 0 41 false }
  , { sourceStart := 64, length := 3, resolution := .traceEliminated }
  , { sourceStart := 67, length := 1, resolution := .direct 492 0 41 false }
  , { sourceStart := 68, length := 3, resolution := .traceEliminated }
  , { sourceStart := 71, length := 1, resolution := .direct 533 0 41 false }
  , { sourceStart := 72, length := 3, resolution := .traceEliminated }
  , { sourceStart := 75, length := 1, resolution := .direct 574 0 41 false }
  , { sourceStart := 76, length := 3, resolution := .traceEliminated }
  , { sourceStart := 79, length := 1, resolution := .direct 615 0 41 false }
  , { sourceStart := 80, length := 11, resolution := .traceEliminated }
  , { sourceStart := 91, length := 1, resolution := .direct 656 0 41 false }
  , { sourceStart := 92, length := 3, resolution := .traceEliminated }
  , { sourceStart := 95, length := 1, resolution := .direct 697 0 41 false }
  , { sourceStart := 96, length := 3, resolution := .traceEliminated }
  , { sourceStart := 99, length := 1, resolution := .direct 738 0 41 false }
  , { sourceStart := 100, length := 3, resolution := .traceEliminated }
  , { sourceStart := 103, length := 1, resolution := .direct 779 0 41 false }
  , { sourceStart := 104, length := 3, resolution := .traceEliminated }
  , { sourceStart := 107, length := 1, resolution := .direct 820 0 41 false }
  , { sourceStart := 108, length := 3, resolution := .traceEliminated }
  , { sourceStart := 111, length := 1, resolution := .direct 861 0 41 false }
  , { sourceStart := 112, length := 3, resolution := .traceEliminated }
  , { sourceStart := 115, length := 1, resolution := .direct 902 0 41 false }
  , { sourceStart := 116, length := 3, resolution := .traceEliminated }
  , { sourceStart := 119, length := 1, resolution := .direct 943 0 41 false }
  , { sourceStart := 120, length := 11, resolution := .traceEliminated }
  , { sourceStart := 131, length := 1, resolution := .direct 984 0 41 false }
  , { sourceStart := 132, length := 3, resolution := .traceEliminated }
  , { sourceStart := 135, length := 1, resolution := .direct 1025 0 41 false }
  , { sourceStart := 136, length := 3, resolution := .traceEliminated }
  , { sourceStart := 139, length := 1, resolution := .direct 1066 0 41 false }
  , { sourceStart := 140, length := 3, resolution := .traceEliminated }
  , { sourceStart := 143, length := 1, resolution := .direct 1107 0 41 false }
  , { sourceStart := 144, length := 3, resolution := .traceEliminated }
  , { sourceStart := 147, length := 1, resolution := .direct 1148 0 41 false }
  , { sourceStart := 148, length := 3, resolution := .traceEliminated }
  , { sourceStart := 151, length := 1, resolution := .direct 1189 0 41 false }
  , { sourceStart := 152, length := 3, resolution := .traceEliminated }
  , { sourceStart := 155, length := 1, resolution := .direct 1230 0 41 false }
  , { sourceStart := 156, length := 3, resolution := .traceEliminated }
  , { sourceStart := 159, length := 1, resolution := .direct 1271 0 41 false }
  , { sourceStart := 160, length := 11, resolution := .traceEliminated }
  , { sourceStart := 171, length := 1, resolution := .direct 1312 0 41 false }
  , { sourceStart := 172, length := 11, resolution := .traceEliminated }
  , { sourceStart := 183, length := 1, resolution := .direct 1353 0 41 false }
  , { sourceStart := 184, length := 11, resolution := .traceEliminated }
  , { sourceStart := 195, length := 1, resolution := .direct 1394 0 41 false }
  , { sourceStart := 196, length := 11, resolution := .traceEliminated }
  , { sourceStart := 207, length := 1, resolution := .direct 1435 0 41 false }
  , { sourceStart := 208, length := 11, resolution := .traceEliminated }
  , { sourceStart := 219, length := 1, resolution := .direct 1476 0 41 false }
  , { sourceStart := 220, length := 11, resolution := .traceEliminated }
  , { sourceStart := 231, length := 1, resolution := .direct 1517 0 41 false }
  , { sourceStart := 232, length := 11, resolution := .traceEliminated }
  , { sourceStart := 243, length := 1, resolution := .direct 1558 0 41 false }
  , { sourceStart := 244, length := 11, resolution := .traceEliminated }
  , { sourceStart := 255, length := 1, resolution := .direct 1599 0 41 false }
  , { sourceStart := 256, length := 11, resolution := .traceEliminated }
  , { sourceStart := 267, length := 1, resolution := .direct 1640 0 41 false }
  , { sourceStart := 268, length := 11, resolution := .traceEliminated }
  , { sourceStart := 279, length := 1, resolution := .direct 1681 0 41 false }
  , { sourceStart := 280, length := 11, resolution := .traceEliminated }
  , { sourceStart := 291, length := 1, resolution := .direct 1722 0 41 false }
  , { sourceStart := 292, length := 11, resolution := .traceEliminated }
  , { sourceStart := 303, length := 1, resolution := .direct 1763 0 41 false }
  , { sourceStart := 304, length := 11, resolution := .traceEliminated }
  , { sourceStart := 315, length := 1, resolution := .direct 1804 0 41 false }
  , { sourceStart := 316, length := 11, resolution := .traceEliminated }
  , { sourceStart := 327, length := 1, resolution := .direct 1845 0 41 false }
  , { sourceStart := 328, length := 11, resolution := .traceEliminated }
  , { sourceStart := 339, length := 1, resolution := .direct 1886 0 41 false }
  , { sourceStart := 340, length := 11, resolution := .traceEliminated }
  , { sourceStart := 351, length := 1, resolution := .direct 1927 0 41 false }
  , { sourceStart := 352, length := 11, resolution := .traceEliminated }
  , { sourceStart := 363, length := 1, resolution := .direct 1968 0 41 false }
  , { sourceStart := 364, length := 11, resolution := .traceEliminated }
  , { sourceStart := 375, length := 1, resolution := .direct 2009 0 41 false }
  , { sourceStart := 376, length := 11, resolution := .traceEliminated }
  , { sourceStart := 387, length := 1, resolution := .direct 2050 0 41 false }
  , { sourceStart := 388, length := 11, resolution := .traceEliminated }
  , { sourceStart := 399, length := 1, resolution := .direct 2091 0 41 false }
  , { sourceStart := 400, length := 11, resolution := .traceEliminated }
  , { sourceStart := 411, length := 1, resolution := .direct 2132 0 41 false }
  , { sourceStart := 412, length := 11, resolution := .traceEliminated }
  , { sourceStart := 423, length := 1, resolution := .direct 2173 0 41 false }
  , { sourceStart := 424, length := 11, resolution := .traceEliminated }
  , { sourceStart := 435, length := 1, resolution := .direct 2214 0 41 false }
  , { sourceStart := 436, length := 3, resolution := .traceEliminated }
  , { sourceStart := 439, length := 1, resolution := .direct 2255 0 41 false }
  , { sourceStart := 440, length := 3, resolution := .traceEliminated }
  , { sourceStart := 443, length := 1, resolution := .direct 2296 0 41 false }
  , { sourceStart := 444, length := 3, resolution := .traceEliminated }
  , { sourceStart := 447, length := 1, resolution := .direct 2337 0 41 false }
  , { sourceStart := 448, length := 3, resolution := .traceEliminated }
  , { sourceStart := 451, length := 1, resolution := .direct 2378 0 41 false }
  , { sourceStart := 452, length := 3, resolution := .traceEliminated }
  , { sourceStart := 455, length := 1, resolution := .direct 2419 0 41 false }
  , { sourceStart := 456, length := 3, resolution := .traceEliminated }
  , { sourceStart := 459, length := 1, resolution := .direct 2460 0 41 false }
  , { sourceStart := 460, length := 3, resolution := .traceEliminated }
  , { sourceStart := 463, length := 1, resolution := .direct 2501 0 41 false }
  , { sourceStart := 464, length := 11, resolution := .traceEliminated }
  , { sourceStart := 475, length := 1, resolution := .direct 2542 0 41 false }
  , { sourceStart := 476, length := 3, resolution := .traceEliminated }
  , { sourceStart := 479, length := 1, resolution := .direct 2583 0 41 false }
  , { sourceStart := 480, length := 3, resolution := .traceEliminated }
  , { sourceStart := 483, length := 1, resolution := .direct 2624 0 41 false }
  , { sourceStart := 484, length := 3, resolution := .traceEliminated }
  , { sourceStart := 487, length := 1, resolution := .direct 2665 0 41 false }
  , { sourceStart := 488, length := 3, resolution := .traceEliminated }
  , { sourceStart := 491, length := 1, resolution := .direct 2706 0 41 false }
  , { sourceStart := 492, length := 3, resolution := .traceEliminated }
  , { sourceStart := 495, length := 1, resolution := .direct 2747 0 41 false }
  , { sourceStart := 496, length := 3, resolution := .traceEliminated }
  , { sourceStart := 499, length := 1, resolution := .direct 2788 0 41 false }
  , { sourceStart := 500, length := 3, resolution := .traceEliminated }
  , { sourceStart := 503, length := 1, resolution := .direct 2829 0 41 false }
  , { sourceStart := 504, length := 11, resolution := .traceEliminated }
  , { sourceStart := 515, length := 1, resolution := .direct 2870 0 41 false }
  , { sourceStart := 516, length := 3, resolution := .traceEliminated }
  , { sourceStart := 519, length := 1, resolution := .direct 2911 0 41 false }
  , { sourceStart := 520, length := 3, resolution := .traceEliminated }
  , { sourceStart := 523, length := 1, resolution := .direct 2952 0 41 false }
  , { sourceStart := 524, length := 3, resolution := .traceEliminated }
  , { sourceStart := 527, length := 1, resolution := .direct 2993 0 41 false }
  , { sourceStart := 528, length := 3, resolution := .traceEliminated }
  , { sourceStart := 531, length := 1, resolution := .direct 3034 0 41 false }
  , { sourceStart := 532, length := 3, resolution := .traceEliminated }
  , { sourceStart := 535, length := 1, resolution := .direct 3075 0 41 false }
  , { sourceStart := 536, length := 3, resolution := .traceEliminated }
  , { sourceStart := 539, length := 1, resolution := .direct 3116 0 41 false }
  , { sourceStart := 540, length := 3, resolution := .traceEliminated }
  , { sourceStart := 543, length := 1, resolution := .direct 3157 0 41 false }
  , { sourceStart := 544, length := 11, resolution := .traceEliminated }
  , { sourceStart := 555, length := 1, resolution := .direct 3198 0 41 false }
  , { sourceStart := 556, length := 3, resolution := .traceEliminated }
  , { sourceStart := 559, length := 1, resolution := .direct 3239 0 41 false }
  , { sourceStart := 560, length := 3, resolution := .traceEliminated }
  , { sourceStart := 563, length := 1, resolution := .direct 3280 0 41 false }
  , { sourceStart := 564, length := 3, resolution := .traceEliminated }
  , { sourceStart := 567, length := 1, resolution := .direct 3321 0 41 false }
  , { sourceStart := 568, length := 3, resolution := .traceEliminated }
  , { sourceStart := 571, length := 1, resolution := .direct 3362 0 41 false }
  , { sourceStart := 572, length := 3, resolution := .traceEliminated }
  , { sourceStart := 575, length := 1, resolution := .direct 3403 0 41 false }
  , { sourceStart := 576, length := 3, resolution := .traceEliminated }
  , { sourceStart := 579, length := 1, resolution := .direct 3444 0 41 false }
  , { sourceStart := 580, length := 3, resolution := .traceEliminated }
  , { sourceStart := 583, length := 1, resolution := .direct 3485 0 41 false }
  , { sourceStart := 584, length := 8, resolution := .traceEliminated }
  , { sourceStart := 592, length := 4, resolution := .direct 3526 64 64 false }
  , { sourceStart := 596, length := 4, resolution := .linearDefinition }
  ]

def templateRules02 : List RawRun :=
  [
    { sourceStart := 0, length := 11, resolution := .traceEliminated }
  , { sourceStart := 11, length := 1, resolution := .direct 0 0 41 false }
  , { sourceStart := 12, length := 3, resolution := .traceEliminated }
  , { sourceStart := 15, length := 1, resolution := .direct 41 0 41 false }
  , { sourceStart := 16, length := 3, resolution := .traceEliminated }
  , { sourceStart := 19, length := 1, resolution := .direct 82 0 41 false }
  , { sourceStart := 20, length := 3, resolution := .traceEliminated }
  , { sourceStart := 23, length := 1, resolution := .direct 123 0 41 false }
  , { sourceStart := 24, length := 3, resolution := .traceEliminated }
  , { sourceStart := 27, length := 1, resolution := .direct 164 0 41 false }
  , { sourceStart := 28, length := 3, resolution := .traceEliminated }
  , { sourceStart := 31, length := 1, resolution := .direct 205 0 41 false }
  , { sourceStart := 32, length := 3, resolution := .traceEliminated }
  , { sourceStart := 35, length := 1, resolution := .direct 246 0 41 false }
  , { sourceStart := 36, length := 3, resolution := .traceEliminated }
  , { sourceStart := 39, length := 1, resolution := .direct 287 0 41 false }
  , { sourceStart := 40, length := 11, resolution := .traceEliminated }
  , { sourceStart := 51, length := 1, resolution := .direct 328 0 41 false }
  , { sourceStart := 52, length := 3, resolution := .traceEliminated }
  , { sourceStart := 55, length := 1, resolution := .direct 369 0 41 false }
  , { sourceStart := 56, length := 3, resolution := .traceEliminated }
  , { sourceStart := 59, length := 1, resolution := .direct 410 0 41 false }
  , { sourceStart := 60, length := 3, resolution := .traceEliminated }
  , { sourceStart := 63, length := 1, resolution := .direct 451 0 41 false }
  , { sourceStart := 64, length := 3, resolution := .traceEliminated }
  , { sourceStart := 67, length := 1, resolution := .direct 492 0 41 false }
  , { sourceStart := 68, length := 3, resolution := .traceEliminated }
  , { sourceStart := 71, length := 1, resolution := .direct 533 0 41 false }
  , { sourceStart := 72, length := 3, resolution := .traceEliminated }
  , { sourceStart := 75, length := 1, resolution := .direct 574 0 41 false }
  , { sourceStart := 76, length := 3, resolution := .traceEliminated }
  , { sourceStart := 79, length := 1, resolution := .direct 615 0 41 false }
  , { sourceStart := 80, length := 11, resolution := .traceEliminated }
  , { sourceStart := 91, length := 1, resolution := .direct 656 0 41 false }
  , { sourceStart := 92, length := 3, resolution := .traceEliminated }
  , { sourceStart := 95, length := 1, resolution := .direct 697 0 41 false }
  , { sourceStart := 96, length := 3, resolution := .traceEliminated }
  , { sourceStart := 99, length := 1, resolution := .direct 738 0 41 false }
  , { sourceStart := 100, length := 3, resolution := .traceEliminated }
  , { sourceStart := 103, length := 1, resolution := .direct 779 0 41 false }
  , { sourceStart := 104, length := 3, resolution := .traceEliminated }
  , { sourceStart := 107, length := 1, resolution := .direct 820 0 41 false }
  , { sourceStart := 108, length := 3, resolution := .traceEliminated }
  , { sourceStart := 111, length := 1, resolution := .direct 861 0 41 false }
  , { sourceStart := 112, length := 3, resolution := .traceEliminated }
  , { sourceStart := 115, length := 1, resolution := .direct 902 0 41 false }
  , { sourceStart := 116, length := 3, resolution := .traceEliminated }
  , { sourceStart := 119, length := 1, resolution := .direct 943 0 41 false }
  , { sourceStart := 120, length := 11, resolution := .traceEliminated }
  , { sourceStart := 131, length := 1, resolution := .direct 984 0 41 false }
  , { sourceStart := 132, length := 3, resolution := .traceEliminated }
  , { sourceStart := 135, length := 1, resolution := .direct 1025 0 41 false }
  , { sourceStart := 136, length := 3, resolution := .traceEliminated }
  , { sourceStart := 139, length := 1, resolution := .direct 1066 0 41 false }
  , { sourceStart := 140, length := 3, resolution := .traceEliminated }
  , { sourceStart := 143, length := 1, resolution := .direct 1107 0 41 false }
  , { sourceStart := 144, length := 3, resolution := .traceEliminated }
  , { sourceStart := 147, length := 1, resolution := .direct 1148 0 41 false }
  , { sourceStart := 148, length := 3, resolution := .traceEliminated }
  , { sourceStart := 151, length := 1, resolution := .direct 1189 0 41 false }
  , { sourceStart := 152, length := 3, resolution := .traceEliminated }
  , { sourceStart := 155, length := 1, resolution := .direct 1230 0 41 false }
  , { sourceStart := 156, length := 3, resolution := .traceEliminated }
  , { sourceStart := 159, length := 1, resolution := .direct 1271 0 41 false }
  , { sourceStart := 160, length := 11, resolution := .traceEliminated }
  , { sourceStart := 171, length := 1, resolution := .direct 1312 0 41 false }
  , { sourceStart := 172, length := 11, resolution := .traceEliminated }
  , { sourceStart := 183, length := 1, resolution := .direct 1353 0 41 false }
  , { sourceStart := 184, length := 11, resolution := .traceEliminated }
  , { sourceStart := 195, length := 1, resolution := .direct 1394 0 41 false }
  , { sourceStart := 196, length := 11, resolution := .traceEliminated }
  , { sourceStart := 207, length := 1, resolution := .direct 1435 0 41 false }
  , { sourceStart := 208, length := 11, resolution := .traceEliminated }
  , { sourceStart := 219, length := 1, resolution := .direct 1476 0 41 false }
  , { sourceStart := 220, length := 11, resolution := .traceEliminated }
  , { sourceStart := 231, length := 1, resolution := .direct 1517 0 41 false }
  , { sourceStart := 232, length := 11, resolution := .traceEliminated }
  , { sourceStart := 243, length := 1, resolution := .direct 1558 0 41 false }
  , { sourceStart := 244, length := 11, resolution := .traceEliminated }
  , { sourceStart := 255, length := 1, resolution := .direct 1599 0 41 false }
  , { sourceStart := 256, length := 11, resolution := .traceEliminated }
  , { sourceStart := 267, length := 1, resolution := .direct 1640 0 41 false }
  , { sourceStart := 268, length := 11, resolution := .traceEliminated }
  , { sourceStart := 279, length := 1, resolution := .direct 1681 0 41 false }
  , { sourceStart := 280, length := 11, resolution := .traceEliminated }
  , { sourceStart := 291, length := 1, resolution := .direct 1722 0 41 false }
  , { sourceStart := 292, length := 11, resolution := .traceEliminated }
  , { sourceStart := 303, length := 1, resolution := .direct 1763 0 41 false }
  , { sourceStart := 304, length := 11, resolution := .traceEliminated }
  , { sourceStart := 315, length := 1, resolution := .direct 1804 0 41 false }
  , { sourceStart := 316, length := 11, resolution := .traceEliminated }
  , { sourceStart := 327, length := 1, resolution := .direct 1845 0 41 false }
  , { sourceStart := 328, length := 11, resolution := .traceEliminated }
  , { sourceStart := 339, length := 1, resolution := .direct 1886 0 41 false }
  , { sourceStart := 340, length := 11, resolution := .traceEliminated }
  , { sourceStart := 351, length := 1, resolution := .direct 1927 0 41 false }
  , { sourceStart := 352, length := 11, resolution := .traceEliminated }
  , { sourceStart := 363, length := 1, resolution := .direct 1968 0 41 false }
  , { sourceStart := 364, length := 11, resolution := .traceEliminated }
  , { sourceStart := 375, length := 1, resolution := .direct 2009 0 41 false }
  , { sourceStart := 376, length := 11, resolution := .traceEliminated }
  , { sourceStart := 387, length := 1, resolution := .direct 2050 0 41 false }
  , { sourceStart := 388, length := 11, resolution := .traceEliminated }
  , { sourceStart := 399, length := 1, resolution := .direct 2091 0 41 false }
  , { sourceStart := 400, length := 11, resolution := .traceEliminated }
  , { sourceStart := 411, length := 1, resolution := .direct 2132 0 41 false }
  , { sourceStart := 412, length := 11, resolution := .traceEliminated }
  , { sourceStart := 423, length := 1, resolution := .direct 2173 0 41 false }
  , { sourceStart := 424, length := 11, resolution := .traceEliminated }
  , { sourceStart := 435, length := 1, resolution := .direct 2214 0 41 false }
  , { sourceStart := 436, length := 3, resolution := .traceEliminated }
  , { sourceStart := 439, length := 1, resolution := .direct 2255 0 41 false }
  , { sourceStart := 440, length := 3, resolution := .traceEliminated }
  , { sourceStart := 443, length := 1, resolution := .direct 2296 0 41 false }
  , { sourceStart := 444, length := 3, resolution := .traceEliminated }
  , { sourceStart := 447, length := 1, resolution := .direct 2337 0 41 false }
  , { sourceStart := 448, length := 3, resolution := .traceEliminated }
  , { sourceStart := 451, length := 1, resolution := .direct 2378 0 41 false }
  , { sourceStart := 452, length := 3, resolution := .traceEliminated }
  , { sourceStart := 455, length := 1, resolution := .direct 2419 0 41 false }
  , { sourceStart := 456, length := 3, resolution := .traceEliminated }
  , { sourceStart := 459, length := 1, resolution := .direct 2460 0 41 false }
  , { sourceStart := 460, length := 3, resolution := .traceEliminated }
  , { sourceStart := 463, length := 1, resolution := .direct 2501 0 41 false }
  , { sourceStart := 464, length := 11, resolution := .traceEliminated }
  , { sourceStart := 475, length := 1, resolution := .direct 2542 0 41 false }
  , { sourceStart := 476, length := 3, resolution := .traceEliminated }
  , { sourceStart := 479, length := 1, resolution := .direct 2583 0 41 false }
  , { sourceStart := 480, length := 3, resolution := .traceEliminated }
  , { sourceStart := 483, length := 1, resolution := .direct 2624 0 41 false }
  , { sourceStart := 484, length := 3, resolution := .traceEliminated }
  , { sourceStart := 487, length := 1, resolution := .direct 2665 0 41 false }
  , { sourceStart := 488, length := 3, resolution := .traceEliminated }
  , { sourceStart := 491, length := 1, resolution := .direct 2706 0 41 false }
  , { sourceStart := 492, length := 3, resolution := .traceEliminated }
  , { sourceStart := 495, length := 1, resolution := .direct 2747 0 41 false }
  , { sourceStart := 496, length := 3, resolution := .traceEliminated }
  , { sourceStart := 499, length := 1, resolution := .direct 2788 0 41 false }
  , { sourceStart := 500, length := 3, resolution := .traceEliminated }
  , { sourceStart := 503, length := 1, resolution := .direct 2829 0 41 false }
  , { sourceStart := 504, length := 11, resolution := .traceEliminated }
  , { sourceStart := 515, length := 1, resolution := .direct 2870 0 41 false }
  , { sourceStart := 516, length := 3, resolution := .traceEliminated }
  , { sourceStart := 519, length := 1, resolution := .direct 2911 0 41 false }
  , { sourceStart := 520, length := 3, resolution := .traceEliminated }
  , { sourceStart := 523, length := 1, resolution := .direct 2952 0 41 false }
  , { sourceStart := 524, length := 3, resolution := .traceEliminated }
  , { sourceStart := 527, length := 1, resolution := .direct 2993 0 41 false }
  , { sourceStart := 528, length := 3, resolution := .traceEliminated }
  , { sourceStart := 531, length := 1, resolution := .direct 3034 0 41 false }
  , { sourceStart := 532, length := 3, resolution := .traceEliminated }
  , { sourceStart := 535, length := 1, resolution := .direct 3075 0 41 false }
  , { sourceStart := 536, length := 3, resolution := .traceEliminated }
  , { sourceStart := 539, length := 1, resolution := .direct 3116 0 41 false }
  , { sourceStart := 540, length := 3, resolution := .traceEliminated }
  , { sourceStart := 543, length := 1, resolution := .direct 3157 0 41 false }
  , { sourceStart := 544, length := 11, resolution := .traceEliminated }
  , { sourceStart := 555, length := 1, resolution := .direct 3198 0 41 false }
  , { sourceStart := 556, length := 3, resolution := .traceEliminated }
  , { sourceStart := 559, length := 1, resolution := .direct 3239 0 41 false }
  , { sourceStart := 560, length := 3, resolution := .traceEliminated }
  , { sourceStart := 563, length := 1, resolution := .direct 3280 0 41 false }
  , { sourceStart := 564, length := 3, resolution := .traceEliminated }
  , { sourceStart := 567, length := 1, resolution := .direct 3321 0 41 false }
  , { sourceStart := 568, length := 3, resolution := .traceEliminated }
  , { sourceStart := 571, length := 1, resolution := .direct 3362 0 41 false }
  , { sourceStart := 572, length := 3, resolution := .traceEliminated }
  , { sourceStart := 575, length := 1, resolution := .direct 3403 0 41 false }
  , { sourceStart := 576, length := 3, resolution := .traceEliminated }
  , { sourceStart := 579, length := 1, resolution := .direct 3444 0 41 false }
  , { sourceStart := 580, length := 3, resolution := .traceEliminated }
  , { sourceStart := 583, length := 1, resolution := .direct 3485 0 41 false }
  , { sourceStart := 584, length := 8, resolution := .traceEliminated }
  , { sourceStart := 592, length := 8, resolution := .linearDefinition }
  ]

def evenTemplateInstances00 : List RawTemplateInstances :=
  [
    { sourceStart := 52144, count := 918, sourceStride := 122, finalStart := 2110685, finalStride := 20, referenceStart := 1559, referenceStride := 1, referenceFinalStart := 38340, referenceFinalStride := 41 }
  ]

def evenTemplateInstances01 : List RawTemplateInstances :=
  [
    { sourceStart := 1295068, count := 1, sourceStride := 0, finalStart := 8815680, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1300518, count := 1, sourceStride := 0, finalStart := 8847838, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  ]

def evenTemplateInstances02 : List RawTemplateInstances :=
  [
    { sourceStart := 166320, count := 242, sourceStride := 600, finalStart := 2218425, finalStride := 3526, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 311541, count := 1, sourceStride := 0, finalStart := 3071929, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 312142, count := 1, sourceStride := 0, finalStart := 3075455, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 312742, count := 260, sourceStride := 600, finalStart := 3078981, finalStride := 3526, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 468743, count := 1, sourceStride := 0, finalStart := 3995741, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 469356, count := 1, sourceStride := 0, finalStart := 3999267, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 469957, count := 261, sourceStride := 600, finalStart := 4002793, finalStride := 3526, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 626558, count := 1, sourceStride := 0, finalStart := 4923079, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 629353, count := 1, sourceStride := 0, finalStart := 4928774, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 629957, count := 545, sourceStride := 604, finalStart := 4932300, finalStride := 3526, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 959134, count := 1, sourceStride := 0, finalStart := 6853970, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 959750, count := 1, sourceStride := 0, finalStart := 6857496, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 960354, count := 545, sourceStride := 604, finalStart := 6861022, finalStride := 3526, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1289531, count := 1, sourceStride := 0, finalStart := 8782692, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1290239, count := 1, sourceStride := 0, finalStart := 8787472, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1290843, count := 1, sourceStride := 0, finalStart := 8790998, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1291447, count := 1, sourceStride := 0, finalStart := 8794524, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1292051, count := 1, sourceStride := 0, finalStart := 8798050, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1292655, count := 1, sourceStride := 0, finalStart := 8801576, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1293259, count := 1, sourceStride := 0, finalStart := 8805102, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1293863, count := 1, sourceStride := 0, finalStart := 8808628, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1294467, count := 1, sourceStride := 0, finalStart := 8812154, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1295689, count := 1, sourceStride := 0, finalStart := 8819630, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1296293, count := 1, sourceStride := 0, finalStart := 8823156, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1296897, count := 1, sourceStride := 0, finalStart := 8826682, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1297501, count := 1, sourceStride := 0, finalStart := 8830208, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1298105, count := 1, sourceStride := 0, finalStart := 8833734, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1298709, count := 1, sourceStride := 0, finalStart := 8837260, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1299313, count := 1, sourceStride := 0, finalStart := 8840786, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1299917, count := 1, sourceStride := 0, finalStart := 8844312, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  ]

def evenTemplates : List RawTemplate :=
  [
    { sourceWidth := 122, relativeRuns := templateRules00, instances := evenTemplateInstances00 }
  , { sourceWidth := 600, relativeRuns := templateRules01, instances := evenTemplateInstances01 }
  , { sourceWidth := 600, relativeRuns := templateRules02, instances := evenTemplateInstances02 }
  ]

def evenResidualBatches : List RawResidualBatch :=
  [
    { sourceStart := 1, instanceCount := 1, instanceStride := 0, width := 640, resolution := .direct 1 1 1 false }
  , { sourceStart := 641, instanceCount := 1, instanceStride := 0, width := 51462, resolution := .direct 702 41 41 false }
  , { sourceStart := 52103, instanceCount := 1, instanceStride := 0, width := 41, resolution := .direct 2110644 1 1 true }
  , { sourceStart := 164140, instanceCount := 1, instanceStride := 0, width := 2180, resolution := .direct 2129045 41 41 false }
  , { sourceStart := 311520, instanceCount := 1, instanceStride := 0, width := 2, resolution := .direct 3071717 64 64 false }
  , { sourceStart := 311522, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 3071845 42 1 false }
  , { sourceStart := 311524, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 3071887 42 1 false }
  , { sourceStart := 311523, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 3071846 42 41 false }
  , { sourceStart := 311525, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 3071888 42 41 false }
  , { sourceStart := 311526, instanceCount := 1, instanceStride := 0, width := 15, resolution := .linearDefinition }
  , { sourceStart := 312141, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 468742, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 469343, instanceCount := 1, instanceStride := 0, width := 13, resolution := .linearDefinition }
  , { sourceStart := 469956, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 626557, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 627158, instanceCount := 1, instanceStride := 0, width := 6, resolution := .linearDefinition }
  , { sourceStart := 627164, instanceCount := 1, instanceStride := 0, width := 2169, resolution := .direct 4926605 1 1 false }
  , { sourceStart := 629333, instanceCount := 1, instanceStride := 0, width := 20, resolution := .linearDefinition }
  , { sourceStart := 629953, instanceCount := 545, instanceStride := 604, width := 4, resolution := .linearDefinition }
  , { sourceStart := 959133, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 959734, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 959735, instanceCount := 1, instanceStride := 0, width := 15, resolution := .linearDefinition }
  , { sourceStart := 960350, instanceCount := 545, instanceStride := 604, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1289530, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 1290160, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 1290131, instanceCount := 1, instanceStride := 0, width := 8, resolution := .direct 8786218 41 41 false }
  , { sourceStart := 1290139, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8786546 0 64 false }
  , { sourceStart := 1290140, instanceCount := 1, instanceStride := 0, width := 20, resolution := .direct 8786610 41 41 false }
  , { sourceStart := 1290161, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1290165, instanceCount := 1, instanceStride := 0, width := 64, resolution := .decompositionAlias 1290139 0 0 1 8786546 1 false }
  , { sourceStart := 1290229, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8787430 32032 1 false }
  , { sourceStart := 1295668, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8819462 32032 1 false }
  , { sourceStart := 1290230, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8787431 32032 41 false }
  , { sourceStart := 1295669, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8819463 32032 41 false }
  , { sourceStart := 1290231, instanceCount := 1, instanceStride := 0, width := 8, resolution := .linearDefinition }
  , { sourceStart := 1290839, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1291443, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1292047, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1292651, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1293255, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1293859, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1294463, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1295067, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 1295676, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 1295670, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8819504 42 1 false }
  , { sourceStart := 1295672, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8819546 42 1 false }
  , { sourceStart := 1295674, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8819588 42 1 false }
  , { sourceStart := 1295671, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8819505 42 41 false }
  , { sourceStart := 1295673, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8819547 42 41 false }
  , { sourceStart := 1295675, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8819589 42 41 false }
  , { sourceStart := 1295677, instanceCount := 1, instanceStride := 0, width := 12, resolution := .linearDefinition }
  , { sourceStart := 1296289, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1296893, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1297497, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1298101, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1298705, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1299309, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1299913, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1300517, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 1301118, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8851620 42 1 false }
  , { sourceStart := 1301120, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8851662 42 1 false }
  , { sourceStart := 1301122, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8851704 42 1 false }
  , { sourceStart := 1301124, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8851746 42 1 false }
  , { sourceStart := 1301119, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8851621 42 41 false }
  , { sourceStart := 1301121, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8851663 42 41 false }
  , { sourceStart := 1301123, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8851705 42 41 false }
  , { sourceStart := 1301125, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8851747 42 41 false }
  ]

def evenCoverGroups : List RawCoverGroup :=
  [
    { sourceStart := 1, count := 1, stride := 640, owners := [.residual 0] }
  , { sourceStart := 641, count := 1, stride := 51462, owners := [.residual 1] }
  , { sourceStart := 52103, count := 1, stride := 41, owners := [.residual 2] }
  , { sourceStart := 52144, count := 918, stride := 122, owners := [.template 0 0] }
  , { sourceStart := 164140, count := 1, stride := 2180, owners := [.residual 3] }
  , { sourceStart := 166320, count := 242, stride := 600, owners := [.template 2 0] }
  , { sourceStart := 311520, count := 1, stride := 2, owners := [.residual 4] }
  , { sourceStart := 311522, count := 1, stride := 1, owners := [.residual 5] }
  , { sourceStart := 311523, count := 1, stride := 1, owners := [.residual 7] }
  , { sourceStart := 311524, count := 1, stride := 1, owners := [.residual 6] }
  , { sourceStart := 311525, count := 1, stride := 1, owners := [.residual 8] }
  , { sourceStart := 311526, count := 1, stride := 15, owners := [.residual 9] }
  , { sourceStart := 311541, count := 1, stride := 600, owners := [.template 2 1] }
  , { sourceStart := 312141, count := 1, stride := 1, owners := [.residual 10] }
  , { sourceStart := 312142, count := 1, stride := 600, owners := [.template 2 2] }
  , { sourceStart := 312742, count := 260, stride := 600, owners := [.template 2 3] }
  , { sourceStart := 468742, count := 1, stride := 1, owners := [.residual 11] }
  , { sourceStart := 468743, count := 1, stride := 600, owners := [.template 2 4] }
  , { sourceStart := 469343, count := 1, stride := 13, owners := [.residual 12] }
  , { sourceStart := 469356, count := 1, stride := 600, owners := [.template 2 5] }
  , { sourceStart := 469956, count := 1, stride := 1, owners := [.residual 13] }
  , { sourceStart := 469957, count := 261, stride := 600, owners := [.template 2 6] }
  , { sourceStart := 626557, count := 1, stride := 1, owners := [.residual 14] }
  , { sourceStart := 626558, count := 1, stride := 600, owners := [.template 2 7] }
  , { sourceStart := 627158, count := 1, stride := 6, owners := [.residual 15] }
  , { sourceStart := 627164, count := 1, stride := 2169, owners := [.residual 16] }
  , { sourceStart := 629333, count := 1, stride := 20, owners := [.residual 17] }
  , { sourceStart := 629353, count := 1, stride := 600, owners := [.template 2 8] }
  , { sourceStart := 629953, count := 545, stride := 604, owners := [.residual 18, .template 2 9] }
  , { sourceStart := 959133, count := 1, stride := 1, owners := [.residual 19] }
  , { sourceStart := 959134, count := 1, stride := 600, owners := [.template 2 10] }
  , { sourceStart := 959734, count := 1, stride := 1, owners := [.residual 20] }
  , { sourceStart := 959735, count := 1, stride := 15, owners := [.residual 21] }
  , { sourceStart := 959750, count := 1, stride := 600, owners := [.template 2 11] }
  , { sourceStart := 960350, count := 545, stride := 604, owners := [.residual 22, .template 2 12] }
  , { sourceStart := 1289530, count := 1, stride := 1, owners := [.residual 23] }
  , { sourceStart := 1289531, count := 1, stride := 600, owners := [.template 2 13] }
  , { sourceStart := 1290131, count := 1, stride := 8, owners := [.residual 25] }
  , { sourceStart := 1290139, count := 1, stride := 1, owners := [.residual 26] }
  , { sourceStart := 1290140, count := 1, stride := 20, owners := [.residual 27] }
  , { sourceStart := 1290160, count := 1, stride := 1, owners := [.residual 24] }
  , { sourceStart := 1290161, count := 1, stride := 4, owners := [.residual 28] }
  , { sourceStart := 1290165, count := 1, stride := 64, owners := [.residual 29] }
  , { sourceStart := 1290229, count := 1, stride := 1, owners := [.residual 30] }
  , { sourceStart := 1290230, count := 1, stride := 1, owners := [.residual 32] }
  , { sourceStart := 1290231, count := 1, stride := 8, owners := [.residual 34] }
  , { sourceStart := 1290239, count := 1, stride := 600, owners := [.template 2 14] }
  , { sourceStart := 1290839, count := 1, stride := 4, owners := [.residual 35] }
  , { sourceStart := 1290843, count := 1, stride := 600, owners := [.template 2 15] }
  , { sourceStart := 1291443, count := 1, stride := 4, owners := [.residual 36] }
  , { sourceStart := 1291447, count := 1, stride := 600, owners := [.template 2 16] }
  , { sourceStart := 1292047, count := 1, stride := 4, owners := [.residual 37] }
  , { sourceStart := 1292051, count := 1, stride := 600, owners := [.template 2 17] }
  , { sourceStart := 1292651, count := 1, stride := 4, owners := [.residual 38] }
  , { sourceStart := 1292655, count := 1, stride := 600, owners := [.template 2 18] }
  , { sourceStart := 1293255, count := 1, stride := 4, owners := [.residual 39] }
  , { sourceStart := 1293259, count := 1, stride := 600, owners := [.template 2 19] }
  , { sourceStart := 1293859, count := 1, stride := 4, owners := [.residual 40] }
  , { sourceStart := 1293863, count := 1, stride := 600, owners := [.template 2 20] }
  , { sourceStart := 1294463, count := 1, stride := 4, owners := [.residual 41] }
  , { sourceStart := 1294467, count := 1, stride := 600, owners := [.template 2 21] }
  , { sourceStart := 1295067, count := 1, stride := 1, owners := [.residual 42] }
  , { sourceStart := 1295068, count := 1, stride := 600, owners := [.template 1 0] }
  , { sourceStart := 1295668, count := 1, stride := 1, owners := [.residual 31] }
  , { sourceStart := 1295669, count := 1, stride := 1, owners := [.residual 33] }
  , { sourceStart := 1295670, count := 1, stride := 1, owners := [.residual 44] }
  , { sourceStart := 1295671, count := 1, stride := 1, owners := [.residual 47] }
  , { sourceStart := 1295672, count := 1, stride := 1, owners := [.residual 45] }
  , { sourceStart := 1295673, count := 1, stride := 1, owners := [.residual 48] }
  , { sourceStart := 1295674, count := 1, stride := 1, owners := [.residual 46] }
  , { sourceStart := 1295675, count := 1, stride := 1, owners := [.residual 49] }
  , { sourceStart := 1295676, count := 1, stride := 1, owners := [.residual 43] }
  , { sourceStart := 1295677, count := 1, stride := 12, owners := [.residual 50] }
  , { sourceStart := 1295689, count := 1, stride := 600, owners := [.template 2 22] }
  , { sourceStart := 1296289, count := 1, stride := 4, owners := [.residual 51] }
  , { sourceStart := 1296293, count := 1, stride := 600, owners := [.template 2 23] }
  , { sourceStart := 1296893, count := 1, stride := 4, owners := [.residual 52] }
  , { sourceStart := 1296897, count := 1, stride := 600, owners := [.template 2 24] }
  , { sourceStart := 1297497, count := 1, stride := 4, owners := [.residual 53] }
  , { sourceStart := 1297501, count := 1, stride := 600, owners := [.template 2 25] }
  , { sourceStart := 1298101, count := 1, stride := 4, owners := [.residual 54] }
  , { sourceStart := 1298105, count := 1, stride := 600, owners := [.template 2 26] }
  , { sourceStart := 1298705, count := 1, stride := 4, owners := [.residual 55] }
  , { sourceStart := 1298709, count := 1, stride := 600, owners := [.template 2 27] }
  , { sourceStart := 1299309, count := 1, stride := 4, owners := [.residual 56] }
  , { sourceStart := 1299313, count := 1, stride := 600, owners := [.template 2 28] }
  , { sourceStart := 1299913, count := 1, stride := 4, owners := [.residual 57] }
  , { sourceStart := 1299917, count := 1, stride := 600, owners := [.template 2 29] }
  , { sourceStart := 1300517, count := 1, stride := 1, owners := [.residual 58] }
  , { sourceStart := 1300518, count := 1, stride := 600, owners := [.template 1 1] }
  , { sourceStart := 1301118, count := 1, stride := 1, owners := [.residual 59] }
  , { sourceStart := 1301119, count := 1, stride := 1, owners := [.residual 63] }
  , { sourceStart := 1301120, count := 1, stride := 1, owners := [.residual 60] }
  , { sourceStart := 1301121, count := 1, stride := 1, owners := [.residual 64] }
  , { sourceStart := 1301122, count := 1, stride := 1, owners := [.residual 61] }
  , { sourceStart := 1301123, count := 1, stride := 1, owners := [.residual 65] }
  , { sourceStart := 1301124, count := 1, stride := 1, owners := [.residual 62] }
  , { sourceStart := 1301125, count := 1, stride := 1, owners := [.residual 66] }
  ]

def evenArm : RawArm where
  schemaVersion := 1
  arm := 0
  sourceStart := 1
  sourceEnd := 1301126
  finalColumns := 8858862
  templates := evenTemplates
  residualBatches := evenResidualBatches
  coverGroups := evenCoverGroups

def oddTemplateInstances00 : List RawTemplateInstances :=
  [
    { sourceStart := 52144, count := 918, sourceStride := 122, finalStart := 2110685, finalStride := 20, referenceStart := 1559, referenceStride := 1, referenceFinalStart := 38340, referenceFinalStride := 41 }
  ]

def oddTemplateInstances01 : List RawTemplateInstances :=
  [
    { sourceStart := 1296268, count := 1, sourceStride := 0, finalStart := 8822732, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1301718, count := 1, sourceStride := 0, finalStart := 8854890, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  ]

def oddTemplateInstances02 : List RawTemplateInstances :=
  [
    { sourceStart := 166320, count := 244, sourceStride := 600, finalStart := 2218425, finalStride := 3526, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 312741, count := 1, sourceStride := 0, finalStart := 3078981, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 313342, count := 1, sourceStride := 0, finalStart := 3082507, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 313942, count := 260, sourceStride := 600, finalStart := 3086033, finalStride := 3526, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 469943, count := 1, sourceStride := 0, finalStart := 4002793, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 470556, count := 1, sourceStride := 0, finalStart := 4006319, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 471157, count := 261, sourceStride := 600, finalStart := 4009845, finalStride := 3526, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 627758, count := 1, sourceStride := 0, finalStart := 4930131, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 630553, count := 1, sourceStride := 0, finalStart := 4935826, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 631157, count := 545, sourceStride := 604, finalStart := 4939352, finalStride := 3526, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 960334, count := 1, sourceStride := 0, finalStart := 6861022, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 960950, count := 1, sourceStride := 0, finalStart := 6864548, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 961554, count := 545, sourceStride := 604, finalStart := 6868074, finalStride := 3526, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1290731, count := 1, sourceStride := 0, finalStart := 8789744, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1291439, count := 1, sourceStride := 0, finalStart := 8794524, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1292043, count := 1, sourceStride := 0, finalStart := 8798050, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1292647, count := 1, sourceStride := 0, finalStart := 8801576, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1293251, count := 1, sourceStride := 0, finalStart := 8805102, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1293855, count := 1, sourceStride := 0, finalStart := 8808628, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1294459, count := 1, sourceStride := 0, finalStart := 8812154, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1295063, count := 1, sourceStride := 0, finalStart := 8815680, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1295667, count := 1, sourceStride := 0, finalStart := 8819206, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1296889, count := 1, sourceStride := 0, finalStart := 8826682, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1297493, count := 1, sourceStride := 0, finalStart := 8830208, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1298097, count := 1, sourceStride := 0, finalStart := 8833734, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1298701, count := 1, sourceStride := 0, finalStart := 8837260, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1299305, count := 1, sourceStride := 0, finalStart := 8840786, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1299909, count := 1, sourceStride := 0, finalStart := 8844312, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1300513, count := 1, sourceStride := 0, finalStart := 8847838, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 1301117, count := 1, sourceStride := 0, finalStart := 8851364, finalStride := 0, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  ]

def oddTemplates : List RawTemplate :=
  [
    { sourceWidth := 122, relativeRuns := templateRules00, instances := oddTemplateInstances00 }
  , { sourceWidth := 600, relativeRuns := templateRules01, instances := oddTemplateInstances01 }
  , { sourceWidth := 600, relativeRuns := templateRules02, instances := oddTemplateInstances02 }
  ]

def oddResidualBatches : List RawResidualBatch :=
  [
    { sourceStart := 1, instanceCount := 1, instanceStride := 0, width := 640, resolution := .direct 1 1 1 false }
  , { sourceStart := 641, instanceCount := 1, instanceStride := 0, width := 51462, resolution := .direct 702 41 41 false }
  , { sourceStart := 52103, instanceCount := 1, instanceStride := 0, width := 41, resolution := .direct 2110644 1 1 true }
  , { sourceStart := 164140, instanceCount := 1, instanceStride := 0, width := 2180, resolution := .direct 2129045 41 41 false }
  , { sourceStart := 312720, instanceCount := 1, instanceStride := 0, width := 2, resolution := .direct 3078769 64 64 false }
  , { sourceStart := 312722, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 3078897 42 1 false }
  , { sourceStart := 312724, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 3078939 42 1 false }
  , { sourceStart := 312723, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 3078898 42 41 false }
  , { sourceStart := 312725, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 3078940 42 41 false }
  , { sourceStart := 312726, instanceCount := 1, instanceStride := 0, width := 15, resolution := .linearDefinition }
  , { sourceStart := 313341, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 469942, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 470543, instanceCount := 1, instanceStride := 0, width := 13, resolution := .linearDefinition }
  , { sourceStart := 471156, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 627757, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 628358, instanceCount := 1, instanceStride := 0, width := 6, resolution := .linearDefinition }
  , { sourceStart := 628364, instanceCount := 1, instanceStride := 0, width := 2169, resolution := .direct 4933657 1 1 false }
  , { sourceStart := 630533, instanceCount := 1, instanceStride := 0, width := 20, resolution := .linearDefinition }
  , { sourceStart := 631153, instanceCount := 545, instanceStride := 604, width := 4, resolution := .linearDefinition }
  , { sourceStart := 960333, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 960934, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 960935, instanceCount := 1, instanceStride := 0, width := 15, resolution := .linearDefinition }
  , { sourceStart := 961550, instanceCount := 545, instanceStride := 604, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1290730, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 1291360, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 1291331, instanceCount := 1, instanceStride := 0, width := 8, resolution := .direct 8793270 41 41 false }
  , { sourceStart := 1291339, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8793598 0 64 false }
  , { sourceStart := 1291340, instanceCount := 1, instanceStride := 0, width := 20, resolution := .direct 8793662 41 41 false }
  , { sourceStart := 1291361, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1291365, instanceCount := 1, instanceStride := 0, width := 64, resolution := .decompositionAlias 1291339 0 0 1 8793598 1 false }
  , { sourceStart := 1291429, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8794482 32032 1 false }
  , { sourceStart := 1296868, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8826514 32032 1 false }
  , { sourceStart := 1291430, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8794483 32032 41 false }
  , { sourceStart := 1296869, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8826515 32032 41 false }
  , { sourceStart := 1291431, instanceCount := 1, instanceStride := 0, width := 8, resolution := .linearDefinition }
  , { sourceStart := 1292039, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1292643, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1293247, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1293851, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1294455, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1295059, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1295663, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1296267, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 1296876, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 1296870, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8826556 42 1 false }
  , { sourceStart := 1296872, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8826598 42 1 false }
  , { sourceStart := 1296874, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8826640 42 1 false }
  , { sourceStart := 1296871, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8826557 42 41 false }
  , { sourceStart := 1296873, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8826599 42 41 false }
  , { sourceStart := 1296875, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8826641 42 41 false }
  , { sourceStart := 1296877, instanceCount := 1, instanceStride := 0, width := 12, resolution := .linearDefinition }
  , { sourceStart := 1297489, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1298093, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1298697, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1299301, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1299905, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1300509, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1301113, instanceCount := 1, instanceStride := 0, width := 4, resolution := .linearDefinition }
  , { sourceStart := 1301717, instanceCount := 1, instanceStride := 0, width := 1, resolution := .linearDefinition }
  , { sourceStart := 1302318, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8858672 42 1 false }
  , { sourceStart := 1302320, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8858714 42 1 false }
  , { sourceStart := 1302322, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8858756 42 1 false }
  , { sourceStart := 1302324, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8858798 42 1 false }
  , { sourceStart := 1302319, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8858673 42 41 false }
  , { sourceStart := 1302321, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8858715 42 41 false }
  , { sourceStart := 1302323, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8858757 42 41 false }
  , { sourceStart := 1302325, instanceCount := 1, instanceStride := 0, width := 1, resolution := .direct 8858799 42 41 false }
  ]

def oddCoverGroups : List RawCoverGroup :=
  [
    { sourceStart := 1, count := 1, stride := 640, owners := [.residual 0] }
  , { sourceStart := 641, count := 1, stride := 51462, owners := [.residual 1] }
  , { sourceStart := 52103, count := 1, stride := 41, owners := [.residual 2] }
  , { sourceStart := 52144, count := 918, stride := 122, owners := [.template 0 0] }
  , { sourceStart := 164140, count := 1, stride := 2180, owners := [.residual 3] }
  , { sourceStart := 166320, count := 244, stride := 600, owners := [.template 2 0] }
  , { sourceStart := 312720, count := 1, stride := 2, owners := [.residual 4] }
  , { sourceStart := 312722, count := 1, stride := 1, owners := [.residual 5] }
  , { sourceStart := 312723, count := 1, stride := 1, owners := [.residual 7] }
  , { sourceStart := 312724, count := 1, stride := 1, owners := [.residual 6] }
  , { sourceStart := 312725, count := 1, stride := 1, owners := [.residual 8] }
  , { sourceStart := 312726, count := 1, stride := 15, owners := [.residual 9] }
  , { sourceStart := 312741, count := 1, stride := 600, owners := [.template 2 1] }
  , { sourceStart := 313341, count := 1, stride := 1, owners := [.residual 10] }
  , { sourceStart := 313342, count := 1, stride := 600, owners := [.template 2 2] }
  , { sourceStart := 313942, count := 260, stride := 600, owners := [.template 2 3] }
  , { sourceStart := 469942, count := 1, stride := 1, owners := [.residual 11] }
  , { sourceStart := 469943, count := 1, stride := 600, owners := [.template 2 4] }
  , { sourceStart := 470543, count := 1, stride := 13, owners := [.residual 12] }
  , { sourceStart := 470556, count := 1, stride := 600, owners := [.template 2 5] }
  , { sourceStart := 471156, count := 1, stride := 1, owners := [.residual 13] }
  , { sourceStart := 471157, count := 261, stride := 600, owners := [.template 2 6] }
  , { sourceStart := 627757, count := 1, stride := 1, owners := [.residual 14] }
  , { sourceStart := 627758, count := 1, stride := 600, owners := [.template 2 7] }
  , { sourceStart := 628358, count := 1, stride := 6, owners := [.residual 15] }
  , { sourceStart := 628364, count := 1, stride := 2169, owners := [.residual 16] }
  , { sourceStart := 630533, count := 1, stride := 20, owners := [.residual 17] }
  , { sourceStart := 630553, count := 1, stride := 600, owners := [.template 2 8] }
  , { sourceStart := 631153, count := 545, stride := 604, owners := [.residual 18, .template 2 9] }
  , { sourceStart := 960333, count := 1, stride := 1, owners := [.residual 19] }
  , { sourceStart := 960334, count := 1, stride := 600, owners := [.template 2 10] }
  , { sourceStart := 960934, count := 1, stride := 1, owners := [.residual 20] }
  , { sourceStart := 960935, count := 1, stride := 15, owners := [.residual 21] }
  , { sourceStart := 960950, count := 1, stride := 600, owners := [.template 2 11] }
  , { sourceStart := 961550, count := 545, stride := 604, owners := [.residual 22, .template 2 12] }
  , { sourceStart := 1290730, count := 1, stride := 1, owners := [.residual 23] }
  , { sourceStart := 1290731, count := 1, stride := 600, owners := [.template 2 13] }
  , { sourceStart := 1291331, count := 1, stride := 8, owners := [.residual 25] }
  , { sourceStart := 1291339, count := 1, stride := 1, owners := [.residual 26] }
  , { sourceStart := 1291340, count := 1, stride := 20, owners := [.residual 27] }
  , { sourceStart := 1291360, count := 1, stride := 1, owners := [.residual 24] }
  , { sourceStart := 1291361, count := 1, stride := 4, owners := [.residual 28] }
  , { sourceStart := 1291365, count := 1, stride := 64, owners := [.residual 29] }
  , { sourceStart := 1291429, count := 1, stride := 1, owners := [.residual 30] }
  , { sourceStart := 1291430, count := 1, stride := 1, owners := [.residual 32] }
  , { sourceStart := 1291431, count := 1, stride := 8, owners := [.residual 34] }
  , { sourceStart := 1291439, count := 1, stride := 600, owners := [.template 2 14] }
  , { sourceStart := 1292039, count := 1, stride := 4, owners := [.residual 35] }
  , { sourceStart := 1292043, count := 1, stride := 600, owners := [.template 2 15] }
  , { sourceStart := 1292643, count := 1, stride := 4, owners := [.residual 36] }
  , { sourceStart := 1292647, count := 1, stride := 600, owners := [.template 2 16] }
  , { sourceStart := 1293247, count := 1, stride := 4, owners := [.residual 37] }
  , { sourceStart := 1293251, count := 1, stride := 600, owners := [.template 2 17] }
  , { sourceStart := 1293851, count := 1, stride := 4, owners := [.residual 38] }
  , { sourceStart := 1293855, count := 1, stride := 600, owners := [.template 2 18] }
  , { sourceStart := 1294455, count := 1, stride := 4, owners := [.residual 39] }
  , { sourceStart := 1294459, count := 1, stride := 600, owners := [.template 2 19] }
  , { sourceStart := 1295059, count := 1, stride := 4, owners := [.residual 40] }
  , { sourceStart := 1295063, count := 1, stride := 600, owners := [.template 2 20] }
  , { sourceStart := 1295663, count := 1, stride := 4, owners := [.residual 41] }
  , { sourceStart := 1295667, count := 1, stride := 600, owners := [.template 2 21] }
  , { sourceStart := 1296267, count := 1, stride := 1, owners := [.residual 42] }
  , { sourceStart := 1296268, count := 1, stride := 600, owners := [.template 1 0] }
  , { sourceStart := 1296868, count := 1, stride := 1, owners := [.residual 31] }
  , { sourceStart := 1296869, count := 1, stride := 1, owners := [.residual 33] }
  , { sourceStart := 1296870, count := 1, stride := 1, owners := [.residual 44] }
  , { sourceStart := 1296871, count := 1, stride := 1, owners := [.residual 47] }
  , { sourceStart := 1296872, count := 1, stride := 1, owners := [.residual 45] }
  , { sourceStart := 1296873, count := 1, stride := 1, owners := [.residual 48] }
  , { sourceStart := 1296874, count := 1, stride := 1, owners := [.residual 46] }
  , { sourceStart := 1296875, count := 1, stride := 1, owners := [.residual 49] }
  , { sourceStart := 1296876, count := 1, stride := 1, owners := [.residual 43] }
  , { sourceStart := 1296877, count := 1, stride := 12, owners := [.residual 50] }
  , { sourceStart := 1296889, count := 1, stride := 600, owners := [.template 2 22] }
  , { sourceStart := 1297489, count := 1, stride := 4, owners := [.residual 51] }
  , { sourceStart := 1297493, count := 1, stride := 600, owners := [.template 2 23] }
  , { sourceStart := 1298093, count := 1, stride := 4, owners := [.residual 52] }
  , { sourceStart := 1298097, count := 1, stride := 600, owners := [.template 2 24] }
  , { sourceStart := 1298697, count := 1, stride := 4, owners := [.residual 53] }
  , { sourceStart := 1298701, count := 1, stride := 600, owners := [.template 2 25] }
  , { sourceStart := 1299301, count := 1, stride := 4, owners := [.residual 54] }
  , { sourceStart := 1299305, count := 1, stride := 600, owners := [.template 2 26] }
  , { sourceStart := 1299905, count := 1, stride := 4, owners := [.residual 55] }
  , { sourceStart := 1299909, count := 1, stride := 600, owners := [.template 2 27] }
  , { sourceStart := 1300509, count := 1, stride := 4, owners := [.residual 56] }
  , { sourceStart := 1300513, count := 1, stride := 600, owners := [.template 2 28] }
  , { sourceStart := 1301113, count := 1, stride := 4, owners := [.residual 57] }
  , { sourceStart := 1301117, count := 1, stride := 600, owners := [.template 2 29] }
  , { sourceStart := 1301717, count := 1, stride := 1, owners := [.residual 58] }
  , { sourceStart := 1301718, count := 1, stride := 600, owners := [.template 1 1] }
  , { sourceStart := 1302318, count := 1, stride := 1, owners := [.residual 59] }
  , { sourceStart := 1302319, count := 1, stride := 1, owners := [.residual 63] }
  , { sourceStart := 1302320, count := 1, stride := 1, owners := [.residual 60] }
  , { sourceStart := 1302321, count := 1, stride := 1, owners := [.residual 64] }
  , { sourceStart := 1302322, count := 1, stride := 1, owners := [.residual 61] }
  , { sourceStart := 1302323, count := 1, stride := 1, owners := [.residual 65] }
  , { sourceStart := 1302324, count := 1, stride := 1, owners := [.residual 62] }
  , { sourceStart := 1302325, count := 1, stride := 1, owners := [.residual 66] }
  ]

def oddArm : RawArm where
  schemaVersion := 1
  arm := 1
  sourceStart := 1
  sourceEnd := 1302326
  finalColumns := 8858862
  templates := oddTemplates
  residualBatches := oddResidualBatches
  coverGroups := oddCoverGroups

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder
