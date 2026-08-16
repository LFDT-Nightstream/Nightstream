import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema

/-! Generated file: exact compact source-to-final decoder for both production
PiRLC parity bodies.

Owns: the two source ranges, final normalized column bound, three shared
decoder templates, exact affine template instances, and residual strided
rules emitted from the norm-base-four production selective layout.

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
  , { sourceStart := 11, length := 1, resolution := .direct 0 0 23 false }
  , { sourceStart := 12, length := 3, resolution := .traceEliminated }
  , { sourceStart := 15, length := 1, resolution := .direct 23 0 23 false }
  , { sourceStart := 16, length := 3, resolution := .traceEliminated }
  , { sourceStart := 19, length := 1, resolution := .direct 46 0 23 false }
  , { sourceStart := 20, length := 3, resolution := .traceEliminated }
  , { sourceStart := 23, length := 1, resolution := .direct 69 0 23 false }
  , { sourceStart := 24, length := 3, resolution := .traceEliminated }
  , { sourceStart := 27, length := 1, resolution := .direct 92 0 23 false }
  , { sourceStart := 28, length := 3, resolution := .traceEliminated }
  , { sourceStart := 31, length := 1, resolution := .direct 115 0 23 false }
  , { sourceStart := 32, length := 3, resolution := .traceEliminated }
  , { sourceStart := 35, length := 1, resolution := .direct 138 0 23 false }
  , { sourceStart := 36, length := 3, resolution := .traceEliminated }
  , { sourceStart := 39, length := 1, resolution := .direct 161 0 23 false }
  , { sourceStart := 40, length := 11, resolution := .traceEliminated }
  , { sourceStart := 51, length := 1, resolution := .direct 184 0 23 false }
  , { sourceStart := 52, length := 3, resolution := .traceEliminated }
  , { sourceStart := 55, length := 1, resolution := .direct 207 0 23 false }
  , { sourceStart := 56, length := 3, resolution := .traceEliminated }
  , { sourceStart := 59, length := 1, resolution := .direct 230 0 23 false }
  , { sourceStart := 60, length := 3, resolution := .traceEliminated }
  , { sourceStart := 63, length := 1, resolution := .direct 253 0 23 false }
  , { sourceStart := 64, length := 3, resolution := .traceEliminated }
  , { sourceStart := 67, length := 1, resolution := .direct 276 0 23 false }
  , { sourceStart := 68, length := 3, resolution := .traceEliminated }
  , { sourceStart := 71, length := 1, resolution := .direct 299 0 23 false }
  , { sourceStart := 72, length := 3, resolution := .traceEliminated }
  , { sourceStart := 75, length := 1, resolution := .direct 322 0 23 false }
  , { sourceStart := 76, length := 3, resolution := .traceEliminated }
  , { sourceStart := 79, length := 1, resolution := .direct 345 0 23 false }
  , { sourceStart := 80, length := 11, resolution := .traceEliminated }
  , { sourceStart := 91, length := 1, resolution := .direct 368 0 23 false }
  , { sourceStart := 92, length := 3, resolution := .traceEliminated }
  , { sourceStart := 95, length := 1, resolution := .direct 391 0 23 false }
  , { sourceStart := 96, length := 3, resolution := .traceEliminated }
  , { sourceStart := 99, length := 1, resolution := .direct 414 0 23 false }
  , { sourceStart := 100, length := 3, resolution := .traceEliminated }
  , { sourceStart := 103, length := 1, resolution := .direct 437 0 23 false }
  , { sourceStart := 104, length := 3, resolution := .traceEliminated }
  , { sourceStart := 107, length := 1, resolution := .direct 460 0 23 false }
  , { sourceStart := 108, length := 3, resolution := .traceEliminated }
  , { sourceStart := 111, length := 1, resolution := .direct 483 0 23 false }
  , { sourceStart := 112, length := 3, resolution := .traceEliminated }
  , { sourceStart := 115, length := 1, resolution := .direct 506 0 23 false }
  , { sourceStart := 116, length := 3, resolution := .traceEliminated }
  , { sourceStart := 119, length := 1, resolution := .direct 529 0 23 false }
  , { sourceStart := 120, length := 11, resolution := .traceEliminated }
  , { sourceStart := 131, length := 1, resolution := .direct 552 0 23 false }
  , { sourceStart := 132, length := 3, resolution := .traceEliminated }
  , { sourceStart := 135, length := 1, resolution := .direct 575 0 23 false }
  , { sourceStart := 136, length := 3, resolution := .traceEliminated }
  , { sourceStart := 139, length := 1, resolution := .direct 598 0 23 false }
  , { sourceStart := 140, length := 3, resolution := .traceEliminated }
  , { sourceStart := 143, length := 1, resolution := .direct 621 0 23 false }
  , { sourceStart := 144, length := 3, resolution := .traceEliminated }
  , { sourceStart := 147, length := 1, resolution := .direct 644 0 23 false }
  , { sourceStart := 148, length := 3, resolution := .traceEliminated }
  , { sourceStart := 151, length := 1, resolution := .direct 667 0 23 false }
  , { sourceStart := 152, length := 3, resolution := .traceEliminated }
  , { sourceStart := 155, length := 1, resolution := .direct 690 0 23 false }
  , { sourceStart := 156, length := 3, resolution := .traceEliminated }
  , { sourceStart := 159, length := 1, resolution := .direct 713 0 23 false }
  , { sourceStart := 160, length := 11, resolution := .traceEliminated }
  , { sourceStart := 171, length := 1, resolution := .direct 736 0 23 false }
  , { sourceStart := 172, length := 11, resolution := .traceEliminated }
  , { sourceStart := 183, length := 1, resolution := .direct 759 0 23 false }
  , { sourceStart := 184, length := 11, resolution := .traceEliminated }
  , { sourceStart := 195, length := 1, resolution := .direct 782 0 23 false }
  , { sourceStart := 196, length := 11, resolution := .traceEliminated }
  , { sourceStart := 207, length := 1, resolution := .direct 805 0 23 false }
  , { sourceStart := 208, length := 11, resolution := .traceEliminated }
  , { sourceStart := 219, length := 1, resolution := .direct 828 0 23 false }
  , { sourceStart := 220, length := 11, resolution := .traceEliminated }
  , { sourceStart := 231, length := 1, resolution := .direct 851 0 23 false }
  , { sourceStart := 232, length := 11, resolution := .traceEliminated }
  , { sourceStart := 243, length := 1, resolution := .direct 874 0 23 false }
  , { sourceStart := 244, length := 11, resolution := .traceEliminated }
  , { sourceStart := 255, length := 1, resolution := .direct 897 0 23 false }
  , { sourceStart := 256, length := 11, resolution := .traceEliminated }
  , { sourceStart := 267, length := 1, resolution := .direct 920 0 23 false }
  , { sourceStart := 268, length := 11, resolution := .traceEliminated }
  , { sourceStart := 279, length := 1, resolution := .direct 943 0 23 false }
  , { sourceStart := 280, length := 11, resolution := .traceEliminated }
  , { sourceStart := 291, length := 1, resolution := .direct 966 0 23 false }
  , { sourceStart := 292, length := 11, resolution := .traceEliminated }
  , { sourceStart := 303, length := 1, resolution := .direct 989 0 23 false }
  , { sourceStart := 304, length := 11, resolution := .traceEliminated }
  , { sourceStart := 315, length := 1, resolution := .direct 1012 0 23 false }
  , { sourceStart := 316, length := 11, resolution := .traceEliminated }
  , { sourceStart := 327, length := 1, resolution := .direct 1035 0 23 false }
  , { sourceStart := 328, length := 11, resolution := .traceEliminated }
  , { sourceStart := 339, length := 1, resolution := .direct 1058 0 23 false }
  , { sourceStart := 340, length := 11, resolution := .traceEliminated }
  , { sourceStart := 351, length := 1, resolution := .direct 1081 0 23 false }
  , { sourceStart := 352, length := 11, resolution := .traceEliminated }
  , { sourceStart := 363, length := 1, resolution := .direct 1104 0 23 false }
  , { sourceStart := 364, length := 11, resolution := .traceEliminated }
  , { sourceStart := 375, length := 1, resolution := .direct 1127 0 23 false }
  , { sourceStart := 376, length := 11, resolution := .traceEliminated }
  , { sourceStart := 387, length := 1, resolution := .direct 1150 0 23 false }
  , { sourceStart := 388, length := 11, resolution := .traceEliminated }
  , { sourceStart := 399, length := 1, resolution := .direct 1173 0 23 false }
  , { sourceStart := 400, length := 11, resolution := .traceEliminated }
  , { sourceStart := 411, length := 1, resolution := .direct 1196 0 23 false }
  , { sourceStart := 412, length := 11, resolution := .traceEliminated }
  , { sourceStart := 423, length := 1, resolution := .direct 1219 0 23 false }
  , { sourceStart := 424, length := 11, resolution := .traceEliminated }
  , { sourceStart := 435, length := 1, resolution := .direct 1242 0 23 false }
  , { sourceStart := 436, length := 3, resolution := .traceEliminated }
  , { sourceStart := 439, length := 1, resolution := .direct 1265 0 23 false }
  , { sourceStart := 440, length := 3, resolution := .traceEliminated }
  , { sourceStart := 443, length := 1, resolution := .direct 1288 0 23 false }
  , { sourceStart := 444, length := 3, resolution := .traceEliminated }
  , { sourceStart := 447, length := 1, resolution := .direct 1311 0 23 false }
  , { sourceStart := 448, length := 3, resolution := .traceEliminated }
  , { sourceStart := 451, length := 1, resolution := .direct 1334 0 23 false }
  , { sourceStart := 452, length := 3, resolution := .traceEliminated }
  , { sourceStart := 455, length := 1, resolution := .direct 1357 0 23 false }
  , { sourceStart := 456, length := 3, resolution := .traceEliminated }
  , { sourceStart := 459, length := 1, resolution := .direct 1380 0 23 false }
  , { sourceStart := 460, length := 3, resolution := .traceEliminated }
  , { sourceStart := 463, length := 1, resolution := .direct 1403 0 23 false }
  , { sourceStart := 464, length := 11, resolution := .traceEliminated }
  , { sourceStart := 475, length := 1, resolution := .direct 1426 0 23 false }
  , { sourceStart := 476, length := 3, resolution := .traceEliminated }
  , { sourceStart := 479, length := 1, resolution := .direct 1449 0 23 false }
  , { sourceStart := 480, length := 3, resolution := .traceEliminated }
  , { sourceStart := 483, length := 1, resolution := .direct 1472 0 23 false }
  , { sourceStart := 484, length := 3, resolution := .traceEliminated }
  , { sourceStart := 487, length := 1, resolution := .direct 1495 0 23 false }
  , { sourceStart := 488, length := 3, resolution := .traceEliminated }
  , { sourceStart := 491, length := 1, resolution := .direct 1518 0 23 false }
  , { sourceStart := 492, length := 3, resolution := .traceEliminated }
  , { sourceStart := 495, length := 1, resolution := .direct 1541 0 23 false }
  , { sourceStart := 496, length := 3, resolution := .traceEliminated }
  , { sourceStart := 499, length := 1, resolution := .direct 1564 0 23 false }
  , { sourceStart := 500, length := 3, resolution := .traceEliminated }
  , { sourceStart := 503, length := 1, resolution := .direct 1587 0 23 false }
  , { sourceStart := 504, length := 11, resolution := .traceEliminated }
  , { sourceStart := 515, length := 1, resolution := .direct 1610 0 23 false }
  , { sourceStart := 516, length := 3, resolution := .traceEliminated }
  , { sourceStart := 519, length := 1, resolution := .direct 1633 0 23 false }
  , { sourceStart := 520, length := 3, resolution := .traceEliminated }
  , { sourceStart := 523, length := 1, resolution := .direct 1656 0 23 false }
  , { sourceStart := 524, length := 3, resolution := .traceEliminated }
  , { sourceStart := 527, length := 1, resolution := .direct 1679 0 23 false }
  , { sourceStart := 528, length := 3, resolution := .traceEliminated }
  , { sourceStart := 531, length := 1, resolution := .direct 1702 0 23 false }
  , { sourceStart := 532, length := 3, resolution := .traceEliminated }
  , { sourceStart := 535, length := 1, resolution := .direct 1725 0 23 false }
  , { sourceStart := 536, length := 3, resolution := .traceEliminated }
  , { sourceStart := 539, length := 1, resolution := .direct 1748 0 23 false }
  , { sourceStart := 540, length := 3, resolution := .traceEliminated }
  , { sourceStart := 543, length := 1, resolution := .direct 1771 0 23 false }
  , { sourceStart := 544, length := 11, resolution := .traceEliminated }
  , { sourceStart := 555, length := 1, resolution := .direct 1794 0 23 false }
  , { sourceStart := 556, length := 3, resolution := .traceEliminated }
  , { sourceStart := 559, length := 1, resolution := .direct 1817 0 23 false }
  , { sourceStart := 560, length := 3, resolution := .traceEliminated }
  , { sourceStart := 563, length := 1, resolution := .direct 1840 0 23 false }
  , { sourceStart := 564, length := 3, resolution := .traceEliminated }
  , { sourceStart := 567, length := 1, resolution := .direct 1863 0 23 false }
  , { sourceStart := 568, length := 3, resolution := .traceEliminated }
  , { sourceStart := 571, length := 1, resolution := .direct 1886 0 23 false }
  , { sourceStart := 572, length := 3, resolution := .traceEliminated }
  , { sourceStart := 575, length := 1, resolution := .direct 1909 0 23 false }
  , { sourceStart := 576, length := 3, resolution := .traceEliminated }
  , { sourceStart := 579, length := 1, resolution := .direct 1932 0 23 false }
  , { sourceStart := 580, length := 3, resolution := .traceEliminated }
  , { sourceStart := 583, length := 1, resolution := .direct 1955 0 23 false }
  , { sourceStart := 584, length := 8, resolution := .traceEliminated }
  , { sourceStart := 592, length := 4, resolution := .direct 1978 64 64 false }
  , { sourceStart := 596, length := 4, resolution := .linearDefinition }
  ]

def templateRules02 : List RawRun :=
  [
    { sourceStart := 0, length := 11, resolution := .traceEliminated }
  , { sourceStart := 11, length := 1, resolution := .direct 0 0 23 false }
  , { sourceStart := 12, length := 3, resolution := .traceEliminated }
  , { sourceStart := 15, length := 1, resolution := .direct 23 0 23 false }
  , { sourceStart := 16, length := 3, resolution := .traceEliminated }
  , { sourceStart := 19, length := 1, resolution := .direct 46 0 23 false }
  , { sourceStart := 20, length := 3, resolution := .traceEliminated }
  , { sourceStart := 23, length := 1, resolution := .direct 69 0 23 false }
  , { sourceStart := 24, length := 3, resolution := .traceEliminated }
  , { sourceStart := 27, length := 1, resolution := .direct 92 0 23 false }
  , { sourceStart := 28, length := 3, resolution := .traceEliminated }
  , { sourceStart := 31, length := 1, resolution := .direct 115 0 23 false }
  , { sourceStart := 32, length := 3, resolution := .traceEliminated }
  , { sourceStart := 35, length := 1, resolution := .direct 138 0 23 false }
  , { sourceStart := 36, length := 3, resolution := .traceEliminated }
  , { sourceStart := 39, length := 1, resolution := .direct 161 0 23 false }
  , { sourceStart := 40, length := 11, resolution := .traceEliminated }
  , { sourceStart := 51, length := 1, resolution := .direct 184 0 23 false }
  , { sourceStart := 52, length := 3, resolution := .traceEliminated }
  , { sourceStart := 55, length := 1, resolution := .direct 207 0 23 false }
  , { sourceStart := 56, length := 3, resolution := .traceEliminated }
  , { sourceStart := 59, length := 1, resolution := .direct 230 0 23 false }
  , { sourceStart := 60, length := 3, resolution := .traceEliminated }
  , { sourceStart := 63, length := 1, resolution := .direct 253 0 23 false }
  , { sourceStart := 64, length := 3, resolution := .traceEliminated }
  , { sourceStart := 67, length := 1, resolution := .direct 276 0 23 false }
  , { sourceStart := 68, length := 3, resolution := .traceEliminated }
  , { sourceStart := 71, length := 1, resolution := .direct 299 0 23 false }
  , { sourceStart := 72, length := 3, resolution := .traceEliminated }
  , { sourceStart := 75, length := 1, resolution := .direct 322 0 23 false }
  , { sourceStart := 76, length := 3, resolution := .traceEliminated }
  , { sourceStart := 79, length := 1, resolution := .direct 345 0 23 false }
  , { sourceStart := 80, length := 11, resolution := .traceEliminated }
  , { sourceStart := 91, length := 1, resolution := .direct 368 0 23 false }
  , { sourceStart := 92, length := 3, resolution := .traceEliminated }
  , { sourceStart := 95, length := 1, resolution := .direct 391 0 23 false }
  , { sourceStart := 96, length := 3, resolution := .traceEliminated }
  , { sourceStart := 99, length := 1, resolution := .direct 414 0 23 false }
  , { sourceStart := 100, length := 3, resolution := .traceEliminated }
  , { sourceStart := 103, length := 1, resolution := .direct 437 0 23 false }
  , { sourceStart := 104, length := 3, resolution := .traceEliminated }
  , { sourceStart := 107, length := 1, resolution := .direct 460 0 23 false }
  , { sourceStart := 108, length := 3, resolution := .traceEliminated }
  , { sourceStart := 111, length := 1, resolution := .direct 483 0 23 false }
  , { sourceStart := 112, length := 3, resolution := .traceEliminated }
  , { sourceStart := 115, length := 1, resolution := .direct 506 0 23 false }
  , { sourceStart := 116, length := 3, resolution := .traceEliminated }
  , { sourceStart := 119, length := 1, resolution := .direct 529 0 23 false }
  , { sourceStart := 120, length := 11, resolution := .traceEliminated }
  , { sourceStart := 131, length := 1, resolution := .direct 552 0 23 false }
  , { sourceStart := 132, length := 3, resolution := .traceEliminated }
  , { sourceStart := 135, length := 1, resolution := .direct 575 0 23 false }
  , { sourceStart := 136, length := 3, resolution := .traceEliminated }
  , { sourceStart := 139, length := 1, resolution := .direct 598 0 23 false }
  , { sourceStart := 140, length := 3, resolution := .traceEliminated }
  , { sourceStart := 143, length := 1, resolution := .direct 621 0 23 false }
  , { sourceStart := 144, length := 3, resolution := .traceEliminated }
  , { sourceStart := 147, length := 1, resolution := .direct 644 0 23 false }
  , { sourceStart := 148, length := 3, resolution := .traceEliminated }
  , { sourceStart := 151, length := 1, resolution := .direct 667 0 23 false }
  , { sourceStart := 152, length := 3, resolution := .traceEliminated }
  , { sourceStart := 155, length := 1, resolution := .direct 690 0 23 false }
  , { sourceStart := 156, length := 3, resolution := .traceEliminated }
  , { sourceStart := 159, length := 1, resolution := .direct 713 0 23 false }
  , { sourceStart := 160, length := 11, resolution := .traceEliminated }
  , { sourceStart := 171, length := 1, resolution := .direct 736 0 23 false }
  , { sourceStart := 172, length := 11, resolution := .traceEliminated }
  , { sourceStart := 183, length := 1, resolution := .direct 759 0 23 false }
  , { sourceStart := 184, length := 11, resolution := .traceEliminated }
  , { sourceStart := 195, length := 1, resolution := .direct 782 0 23 false }
  , { sourceStart := 196, length := 11, resolution := .traceEliminated }
  , { sourceStart := 207, length := 1, resolution := .direct 805 0 23 false }
  , { sourceStart := 208, length := 11, resolution := .traceEliminated }
  , { sourceStart := 219, length := 1, resolution := .direct 828 0 23 false }
  , { sourceStart := 220, length := 11, resolution := .traceEliminated }
  , { sourceStart := 231, length := 1, resolution := .direct 851 0 23 false }
  , { sourceStart := 232, length := 11, resolution := .traceEliminated }
  , { sourceStart := 243, length := 1, resolution := .direct 874 0 23 false }
  , { sourceStart := 244, length := 11, resolution := .traceEliminated }
  , { sourceStart := 255, length := 1, resolution := .direct 897 0 23 false }
  , { sourceStart := 256, length := 11, resolution := .traceEliminated }
  , { sourceStart := 267, length := 1, resolution := .direct 920 0 23 false }
  , { sourceStart := 268, length := 11, resolution := .traceEliminated }
  , { sourceStart := 279, length := 1, resolution := .direct 943 0 23 false }
  , { sourceStart := 280, length := 11, resolution := .traceEliminated }
  , { sourceStart := 291, length := 1, resolution := .direct 966 0 23 false }
  , { sourceStart := 292, length := 11, resolution := .traceEliminated }
  , { sourceStart := 303, length := 1, resolution := .direct 989 0 23 false }
  , { sourceStart := 304, length := 11, resolution := .traceEliminated }
  , { sourceStart := 315, length := 1, resolution := .direct 1012 0 23 false }
  , { sourceStart := 316, length := 11, resolution := .traceEliminated }
  , { sourceStart := 327, length := 1, resolution := .direct 1035 0 23 false }
  , { sourceStart := 328, length := 11, resolution := .traceEliminated }
  , { sourceStart := 339, length := 1, resolution := .direct 1058 0 23 false }
  , { sourceStart := 340, length := 11, resolution := .traceEliminated }
  , { sourceStart := 351, length := 1, resolution := .direct 1081 0 23 false }
  , { sourceStart := 352, length := 11, resolution := .traceEliminated }
  , { sourceStart := 363, length := 1, resolution := .direct 1104 0 23 false }
  , { sourceStart := 364, length := 11, resolution := .traceEliminated }
  , { sourceStart := 375, length := 1, resolution := .direct 1127 0 23 false }
  , { sourceStart := 376, length := 11, resolution := .traceEliminated }
  , { sourceStart := 387, length := 1, resolution := .direct 1150 0 23 false }
  , { sourceStart := 388, length := 11, resolution := .traceEliminated }
  , { sourceStart := 399, length := 1, resolution := .direct 1173 0 23 false }
  , { sourceStart := 400, length := 11, resolution := .traceEliminated }
  , { sourceStart := 411, length := 1, resolution := .direct 1196 0 23 false }
  , { sourceStart := 412, length := 11, resolution := .traceEliminated }
  , { sourceStart := 423, length := 1, resolution := .direct 1219 0 23 false }
  , { sourceStart := 424, length := 11, resolution := .traceEliminated }
  , { sourceStart := 435, length := 1, resolution := .direct 1242 0 23 false }
  , { sourceStart := 436, length := 3, resolution := .traceEliminated }
  , { sourceStart := 439, length := 1, resolution := .direct 1265 0 23 false }
  , { sourceStart := 440, length := 3, resolution := .traceEliminated }
  , { sourceStart := 443, length := 1, resolution := .direct 1288 0 23 false }
  , { sourceStart := 444, length := 3, resolution := .traceEliminated }
  , { sourceStart := 447, length := 1, resolution := .direct 1311 0 23 false }
  , { sourceStart := 448, length := 3, resolution := .traceEliminated }
  , { sourceStart := 451, length := 1, resolution := .direct 1334 0 23 false }
  , { sourceStart := 452, length := 3, resolution := .traceEliminated }
  , { sourceStart := 455, length := 1, resolution := .direct 1357 0 23 false }
  , { sourceStart := 456, length := 3, resolution := .traceEliminated }
  , { sourceStart := 459, length := 1, resolution := .direct 1380 0 23 false }
  , { sourceStart := 460, length := 3, resolution := .traceEliminated }
  , { sourceStart := 463, length := 1, resolution := .direct 1403 0 23 false }
  , { sourceStart := 464, length := 11, resolution := .traceEliminated }
  , { sourceStart := 475, length := 1, resolution := .direct 1426 0 23 false }
  , { sourceStart := 476, length := 3, resolution := .traceEliminated }
  , { sourceStart := 479, length := 1, resolution := .direct 1449 0 23 false }
  , { sourceStart := 480, length := 3, resolution := .traceEliminated }
  , { sourceStart := 483, length := 1, resolution := .direct 1472 0 23 false }
  , { sourceStart := 484, length := 3, resolution := .traceEliminated }
  , { sourceStart := 487, length := 1, resolution := .direct 1495 0 23 false }
  , { sourceStart := 488, length := 3, resolution := .traceEliminated }
  , { sourceStart := 491, length := 1, resolution := .direct 1518 0 23 false }
  , { sourceStart := 492, length := 3, resolution := .traceEliminated }
  , { sourceStart := 495, length := 1, resolution := .direct 1541 0 23 false }
  , { sourceStart := 496, length := 3, resolution := .traceEliminated }
  , { sourceStart := 499, length := 1, resolution := .direct 1564 0 23 false }
  , { sourceStart := 500, length := 3, resolution := .traceEliminated }
  , { sourceStart := 503, length := 1, resolution := .direct 1587 0 23 false }
  , { sourceStart := 504, length := 11, resolution := .traceEliminated }
  , { sourceStart := 515, length := 1, resolution := .direct 1610 0 23 false }
  , { sourceStart := 516, length := 3, resolution := .traceEliminated }
  , { sourceStart := 519, length := 1, resolution := .direct 1633 0 23 false }
  , { sourceStart := 520, length := 3, resolution := .traceEliminated }
  , { sourceStart := 523, length := 1, resolution := .direct 1656 0 23 false }
  , { sourceStart := 524, length := 3, resolution := .traceEliminated }
  , { sourceStart := 527, length := 1, resolution := .direct 1679 0 23 false }
  , { sourceStart := 528, length := 3, resolution := .traceEliminated }
  , { sourceStart := 531, length := 1, resolution := .direct 1702 0 23 false }
  , { sourceStart := 532, length := 3, resolution := .traceEliminated }
  , { sourceStart := 535, length := 1, resolution := .direct 1725 0 23 false }
  , { sourceStart := 536, length := 3, resolution := .traceEliminated }
  , { sourceStart := 539, length := 1, resolution := .direct 1748 0 23 false }
  , { sourceStart := 540, length := 3, resolution := .traceEliminated }
  , { sourceStart := 543, length := 1, resolution := .direct 1771 0 23 false }
  , { sourceStart := 544, length := 11, resolution := .traceEliminated }
  , { sourceStart := 555, length := 1, resolution := .direct 1794 0 23 false }
  , { sourceStart := 556, length := 3, resolution := .traceEliminated }
  , { sourceStart := 559, length := 1, resolution := .direct 1817 0 23 false }
  , { sourceStart := 560, length := 3, resolution := .traceEliminated }
  , { sourceStart := 563, length := 1, resolution := .direct 1840 0 23 false }
  , { sourceStart := 564, length := 3, resolution := .traceEliminated }
  , { sourceStart := 567, length := 1, resolution := .direct 1863 0 23 false }
  , { sourceStart := 568, length := 3, resolution := .traceEliminated }
  , { sourceStart := 571, length := 1, resolution := .direct 1886 0 23 false }
  , { sourceStart := 572, length := 3, resolution := .traceEliminated }
  , { sourceStart := 575, length := 1, resolution := .direct 1909 0 23 false }
  , { sourceStart := 576, length := 3, resolution := .traceEliminated }
  , { sourceStart := 579, length := 1, resolution := .direct 1932 0 23 false }
  , { sourceStart := 580, length := 3, resolution := .traceEliminated }
  , { sourceStart := 583, length := 1, resolution := .direct 1955 0 23 false }
  , { sourceStart := 584, length := 8, resolution := .traceEliminated }
  , { sourceStart := 592, length := 8, resolution := .linearDefinition }
  ]

def evenTemplateInstances00 : List RawTemplateInstances :=
  [
    { sourceStart := 46096, count := 810, sourceStride := 122, finalStart := 1059845, finalStride := 20, referenceStart := 1451, referenceStride := 1, referenceFinalStart := 19332, referenceFinalStride := 41 }
  ]

def evenTemplateInstances01 : List RawTemplateInstances :=
  [
    { sourceStart := 564057, count := 2, sourceStride := 5450, finalStart := 2496835, finalStride := 18154, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  ]

def evenTemplateInstances02 : List RawTemplateInstances :=
  [
    { sourceStart := 146880, count := 215, sourceStride := 600, finalStart := 1121217, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 275901, count := 2, sourceStride := 601, finalStart := 1546663, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 277102, count := 233, sourceStride := 600, finalStart := 1550619, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 416903, count := 2, sourceStride := 613, finalStart := 2011493, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 418117, count := 234, sourceStride := 600, finalStart := 2015449, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 558518, count := 2, sourceStride := 710, finalStart := 2478301, finalStride := 2710, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 559832, count := 7, sourceStride := 604, finalStart := 2482989, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 564678, count := 8, sourceStride := 604, finalStart := 2499165, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  ]

def evenTemplates : List RawTemplate :=
  [
    { sourceWidth := 122, relativeRuns := templateRules00, instances := evenTemplateInstances00 }
  , { sourceWidth := 600, relativeRuns := templateRules01, instances := evenTemplateInstances01 }
  , { sourceWidth := 600, relativeRuns := templateRules02, instances := evenTemplateInstances02 }
  ]

def evenResidualRuns : List RawStridedRun :=
  [
    { sourceStart := 1, count := 640, sourceStride := 1, resolution := .direct 1 1 1 false }
  , { sourceStart := 641, count := 810, sourceStride := 1, resolution := .direct 702 23 23 false }
  , { sourceStart := 1451, count := 810, sourceStride := 1, resolution := .direct 19332 41 41 false }
  , { sourceStart := 2261, count := 43794, sourceStride := 1, resolution := .direct 52542 23 23 false }
  , { sourceStart := 46055, count := 41, sourceStride := 1, resolution := .direct 1059804 1 1 true }
  , { sourceStart := 144916, count := 1964, sourceStride := 1, resolution := .direct 1076045 23 23 false }
  , { sourceStart := 275880, count := 2, sourceStride := 1, resolution := .direct 1546487 64 64 false }
  , { sourceStart := 275882, count := 2, sourceStride := 2, resolution := .direct 1546615 24 1 false }
  , { sourceStart := 275883, count := 2, sourceStride := 2, resolution := .direct 1546616 24 23 false }
  , { sourceStart := 275886, count := 15, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 276501, count := 2, sourceStride := 140401, resolution := .linearDefinition }
  , { sourceStart := 417503, count := 13, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 418116, count := 2, sourceStride := 140401, resolution := .linearDefinition }
  , { sourceStart := 559118, count := 2, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 559120, count := 8, sourceStride := 1, resolution := .direct 2480279 23 23 false }
  , { sourceStart := 559128, count := 1, sourceStride := 1, resolution := .direct 2480463 0 64 false }
  , { sourceStart := 559129, count := 20, sourceStride := 1, resolution := .direct 2480527 23 23 false }
  , { sourceStart := 559149, count := 5, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 559154, count := 64, sourceStride := 1, resolution := .decompositionAlias 559128 0 0 1 2480463 1 false }
  , { sourceStart := 559218, count := 2, sourceStride := 5439, resolution := .direct 2480987 18082 1 false }
  , { sourceStart := 559219, count := 2, sourceStride := 5439, resolution := .direct 2480988 18082 23 false }
  , { sourceStart := 559220, count := 8, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 559828, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 560432, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 561036, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 561640, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 562244, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 562848, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 563452, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 564056, count := 2, sourceStride := 609, resolution := .linearDefinition }
  , { sourceStart := 564659, count := 3, sourceStride := 2, resolution := .direct 2499093 24 1 false }
  , { sourceStart := 564660, count := 3, sourceStride := 2, resolution := .direct 2499094 24 23 false }
  , { sourceStart := 564666, count := 12, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 565278, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 565882, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 566486, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 567090, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 567694, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 568298, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 568902, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 569506, count := 1, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 570107, count := 4, sourceStride := 2, resolution := .direct 2517223 24 1 false }
  , { sourceStart := 570108, count := 4, sourceStride := 2, resolution := .direct 2517224 24 23 false }
  ]

def evenArm : RawArm where
  schemaVersion := 1
  arm := 0
  sourceStart := 1
  sourceEnd := 570115
  finalColumns := 2521314
  templates := evenTemplates
  residualRuns := evenResidualRuns

def oddTemplateInstances00 : List RawTemplateInstances :=
  [
    { sourceStart := 46096, count := 810, sourceStride := 122, finalStart := 1059845, finalStride := 20, referenceStart := 1451, referenceStride := 1, referenceFinalStart := 19332, referenceFinalStride := 41 }
  ]

def oddTemplateInstances01 : List RawTemplateInstances :=
  [
    { sourceStart := 565257, count := 2, sourceStride := 5450, finalStart := 2500791, finalStride := 18154, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  ]

def oddTemplateInstances02 : List RawTemplateInstances :=
  [
    { sourceStart := 146880, count := 217, sourceStride := 600, finalStart := 1121217, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 277101, count := 2, sourceStride := 601, finalStart := 1550619, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 278302, count := 233, sourceStride := 600, finalStart := 1554575, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 418103, count := 2, sourceStride := 613, finalStart := 2015449, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 419317, count := 234, sourceStride := 600, finalStart := 2019405, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 559718, count := 2, sourceStride := 710, finalStart := 2482257, finalStride := 2710, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 561032, count := 7, sourceStride := 604, finalStart := 2486945, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  , { sourceStart := 565878, count := 8, sourceStride := 604, finalStart := 2503121, finalStride := 1978, referenceStart := 0, referenceStride := 0, referenceFinalStart := 0, referenceFinalStride := 0 }
  ]

def oddTemplates : List RawTemplate :=
  [
    { sourceWidth := 122, relativeRuns := templateRules00, instances := oddTemplateInstances00 }
  , { sourceWidth := 600, relativeRuns := templateRules01, instances := oddTemplateInstances01 }
  , { sourceWidth := 600, relativeRuns := templateRules02, instances := oddTemplateInstances02 }
  ]

def oddResidualRuns : List RawStridedRun :=
  [
    { sourceStart := 1, count := 640, sourceStride := 1, resolution := .direct 1 1 1 false }
  , { sourceStart := 641, count := 810, sourceStride := 1, resolution := .direct 702 23 23 false }
  , { sourceStart := 1451, count := 810, sourceStride := 1, resolution := .direct 19332 41 41 false }
  , { sourceStart := 2261, count := 43794, sourceStride := 1, resolution := .direct 52542 23 23 false }
  , { sourceStart := 46055, count := 41, sourceStride := 1, resolution := .direct 1059804 1 1 true }
  , { sourceStart := 144916, count := 1964, sourceStride := 1, resolution := .direct 1076045 23 23 false }
  , { sourceStart := 277080, count := 2, sourceStride := 1, resolution := .direct 1550443 64 64 false }
  , { sourceStart := 277082, count := 2, sourceStride := 2, resolution := .direct 1550571 24 1 false }
  , { sourceStart := 277083, count := 2, sourceStride := 2, resolution := .direct 1550572 24 23 false }
  , { sourceStart := 277086, count := 15, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 277701, count := 2, sourceStride := 140401, resolution := .linearDefinition }
  , { sourceStart := 418703, count := 13, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 419316, count := 2, sourceStride := 140401, resolution := .linearDefinition }
  , { sourceStart := 560318, count := 2, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 560320, count := 8, sourceStride := 1, resolution := .direct 2484235 23 23 false }
  , { sourceStart := 560328, count := 1, sourceStride := 1, resolution := .direct 2484419 0 64 false }
  , { sourceStart := 560329, count := 20, sourceStride := 1, resolution := .direct 2484483 23 23 false }
  , { sourceStart := 560349, count := 5, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 560354, count := 64, sourceStride := 1, resolution := .decompositionAlias 560328 0 0 1 2484419 1 false }
  , { sourceStart := 560418, count := 2, sourceStride := 5439, resolution := .direct 2484943 18082 1 false }
  , { sourceStart := 560419, count := 2, sourceStride := 5439, resolution := .direct 2484944 18082 23 false }
  , { sourceStart := 560420, count := 8, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 561028, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 561632, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 562236, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 562840, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 563444, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 564048, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 564652, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 565256, count := 2, sourceStride := 609, resolution := .linearDefinition }
  , { sourceStart := 565859, count := 3, sourceStride := 2, resolution := .direct 2503049 24 1 false }
  , { sourceStart := 565860, count := 3, sourceStride := 2, resolution := .direct 2503050 24 23 false }
  , { sourceStart := 565866, count := 12, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 566478, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 567082, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 567686, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 568290, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 568894, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 569498, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 570102, count := 4, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 570706, count := 1, sourceStride := 1, resolution := .linearDefinition }
  , { sourceStart := 571307, count := 4, sourceStride := 2, resolution := .direct 2521179 24 1 false }
  , { sourceStart := 571308, count := 4, sourceStride := 2, resolution := .direct 2521180 24 23 false }
  ]

def oddArm : RawArm where
  schemaVersion := 1
  arm := 1
  sourceStart := 1
  sourceEnd := 571315
  finalColumns := 2521314
  templates := oddTemplates
  residualRuns := oddResidualRuns

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder
