import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder.Schema

/-!
Generated file: exact production full-`Z` decoder layout; do not hand-edit.

The artifact records the stabilized relation width and the same column-major
`(lane, block)` convention exercised through Rust's actual
`decode_superneo_coeffs_from_witness_mat` implementation. It is compact:
the full logical-coordinate map is represented by one affine block stride
and 64 proof-free lane records, not by enumerating every witness cell.

Owns: production packed-witness dimensions and live/virtual lane provenance.

The bounded constructor probe additionally records every one-hot logical
column exercised through `CcsInstance::from_low_norm_assignment`, the actual
Ajtai verifier-key dimensions, commitment recomputation, and column-major
commitment-data indexing.

Does not own: witness values, commitment binding, NC acceptance, generated
delayed-projection rows, transcript scheduling, or row-removal permission.

Emits constraints: none; generated direct-dataflow evidence only.

| Stable stage path | Obligation | Authority |
|---|---|---|
| `pi_ccs_nc.delayed_projection.full_z_decoder.generated` | exact full-width block/lane decoder and 54+10 lane partition | computed artifact |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder.Generated.Layout

def schemaVersion : Nat := 2
def relationRows : Nat := 14944219
def logicalWidth : Nat := 11437038
def childCount : Nat := 14
def matrixRows : Nat := 54
def matrixColumns : Nat := 211797
def booleanLaneCount : Nat := 64
def fixtureCommitmentWidth : Nat := 4
def fixtureCommitmentDataLength : Nat := 216
def productionCommitmentWidth : Nat := 18
def productionCommitmentDataLength : Nat := 972
def commitmentProbeBlocks : Nat := 2
def laneSources : List LaneSourceRecord := [
  { booleanLane := 0, witnessLane := some 0 }
, { booleanLane := 1, witnessLane := some 1 }
, { booleanLane := 2, witnessLane := some 2 }
, { booleanLane := 3, witnessLane := some 3 }
, { booleanLane := 4, witnessLane := some 4 }
, { booleanLane := 5, witnessLane := some 5 }
, { booleanLane := 6, witnessLane := some 6 }
, { booleanLane := 7, witnessLane := some 7 }
, { booleanLane := 8, witnessLane := some 8 }
, { booleanLane := 9, witnessLane := some 9 }
, { booleanLane := 10, witnessLane := some 10 }
, { booleanLane := 11, witnessLane := some 11 }
, { booleanLane := 12, witnessLane := some 12 }
, { booleanLane := 13, witnessLane := some 13 }
, { booleanLane := 14, witnessLane := some 14 }
, { booleanLane := 15, witnessLane := some 15 }
, { booleanLane := 16, witnessLane := some 16 }
, { booleanLane := 17, witnessLane := some 17 }
, { booleanLane := 18, witnessLane := some 18 }
, { booleanLane := 19, witnessLane := some 19 }
, { booleanLane := 20, witnessLane := some 20 }
, { booleanLane := 21, witnessLane := some 21 }
, { booleanLane := 22, witnessLane := some 22 }
, { booleanLane := 23, witnessLane := some 23 }
, { booleanLane := 24, witnessLane := some 24 }
, { booleanLane := 25, witnessLane := some 25 }
, { booleanLane := 26, witnessLane := some 26 }
, { booleanLane := 27, witnessLane := some 27 }
, { booleanLane := 28, witnessLane := some 28 }
, { booleanLane := 29, witnessLane := some 29 }
, { booleanLane := 30, witnessLane := some 30 }
, { booleanLane := 31, witnessLane := some 31 }
, { booleanLane := 32, witnessLane := some 32 }
, { booleanLane := 33, witnessLane := some 33 }
, { booleanLane := 34, witnessLane := some 34 }
, { booleanLane := 35, witnessLane := some 35 }
, { booleanLane := 36, witnessLane := some 36 }
, { booleanLane := 37, witnessLane := some 37 }
, { booleanLane := 38, witnessLane := some 38 }
, { booleanLane := 39, witnessLane := some 39 }
, { booleanLane := 40, witnessLane := some 40 }
, { booleanLane := 41, witnessLane := some 41 }
, { booleanLane := 42, witnessLane := some 42 }
, { booleanLane := 43, witnessLane := some 43 }
, { booleanLane := 44, witnessLane := some 44 }
, { booleanLane := 45, witnessLane := some 45 }
, { booleanLane := 46, witnessLane := some 46 }
, { booleanLane := 47, witnessLane := some 47 }
, { booleanLane := 48, witnessLane := some 48 }
, { booleanLane := 49, witnessLane := some 49 }
, { booleanLane := 50, witnessLane := some 50 }
, { booleanLane := 51, witnessLane := some 51 }
, { booleanLane := 52, witnessLane := some 52 }
, { booleanLane := 53, witnessLane := some 53 }
, { booleanLane := 54, witnessLane := none }
, { booleanLane := 55, witnessLane := none }
, { booleanLane := 56, witnessLane := none }
, { booleanLane := 57, witnessLane := none }
, { booleanLane := 58, witnessLane := none }
, { booleanLane := 59, witnessLane := none }
, { booleanLane := 60, witnessLane := none }
, { booleanLane := 61, witnessLane := none }
, { booleanLane := 62, witnessLane := none }
, { booleanLane := 63, witnessLane := none }
]
def commitmentProbeColumns : List Nat := [
  0
, 1
, 2
, 3
, 4
, 5
, 6
, 7
, 8
, 9
, 10
, 11
, 12
, 13
, 14
, 15
, 16
, 17
, 18
, 19
, 20
, 21
, 22
, 23
, 24
, 25
, 26
, 27
, 28
, 29
, 30
, 31
, 32
, 33
, 34
, 35
, 36
, 37
, 38
, 39
, 40
, 41
, 42
, 43
, 44
, 45
, 46
, 47
, 48
, 49
, 50
, 51
, 52
, 53
, 54
, 55
, 56
, 57
, 58
, 59
, 60
, 61
, 62
, 63
, 64
, 65
, 66
, 67
, 68
, 69
, 70
, 71
, 72
, 73
, 74
, 75
, 76
, 77
, 78
, 79
, 80
, 81
, 82
, 83
, 84
, 85
, 86
, 87
, 88
, 89
, 90
, 91
, 92
, 93
, 94
, 95
, 96
, 97
, 98
, 99
, 100
, 101
, 102
, 103
, 104
, 105
, 106
, 107
]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder.Generated.Layout
