import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Schema

/-!
Generated file: authoritative raw-running assignment decoder chunk; do not
hand-edit.

Each provenance record carries both the normalized source-arm column and its
complete final selective-assignment scalar encoding. The generator fails
closed unless the final interval and encoding kind come from the exact direct
slot for the record's actual
`running[child].x[(logicalColumn % 54) * x_cols + logicalColumn / 54]` wire.

`balancedTernary` means the field value is reconstructed as
`sum(digit[i] * 3^i)` from exactly 41 signed-unit digits. It is not a binary
encoding and the first digit is not the scalar value.

This data does not establish delayed-projection acceptance, raw-child semantic
authority, commitment binding, or row-removal permission.

Owns: one exact 252-record raw-running physical-column provenance shard.

Does not own: assignment values, combined-NC acceptance, transcript scheduling,
commitment binding, or permission to remove rows.

Emits constraints: none; generated data only.

| Stable stage path | Obligation | Authority |
|---|---|---|
| `pi_ccs_nc.delayed_projection.raw_running_decoder.generated.chunk` | Exact generated coordinate-to-column records | computed artifact |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk0

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 0, logicalColumn := 0, sourceArmColumn := 21819, finalStart := 133997, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 1, sourceArmColumn := 21824, finalStart := 134202, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 2, sourceArmColumn := 21829, finalStart := 134407, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 3, sourceArmColumn := 21834, finalStart := 134612, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 4, sourceArmColumn := 21839, finalStart := 134817, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 5, sourceArmColumn := 21844, finalStart := 135022, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 6, sourceArmColumn := 21849, finalStart := 135227, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 7, sourceArmColumn := 21854, finalStart := 135432, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 8, sourceArmColumn := 21859, finalStart := 135637, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 9, sourceArmColumn := 21864, finalStart := 135842, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 10, sourceArmColumn := 21869, finalStart := 136047, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 11, sourceArmColumn := 21874, finalStart := 136252, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 12, sourceArmColumn := 21879, finalStart := 136457, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 13, sourceArmColumn := 21884, finalStart := 136662, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 14, sourceArmColumn := 21889, finalStart := 136867, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 15, sourceArmColumn := 21894, finalStart := 137072, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 16, sourceArmColumn := 21899, finalStart := 137277, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 17, sourceArmColumn := 21904, finalStart := 137482, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 18, sourceArmColumn := 21909, finalStart := 137687, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 19, sourceArmColumn := 21914, finalStart := 137892, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 20, sourceArmColumn := 21919, finalStart := 138097, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 21, sourceArmColumn := 21924, finalStart := 138302, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 22, sourceArmColumn := 21929, finalStart := 138507, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 23, sourceArmColumn := 21934, finalStart := 138712, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 24, sourceArmColumn := 21939, finalStart := 138917, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 25, sourceArmColumn := 21944, finalStart := 139122, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 26, sourceArmColumn := 21949, finalStart := 139327, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 27, sourceArmColumn := 21954, finalStart := 139532, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 28, sourceArmColumn := 21959, finalStart := 139737, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 29, sourceArmColumn := 21964, finalStart := 139942, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 30, sourceArmColumn := 21969, finalStart := 140147, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 31, sourceArmColumn := 21974, finalStart := 140352, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 32, sourceArmColumn := 21979, finalStart := 140557, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 33, sourceArmColumn := 21984, finalStart := 140762, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 34, sourceArmColumn := 21989, finalStart := 140967, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 35, sourceArmColumn := 21994, finalStart := 141172, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 36, sourceArmColumn := 21999, finalStart := 141377, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 37, sourceArmColumn := 22004, finalStart := 141582, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 38, sourceArmColumn := 22009, finalStart := 141787, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 39, sourceArmColumn := 22014, finalStart := 141992, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 40, sourceArmColumn := 22019, finalStart := 142197, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 41, sourceArmColumn := 22024, finalStart := 142402, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 42, sourceArmColumn := 22029, finalStart := 142607, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 43, sourceArmColumn := 22034, finalStart := 142812, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 44, sourceArmColumn := 22039, finalStart := 143017, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 45, sourceArmColumn := 22044, finalStart := 143222, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 46, sourceArmColumn := 22049, finalStart := 143427, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 47, sourceArmColumn := 22054, finalStart := 143632, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 48, sourceArmColumn := 22059, finalStart := 143837, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 49, sourceArmColumn := 22064, finalStart := 144042, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 50, sourceArmColumn := 22069, finalStart := 144247, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 51, sourceArmColumn := 22074, finalStart := 144452, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 52, sourceArmColumn := 22079, finalStart := 144657, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 53, sourceArmColumn := 22084, finalStart := 144862, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 54, sourceArmColumn := 21820, finalStart := 134038, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 55, sourceArmColumn := 21825, finalStart := 134243, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 56, sourceArmColumn := 21830, finalStart := 134448, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 57, sourceArmColumn := 21835, finalStart := 134653, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 58, sourceArmColumn := 21840, finalStart := 134858, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 59, sourceArmColumn := 21845, finalStart := 135063, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 60, sourceArmColumn := 21850, finalStart := 135268, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 61, sourceArmColumn := 21855, finalStart := 135473, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 62, sourceArmColumn := 21860, finalStart := 135678, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 63, sourceArmColumn := 21865, finalStart := 135883, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 64, sourceArmColumn := 21870, finalStart := 136088, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 65, sourceArmColumn := 21875, finalStart := 136293, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 66, sourceArmColumn := 21880, finalStart := 136498, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 67, sourceArmColumn := 21885, finalStart := 136703, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 68, sourceArmColumn := 21890, finalStart := 136908, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 69, sourceArmColumn := 21895, finalStart := 137113, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 70, sourceArmColumn := 21900, finalStart := 137318, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 71, sourceArmColumn := 21905, finalStart := 137523, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 72, sourceArmColumn := 21910, finalStart := 137728, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 73, sourceArmColumn := 21915, finalStart := 137933, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 74, sourceArmColumn := 21920, finalStart := 138138, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 75, sourceArmColumn := 21925, finalStart := 138343, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 76, sourceArmColumn := 21930, finalStart := 138548, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 77, sourceArmColumn := 21935, finalStart := 138753, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 78, sourceArmColumn := 21940, finalStart := 138958, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 79, sourceArmColumn := 21945, finalStart := 139163, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 80, sourceArmColumn := 21950, finalStart := 139368, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 81, sourceArmColumn := 21955, finalStart := 139573, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 82, sourceArmColumn := 21960, finalStart := 139778, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 83, sourceArmColumn := 21965, finalStart := 139983, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 84, sourceArmColumn := 21970, finalStart := 140188, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 85, sourceArmColumn := 21975, finalStart := 140393, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 86, sourceArmColumn := 21980, finalStart := 140598, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 87, sourceArmColumn := 21985, finalStart := 140803, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 88, sourceArmColumn := 21990, finalStart := 141008, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 89, sourceArmColumn := 21995, finalStart := 141213, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 90, sourceArmColumn := 22000, finalStart := 141418, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 91, sourceArmColumn := 22005, finalStart := 141623, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 92, sourceArmColumn := 22010, finalStart := 141828, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 93, sourceArmColumn := 22015, finalStart := 142033, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 94, sourceArmColumn := 22020, finalStart := 142238, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 95, sourceArmColumn := 22025, finalStart := 142443, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 96, sourceArmColumn := 22030, finalStart := 142648, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 97, sourceArmColumn := 22035, finalStart := 142853, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 98, sourceArmColumn := 22040, finalStart := 143058, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 99, sourceArmColumn := 22045, finalStart := 143263, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 100, sourceArmColumn := 22050, finalStart := 143468, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 101, sourceArmColumn := 22055, finalStart := 143673, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 102, sourceArmColumn := 22060, finalStart := 143878, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 103, sourceArmColumn := 22065, finalStart := 144083, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 104, sourceArmColumn := 22070, finalStart := 144288, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 105, sourceArmColumn := 22075, finalStart := 144493, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 106, sourceArmColumn := 22080, finalStart := 144698, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 107, sourceArmColumn := 22085, finalStart := 144903, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 108, sourceArmColumn := 21821, finalStart := 134079, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 109, sourceArmColumn := 21826, finalStart := 134284, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 110, sourceArmColumn := 21831, finalStart := 134489, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 111, sourceArmColumn := 21836, finalStart := 134694, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 112, sourceArmColumn := 21841, finalStart := 134899, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 113, sourceArmColumn := 21846, finalStart := 135104, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 114, sourceArmColumn := 21851, finalStart := 135309, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 115, sourceArmColumn := 21856, finalStart := 135514, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 116, sourceArmColumn := 21861, finalStart := 135719, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 117, sourceArmColumn := 21866, finalStart := 135924, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 118, sourceArmColumn := 21871, finalStart := 136129, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 119, sourceArmColumn := 21876, finalStart := 136334, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 120, sourceArmColumn := 21881, finalStart := 136539, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 121, sourceArmColumn := 21886, finalStart := 136744, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 122, sourceArmColumn := 21891, finalStart := 136949, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 123, sourceArmColumn := 21896, finalStart := 137154, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 124, sourceArmColumn := 21901, finalStart := 137359, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 125, sourceArmColumn := 21906, finalStart := 137564, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 126, sourceArmColumn := 21911, finalStart := 137769, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 127, sourceArmColumn := 21916, finalStart := 137974, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 128, sourceArmColumn := 21921, finalStart := 138179, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 129, sourceArmColumn := 21926, finalStart := 138384, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 130, sourceArmColumn := 21931, finalStart := 138589, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 131, sourceArmColumn := 21936, finalStart := 138794, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 132, sourceArmColumn := 21941, finalStart := 138999, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 133, sourceArmColumn := 21946, finalStart := 139204, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 134, sourceArmColumn := 21951, finalStart := 139409, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 135, sourceArmColumn := 21956, finalStart := 139614, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 136, sourceArmColumn := 21961, finalStart := 139819, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 137, sourceArmColumn := 21966, finalStart := 140024, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 138, sourceArmColumn := 21971, finalStart := 140229, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 139, sourceArmColumn := 21976, finalStart := 140434, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 140, sourceArmColumn := 21981, finalStart := 140639, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 141, sourceArmColumn := 21986, finalStart := 140844, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 142, sourceArmColumn := 21991, finalStart := 141049, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 143, sourceArmColumn := 21996, finalStart := 141254, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 144, sourceArmColumn := 22001, finalStart := 141459, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 145, sourceArmColumn := 22006, finalStart := 141664, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 146, sourceArmColumn := 22011, finalStart := 141869, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 147, sourceArmColumn := 22016, finalStart := 142074, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 148, sourceArmColumn := 22021, finalStart := 142279, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 149, sourceArmColumn := 22026, finalStart := 142484, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 150, sourceArmColumn := 22031, finalStart := 142689, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 151, sourceArmColumn := 22036, finalStart := 142894, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 152, sourceArmColumn := 22041, finalStart := 143099, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 153, sourceArmColumn := 22046, finalStart := 143304, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 154, sourceArmColumn := 22051, finalStart := 143509, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 155, sourceArmColumn := 22056, finalStart := 143714, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 156, sourceArmColumn := 22061, finalStart := 143919, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 157, sourceArmColumn := 22066, finalStart := 144124, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 158, sourceArmColumn := 22071, finalStart := 144329, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 159, sourceArmColumn := 22076, finalStart := 144534, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 160, sourceArmColumn := 22081, finalStart := 144739, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 161, sourceArmColumn := 22086, finalStart := 144944, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 162, sourceArmColumn := 21822, finalStart := 134120, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 163, sourceArmColumn := 21827, finalStart := 134325, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 164, sourceArmColumn := 21832, finalStart := 134530, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 165, sourceArmColumn := 21837, finalStart := 134735, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 166, sourceArmColumn := 21842, finalStart := 134940, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 167, sourceArmColumn := 21847, finalStart := 135145, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 168, sourceArmColumn := 21852, finalStart := 135350, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 169, sourceArmColumn := 21857, finalStart := 135555, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 170, sourceArmColumn := 21862, finalStart := 135760, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 171, sourceArmColumn := 21867, finalStart := 135965, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 172, sourceArmColumn := 21872, finalStart := 136170, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 173, sourceArmColumn := 21877, finalStart := 136375, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 174, sourceArmColumn := 21882, finalStart := 136580, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 175, sourceArmColumn := 21887, finalStart := 136785, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 176, sourceArmColumn := 21892, finalStart := 136990, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 177, sourceArmColumn := 21897, finalStart := 137195, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 178, sourceArmColumn := 21902, finalStart := 137400, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 179, sourceArmColumn := 21907, finalStart := 137605, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 180, sourceArmColumn := 21912, finalStart := 137810, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 181, sourceArmColumn := 21917, finalStart := 138015, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 182, sourceArmColumn := 21922, finalStart := 138220, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 183, sourceArmColumn := 21927, finalStart := 138425, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 184, sourceArmColumn := 21932, finalStart := 138630, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 185, sourceArmColumn := 21937, finalStart := 138835, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 186, sourceArmColumn := 21942, finalStart := 139040, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 187, sourceArmColumn := 21947, finalStart := 139245, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 188, sourceArmColumn := 21952, finalStart := 139450, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 189, sourceArmColumn := 21957, finalStart := 139655, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 190, sourceArmColumn := 21962, finalStart := 139860, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 191, sourceArmColumn := 21967, finalStart := 140065, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 192, sourceArmColumn := 21972, finalStart := 140270, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 193, sourceArmColumn := 21977, finalStart := 140475, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 194, sourceArmColumn := 21982, finalStart := 140680, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 195, sourceArmColumn := 21987, finalStart := 140885, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 196, sourceArmColumn := 21992, finalStart := 141090, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 197, sourceArmColumn := 21997, finalStart := 141295, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 198, sourceArmColumn := 22002, finalStart := 141500, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 199, sourceArmColumn := 22007, finalStart := 141705, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 200, sourceArmColumn := 22012, finalStart := 141910, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 201, sourceArmColumn := 22017, finalStart := 142115, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 202, sourceArmColumn := 22022, finalStart := 142320, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 203, sourceArmColumn := 22027, finalStart := 142525, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 204, sourceArmColumn := 22032, finalStart := 142730, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 205, sourceArmColumn := 22037, finalStart := 142935, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 206, sourceArmColumn := 22042, finalStart := 143140, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 207, sourceArmColumn := 22047, finalStart := 143345, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 208, sourceArmColumn := 22052, finalStart := 143550, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 209, sourceArmColumn := 22057, finalStart := 143755, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 210, sourceArmColumn := 22062, finalStart := 143960, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 211, sourceArmColumn := 22067, finalStart := 144165, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 212, sourceArmColumn := 22072, finalStart := 144370, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 213, sourceArmColumn := 22077, finalStart := 144575, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 214, sourceArmColumn := 22082, finalStart := 144780, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 215, sourceArmColumn := 22087, finalStart := 144985, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 216, sourceArmColumn := 21823, finalStart := 134161, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 217, sourceArmColumn := 21828, finalStart := 134366, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 218, sourceArmColumn := 21833, finalStart := 134571, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 219, sourceArmColumn := 21838, finalStart := 134776, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 220, sourceArmColumn := 21843, finalStart := 134981, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 221, sourceArmColumn := 21848, finalStart := 135186, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 222, sourceArmColumn := 21853, finalStart := 135391, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 223, sourceArmColumn := 21858, finalStart := 135596, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 224, sourceArmColumn := 21863, finalStart := 135801, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 225, sourceArmColumn := 21868, finalStart := 136006, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 226, sourceArmColumn := 21873, finalStart := 136211, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 227, sourceArmColumn := 21878, finalStart := 136416, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 228, sourceArmColumn := 21883, finalStart := 136621, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 229, sourceArmColumn := 21888, finalStart := 136826, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 230, sourceArmColumn := 21893, finalStart := 137031, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 231, sourceArmColumn := 21898, finalStart := 137236, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 232, sourceArmColumn := 21903, finalStart := 137441, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 233, sourceArmColumn := 21908, finalStart := 137646, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 234, sourceArmColumn := 21913, finalStart := 137851, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 235, sourceArmColumn := 21918, finalStart := 138056, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 236, sourceArmColumn := 21923, finalStart := 138261, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 237, sourceArmColumn := 21928, finalStart := 138466, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 238, sourceArmColumn := 21933, finalStart := 138671, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 239, sourceArmColumn := 21938, finalStart := 138876, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 240, sourceArmColumn := 21943, finalStart := 139081, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 241, sourceArmColumn := 21948, finalStart := 139286, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 242, sourceArmColumn := 21953, finalStart := 139491, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 243, sourceArmColumn := 21958, finalStart := 139696, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 244, sourceArmColumn := 21963, finalStart := 139901, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 245, sourceArmColumn := 21968, finalStart := 140106, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 246, sourceArmColumn := 21973, finalStart := 140311, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 247, sourceArmColumn := 21978, finalStart := 140516, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 248, sourceArmColumn := 21983, finalStart := 140721, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 249, sourceArmColumn := 21988, finalStart := 140926, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 250, sourceArmColumn := 21993, finalStart := 141131, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 251, sourceArmColumn := 21998, finalStart := 141336, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk0
