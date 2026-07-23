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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk12

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 11, logicalColumn := 54, sourceArmColumn := 46812, finalStart := 990118, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 55, sourceArmColumn := 46817, finalStart := 990323, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 56, sourceArmColumn := 46822, finalStart := 990528, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 57, sourceArmColumn := 46827, finalStart := 990733, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 58, sourceArmColumn := 46832, finalStart := 990938, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 59, sourceArmColumn := 46837, finalStart := 991143, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 60, sourceArmColumn := 46842, finalStart := 991348, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 61, sourceArmColumn := 46847, finalStart := 991553, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 62, sourceArmColumn := 46852, finalStart := 991758, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 63, sourceArmColumn := 46857, finalStart := 991963, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 64, sourceArmColumn := 46862, finalStart := 992168, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 65, sourceArmColumn := 46867, finalStart := 992373, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 66, sourceArmColumn := 46872, finalStart := 992578, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 67, sourceArmColumn := 46877, finalStart := 992783, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 68, sourceArmColumn := 46882, finalStart := 992988, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 69, sourceArmColumn := 46887, finalStart := 993193, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 70, sourceArmColumn := 46892, finalStart := 993398, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 71, sourceArmColumn := 46897, finalStart := 993603, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 72, sourceArmColumn := 46902, finalStart := 993808, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 73, sourceArmColumn := 46907, finalStart := 994013, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 74, sourceArmColumn := 46912, finalStart := 994218, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 75, sourceArmColumn := 46917, finalStart := 994423, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 76, sourceArmColumn := 46922, finalStart := 994628, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 77, sourceArmColumn := 46927, finalStart := 994833, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 78, sourceArmColumn := 46932, finalStart := 995038, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 79, sourceArmColumn := 46937, finalStart := 995243, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 80, sourceArmColumn := 46942, finalStart := 995448, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 81, sourceArmColumn := 46947, finalStart := 995653, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 82, sourceArmColumn := 46952, finalStart := 995858, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 83, sourceArmColumn := 46957, finalStart := 996063, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 84, sourceArmColumn := 46962, finalStart := 996268, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 85, sourceArmColumn := 46967, finalStart := 996473, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 86, sourceArmColumn := 46972, finalStart := 996678, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 87, sourceArmColumn := 46977, finalStart := 996883, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 88, sourceArmColumn := 46982, finalStart := 997088, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 89, sourceArmColumn := 46987, finalStart := 997293, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 90, sourceArmColumn := 46992, finalStart := 997498, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 91, sourceArmColumn := 46997, finalStart := 997703, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 92, sourceArmColumn := 47002, finalStart := 997908, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 93, sourceArmColumn := 47007, finalStart := 998113, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 94, sourceArmColumn := 47012, finalStart := 998318, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 95, sourceArmColumn := 47017, finalStart := 998523, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 96, sourceArmColumn := 47022, finalStart := 998728, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 97, sourceArmColumn := 47027, finalStart := 998933, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 98, sourceArmColumn := 47032, finalStart := 999138, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 99, sourceArmColumn := 47037, finalStart := 999343, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 100, sourceArmColumn := 47042, finalStart := 999548, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 101, sourceArmColumn := 47047, finalStart := 999753, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 102, sourceArmColumn := 47052, finalStart := 999958, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 103, sourceArmColumn := 47057, finalStart := 1000163, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 104, sourceArmColumn := 47062, finalStart := 1000368, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 105, sourceArmColumn := 47067, finalStart := 1000573, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 106, sourceArmColumn := 47072, finalStart := 1000778, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 107, sourceArmColumn := 47077, finalStart := 1000983, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 108, sourceArmColumn := 46813, finalStart := 990159, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 109, sourceArmColumn := 46818, finalStart := 990364, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 110, sourceArmColumn := 46823, finalStart := 990569, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 111, sourceArmColumn := 46828, finalStart := 990774, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 112, sourceArmColumn := 46833, finalStart := 990979, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 113, sourceArmColumn := 46838, finalStart := 991184, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 114, sourceArmColumn := 46843, finalStart := 991389, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 115, sourceArmColumn := 46848, finalStart := 991594, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 116, sourceArmColumn := 46853, finalStart := 991799, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 117, sourceArmColumn := 46858, finalStart := 992004, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 118, sourceArmColumn := 46863, finalStart := 992209, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 119, sourceArmColumn := 46868, finalStart := 992414, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 120, sourceArmColumn := 46873, finalStart := 992619, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 121, sourceArmColumn := 46878, finalStart := 992824, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 122, sourceArmColumn := 46883, finalStart := 993029, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 123, sourceArmColumn := 46888, finalStart := 993234, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 124, sourceArmColumn := 46893, finalStart := 993439, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 125, sourceArmColumn := 46898, finalStart := 993644, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 126, sourceArmColumn := 46903, finalStart := 993849, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 127, sourceArmColumn := 46908, finalStart := 994054, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 128, sourceArmColumn := 46913, finalStart := 994259, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 129, sourceArmColumn := 46918, finalStart := 994464, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 130, sourceArmColumn := 46923, finalStart := 994669, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 131, sourceArmColumn := 46928, finalStart := 994874, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 132, sourceArmColumn := 46933, finalStart := 995079, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 133, sourceArmColumn := 46938, finalStart := 995284, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 134, sourceArmColumn := 46943, finalStart := 995489, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 135, sourceArmColumn := 46948, finalStart := 995694, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 136, sourceArmColumn := 46953, finalStart := 995899, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 137, sourceArmColumn := 46958, finalStart := 996104, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 138, sourceArmColumn := 46963, finalStart := 996309, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 139, sourceArmColumn := 46968, finalStart := 996514, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 140, sourceArmColumn := 46973, finalStart := 996719, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 141, sourceArmColumn := 46978, finalStart := 996924, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 142, sourceArmColumn := 46983, finalStart := 997129, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 143, sourceArmColumn := 46988, finalStart := 997334, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 144, sourceArmColumn := 46993, finalStart := 997539, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 145, sourceArmColumn := 46998, finalStart := 997744, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 146, sourceArmColumn := 47003, finalStart := 997949, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 147, sourceArmColumn := 47008, finalStart := 998154, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 148, sourceArmColumn := 47013, finalStart := 998359, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 149, sourceArmColumn := 47018, finalStart := 998564, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 150, sourceArmColumn := 47023, finalStart := 998769, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 151, sourceArmColumn := 47028, finalStart := 998974, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 152, sourceArmColumn := 47033, finalStart := 999179, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 153, sourceArmColumn := 47038, finalStart := 999384, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 154, sourceArmColumn := 47043, finalStart := 999589, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 155, sourceArmColumn := 47048, finalStart := 999794, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 156, sourceArmColumn := 47053, finalStart := 999999, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 157, sourceArmColumn := 47058, finalStart := 1000204, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 158, sourceArmColumn := 47063, finalStart := 1000409, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 159, sourceArmColumn := 47068, finalStart := 1000614, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 160, sourceArmColumn := 47073, finalStart := 1000819, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 161, sourceArmColumn := 47078, finalStart := 1001024, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 162, sourceArmColumn := 46814, finalStart := 990200, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 163, sourceArmColumn := 46819, finalStart := 990405, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 164, sourceArmColumn := 46824, finalStart := 990610, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 165, sourceArmColumn := 46829, finalStart := 990815, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 166, sourceArmColumn := 46834, finalStart := 991020, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 167, sourceArmColumn := 46839, finalStart := 991225, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 168, sourceArmColumn := 46844, finalStart := 991430, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 169, sourceArmColumn := 46849, finalStart := 991635, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 170, sourceArmColumn := 46854, finalStart := 991840, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 171, sourceArmColumn := 46859, finalStart := 992045, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 172, sourceArmColumn := 46864, finalStart := 992250, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 173, sourceArmColumn := 46869, finalStart := 992455, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 174, sourceArmColumn := 46874, finalStart := 992660, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 175, sourceArmColumn := 46879, finalStart := 992865, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 176, sourceArmColumn := 46884, finalStart := 993070, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 177, sourceArmColumn := 46889, finalStart := 993275, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 178, sourceArmColumn := 46894, finalStart := 993480, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 179, sourceArmColumn := 46899, finalStart := 993685, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 180, sourceArmColumn := 46904, finalStart := 993890, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 181, sourceArmColumn := 46909, finalStart := 994095, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 182, sourceArmColumn := 46914, finalStart := 994300, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 183, sourceArmColumn := 46919, finalStart := 994505, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 184, sourceArmColumn := 46924, finalStart := 994710, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 185, sourceArmColumn := 46929, finalStart := 994915, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 186, sourceArmColumn := 46934, finalStart := 995120, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 187, sourceArmColumn := 46939, finalStart := 995325, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 188, sourceArmColumn := 46944, finalStart := 995530, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 189, sourceArmColumn := 46949, finalStart := 995735, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 190, sourceArmColumn := 46954, finalStart := 995940, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 191, sourceArmColumn := 46959, finalStart := 996145, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 192, sourceArmColumn := 46964, finalStart := 996350, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 193, sourceArmColumn := 46969, finalStart := 996555, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 194, sourceArmColumn := 46974, finalStart := 996760, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 195, sourceArmColumn := 46979, finalStart := 996965, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 196, sourceArmColumn := 46984, finalStart := 997170, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 197, sourceArmColumn := 46989, finalStart := 997375, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 198, sourceArmColumn := 46994, finalStart := 997580, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 199, sourceArmColumn := 46999, finalStart := 997785, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 200, sourceArmColumn := 47004, finalStart := 997990, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 201, sourceArmColumn := 47009, finalStart := 998195, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 202, sourceArmColumn := 47014, finalStart := 998400, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 203, sourceArmColumn := 47019, finalStart := 998605, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 204, sourceArmColumn := 47024, finalStart := 998810, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 205, sourceArmColumn := 47029, finalStart := 999015, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 206, sourceArmColumn := 47034, finalStart := 999220, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 207, sourceArmColumn := 47039, finalStart := 999425, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 208, sourceArmColumn := 47044, finalStart := 999630, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 209, sourceArmColumn := 47049, finalStart := 999835, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 210, sourceArmColumn := 47054, finalStart := 1000040, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 211, sourceArmColumn := 47059, finalStart := 1000245, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 212, sourceArmColumn := 47064, finalStart := 1000450, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 213, sourceArmColumn := 47069, finalStart := 1000655, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 214, sourceArmColumn := 47074, finalStart := 1000860, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 215, sourceArmColumn := 47079, finalStart := 1001065, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 216, sourceArmColumn := 46815, finalStart := 990241, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 217, sourceArmColumn := 46820, finalStart := 990446, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 218, sourceArmColumn := 46825, finalStart := 990651, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 219, sourceArmColumn := 46830, finalStart := 990856, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 220, sourceArmColumn := 46835, finalStart := 991061, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 221, sourceArmColumn := 46840, finalStart := 991266, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 222, sourceArmColumn := 46845, finalStart := 991471, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 223, sourceArmColumn := 46850, finalStart := 991676, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 224, sourceArmColumn := 46855, finalStart := 991881, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 225, sourceArmColumn := 46860, finalStart := 992086, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 226, sourceArmColumn := 46865, finalStart := 992291, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 227, sourceArmColumn := 46870, finalStart := 992496, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 228, sourceArmColumn := 46875, finalStart := 992701, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 229, sourceArmColumn := 46880, finalStart := 992906, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 230, sourceArmColumn := 46885, finalStart := 993111, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 231, sourceArmColumn := 46890, finalStart := 993316, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 232, sourceArmColumn := 46895, finalStart := 993521, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 233, sourceArmColumn := 46900, finalStart := 993726, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 234, sourceArmColumn := 46905, finalStart := 993931, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 235, sourceArmColumn := 46910, finalStart := 994136, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 236, sourceArmColumn := 46915, finalStart := 994341, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 237, sourceArmColumn := 46920, finalStart := 994546, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 238, sourceArmColumn := 46925, finalStart := 994751, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 239, sourceArmColumn := 46930, finalStart := 994956, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 240, sourceArmColumn := 46935, finalStart := 995161, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 241, sourceArmColumn := 46940, finalStart := 995366, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 242, sourceArmColumn := 46945, finalStart := 995571, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 243, sourceArmColumn := 46950, finalStart := 995776, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 244, sourceArmColumn := 46955, finalStart := 995981, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 245, sourceArmColumn := 46960, finalStart := 996186, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 246, sourceArmColumn := 46965, finalStart := 996391, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 247, sourceArmColumn := 46970, finalStart := 996596, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 248, sourceArmColumn := 46975, finalStart := 996801, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 249, sourceArmColumn := 46980, finalStart := 997006, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 250, sourceArmColumn := 46985, finalStart := 997211, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 251, sourceArmColumn := 46990, finalStart := 997416, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 252, sourceArmColumn := 46995, finalStart := 997621, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 253, sourceArmColumn := 47000, finalStart := 997826, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 254, sourceArmColumn := 47005, finalStart := 998031, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 255, sourceArmColumn := 47010, finalStart := 998236, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 256, sourceArmColumn := 47015, finalStart := 998441, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 257, sourceArmColumn := 47020, finalStart := 998646, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 258, sourceArmColumn := 47025, finalStart := 998851, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 259, sourceArmColumn := 47030, finalStart := 999056, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 260, sourceArmColumn := 47035, finalStart := 999261, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 261, sourceArmColumn := 47040, finalStart := 999466, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 262, sourceArmColumn := 47045, finalStart := 999671, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 263, sourceArmColumn := 47050, finalStart := 999876, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 264, sourceArmColumn := 47055, finalStart := 1000081, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 265, sourceArmColumn := 47060, finalStart := 1000286, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 266, sourceArmColumn := 47065, finalStart := 1000491, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 267, sourceArmColumn := 47070, finalStart := 1000696, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 268, sourceArmColumn := 47075, finalStart := 1000901, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 269, sourceArmColumn := 47080, finalStart := 1001106, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 0, sourceArmColumn := 49083, finalStart := 1067567, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 1, sourceArmColumn := 49088, finalStart := 1067772, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 2, sourceArmColumn := 49093, finalStart := 1067977, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 3, sourceArmColumn := 49098, finalStart := 1068182, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 4, sourceArmColumn := 49103, finalStart := 1068387, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 5, sourceArmColumn := 49108, finalStart := 1068592, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 6, sourceArmColumn := 49113, finalStart := 1068797, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 7, sourceArmColumn := 49118, finalStart := 1069002, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 8, sourceArmColumn := 49123, finalStart := 1069207, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 9, sourceArmColumn := 49128, finalStart := 1069412, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 10, sourceArmColumn := 49133, finalStart := 1069617, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 11, sourceArmColumn := 49138, finalStart := 1069822, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 12, sourceArmColumn := 49143, finalStart := 1070027, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 13, sourceArmColumn := 49148, finalStart := 1070232, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 14, sourceArmColumn := 49153, finalStart := 1070437, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 15, sourceArmColumn := 49158, finalStart := 1070642, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 16, sourceArmColumn := 49163, finalStart := 1070847, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 17, sourceArmColumn := 49168, finalStart := 1071052, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 18, sourceArmColumn := 49173, finalStart := 1071257, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 19, sourceArmColumn := 49178, finalStart := 1071462, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 20, sourceArmColumn := 49183, finalStart := 1071667, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 21, sourceArmColumn := 49188, finalStart := 1071872, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 22, sourceArmColumn := 49193, finalStart := 1072077, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 23, sourceArmColumn := 49198, finalStart := 1072282, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 24, sourceArmColumn := 49203, finalStart := 1072487, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 25, sourceArmColumn := 49208, finalStart := 1072692, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 26, sourceArmColumn := 49213, finalStart := 1072897, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 27, sourceArmColumn := 49218, finalStart := 1073102, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 28, sourceArmColumn := 49223, finalStart := 1073307, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 29, sourceArmColumn := 49228, finalStart := 1073512, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 30, sourceArmColumn := 49233, finalStart := 1073717, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 31, sourceArmColumn := 49238, finalStart := 1073922, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 32, sourceArmColumn := 49243, finalStart := 1074127, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 33, sourceArmColumn := 49248, finalStart := 1074332, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 34, sourceArmColumn := 49253, finalStart := 1074537, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 35, sourceArmColumn := 49258, finalStart := 1074742, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk12
