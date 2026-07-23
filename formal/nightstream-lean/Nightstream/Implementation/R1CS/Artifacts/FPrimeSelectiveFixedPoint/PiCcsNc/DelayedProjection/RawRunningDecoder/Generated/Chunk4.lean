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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk4

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 3, logicalColumn := 198, sourceArmColumn := 28818, finalStart := 377660, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 199, sourceArmColumn := 28823, finalStart := 377865, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 200, sourceArmColumn := 28828, finalStart := 378070, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 201, sourceArmColumn := 28833, finalStart := 378275, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 202, sourceArmColumn := 28838, finalStart := 378480, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 203, sourceArmColumn := 28843, finalStart := 378685, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 204, sourceArmColumn := 28848, finalStart := 378890, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 205, sourceArmColumn := 28853, finalStart := 379095, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 206, sourceArmColumn := 28858, finalStart := 379300, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 207, sourceArmColumn := 28863, finalStart := 379505, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 208, sourceArmColumn := 28868, finalStart := 379710, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 209, sourceArmColumn := 28873, finalStart := 379915, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 210, sourceArmColumn := 28878, finalStart := 380120, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 211, sourceArmColumn := 28883, finalStart := 380325, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 212, sourceArmColumn := 28888, finalStart := 380530, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 213, sourceArmColumn := 28893, finalStart := 380735, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 214, sourceArmColumn := 28898, finalStart := 380940, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 215, sourceArmColumn := 28903, finalStart := 381145, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 216, sourceArmColumn := 28639, finalStart := 370321, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 217, sourceArmColumn := 28644, finalStart := 370526, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 218, sourceArmColumn := 28649, finalStart := 370731, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 219, sourceArmColumn := 28654, finalStart := 370936, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 220, sourceArmColumn := 28659, finalStart := 371141, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 221, sourceArmColumn := 28664, finalStart := 371346, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 222, sourceArmColumn := 28669, finalStart := 371551, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 223, sourceArmColumn := 28674, finalStart := 371756, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 224, sourceArmColumn := 28679, finalStart := 371961, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 225, sourceArmColumn := 28684, finalStart := 372166, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 226, sourceArmColumn := 28689, finalStart := 372371, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 227, sourceArmColumn := 28694, finalStart := 372576, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 228, sourceArmColumn := 28699, finalStart := 372781, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 229, sourceArmColumn := 28704, finalStart := 372986, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 230, sourceArmColumn := 28709, finalStart := 373191, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 231, sourceArmColumn := 28714, finalStart := 373396, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 232, sourceArmColumn := 28719, finalStart := 373601, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 233, sourceArmColumn := 28724, finalStart := 373806, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 234, sourceArmColumn := 28729, finalStart := 374011, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 235, sourceArmColumn := 28734, finalStart := 374216, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 236, sourceArmColumn := 28739, finalStart := 374421, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 237, sourceArmColumn := 28744, finalStart := 374626, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 238, sourceArmColumn := 28749, finalStart := 374831, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 239, sourceArmColumn := 28754, finalStart := 375036, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 240, sourceArmColumn := 28759, finalStart := 375241, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 241, sourceArmColumn := 28764, finalStart := 375446, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 242, sourceArmColumn := 28769, finalStart := 375651, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 243, sourceArmColumn := 28774, finalStart := 375856, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 244, sourceArmColumn := 28779, finalStart := 376061, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 245, sourceArmColumn := 28784, finalStart := 376266, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 246, sourceArmColumn := 28789, finalStart := 376471, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 247, sourceArmColumn := 28794, finalStart := 376676, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 248, sourceArmColumn := 28799, finalStart := 376881, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 249, sourceArmColumn := 28804, finalStart := 377086, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 250, sourceArmColumn := 28809, finalStart := 377291, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 251, sourceArmColumn := 28814, finalStart := 377496, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 252, sourceArmColumn := 28819, finalStart := 377701, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 253, sourceArmColumn := 28824, finalStart := 377906, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 254, sourceArmColumn := 28829, finalStart := 378111, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 255, sourceArmColumn := 28834, finalStart := 378316, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 256, sourceArmColumn := 28839, finalStart := 378521, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 257, sourceArmColumn := 28844, finalStart := 378726, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 258, sourceArmColumn := 28849, finalStart := 378931, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 259, sourceArmColumn := 28854, finalStart := 379136, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 260, sourceArmColumn := 28859, finalStart := 379341, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 261, sourceArmColumn := 28864, finalStart := 379546, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 262, sourceArmColumn := 28869, finalStart := 379751, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 263, sourceArmColumn := 28874, finalStart := 379956, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 264, sourceArmColumn := 28879, finalStart := 380161, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 265, sourceArmColumn := 28884, finalStart := 380366, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 266, sourceArmColumn := 28889, finalStart := 380571, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 267, sourceArmColumn := 28894, finalStart := 380776, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 268, sourceArmColumn := 28899, finalStart := 380981, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 269, sourceArmColumn := 28904, finalStart := 381186, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 0, sourceArmColumn := 30907, finalStart := 447647, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 1, sourceArmColumn := 30912, finalStart := 447852, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 2, sourceArmColumn := 30917, finalStart := 448057, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 3, sourceArmColumn := 30922, finalStart := 448262, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 4, sourceArmColumn := 30927, finalStart := 448467, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 5, sourceArmColumn := 30932, finalStart := 448672, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 6, sourceArmColumn := 30937, finalStart := 448877, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 7, sourceArmColumn := 30942, finalStart := 449082, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 8, sourceArmColumn := 30947, finalStart := 449287, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 9, sourceArmColumn := 30952, finalStart := 449492, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 10, sourceArmColumn := 30957, finalStart := 449697, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 11, sourceArmColumn := 30962, finalStart := 449902, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 12, sourceArmColumn := 30967, finalStart := 450107, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 13, sourceArmColumn := 30972, finalStart := 450312, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 14, sourceArmColumn := 30977, finalStart := 450517, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 15, sourceArmColumn := 30982, finalStart := 450722, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 16, sourceArmColumn := 30987, finalStart := 450927, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 17, sourceArmColumn := 30992, finalStart := 451132, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 18, sourceArmColumn := 30997, finalStart := 451337, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 19, sourceArmColumn := 31002, finalStart := 451542, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 20, sourceArmColumn := 31007, finalStart := 451747, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 21, sourceArmColumn := 31012, finalStart := 451952, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 22, sourceArmColumn := 31017, finalStart := 452157, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 23, sourceArmColumn := 31022, finalStart := 452362, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 24, sourceArmColumn := 31027, finalStart := 452567, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 25, sourceArmColumn := 31032, finalStart := 452772, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 26, sourceArmColumn := 31037, finalStart := 452977, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 27, sourceArmColumn := 31042, finalStart := 453182, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 28, sourceArmColumn := 31047, finalStart := 453387, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 29, sourceArmColumn := 31052, finalStart := 453592, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 30, sourceArmColumn := 31057, finalStart := 453797, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 31, sourceArmColumn := 31062, finalStart := 454002, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 32, sourceArmColumn := 31067, finalStart := 454207, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 33, sourceArmColumn := 31072, finalStart := 454412, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 34, sourceArmColumn := 31077, finalStart := 454617, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 35, sourceArmColumn := 31082, finalStart := 454822, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 36, sourceArmColumn := 31087, finalStart := 455027, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 37, sourceArmColumn := 31092, finalStart := 455232, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 38, sourceArmColumn := 31097, finalStart := 455437, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 39, sourceArmColumn := 31102, finalStart := 455642, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 40, sourceArmColumn := 31107, finalStart := 455847, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 41, sourceArmColumn := 31112, finalStart := 456052, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 42, sourceArmColumn := 31117, finalStart := 456257, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 43, sourceArmColumn := 31122, finalStart := 456462, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 44, sourceArmColumn := 31127, finalStart := 456667, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 45, sourceArmColumn := 31132, finalStart := 456872, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 46, sourceArmColumn := 31137, finalStart := 457077, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 47, sourceArmColumn := 31142, finalStart := 457282, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 48, sourceArmColumn := 31147, finalStart := 457487, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 49, sourceArmColumn := 31152, finalStart := 457692, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 50, sourceArmColumn := 31157, finalStart := 457897, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 51, sourceArmColumn := 31162, finalStart := 458102, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 52, sourceArmColumn := 31167, finalStart := 458307, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 53, sourceArmColumn := 31172, finalStart := 458512, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 54, sourceArmColumn := 30908, finalStart := 447688, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 55, sourceArmColumn := 30913, finalStart := 447893, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 56, sourceArmColumn := 30918, finalStart := 448098, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 57, sourceArmColumn := 30923, finalStart := 448303, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 58, sourceArmColumn := 30928, finalStart := 448508, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 59, sourceArmColumn := 30933, finalStart := 448713, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 60, sourceArmColumn := 30938, finalStart := 448918, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 61, sourceArmColumn := 30943, finalStart := 449123, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 62, sourceArmColumn := 30948, finalStart := 449328, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 63, sourceArmColumn := 30953, finalStart := 449533, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 64, sourceArmColumn := 30958, finalStart := 449738, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 65, sourceArmColumn := 30963, finalStart := 449943, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 66, sourceArmColumn := 30968, finalStart := 450148, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 67, sourceArmColumn := 30973, finalStart := 450353, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 68, sourceArmColumn := 30978, finalStart := 450558, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 69, sourceArmColumn := 30983, finalStart := 450763, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 70, sourceArmColumn := 30988, finalStart := 450968, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 71, sourceArmColumn := 30993, finalStart := 451173, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 72, sourceArmColumn := 30998, finalStart := 451378, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 73, sourceArmColumn := 31003, finalStart := 451583, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 74, sourceArmColumn := 31008, finalStart := 451788, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 75, sourceArmColumn := 31013, finalStart := 451993, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 76, sourceArmColumn := 31018, finalStart := 452198, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 77, sourceArmColumn := 31023, finalStart := 452403, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 78, sourceArmColumn := 31028, finalStart := 452608, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 79, sourceArmColumn := 31033, finalStart := 452813, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 80, sourceArmColumn := 31038, finalStart := 453018, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 81, sourceArmColumn := 31043, finalStart := 453223, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 82, sourceArmColumn := 31048, finalStart := 453428, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 83, sourceArmColumn := 31053, finalStart := 453633, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 84, sourceArmColumn := 31058, finalStart := 453838, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 85, sourceArmColumn := 31063, finalStart := 454043, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 86, sourceArmColumn := 31068, finalStart := 454248, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 87, sourceArmColumn := 31073, finalStart := 454453, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 88, sourceArmColumn := 31078, finalStart := 454658, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 89, sourceArmColumn := 31083, finalStart := 454863, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 90, sourceArmColumn := 31088, finalStart := 455068, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 91, sourceArmColumn := 31093, finalStart := 455273, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 92, sourceArmColumn := 31098, finalStart := 455478, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 93, sourceArmColumn := 31103, finalStart := 455683, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 94, sourceArmColumn := 31108, finalStart := 455888, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 95, sourceArmColumn := 31113, finalStart := 456093, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 96, sourceArmColumn := 31118, finalStart := 456298, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 97, sourceArmColumn := 31123, finalStart := 456503, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 98, sourceArmColumn := 31128, finalStart := 456708, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 99, sourceArmColumn := 31133, finalStart := 456913, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 100, sourceArmColumn := 31138, finalStart := 457118, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 101, sourceArmColumn := 31143, finalStart := 457323, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 102, sourceArmColumn := 31148, finalStart := 457528, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 103, sourceArmColumn := 31153, finalStart := 457733, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 104, sourceArmColumn := 31158, finalStart := 457938, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 105, sourceArmColumn := 31163, finalStart := 458143, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 106, sourceArmColumn := 31168, finalStart := 458348, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 107, sourceArmColumn := 31173, finalStart := 458553, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 108, sourceArmColumn := 30909, finalStart := 447729, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 109, sourceArmColumn := 30914, finalStart := 447934, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 110, sourceArmColumn := 30919, finalStart := 448139, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 111, sourceArmColumn := 30924, finalStart := 448344, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 112, sourceArmColumn := 30929, finalStart := 448549, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 113, sourceArmColumn := 30934, finalStart := 448754, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 114, sourceArmColumn := 30939, finalStart := 448959, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 115, sourceArmColumn := 30944, finalStart := 449164, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 116, sourceArmColumn := 30949, finalStart := 449369, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 117, sourceArmColumn := 30954, finalStart := 449574, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 118, sourceArmColumn := 30959, finalStart := 449779, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 119, sourceArmColumn := 30964, finalStart := 449984, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 120, sourceArmColumn := 30969, finalStart := 450189, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 121, sourceArmColumn := 30974, finalStart := 450394, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 122, sourceArmColumn := 30979, finalStart := 450599, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 123, sourceArmColumn := 30984, finalStart := 450804, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 124, sourceArmColumn := 30989, finalStart := 451009, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 125, sourceArmColumn := 30994, finalStart := 451214, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 126, sourceArmColumn := 30999, finalStart := 451419, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 127, sourceArmColumn := 31004, finalStart := 451624, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 128, sourceArmColumn := 31009, finalStart := 451829, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 129, sourceArmColumn := 31014, finalStart := 452034, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 130, sourceArmColumn := 31019, finalStart := 452239, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 131, sourceArmColumn := 31024, finalStart := 452444, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 132, sourceArmColumn := 31029, finalStart := 452649, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 133, sourceArmColumn := 31034, finalStart := 452854, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 134, sourceArmColumn := 31039, finalStart := 453059, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 135, sourceArmColumn := 31044, finalStart := 453264, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 136, sourceArmColumn := 31049, finalStart := 453469, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 137, sourceArmColumn := 31054, finalStart := 453674, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 138, sourceArmColumn := 31059, finalStart := 453879, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 139, sourceArmColumn := 31064, finalStart := 454084, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 140, sourceArmColumn := 31069, finalStart := 454289, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 141, sourceArmColumn := 31074, finalStart := 454494, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 142, sourceArmColumn := 31079, finalStart := 454699, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 143, sourceArmColumn := 31084, finalStart := 454904, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 144, sourceArmColumn := 31089, finalStart := 455109, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 145, sourceArmColumn := 31094, finalStart := 455314, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 146, sourceArmColumn := 31099, finalStart := 455519, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 147, sourceArmColumn := 31104, finalStart := 455724, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 148, sourceArmColumn := 31109, finalStart := 455929, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 149, sourceArmColumn := 31114, finalStart := 456134, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 150, sourceArmColumn := 31119, finalStart := 456339, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 151, sourceArmColumn := 31124, finalStart := 456544, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 152, sourceArmColumn := 31129, finalStart := 456749, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 153, sourceArmColumn := 31134, finalStart := 456954, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 154, sourceArmColumn := 31139, finalStart := 457159, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 155, sourceArmColumn := 31144, finalStart := 457364, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 156, sourceArmColumn := 31149, finalStart := 457569, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 157, sourceArmColumn := 31154, finalStart := 457774, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 158, sourceArmColumn := 31159, finalStart := 457979, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 159, sourceArmColumn := 31164, finalStart := 458184, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 160, sourceArmColumn := 31169, finalStart := 458389, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 161, sourceArmColumn := 31174, finalStart := 458594, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 162, sourceArmColumn := 30910, finalStart := 447770, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 163, sourceArmColumn := 30915, finalStart := 447975, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 164, sourceArmColumn := 30920, finalStart := 448180, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 165, sourceArmColumn := 30925, finalStart := 448385, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 166, sourceArmColumn := 30930, finalStart := 448590, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 167, sourceArmColumn := 30935, finalStart := 448795, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 168, sourceArmColumn := 30940, finalStart := 449000, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 169, sourceArmColumn := 30945, finalStart := 449205, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 170, sourceArmColumn := 30950, finalStart := 449410, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 171, sourceArmColumn := 30955, finalStart := 449615, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 172, sourceArmColumn := 30960, finalStart := 449820, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 173, sourceArmColumn := 30965, finalStart := 450025, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 174, sourceArmColumn := 30970, finalStart := 450230, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 175, sourceArmColumn := 30975, finalStart := 450435, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 176, sourceArmColumn := 30980, finalStart := 450640, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 177, sourceArmColumn := 30985, finalStart := 450845, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 178, sourceArmColumn := 30990, finalStart := 451050, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 179, sourceArmColumn := 30995, finalStart := 451255, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk4
