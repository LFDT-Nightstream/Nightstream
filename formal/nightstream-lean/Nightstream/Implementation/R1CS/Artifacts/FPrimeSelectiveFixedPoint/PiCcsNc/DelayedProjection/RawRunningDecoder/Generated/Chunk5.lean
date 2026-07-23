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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk5

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 4, logicalColumn := 180, sourceArmColumn := 31000, finalStart := 451460, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 181, sourceArmColumn := 31005, finalStart := 451665, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 182, sourceArmColumn := 31010, finalStart := 451870, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 183, sourceArmColumn := 31015, finalStart := 452075, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 184, sourceArmColumn := 31020, finalStart := 452280, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 185, sourceArmColumn := 31025, finalStart := 452485, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 186, sourceArmColumn := 31030, finalStart := 452690, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 187, sourceArmColumn := 31035, finalStart := 452895, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 188, sourceArmColumn := 31040, finalStart := 453100, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 189, sourceArmColumn := 31045, finalStart := 453305, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 190, sourceArmColumn := 31050, finalStart := 453510, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 191, sourceArmColumn := 31055, finalStart := 453715, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 192, sourceArmColumn := 31060, finalStart := 453920, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 193, sourceArmColumn := 31065, finalStart := 454125, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 194, sourceArmColumn := 31070, finalStart := 454330, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 195, sourceArmColumn := 31075, finalStart := 454535, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 196, sourceArmColumn := 31080, finalStart := 454740, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 197, sourceArmColumn := 31085, finalStart := 454945, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 198, sourceArmColumn := 31090, finalStart := 455150, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 199, sourceArmColumn := 31095, finalStart := 455355, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 200, sourceArmColumn := 31100, finalStart := 455560, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 201, sourceArmColumn := 31105, finalStart := 455765, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 202, sourceArmColumn := 31110, finalStart := 455970, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 203, sourceArmColumn := 31115, finalStart := 456175, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 204, sourceArmColumn := 31120, finalStart := 456380, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 205, sourceArmColumn := 31125, finalStart := 456585, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 206, sourceArmColumn := 31130, finalStart := 456790, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 207, sourceArmColumn := 31135, finalStart := 456995, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 208, sourceArmColumn := 31140, finalStart := 457200, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 209, sourceArmColumn := 31145, finalStart := 457405, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 210, sourceArmColumn := 31150, finalStart := 457610, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 211, sourceArmColumn := 31155, finalStart := 457815, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 212, sourceArmColumn := 31160, finalStart := 458020, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 213, sourceArmColumn := 31165, finalStart := 458225, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 214, sourceArmColumn := 31170, finalStart := 458430, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 215, sourceArmColumn := 31175, finalStart := 458635, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 216, sourceArmColumn := 30911, finalStart := 447811, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 217, sourceArmColumn := 30916, finalStart := 448016, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 218, sourceArmColumn := 30921, finalStart := 448221, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 219, sourceArmColumn := 30926, finalStart := 448426, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 220, sourceArmColumn := 30931, finalStart := 448631, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 221, sourceArmColumn := 30936, finalStart := 448836, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 222, sourceArmColumn := 30941, finalStart := 449041, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 223, sourceArmColumn := 30946, finalStart := 449246, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 224, sourceArmColumn := 30951, finalStart := 449451, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 225, sourceArmColumn := 30956, finalStart := 449656, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 226, sourceArmColumn := 30961, finalStart := 449861, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 227, sourceArmColumn := 30966, finalStart := 450066, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 228, sourceArmColumn := 30971, finalStart := 450271, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 229, sourceArmColumn := 30976, finalStart := 450476, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 230, sourceArmColumn := 30981, finalStart := 450681, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 231, sourceArmColumn := 30986, finalStart := 450886, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 232, sourceArmColumn := 30991, finalStart := 451091, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 233, sourceArmColumn := 30996, finalStart := 451296, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 234, sourceArmColumn := 31001, finalStart := 451501, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 235, sourceArmColumn := 31006, finalStart := 451706, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 236, sourceArmColumn := 31011, finalStart := 451911, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 237, sourceArmColumn := 31016, finalStart := 452116, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 238, sourceArmColumn := 31021, finalStart := 452321, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 239, sourceArmColumn := 31026, finalStart := 452526, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 240, sourceArmColumn := 31031, finalStart := 452731, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 241, sourceArmColumn := 31036, finalStart := 452936, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 242, sourceArmColumn := 31041, finalStart := 453141, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 243, sourceArmColumn := 31046, finalStart := 453346, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 244, sourceArmColumn := 31051, finalStart := 453551, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 245, sourceArmColumn := 31056, finalStart := 453756, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 246, sourceArmColumn := 31061, finalStart := 453961, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 247, sourceArmColumn := 31066, finalStart := 454166, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 248, sourceArmColumn := 31071, finalStart := 454371, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 249, sourceArmColumn := 31076, finalStart := 454576, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 250, sourceArmColumn := 31081, finalStart := 454781, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 251, sourceArmColumn := 31086, finalStart := 454986, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 252, sourceArmColumn := 31091, finalStart := 455191, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 253, sourceArmColumn := 31096, finalStart := 455396, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 254, sourceArmColumn := 31101, finalStart := 455601, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 255, sourceArmColumn := 31106, finalStart := 455806, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 256, sourceArmColumn := 31111, finalStart := 456011, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 257, sourceArmColumn := 31116, finalStart := 456216, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 258, sourceArmColumn := 31121, finalStart := 456421, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 259, sourceArmColumn := 31126, finalStart := 456626, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 260, sourceArmColumn := 31131, finalStart := 456831, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 261, sourceArmColumn := 31136, finalStart := 457036, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 262, sourceArmColumn := 31141, finalStart := 457241, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 263, sourceArmColumn := 31146, finalStart := 457446, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 264, sourceArmColumn := 31151, finalStart := 457651, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 265, sourceArmColumn := 31156, finalStart := 457856, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 266, sourceArmColumn := 31161, finalStart := 458061, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 267, sourceArmColumn := 31166, finalStart := 458266, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 268, sourceArmColumn := 31171, finalStart := 458471, width := 41, encoding := .balancedTernary }
, { child := 4, logicalColumn := 269, sourceArmColumn := 31176, finalStart := 458676, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 0, sourceArmColumn := 33179, finalStart := 525137, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 1, sourceArmColumn := 33184, finalStart := 525342, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 2, sourceArmColumn := 33189, finalStart := 525547, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 3, sourceArmColumn := 33194, finalStart := 525752, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 4, sourceArmColumn := 33199, finalStart := 525957, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 5, sourceArmColumn := 33204, finalStart := 526162, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 6, sourceArmColumn := 33209, finalStart := 526367, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 7, sourceArmColumn := 33214, finalStart := 526572, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 8, sourceArmColumn := 33219, finalStart := 526777, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 9, sourceArmColumn := 33224, finalStart := 526982, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 10, sourceArmColumn := 33229, finalStart := 527187, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 11, sourceArmColumn := 33234, finalStart := 527392, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 12, sourceArmColumn := 33239, finalStart := 527597, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 13, sourceArmColumn := 33244, finalStart := 527802, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 14, sourceArmColumn := 33249, finalStart := 528007, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 15, sourceArmColumn := 33254, finalStart := 528212, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 16, sourceArmColumn := 33259, finalStart := 528417, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 17, sourceArmColumn := 33264, finalStart := 528622, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 18, sourceArmColumn := 33269, finalStart := 528827, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 19, sourceArmColumn := 33274, finalStart := 529032, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 20, sourceArmColumn := 33279, finalStart := 529237, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 21, sourceArmColumn := 33284, finalStart := 529442, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 22, sourceArmColumn := 33289, finalStart := 529647, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 23, sourceArmColumn := 33294, finalStart := 529852, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 24, sourceArmColumn := 33299, finalStart := 530057, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 25, sourceArmColumn := 33304, finalStart := 530262, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 26, sourceArmColumn := 33309, finalStart := 530467, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 27, sourceArmColumn := 33314, finalStart := 530672, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 28, sourceArmColumn := 33319, finalStart := 530877, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 29, sourceArmColumn := 33324, finalStart := 531082, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 30, sourceArmColumn := 33329, finalStart := 531287, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 31, sourceArmColumn := 33334, finalStart := 531492, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 32, sourceArmColumn := 33339, finalStart := 531697, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 33, sourceArmColumn := 33344, finalStart := 531902, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 34, sourceArmColumn := 33349, finalStart := 532107, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 35, sourceArmColumn := 33354, finalStart := 532312, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 36, sourceArmColumn := 33359, finalStart := 532517, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 37, sourceArmColumn := 33364, finalStart := 532722, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 38, sourceArmColumn := 33369, finalStart := 532927, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 39, sourceArmColumn := 33374, finalStart := 533132, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 40, sourceArmColumn := 33379, finalStart := 533337, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 41, sourceArmColumn := 33384, finalStart := 533542, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 42, sourceArmColumn := 33389, finalStart := 533747, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 43, sourceArmColumn := 33394, finalStart := 533952, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 44, sourceArmColumn := 33399, finalStart := 534157, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 45, sourceArmColumn := 33404, finalStart := 534362, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 46, sourceArmColumn := 33409, finalStart := 534567, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 47, sourceArmColumn := 33414, finalStart := 534772, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 48, sourceArmColumn := 33419, finalStart := 534977, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 49, sourceArmColumn := 33424, finalStart := 535182, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 50, sourceArmColumn := 33429, finalStart := 535387, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 51, sourceArmColumn := 33434, finalStart := 535592, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 52, sourceArmColumn := 33439, finalStart := 535797, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 53, sourceArmColumn := 33444, finalStart := 536002, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 54, sourceArmColumn := 33180, finalStart := 525178, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 55, sourceArmColumn := 33185, finalStart := 525383, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 56, sourceArmColumn := 33190, finalStart := 525588, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 57, sourceArmColumn := 33195, finalStart := 525793, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 58, sourceArmColumn := 33200, finalStart := 525998, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 59, sourceArmColumn := 33205, finalStart := 526203, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 60, sourceArmColumn := 33210, finalStart := 526408, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 61, sourceArmColumn := 33215, finalStart := 526613, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 62, sourceArmColumn := 33220, finalStart := 526818, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 63, sourceArmColumn := 33225, finalStart := 527023, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 64, sourceArmColumn := 33230, finalStart := 527228, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 65, sourceArmColumn := 33235, finalStart := 527433, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 66, sourceArmColumn := 33240, finalStart := 527638, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 67, sourceArmColumn := 33245, finalStart := 527843, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 68, sourceArmColumn := 33250, finalStart := 528048, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 69, sourceArmColumn := 33255, finalStart := 528253, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 70, sourceArmColumn := 33260, finalStart := 528458, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 71, sourceArmColumn := 33265, finalStart := 528663, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 72, sourceArmColumn := 33270, finalStart := 528868, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 73, sourceArmColumn := 33275, finalStart := 529073, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 74, sourceArmColumn := 33280, finalStart := 529278, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 75, sourceArmColumn := 33285, finalStart := 529483, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 76, sourceArmColumn := 33290, finalStart := 529688, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 77, sourceArmColumn := 33295, finalStart := 529893, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 78, sourceArmColumn := 33300, finalStart := 530098, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 79, sourceArmColumn := 33305, finalStart := 530303, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 80, sourceArmColumn := 33310, finalStart := 530508, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 81, sourceArmColumn := 33315, finalStart := 530713, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 82, sourceArmColumn := 33320, finalStart := 530918, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 83, sourceArmColumn := 33325, finalStart := 531123, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 84, sourceArmColumn := 33330, finalStart := 531328, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 85, sourceArmColumn := 33335, finalStart := 531533, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 86, sourceArmColumn := 33340, finalStart := 531738, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 87, sourceArmColumn := 33345, finalStart := 531943, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 88, sourceArmColumn := 33350, finalStart := 532148, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 89, sourceArmColumn := 33355, finalStart := 532353, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 90, sourceArmColumn := 33360, finalStart := 532558, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 91, sourceArmColumn := 33365, finalStart := 532763, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 92, sourceArmColumn := 33370, finalStart := 532968, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 93, sourceArmColumn := 33375, finalStart := 533173, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 94, sourceArmColumn := 33380, finalStart := 533378, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 95, sourceArmColumn := 33385, finalStart := 533583, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 96, sourceArmColumn := 33390, finalStart := 533788, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 97, sourceArmColumn := 33395, finalStart := 533993, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 98, sourceArmColumn := 33400, finalStart := 534198, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 99, sourceArmColumn := 33405, finalStart := 534403, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 100, sourceArmColumn := 33410, finalStart := 534608, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 101, sourceArmColumn := 33415, finalStart := 534813, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 102, sourceArmColumn := 33420, finalStart := 535018, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 103, sourceArmColumn := 33425, finalStart := 535223, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 104, sourceArmColumn := 33430, finalStart := 535428, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 105, sourceArmColumn := 33435, finalStart := 535633, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 106, sourceArmColumn := 33440, finalStart := 535838, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 107, sourceArmColumn := 33445, finalStart := 536043, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 108, sourceArmColumn := 33181, finalStart := 525219, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 109, sourceArmColumn := 33186, finalStart := 525424, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 110, sourceArmColumn := 33191, finalStart := 525629, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 111, sourceArmColumn := 33196, finalStart := 525834, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 112, sourceArmColumn := 33201, finalStart := 526039, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 113, sourceArmColumn := 33206, finalStart := 526244, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 114, sourceArmColumn := 33211, finalStart := 526449, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 115, sourceArmColumn := 33216, finalStart := 526654, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 116, sourceArmColumn := 33221, finalStart := 526859, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 117, sourceArmColumn := 33226, finalStart := 527064, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 118, sourceArmColumn := 33231, finalStart := 527269, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 119, sourceArmColumn := 33236, finalStart := 527474, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 120, sourceArmColumn := 33241, finalStart := 527679, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 121, sourceArmColumn := 33246, finalStart := 527884, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 122, sourceArmColumn := 33251, finalStart := 528089, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 123, sourceArmColumn := 33256, finalStart := 528294, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 124, sourceArmColumn := 33261, finalStart := 528499, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 125, sourceArmColumn := 33266, finalStart := 528704, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 126, sourceArmColumn := 33271, finalStart := 528909, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 127, sourceArmColumn := 33276, finalStart := 529114, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 128, sourceArmColumn := 33281, finalStart := 529319, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 129, sourceArmColumn := 33286, finalStart := 529524, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 130, sourceArmColumn := 33291, finalStart := 529729, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 131, sourceArmColumn := 33296, finalStart := 529934, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 132, sourceArmColumn := 33301, finalStart := 530139, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 133, sourceArmColumn := 33306, finalStart := 530344, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 134, sourceArmColumn := 33311, finalStart := 530549, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 135, sourceArmColumn := 33316, finalStart := 530754, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 136, sourceArmColumn := 33321, finalStart := 530959, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 137, sourceArmColumn := 33326, finalStart := 531164, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 138, sourceArmColumn := 33331, finalStart := 531369, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 139, sourceArmColumn := 33336, finalStart := 531574, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 140, sourceArmColumn := 33341, finalStart := 531779, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 141, sourceArmColumn := 33346, finalStart := 531984, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 142, sourceArmColumn := 33351, finalStart := 532189, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 143, sourceArmColumn := 33356, finalStart := 532394, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 144, sourceArmColumn := 33361, finalStart := 532599, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 145, sourceArmColumn := 33366, finalStart := 532804, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 146, sourceArmColumn := 33371, finalStart := 533009, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 147, sourceArmColumn := 33376, finalStart := 533214, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 148, sourceArmColumn := 33381, finalStart := 533419, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 149, sourceArmColumn := 33386, finalStart := 533624, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 150, sourceArmColumn := 33391, finalStart := 533829, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 151, sourceArmColumn := 33396, finalStart := 534034, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 152, sourceArmColumn := 33401, finalStart := 534239, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 153, sourceArmColumn := 33406, finalStart := 534444, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 154, sourceArmColumn := 33411, finalStart := 534649, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 155, sourceArmColumn := 33416, finalStart := 534854, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 156, sourceArmColumn := 33421, finalStart := 535059, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 157, sourceArmColumn := 33426, finalStart := 535264, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 158, sourceArmColumn := 33431, finalStart := 535469, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 159, sourceArmColumn := 33436, finalStart := 535674, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 160, sourceArmColumn := 33441, finalStart := 535879, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 161, sourceArmColumn := 33446, finalStart := 536084, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk5
