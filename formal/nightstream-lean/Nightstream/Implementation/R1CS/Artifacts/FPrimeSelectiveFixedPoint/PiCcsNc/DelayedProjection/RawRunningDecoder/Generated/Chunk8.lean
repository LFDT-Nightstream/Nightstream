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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk8

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 7, logicalColumn := 126, sourceArmColumn := 37815, finalStart := 683889, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 127, sourceArmColumn := 37820, finalStart := 684094, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 128, sourceArmColumn := 37825, finalStart := 684299, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 129, sourceArmColumn := 37830, finalStart := 684504, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 130, sourceArmColumn := 37835, finalStart := 684709, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 131, sourceArmColumn := 37840, finalStart := 684914, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 132, sourceArmColumn := 37845, finalStart := 685119, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 133, sourceArmColumn := 37850, finalStart := 685324, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 134, sourceArmColumn := 37855, finalStart := 685529, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 135, sourceArmColumn := 37860, finalStart := 685734, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 136, sourceArmColumn := 37865, finalStart := 685939, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 137, sourceArmColumn := 37870, finalStart := 686144, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 138, sourceArmColumn := 37875, finalStart := 686349, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 139, sourceArmColumn := 37880, finalStart := 686554, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 140, sourceArmColumn := 37885, finalStart := 686759, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 141, sourceArmColumn := 37890, finalStart := 686964, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 142, sourceArmColumn := 37895, finalStart := 687169, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 143, sourceArmColumn := 37900, finalStart := 687374, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 144, sourceArmColumn := 37905, finalStart := 687579, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 145, sourceArmColumn := 37910, finalStart := 687784, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 146, sourceArmColumn := 37915, finalStart := 687989, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 147, sourceArmColumn := 37920, finalStart := 688194, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 148, sourceArmColumn := 37925, finalStart := 688399, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 149, sourceArmColumn := 37930, finalStart := 688604, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 150, sourceArmColumn := 37935, finalStart := 688809, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 151, sourceArmColumn := 37940, finalStart := 689014, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 152, sourceArmColumn := 37945, finalStart := 689219, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 153, sourceArmColumn := 37950, finalStart := 689424, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 154, sourceArmColumn := 37955, finalStart := 689629, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 155, sourceArmColumn := 37960, finalStart := 689834, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 156, sourceArmColumn := 37965, finalStart := 690039, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 157, sourceArmColumn := 37970, finalStart := 690244, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 158, sourceArmColumn := 37975, finalStart := 690449, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 159, sourceArmColumn := 37980, finalStart := 690654, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 160, sourceArmColumn := 37985, finalStart := 690859, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 161, sourceArmColumn := 37990, finalStart := 691064, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 162, sourceArmColumn := 37726, finalStart := 680240, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 163, sourceArmColumn := 37731, finalStart := 680445, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 164, sourceArmColumn := 37736, finalStart := 680650, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 165, sourceArmColumn := 37741, finalStart := 680855, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 166, sourceArmColumn := 37746, finalStart := 681060, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 167, sourceArmColumn := 37751, finalStart := 681265, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 168, sourceArmColumn := 37756, finalStart := 681470, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 169, sourceArmColumn := 37761, finalStart := 681675, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 170, sourceArmColumn := 37766, finalStart := 681880, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 171, sourceArmColumn := 37771, finalStart := 682085, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 172, sourceArmColumn := 37776, finalStart := 682290, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 173, sourceArmColumn := 37781, finalStart := 682495, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 174, sourceArmColumn := 37786, finalStart := 682700, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 175, sourceArmColumn := 37791, finalStart := 682905, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 176, sourceArmColumn := 37796, finalStart := 683110, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 177, sourceArmColumn := 37801, finalStart := 683315, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 178, sourceArmColumn := 37806, finalStart := 683520, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 179, sourceArmColumn := 37811, finalStart := 683725, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 180, sourceArmColumn := 37816, finalStart := 683930, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 181, sourceArmColumn := 37821, finalStart := 684135, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 182, sourceArmColumn := 37826, finalStart := 684340, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 183, sourceArmColumn := 37831, finalStart := 684545, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 184, sourceArmColumn := 37836, finalStart := 684750, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 185, sourceArmColumn := 37841, finalStart := 684955, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 186, sourceArmColumn := 37846, finalStart := 685160, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 187, sourceArmColumn := 37851, finalStart := 685365, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 188, sourceArmColumn := 37856, finalStart := 685570, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 189, sourceArmColumn := 37861, finalStart := 685775, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 190, sourceArmColumn := 37866, finalStart := 685980, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 191, sourceArmColumn := 37871, finalStart := 686185, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 192, sourceArmColumn := 37876, finalStart := 686390, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 193, sourceArmColumn := 37881, finalStart := 686595, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 194, sourceArmColumn := 37886, finalStart := 686800, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 195, sourceArmColumn := 37891, finalStart := 687005, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 196, sourceArmColumn := 37896, finalStart := 687210, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 197, sourceArmColumn := 37901, finalStart := 687415, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 198, sourceArmColumn := 37906, finalStart := 687620, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 199, sourceArmColumn := 37911, finalStart := 687825, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 200, sourceArmColumn := 37916, finalStart := 688030, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 201, sourceArmColumn := 37921, finalStart := 688235, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 202, sourceArmColumn := 37926, finalStart := 688440, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 203, sourceArmColumn := 37931, finalStart := 688645, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 204, sourceArmColumn := 37936, finalStart := 688850, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 205, sourceArmColumn := 37941, finalStart := 689055, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 206, sourceArmColumn := 37946, finalStart := 689260, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 207, sourceArmColumn := 37951, finalStart := 689465, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 208, sourceArmColumn := 37956, finalStart := 689670, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 209, sourceArmColumn := 37961, finalStart := 689875, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 210, sourceArmColumn := 37966, finalStart := 690080, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 211, sourceArmColumn := 37971, finalStart := 690285, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 212, sourceArmColumn := 37976, finalStart := 690490, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 213, sourceArmColumn := 37981, finalStart := 690695, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 214, sourceArmColumn := 37986, finalStart := 690900, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 215, sourceArmColumn := 37991, finalStart := 691105, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 216, sourceArmColumn := 37727, finalStart := 680281, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 217, sourceArmColumn := 37732, finalStart := 680486, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 218, sourceArmColumn := 37737, finalStart := 680691, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 219, sourceArmColumn := 37742, finalStart := 680896, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 220, sourceArmColumn := 37747, finalStart := 681101, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 221, sourceArmColumn := 37752, finalStart := 681306, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 222, sourceArmColumn := 37757, finalStart := 681511, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 223, sourceArmColumn := 37762, finalStart := 681716, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 224, sourceArmColumn := 37767, finalStart := 681921, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 225, sourceArmColumn := 37772, finalStart := 682126, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 226, sourceArmColumn := 37777, finalStart := 682331, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 227, sourceArmColumn := 37782, finalStart := 682536, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 228, sourceArmColumn := 37787, finalStart := 682741, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 229, sourceArmColumn := 37792, finalStart := 682946, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 230, sourceArmColumn := 37797, finalStart := 683151, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 231, sourceArmColumn := 37802, finalStart := 683356, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 232, sourceArmColumn := 37807, finalStart := 683561, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 233, sourceArmColumn := 37812, finalStart := 683766, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 234, sourceArmColumn := 37817, finalStart := 683971, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 235, sourceArmColumn := 37822, finalStart := 684176, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 236, sourceArmColumn := 37827, finalStart := 684381, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 237, sourceArmColumn := 37832, finalStart := 684586, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 238, sourceArmColumn := 37837, finalStart := 684791, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 239, sourceArmColumn := 37842, finalStart := 684996, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 240, sourceArmColumn := 37847, finalStart := 685201, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 241, sourceArmColumn := 37852, finalStart := 685406, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 242, sourceArmColumn := 37857, finalStart := 685611, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 243, sourceArmColumn := 37862, finalStart := 685816, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 244, sourceArmColumn := 37867, finalStart := 686021, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 245, sourceArmColumn := 37872, finalStart := 686226, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 246, sourceArmColumn := 37877, finalStart := 686431, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 247, sourceArmColumn := 37882, finalStart := 686636, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 248, sourceArmColumn := 37887, finalStart := 686841, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 249, sourceArmColumn := 37892, finalStart := 687046, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 250, sourceArmColumn := 37897, finalStart := 687251, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 251, sourceArmColumn := 37902, finalStart := 687456, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 252, sourceArmColumn := 37907, finalStart := 687661, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 253, sourceArmColumn := 37912, finalStart := 687866, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 254, sourceArmColumn := 37917, finalStart := 688071, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 255, sourceArmColumn := 37922, finalStart := 688276, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 256, sourceArmColumn := 37927, finalStart := 688481, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 257, sourceArmColumn := 37932, finalStart := 688686, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 258, sourceArmColumn := 37937, finalStart := 688891, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 259, sourceArmColumn := 37942, finalStart := 689096, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 260, sourceArmColumn := 37947, finalStart := 689301, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 261, sourceArmColumn := 37952, finalStart := 689506, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 262, sourceArmColumn := 37957, finalStart := 689711, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 263, sourceArmColumn := 37962, finalStart := 689916, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 264, sourceArmColumn := 37967, finalStart := 690121, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 265, sourceArmColumn := 37972, finalStart := 690326, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 266, sourceArmColumn := 37977, finalStart := 690531, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 267, sourceArmColumn := 37982, finalStart := 690736, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 268, sourceArmColumn := 37987, finalStart := 690941, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 269, sourceArmColumn := 37992, finalStart := 691146, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 0, sourceArmColumn := 39995, finalStart := 757607, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 1, sourceArmColumn := 40000, finalStart := 757812, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 2, sourceArmColumn := 40005, finalStart := 758017, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 3, sourceArmColumn := 40010, finalStart := 758222, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 4, sourceArmColumn := 40015, finalStart := 758427, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 5, sourceArmColumn := 40020, finalStart := 758632, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 6, sourceArmColumn := 40025, finalStart := 758837, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 7, sourceArmColumn := 40030, finalStart := 759042, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 8, sourceArmColumn := 40035, finalStart := 759247, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 9, sourceArmColumn := 40040, finalStart := 759452, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 10, sourceArmColumn := 40045, finalStart := 759657, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 11, sourceArmColumn := 40050, finalStart := 759862, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 12, sourceArmColumn := 40055, finalStart := 760067, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 13, sourceArmColumn := 40060, finalStart := 760272, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 14, sourceArmColumn := 40065, finalStart := 760477, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 15, sourceArmColumn := 40070, finalStart := 760682, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 16, sourceArmColumn := 40075, finalStart := 760887, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 17, sourceArmColumn := 40080, finalStart := 761092, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 18, sourceArmColumn := 40085, finalStart := 761297, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 19, sourceArmColumn := 40090, finalStart := 761502, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 20, sourceArmColumn := 40095, finalStart := 761707, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 21, sourceArmColumn := 40100, finalStart := 761912, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 22, sourceArmColumn := 40105, finalStart := 762117, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 23, sourceArmColumn := 40110, finalStart := 762322, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 24, sourceArmColumn := 40115, finalStart := 762527, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 25, sourceArmColumn := 40120, finalStart := 762732, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 26, sourceArmColumn := 40125, finalStart := 762937, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 27, sourceArmColumn := 40130, finalStart := 763142, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 28, sourceArmColumn := 40135, finalStart := 763347, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 29, sourceArmColumn := 40140, finalStart := 763552, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 30, sourceArmColumn := 40145, finalStart := 763757, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 31, sourceArmColumn := 40150, finalStart := 763962, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 32, sourceArmColumn := 40155, finalStart := 764167, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 33, sourceArmColumn := 40160, finalStart := 764372, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 34, sourceArmColumn := 40165, finalStart := 764577, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 35, sourceArmColumn := 40170, finalStart := 764782, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 36, sourceArmColumn := 40175, finalStart := 764987, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 37, sourceArmColumn := 40180, finalStart := 765192, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 38, sourceArmColumn := 40185, finalStart := 765397, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 39, sourceArmColumn := 40190, finalStart := 765602, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 40, sourceArmColumn := 40195, finalStart := 765807, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 41, sourceArmColumn := 40200, finalStart := 766012, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 42, sourceArmColumn := 40205, finalStart := 766217, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 43, sourceArmColumn := 40210, finalStart := 766422, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 44, sourceArmColumn := 40215, finalStart := 766627, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 45, sourceArmColumn := 40220, finalStart := 766832, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 46, sourceArmColumn := 40225, finalStart := 767037, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 47, sourceArmColumn := 40230, finalStart := 767242, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 48, sourceArmColumn := 40235, finalStart := 767447, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 49, sourceArmColumn := 40240, finalStart := 767652, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 50, sourceArmColumn := 40245, finalStart := 767857, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 51, sourceArmColumn := 40250, finalStart := 768062, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 52, sourceArmColumn := 40255, finalStart := 768267, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 53, sourceArmColumn := 40260, finalStart := 768472, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 54, sourceArmColumn := 39996, finalStart := 757648, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 55, sourceArmColumn := 40001, finalStart := 757853, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 56, sourceArmColumn := 40006, finalStart := 758058, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 57, sourceArmColumn := 40011, finalStart := 758263, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 58, sourceArmColumn := 40016, finalStart := 758468, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 59, sourceArmColumn := 40021, finalStart := 758673, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 60, sourceArmColumn := 40026, finalStart := 758878, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 61, sourceArmColumn := 40031, finalStart := 759083, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 62, sourceArmColumn := 40036, finalStart := 759288, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 63, sourceArmColumn := 40041, finalStart := 759493, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 64, sourceArmColumn := 40046, finalStart := 759698, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 65, sourceArmColumn := 40051, finalStart := 759903, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 66, sourceArmColumn := 40056, finalStart := 760108, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 67, sourceArmColumn := 40061, finalStart := 760313, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 68, sourceArmColumn := 40066, finalStart := 760518, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 69, sourceArmColumn := 40071, finalStart := 760723, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 70, sourceArmColumn := 40076, finalStart := 760928, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 71, sourceArmColumn := 40081, finalStart := 761133, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 72, sourceArmColumn := 40086, finalStart := 761338, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 73, sourceArmColumn := 40091, finalStart := 761543, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 74, sourceArmColumn := 40096, finalStart := 761748, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 75, sourceArmColumn := 40101, finalStart := 761953, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 76, sourceArmColumn := 40106, finalStart := 762158, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 77, sourceArmColumn := 40111, finalStart := 762363, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 78, sourceArmColumn := 40116, finalStart := 762568, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 79, sourceArmColumn := 40121, finalStart := 762773, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 80, sourceArmColumn := 40126, finalStart := 762978, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 81, sourceArmColumn := 40131, finalStart := 763183, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 82, sourceArmColumn := 40136, finalStart := 763388, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 83, sourceArmColumn := 40141, finalStart := 763593, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 84, sourceArmColumn := 40146, finalStart := 763798, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 85, sourceArmColumn := 40151, finalStart := 764003, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 86, sourceArmColumn := 40156, finalStart := 764208, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 87, sourceArmColumn := 40161, finalStart := 764413, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 88, sourceArmColumn := 40166, finalStart := 764618, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 89, sourceArmColumn := 40171, finalStart := 764823, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 90, sourceArmColumn := 40176, finalStart := 765028, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 91, sourceArmColumn := 40181, finalStart := 765233, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 92, sourceArmColumn := 40186, finalStart := 765438, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 93, sourceArmColumn := 40191, finalStart := 765643, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 94, sourceArmColumn := 40196, finalStart := 765848, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 95, sourceArmColumn := 40201, finalStart := 766053, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 96, sourceArmColumn := 40206, finalStart := 766258, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 97, sourceArmColumn := 40211, finalStart := 766463, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 98, sourceArmColumn := 40216, finalStart := 766668, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 99, sourceArmColumn := 40221, finalStart := 766873, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 100, sourceArmColumn := 40226, finalStart := 767078, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 101, sourceArmColumn := 40231, finalStart := 767283, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 102, sourceArmColumn := 40236, finalStart := 767488, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 103, sourceArmColumn := 40241, finalStart := 767693, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 104, sourceArmColumn := 40246, finalStart := 767898, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 105, sourceArmColumn := 40251, finalStart := 768103, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 106, sourceArmColumn := 40256, finalStart := 768308, width := 41, encoding := .balancedTernary }
, { child := 8, logicalColumn := 107, sourceArmColumn := 40261, finalStart := 768513, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk8
