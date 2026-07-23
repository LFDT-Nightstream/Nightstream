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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk3

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 2, logicalColumn := 216, sourceArmColumn := 26367, finalStart := 292831, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 217, sourceArmColumn := 26372, finalStart := 293036, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 218, sourceArmColumn := 26377, finalStart := 293241, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 219, sourceArmColumn := 26382, finalStart := 293446, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 220, sourceArmColumn := 26387, finalStart := 293651, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 221, sourceArmColumn := 26392, finalStart := 293856, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 222, sourceArmColumn := 26397, finalStart := 294061, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 223, sourceArmColumn := 26402, finalStart := 294266, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 224, sourceArmColumn := 26407, finalStart := 294471, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 225, sourceArmColumn := 26412, finalStart := 294676, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 226, sourceArmColumn := 26417, finalStart := 294881, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 227, sourceArmColumn := 26422, finalStart := 295086, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 228, sourceArmColumn := 26427, finalStart := 295291, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 229, sourceArmColumn := 26432, finalStart := 295496, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 230, sourceArmColumn := 26437, finalStart := 295701, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 231, sourceArmColumn := 26442, finalStart := 295906, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 232, sourceArmColumn := 26447, finalStart := 296111, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 233, sourceArmColumn := 26452, finalStart := 296316, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 234, sourceArmColumn := 26457, finalStart := 296521, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 235, sourceArmColumn := 26462, finalStart := 296726, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 236, sourceArmColumn := 26467, finalStart := 296931, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 237, sourceArmColumn := 26472, finalStart := 297136, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 238, sourceArmColumn := 26477, finalStart := 297341, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 239, sourceArmColumn := 26482, finalStart := 297546, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 240, sourceArmColumn := 26487, finalStart := 297751, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 241, sourceArmColumn := 26492, finalStart := 297956, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 242, sourceArmColumn := 26497, finalStart := 298161, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 243, sourceArmColumn := 26502, finalStart := 298366, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 244, sourceArmColumn := 26507, finalStart := 298571, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 245, sourceArmColumn := 26512, finalStart := 298776, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 246, sourceArmColumn := 26517, finalStart := 298981, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 247, sourceArmColumn := 26522, finalStart := 299186, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 248, sourceArmColumn := 26527, finalStart := 299391, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 249, sourceArmColumn := 26532, finalStart := 299596, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 250, sourceArmColumn := 26537, finalStart := 299801, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 251, sourceArmColumn := 26542, finalStart := 300006, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 252, sourceArmColumn := 26547, finalStart := 300211, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 253, sourceArmColumn := 26552, finalStart := 300416, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 254, sourceArmColumn := 26557, finalStart := 300621, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 255, sourceArmColumn := 26562, finalStart := 300826, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 256, sourceArmColumn := 26567, finalStart := 301031, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 257, sourceArmColumn := 26572, finalStart := 301236, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 258, sourceArmColumn := 26577, finalStart := 301441, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 259, sourceArmColumn := 26582, finalStart := 301646, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 260, sourceArmColumn := 26587, finalStart := 301851, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 261, sourceArmColumn := 26592, finalStart := 302056, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 262, sourceArmColumn := 26597, finalStart := 302261, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 263, sourceArmColumn := 26602, finalStart := 302466, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 264, sourceArmColumn := 26607, finalStart := 302671, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 265, sourceArmColumn := 26612, finalStart := 302876, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 266, sourceArmColumn := 26617, finalStart := 303081, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 267, sourceArmColumn := 26622, finalStart := 303286, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 268, sourceArmColumn := 26627, finalStart := 303491, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 269, sourceArmColumn := 26632, finalStart := 303696, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 0, sourceArmColumn := 28635, finalStart := 370157, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 1, sourceArmColumn := 28640, finalStart := 370362, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 2, sourceArmColumn := 28645, finalStart := 370567, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 3, sourceArmColumn := 28650, finalStart := 370772, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 4, sourceArmColumn := 28655, finalStart := 370977, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 5, sourceArmColumn := 28660, finalStart := 371182, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 6, sourceArmColumn := 28665, finalStart := 371387, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 7, sourceArmColumn := 28670, finalStart := 371592, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 8, sourceArmColumn := 28675, finalStart := 371797, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 9, sourceArmColumn := 28680, finalStart := 372002, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 10, sourceArmColumn := 28685, finalStart := 372207, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 11, sourceArmColumn := 28690, finalStart := 372412, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 12, sourceArmColumn := 28695, finalStart := 372617, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 13, sourceArmColumn := 28700, finalStart := 372822, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 14, sourceArmColumn := 28705, finalStart := 373027, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 15, sourceArmColumn := 28710, finalStart := 373232, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 16, sourceArmColumn := 28715, finalStart := 373437, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 17, sourceArmColumn := 28720, finalStart := 373642, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 18, sourceArmColumn := 28725, finalStart := 373847, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 19, sourceArmColumn := 28730, finalStart := 374052, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 20, sourceArmColumn := 28735, finalStart := 374257, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 21, sourceArmColumn := 28740, finalStart := 374462, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 22, sourceArmColumn := 28745, finalStart := 374667, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 23, sourceArmColumn := 28750, finalStart := 374872, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 24, sourceArmColumn := 28755, finalStart := 375077, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 25, sourceArmColumn := 28760, finalStart := 375282, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 26, sourceArmColumn := 28765, finalStart := 375487, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 27, sourceArmColumn := 28770, finalStart := 375692, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 28, sourceArmColumn := 28775, finalStart := 375897, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 29, sourceArmColumn := 28780, finalStart := 376102, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 30, sourceArmColumn := 28785, finalStart := 376307, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 31, sourceArmColumn := 28790, finalStart := 376512, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 32, sourceArmColumn := 28795, finalStart := 376717, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 33, sourceArmColumn := 28800, finalStart := 376922, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 34, sourceArmColumn := 28805, finalStart := 377127, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 35, sourceArmColumn := 28810, finalStart := 377332, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 36, sourceArmColumn := 28815, finalStart := 377537, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 37, sourceArmColumn := 28820, finalStart := 377742, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 38, sourceArmColumn := 28825, finalStart := 377947, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 39, sourceArmColumn := 28830, finalStart := 378152, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 40, sourceArmColumn := 28835, finalStart := 378357, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 41, sourceArmColumn := 28840, finalStart := 378562, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 42, sourceArmColumn := 28845, finalStart := 378767, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 43, sourceArmColumn := 28850, finalStart := 378972, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 44, sourceArmColumn := 28855, finalStart := 379177, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 45, sourceArmColumn := 28860, finalStart := 379382, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 46, sourceArmColumn := 28865, finalStart := 379587, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 47, sourceArmColumn := 28870, finalStart := 379792, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 48, sourceArmColumn := 28875, finalStart := 379997, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 49, sourceArmColumn := 28880, finalStart := 380202, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 50, sourceArmColumn := 28885, finalStart := 380407, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 51, sourceArmColumn := 28890, finalStart := 380612, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 52, sourceArmColumn := 28895, finalStart := 380817, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 53, sourceArmColumn := 28900, finalStart := 381022, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 54, sourceArmColumn := 28636, finalStart := 370198, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 55, sourceArmColumn := 28641, finalStart := 370403, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 56, sourceArmColumn := 28646, finalStart := 370608, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 57, sourceArmColumn := 28651, finalStart := 370813, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 58, sourceArmColumn := 28656, finalStart := 371018, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 59, sourceArmColumn := 28661, finalStart := 371223, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 60, sourceArmColumn := 28666, finalStart := 371428, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 61, sourceArmColumn := 28671, finalStart := 371633, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 62, sourceArmColumn := 28676, finalStart := 371838, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 63, sourceArmColumn := 28681, finalStart := 372043, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 64, sourceArmColumn := 28686, finalStart := 372248, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 65, sourceArmColumn := 28691, finalStart := 372453, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 66, sourceArmColumn := 28696, finalStart := 372658, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 67, sourceArmColumn := 28701, finalStart := 372863, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 68, sourceArmColumn := 28706, finalStart := 373068, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 69, sourceArmColumn := 28711, finalStart := 373273, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 70, sourceArmColumn := 28716, finalStart := 373478, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 71, sourceArmColumn := 28721, finalStart := 373683, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 72, sourceArmColumn := 28726, finalStart := 373888, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 73, sourceArmColumn := 28731, finalStart := 374093, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 74, sourceArmColumn := 28736, finalStart := 374298, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 75, sourceArmColumn := 28741, finalStart := 374503, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 76, sourceArmColumn := 28746, finalStart := 374708, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 77, sourceArmColumn := 28751, finalStart := 374913, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 78, sourceArmColumn := 28756, finalStart := 375118, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 79, sourceArmColumn := 28761, finalStart := 375323, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 80, sourceArmColumn := 28766, finalStart := 375528, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 81, sourceArmColumn := 28771, finalStart := 375733, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 82, sourceArmColumn := 28776, finalStart := 375938, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 83, sourceArmColumn := 28781, finalStart := 376143, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 84, sourceArmColumn := 28786, finalStart := 376348, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 85, sourceArmColumn := 28791, finalStart := 376553, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 86, sourceArmColumn := 28796, finalStart := 376758, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 87, sourceArmColumn := 28801, finalStart := 376963, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 88, sourceArmColumn := 28806, finalStart := 377168, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 89, sourceArmColumn := 28811, finalStart := 377373, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 90, sourceArmColumn := 28816, finalStart := 377578, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 91, sourceArmColumn := 28821, finalStart := 377783, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 92, sourceArmColumn := 28826, finalStart := 377988, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 93, sourceArmColumn := 28831, finalStart := 378193, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 94, sourceArmColumn := 28836, finalStart := 378398, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 95, sourceArmColumn := 28841, finalStart := 378603, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 96, sourceArmColumn := 28846, finalStart := 378808, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 97, sourceArmColumn := 28851, finalStart := 379013, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 98, sourceArmColumn := 28856, finalStart := 379218, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 99, sourceArmColumn := 28861, finalStart := 379423, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 100, sourceArmColumn := 28866, finalStart := 379628, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 101, sourceArmColumn := 28871, finalStart := 379833, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 102, sourceArmColumn := 28876, finalStart := 380038, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 103, sourceArmColumn := 28881, finalStart := 380243, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 104, sourceArmColumn := 28886, finalStart := 380448, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 105, sourceArmColumn := 28891, finalStart := 380653, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 106, sourceArmColumn := 28896, finalStart := 380858, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 107, sourceArmColumn := 28901, finalStart := 381063, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 108, sourceArmColumn := 28637, finalStart := 370239, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 109, sourceArmColumn := 28642, finalStart := 370444, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 110, sourceArmColumn := 28647, finalStart := 370649, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 111, sourceArmColumn := 28652, finalStart := 370854, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 112, sourceArmColumn := 28657, finalStart := 371059, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 113, sourceArmColumn := 28662, finalStart := 371264, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 114, sourceArmColumn := 28667, finalStart := 371469, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 115, sourceArmColumn := 28672, finalStart := 371674, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 116, sourceArmColumn := 28677, finalStart := 371879, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 117, sourceArmColumn := 28682, finalStart := 372084, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 118, sourceArmColumn := 28687, finalStart := 372289, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 119, sourceArmColumn := 28692, finalStart := 372494, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 120, sourceArmColumn := 28697, finalStart := 372699, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 121, sourceArmColumn := 28702, finalStart := 372904, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 122, sourceArmColumn := 28707, finalStart := 373109, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 123, sourceArmColumn := 28712, finalStart := 373314, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 124, sourceArmColumn := 28717, finalStart := 373519, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 125, sourceArmColumn := 28722, finalStart := 373724, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 126, sourceArmColumn := 28727, finalStart := 373929, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 127, sourceArmColumn := 28732, finalStart := 374134, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 128, sourceArmColumn := 28737, finalStart := 374339, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 129, sourceArmColumn := 28742, finalStart := 374544, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 130, sourceArmColumn := 28747, finalStart := 374749, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 131, sourceArmColumn := 28752, finalStart := 374954, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 132, sourceArmColumn := 28757, finalStart := 375159, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 133, sourceArmColumn := 28762, finalStart := 375364, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 134, sourceArmColumn := 28767, finalStart := 375569, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 135, sourceArmColumn := 28772, finalStart := 375774, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 136, sourceArmColumn := 28777, finalStart := 375979, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 137, sourceArmColumn := 28782, finalStart := 376184, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 138, sourceArmColumn := 28787, finalStart := 376389, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 139, sourceArmColumn := 28792, finalStart := 376594, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 140, sourceArmColumn := 28797, finalStart := 376799, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 141, sourceArmColumn := 28802, finalStart := 377004, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 142, sourceArmColumn := 28807, finalStart := 377209, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 143, sourceArmColumn := 28812, finalStart := 377414, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 144, sourceArmColumn := 28817, finalStart := 377619, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 145, sourceArmColumn := 28822, finalStart := 377824, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 146, sourceArmColumn := 28827, finalStart := 378029, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 147, sourceArmColumn := 28832, finalStart := 378234, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 148, sourceArmColumn := 28837, finalStart := 378439, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 149, sourceArmColumn := 28842, finalStart := 378644, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 150, sourceArmColumn := 28847, finalStart := 378849, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 151, sourceArmColumn := 28852, finalStart := 379054, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 152, sourceArmColumn := 28857, finalStart := 379259, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 153, sourceArmColumn := 28862, finalStart := 379464, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 154, sourceArmColumn := 28867, finalStart := 379669, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 155, sourceArmColumn := 28872, finalStart := 379874, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 156, sourceArmColumn := 28877, finalStart := 380079, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 157, sourceArmColumn := 28882, finalStart := 380284, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 158, sourceArmColumn := 28887, finalStart := 380489, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 159, sourceArmColumn := 28892, finalStart := 380694, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 160, sourceArmColumn := 28897, finalStart := 380899, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 161, sourceArmColumn := 28902, finalStart := 381104, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 162, sourceArmColumn := 28638, finalStart := 370280, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 163, sourceArmColumn := 28643, finalStart := 370485, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 164, sourceArmColumn := 28648, finalStart := 370690, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 165, sourceArmColumn := 28653, finalStart := 370895, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 166, sourceArmColumn := 28658, finalStart := 371100, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 167, sourceArmColumn := 28663, finalStart := 371305, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 168, sourceArmColumn := 28668, finalStart := 371510, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 169, sourceArmColumn := 28673, finalStart := 371715, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 170, sourceArmColumn := 28678, finalStart := 371920, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 171, sourceArmColumn := 28683, finalStart := 372125, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 172, sourceArmColumn := 28688, finalStart := 372330, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 173, sourceArmColumn := 28693, finalStart := 372535, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 174, sourceArmColumn := 28698, finalStart := 372740, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 175, sourceArmColumn := 28703, finalStart := 372945, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 176, sourceArmColumn := 28708, finalStart := 373150, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 177, sourceArmColumn := 28713, finalStart := 373355, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 178, sourceArmColumn := 28718, finalStart := 373560, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 179, sourceArmColumn := 28723, finalStart := 373765, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 180, sourceArmColumn := 28728, finalStart := 373970, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 181, sourceArmColumn := 28733, finalStart := 374175, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 182, sourceArmColumn := 28738, finalStart := 374380, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 183, sourceArmColumn := 28743, finalStart := 374585, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 184, sourceArmColumn := 28748, finalStart := 374790, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 185, sourceArmColumn := 28753, finalStart := 374995, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 186, sourceArmColumn := 28758, finalStart := 375200, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 187, sourceArmColumn := 28763, finalStart := 375405, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 188, sourceArmColumn := 28768, finalStart := 375610, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 189, sourceArmColumn := 28773, finalStart := 375815, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 190, sourceArmColumn := 28778, finalStart := 376020, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 191, sourceArmColumn := 28783, finalStart := 376225, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 192, sourceArmColumn := 28788, finalStart := 376430, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 193, sourceArmColumn := 28793, finalStart := 376635, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 194, sourceArmColumn := 28798, finalStart := 376840, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 195, sourceArmColumn := 28803, finalStart := 377045, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 196, sourceArmColumn := 28808, finalStart := 377250, width := 41, encoding := .balancedTernary }
, { child := 3, logicalColumn := 197, sourceArmColumn := 28813, finalStart := 377455, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk3
