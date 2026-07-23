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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk11

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 10, logicalColumn := 72, sourceArmColumn := 44630, finalStart := 916318, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 73, sourceArmColumn := 44635, finalStart := 916523, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 74, sourceArmColumn := 44640, finalStart := 916728, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 75, sourceArmColumn := 44645, finalStart := 916933, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 76, sourceArmColumn := 44650, finalStart := 917138, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 77, sourceArmColumn := 44655, finalStart := 917343, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 78, sourceArmColumn := 44660, finalStart := 917548, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 79, sourceArmColumn := 44665, finalStart := 917753, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 80, sourceArmColumn := 44670, finalStart := 917958, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 81, sourceArmColumn := 44675, finalStart := 918163, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 82, sourceArmColumn := 44680, finalStart := 918368, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 83, sourceArmColumn := 44685, finalStart := 918573, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 84, sourceArmColumn := 44690, finalStart := 918778, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 85, sourceArmColumn := 44695, finalStart := 918983, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 86, sourceArmColumn := 44700, finalStart := 919188, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 87, sourceArmColumn := 44705, finalStart := 919393, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 88, sourceArmColumn := 44710, finalStart := 919598, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 89, sourceArmColumn := 44715, finalStart := 919803, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 90, sourceArmColumn := 44720, finalStart := 920008, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 91, sourceArmColumn := 44725, finalStart := 920213, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 92, sourceArmColumn := 44730, finalStart := 920418, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 93, sourceArmColumn := 44735, finalStart := 920623, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 94, sourceArmColumn := 44740, finalStart := 920828, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 95, sourceArmColumn := 44745, finalStart := 921033, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 96, sourceArmColumn := 44750, finalStart := 921238, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 97, sourceArmColumn := 44755, finalStart := 921443, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 98, sourceArmColumn := 44760, finalStart := 921648, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 99, sourceArmColumn := 44765, finalStart := 921853, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 100, sourceArmColumn := 44770, finalStart := 922058, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 101, sourceArmColumn := 44775, finalStart := 922263, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 102, sourceArmColumn := 44780, finalStart := 922468, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 103, sourceArmColumn := 44785, finalStart := 922673, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 104, sourceArmColumn := 44790, finalStart := 922878, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 105, sourceArmColumn := 44795, finalStart := 923083, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 106, sourceArmColumn := 44800, finalStart := 923288, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 107, sourceArmColumn := 44805, finalStart := 923493, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 108, sourceArmColumn := 44541, finalStart := 912669, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 109, sourceArmColumn := 44546, finalStart := 912874, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 110, sourceArmColumn := 44551, finalStart := 913079, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 111, sourceArmColumn := 44556, finalStart := 913284, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 112, sourceArmColumn := 44561, finalStart := 913489, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 113, sourceArmColumn := 44566, finalStart := 913694, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 114, sourceArmColumn := 44571, finalStart := 913899, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 115, sourceArmColumn := 44576, finalStart := 914104, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 116, sourceArmColumn := 44581, finalStart := 914309, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 117, sourceArmColumn := 44586, finalStart := 914514, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 118, sourceArmColumn := 44591, finalStart := 914719, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 119, sourceArmColumn := 44596, finalStart := 914924, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 120, sourceArmColumn := 44601, finalStart := 915129, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 121, sourceArmColumn := 44606, finalStart := 915334, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 122, sourceArmColumn := 44611, finalStart := 915539, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 123, sourceArmColumn := 44616, finalStart := 915744, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 124, sourceArmColumn := 44621, finalStart := 915949, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 125, sourceArmColumn := 44626, finalStart := 916154, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 126, sourceArmColumn := 44631, finalStart := 916359, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 127, sourceArmColumn := 44636, finalStart := 916564, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 128, sourceArmColumn := 44641, finalStart := 916769, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 129, sourceArmColumn := 44646, finalStart := 916974, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 130, sourceArmColumn := 44651, finalStart := 917179, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 131, sourceArmColumn := 44656, finalStart := 917384, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 132, sourceArmColumn := 44661, finalStart := 917589, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 133, sourceArmColumn := 44666, finalStart := 917794, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 134, sourceArmColumn := 44671, finalStart := 917999, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 135, sourceArmColumn := 44676, finalStart := 918204, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 136, sourceArmColumn := 44681, finalStart := 918409, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 137, sourceArmColumn := 44686, finalStart := 918614, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 138, sourceArmColumn := 44691, finalStart := 918819, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 139, sourceArmColumn := 44696, finalStart := 919024, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 140, sourceArmColumn := 44701, finalStart := 919229, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 141, sourceArmColumn := 44706, finalStart := 919434, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 142, sourceArmColumn := 44711, finalStart := 919639, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 143, sourceArmColumn := 44716, finalStart := 919844, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 144, sourceArmColumn := 44721, finalStart := 920049, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 145, sourceArmColumn := 44726, finalStart := 920254, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 146, sourceArmColumn := 44731, finalStart := 920459, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 147, sourceArmColumn := 44736, finalStart := 920664, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 148, sourceArmColumn := 44741, finalStart := 920869, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 149, sourceArmColumn := 44746, finalStart := 921074, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 150, sourceArmColumn := 44751, finalStart := 921279, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 151, sourceArmColumn := 44756, finalStart := 921484, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 152, sourceArmColumn := 44761, finalStart := 921689, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 153, sourceArmColumn := 44766, finalStart := 921894, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 154, sourceArmColumn := 44771, finalStart := 922099, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 155, sourceArmColumn := 44776, finalStart := 922304, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 156, sourceArmColumn := 44781, finalStart := 922509, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 157, sourceArmColumn := 44786, finalStart := 922714, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 158, sourceArmColumn := 44791, finalStart := 922919, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 159, sourceArmColumn := 44796, finalStart := 923124, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 160, sourceArmColumn := 44801, finalStart := 923329, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 161, sourceArmColumn := 44806, finalStart := 923534, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 162, sourceArmColumn := 44542, finalStart := 912710, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 163, sourceArmColumn := 44547, finalStart := 912915, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 164, sourceArmColumn := 44552, finalStart := 913120, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 165, sourceArmColumn := 44557, finalStart := 913325, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 166, sourceArmColumn := 44562, finalStart := 913530, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 167, sourceArmColumn := 44567, finalStart := 913735, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 168, sourceArmColumn := 44572, finalStart := 913940, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 169, sourceArmColumn := 44577, finalStart := 914145, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 170, sourceArmColumn := 44582, finalStart := 914350, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 171, sourceArmColumn := 44587, finalStart := 914555, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 172, sourceArmColumn := 44592, finalStart := 914760, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 173, sourceArmColumn := 44597, finalStart := 914965, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 174, sourceArmColumn := 44602, finalStart := 915170, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 175, sourceArmColumn := 44607, finalStart := 915375, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 176, sourceArmColumn := 44612, finalStart := 915580, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 177, sourceArmColumn := 44617, finalStart := 915785, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 178, sourceArmColumn := 44622, finalStart := 915990, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 179, sourceArmColumn := 44627, finalStart := 916195, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 180, sourceArmColumn := 44632, finalStart := 916400, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 181, sourceArmColumn := 44637, finalStart := 916605, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 182, sourceArmColumn := 44642, finalStart := 916810, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 183, sourceArmColumn := 44647, finalStart := 917015, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 184, sourceArmColumn := 44652, finalStart := 917220, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 185, sourceArmColumn := 44657, finalStart := 917425, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 186, sourceArmColumn := 44662, finalStart := 917630, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 187, sourceArmColumn := 44667, finalStart := 917835, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 188, sourceArmColumn := 44672, finalStart := 918040, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 189, sourceArmColumn := 44677, finalStart := 918245, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 190, sourceArmColumn := 44682, finalStart := 918450, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 191, sourceArmColumn := 44687, finalStart := 918655, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 192, sourceArmColumn := 44692, finalStart := 918860, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 193, sourceArmColumn := 44697, finalStart := 919065, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 194, sourceArmColumn := 44702, finalStart := 919270, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 195, sourceArmColumn := 44707, finalStart := 919475, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 196, sourceArmColumn := 44712, finalStart := 919680, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 197, sourceArmColumn := 44717, finalStart := 919885, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 198, sourceArmColumn := 44722, finalStart := 920090, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 199, sourceArmColumn := 44727, finalStart := 920295, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 200, sourceArmColumn := 44732, finalStart := 920500, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 201, sourceArmColumn := 44737, finalStart := 920705, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 202, sourceArmColumn := 44742, finalStart := 920910, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 203, sourceArmColumn := 44747, finalStart := 921115, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 204, sourceArmColumn := 44752, finalStart := 921320, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 205, sourceArmColumn := 44757, finalStart := 921525, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 206, sourceArmColumn := 44762, finalStart := 921730, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 207, sourceArmColumn := 44767, finalStart := 921935, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 208, sourceArmColumn := 44772, finalStart := 922140, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 209, sourceArmColumn := 44777, finalStart := 922345, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 210, sourceArmColumn := 44782, finalStart := 922550, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 211, sourceArmColumn := 44787, finalStart := 922755, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 212, sourceArmColumn := 44792, finalStart := 922960, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 213, sourceArmColumn := 44797, finalStart := 923165, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 214, sourceArmColumn := 44802, finalStart := 923370, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 215, sourceArmColumn := 44807, finalStart := 923575, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 216, sourceArmColumn := 44543, finalStart := 912751, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 217, sourceArmColumn := 44548, finalStart := 912956, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 218, sourceArmColumn := 44553, finalStart := 913161, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 219, sourceArmColumn := 44558, finalStart := 913366, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 220, sourceArmColumn := 44563, finalStart := 913571, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 221, sourceArmColumn := 44568, finalStart := 913776, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 222, sourceArmColumn := 44573, finalStart := 913981, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 223, sourceArmColumn := 44578, finalStart := 914186, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 224, sourceArmColumn := 44583, finalStart := 914391, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 225, sourceArmColumn := 44588, finalStart := 914596, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 226, sourceArmColumn := 44593, finalStart := 914801, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 227, sourceArmColumn := 44598, finalStart := 915006, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 228, sourceArmColumn := 44603, finalStart := 915211, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 229, sourceArmColumn := 44608, finalStart := 915416, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 230, sourceArmColumn := 44613, finalStart := 915621, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 231, sourceArmColumn := 44618, finalStart := 915826, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 232, sourceArmColumn := 44623, finalStart := 916031, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 233, sourceArmColumn := 44628, finalStart := 916236, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 234, sourceArmColumn := 44633, finalStart := 916441, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 235, sourceArmColumn := 44638, finalStart := 916646, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 236, sourceArmColumn := 44643, finalStart := 916851, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 237, sourceArmColumn := 44648, finalStart := 917056, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 238, sourceArmColumn := 44653, finalStart := 917261, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 239, sourceArmColumn := 44658, finalStart := 917466, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 240, sourceArmColumn := 44663, finalStart := 917671, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 241, sourceArmColumn := 44668, finalStart := 917876, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 242, sourceArmColumn := 44673, finalStart := 918081, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 243, sourceArmColumn := 44678, finalStart := 918286, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 244, sourceArmColumn := 44683, finalStart := 918491, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 245, sourceArmColumn := 44688, finalStart := 918696, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 246, sourceArmColumn := 44693, finalStart := 918901, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 247, sourceArmColumn := 44698, finalStart := 919106, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 248, sourceArmColumn := 44703, finalStart := 919311, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 249, sourceArmColumn := 44708, finalStart := 919516, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 250, sourceArmColumn := 44713, finalStart := 919721, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 251, sourceArmColumn := 44718, finalStart := 919926, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 252, sourceArmColumn := 44723, finalStart := 920131, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 253, sourceArmColumn := 44728, finalStart := 920336, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 254, sourceArmColumn := 44733, finalStart := 920541, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 255, sourceArmColumn := 44738, finalStart := 920746, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 256, sourceArmColumn := 44743, finalStart := 920951, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 257, sourceArmColumn := 44748, finalStart := 921156, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 258, sourceArmColumn := 44753, finalStart := 921361, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 259, sourceArmColumn := 44758, finalStart := 921566, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 260, sourceArmColumn := 44763, finalStart := 921771, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 261, sourceArmColumn := 44768, finalStart := 921976, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 262, sourceArmColumn := 44773, finalStart := 922181, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 263, sourceArmColumn := 44778, finalStart := 922386, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 264, sourceArmColumn := 44783, finalStart := 922591, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 265, sourceArmColumn := 44788, finalStart := 922796, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 266, sourceArmColumn := 44793, finalStart := 923001, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 267, sourceArmColumn := 44798, finalStart := 923206, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 268, sourceArmColumn := 44803, finalStart := 923411, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 269, sourceArmColumn := 44808, finalStart := 923616, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 0, sourceArmColumn := 46811, finalStart := 990077, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 1, sourceArmColumn := 46816, finalStart := 990282, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 2, sourceArmColumn := 46821, finalStart := 990487, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 3, sourceArmColumn := 46826, finalStart := 990692, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 4, sourceArmColumn := 46831, finalStart := 990897, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 5, sourceArmColumn := 46836, finalStart := 991102, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 6, sourceArmColumn := 46841, finalStart := 991307, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 7, sourceArmColumn := 46846, finalStart := 991512, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 8, sourceArmColumn := 46851, finalStart := 991717, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 9, sourceArmColumn := 46856, finalStart := 991922, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 10, sourceArmColumn := 46861, finalStart := 992127, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 11, sourceArmColumn := 46866, finalStart := 992332, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 12, sourceArmColumn := 46871, finalStart := 992537, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 13, sourceArmColumn := 46876, finalStart := 992742, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 14, sourceArmColumn := 46881, finalStart := 992947, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 15, sourceArmColumn := 46886, finalStart := 993152, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 16, sourceArmColumn := 46891, finalStart := 993357, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 17, sourceArmColumn := 46896, finalStart := 993562, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 18, sourceArmColumn := 46901, finalStart := 993767, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 19, sourceArmColumn := 46906, finalStart := 993972, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 20, sourceArmColumn := 46911, finalStart := 994177, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 21, sourceArmColumn := 46916, finalStart := 994382, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 22, sourceArmColumn := 46921, finalStart := 994587, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 23, sourceArmColumn := 46926, finalStart := 994792, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 24, sourceArmColumn := 46931, finalStart := 994997, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 25, sourceArmColumn := 46936, finalStart := 995202, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 26, sourceArmColumn := 46941, finalStart := 995407, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 27, sourceArmColumn := 46946, finalStart := 995612, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 28, sourceArmColumn := 46951, finalStart := 995817, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 29, sourceArmColumn := 46956, finalStart := 996022, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 30, sourceArmColumn := 46961, finalStart := 996227, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 31, sourceArmColumn := 46966, finalStart := 996432, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 32, sourceArmColumn := 46971, finalStart := 996637, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 33, sourceArmColumn := 46976, finalStart := 996842, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 34, sourceArmColumn := 46981, finalStart := 997047, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 35, sourceArmColumn := 46986, finalStart := 997252, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 36, sourceArmColumn := 46991, finalStart := 997457, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 37, sourceArmColumn := 46996, finalStart := 997662, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 38, sourceArmColumn := 47001, finalStart := 997867, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 39, sourceArmColumn := 47006, finalStart := 998072, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 40, sourceArmColumn := 47011, finalStart := 998277, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 41, sourceArmColumn := 47016, finalStart := 998482, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 42, sourceArmColumn := 47021, finalStart := 998687, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 43, sourceArmColumn := 47026, finalStart := 998892, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 44, sourceArmColumn := 47031, finalStart := 999097, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 45, sourceArmColumn := 47036, finalStart := 999302, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 46, sourceArmColumn := 47041, finalStart := 999507, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 47, sourceArmColumn := 47046, finalStart := 999712, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 48, sourceArmColumn := 47051, finalStart := 999917, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 49, sourceArmColumn := 47056, finalStart := 1000122, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 50, sourceArmColumn := 47061, finalStart := 1000327, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 51, sourceArmColumn := 47066, finalStart := 1000532, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 52, sourceArmColumn := 47071, finalStart := 1000737, width := 41, encoding := .balancedTernary }
, { child := 11, logicalColumn := 53, sourceArmColumn := 47076, finalStart := 1000942, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk11
