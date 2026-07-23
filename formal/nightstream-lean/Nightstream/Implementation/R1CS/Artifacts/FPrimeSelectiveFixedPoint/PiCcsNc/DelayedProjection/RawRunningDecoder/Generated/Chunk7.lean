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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk7

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 6, logicalColumn := 144, sourceArmColumn := 35633, finalStart := 610089, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 145, sourceArmColumn := 35638, finalStart := 610294, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 146, sourceArmColumn := 35643, finalStart := 610499, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 147, sourceArmColumn := 35648, finalStart := 610704, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 148, sourceArmColumn := 35653, finalStart := 610909, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 149, sourceArmColumn := 35658, finalStart := 611114, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 150, sourceArmColumn := 35663, finalStart := 611319, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 151, sourceArmColumn := 35668, finalStart := 611524, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 152, sourceArmColumn := 35673, finalStart := 611729, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 153, sourceArmColumn := 35678, finalStart := 611934, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 154, sourceArmColumn := 35683, finalStart := 612139, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 155, sourceArmColumn := 35688, finalStart := 612344, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 156, sourceArmColumn := 35693, finalStart := 612549, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 157, sourceArmColumn := 35698, finalStart := 612754, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 158, sourceArmColumn := 35703, finalStart := 612959, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 159, sourceArmColumn := 35708, finalStart := 613164, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 160, sourceArmColumn := 35713, finalStart := 613369, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 161, sourceArmColumn := 35718, finalStart := 613574, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 162, sourceArmColumn := 35454, finalStart := 602750, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 163, sourceArmColumn := 35459, finalStart := 602955, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 164, sourceArmColumn := 35464, finalStart := 603160, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 165, sourceArmColumn := 35469, finalStart := 603365, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 166, sourceArmColumn := 35474, finalStart := 603570, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 167, sourceArmColumn := 35479, finalStart := 603775, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 168, sourceArmColumn := 35484, finalStart := 603980, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 169, sourceArmColumn := 35489, finalStart := 604185, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 170, sourceArmColumn := 35494, finalStart := 604390, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 171, sourceArmColumn := 35499, finalStart := 604595, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 172, sourceArmColumn := 35504, finalStart := 604800, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 173, sourceArmColumn := 35509, finalStart := 605005, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 174, sourceArmColumn := 35514, finalStart := 605210, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 175, sourceArmColumn := 35519, finalStart := 605415, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 176, sourceArmColumn := 35524, finalStart := 605620, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 177, sourceArmColumn := 35529, finalStart := 605825, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 178, sourceArmColumn := 35534, finalStart := 606030, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 179, sourceArmColumn := 35539, finalStart := 606235, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 180, sourceArmColumn := 35544, finalStart := 606440, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 181, sourceArmColumn := 35549, finalStart := 606645, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 182, sourceArmColumn := 35554, finalStart := 606850, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 183, sourceArmColumn := 35559, finalStart := 607055, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 184, sourceArmColumn := 35564, finalStart := 607260, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 185, sourceArmColumn := 35569, finalStart := 607465, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 186, sourceArmColumn := 35574, finalStart := 607670, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 187, sourceArmColumn := 35579, finalStart := 607875, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 188, sourceArmColumn := 35584, finalStart := 608080, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 189, sourceArmColumn := 35589, finalStart := 608285, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 190, sourceArmColumn := 35594, finalStart := 608490, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 191, sourceArmColumn := 35599, finalStart := 608695, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 192, sourceArmColumn := 35604, finalStart := 608900, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 193, sourceArmColumn := 35609, finalStart := 609105, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 194, sourceArmColumn := 35614, finalStart := 609310, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 195, sourceArmColumn := 35619, finalStart := 609515, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 196, sourceArmColumn := 35624, finalStart := 609720, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 197, sourceArmColumn := 35629, finalStart := 609925, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 198, sourceArmColumn := 35634, finalStart := 610130, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 199, sourceArmColumn := 35639, finalStart := 610335, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 200, sourceArmColumn := 35644, finalStart := 610540, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 201, sourceArmColumn := 35649, finalStart := 610745, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 202, sourceArmColumn := 35654, finalStart := 610950, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 203, sourceArmColumn := 35659, finalStart := 611155, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 204, sourceArmColumn := 35664, finalStart := 611360, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 205, sourceArmColumn := 35669, finalStart := 611565, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 206, sourceArmColumn := 35674, finalStart := 611770, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 207, sourceArmColumn := 35679, finalStart := 611975, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 208, sourceArmColumn := 35684, finalStart := 612180, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 209, sourceArmColumn := 35689, finalStart := 612385, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 210, sourceArmColumn := 35694, finalStart := 612590, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 211, sourceArmColumn := 35699, finalStart := 612795, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 212, sourceArmColumn := 35704, finalStart := 613000, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 213, sourceArmColumn := 35709, finalStart := 613205, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 214, sourceArmColumn := 35714, finalStart := 613410, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 215, sourceArmColumn := 35719, finalStart := 613615, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 216, sourceArmColumn := 35455, finalStart := 602791, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 217, sourceArmColumn := 35460, finalStart := 602996, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 218, sourceArmColumn := 35465, finalStart := 603201, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 219, sourceArmColumn := 35470, finalStart := 603406, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 220, sourceArmColumn := 35475, finalStart := 603611, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 221, sourceArmColumn := 35480, finalStart := 603816, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 222, sourceArmColumn := 35485, finalStart := 604021, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 223, sourceArmColumn := 35490, finalStart := 604226, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 224, sourceArmColumn := 35495, finalStart := 604431, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 225, sourceArmColumn := 35500, finalStart := 604636, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 226, sourceArmColumn := 35505, finalStart := 604841, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 227, sourceArmColumn := 35510, finalStart := 605046, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 228, sourceArmColumn := 35515, finalStart := 605251, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 229, sourceArmColumn := 35520, finalStart := 605456, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 230, sourceArmColumn := 35525, finalStart := 605661, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 231, sourceArmColumn := 35530, finalStart := 605866, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 232, sourceArmColumn := 35535, finalStart := 606071, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 233, sourceArmColumn := 35540, finalStart := 606276, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 234, sourceArmColumn := 35545, finalStart := 606481, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 235, sourceArmColumn := 35550, finalStart := 606686, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 236, sourceArmColumn := 35555, finalStart := 606891, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 237, sourceArmColumn := 35560, finalStart := 607096, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 238, sourceArmColumn := 35565, finalStart := 607301, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 239, sourceArmColumn := 35570, finalStart := 607506, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 240, sourceArmColumn := 35575, finalStart := 607711, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 241, sourceArmColumn := 35580, finalStart := 607916, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 242, sourceArmColumn := 35585, finalStart := 608121, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 243, sourceArmColumn := 35590, finalStart := 608326, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 244, sourceArmColumn := 35595, finalStart := 608531, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 245, sourceArmColumn := 35600, finalStart := 608736, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 246, sourceArmColumn := 35605, finalStart := 608941, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 247, sourceArmColumn := 35610, finalStart := 609146, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 248, sourceArmColumn := 35615, finalStart := 609351, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 249, sourceArmColumn := 35620, finalStart := 609556, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 250, sourceArmColumn := 35625, finalStart := 609761, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 251, sourceArmColumn := 35630, finalStart := 609966, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 252, sourceArmColumn := 35635, finalStart := 610171, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 253, sourceArmColumn := 35640, finalStart := 610376, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 254, sourceArmColumn := 35645, finalStart := 610581, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 255, sourceArmColumn := 35650, finalStart := 610786, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 256, sourceArmColumn := 35655, finalStart := 610991, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 257, sourceArmColumn := 35660, finalStart := 611196, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 258, sourceArmColumn := 35665, finalStart := 611401, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 259, sourceArmColumn := 35670, finalStart := 611606, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 260, sourceArmColumn := 35675, finalStart := 611811, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 261, sourceArmColumn := 35680, finalStart := 612016, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 262, sourceArmColumn := 35685, finalStart := 612221, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 263, sourceArmColumn := 35690, finalStart := 612426, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 264, sourceArmColumn := 35695, finalStart := 612631, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 265, sourceArmColumn := 35700, finalStart := 612836, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 266, sourceArmColumn := 35705, finalStart := 613041, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 267, sourceArmColumn := 35710, finalStart := 613246, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 268, sourceArmColumn := 35715, finalStart := 613451, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 269, sourceArmColumn := 35720, finalStart := 613656, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 0, sourceArmColumn := 37723, finalStart := 680117, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 1, sourceArmColumn := 37728, finalStart := 680322, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 2, sourceArmColumn := 37733, finalStart := 680527, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 3, sourceArmColumn := 37738, finalStart := 680732, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 4, sourceArmColumn := 37743, finalStart := 680937, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 5, sourceArmColumn := 37748, finalStart := 681142, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 6, sourceArmColumn := 37753, finalStart := 681347, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 7, sourceArmColumn := 37758, finalStart := 681552, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 8, sourceArmColumn := 37763, finalStart := 681757, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 9, sourceArmColumn := 37768, finalStart := 681962, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 10, sourceArmColumn := 37773, finalStart := 682167, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 11, sourceArmColumn := 37778, finalStart := 682372, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 12, sourceArmColumn := 37783, finalStart := 682577, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 13, sourceArmColumn := 37788, finalStart := 682782, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 14, sourceArmColumn := 37793, finalStart := 682987, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 15, sourceArmColumn := 37798, finalStart := 683192, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 16, sourceArmColumn := 37803, finalStart := 683397, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 17, sourceArmColumn := 37808, finalStart := 683602, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 18, sourceArmColumn := 37813, finalStart := 683807, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 19, sourceArmColumn := 37818, finalStart := 684012, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 20, sourceArmColumn := 37823, finalStart := 684217, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 21, sourceArmColumn := 37828, finalStart := 684422, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 22, sourceArmColumn := 37833, finalStart := 684627, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 23, sourceArmColumn := 37838, finalStart := 684832, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 24, sourceArmColumn := 37843, finalStart := 685037, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 25, sourceArmColumn := 37848, finalStart := 685242, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 26, sourceArmColumn := 37853, finalStart := 685447, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 27, sourceArmColumn := 37858, finalStart := 685652, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 28, sourceArmColumn := 37863, finalStart := 685857, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 29, sourceArmColumn := 37868, finalStart := 686062, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 30, sourceArmColumn := 37873, finalStart := 686267, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 31, sourceArmColumn := 37878, finalStart := 686472, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 32, sourceArmColumn := 37883, finalStart := 686677, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 33, sourceArmColumn := 37888, finalStart := 686882, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 34, sourceArmColumn := 37893, finalStart := 687087, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 35, sourceArmColumn := 37898, finalStart := 687292, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 36, sourceArmColumn := 37903, finalStart := 687497, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 37, sourceArmColumn := 37908, finalStart := 687702, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 38, sourceArmColumn := 37913, finalStart := 687907, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 39, sourceArmColumn := 37918, finalStart := 688112, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 40, sourceArmColumn := 37923, finalStart := 688317, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 41, sourceArmColumn := 37928, finalStart := 688522, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 42, sourceArmColumn := 37933, finalStart := 688727, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 43, sourceArmColumn := 37938, finalStart := 688932, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 44, sourceArmColumn := 37943, finalStart := 689137, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 45, sourceArmColumn := 37948, finalStart := 689342, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 46, sourceArmColumn := 37953, finalStart := 689547, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 47, sourceArmColumn := 37958, finalStart := 689752, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 48, sourceArmColumn := 37963, finalStart := 689957, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 49, sourceArmColumn := 37968, finalStart := 690162, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 50, sourceArmColumn := 37973, finalStart := 690367, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 51, sourceArmColumn := 37978, finalStart := 690572, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 52, sourceArmColumn := 37983, finalStart := 690777, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 53, sourceArmColumn := 37988, finalStart := 690982, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 54, sourceArmColumn := 37724, finalStart := 680158, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 55, sourceArmColumn := 37729, finalStart := 680363, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 56, sourceArmColumn := 37734, finalStart := 680568, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 57, sourceArmColumn := 37739, finalStart := 680773, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 58, sourceArmColumn := 37744, finalStart := 680978, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 59, sourceArmColumn := 37749, finalStart := 681183, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 60, sourceArmColumn := 37754, finalStart := 681388, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 61, sourceArmColumn := 37759, finalStart := 681593, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 62, sourceArmColumn := 37764, finalStart := 681798, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 63, sourceArmColumn := 37769, finalStart := 682003, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 64, sourceArmColumn := 37774, finalStart := 682208, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 65, sourceArmColumn := 37779, finalStart := 682413, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 66, sourceArmColumn := 37784, finalStart := 682618, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 67, sourceArmColumn := 37789, finalStart := 682823, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 68, sourceArmColumn := 37794, finalStart := 683028, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 69, sourceArmColumn := 37799, finalStart := 683233, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 70, sourceArmColumn := 37804, finalStart := 683438, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 71, sourceArmColumn := 37809, finalStart := 683643, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 72, sourceArmColumn := 37814, finalStart := 683848, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 73, sourceArmColumn := 37819, finalStart := 684053, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 74, sourceArmColumn := 37824, finalStart := 684258, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 75, sourceArmColumn := 37829, finalStart := 684463, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 76, sourceArmColumn := 37834, finalStart := 684668, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 77, sourceArmColumn := 37839, finalStart := 684873, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 78, sourceArmColumn := 37844, finalStart := 685078, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 79, sourceArmColumn := 37849, finalStart := 685283, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 80, sourceArmColumn := 37854, finalStart := 685488, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 81, sourceArmColumn := 37859, finalStart := 685693, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 82, sourceArmColumn := 37864, finalStart := 685898, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 83, sourceArmColumn := 37869, finalStart := 686103, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 84, sourceArmColumn := 37874, finalStart := 686308, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 85, sourceArmColumn := 37879, finalStart := 686513, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 86, sourceArmColumn := 37884, finalStart := 686718, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 87, sourceArmColumn := 37889, finalStart := 686923, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 88, sourceArmColumn := 37894, finalStart := 687128, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 89, sourceArmColumn := 37899, finalStart := 687333, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 90, sourceArmColumn := 37904, finalStart := 687538, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 91, sourceArmColumn := 37909, finalStart := 687743, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 92, sourceArmColumn := 37914, finalStart := 687948, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 93, sourceArmColumn := 37919, finalStart := 688153, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 94, sourceArmColumn := 37924, finalStart := 688358, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 95, sourceArmColumn := 37929, finalStart := 688563, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 96, sourceArmColumn := 37934, finalStart := 688768, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 97, sourceArmColumn := 37939, finalStart := 688973, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 98, sourceArmColumn := 37944, finalStart := 689178, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 99, sourceArmColumn := 37949, finalStart := 689383, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 100, sourceArmColumn := 37954, finalStart := 689588, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 101, sourceArmColumn := 37959, finalStart := 689793, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 102, sourceArmColumn := 37964, finalStart := 689998, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 103, sourceArmColumn := 37969, finalStart := 690203, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 104, sourceArmColumn := 37974, finalStart := 690408, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 105, sourceArmColumn := 37979, finalStart := 690613, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 106, sourceArmColumn := 37984, finalStart := 690818, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 107, sourceArmColumn := 37989, finalStart := 691023, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 108, sourceArmColumn := 37725, finalStart := 680199, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 109, sourceArmColumn := 37730, finalStart := 680404, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 110, sourceArmColumn := 37735, finalStart := 680609, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 111, sourceArmColumn := 37740, finalStart := 680814, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 112, sourceArmColumn := 37745, finalStart := 681019, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 113, sourceArmColumn := 37750, finalStart := 681224, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 114, sourceArmColumn := 37755, finalStart := 681429, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 115, sourceArmColumn := 37760, finalStart := 681634, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 116, sourceArmColumn := 37765, finalStart := 681839, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 117, sourceArmColumn := 37770, finalStart := 682044, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 118, sourceArmColumn := 37775, finalStart := 682249, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 119, sourceArmColumn := 37780, finalStart := 682454, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 120, sourceArmColumn := 37785, finalStart := 682659, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 121, sourceArmColumn := 37790, finalStart := 682864, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 122, sourceArmColumn := 37795, finalStart := 683069, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 123, sourceArmColumn := 37800, finalStart := 683274, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 124, sourceArmColumn := 37805, finalStart := 683479, width := 41, encoding := .balancedTernary }
, { child := 7, logicalColumn := 125, sourceArmColumn := 37810, finalStart := 683684, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk7
