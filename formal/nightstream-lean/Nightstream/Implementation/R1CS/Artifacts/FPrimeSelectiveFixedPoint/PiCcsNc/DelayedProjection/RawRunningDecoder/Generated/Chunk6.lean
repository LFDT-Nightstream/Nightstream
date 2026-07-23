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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk6

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 5, logicalColumn := 162, sourceArmColumn := 33182, finalStart := 525260, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 163, sourceArmColumn := 33187, finalStart := 525465, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 164, sourceArmColumn := 33192, finalStart := 525670, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 165, sourceArmColumn := 33197, finalStart := 525875, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 166, sourceArmColumn := 33202, finalStart := 526080, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 167, sourceArmColumn := 33207, finalStart := 526285, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 168, sourceArmColumn := 33212, finalStart := 526490, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 169, sourceArmColumn := 33217, finalStart := 526695, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 170, sourceArmColumn := 33222, finalStart := 526900, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 171, sourceArmColumn := 33227, finalStart := 527105, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 172, sourceArmColumn := 33232, finalStart := 527310, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 173, sourceArmColumn := 33237, finalStart := 527515, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 174, sourceArmColumn := 33242, finalStart := 527720, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 175, sourceArmColumn := 33247, finalStart := 527925, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 176, sourceArmColumn := 33252, finalStart := 528130, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 177, sourceArmColumn := 33257, finalStart := 528335, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 178, sourceArmColumn := 33262, finalStart := 528540, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 179, sourceArmColumn := 33267, finalStart := 528745, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 180, sourceArmColumn := 33272, finalStart := 528950, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 181, sourceArmColumn := 33277, finalStart := 529155, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 182, sourceArmColumn := 33282, finalStart := 529360, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 183, sourceArmColumn := 33287, finalStart := 529565, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 184, sourceArmColumn := 33292, finalStart := 529770, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 185, sourceArmColumn := 33297, finalStart := 529975, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 186, sourceArmColumn := 33302, finalStart := 530180, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 187, sourceArmColumn := 33307, finalStart := 530385, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 188, sourceArmColumn := 33312, finalStart := 530590, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 189, sourceArmColumn := 33317, finalStart := 530795, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 190, sourceArmColumn := 33322, finalStart := 531000, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 191, sourceArmColumn := 33327, finalStart := 531205, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 192, sourceArmColumn := 33332, finalStart := 531410, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 193, sourceArmColumn := 33337, finalStart := 531615, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 194, sourceArmColumn := 33342, finalStart := 531820, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 195, sourceArmColumn := 33347, finalStart := 532025, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 196, sourceArmColumn := 33352, finalStart := 532230, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 197, sourceArmColumn := 33357, finalStart := 532435, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 198, sourceArmColumn := 33362, finalStart := 532640, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 199, sourceArmColumn := 33367, finalStart := 532845, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 200, sourceArmColumn := 33372, finalStart := 533050, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 201, sourceArmColumn := 33377, finalStart := 533255, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 202, sourceArmColumn := 33382, finalStart := 533460, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 203, sourceArmColumn := 33387, finalStart := 533665, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 204, sourceArmColumn := 33392, finalStart := 533870, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 205, sourceArmColumn := 33397, finalStart := 534075, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 206, sourceArmColumn := 33402, finalStart := 534280, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 207, sourceArmColumn := 33407, finalStart := 534485, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 208, sourceArmColumn := 33412, finalStart := 534690, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 209, sourceArmColumn := 33417, finalStart := 534895, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 210, sourceArmColumn := 33422, finalStart := 535100, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 211, sourceArmColumn := 33427, finalStart := 535305, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 212, sourceArmColumn := 33432, finalStart := 535510, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 213, sourceArmColumn := 33437, finalStart := 535715, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 214, sourceArmColumn := 33442, finalStart := 535920, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 215, sourceArmColumn := 33447, finalStart := 536125, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 216, sourceArmColumn := 33183, finalStart := 525301, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 217, sourceArmColumn := 33188, finalStart := 525506, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 218, sourceArmColumn := 33193, finalStart := 525711, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 219, sourceArmColumn := 33198, finalStart := 525916, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 220, sourceArmColumn := 33203, finalStart := 526121, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 221, sourceArmColumn := 33208, finalStart := 526326, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 222, sourceArmColumn := 33213, finalStart := 526531, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 223, sourceArmColumn := 33218, finalStart := 526736, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 224, sourceArmColumn := 33223, finalStart := 526941, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 225, sourceArmColumn := 33228, finalStart := 527146, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 226, sourceArmColumn := 33233, finalStart := 527351, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 227, sourceArmColumn := 33238, finalStart := 527556, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 228, sourceArmColumn := 33243, finalStart := 527761, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 229, sourceArmColumn := 33248, finalStart := 527966, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 230, sourceArmColumn := 33253, finalStart := 528171, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 231, sourceArmColumn := 33258, finalStart := 528376, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 232, sourceArmColumn := 33263, finalStart := 528581, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 233, sourceArmColumn := 33268, finalStart := 528786, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 234, sourceArmColumn := 33273, finalStart := 528991, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 235, sourceArmColumn := 33278, finalStart := 529196, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 236, sourceArmColumn := 33283, finalStart := 529401, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 237, sourceArmColumn := 33288, finalStart := 529606, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 238, sourceArmColumn := 33293, finalStart := 529811, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 239, sourceArmColumn := 33298, finalStart := 530016, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 240, sourceArmColumn := 33303, finalStart := 530221, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 241, sourceArmColumn := 33308, finalStart := 530426, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 242, sourceArmColumn := 33313, finalStart := 530631, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 243, sourceArmColumn := 33318, finalStart := 530836, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 244, sourceArmColumn := 33323, finalStart := 531041, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 245, sourceArmColumn := 33328, finalStart := 531246, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 246, sourceArmColumn := 33333, finalStart := 531451, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 247, sourceArmColumn := 33338, finalStart := 531656, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 248, sourceArmColumn := 33343, finalStart := 531861, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 249, sourceArmColumn := 33348, finalStart := 532066, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 250, sourceArmColumn := 33353, finalStart := 532271, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 251, sourceArmColumn := 33358, finalStart := 532476, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 252, sourceArmColumn := 33363, finalStart := 532681, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 253, sourceArmColumn := 33368, finalStart := 532886, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 254, sourceArmColumn := 33373, finalStart := 533091, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 255, sourceArmColumn := 33378, finalStart := 533296, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 256, sourceArmColumn := 33383, finalStart := 533501, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 257, sourceArmColumn := 33388, finalStart := 533706, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 258, sourceArmColumn := 33393, finalStart := 533911, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 259, sourceArmColumn := 33398, finalStart := 534116, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 260, sourceArmColumn := 33403, finalStart := 534321, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 261, sourceArmColumn := 33408, finalStart := 534526, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 262, sourceArmColumn := 33413, finalStart := 534731, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 263, sourceArmColumn := 33418, finalStart := 534936, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 264, sourceArmColumn := 33423, finalStart := 535141, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 265, sourceArmColumn := 33428, finalStart := 535346, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 266, sourceArmColumn := 33433, finalStart := 535551, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 267, sourceArmColumn := 33438, finalStart := 535756, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 268, sourceArmColumn := 33443, finalStart := 535961, width := 41, encoding := .balancedTernary }
, { child := 5, logicalColumn := 269, sourceArmColumn := 33448, finalStart := 536166, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 0, sourceArmColumn := 35451, finalStart := 602627, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 1, sourceArmColumn := 35456, finalStart := 602832, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 2, sourceArmColumn := 35461, finalStart := 603037, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 3, sourceArmColumn := 35466, finalStart := 603242, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 4, sourceArmColumn := 35471, finalStart := 603447, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 5, sourceArmColumn := 35476, finalStart := 603652, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 6, sourceArmColumn := 35481, finalStart := 603857, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 7, sourceArmColumn := 35486, finalStart := 604062, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 8, sourceArmColumn := 35491, finalStart := 604267, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 9, sourceArmColumn := 35496, finalStart := 604472, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 10, sourceArmColumn := 35501, finalStart := 604677, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 11, sourceArmColumn := 35506, finalStart := 604882, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 12, sourceArmColumn := 35511, finalStart := 605087, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 13, sourceArmColumn := 35516, finalStart := 605292, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 14, sourceArmColumn := 35521, finalStart := 605497, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 15, sourceArmColumn := 35526, finalStart := 605702, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 16, sourceArmColumn := 35531, finalStart := 605907, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 17, sourceArmColumn := 35536, finalStart := 606112, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 18, sourceArmColumn := 35541, finalStart := 606317, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 19, sourceArmColumn := 35546, finalStart := 606522, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 20, sourceArmColumn := 35551, finalStart := 606727, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 21, sourceArmColumn := 35556, finalStart := 606932, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 22, sourceArmColumn := 35561, finalStart := 607137, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 23, sourceArmColumn := 35566, finalStart := 607342, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 24, sourceArmColumn := 35571, finalStart := 607547, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 25, sourceArmColumn := 35576, finalStart := 607752, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 26, sourceArmColumn := 35581, finalStart := 607957, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 27, sourceArmColumn := 35586, finalStart := 608162, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 28, sourceArmColumn := 35591, finalStart := 608367, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 29, sourceArmColumn := 35596, finalStart := 608572, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 30, sourceArmColumn := 35601, finalStart := 608777, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 31, sourceArmColumn := 35606, finalStart := 608982, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 32, sourceArmColumn := 35611, finalStart := 609187, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 33, sourceArmColumn := 35616, finalStart := 609392, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 34, sourceArmColumn := 35621, finalStart := 609597, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 35, sourceArmColumn := 35626, finalStart := 609802, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 36, sourceArmColumn := 35631, finalStart := 610007, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 37, sourceArmColumn := 35636, finalStart := 610212, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 38, sourceArmColumn := 35641, finalStart := 610417, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 39, sourceArmColumn := 35646, finalStart := 610622, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 40, sourceArmColumn := 35651, finalStart := 610827, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 41, sourceArmColumn := 35656, finalStart := 611032, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 42, sourceArmColumn := 35661, finalStart := 611237, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 43, sourceArmColumn := 35666, finalStart := 611442, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 44, sourceArmColumn := 35671, finalStart := 611647, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 45, sourceArmColumn := 35676, finalStart := 611852, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 46, sourceArmColumn := 35681, finalStart := 612057, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 47, sourceArmColumn := 35686, finalStart := 612262, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 48, sourceArmColumn := 35691, finalStart := 612467, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 49, sourceArmColumn := 35696, finalStart := 612672, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 50, sourceArmColumn := 35701, finalStart := 612877, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 51, sourceArmColumn := 35706, finalStart := 613082, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 52, sourceArmColumn := 35711, finalStart := 613287, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 53, sourceArmColumn := 35716, finalStart := 613492, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 54, sourceArmColumn := 35452, finalStart := 602668, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 55, sourceArmColumn := 35457, finalStart := 602873, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 56, sourceArmColumn := 35462, finalStart := 603078, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 57, sourceArmColumn := 35467, finalStart := 603283, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 58, sourceArmColumn := 35472, finalStart := 603488, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 59, sourceArmColumn := 35477, finalStart := 603693, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 60, sourceArmColumn := 35482, finalStart := 603898, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 61, sourceArmColumn := 35487, finalStart := 604103, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 62, sourceArmColumn := 35492, finalStart := 604308, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 63, sourceArmColumn := 35497, finalStart := 604513, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 64, sourceArmColumn := 35502, finalStart := 604718, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 65, sourceArmColumn := 35507, finalStart := 604923, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 66, sourceArmColumn := 35512, finalStart := 605128, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 67, sourceArmColumn := 35517, finalStart := 605333, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 68, sourceArmColumn := 35522, finalStart := 605538, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 69, sourceArmColumn := 35527, finalStart := 605743, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 70, sourceArmColumn := 35532, finalStart := 605948, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 71, sourceArmColumn := 35537, finalStart := 606153, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 72, sourceArmColumn := 35542, finalStart := 606358, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 73, sourceArmColumn := 35547, finalStart := 606563, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 74, sourceArmColumn := 35552, finalStart := 606768, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 75, sourceArmColumn := 35557, finalStart := 606973, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 76, sourceArmColumn := 35562, finalStart := 607178, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 77, sourceArmColumn := 35567, finalStart := 607383, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 78, sourceArmColumn := 35572, finalStart := 607588, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 79, sourceArmColumn := 35577, finalStart := 607793, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 80, sourceArmColumn := 35582, finalStart := 607998, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 81, sourceArmColumn := 35587, finalStart := 608203, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 82, sourceArmColumn := 35592, finalStart := 608408, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 83, sourceArmColumn := 35597, finalStart := 608613, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 84, sourceArmColumn := 35602, finalStart := 608818, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 85, sourceArmColumn := 35607, finalStart := 609023, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 86, sourceArmColumn := 35612, finalStart := 609228, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 87, sourceArmColumn := 35617, finalStart := 609433, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 88, sourceArmColumn := 35622, finalStart := 609638, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 89, sourceArmColumn := 35627, finalStart := 609843, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 90, sourceArmColumn := 35632, finalStart := 610048, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 91, sourceArmColumn := 35637, finalStart := 610253, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 92, sourceArmColumn := 35642, finalStart := 610458, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 93, sourceArmColumn := 35647, finalStart := 610663, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 94, sourceArmColumn := 35652, finalStart := 610868, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 95, sourceArmColumn := 35657, finalStart := 611073, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 96, sourceArmColumn := 35662, finalStart := 611278, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 97, sourceArmColumn := 35667, finalStart := 611483, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 98, sourceArmColumn := 35672, finalStart := 611688, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 99, sourceArmColumn := 35677, finalStart := 611893, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 100, sourceArmColumn := 35682, finalStart := 612098, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 101, sourceArmColumn := 35687, finalStart := 612303, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 102, sourceArmColumn := 35692, finalStart := 612508, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 103, sourceArmColumn := 35697, finalStart := 612713, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 104, sourceArmColumn := 35702, finalStart := 612918, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 105, sourceArmColumn := 35707, finalStart := 613123, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 106, sourceArmColumn := 35712, finalStart := 613328, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 107, sourceArmColumn := 35717, finalStart := 613533, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 108, sourceArmColumn := 35453, finalStart := 602709, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 109, sourceArmColumn := 35458, finalStart := 602914, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 110, sourceArmColumn := 35463, finalStart := 603119, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 111, sourceArmColumn := 35468, finalStart := 603324, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 112, sourceArmColumn := 35473, finalStart := 603529, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 113, sourceArmColumn := 35478, finalStart := 603734, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 114, sourceArmColumn := 35483, finalStart := 603939, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 115, sourceArmColumn := 35488, finalStart := 604144, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 116, sourceArmColumn := 35493, finalStart := 604349, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 117, sourceArmColumn := 35498, finalStart := 604554, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 118, sourceArmColumn := 35503, finalStart := 604759, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 119, sourceArmColumn := 35508, finalStart := 604964, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 120, sourceArmColumn := 35513, finalStart := 605169, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 121, sourceArmColumn := 35518, finalStart := 605374, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 122, sourceArmColumn := 35523, finalStart := 605579, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 123, sourceArmColumn := 35528, finalStart := 605784, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 124, sourceArmColumn := 35533, finalStart := 605989, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 125, sourceArmColumn := 35538, finalStart := 606194, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 126, sourceArmColumn := 35543, finalStart := 606399, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 127, sourceArmColumn := 35548, finalStart := 606604, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 128, sourceArmColumn := 35553, finalStart := 606809, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 129, sourceArmColumn := 35558, finalStart := 607014, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 130, sourceArmColumn := 35563, finalStart := 607219, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 131, sourceArmColumn := 35568, finalStart := 607424, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 132, sourceArmColumn := 35573, finalStart := 607629, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 133, sourceArmColumn := 35578, finalStart := 607834, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 134, sourceArmColumn := 35583, finalStart := 608039, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 135, sourceArmColumn := 35588, finalStart := 608244, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 136, sourceArmColumn := 35593, finalStart := 608449, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 137, sourceArmColumn := 35598, finalStart := 608654, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 138, sourceArmColumn := 35603, finalStart := 608859, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 139, sourceArmColumn := 35608, finalStart := 609064, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 140, sourceArmColumn := 35613, finalStart := 609269, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 141, sourceArmColumn := 35618, finalStart := 609474, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 142, sourceArmColumn := 35623, finalStart := 609679, width := 41, encoding := .balancedTernary }
, { child := 6, logicalColumn := 143, sourceArmColumn := 35628, finalStart := 609884, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk6
