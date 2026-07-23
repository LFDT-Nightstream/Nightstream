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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk14

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 13, logicalColumn := 18, sourceArmColumn := 51445, finalStart := 1148747, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 19, sourceArmColumn := 51450, finalStart := 1148952, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 20, sourceArmColumn := 51455, finalStart := 1149157, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 21, sourceArmColumn := 51460, finalStart := 1149362, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 22, sourceArmColumn := 51465, finalStart := 1149567, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 23, sourceArmColumn := 51470, finalStart := 1149772, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 24, sourceArmColumn := 51475, finalStart := 1149977, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 25, sourceArmColumn := 51480, finalStart := 1150182, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 26, sourceArmColumn := 51485, finalStart := 1150387, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 27, sourceArmColumn := 51490, finalStart := 1150592, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 28, sourceArmColumn := 51495, finalStart := 1150797, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 29, sourceArmColumn := 51500, finalStart := 1151002, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 30, sourceArmColumn := 51505, finalStart := 1151207, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 31, sourceArmColumn := 51510, finalStart := 1151412, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 32, sourceArmColumn := 51515, finalStart := 1151617, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 33, sourceArmColumn := 51520, finalStart := 1151822, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 34, sourceArmColumn := 51525, finalStart := 1152027, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 35, sourceArmColumn := 51530, finalStart := 1152232, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 36, sourceArmColumn := 51535, finalStart := 1152437, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 37, sourceArmColumn := 51540, finalStart := 1152642, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 38, sourceArmColumn := 51545, finalStart := 1152847, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 39, sourceArmColumn := 51550, finalStart := 1153052, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 40, sourceArmColumn := 51555, finalStart := 1153257, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 41, sourceArmColumn := 51560, finalStart := 1153462, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 42, sourceArmColumn := 51565, finalStart := 1153667, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 43, sourceArmColumn := 51570, finalStart := 1153872, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 44, sourceArmColumn := 51575, finalStart := 1154077, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 45, sourceArmColumn := 51580, finalStart := 1154282, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 46, sourceArmColumn := 51585, finalStart := 1154487, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 47, sourceArmColumn := 51590, finalStart := 1154692, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 48, sourceArmColumn := 51595, finalStart := 1154897, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 49, sourceArmColumn := 51600, finalStart := 1155102, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 50, sourceArmColumn := 51605, finalStart := 1155307, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 51, sourceArmColumn := 51610, finalStart := 1155512, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 52, sourceArmColumn := 51615, finalStart := 1155717, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 53, sourceArmColumn := 51620, finalStart := 1155922, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 54, sourceArmColumn := 51356, finalStart := 1145098, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 55, sourceArmColumn := 51361, finalStart := 1145303, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 56, sourceArmColumn := 51366, finalStart := 1145508, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 57, sourceArmColumn := 51371, finalStart := 1145713, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 58, sourceArmColumn := 51376, finalStart := 1145918, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 59, sourceArmColumn := 51381, finalStart := 1146123, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 60, sourceArmColumn := 51386, finalStart := 1146328, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 61, sourceArmColumn := 51391, finalStart := 1146533, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 62, sourceArmColumn := 51396, finalStart := 1146738, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 63, sourceArmColumn := 51401, finalStart := 1146943, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 64, sourceArmColumn := 51406, finalStart := 1147148, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 65, sourceArmColumn := 51411, finalStart := 1147353, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 66, sourceArmColumn := 51416, finalStart := 1147558, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 67, sourceArmColumn := 51421, finalStart := 1147763, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 68, sourceArmColumn := 51426, finalStart := 1147968, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 69, sourceArmColumn := 51431, finalStart := 1148173, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 70, sourceArmColumn := 51436, finalStart := 1148378, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 71, sourceArmColumn := 51441, finalStart := 1148583, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 72, sourceArmColumn := 51446, finalStart := 1148788, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 73, sourceArmColumn := 51451, finalStart := 1148993, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 74, sourceArmColumn := 51456, finalStart := 1149198, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 75, sourceArmColumn := 51461, finalStart := 1149403, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 76, sourceArmColumn := 51466, finalStart := 1149608, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 77, sourceArmColumn := 51471, finalStart := 1149813, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 78, sourceArmColumn := 51476, finalStart := 1150018, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 79, sourceArmColumn := 51481, finalStart := 1150223, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 80, sourceArmColumn := 51486, finalStart := 1150428, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 81, sourceArmColumn := 51491, finalStart := 1150633, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 82, sourceArmColumn := 51496, finalStart := 1150838, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 83, sourceArmColumn := 51501, finalStart := 1151043, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 84, sourceArmColumn := 51506, finalStart := 1151248, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 85, sourceArmColumn := 51511, finalStart := 1151453, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 86, sourceArmColumn := 51516, finalStart := 1151658, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 87, sourceArmColumn := 51521, finalStart := 1151863, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 88, sourceArmColumn := 51526, finalStart := 1152068, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 89, sourceArmColumn := 51531, finalStart := 1152273, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 90, sourceArmColumn := 51536, finalStart := 1152478, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 91, sourceArmColumn := 51541, finalStart := 1152683, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 92, sourceArmColumn := 51546, finalStart := 1152888, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 93, sourceArmColumn := 51551, finalStart := 1153093, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 94, sourceArmColumn := 51556, finalStart := 1153298, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 95, sourceArmColumn := 51561, finalStart := 1153503, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 96, sourceArmColumn := 51566, finalStart := 1153708, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 97, sourceArmColumn := 51571, finalStart := 1153913, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 98, sourceArmColumn := 51576, finalStart := 1154118, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 99, sourceArmColumn := 51581, finalStart := 1154323, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 100, sourceArmColumn := 51586, finalStart := 1154528, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 101, sourceArmColumn := 51591, finalStart := 1154733, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 102, sourceArmColumn := 51596, finalStart := 1154938, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 103, sourceArmColumn := 51601, finalStart := 1155143, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 104, sourceArmColumn := 51606, finalStart := 1155348, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 105, sourceArmColumn := 51611, finalStart := 1155553, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 106, sourceArmColumn := 51616, finalStart := 1155758, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 107, sourceArmColumn := 51621, finalStart := 1155963, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 108, sourceArmColumn := 51357, finalStart := 1145139, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 109, sourceArmColumn := 51362, finalStart := 1145344, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 110, sourceArmColumn := 51367, finalStart := 1145549, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 111, sourceArmColumn := 51372, finalStart := 1145754, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 112, sourceArmColumn := 51377, finalStart := 1145959, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 113, sourceArmColumn := 51382, finalStart := 1146164, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 114, sourceArmColumn := 51387, finalStart := 1146369, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 115, sourceArmColumn := 51392, finalStart := 1146574, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 116, sourceArmColumn := 51397, finalStart := 1146779, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 117, sourceArmColumn := 51402, finalStart := 1146984, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 118, sourceArmColumn := 51407, finalStart := 1147189, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 119, sourceArmColumn := 51412, finalStart := 1147394, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 120, sourceArmColumn := 51417, finalStart := 1147599, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 121, sourceArmColumn := 51422, finalStart := 1147804, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 122, sourceArmColumn := 51427, finalStart := 1148009, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 123, sourceArmColumn := 51432, finalStart := 1148214, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 124, sourceArmColumn := 51437, finalStart := 1148419, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 125, sourceArmColumn := 51442, finalStart := 1148624, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 126, sourceArmColumn := 51447, finalStart := 1148829, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 127, sourceArmColumn := 51452, finalStart := 1149034, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 128, sourceArmColumn := 51457, finalStart := 1149239, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 129, sourceArmColumn := 51462, finalStart := 1149444, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 130, sourceArmColumn := 51467, finalStart := 1149649, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 131, sourceArmColumn := 51472, finalStart := 1149854, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 132, sourceArmColumn := 51477, finalStart := 1150059, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 133, sourceArmColumn := 51482, finalStart := 1150264, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 134, sourceArmColumn := 51487, finalStart := 1150469, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 135, sourceArmColumn := 51492, finalStart := 1150674, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 136, sourceArmColumn := 51497, finalStart := 1150879, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 137, sourceArmColumn := 51502, finalStart := 1151084, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 138, sourceArmColumn := 51507, finalStart := 1151289, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 139, sourceArmColumn := 51512, finalStart := 1151494, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 140, sourceArmColumn := 51517, finalStart := 1151699, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 141, sourceArmColumn := 51522, finalStart := 1151904, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 142, sourceArmColumn := 51527, finalStart := 1152109, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 143, sourceArmColumn := 51532, finalStart := 1152314, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 144, sourceArmColumn := 51537, finalStart := 1152519, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 145, sourceArmColumn := 51542, finalStart := 1152724, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 146, sourceArmColumn := 51547, finalStart := 1152929, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 147, sourceArmColumn := 51552, finalStart := 1153134, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 148, sourceArmColumn := 51557, finalStart := 1153339, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 149, sourceArmColumn := 51562, finalStart := 1153544, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 150, sourceArmColumn := 51567, finalStart := 1153749, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 151, sourceArmColumn := 51572, finalStart := 1153954, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 152, sourceArmColumn := 51577, finalStart := 1154159, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 153, sourceArmColumn := 51582, finalStart := 1154364, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 154, sourceArmColumn := 51587, finalStart := 1154569, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 155, sourceArmColumn := 51592, finalStart := 1154774, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 156, sourceArmColumn := 51597, finalStart := 1154979, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 157, sourceArmColumn := 51602, finalStart := 1155184, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 158, sourceArmColumn := 51607, finalStart := 1155389, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 159, sourceArmColumn := 51612, finalStart := 1155594, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 160, sourceArmColumn := 51617, finalStart := 1155799, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 161, sourceArmColumn := 51622, finalStart := 1156004, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 162, sourceArmColumn := 51358, finalStart := 1145180, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 163, sourceArmColumn := 51363, finalStart := 1145385, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 164, sourceArmColumn := 51368, finalStart := 1145590, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 165, sourceArmColumn := 51373, finalStart := 1145795, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 166, sourceArmColumn := 51378, finalStart := 1146000, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 167, sourceArmColumn := 51383, finalStart := 1146205, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 168, sourceArmColumn := 51388, finalStart := 1146410, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 169, sourceArmColumn := 51393, finalStart := 1146615, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 170, sourceArmColumn := 51398, finalStart := 1146820, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 171, sourceArmColumn := 51403, finalStart := 1147025, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 172, sourceArmColumn := 51408, finalStart := 1147230, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 173, sourceArmColumn := 51413, finalStart := 1147435, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 174, sourceArmColumn := 51418, finalStart := 1147640, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 175, sourceArmColumn := 51423, finalStart := 1147845, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 176, sourceArmColumn := 51428, finalStart := 1148050, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 177, sourceArmColumn := 51433, finalStart := 1148255, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 178, sourceArmColumn := 51438, finalStart := 1148460, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 179, sourceArmColumn := 51443, finalStart := 1148665, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 180, sourceArmColumn := 51448, finalStart := 1148870, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 181, sourceArmColumn := 51453, finalStart := 1149075, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 182, sourceArmColumn := 51458, finalStart := 1149280, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 183, sourceArmColumn := 51463, finalStart := 1149485, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 184, sourceArmColumn := 51468, finalStart := 1149690, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 185, sourceArmColumn := 51473, finalStart := 1149895, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 186, sourceArmColumn := 51478, finalStart := 1150100, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 187, sourceArmColumn := 51483, finalStart := 1150305, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 188, sourceArmColumn := 51488, finalStart := 1150510, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 189, sourceArmColumn := 51493, finalStart := 1150715, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 190, sourceArmColumn := 51498, finalStart := 1150920, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 191, sourceArmColumn := 51503, finalStart := 1151125, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 192, sourceArmColumn := 51508, finalStart := 1151330, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 193, sourceArmColumn := 51513, finalStart := 1151535, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 194, sourceArmColumn := 51518, finalStart := 1151740, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 195, sourceArmColumn := 51523, finalStart := 1151945, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 196, sourceArmColumn := 51528, finalStart := 1152150, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 197, sourceArmColumn := 51533, finalStart := 1152355, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 198, sourceArmColumn := 51538, finalStart := 1152560, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 199, sourceArmColumn := 51543, finalStart := 1152765, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 200, sourceArmColumn := 51548, finalStart := 1152970, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 201, sourceArmColumn := 51553, finalStart := 1153175, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 202, sourceArmColumn := 51558, finalStart := 1153380, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 203, sourceArmColumn := 51563, finalStart := 1153585, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 204, sourceArmColumn := 51568, finalStart := 1153790, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 205, sourceArmColumn := 51573, finalStart := 1153995, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 206, sourceArmColumn := 51578, finalStart := 1154200, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 207, sourceArmColumn := 51583, finalStart := 1154405, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 208, sourceArmColumn := 51588, finalStart := 1154610, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 209, sourceArmColumn := 51593, finalStart := 1154815, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 210, sourceArmColumn := 51598, finalStart := 1155020, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 211, sourceArmColumn := 51603, finalStart := 1155225, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 212, sourceArmColumn := 51608, finalStart := 1155430, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 213, sourceArmColumn := 51613, finalStart := 1155635, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 214, sourceArmColumn := 51618, finalStart := 1155840, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 215, sourceArmColumn := 51623, finalStart := 1156045, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 216, sourceArmColumn := 51359, finalStart := 1145221, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 217, sourceArmColumn := 51364, finalStart := 1145426, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 218, sourceArmColumn := 51369, finalStart := 1145631, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 219, sourceArmColumn := 51374, finalStart := 1145836, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 220, sourceArmColumn := 51379, finalStart := 1146041, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 221, sourceArmColumn := 51384, finalStart := 1146246, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 222, sourceArmColumn := 51389, finalStart := 1146451, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 223, sourceArmColumn := 51394, finalStart := 1146656, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 224, sourceArmColumn := 51399, finalStart := 1146861, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 225, sourceArmColumn := 51404, finalStart := 1147066, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 226, sourceArmColumn := 51409, finalStart := 1147271, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 227, sourceArmColumn := 51414, finalStart := 1147476, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 228, sourceArmColumn := 51419, finalStart := 1147681, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 229, sourceArmColumn := 51424, finalStart := 1147886, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 230, sourceArmColumn := 51429, finalStart := 1148091, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 231, sourceArmColumn := 51434, finalStart := 1148296, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 232, sourceArmColumn := 51439, finalStart := 1148501, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 233, sourceArmColumn := 51444, finalStart := 1148706, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 234, sourceArmColumn := 51449, finalStart := 1148911, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 235, sourceArmColumn := 51454, finalStart := 1149116, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 236, sourceArmColumn := 51459, finalStart := 1149321, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 237, sourceArmColumn := 51464, finalStart := 1149526, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 238, sourceArmColumn := 51469, finalStart := 1149731, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 239, sourceArmColumn := 51474, finalStart := 1149936, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 240, sourceArmColumn := 51479, finalStart := 1150141, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 241, sourceArmColumn := 51484, finalStart := 1150346, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 242, sourceArmColumn := 51489, finalStart := 1150551, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 243, sourceArmColumn := 51494, finalStart := 1150756, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 244, sourceArmColumn := 51499, finalStart := 1150961, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 245, sourceArmColumn := 51504, finalStart := 1151166, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 246, sourceArmColumn := 51509, finalStart := 1151371, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 247, sourceArmColumn := 51514, finalStart := 1151576, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 248, sourceArmColumn := 51519, finalStart := 1151781, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 249, sourceArmColumn := 51524, finalStart := 1151986, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 250, sourceArmColumn := 51529, finalStart := 1152191, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 251, sourceArmColumn := 51534, finalStart := 1152396, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 252, sourceArmColumn := 51539, finalStart := 1152601, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 253, sourceArmColumn := 51544, finalStart := 1152806, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 254, sourceArmColumn := 51549, finalStart := 1153011, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 255, sourceArmColumn := 51554, finalStart := 1153216, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 256, sourceArmColumn := 51559, finalStart := 1153421, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 257, sourceArmColumn := 51564, finalStart := 1153626, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 258, sourceArmColumn := 51569, finalStart := 1153831, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 259, sourceArmColumn := 51574, finalStart := 1154036, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 260, sourceArmColumn := 51579, finalStart := 1154241, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 261, sourceArmColumn := 51584, finalStart := 1154446, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 262, sourceArmColumn := 51589, finalStart := 1154651, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 263, sourceArmColumn := 51594, finalStart := 1154856, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 264, sourceArmColumn := 51599, finalStart := 1155061, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 265, sourceArmColumn := 51604, finalStart := 1155266, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 266, sourceArmColumn := 51609, finalStart := 1155471, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 267, sourceArmColumn := 51614, finalStart := 1155676, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 268, sourceArmColumn := 51619, finalStart := 1155881, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 269, sourceArmColumn := 51624, finalStart := 1156086, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk14
