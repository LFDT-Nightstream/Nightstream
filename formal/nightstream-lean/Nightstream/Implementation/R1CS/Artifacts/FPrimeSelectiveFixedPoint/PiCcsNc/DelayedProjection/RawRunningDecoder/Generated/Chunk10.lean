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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk10

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 9, logicalColumn := 90, sourceArmColumn := 42448, finalStart := 842518, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 91, sourceArmColumn := 42453, finalStart := 842723, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 92, sourceArmColumn := 42458, finalStart := 842928, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 93, sourceArmColumn := 42463, finalStart := 843133, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 94, sourceArmColumn := 42468, finalStart := 843338, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 95, sourceArmColumn := 42473, finalStart := 843543, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 96, sourceArmColumn := 42478, finalStart := 843748, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 97, sourceArmColumn := 42483, finalStart := 843953, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 98, sourceArmColumn := 42488, finalStart := 844158, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 99, sourceArmColumn := 42493, finalStart := 844363, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 100, sourceArmColumn := 42498, finalStart := 844568, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 101, sourceArmColumn := 42503, finalStart := 844773, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 102, sourceArmColumn := 42508, finalStart := 844978, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 103, sourceArmColumn := 42513, finalStart := 845183, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 104, sourceArmColumn := 42518, finalStart := 845388, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 105, sourceArmColumn := 42523, finalStart := 845593, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 106, sourceArmColumn := 42528, finalStart := 845798, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 107, sourceArmColumn := 42533, finalStart := 846003, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 108, sourceArmColumn := 42269, finalStart := 835179, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 109, sourceArmColumn := 42274, finalStart := 835384, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 110, sourceArmColumn := 42279, finalStart := 835589, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 111, sourceArmColumn := 42284, finalStart := 835794, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 112, sourceArmColumn := 42289, finalStart := 835999, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 113, sourceArmColumn := 42294, finalStart := 836204, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 114, sourceArmColumn := 42299, finalStart := 836409, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 115, sourceArmColumn := 42304, finalStart := 836614, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 116, sourceArmColumn := 42309, finalStart := 836819, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 117, sourceArmColumn := 42314, finalStart := 837024, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 118, sourceArmColumn := 42319, finalStart := 837229, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 119, sourceArmColumn := 42324, finalStart := 837434, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 120, sourceArmColumn := 42329, finalStart := 837639, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 121, sourceArmColumn := 42334, finalStart := 837844, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 122, sourceArmColumn := 42339, finalStart := 838049, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 123, sourceArmColumn := 42344, finalStart := 838254, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 124, sourceArmColumn := 42349, finalStart := 838459, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 125, sourceArmColumn := 42354, finalStart := 838664, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 126, sourceArmColumn := 42359, finalStart := 838869, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 127, sourceArmColumn := 42364, finalStart := 839074, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 128, sourceArmColumn := 42369, finalStart := 839279, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 129, sourceArmColumn := 42374, finalStart := 839484, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 130, sourceArmColumn := 42379, finalStart := 839689, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 131, sourceArmColumn := 42384, finalStart := 839894, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 132, sourceArmColumn := 42389, finalStart := 840099, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 133, sourceArmColumn := 42394, finalStart := 840304, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 134, sourceArmColumn := 42399, finalStart := 840509, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 135, sourceArmColumn := 42404, finalStart := 840714, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 136, sourceArmColumn := 42409, finalStart := 840919, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 137, sourceArmColumn := 42414, finalStart := 841124, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 138, sourceArmColumn := 42419, finalStart := 841329, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 139, sourceArmColumn := 42424, finalStart := 841534, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 140, sourceArmColumn := 42429, finalStart := 841739, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 141, sourceArmColumn := 42434, finalStart := 841944, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 142, sourceArmColumn := 42439, finalStart := 842149, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 143, sourceArmColumn := 42444, finalStart := 842354, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 144, sourceArmColumn := 42449, finalStart := 842559, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 145, sourceArmColumn := 42454, finalStart := 842764, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 146, sourceArmColumn := 42459, finalStart := 842969, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 147, sourceArmColumn := 42464, finalStart := 843174, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 148, sourceArmColumn := 42469, finalStart := 843379, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 149, sourceArmColumn := 42474, finalStart := 843584, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 150, sourceArmColumn := 42479, finalStart := 843789, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 151, sourceArmColumn := 42484, finalStart := 843994, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 152, sourceArmColumn := 42489, finalStart := 844199, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 153, sourceArmColumn := 42494, finalStart := 844404, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 154, sourceArmColumn := 42499, finalStart := 844609, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 155, sourceArmColumn := 42504, finalStart := 844814, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 156, sourceArmColumn := 42509, finalStart := 845019, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 157, sourceArmColumn := 42514, finalStart := 845224, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 158, sourceArmColumn := 42519, finalStart := 845429, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 159, sourceArmColumn := 42524, finalStart := 845634, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 160, sourceArmColumn := 42529, finalStart := 845839, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 161, sourceArmColumn := 42534, finalStart := 846044, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 162, sourceArmColumn := 42270, finalStart := 835220, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 163, sourceArmColumn := 42275, finalStart := 835425, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 164, sourceArmColumn := 42280, finalStart := 835630, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 165, sourceArmColumn := 42285, finalStart := 835835, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 166, sourceArmColumn := 42290, finalStart := 836040, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 167, sourceArmColumn := 42295, finalStart := 836245, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 168, sourceArmColumn := 42300, finalStart := 836450, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 169, sourceArmColumn := 42305, finalStart := 836655, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 170, sourceArmColumn := 42310, finalStart := 836860, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 171, sourceArmColumn := 42315, finalStart := 837065, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 172, sourceArmColumn := 42320, finalStart := 837270, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 173, sourceArmColumn := 42325, finalStart := 837475, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 174, sourceArmColumn := 42330, finalStart := 837680, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 175, sourceArmColumn := 42335, finalStart := 837885, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 176, sourceArmColumn := 42340, finalStart := 838090, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 177, sourceArmColumn := 42345, finalStart := 838295, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 178, sourceArmColumn := 42350, finalStart := 838500, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 179, sourceArmColumn := 42355, finalStart := 838705, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 180, sourceArmColumn := 42360, finalStart := 838910, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 181, sourceArmColumn := 42365, finalStart := 839115, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 182, sourceArmColumn := 42370, finalStart := 839320, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 183, sourceArmColumn := 42375, finalStart := 839525, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 184, sourceArmColumn := 42380, finalStart := 839730, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 185, sourceArmColumn := 42385, finalStart := 839935, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 186, sourceArmColumn := 42390, finalStart := 840140, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 187, sourceArmColumn := 42395, finalStart := 840345, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 188, sourceArmColumn := 42400, finalStart := 840550, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 189, sourceArmColumn := 42405, finalStart := 840755, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 190, sourceArmColumn := 42410, finalStart := 840960, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 191, sourceArmColumn := 42415, finalStart := 841165, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 192, sourceArmColumn := 42420, finalStart := 841370, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 193, sourceArmColumn := 42425, finalStart := 841575, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 194, sourceArmColumn := 42430, finalStart := 841780, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 195, sourceArmColumn := 42435, finalStart := 841985, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 196, sourceArmColumn := 42440, finalStart := 842190, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 197, sourceArmColumn := 42445, finalStart := 842395, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 198, sourceArmColumn := 42450, finalStart := 842600, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 199, sourceArmColumn := 42455, finalStart := 842805, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 200, sourceArmColumn := 42460, finalStart := 843010, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 201, sourceArmColumn := 42465, finalStart := 843215, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 202, sourceArmColumn := 42470, finalStart := 843420, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 203, sourceArmColumn := 42475, finalStart := 843625, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 204, sourceArmColumn := 42480, finalStart := 843830, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 205, sourceArmColumn := 42485, finalStart := 844035, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 206, sourceArmColumn := 42490, finalStart := 844240, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 207, sourceArmColumn := 42495, finalStart := 844445, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 208, sourceArmColumn := 42500, finalStart := 844650, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 209, sourceArmColumn := 42505, finalStart := 844855, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 210, sourceArmColumn := 42510, finalStart := 845060, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 211, sourceArmColumn := 42515, finalStart := 845265, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 212, sourceArmColumn := 42520, finalStart := 845470, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 213, sourceArmColumn := 42525, finalStart := 845675, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 214, sourceArmColumn := 42530, finalStart := 845880, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 215, sourceArmColumn := 42535, finalStart := 846085, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 216, sourceArmColumn := 42271, finalStart := 835261, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 217, sourceArmColumn := 42276, finalStart := 835466, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 218, sourceArmColumn := 42281, finalStart := 835671, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 219, sourceArmColumn := 42286, finalStart := 835876, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 220, sourceArmColumn := 42291, finalStart := 836081, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 221, sourceArmColumn := 42296, finalStart := 836286, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 222, sourceArmColumn := 42301, finalStart := 836491, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 223, sourceArmColumn := 42306, finalStart := 836696, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 224, sourceArmColumn := 42311, finalStart := 836901, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 225, sourceArmColumn := 42316, finalStart := 837106, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 226, sourceArmColumn := 42321, finalStart := 837311, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 227, sourceArmColumn := 42326, finalStart := 837516, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 228, sourceArmColumn := 42331, finalStart := 837721, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 229, sourceArmColumn := 42336, finalStart := 837926, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 230, sourceArmColumn := 42341, finalStart := 838131, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 231, sourceArmColumn := 42346, finalStart := 838336, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 232, sourceArmColumn := 42351, finalStart := 838541, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 233, sourceArmColumn := 42356, finalStart := 838746, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 234, sourceArmColumn := 42361, finalStart := 838951, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 235, sourceArmColumn := 42366, finalStart := 839156, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 236, sourceArmColumn := 42371, finalStart := 839361, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 237, sourceArmColumn := 42376, finalStart := 839566, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 238, sourceArmColumn := 42381, finalStart := 839771, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 239, sourceArmColumn := 42386, finalStart := 839976, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 240, sourceArmColumn := 42391, finalStart := 840181, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 241, sourceArmColumn := 42396, finalStart := 840386, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 242, sourceArmColumn := 42401, finalStart := 840591, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 243, sourceArmColumn := 42406, finalStart := 840796, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 244, sourceArmColumn := 42411, finalStart := 841001, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 245, sourceArmColumn := 42416, finalStart := 841206, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 246, sourceArmColumn := 42421, finalStart := 841411, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 247, sourceArmColumn := 42426, finalStart := 841616, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 248, sourceArmColumn := 42431, finalStart := 841821, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 249, sourceArmColumn := 42436, finalStart := 842026, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 250, sourceArmColumn := 42441, finalStart := 842231, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 251, sourceArmColumn := 42446, finalStart := 842436, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 252, sourceArmColumn := 42451, finalStart := 842641, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 253, sourceArmColumn := 42456, finalStart := 842846, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 254, sourceArmColumn := 42461, finalStart := 843051, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 255, sourceArmColumn := 42466, finalStart := 843256, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 256, sourceArmColumn := 42471, finalStart := 843461, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 257, sourceArmColumn := 42476, finalStart := 843666, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 258, sourceArmColumn := 42481, finalStart := 843871, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 259, sourceArmColumn := 42486, finalStart := 844076, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 260, sourceArmColumn := 42491, finalStart := 844281, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 261, sourceArmColumn := 42496, finalStart := 844486, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 262, sourceArmColumn := 42501, finalStart := 844691, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 263, sourceArmColumn := 42506, finalStart := 844896, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 264, sourceArmColumn := 42511, finalStart := 845101, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 265, sourceArmColumn := 42516, finalStart := 845306, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 266, sourceArmColumn := 42521, finalStart := 845511, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 267, sourceArmColumn := 42526, finalStart := 845716, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 268, sourceArmColumn := 42531, finalStart := 845921, width := 41, encoding := .balancedTernary }
, { child := 9, logicalColumn := 269, sourceArmColumn := 42536, finalStart := 846126, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 0, sourceArmColumn := 44539, finalStart := 912587, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 1, sourceArmColumn := 44544, finalStart := 912792, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 2, sourceArmColumn := 44549, finalStart := 912997, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 3, sourceArmColumn := 44554, finalStart := 913202, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 4, sourceArmColumn := 44559, finalStart := 913407, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 5, sourceArmColumn := 44564, finalStart := 913612, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 6, sourceArmColumn := 44569, finalStart := 913817, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 7, sourceArmColumn := 44574, finalStart := 914022, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 8, sourceArmColumn := 44579, finalStart := 914227, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 9, sourceArmColumn := 44584, finalStart := 914432, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 10, sourceArmColumn := 44589, finalStart := 914637, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 11, sourceArmColumn := 44594, finalStart := 914842, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 12, sourceArmColumn := 44599, finalStart := 915047, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 13, sourceArmColumn := 44604, finalStart := 915252, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 14, sourceArmColumn := 44609, finalStart := 915457, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 15, sourceArmColumn := 44614, finalStart := 915662, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 16, sourceArmColumn := 44619, finalStart := 915867, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 17, sourceArmColumn := 44624, finalStart := 916072, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 18, sourceArmColumn := 44629, finalStart := 916277, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 19, sourceArmColumn := 44634, finalStart := 916482, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 20, sourceArmColumn := 44639, finalStart := 916687, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 21, sourceArmColumn := 44644, finalStart := 916892, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 22, sourceArmColumn := 44649, finalStart := 917097, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 23, sourceArmColumn := 44654, finalStart := 917302, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 24, sourceArmColumn := 44659, finalStart := 917507, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 25, sourceArmColumn := 44664, finalStart := 917712, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 26, sourceArmColumn := 44669, finalStart := 917917, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 27, sourceArmColumn := 44674, finalStart := 918122, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 28, sourceArmColumn := 44679, finalStart := 918327, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 29, sourceArmColumn := 44684, finalStart := 918532, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 30, sourceArmColumn := 44689, finalStart := 918737, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 31, sourceArmColumn := 44694, finalStart := 918942, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 32, sourceArmColumn := 44699, finalStart := 919147, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 33, sourceArmColumn := 44704, finalStart := 919352, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 34, sourceArmColumn := 44709, finalStart := 919557, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 35, sourceArmColumn := 44714, finalStart := 919762, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 36, sourceArmColumn := 44719, finalStart := 919967, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 37, sourceArmColumn := 44724, finalStart := 920172, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 38, sourceArmColumn := 44729, finalStart := 920377, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 39, sourceArmColumn := 44734, finalStart := 920582, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 40, sourceArmColumn := 44739, finalStart := 920787, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 41, sourceArmColumn := 44744, finalStart := 920992, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 42, sourceArmColumn := 44749, finalStart := 921197, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 43, sourceArmColumn := 44754, finalStart := 921402, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 44, sourceArmColumn := 44759, finalStart := 921607, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 45, sourceArmColumn := 44764, finalStart := 921812, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 46, sourceArmColumn := 44769, finalStart := 922017, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 47, sourceArmColumn := 44774, finalStart := 922222, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 48, sourceArmColumn := 44779, finalStart := 922427, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 49, sourceArmColumn := 44784, finalStart := 922632, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 50, sourceArmColumn := 44789, finalStart := 922837, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 51, sourceArmColumn := 44794, finalStart := 923042, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 52, sourceArmColumn := 44799, finalStart := 923247, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 53, sourceArmColumn := 44804, finalStart := 923452, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 54, sourceArmColumn := 44540, finalStart := 912628, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 55, sourceArmColumn := 44545, finalStart := 912833, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 56, sourceArmColumn := 44550, finalStart := 913038, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 57, sourceArmColumn := 44555, finalStart := 913243, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 58, sourceArmColumn := 44560, finalStart := 913448, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 59, sourceArmColumn := 44565, finalStart := 913653, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 60, sourceArmColumn := 44570, finalStart := 913858, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 61, sourceArmColumn := 44575, finalStart := 914063, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 62, sourceArmColumn := 44580, finalStart := 914268, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 63, sourceArmColumn := 44585, finalStart := 914473, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 64, sourceArmColumn := 44590, finalStart := 914678, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 65, sourceArmColumn := 44595, finalStart := 914883, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 66, sourceArmColumn := 44600, finalStart := 915088, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 67, sourceArmColumn := 44605, finalStart := 915293, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 68, sourceArmColumn := 44610, finalStart := 915498, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 69, sourceArmColumn := 44615, finalStart := 915703, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 70, sourceArmColumn := 44620, finalStart := 915908, width := 41, encoding := .balancedTernary }
, { child := 10, logicalColumn := 71, sourceArmColumn := 44625, finalStart := 916113, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk10
