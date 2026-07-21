import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Schema

/-!
Generated file: authoritative raw-running assignment decoder chunk; do not
hand-edit.

Each provenance record carries both the normalized source-arm column and its
final selective-assignment column. The generator fails closed unless the final
column is the exact direct, centered, width-one selective slot for the record's
actual
`running[child].x[(logicalColumn % 54) * x_cols + logicalColumn / 54]` wire.

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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk2

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 1, logicalColumn := 234, sourceArmColumn := 24185, finalColumn := 204471 }
, { child := 1, logicalColumn := 235, sourceArmColumn := 24190, finalColumn := 204476 }
, { child := 1, logicalColumn := 236, sourceArmColumn := 24195, finalColumn := 204481 }
, { child := 1, logicalColumn := 237, sourceArmColumn := 24200, finalColumn := 204486 }
, { child := 1, logicalColumn := 238, sourceArmColumn := 24205, finalColumn := 204491 }
, { child := 1, logicalColumn := 239, sourceArmColumn := 24210, finalColumn := 204496 }
, { child := 1, logicalColumn := 240, sourceArmColumn := 24215, finalColumn := 204501 }
, { child := 1, logicalColumn := 241, sourceArmColumn := 24220, finalColumn := 204506 }
, { child := 1, logicalColumn := 242, sourceArmColumn := 24225, finalColumn := 204511 }
, { child := 1, logicalColumn := 243, sourceArmColumn := 24230, finalColumn := 204516 }
, { child := 1, logicalColumn := 244, sourceArmColumn := 24235, finalColumn := 204521 }
, { child := 1, logicalColumn := 245, sourceArmColumn := 24240, finalColumn := 204526 }
, { child := 1, logicalColumn := 246, sourceArmColumn := 24245, finalColumn := 204531 }
, { child := 1, logicalColumn := 247, sourceArmColumn := 24250, finalColumn := 204536 }
, { child := 1, logicalColumn := 248, sourceArmColumn := 24255, finalColumn := 204541 }
, { child := 1, logicalColumn := 249, sourceArmColumn := 24260, finalColumn := 204546 }
, { child := 1, logicalColumn := 250, sourceArmColumn := 24265, finalColumn := 204551 }
, { child := 1, logicalColumn := 251, sourceArmColumn := 24270, finalColumn := 204556 }
, { child := 1, logicalColumn := 252, sourceArmColumn := 24275, finalColumn := 204561 }
, { child := 1, logicalColumn := 253, sourceArmColumn := 24280, finalColumn := 204566 }
, { child := 1, logicalColumn := 254, sourceArmColumn := 24285, finalColumn := 204571 }
, { child := 1, logicalColumn := 255, sourceArmColumn := 24290, finalColumn := 204576 }
, { child := 1, logicalColumn := 256, sourceArmColumn := 24295, finalColumn := 204581 }
, { child := 1, logicalColumn := 257, sourceArmColumn := 24300, finalColumn := 204586 }
, { child := 1, logicalColumn := 258, sourceArmColumn := 24305, finalColumn := 204591 }
, { child := 1, logicalColumn := 259, sourceArmColumn := 24310, finalColumn := 204596 }
, { child := 1, logicalColumn := 260, sourceArmColumn := 24315, finalColumn := 204601 }
, { child := 1, logicalColumn := 261, sourceArmColumn := 24320, finalColumn := 204606 }
, { child := 1, logicalColumn := 262, sourceArmColumn := 24325, finalColumn := 204611 }
, { child := 1, logicalColumn := 263, sourceArmColumn := 24330, finalColumn := 204616 }
, { child := 1, logicalColumn := 264, sourceArmColumn := 24335, finalColumn := 204621 }
, { child := 1, logicalColumn := 265, sourceArmColumn := 24340, finalColumn := 204626 }
, { child := 1, logicalColumn := 266, sourceArmColumn := 24345, finalColumn := 204631 }
, { child := 1, logicalColumn := 267, sourceArmColumn := 24350, finalColumn := 204636 }
, { child := 1, logicalColumn := 268, sourceArmColumn := 24355, finalColumn := 204641 }
, { child := 1, logicalColumn := 269, sourceArmColumn := 24360, finalColumn := 204646 }
, { child := 2, logicalColumn := 0, sourceArmColumn := 26363, finalColumn := 271067 }
, { child := 2, logicalColumn := 1, sourceArmColumn := 26368, finalColumn := 271072 }
, { child := 2, logicalColumn := 2, sourceArmColumn := 26373, finalColumn := 271077 }
, { child := 2, logicalColumn := 3, sourceArmColumn := 26378, finalColumn := 271082 }
, { child := 2, logicalColumn := 4, sourceArmColumn := 26383, finalColumn := 271087 }
, { child := 2, logicalColumn := 5, sourceArmColumn := 26388, finalColumn := 271092 }
, { child := 2, logicalColumn := 6, sourceArmColumn := 26393, finalColumn := 271097 }
, { child := 2, logicalColumn := 7, sourceArmColumn := 26398, finalColumn := 271102 }
, { child := 2, logicalColumn := 8, sourceArmColumn := 26403, finalColumn := 271107 }
, { child := 2, logicalColumn := 9, sourceArmColumn := 26408, finalColumn := 271112 }
, { child := 2, logicalColumn := 10, sourceArmColumn := 26413, finalColumn := 271117 }
, { child := 2, logicalColumn := 11, sourceArmColumn := 26418, finalColumn := 271122 }
, { child := 2, logicalColumn := 12, sourceArmColumn := 26423, finalColumn := 271127 }
, { child := 2, logicalColumn := 13, sourceArmColumn := 26428, finalColumn := 271132 }
, { child := 2, logicalColumn := 14, sourceArmColumn := 26433, finalColumn := 271137 }
, { child := 2, logicalColumn := 15, sourceArmColumn := 26438, finalColumn := 271142 }
, { child := 2, logicalColumn := 16, sourceArmColumn := 26443, finalColumn := 271147 }
, { child := 2, logicalColumn := 17, sourceArmColumn := 26448, finalColumn := 271152 }
, { child := 2, logicalColumn := 18, sourceArmColumn := 26453, finalColumn := 271157 }
, { child := 2, logicalColumn := 19, sourceArmColumn := 26458, finalColumn := 271162 }
, { child := 2, logicalColumn := 20, sourceArmColumn := 26463, finalColumn := 271167 }
, { child := 2, logicalColumn := 21, sourceArmColumn := 26468, finalColumn := 271172 }
, { child := 2, logicalColumn := 22, sourceArmColumn := 26473, finalColumn := 271177 }
, { child := 2, logicalColumn := 23, sourceArmColumn := 26478, finalColumn := 271182 }
, { child := 2, logicalColumn := 24, sourceArmColumn := 26483, finalColumn := 271187 }
, { child := 2, logicalColumn := 25, sourceArmColumn := 26488, finalColumn := 271192 }
, { child := 2, logicalColumn := 26, sourceArmColumn := 26493, finalColumn := 271197 }
, { child := 2, logicalColumn := 27, sourceArmColumn := 26498, finalColumn := 271202 }
, { child := 2, logicalColumn := 28, sourceArmColumn := 26503, finalColumn := 271207 }
, { child := 2, logicalColumn := 29, sourceArmColumn := 26508, finalColumn := 271212 }
, { child := 2, logicalColumn := 30, sourceArmColumn := 26513, finalColumn := 271217 }
, { child := 2, logicalColumn := 31, sourceArmColumn := 26518, finalColumn := 271222 }
, { child := 2, logicalColumn := 32, sourceArmColumn := 26523, finalColumn := 271227 }
, { child := 2, logicalColumn := 33, sourceArmColumn := 26528, finalColumn := 271232 }
, { child := 2, logicalColumn := 34, sourceArmColumn := 26533, finalColumn := 271237 }
, { child := 2, logicalColumn := 35, sourceArmColumn := 26538, finalColumn := 271242 }
, { child := 2, logicalColumn := 36, sourceArmColumn := 26543, finalColumn := 271247 }
, { child := 2, logicalColumn := 37, sourceArmColumn := 26548, finalColumn := 271252 }
, { child := 2, logicalColumn := 38, sourceArmColumn := 26553, finalColumn := 271257 }
, { child := 2, logicalColumn := 39, sourceArmColumn := 26558, finalColumn := 271262 }
, { child := 2, logicalColumn := 40, sourceArmColumn := 26563, finalColumn := 271267 }
, { child := 2, logicalColumn := 41, sourceArmColumn := 26568, finalColumn := 271272 }
, { child := 2, logicalColumn := 42, sourceArmColumn := 26573, finalColumn := 271277 }
, { child := 2, logicalColumn := 43, sourceArmColumn := 26578, finalColumn := 271282 }
, { child := 2, logicalColumn := 44, sourceArmColumn := 26583, finalColumn := 271287 }
, { child := 2, logicalColumn := 45, sourceArmColumn := 26588, finalColumn := 271292 }
, { child := 2, logicalColumn := 46, sourceArmColumn := 26593, finalColumn := 271297 }
, { child := 2, logicalColumn := 47, sourceArmColumn := 26598, finalColumn := 271302 }
, { child := 2, logicalColumn := 48, sourceArmColumn := 26603, finalColumn := 271307 }
, { child := 2, logicalColumn := 49, sourceArmColumn := 26608, finalColumn := 271312 }
, { child := 2, logicalColumn := 50, sourceArmColumn := 26613, finalColumn := 271317 }
, { child := 2, logicalColumn := 51, sourceArmColumn := 26618, finalColumn := 271322 }
, { child := 2, logicalColumn := 52, sourceArmColumn := 26623, finalColumn := 271327 }
, { child := 2, logicalColumn := 53, sourceArmColumn := 26628, finalColumn := 271332 }
, { child := 2, logicalColumn := 54, sourceArmColumn := 26364, finalColumn := 271068 }
, { child := 2, logicalColumn := 55, sourceArmColumn := 26369, finalColumn := 271073 }
, { child := 2, logicalColumn := 56, sourceArmColumn := 26374, finalColumn := 271078 }
, { child := 2, logicalColumn := 57, sourceArmColumn := 26379, finalColumn := 271083 }
, { child := 2, logicalColumn := 58, sourceArmColumn := 26384, finalColumn := 271088 }
, { child := 2, logicalColumn := 59, sourceArmColumn := 26389, finalColumn := 271093 }
, { child := 2, logicalColumn := 60, sourceArmColumn := 26394, finalColumn := 271098 }
, { child := 2, logicalColumn := 61, sourceArmColumn := 26399, finalColumn := 271103 }
, { child := 2, logicalColumn := 62, sourceArmColumn := 26404, finalColumn := 271108 }
, { child := 2, logicalColumn := 63, sourceArmColumn := 26409, finalColumn := 271113 }
, { child := 2, logicalColumn := 64, sourceArmColumn := 26414, finalColumn := 271118 }
, { child := 2, logicalColumn := 65, sourceArmColumn := 26419, finalColumn := 271123 }
, { child := 2, logicalColumn := 66, sourceArmColumn := 26424, finalColumn := 271128 }
, { child := 2, logicalColumn := 67, sourceArmColumn := 26429, finalColumn := 271133 }
, { child := 2, logicalColumn := 68, sourceArmColumn := 26434, finalColumn := 271138 }
, { child := 2, logicalColumn := 69, sourceArmColumn := 26439, finalColumn := 271143 }
, { child := 2, logicalColumn := 70, sourceArmColumn := 26444, finalColumn := 271148 }
, { child := 2, logicalColumn := 71, sourceArmColumn := 26449, finalColumn := 271153 }
, { child := 2, logicalColumn := 72, sourceArmColumn := 26454, finalColumn := 271158 }
, { child := 2, logicalColumn := 73, sourceArmColumn := 26459, finalColumn := 271163 }
, { child := 2, logicalColumn := 74, sourceArmColumn := 26464, finalColumn := 271168 }
, { child := 2, logicalColumn := 75, sourceArmColumn := 26469, finalColumn := 271173 }
, { child := 2, logicalColumn := 76, sourceArmColumn := 26474, finalColumn := 271178 }
, { child := 2, logicalColumn := 77, sourceArmColumn := 26479, finalColumn := 271183 }
, { child := 2, logicalColumn := 78, sourceArmColumn := 26484, finalColumn := 271188 }
, { child := 2, logicalColumn := 79, sourceArmColumn := 26489, finalColumn := 271193 }
, { child := 2, logicalColumn := 80, sourceArmColumn := 26494, finalColumn := 271198 }
, { child := 2, logicalColumn := 81, sourceArmColumn := 26499, finalColumn := 271203 }
, { child := 2, logicalColumn := 82, sourceArmColumn := 26504, finalColumn := 271208 }
, { child := 2, logicalColumn := 83, sourceArmColumn := 26509, finalColumn := 271213 }
, { child := 2, logicalColumn := 84, sourceArmColumn := 26514, finalColumn := 271218 }
, { child := 2, logicalColumn := 85, sourceArmColumn := 26519, finalColumn := 271223 }
, { child := 2, logicalColumn := 86, sourceArmColumn := 26524, finalColumn := 271228 }
, { child := 2, logicalColumn := 87, sourceArmColumn := 26529, finalColumn := 271233 }
, { child := 2, logicalColumn := 88, sourceArmColumn := 26534, finalColumn := 271238 }
, { child := 2, logicalColumn := 89, sourceArmColumn := 26539, finalColumn := 271243 }
, { child := 2, logicalColumn := 90, sourceArmColumn := 26544, finalColumn := 271248 }
, { child := 2, logicalColumn := 91, sourceArmColumn := 26549, finalColumn := 271253 }
, { child := 2, logicalColumn := 92, sourceArmColumn := 26554, finalColumn := 271258 }
, { child := 2, logicalColumn := 93, sourceArmColumn := 26559, finalColumn := 271263 }
, { child := 2, logicalColumn := 94, sourceArmColumn := 26564, finalColumn := 271268 }
, { child := 2, logicalColumn := 95, sourceArmColumn := 26569, finalColumn := 271273 }
, { child := 2, logicalColumn := 96, sourceArmColumn := 26574, finalColumn := 271278 }
, { child := 2, logicalColumn := 97, sourceArmColumn := 26579, finalColumn := 271283 }
, { child := 2, logicalColumn := 98, sourceArmColumn := 26584, finalColumn := 271288 }
, { child := 2, logicalColumn := 99, sourceArmColumn := 26589, finalColumn := 271293 }
, { child := 2, logicalColumn := 100, sourceArmColumn := 26594, finalColumn := 271298 }
, { child := 2, logicalColumn := 101, sourceArmColumn := 26599, finalColumn := 271303 }
, { child := 2, logicalColumn := 102, sourceArmColumn := 26604, finalColumn := 271308 }
, { child := 2, logicalColumn := 103, sourceArmColumn := 26609, finalColumn := 271313 }
, { child := 2, logicalColumn := 104, sourceArmColumn := 26614, finalColumn := 271318 }
, { child := 2, logicalColumn := 105, sourceArmColumn := 26619, finalColumn := 271323 }
, { child := 2, logicalColumn := 106, sourceArmColumn := 26624, finalColumn := 271328 }
, { child := 2, logicalColumn := 107, sourceArmColumn := 26629, finalColumn := 271333 }
, { child := 2, logicalColumn := 108, sourceArmColumn := 26365, finalColumn := 271069 }
, { child := 2, logicalColumn := 109, sourceArmColumn := 26370, finalColumn := 271074 }
, { child := 2, logicalColumn := 110, sourceArmColumn := 26375, finalColumn := 271079 }
, { child := 2, logicalColumn := 111, sourceArmColumn := 26380, finalColumn := 271084 }
, { child := 2, logicalColumn := 112, sourceArmColumn := 26385, finalColumn := 271089 }
, { child := 2, logicalColumn := 113, sourceArmColumn := 26390, finalColumn := 271094 }
, { child := 2, logicalColumn := 114, sourceArmColumn := 26395, finalColumn := 271099 }
, { child := 2, logicalColumn := 115, sourceArmColumn := 26400, finalColumn := 271104 }
, { child := 2, logicalColumn := 116, sourceArmColumn := 26405, finalColumn := 271109 }
, { child := 2, logicalColumn := 117, sourceArmColumn := 26410, finalColumn := 271114 }
, { child := 2, logicalColumn := 118, sourceArmColumn := 26415, finalColumn := 271119 }
, { child := 2, logicalColumn := 119, sourceArmColumn := 26420, finalColumn := 271124 }
, { child := 2, logicalColumn := 120, sourceArmColumn := 26425, finalColumn := 271129 }
, { child := 2, logicalColumn := 121, sourceArmColumn := 26430, finalColumn := 271134 }
, { child := 2, logicalColumn := 122, sourceArmColumn := 26435, finalColumn := 271139 }
, { child := 2, logicalColumn := 123, sourceArmColumn := 26440, finalColumn := 271144 }
, { child := 2, logicalColumn := 124, sourceArmColumn := 26445, finalColumn := 271149 }
, { child := 2, logicalColumn := 125, sourceArmColumn := 26450, finalColumn := 271154 }
, { child := 2, logicalColumn := 126, sourceArmColumn := 26455, finalColumn := 271159 }
, { child := 2, logicalColumn := 127, sourceArmColumn := 26460, finalColumn := 271164 }
, { child := 2, logicalColumn := 128, sourceArmColumn := 26465, finalColumn := 271169 }
, { child := 2, logicalColumn := 129, sourceArmColumn := 26470, finalColumn := 271174 }
, { child := 2, logicalColumn := 130, sourceArmColumn := 26475, finalColumn := 271179 }
, { child := 2, logicalColumn := 131, sourceArmColumn := 26480, finalColumn := 271184 }
, { child := 2, logicalColumn := 132, sourceArmColumn := 26485, finalColumn := 271189 }
, { child := 2, logicalColumn := 133, sourceArmColumn := 26490, finalColumn := 271194 }
, { child := 2, logicalColumn := 134, sourceArmColumn := 26495, finalColumn := 271199 }
, { child := 2, logicalColumn := 135, sourceArmColumn := 26500, finalColumn := 271204 }
, { child := 2, logicalColumn := 136, sourceArmColumn := 26505, finalColumn := 271209 }
, { child := 2, logicalColumn := 137, sourceArmColumn := 26510, finalColumn := 271214 }
, { child := 2, logicalColumn := 138, sourceArmColumn := 26515, finalColumn := 271219 }
, { child := 2, logicalColumn := 139, sourceArmColumn := 26520, finalColumn := 271224 }
, { child := 2, logicalColumn := 140, sourceArmColumn := 26525, finalColumn := 271229 }
, { child := 2, logicalColumn := 141, sourceArmColumn := 26530, finalColumn := 271234 }
, { child := 2, logicalColumn := 142, sourceArmColumn := 26535, finalColumn := 271239 }
, { child := 2, logicalColumn := 143, sourceArmColumn := 26540, finalColumn := 271244 }
, { child := 2, logicalColumn := 144, sourceArmColumn := 26545, finalColumn := 271249 }
, { child := 2, logicalColumn := 145, sourceArmColumn := 26550, finalColumn := 271254 }
, { child := 2, logicalColumn := 146, sourceArmColumn := 26555, finalColumn := 271259 }
, { child := 2, logicalColumn := 147, sourceArmColumn := 26560, finalColumn := 271264 }
, { child := 2, logicalColumn := 148, sourceArmColumn := 26565, finalColumn := 271269 }
, { child := 2, logicalColumn := 149, sourceArmColumn := 26570, finalColumn := 271274 }
, { child := 2, logicalColumn := 150, sourceArmColumn := 26575, finalColumn := 271279 }
, { child := 2, logicalColumn := 151, sourceArmColumn := 26580, finalColumn := 271284 }
, { child := 2, logicalColumn := 152, sourceArmColumn := 26585, finalColumn := 271289 }
, { child := 2, logicalColumn := 153, sourceArmColumn := 26590, finalColumn := 271294 }
, { child := 2, logicalColumn := 154, sourceArmColumn := 26595, finalColumn := 271299 }
, { child := 2, logicalColumn := 155, sourceArmColumn := 26600, finalColumn := 271304 }
, { child := 2, logicalColumn := 156, sourceArmColumn := 26605, finalColumn := 271309 }
, { child := 2, logicalColumn := 157, sourceArmColumn := 26610, finalColumn := 271314 }
, { child := 2, logicalColumn := 158, sourceArmColumn := 26615, finalColumn := 271319 }
, { child := 2, logicalColumn := 159, sourceArmColumn := 26620, finalColumn := 271324 }
, { child := 2, logicalColumn := 160, sourceArmColumn := 26625, finalColumn := 271329 }
, { child := 2, logicalColumn := 161, sourceArmColumn := 26630, finalColumn := 271334 }
, { child := 2, logicalColumn := 162, sourceArmColumn := 26366, finalColumn := 271070 }
, { child := 2, logicalColumn := 163, sourceArmColumn := 26371, finalColumn := 271075 }
, { child := 2, logicalColumn := 164, sourceArmColumn := 26376, finalColumn := 271080 }
, { child := 2, logicalColumn := 165, sourceArmColumn := 26381, finalColumn := 271085 }
, { child := 2, logicalColumn := 166, sourceArmColumn := 26386, finalColumn := 271090 }
, { child := 2, logicalColumn := 167, sourceArmColumn := 26391, finalColumn := 271095 }
, { child := 2, logicalColumn := 168, sourceArmColumn := 26396, finalColumn := 271100 }
, { child := 2, logicalColumn := 169, sourceArmColumn := 26401, finalColumn := 271105 }
, { child := 2, logicalColumn := 170, sourceArmColumn := 26406, finalColumn := 271110 }
, { child := 2, logicalColumn := 171, sourceArmColumn := 26411, finalColumn := 271115 }
, { child := 2, logicalColumn := 172, sourceArmColumn := 26416, finalColumn := 271120 }
, { child := 2, logicalColumn := 173, sourceArmColumn := 26421, finalColumn := 271125 }
, { child := 2, logicalColumn := 174, sourceArmColumn := 26426, finalColumn := 271130 }
, { child := 2, logicalColumn := 175, sourceArmColumn := 26431, finalColumn := 271135 }
, { child := 2, logicalColumn := 176, sourceArmColumn := 26436, finalColumn := 271140 }
, { child := 2, logicalColumn := 177, sourceArmColumn := 26441, finalColumn := 271145 }
, { child := 2, logicalColumn := 178, sourceArmColumn := 26446, finalColumn := 271150 }
, { child := 2, logicalColumn := 179, sourceArmColumn := 26451, finalColumn := 271155 }
, { child := 2, logicalColumn := 180, sourceArmColumn := 26456, finalColumn := 271160 }
, { child := 2, logicalColumn := 181, sourceArmColumn := 26461, finalColumn := 271165 }
, { child := 2, logicalColumn := 182, sourceArmColumn := 26466, finalColumn := 271170 }
, { child := 2, logicalColumn := 183, sourceArmColumn := 26471, finalColumn := 271175 }
, { child := 2, logicalColumn := 184, sourceArmColumn := 26476, finalColumn := 271180 }
, { child := 2, logicalColumn := 185, sourceArmColumn := 26481, finalColumn := 271185 }
, { child := 2, logicalColumn := 186, sourceArmColumn := 26486, finalColumn := 271190 }
, { child := 2, logicalColumn := 187, sourceArmColumn := 26491, finalColumn := 271195 }
, { child := 2, logicalColumn := 188, sourceArmColumn := 26496, finalColumn := 271200 }
, { child := 2, logicalColumn := 189, sourceArmColumn := 26501, finalColumn := 271205 }
, { child := 2, logicalColumn := 190, sourceArmColumn := 26506, finalColumn := 271210 }
, { child := 2, logicalColumn := 191, sourceArmColumn := 26511, finalColumn := 271215 }
, { child := 2, logicalColumn := 192, sourceArmColumn := 26516, finalColumn := 271220 }
, { child := 2, logicalColumn := 193, sourceArmColumn := 26521, finalColumn := 271225 }
, { child := 2, logicalColumn := 194, sourceArmColumn := 26526, finalColumn := 271230 }
, { child := 2, logicalColumn := 195, sourceArmColumn := 26531, finalColumn := 271235 }
, { child := 2, logicalColumn := 196, sourceArmColumn := 26536, finalColumn := 271240 }
, { child := 2, logicalColumn := 197, sourceArmColumn := 26541, finalColumn := 271245 }
, { child := 2, logicalColumn := 198, sourceArmColumn := 26546, finalColumn := 271250 }
, { child := 2, logicalColumn := 199, sourceArmColumn := 26551, finalColumn := 271255 }
, { child := 2, logicalColumn := 200, sourceArmColumn := 26556, finalColumn := 271260 }
, { child := 2, logicalColumn := 201, sourceArmColumn := 26561, finalColumn := 271265 }
, { child := 2, logicalColumn := 202, sourceArmColumn := 26566, finalColumn := 271270 }
, { child := 2, logicalColumn := 203, sourceArmColumn := 26571, finalColumn := 271275 }
, { child := 2, logicalColumn := 204, sourceArmColumn := 26576, finalColumn := 271280 }
, { child := 2, logicalColumn := 205, sourceArmColumn := 26581, finalColumn := 271285 }
, { child := 2, logicalColumn := 206, sourceArmColumn := 26586, finalColumn := 271290 }
, { child := 2, logicalColumn := 207, sourceArmColumn := 26591, finalColumn := 271295 }
, { child := 2, logicalColumn := 208, sourceArmColumn := 26596, finalColumn := 271300 }
, { child := 2, logicalColumn := 209, sourceArmColumn := 26601, finalColumn := 271305 }
, { child := 2, logicalColumn := 210, sourceArmColumn := 26606, finalColumn := 271310 }
, { child := 2, logicalColumn := 211, sourceArmColumn := 26611, finalColumn := 271315 }
, { child := 2, logicalColumn := 212, sourceArmColumn := 26616, finalColumn := 271320 }
, { child := 2, logicalColumn := 213, sourceArmColumn := 26621, finalColumn := 271325 }
, { child := 2, logicalColumn := 214, sourceArmColumn := 26626, finalColumn := 271330 }
, { child := 2, logicalColumn := 215, sourceArmColumn := 26631, finalColumn := 271335 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk2
