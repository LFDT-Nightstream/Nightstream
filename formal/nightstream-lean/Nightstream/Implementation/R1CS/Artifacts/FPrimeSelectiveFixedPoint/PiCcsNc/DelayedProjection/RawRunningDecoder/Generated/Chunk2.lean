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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk2

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 1, logicalColumn := 234, sourceArmColumn := 24185, finalStart := 219031, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 235, sourceArmColumn := 24190, finalStart := 219236, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 236, sourceArmColumn := 24195, finalStart := 219441, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 237, sourceArmColumn := 24200, finalStart := 219646, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 238, sourceArmColumn := 24205, finalStart := 219851, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 239, sourceArmColumn := 24210, finalStart := 220056, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 240, sourceArmColumn := 24215, finalStart := 220261, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 241, sourceArmColumn := 24220, finalStart := 220466, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 242, sourceArmColumn := 24225, finalStart := 220671, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 243, sourceArmColumn := 24230, finalStart := 220876, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 244, sourceArmColumn := 24235, finalStart := 221081, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 245, sourceArmColumn := 24240, finalStart := 221286, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 246, sourceArmColumn := 24245, finalStart := 221491, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 247, sourceArmColumn := 24250, finalStart := 221696, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 248, sourceArmColumn := 24255, finalStart := 221901, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 249, sourceArmColumn := 24260, finalStart := 222106, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 250, sourceArmColumn := 24265, finalStart := 222311, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 251, sourceArmColumn := 24270, finalStart := 222516, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 252, sourceArmColumn := 24275, finalStart := 222721, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 253, sourceArmColumn := 24280, finalStart := 222926, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 254, sourceArmColumn := 24285, finalStart := 223131, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 255, sourceArmColumn := 24290, finalStart := 223336, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 256, sourceArmColumn := 24295, finalStart := 223541, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 257, sourceArmColumn := 24300, finalStart := 223746, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 258, sourceArmColumn := 24305, finalStart := 223951, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 259, sourceArmColumn := 24310, finalStart := 224156, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 260, sourceArmColumn := 24315, finalStart := 224361, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 261, sourceArmColumn := 24320, finalStart := 224566, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 262, sourceArmColumn := 24325, finalStart := 224771, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 263, sourceArmColumn := 24330, finalStart := 224976, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 264, sourceArmColumn := 24335, finalStart := 225181, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 265, sourceArmColumn := 24340, finalStart := 225386, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 266, sourceArmColumn := 24345, finalStart := 225591, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 267, sourceArmColumn := 24350, finalStart := 225796, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 268, sourceArmColumn := 24355, finalStart := 226001, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 269, sourceArmColumn := 24360, finalStart := 226206, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 0, sourceArmColumn := 26363, finalStart := 292667, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 1, sourceArmColumn := 26368, finalStart := 292872, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 2, sourceArmColumn := 26373, finalStart := 293077, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 3, sourceArmColumn := 26378, finalStart := 293282, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 4, sourceArmColumn := 26383, finalStart := 293487, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 5, sourceArmColumn := 26388, finalStart := 293692, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 6, sourceArmColumn := 26393, finalStart := 293897, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 7, sourceArmColumn := 26398, finalStart := 294102, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 8, sourceArmColumn := 26403, finalStart := 294307, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 9, sourceArmColumn := 26408, finalStart := 294512, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 10, sourceArmColumn := 26413, finalStart := 294717, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 11, sourceArmColumn := 26418, finalStart := 294922, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 12, sourceArmColumn := 26423, finalStart := 295127, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 13, sourceArmColumn := 26428, finalStart := 295332, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 14, sourceArmColumn := 26433, finalStart := 295537, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 15, sourceArmColumn := 26438, finalStart := 295742, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 16, sourceArmColumn := 26443, finalStart := 295947, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 17, sourceArmColumn := 26448, finalStart := 296152, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 18, sourceArmColumn := 26453, finalStart := 296357, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 19, sourceArmColumn := 26458, finalStart := 296562, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 20, sourceArmColumn := 26463, finalStart := 296767, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 21, sourceArmColumn := 26468, finalStart := 296972, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 22, sourceArmColumn := 26473, finalStart := 297177, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 23, sourceArmColumn := 26478, finalStart := 297382, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 24, sourceArmColumn := 26483, finalStart := 297587, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 25, sourceArmColumn := 26488, finalStart := 297792, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 26, sourceArmColumn := 26493, finalStart := 297997, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 27, sourceArmColumn := 26498, finalStart := 298202, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 28, sourceArmColumn := 26503, finalStart := 298407, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 29, sourceArmColumn := 26508, finalStart := 298612, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 30, sourceArmColumn := 26513, finalStart := 298817, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 31, sourceArmColumn := 26518, finalStart := 299022, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 32, sourceArmColumn := 26523, finalStart := 299227, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 33, sourceArmColumn := 26528, finalStart := 299432, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 34, sourceArmColumn := 26533, finalStart := 299637, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 35, sourceArmColumn := 26538, finalStart := 299842, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 36, sourceArmColumn := 26543, finalStart := 300047, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 37, sourceArmColumn := 26548, finalStart := 300252, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 38, sourceArmColumn := 26553, finalStart := 300457, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 39, sourceArmColumn := 26558, finalStart := 300662, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 40, sourceArmColumn := 26563, finalStart := 300867, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 41, sourceArmColumn := 26568, finalStart := 301072, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 42, sourceArmColumn := 26573, finalStart := 301277, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 43, sourceArmColumn := 26578, finalStart := 301482, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 44, sourceArmColumn := 26583, finalStart := 301687, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 45, sourceArmColumn := 26588, finalStart := 301892, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 46, sourceArmColumn := 26593, finalStart := 302097, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 47, sourceArmColumn := 26598, finalStart := 302302, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 48, sourceArmColumn := 26603, finalStart := 302507, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 49, sourceArmColumn := 26608, finalStart := 302712, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 50, sourceArmColumn := 26613, finalStart := 302917, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 51, sourceArmColumn := 26618, finalStart := 303122, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 52, sourceArmColumn := 26623, finalStart := 303327, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 53, sourceArmColumn := 26628, finalStart := 303532, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 54, sourceArmColumn := 26364, finalStart := 292708, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 55, sourceArmColumn := 26369, finalStart := 292913, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 56, sourceArmColumn := 26374, finalStart := 293118, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 57, sourceArmColumn := 26379, finalStart := 293323, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 58, sourceArmColumn := 26384, finalStart := 293528, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 59, sourceArmColumn := 26389, finalStart := 293733, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 60, sourceArmColumn := 26394, finalStart := 293938, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 61, sourceArmColumn := 26399, finalStart := 294143, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 62, sourceArmColumn := 26404, finalStart := 294348, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 63, sourceArmColumn := 26409, finalStart := 294553, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 64, sourceArmColumn := 26414, finalStart := 294758, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 65, sourceArmColumn := 26419, finalStart := 294963, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 66, sourceArmColumn := 26424, finalStart := 295168, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 67, sourceArmColumn := 26429, finalStart := 295373, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 68, sourceArmColumn := 26434, finalStart := 295578, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 69, sourceArmColumn := 26439, finalStart := 295783, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 70, sourceArmColumn := 26444, finalStart := 295988, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 71, sourceArmColumn := 26449, finalStart := 296193, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 72, sourceArmColumn := 26454, finalStart := 296398, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 73, sourceArmColumn := 26459, finalStart := 296603, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 74, sourceArmColumn := 26464, finalStart := 296808, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 75, sourceArmColumn := 26469, finalStart := 297013, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 76, sourceArmColumn := 26474, finalStart := 297218, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 77, sourceArmColumn := 26479, finalStart := 297423, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 78, sourceArmColumn := 26484, finalStart := 297628, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 79, sourceArmColumn := 26489, finalStart := 297833, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 80, sourceArmColumn := 26494, finalStart := 298038, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 81, sourceArmColumn := 26499, finalStart := 298243, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 82, sourceArmColumn := 26504, finalStart := 298448, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 83, sourceArmColumn := 26509, finalStart := 298653, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 84, sourceArmColumn := 26514, finalStart := 298858, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 85, sourceArmColumn := 26519, finalStart := 299063, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 86, sourceArmColumn := 26524, finalStart := 299268, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 87, sourceArmColumn := 26529, finalStart := 299473, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 88, sourceArmColumn := 26534, finalStart := 299678, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 89, sourceArmColumn := 26539, finalStart := 299883, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 90, sourceArmColumn := 26544, finalStart := 300088, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 91, sourceArmColumn := 26549, finalStart := 300293, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 92, sourceArmColumn := 26554, finalStart := 300498, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 93, sourceArmColumn := 26559, finalStart := 300703, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 94, sourceArmColumn := 26564, finalStart := 300908, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 95, sourceArmColumn := 26569, finalStart := 301113, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 96, sourceArmColumn := 26574, finalStart := 301318, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 97, sourceArmColumn := 26579, finalStart := 301523, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 98, sourceArmColumn := 26584, finalStart := 301728, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 99, sourceArmColumn := 26589, finalStart := 301933, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 100, sourceArmColumn := 26594, finalStart := 302138, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 101, sourceArmColumn := 26599, finalStart := 302343, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 102, sourceArmColumn := 26604, finalStart := 302548, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 103, sourceArmColumn := 26609, finalStart := 302753, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 104, sourceArmColumn := 26614, finalStart := 302958, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 105, sourceArmColumn := 26619, finalStart := 303163, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 106, sourceArmColumn := 26624, finalStart := 303368, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 107, sourceArmColumn := 26629, finalStart := 303573, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 108, sourceArmColumn := 26365, finalStart := 292749, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 109, sourceArmColumn := 26370, finalStart := 292954, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 110, sourceArmColumn := 26375, finalStart := 293159, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 111, sourceArmColumn := 26380, finalStart := 293364, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 112, sourceArmColumn := 26385, finalStart := 293569, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 113, sourceArmColumn := 26390, finalStart := 293774, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 114, sourceArmColumn := 26395, finalStart := 293979, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 115, sourceArmColumn := 26400, finalStart := 294184, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 116, sourceArmColumn := 26405, finalStart := 294389, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 117, sourceArmColumn := 26410, finalStart := 294594, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 118, sourceArmColumn := 26415, finalStart := 294799, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 119, sourceArmColumn := 26420, finalStart := 295004, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 120, sourceArmColumn := 26425, finalStart := 295209, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 121, sourceArmColumn := 26430, finalStart := 295414, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 122, sourceArmColumn := 26435, finalStart := 295619, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 123, sourceArmColumn := 26440, finalStart := 295824, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 124, sourceArmColumn := 26445, finalStart := 296029, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 125, sourceArmColumn := 26450, finalStart := 296234, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 126, sourceArmColumn := 26455, finalStart := 296439, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 127, sourceArmColumn := 26460, finalStart := 296644, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 128, sourceArmColumn := 26465, finalStart := 296849, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 129, sourceArmColumn := 26470, finalStart := 297054, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 130, sourceArmColumn := 26475, finalStart := 297259, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 131, sourceArmColumn := 26480, finalStart := 297464, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 132, sourceArmColumn := 26485, finalStart := 297669, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 133, sourceArmColumn := 26490, finalStart := 297874, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 134, sourceArmColumn := 26495, finalStart := 298079, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 135, sourceArmColumn := 26500, finalStart := 298284, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 136, sourceArmColumn := 26505, finalStart := 298489, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 137, sourceArmColumn := 26510, finalStart := 298694, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 138, sourceArmColumn := 26515, finalStart := 298899, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 139, sourceArmColumn := 26520, finalStart := 299104, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 140, sourceArmColumn := 26525, finalStart := 299309, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 141, sourceArmColumn := 26530, finalStart := 299514, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 142, sourceArmColumn := 26535, finalStart := 299719, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 143, sourceArmColumn := 26540, finalStart := 299924, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 144, sourceArmColumn := 26545, finalStart := 300129, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 145, sourceArmColumn := 26550, finalStart := 300334, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 146, sourceArmColumn := 26555, finalStart := 300539, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 147, sourceArmColumn := 26560, finalStart := 300744, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 148, sourceArmColumn := 26565, finalStart := 300949, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 149, sourceArmColumn := 26570, finalStart := 301154, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 150, sourceArmColumn := 26575, finalStart := 301359, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 151, sourceArmColumn := 26580, finalStart := 301564, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 152, sourceArmColumn := 26585, finalStart := 301769, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 153, sourceArmColumn := 26590, finalStart := 301974, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 154, sourceArmColumn := 26595, finalStart := 302179, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 155, sourceArmColumn := 26600, finalStart := 302384, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 156, sourceArmColumn := 26605, finalStart := 302589, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 157, sourceArmColumn := 26610, finalStart := 302794, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 158, sourceArmColumn := 26615, finalStart := 302999, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 159, sourceArmColumn := 26620, finalStart := 303204, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 160, sourceArmColumn := 26625, finalStart := 303409, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 161, sourceArmColumn := 26630, finalStart := 303614, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 162, sourceArmColumn := 26366, finalStart := 292790, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 163, sourceArmColumn := 26371, finalStart := 292995, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 164, sourceArmColumn := 26376, finalStart := 293200, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 165, sourceArmColumn := 26381, finalStart := 293405, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 166, sourceArmColumn := 26386, finalStart := 293610, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 167, sourceArmColumn := 26391, finalStart := 293815, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 168, sourceArmColumn := 26396, finalStart := 294020, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 169, sourceArmColumn := 26401, finalStart := 294225, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 170, sourceArmColumn := 26406, finalStart := 294430, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 171, sourceArmColumn := 26411, finalStart := 294635, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 172, sourceArmColumn := 26416, finalStart := 294840, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 173, sourceArmColumn := 26421, finalStart := 295045, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 174, sourceArmColumn := 26426, finalStart := 295250, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 175, sourceArmColumn := 26431, finalStart := 295455, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 176, sourceArmColumn := 26436, finalStart := 295660, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 177, sourceArmColumn := 26441, finalStart := 295865, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 178, sourceArmColumn := 26446, finalStart := 296070, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 179, sourceArmColumn := 26451, finalStart := 296275, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 180, sourceArmColumn := 26456, finalStart := 296480, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 181, sourceArmColumn := 26461, finalStart := 296685, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 182, sourceArmColumn := 26466, finalStart := 296890, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 183, sourceArmColumn := 26471, finalStart := 297095, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 184, sourceArmColumn := 26476, finalStart := 297300, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 185, sourceArmColumn := 26481, finalStart := 297505, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 186, sourceArmColumn := 26486, finalStart := 297710, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 187, sourceArmColumn := 26491, finalStart := 297915, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 188, sourceArmColumn := 26496, finalStart := 298120, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 189, sourceArmColumn := 26501, finalStart := 298325, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 190, sourceArmColumn := 26506, finalStart := 298530, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 191, sourceArmColumn := 26511, finalStart := 298735, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 192, sourceArmColumn := 26516, finalStart := 298940, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 193, sourceArmColumn := 26521, finalStart := 299145, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 194, sourceArmColumn := 26526, finalStart := 299350, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 195, sourceArmColumn := 26531, finalStart := 299555, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 196, sourceArmColumn := 26536, finalStart := 299760, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 197, sourceArmColumn := 26541, finalStart := 299965, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 198, sourceArmColumn := 26546, finalStart := 300170, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 199, sourceArmColumn := 26551, finalStart := 300375, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 200, sourceArmColumn := 26556, finalStart := 300580, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 201, sourceArmColumn := 26561, finalStart := 300785, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 202, sourceArmColumn := 26566, finalStart := 300990, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 203, sourceArmColumn := 26571, finalStart := 301195, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 204, sourceArmColumn := 26576, finalStart := 301400, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 205, sourceArmColumn := 26581, finalStart := 301605, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 206, sourceArmColumn := 26586, finalStart := 301810, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 207, sourceArmColumn := 26591, finalStart := 302015, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 208, sourceArmColumn := 26596, finalStart := 302220, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 209, sourceArmColumn := 26601, finalStart := 302425, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 210, sourceArmColumn := 26606, finalStart := 302630, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 211, sourceArmColumn := 26611, finalStart := 302835, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 212, sourceArmColumn := 26616, finalStart := 303040, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 213, sourceArmColumn := 26621, finalStart := 303245, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 214, sourceArmColumn := 26626, finalStart := 303450, width := 41, encoding := .balancedTernary }
, { child := 2, logicalColumn := 215, sourceArmColumn := 26631, finalStart := 303655, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk2
