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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk1

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 0, logicalColumn := 252, sourceArmColumn := 22003, finalStart := 141541, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 253, sourceArmColumn := 22008, finalStart := 141746, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 254, sourceArmColumn := 22013, finalStart := 141951, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 255, sourceArmColumn := 22018, finalStart := 142156, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 256, sourceArmColumn := 22023, finalStart := 142361, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 257, sourceArmColumn := 22028, finalStart := 142566, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 258, sourceArmColumn := 22033, finalStart := 142771, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 259, sourceArmColumn := 22038, finalStart := 142976, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 260, sourceArmColumn := 22043, finalStart := 143181, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 261, sourceArmColumn := 22048, finalStart := 143386, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 262, sourceArmColumn := 22053, finalStart := 143591, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 263, sourceArmColumn := 22058, finalStart := 143796, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 264, sourceArmColumn := 22063, finalStart := 144001, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 265, sourceArmColumn := 22068, finalStart := 144206, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 266, sourceArmColumn := 22073, finalStart := 144411, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 267, sourceArmColumn := 22078, finalStart := 144616, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 268, sourceArmColumn := 22083, finalStart := 144821, width := 41, encoding := .balancedTernary }
, { child := 0, logicalColumn := 269, sourceArmColumn := 22088, finalStart := 145026, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 0, sourceArmColumn := 24091, finalStart := 215177, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 1, sourceArmColumn := 24096, finalStart := 215382, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 2, sourceArmColumn := 24101, finalStart := 215587, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 3, sourceArmColumn := 24106, finalStart := 215792, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 4, sourceArmColumn := 24111, finalStart := 215997, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 5, sourceArmColumn := 24116, finalStart := 216202, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 6, sourceArmColumn := 24121, finalStart := 216407, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 7, sourceArmColumn := 24126, finalStart := 216612, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 8, sourceArmColumn := 24131, finalStart := 216817, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 9, sourceArmColumn := 24136, finalStart := 217022, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 10, sourceArmColumn := 24141, finalStart := 217227, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 11, sourceArmColumn := 24146, finalStart := 217432, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 12, sourceArmColumn := 24151, finalStart := 217637, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 13, sourceArmColumn := 24156, finalStart := 217842, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 14, sourceArmColumn := 24161, finalStart := 218047, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 15, sourceArmColumn := 24166, finalStart := 218252, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 16, sourceArmColumn := 24171, finalStart := 218457, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 17, sourceArmColumn := 24176, finalStart := 218662, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 18, sourceArmColumn := 24181, finalStart := 218867, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 19, sourceArmColumn := 24186, finalStart := 219072, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 20, sourceArmColumn := 24191, finalStart := 219277, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 21, sourceArmColumn := 24196, finalStart := 219482, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 22, sourceArmColumn := 24201, finalStart := 219687, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 23, sourceArmColumn := 24206, finalStart := 219892, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 24, sourceArmColumn := 24211, finalStart := 220097, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 25, sourceArmColumn := 24216, finalStart := 220302, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 26, sourceArmColumn := 24221, finalStart := 220507, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 27, sourceArmColumn := 24226, finalStart := 220712, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 28, sourceArmColumn := 24231, finalStart := 220917, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 29, sourceArmColumn := 24236, finalStart := 221122, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 30, sourceArmColumn := 24241, finalStart := 221327, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 31, sourceArmColumn := 24246, finalStart := 221532, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 32, sourceArmColumn := 24251, finalStart := 221737, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 33, sourceArmColumn := 24256, finalStart := 221942, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 34, sourceArmColumn := 24261, finalStart := 222147, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 35, sourceArmColumn := 24266, finalStart := 222352, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 36, sourceArmColumn := 24271, finalStart := 222557, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 37, sourceArmColumn := 24276, finalStart := 222762, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 38, sourceArmColumn := 24281, finalStart := 222967, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 39, sourceArmColumn := 24286, finalStart := 223172, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 40, sourceArmColumn := 24291, finalStart := 223377, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 41, sourceArmColumn := 24296, finalStart := 223582, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 42, sourceArmColumn := 24301, finalStart := 223787, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 43, sourceArmColumn := 24306, finalStart := 223992, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 44, sourceArmColumn := 24311, finalStart := 224197, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 45, sourceArmColumn := 24316, finalStart := 224402, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 46, sourceArmColumn := 24321, finalStart := 224607, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 47, sourceArmColumn := 24326, finalStart := 224812, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 48, sourceArmColumn := 24331, finalStart := 225017, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 49, sourceArmColumn := 24336, finalStart := 225222, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 50, sourceArmColumn := 24341, finalStart := 225427, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 51, sourceArmColumn := 24346, finalStart := 225632, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 52, sourceArmColumn := 24351, finalStart := 225837, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 53, sourceArmColumn := 24356, finalStart := 226042, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 54, sourceArmColumn := 24092, finalStart := 215218, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 55, sourceArmColumn := 24097, finalStart := 215423, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 56, sourceArmColumn := 24102, finalStart := 215628, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 57, sourceArmColumn := 24107, finalStart := 215833, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 58, sourceArmColumn := 24112, finalStart := 216038, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 59, sourceArmColumn := 24117, finalStart := 216243, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 60, sourceArmColumn := 24122, finalStart := 216448, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 61, sourceArmColumn := 24127, finalStart := 216653, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 62, sourceArmColumn := 24132, finalStart := 216858, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 63, sourceArmColumn := 24137, finalStart := 217063, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 64, sourceArmColumn := 24142, finalStart := 217268, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 65, sourceArmColumn := 24147, finalStart := 217473, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 66, sourceArmColumn := 24152, finalStart := 217678, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 67, sourceArmColumn := 24157, finalStart := 217883, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 68, sourceArmColumn := 24162, finalStart := 218088, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 69, sourceArmColumn := 24167, finalStart := 218293, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 70, sourceArmColumn := 24172, finalStart := 218498, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 71, sourceArmColumn := 24177, finalStart := 218703, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 72, sourceArmColumn := 24182, finalStart := 218908, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 73, sourceArmColumn := 24187, finalStart := 219113, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 74, sourceArmColumn := 24192, finalStart := 219318, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 75, sourceArmColumn := 24197, finalStart := 219523, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 76, sourceArmColumn := 24202, finalStart := 219728, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 77, sourceArmColumn := 24207, finalStart := 219933, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 78, sourceArmColumn := 24212, finalStart := 220138, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 79, sourceArmColumn := 24217, finalStart := 220343, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 80, sourceArmColumn := 24222, finalStart := 220548, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 81, sourceArmColumn := 24227, finalStart := 220753, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 82, sourceArmColumn := 24232, finalStart := 220958, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 83, sourceArmColumn := 24237, finalStart := 221163, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 84, sourceArmColumn := 24242, finalStart := 221368, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 85, sourceArmColumn := 24247, finalStart := 221573, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 86, sourceArmColumn := 24252, finalStart := 221778, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 87, sourceArmColumn := 24257, finalStart := 221983, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 88, sourceArmColumn := 24262, finalStart := 222188, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 89, sourceArmColumn := 24267, finalStart := 222393, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 90, sourceArmColumn := 24272, finalStart := 222598, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 91, sourceArmColumn := 24277, finalStart := 222803, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 92, sourceArmColumn := 24282, finalStart := 223008, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 93, sourceArmColumn := 24287, finalStart := 223213, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 94, sourceArmColumn := 24292, finalStart := 223418, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 95, sourceArmColumn := 24297, finalStart := 223623, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 96, sourceArmColumn := 24302, finalStart := 223828, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 97, sourceArmColumn := 24307, finalStart := 224033, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 98, sourceArmColumn := 24312, finalStart := 224238, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 99, sourceArmColumn := 24317, finalStart := 224443, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 100, sourceArmColumn := 24322, finalStart := 224648, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 101, sourceArmColumn := 24327, finalStart := 224853, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 102, sourceArmColumn := 24332, finalStart := 225058, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 103, sourceArmColumn := 24337, finalStart := 225263, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 104, sourceArmColumn := 24342, finalStart := 225468, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 105, sourceArmColumn := 24347, finalStart := 225673, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 106, sourceArmColumn := 24352, finalStart := 225878, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 107, sourceArmColumn := 24357, finalStart := 226083, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 108, sourceArmColumn := 24093, finalStart := 215259, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 109, sourceArmColumn := 24098, finalStart := 215464, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 110, sourceArmColumn := 24103, finalStart := 215669, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 111, sourceArmColumn := 24108, finalStart := 215874, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 112, sourceArmColumn := 24113, finalStart := 216079, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 113, sourceArmColumn := 24118, finalStart := 216284, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 114, sourceArmColumn := 24123, finalStart := 216489, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 115, sourceArmColumn := 24128, finalStart := 216694, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 116, sourceArmColumn := 24133, finalStart := 216899, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 117, sourceArmColumn := 24138, finalStart := 217104, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 118, sourceArmColumn := 24143, finalStart := 217309, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 119, sourceArmColumn := 24148, finalStart := 217514, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 120, sourceArmColumn := 24153, finalStart := 217719, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 121, sourceArmColumn := 24158, finalStart := 217924, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 122, sourceArmColumn := 24163, finalStart := 218129, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 123, sourceArmColumn := 24168, finalStart := 218334, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 124, sourceArmColumn := 24173, finalStart := 218539, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 125, sourceArmColumn := 24178, finalStart := 218744, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 126, sourceArmColumn := 24183, finalStart := 218949, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 127, sourceArmColumn := 24188, finalStart := 219154, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 128, sourceArmColumn := 24193, finalStart := 219359, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 129, sourceArmColumn := 24198, finalStart := 219564, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 130, sourceArmColumn := 24203, finalStart := 219769, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 131, sourceArmColumn := 24208, finalStart := 219974, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 132, sourceArmColumn := 24213, finalStart := 220179, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 133, sourceArmColumn := 24218, finalStart := 220384, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 134, sourceArmColumn := 24223, finalStart := 220589, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 135, sourceArmColumn := 24228, finalStart := 220794, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 136, sourceArmColumn := 24233, finalStart := 220999, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 137, sourceArmColumn := 24238, finalStart := 221204, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 138, sourceArmColumn := 24243, finalStart := 221409, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 139, sourceArmColumn := 24248, finalStart := 221614, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 140, sourceArmColumn := 24253, finalStart := 221819, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 141, sourceArmColumn := 24258, finalStart := 222024, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 142, sourceArmColumn := 24263, finalStart := 222229, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 143, sourceArmColumn := 24268, finalStart := 222434, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 144, sourceArmColumn := 24273, finalStart := 222639, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 145, sourceArmColumn := 24278, finalStart := 222844, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 146, sourceArmColumn := 24283, finalStart := 223049, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 147, sourceArmColumn := 24288, finalStart := 223254, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 148, sourceArmColumn := 24293, finalStart := 223459, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 149, sourceArmColumn := 24298, finalStart := 223664, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 150, sourceArmColumn := 24303, finalStart := 223869, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 151, sourceArmColumn := 24308, finalStart := 224074, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 152, sourceArmColumn := 24313, finalStart := 224279, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 153, sourceArmColumn := 24318, finalStart := 224484, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 154, sourceArmColumn := 24323, finalStart := 224689, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 155, sourceArmColumn := 24328, finalStart := 224894, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 156, sourceArmColumn := 24333, finalStart := 225099, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 157, sourceArmColumn := 24338, finalStart := 225304, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 158, sourceArmColumn := 24343, finalStart := 225509, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 159, sourceArmColumn := 24348, finalStart := 225714, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 160, sourceArmColumn := 24353, finalStart := 225919, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 161, sourceArmColumn := 24358, finalStart := 226124, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 162, sourceArmColumn := 24094, finalStart := 215300, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 163, sourceArmColumn := 24099, finalStart := 215505, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 164, sourceArmColumn := 24104, finalStart := 215710, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 165, sourceArmColumn := 24109, finalStart := 215915, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 166, sourceArmColumn := 24114, finalStart := 216120, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 167, sourceArmColumn := 24119, finalStart := 216325, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 168, sourceArmColumn := 24124, finalStart := 216530, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 169, sourceArmColumn := 24129, finalStart := 216735, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 170, sourceArmColumn := 24134, finalStart := 216940, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 171, sourceArmColumn := 24139, finalStart := 217145, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 172, sourceArmColumn := 24144, finalStart := 217350, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 173, sourceArmColumn := 24149, finalStart := 217555, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 174, sourceArmColumn := 24154, finalStart := 217760, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 175, sourceArmColumn := 24159, finalStart := 217965, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 176, sourceArmColumn := 24164, finalStart := 218170, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 177, sourceArmColumn := 24169, finalStart := 218375, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 178, sourceArmColumn := 24174, finalStart := 218580, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 179, sourceArmColumn := 24179, finalStart := 218785, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 180, sourceArmColumn := 24184, finalStart := 218990, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 181, sourceArmColumn := 24189, finalStart := 219195, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 182, sourceArmColumn := 24194, finalStart := 219400, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 183, sourceArmColumn := 24199, finalStart := 219605, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 184, sourceArmColumn := 24204, finalStart := 219810, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 185, sourceArmColumn := 24209, finalStart := 220015, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 186, sourceArmColumn := 24214, finalStart := 220220, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 187, sourceArmColumn := 24219, finalStart := 220425, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 188, sourceArmColumn := 24224, finalStart := 220630, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 189, sourceArmColumn := 24229, finalStart := 220835, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 190, sourceArmColumn := 24234, finalStart := 221040, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 191, sourceArmColumn := 24239, finalStart := 221245, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 192, sourceArmColumn := 24244, finalStart := 221450, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 193, sourceArmColumn := 24249, finalStart := 221655, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 194, sourceArmColumn := 24254, finalStart := 221860, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 195, sourceArmColumn := 24259, finalStart := 222065, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 196, sourceArmColumn := 24264, finalStart := 222270, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 197, sourceArmColumn := 24269, finalStart := 222475, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 198, sourceArmColumn := 24274, finalStart := 222680, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 199, sourceArmColumn := 24279, finalStart := 222885, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 200, sourceArmColumn := 24284, finalStart := 223090, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 201, sourceArmColumn := 24289, finalStart := 223295, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 202, sourceArmColumn := 24294, finalStart := 223500, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 203, sourceArmColumn := 24299, finalStart := 223705, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 204, sourceArmColumn := 24304, finalStart := 223910, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 205, sourceArmColumn := 24309, finalStart := 224115, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 206, sourceArmColumn := 24314, finalStart := 224320, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 207, sourceArmColumn := 24319, finalStart := 224525, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 208, sourceArmColumn := 24324, finalStart := 224730, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 209, sourceArmColumn := 24329, finalStart := 224935, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 210, sourceArmColumn := 24334, finalStart := 225140, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 211, sourceArmColumn := 24339, finalStart := 225345, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 212, sourceArmColumn := 24344, finalStart := 225550, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 213, sourceArmColumn := 24349, finalStart := 225755, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 214, sourceArmColumn := 24354, finalStart := 225960, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 215, sourceArmColumn := 24359, finalStart := 226165, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 216, sourceArmColumn := 24095, finalStart := 215341, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 217, sourceArmColumn := 24100, finalStart := 215546, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 218, sourceArmColumn := 24105, finalStart := 215751, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 219, sourceArmColumn := 24110, finalStart := 215956, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 220, sourceArmColumn := 24115, finalStart := 216161, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 221, sourceArmColumn := 24120, finalStart := 216366, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 222, sourceArmColumn := 24125, finalStart := 216571, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 223, sourceArmColumn := 24130, finalStart := 216776, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 224, sourceArmColumn := 24135, finalStart := 216981, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 225, sourceArmColumn := 24140, finalStart := 217186, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 226, sourceArmColumn := 24145, finalStart := 217391, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 227, sourceArmColumn := 24150, finalStart := 217596, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 228, sourceArmColumn := 24155, finalStart := 217801, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 229, sourceArmColumn := 24160, finalStart := 218006, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 230, sourceArmColumn := 24165, finalStart := 218211, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 231, sourceArmColumn := 24170, finalStart := 218416, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 232, sourceArmColumn := 24175, finalStart := 218621, width := 41, encoding := .balancedTernary }
, { child := 1, logicalColumn := 233, sourceArmColumn := 24180, finalStart := 218826, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk1
