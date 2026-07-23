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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk13

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 2
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11437038
def allocationRecords : List AllocationRecord := [
  { child := 12, logicalColumn := 36, sourceArmColumn := 49263, finalStart := 1074947, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 37, sourceArmColumn := 49268, finalStart := 1075152, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 38, sourceArmColumn := 49273, finalStart := 1075357, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 39, sourceArmColumn := 49278, finalStart := 1075562, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 40, sourceArmColumn := 49283, finalStart := 1075767, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 41, sourceArmColumn := 49288, finalStart := 1075972, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 42, sourceArmColumn := 49293, finalStart := 1076177, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 43, sourceArmColumn := 49298, finalStart := 1076382, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 44, sourceArmColumn := 49303, finalStart := 1076587, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 45, sourceArmColumn := 49308, finalStart := 1076792, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 46, sourceArmColumn := 49313, finalStart := 1076997, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 47, sourceArmColumn := 49318, finalStart := 1077202, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 48, sourceArmColumn := 49323, finalStart := 1077407, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 49, sourceArmColumn := 49328, finalStart := 1077612, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 50, sourceArmColumn := 49333, finalStart := 1077817, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 51, sourceArmColumn := 49338, finalStart := 1078022, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 52, sourceArmColumn := 49343, finalStart := 1078227, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 53, sourceArmColumn := 49348, finalStart := 1078432, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 54, sourceArmColumn := 49084, finalStart := 1067608, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 55, sourceArmColumn := 49089, finalStart := 1067813, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 56, sourceArmColumn := 49094, finalStart := 1068018, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 57, sourceArmColumn := 49099, finalStart := 1068223, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 58, sourceArmColumn := 49104, finalStart := 1068428, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 59, sourceArmColumn := 49109, finalStart := 1068633, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 60, sourceArmColumn := 49114, finalStart := 1068838, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 61, sourceArmColumn := 49119, finalStart := 1069043, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 62, sourceArmColumn := 49124, finalStart := 1069248, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 63, sourceArmColumn := 49129, finalStart := 1069453, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 64, sourceArmColumn := 49134, finalStart := 1069658, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 65, sourceArmColumn := 49139, finalStart := 1069863, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 66, sourceArmColumn := 49144, finalStart := 1070068, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 67, sourceArmColumn := 49149, finalStart := 1070273, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 68, sourceArmColumn := 49154, finalStart := 1070478, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 69, sourceArmColumn := 49159, finalStart := 1070683, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 70, sourceArmColumn := 49164, finalStart := 1070888, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 71, sourceArmColumn := 49169, finalStart := 1071093, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 72, sourceArmColumn := 49174, finalStart := 1071298, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 73, sourceArmColumn := 49179, finalStart := 1071503, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 74, sourceArmColumn := 49184, finalStart := 1071708, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 75, sourceArmColumn := 49189, finalStart := 1071913, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 76, sourceArmColumn := 49194, finalStart := 1072118, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 77, sourceArmColumn := 49199, finalStart := 1072323, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 78, sourceArmColumn := 49204, finalStart := 1072528, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 79, sourceArmColumn := 49209, finalStart := 1072733, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 80, sourceArmColumn := 49214, finalStart := 1072938, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 81, sourceArmColumn := 49219, finalStart := 1073143, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 82, sourceArmColumn := 49224, finalStart := 1073348, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 83, sourceArmColumn := 49229, finalStart := 1073553, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 84, sourceArmColumn := 49234, finalStart := 1073758, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 85, sourceArmColumn := 49239, finalStart := 1073963, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 86, sourceArmColumn := 49244, finalStart := 1074168, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 87, sourceArmColumn := 49249, finalStart := 1074373, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 88, sourceArmColumn := 49254, finalStart := 1074578, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 89, sourceArmColumn := 49259, finalStart := 1074783, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 90, sourceArmColumn := 49264, finalStart := 1074988, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 91, sourceArmColumn := 49269, finalStart := 1075193, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 92, sourceArmColumn := 49274, finalStart := 1075398, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 93, sourceArmColumn := 49279, finalStart := 1075603, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 94, sourceArmColumn := 49284, finalStart := 1075808, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 95, sourceArmColumn := 49289, finalStart := 1076013, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 96, sourceArmColumn := 49294, finalStart := 1076218, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 97, sourceArmColumn := 49299, finalStart := 1076423, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 98, sourceArmColumn := 49304, finalStart := 1076628, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 99, sourceArmColumn := 49309, finalStart := 1076833, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 100, sourceArmColumn := 49314, finalStart := 1077038, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 101, sourceArmColumn := 49319, finalStart := 1077243, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 102, sourceArmColumn := 49324, finalStart := 1077448, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 103, sourceArmColumn := 49329, finalStart := 1077653, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 104, sourceArmColumn := 49334, finalStart := 1077858, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 105, sourceArmColumn := 49339, finalStart := 1078063, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 106, sourceArmColumn := 49344, finalStart := 1078268, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 107, sourceArmColumn := 49349, finalStart := 1078473, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 108, sourceArmColumn := 49085, finalStart := 1067649, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 109, sourceArmColumn := 49090, finalStart := 1067854, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 110, sourceArmColumn := 49095, finalStart := 1068059, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 111, sourceArmColumn := 49100, finalStart := 1068264, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 112, sourceArmColumn := 49105, finalStart := 1068469, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 113, sourceArmColumn := 49110, finalStart := 1068674, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 114, sourceArmColumn := 49115, finalStart := 1068879, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 115, sourceArmColumn := 49120, finalStart := 1069084, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 116, sourceArmColumn := 49125, finalStart := 1069289, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 117, sourceArmColumn := 49130, finalStart := 1069494, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 118, sourceArmColumn := 49135, finalStart := 1069699, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 119, sourceArmColumn := 49140, finalStart := 1069904, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 120, sourceArmColumn := 49145, finalStart := 1070109, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 121, sourceArmColumn := 49150, finalStart := 1070314, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 122, sourceArmColumn := 49155, finalStart := 1070519, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 123, sourceArmColumn := 49160, finalStart := 1070724, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 124, sourceArmColumn := 49165, finalStart := 1070929, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 125, sourceArmColumn := 49170, finalStart := 1071134, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 126, sourceArmColumn := 49175, finalStart := 1071339, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 127, sourceArmColumn := 49180, finalStart := 1071544, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 128, sourceArmColumn := 49185, finalStart := 1071749, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 129, sourceArmColumn := 49190, finalStart := 1071954, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 130, sourceArmColumn := 49195, finalStart := 1072159, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 131, sourceArmColumn := 49200, finalStart := 1072364, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 132, sourceArmColumn := 49205, finalStart := 1072569, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 133, sourceArmColumn := 49210, finalStart := 1072774, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 134, sourceArmColumn := 49215, finalStart := 1072979, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 135, sourceArmColumn := 49220, finalStart := 1073184, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 136, sourceArmColumn := 49225, finalStart := 1073389, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 137, sourceArmColumn := 49230, finalStart := 1073594, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 138, sourceArmColumn := 49235, finalStart := 1073799, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 139, sourceArmColumn := 49240, finalStart := 1074004, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 140, sourceArmColumn := 49245, finalStart := 1074209, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 141, sourceArmColumn := 49250, finalStart := 1074414, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 142, sourceArmColumn := 49255, finalStart := 1074619, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 143, sourceArmColumn := 49260, finalStart := 1074824, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 144, sourceArmColumn := 49265, finalStart := 1075029, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 145, sourceArmColumn := 49270, finalStart := 1075234, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 146, sourceArmColumn := 49275, finalStart := 1075439, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 147, sourceArmColumn := 49280, finalStart := 1075644, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 148, sourceArmColumn := 49285, finalStart := 1075849, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 149, sourceArmColumn := 49290, finalStart := 1076054, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 150, sourceArmColumn := 49295, finalStart := 1076259, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 151, sourceArmColumn := 49300, finalStart := 1076464, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 152, sourceArmColumn := 49305, finalStart := 1076669, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 153, sourceArmColumn := 49310, finalStart := 1076874, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 154, sourceArmColumn := 49315, finalStart := 1077079, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 155, sourceArmColumn := 49320, finalStart := 1077284, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 156, sourceArmColumn := 49325, finalStart := 1077489, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 157, sourceArmColumn := 49330, finalStart := 1077694, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 158, sourceArmColumn := 49335, finalStart := 1077899, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 159, sourceArmColumn := 49340, finalStart := 1078104, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 160, sourceArmColumn := 49345, finalStart := 1078309, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 161, sourceArmColumn := 49350, finalStart := 1078514, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 162, sourceArmColumn := 49086, finalStart := 1067690, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 163, sourceArmColumn := 49091, finalStart := 1067895, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 164, sourceArmColumn := 49096, finalStart := 1068100, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 165, sourceArmColumn := 49101, finalStart := 1068305, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 166, sourceArmColumn := 49106, finalStart := 1068510, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 167, sourceArmColumn := 49111, finalStart := 1068715, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 168, sourceArmColumn := 49116, finalStart := 1068920, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 169, sourceArmColumn := 49121, finalStart := 1069125, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 170, sourceArmColumn := 49126, finalStart := 1069330, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 171, sourceArmColumn := 49131, finalStart := 1069535, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 172, sourceArmColumn := 49136, finalStart := 1069740, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 173, sourceArmColumn := 49141, finalStart := 1069945, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 174, sourceArmColumn := 49146, finalStart := 1070150, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 175, sourceArmColumn := 49151, finalStart := 1070355, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 176, sourceArmColumn := 49156, finalStart := 1070560, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 177, sourceArmColumn := 49161, finalStart := 1070765, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 178, sourceArmColumn := 49166, finalStart := 1070970, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 179, sourceArmColumn := 49171, finalStart := 1071175, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 180, sourceArmColumn := 49176, finalStart := 1071380, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 181, sourceArmColumn := 49181, finalStart := 1071585, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 182, sourceArmColumn := 49186, finalStart := 1071790, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 183, sourceArmColumn := 49191, finalStart := 1071995, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 184, sourceArmColumn := 49196, finalStart := 1072200, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 185, sourceArmColumn := 49201, finalStart := 1072405, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 186, sourceArmColumn := 49206, finalStart := 1072610, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 187, sourceArmColumn := 49211, finalStart := 1072815, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 188, sourceArmColumn := 49216, finalStart := 1073020, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 189, sourceArmColumn := 49221, finalStart := 1073225, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 190, sourceArmColumn := 49226, finalStart := 1073430, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 191, sourceArmColumn := 49231, finalStart := 1073635, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 192, sourceArmColumn := 49236, finalStart := 1073840, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 193, sourceArmColumn := 49241, finalStart := 1074045, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 194, sourceArmColumn := 49246, finalStart := 1074250, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 195, sourceArmColumn := 49251, finalStart := 1074455, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 196, sourceArmColumn := 49256, finalStart := 1074660, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 197, sourceArmColumn := 49261, finalStart := 1074865, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 198, sourceArmColumn := 49266, finalStart := 1075070, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 199, sourceArmColumn := 49271, finalStart := 1075275, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 200, sourceArmColumn := 49276, finalStart := 1075480, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 201, sourceArmColumn := 49281, finalStart := 1075685, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 202, sourceArmColumn := 49286, finalStart := 1075890, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 203, sourceArmColumn := 49291, finalStart := 1076095, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 204, sourceArmColumn := 49296, finalStart := 1076300, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 205, sourceArmColumn := 49301, finalStart := 1076505, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 206, sourceArmColumn := 49306, finalStart := 1076710, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 207, sourceArmColumn := 49311, finalStart := 1076915, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 208, sourceArmColumn := 49316, finalStart := 1077120, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 209, sourceArmColumn := 49321, finalStart := 1077325, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 210, sourceArmColumn := 49326, finalStart := 1077530, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 211, sourceArmColumn := 49331, finalStart := 1077735, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 212, sourceArmColumn := 49336, finalStart := 1077940, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 213, sourceArmColumn := 49341, finalStart := 1078145, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 214, sourceArmColumn := 49346, finalStart := 1078350, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 215, sourceArmColumn := 49351, finalStart := 1078555, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 216, sourceArmColumn := 49087, finalStart := 1067731, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 217, sourceArmColumn := 49092, finalStart := 1067936, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 218, sourceArmColumn := 49097, finalStart := 1068141, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 219, sourceArmColumn := 49102, finalStart := 1068346, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 220, sourceArmColumn := 49107, finalStart := 1068551, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 221, sourceArmColumn := 49112, finalStart := 1068756, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 222, sourceArmColumn := 49117, finalStart := 1068961, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 223, sourceArmColumn := 49122, finalStart := 1069166, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 224, sourceArmColumn := 49127, finalStart := 1069371, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 225, sourceArmColumn := 49132, finalStart := 1069576, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 226, sourceArmColumn := 49137, finalStart := 1069781, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 227, sourceArmColumn := 49142, finalStart := 1069986, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 228, sourceArmColumn := 49147, finalStart := 1070191, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 229, sourceArmColumn := 49152, finalStart := 1070396, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 230, sourceArmColumn := 49157, finalStart := 1070601, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 231, sourceArmColumn := 49162, finalStart := 1070806, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 232, sourceArmColumn := 49167, finalStart := 1071011, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 233, sourceArmColumn := 49172, finalStart := 1071216, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 234, sourceArmColumn := 49177, finalStart := 1071421, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 235, sourceArmColumn := 49182, finalStart := 1071626, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 236, sourceArmColumn := 49187, finalStart := 1071831, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 237, sourceArmColumn := 49192, finalStart := 1072036, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 238, sourceArmColumn := 49197, finalStart := 1072241, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 239, sourceArmColumn := 49202, finalStart := 1072446, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 240, sourceArmColumn := 49207, finalStart := 1072651, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 241, sourceArmColumn := 49212, finalStart := 1072856, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 242, sourceArmColumn := 49217, finalStart := 1073061, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 243, sourceArmColumn := 49222, finalStart := 1073266, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 244, sourceArmColumn := 49227, finalStart := 1073471, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 245, sourceArmColumn := 49232, finalStart := 1073676, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 246, sourceArmColumn := 49237, finalStart := 1073881, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 247, sourceArmColumn := 49242, finalStart := 1074086, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 248, sourceArmColumn := 49247, finalStart := 1074291, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 249, sourceArmColumn := 49252, finalStart := 1074496, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 250, sourceArmColumn := 49257, finalStart := 1074701, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 251, sourceArmColumn := 49262, finalStart := 1074906, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 252, sourceArmColumn := 49267, finalStart := 1075111, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 253, sourceArmColumn := 49272, finalStart := 1075316, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 254, sourceArmColumn := 49277, finalStart := 1075521, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 255, sourceArmColumn := 49282, finalStart := 1075726, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 256, sourceArmColumn := 49287, finalStart := 1075931, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 257, sourceArmColumn := 49292, finalStart := 1076136, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 258, sourceArmColumn := 49297, finalStart := 1076341, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 259, sourceArmColumn := 49302, finalStart := 1076546, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 260, sourceArmColumn := 49307, finalStart := 1076751, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 261, sourceArmColumn := 49312, finalStart := 1076956, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 262, sourceArmColumn := 49317, finalStart := 1077161, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 263, sourceArmColumn := 49322, finalStart := 1077366, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 264, sourceArmColumn := 49327, finalStart := 1077571, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 265, sourceArmColumn := 49332, finalStart := 1077776, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 266, sourceArmColumn := 49337, finalStart := 1077981, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 267, sourceArmColumn := 49342, finalStart := 1078186, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 268, sourceArmColumn := 49347, finalStart := 1078391, width := 41, encoding := .balancedTernary }
, { child := 12, logicalColumn := 269, sourceArmColumn := 49352, finalStart := 1078596, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 0, sourceArmColumn := 51355, finalStart := 1145057, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 1, sourceArmColumn := 51360, finalStart := 1145262, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 2, sourceArmColumn := 51365, finalStart := 1145467, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 3, sourceArmColumn := 51370, finalStart := 1145672, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 4, sourceArmColumn := 51375, finalStart := 1145877, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 5, sourceArmColumn := 51380, finalStart := 1146082, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 6, sourceArmColumn := 51385, finalStart := 1146287, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 7, sourceArmColumn := 51390, finalStart := 1146492, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 8, sourceArmColumn := 51395, finalStart := 1146697, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 9, sourceArmColumn := 51400, finalStart := 1146902, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 10, sourceArmColumn := 51405, finalStart := 1147107, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 11, sourceArmColumn := 51410, finalStart := 1147312, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 12, sourceArmColumn := 51415, finalStart := 1147517, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 13, sourceArmColumn := 51420, finalStart := 1147722, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 14, sourceArmColumn := 51425, finalStart := 1147927, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 15, sourceArmColumn := 51430, finalStart := 1148132, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 16, sourceArmColumn := 51435, finalStart := 1148337, width := 41, encoding := .balancedTernary }
, { child := 13, logicalColumn := 17, sourceArmColumn := 51440, finalStart := 1148542, width := 41, encoding := .balancedTernary }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk13
