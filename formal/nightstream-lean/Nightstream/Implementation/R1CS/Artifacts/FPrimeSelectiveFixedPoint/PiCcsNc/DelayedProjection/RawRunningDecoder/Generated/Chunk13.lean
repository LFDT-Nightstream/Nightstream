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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk13

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 12, logicalColumn := 36, sourceArmColumn := 49263, finalColumn := 938147 }
, { child := 12, logicalColumn := 37, sourceArmColumn := 49268, finalColumn := 938152 }
, { child := 12, logicalColumn := 38, sourceArmColumn := 49273, finalColumn := 938157 }
, { child := 12, logicalColumn := 39, sourceArmColumn := 49278, finalColumn := 938162 }
, { child := 12, logicalColumn := 40, sourceArmColumn := 49283, finalColumn := 938167 }
, { child := 12, logicalColumn := 41, sourceArmColumn := 49288, finalColumn := 938172 }
, { child := 12, logicalColumn := 42, sourceArmColumn := 49293, finalColumn := 938177 }
, { child := 12, logicalColumn := 43, sourceArmColumn := 49298, finalColumn := 938182 }
, { child := 12, logicalColumn := 44, sourceArmColumn := 49303, finalColumn := 938187 }
, { child := 12, logicalColumn := 45, sourceArmColumn := 49308, finalColumn := 938192 }
, { child := 12, logicalColumn := 46, sourceArmColumn := 49313, finalColumn := 938197 }
, { child := 12, logicalColumn := 47, sourceArmColumn := 49318, finalColumn := 938202 }
, { child := 12, logicalColumn := 48, sourceArmColumn := 49323, finalColumn := 938207 }
, { child := 12, logicalColumn := 49, sourceArmColumn := 49328, finalColumn := 938212 }
, { child := 12, logicalColumn := 50, sourceArmColumn := 49333, finalColumn := 938217 }
, { child := 12, logicalColumn := 51, sourceArmColumn := 49338, finalColumn := 938222 }
, { child := 12, logicalColumn := 52, sourceArmColumn := 49343, finalColumn := 938227 }
, { child := 12, logicalColumn := 53, sourceArmColumn := 49348, finalColumn := 938232 }
, { child := 12, logicalColumn := 54, sourceArmColumn := 49084, finalColumn := 937968 }
, { child := 12, logicalColumn := 55, sourceArmColumn := 49089, finalColumn := 937973 }
, { child := 12, logicalColumn := 56, sourceArmColumn := 49094, finalColumn := 937978 }
, { child := 12, logicalColumn := 57, sourceArmColumn := 49099, finalColumn := 937983 }
, { child := 12, logicalColumn := 58, sourceArmColumn := 49104, finalColumn := 937988 }
, { child := 12, logicalColumn := 59, sourceArmColumn := 49109, finalColumn := 937993 }
, { child := 12, logicalColumn := 60, sourceArmColumn := 49114, finalColumn := 937998 }
, { child := 12, logicalColumn := 61, sourceArmColumn := 49119, finalColumn := 938003 }
, { child := 12, logicalColumn := 62, sourceArmColumn := 49124, finalColumn := 938008 }
, { child := 12, logicalColumn := 63, sourceArmColumn := 49129, finalColumn := 938013 }
, { child := 12, logicalColumn := 64, sourceArmColumn := 49134, finalColumn := 938018 }
, { child := 12, logicalColumn := 65, sourceArmColumn := 49139, finalColumn := 938023 }
, { child := 12, logicalColumn := 66, sourceArmColumn := 49144, finalColumn := 938028 }
, { child := 12, logicalColumn := 67, sourceArmColumn := 49149, finalColumn := 938033 }
, { child := 12, logicalColumn := 68, sourceArmColumn := 49154, finalColumn := 938038 }
, { child := 12, logicalColumn := 69, sourceArmColumn := 49159, finalColumn := 938043 }
, { child := 12, logicalColumn := 70, sourceArmColumn := 49164, finalColumn := 938048 }
, { child := 12, logicalColumn := 71, sourceArmColumn := 49169, finalColumn := 938053 }
, { child := 12, logicalColumn := 72, sourceArmColumn := 49174, finalColumn := 938058 }
, { child := 12, logicalColumn := 73, sourceArmColumn := 49179, finalColumn := 938063 }
, { child := 12, logicalColumn := 74, sourceArmColumn := 49184, finalColumn := 938068 }
, { child := 12, logicalColumn := 75, sourceArmColumn := 49189, finalColumn := 938073 }
, { child := 12, logicalColumn := 76, sourceArmColumn := 49194, finalColumn := 938078 }
, { child := 12, logicalColumn := 77, sourceArmColumn := 49199, finalColumn := 938083 }
, { child := 12, logicalColumn := 78, sourceArmColumn := 49204, finalColumn := 938088 }
, { child := 12, logicalColumn := 79, sourceArmColumn := 49209, finalColumn := 938093 }
, { child := 12, logicalColumn := 80, sourceArmColumn := 49214, finalColumn := 938098 }
, { child := 12, logicalColumn := 81, sourceArmColumn := 49219, finalColumn := 938103 }
, { child := 12, logicalColumn := 82, sourceArmColumn := 49224, finalColumn := 938108 }
, { child := 12, logicalColumn := 83, sourceArmColumn := 49229, finalColumn := 938113 }
, { child := 12, logicalColumn := 84, sourceArmColumn := 49234, finalColumn := 938118 }
, { child := 12, logicalColumn := 85, sourceArmColumn := 49239, finalColumn := 938123 }
, { child := 12, logicalColumn := 86, sourceArmColumn := 49244, finalColumn := 938128 }
, { child := 12, logicalColumn := 87, sourceArmColumn := 49249, finalColumn := 938133 }
, { child := 12, logicalColumn := 88, sourceArmColumn := 49254, finalColumn := 938138 }
, { child := 12, logicalColumn := 89, sourceArmColumn := 49259, finalColumn := 938143 }
, { child := 12, logicalColumn := 90, sourceArmColumn := 49264, finalColumn := 938148 }
, { child := 12, logicalColumn := 91, sourceArmColumn := 49269, finalColumn := 938153 }
, { child := 12, logicalColumn := 92, sourceArmColumn := 49274, finalColumn := 938158 }
, { child := 12, logicalColumn := 93, sourceArmColumn := 49279, finalColumn := 938163 }
, { child := 12, logicalColumn := 94, sourceArmColumn := 49284, finalColumn := 938168 }
, { child := 12, logicalColumn := 95, sourceArmColumn := 49289, finalColumn := 938173 }
, { child := 12, logicalColumn := 96, sourceArmColumn := 49294, finalColumn := 938178 }
, { child := 12, logicalColumn := 97, sourceArmColumn := 49299, finalColumn := 938183 }
, { child := 12, logicalColumn := 98, sourceArmColumn := 49304, finalColumn := 938188 }
, { child := 12, logicalColumn := 99, sourceArmColumn := 49309, finalColumn := 938193 }
, { child := 12, logicalColumn := 100, sourceArmColumn := 49314, finalColumn := 938198 }
, { child := 12, logicalColumn := 101, sourceArmColumn := 49319, finalColumn := 938203 }
, { child := 12, logicalColumn := 102, sourceArmColumn := 49324, finalColumn := 938208 }
, { child := 12, logicalColumn := 103, sourceArmColumn := 49329, finalColumn := 938213 }
, { child := 12, logicalColumn := 104, sourceArmColumn := 49334, finalColumn := 938218 }
, { child := 12, logicalColumn := 105, sourceArmColumn := 49339, finalColumn := 938223 }
, { child := 12, logicalColumn := 106, sourceArmColumn := 49344, finalColumn := 938228 }
, { child := 12, logicalColumn := 107, sourceArmColumn := 49349, finalColumn := 938233 }
, { child := 12, logicalColumn := 108, sourceArmColumn := 49085, finalColumn := 937969 }
, { child := 12, logicalColumn := 109, sourceArmColumn := 49090, finalColumn := 937974 }
, { child := 12, logicalColumn := 110, sourceArmColumn := 49095, finalColumn := 937979 }
, { child := 12, logicalColumn := 111, sourceArmColumn := 49100, finalColumn := 937984 }
, { child := 12, logicalColumn := 112, sourceArmColumn := 49105, finalColumn := 937989 }
, { child := 12, logicalColumn := 113, sourceArmColumn := 49110, finalColumn := 937994 }
, { child := 12, logicalColumn := 114, sourceArmColumn := 49115, finalColumn := 937999 }
, { child := 12, logicalColumn := 115, sourceArmColumn := 49120, finalColumn := 938004 }
, { child := 12, logicalColumn := 116, sourceArmColumn := 49125, finalColumn := 938009 }
, { child := 12, logicalColumn := 117, sourceArmColumn := 49130, finalColumn := 938014 }
, { child := 12, logicalColumn := 118, sourceArmColumn := 49135, finalColumn := 938019 }
, { child := 12, logicalColumn := 119, sourceArmColumn := 49140, finalColumn := 938024 }
, { child := 12, logicalColumn := 120, sourceArmColumn := 49145, finalColumn := 938029 }
, { child := 12, logicalColumn := 121, sourceArmColumn := 49150, finalColumn := 938034 }
, { child := 12, logicalColumn := 122, sourceArmColumn := 49155, finalColumn := 938039 }
, { child := 12, logicalColumn := 123, sourceArmColumn := 49160, finalColumn := 938044 }
, { child := 12, logicalColumn := 124, sourceArmColumn := 49165, finalColumn := 938049 }
, { child := 12, logicalColumn := 125, sourceArmColumn := 49170, finalColumn := 938054 }
, { child := 12, logicalColumn := 126, sourceArmColumn := 49175, finalColumn := 938059 }
, { child := 12, logicalColumn := 127, sourceArmColumn := 49180, finalColumn := 938064 }
, { child := 12, logicalColumn := 128, sourceArmColumn := 49185, finalColumn := 938069 }
, { child := 12, logicalColumn := 129, sourceArmColumn := 49190, finalColumn := 938074 }
, { child := 12, logicalColumn := 130, sourceArmColumn := 49195, finalColumn := 938079 }
, { child := 12, logicalColumn := 131, sourceArmColumn := 49200, finalColumn := 938084 }
, { child := 12, logicalColumn := 132, sourceArmColumn := 49205, finalColumn := 938089 }
, { child := 12, logicalColumn := 133, sourceArmColumn := 49210, finalColumn := 938094 }
, { child := 12, logicalColumn := 134, sourceArmColumn := 49215, finalColumn := 938099 }
, { child := 12, logicalColumn := 135, sourceArmColumn := 49220, finalColumn := 938104 }
, { child := 12, logicalColumn := 136, sourceArmColumn := 49225, finalColumn := 938109 }
, { child := 12, logicalColumn := 137, sourceArmColumn := 49230, finalColumn := 938114 }
, { child := 12, logicalColumn := 138, sourceArmColumn := 49235, finalColumn := 938119 }
, { child := 12, logicalColumn := 139, sourceArmColumn := 49240, finalColumn := 938124 }
, { child := 12, logicalColumn := 140, sourceArmColumn := 49245, finalColumn := 938129 }
, { child := 12, logicalColumn := 141, sourceArmColumn := 49250, finalColumn := 938134 }
, { child := 12, logicalColumn := 142, sourceArmColumn := 49255, finalColumn := 938139 }
, { child := 12, logicalColumn := 143, sourceArmColumn := 49260, finalColumn := 938144 }
, { child := 12, logicalColumn := 144, sourceArmColumn := 49265, finalColumn := 938149 }
, { child := 12, logicalColumn := 145, sourceArmColumn := 49270, finalColumn := 938154 }
, { child := 12, logicalColumn := 146, sourceArmColumn := 49275, finalColumn := 938159 }
, { child := 12, logicalColumn := 147, sourceArmColumn := 49280, finalColumn := 938164 }
, { child := 12, logicalColumn := 148, sourceArmColumn := 49285, finalColumn := 938169 }
, { child := 12, logicalColumn := 149, sourceArmColumn := 49290, finalColumn := 938174 }
, { child := 12, logicalColumn := 150, sourceArmColumn := 49295, finalColumn := 938179 }
, { child := 12, logicalColumn := 151, sourceArmColumn := 49300, finalColumn := 938184 }
, { child := 12, logicalColumn := 152, sourceArmColumn := 49305, finalColumn := 938189 }
, { child := 12, logicalColumn := 153, sourceArmColumn := 49310, finalColumn := 938194 }
, { child := 12, logicalColumn := 154, sourceArmColumn := 49315, finalColumn := 938199 }
, { child := 12, logicalColumn := 155, sourceArmColumn := 49320, finalColumn := 938204 }
, { child := 12, logicalColumn := 156, sourceArmColumn := 49325, finalColumn := 938209 }
, { child := 12, logicalColumn := 157, sourceArmColumn := 49330, finalColumn := 938214 }
, { child := 12, logicalColumn := 158, sourceArmColumn := 49335, finalColumn := 938219 }
, { child := 12, logicalColumn := 159, sourceArmColumn := 49340, finalColumn := 938224 }
, { child := 12, logicalColumn := 160, sourceArmColumn := 49345, finalColumn := 938229 }
, { child := 12, logicalColumn := 161, sourceArmColumn := 49350, finalColumn := 938234 }
, { child := 12, logicalColumn := 162, sourceArmColumn := 49086, finalColumn := 937970 }
, { child := 12, logicalColumn := 163, sourceArmColumn := 49091, finalColumn := 937975 }
, { child := 12, logicalColumn := 164, sourceArmColumn := 49096, finalColumn := 937980 }
, { child := 12, logicalColumn := 165, sourceArmColumn := 49101, finalColumn := 937985 }
, { child := 12, logicalColumn := 166, sourceArmColumn := 49106, finalColumn := 937990 }
, { child := 12, logicalColumn := 167, sourceArmColumn := 49111, finalColumn := 937995 }
, { child := 12, logicalColumn := 168, sourceArmColumn := 49116, finalColumn := 938000 }
, { child := 12, logicalColumn := 169, sourceArmColumn := 49121, finalColumn := 938005 }
, { child := 12, logicalColumn := 170, sourceArmColumn := 49126, finalColumn := 938010 }
, { child := 12, logicalColumn := 171, sourceArmColumn := 49131, finalColumn := 938015 }
, { child := 12, logicalColumn := 172, sourceArmColumn := 49136, finalColumn := 938020 }
, { child := 12, logicalColumn := 173, sourceArmColumn := 49141, finalColumn := 938025 }
, { child := 12, logicalColumn := 174, sourceArmColumn := 49146, finalColumn := 938030 }
, { child := 12, logicalColumn := 175, sourceArmColumn := 49151, finalColumn := 938035 }
, { child := 12, logicalColumn := 176, sourceArmColumn := 49156, finalColumn := 938040 }
, { child := 12, logicalColumn := 177, sourceArmColumn := 49161, finalColumn := 938045 }
, { child := 12, logicalColumn := 178, sourceArmColumn := 49166, finalColumn := 938050 }
, { child := 12, logicalColumn := 179, sourceArmColumn := 49171, finalColumn := 938055 }
, { child := 12, logicalColumn := 180, sourceArmColumn := 49176, finalColumn := 938060 }
, { child := 12, logicalColumn := 181, sourceArmColumn := 49181, finalColumn := 938065 }
, { child := 12, logicalColumn := 182, sourceArmColumn := 49186, finalColumn := 938070 }
, { child := 12, logicalColumn := 183, sourceArmColumn := 49191, finalColumn := 938075 }
, { child := 12, logicalColumn := 184, sourceArmColumn := 49196, finalColumn := 938080 }
, { child := 12, logicalColumn := 185, sourceArmColumn := 49201, finalColumn := 938085 }
, { child := 12, logicalColumn := 186, sourceArmColumn := 49206, finalColumn := 938090 }
, { child := 12, logicalColumn := 187, sourceArmColumn := 49211, finalColumn := 938095 }
, { child := 12, logicalColumn := 188, sourceArmColumn := 49216, finalColumn := 938100 }
, { child := 12, logicalColumn := 189, sourceArmColumn := 49221, finalColumn := 938105 }
, { child := 12, logicalColumn := 190, sourceArmColumn := 49226, finalColumn := 938110 }
, { child := 12, logicalColumn := 191, sourceArmColumn := 49231, finalColumn := 938115 }
, { child := 12, logicalColumn := 192, sourceArmColumn := 49236, finalColumn := 938120 }
, { child := 12, logicalColumn := 193, sourceArmColumn := 49241, finalColumn := 938125 }
, { child := 12, logicalColumn := 194, sourceArmColumn := 49246, finalColumn := 938130 }
, { child := 12, logicalColumn := 195, sourceArmColumn := 49251, finalColumn := 938135 }
, { child := 12, logicalColumn := 196, sourceArmColumn := 49256, finalColumn := 938140 }
, { child := 12, logicalColumn := 197, sourceArmColumn := 49261, finalColumn := 938145 }
, { child := 12, logicalColumn := 198, sourceArmColumn := 49266, finalColumn := 938150 }
, { child := 12, logicalColumn := 199, sourceArmColumn := 49271, finalColumn := 938155 }
, { child := 12, logicalColumn := 200, sourceArmColumn := 49276, finalColumn := 938160 }
, { child := 12, logicalColumn := 201, sourceArmColumn := 49281, finalColumn := 938165 }
, { child := 12, logicalColumn := 202, sourceArmColumn := 49286, finalColumn := 938170 }
, { child := 12, logicalColumn := 203, sourceArmColumn := 49291, finalColumn := 938175 }
, { child := 12, logicalColumn := 204, sourceArmColumn := 49296, finalColumn := 938180 }
, { child := 12, logicalColumn := 205, sourceArmColumn := 49301, finalColumn := 938185 }
, { child := 12, logicalColumn := 206, sourceArmColumn := 49306, finalColumn := 938190 }
, { child := 12, logicalColumn := 207, sourceArmColumn := 49311, finalColumn := 938195 }
, { child := 12, logicalColumn := 208, sourceArmColumn := 49316, finalColumn := 938200 }
, { child := 12, logicalColumn := 209, sourceArmColumn := 49321, finalColumn := 938205 }
, { child := 12, logicalColumn := 210, sourceArmColumn := 49326, finalColumn := 938210 }
, { child := 12, logicalColumn := 211, sourceArmColumn := 49331, finalColumn := 938215 }
, { child := 12, logicalColumn := 212, sourceArmColumn := 49336, finalColumn := 938220 }
, { child := 12, logicalColumn := 213, sourceArmColumn := 49341, finalColumn := 938225 }
, { child := 12, logicalColumn := 214, sourceArmColumn := 49346, finalColumn := 938230 }
, { child := 12, logicalColumn := 215, sourceArmColumn := 49351, finalColumn := 938235 }
, { child := 12, logicalColumn := 216, sourceArmColumn := 49087, finalColumn := 937971 }
, { child := 12, logicalColumn := 217, sourceArmColumn := 49092, finalColumn := 937976 }
, { child := 12, logicalColumn := 218, sourceArmColumn := 49097, finalColumn := 937981 }
, { child := 12, logicalColumn := 219, sourceArmColumn := 49102, finalColumn := 937986 }
, { child := 12, logicalColumn := 220, sourceArmColumn := 49107, finalColumn := 937991 }
, { child := 12, logicalColumn := 221, sourceArmColumn := 49112, finalColumn := 937996 }
, { child := 12, logicalColumn := 222, sourceArmColumn := 49117, finalColumn := 938001 }
, { child := 12, logicalColumn := 223, sourceArmColumn := 49122, finalColumn := 938006 }
, { child := 12, logicalColumn := 224, sourceArmColumn := 49127, finalColumn := 938011 }
, { child := 12, logicalColumn := 225, sourceArmColumn := 49132, finalColumn := 938016 }
, { child := 12, logicalColumn := 226, sourceArmColumn := 49137, finalColumn := 938021 }
, { child := 12, logicalColumn := 227, sourceArmColumn := 49142, finalColumn := 938026 }
, { child := 12, logicalColumn := 228, sourceArmColumn := 49147, finalColumn := 938031 }
, { child := 12, logicalColumn := 229, sourceArmColumn := 49152, finalColumn := 938036 }
, { child := 12, logicalColumn := 230, sourceArmColumn := 49157, finalColumn := 938041 }
, { child := 12, logicalColumn := 231, sourceArmColumn := 49162, finalColumn := 938046 }
, { child := 12, logicalColumn := 232, sourceArmColumn := 49167, finalColumn := 938051 }
, { child := 12, logicalColumn := 233, sourceArmColumn := 49172, finalColumn := 938056 }
, { child := 12, logicalColumn := 234, sourceArmColumn := 49177, finalColumn := 938061 }
, { child := 12, logicalColumn := 235, sourceArmColumn := 49182, finalColumn := 938066 }
, { child := 12, logicalColumn := 236, sourceArmColumn := 49187, finalColumn := 938071 }
, { child := 12, logicalColumn := 237, sourceArmColumn := 49192, finalColumn := 938076 }
, { child := 12, logicalColumn := 238, sourceArmColumn := 49197, finalColumn := 938081 }
, { child := 12, logicalColumn := 239, sourceArmColumn := 49202, finalColumn := 938086 }
, { child := 12, logicalColumn := 240, sourceArmColumn := 49207, finalColumn := 938091 }
, { child := 12, logicalColumn := 241, sourceArmColumn := 49212, finalColumn := 938096 }
, { child := 12, logicalColumn := 242, sourceArmColumn := 49217, finalColumn := 938101 }
, { child := 12, logicalColumn := 243, sourceArmColumn := 49222, finalColumn := 938106 }
, { child := 12, logicalColumn := 244, sourceArmColumn := 49227, finalColumn := 938111 }
, { child := 12, logicalColumn := 245, sourceArmColumn := 49232, finalColumn := 938116 }
, { child := 12, logicalColumn := 246, sourceArmColumn := 49237, finalColumn := 938121 }
, { child := 12, logicalColumn := 247, sourceArmColumn := 49242, finalColumn := 938126 }
, { child := 12, logicalColumn := 248, sourceArmColumn := 49247, finalColumn := 938131 }
, { child := 12, logicalColumn := 249, sourceArmColumn := 49252, finalColumn := 938136 }
, { child := 12, logicalColumn := 250, sourceArmColumn := 49257, finalColumn := 938141 }
, { child := 12, logicalColumn := 251, sourceArmColumn := 49262, finalColumn := 938146 }
, { child := 12, logicalColumn := 252, sourceArmColumn := 49267, finalColumn := 938151 }
, { child := 12, logicalColumn := 253, sourceArmColumn := 49272, finalColumn := 938156 }
, { child := 12, logicalColumn := 254, sourceArmColumn := 49277, finalColumn := 938161 }
, { child := 12, logicalColumn := 255, sourceArmColumn := 49282, finalColumn := 938166 }
, { child := 12, logicalColumn := 256, sourceArmColumn := 49287, finalColumn := 938171 }
, { child := 12, logicalColumn := 257, sourceArmColumn := 49292, finalColumn := 938176 }
, { child := 12, logicalColumn := 258, sourceArmColumn := 49297, finalColumn := 938181 }
, { child := 12, logicalColumn := 259, sourceArmColumn := 49302, finalColumn := 938186 }
, { child := 12, logicalColumn := 260, sourceArmColumn := 49307, finalColumn := 938191 }
, { child := 12, logicalColumn := 261, sourceArmColumn := 49312, finalColumn := 938196 }
, { child := 12, logicalColumn := 262, sourceArmColumn := 49317, finalColumn := 938201 }
, { child := 12, logicalColumn := 263, sourceArmColumn := 49322, finalColumn := 938206 }
, { child := 12, logicalColumn := 264, sourceArmColumn := 49327, finalColumn := 938211 }
, { child := 12, logicalColumn := 265, sourceArmColumn := 49332, finalColumn := 938216 }
, { child := 12, logicalColumn := 266, sourceArmColumn := 49337, finalColumn := 938221 }
, { child := 12, logicalColumn := 267, sourceArmColumn := 49342, finalColumn := 938226 }
, { child := 12, logicalColumn := 268, sourceArmColumn := 49347, finalColumn := 938231 }
, { child := 12, logicalColumn := 269, sourceArmColumn := 49352, finalColumn := 938236 }
, { child := 13, logicalColumn := 0, sourceArmColumn := 51355, finalColumn := 1004657 }
, { child := 13, logicalColumn := 1, sourceArmColumn := 51360, finalColumn := 1004662 }
, { child := 13, logicalColumn := 2, sourceArmColumn := 51365, finalColumn := 1004667 }
, { child := 13, logicalColumn := 3, sourceArmColumn := 51370, finalColumn := 1004672 }
, { child := 13, logicalColumn := 4, sourceArmColumn := 51375, finalColumn := 1004677 }
, { child := 13, logicalColumn := 5, sourceArmColumn := 51380, finalColumn := 1004682 }
, { child := 13, logicalColumn := 6, sourceArmColumn := 51385, finalColumn := 1004687 }
, { child := 13, logicalColumn := 7, sourceArmColumn := 51390, finalColumn := 1004692 }
, { child := 13, logicalColumn := 8, sourceArmColumn := 51395, finalColumn := 1004697 }
, { child := 13, logicalColumn := 9, sourceArmColumn := 51400, finalColumn := 1004702 }
, { child := 13, logicalColumn := 10, sourceArmColumn := 51405, finalColumn := 1004707 }
, { child := 13, logicalColumn := 11, sourceArmColumn := 51410, finalColumn := 1004712 }
, { child := 13, logicalColumn := 12, sourceArmColumn := 51415, finalColumn := 1004717 }
, { child := 13, logicalColumn := 13, sourceArmColumn := 51420, finalColumn := 1004722 }
, { child := 13, logicalColumn := 14, sourceArmColumn := 51425, finalColumn := 1004727 }
, { child := 13, logicalColumn := 15, sourceArmColumn := 51430, finalColumn := 1004732 }
, { child := 13, logicalColumn := 16, sourceArmColumn := 51435, finalColumn := 1004737 }
, { child := 13, logicalColumn := 17, sourceArmColumn := 51440, finalColumn := 1004742 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk13
