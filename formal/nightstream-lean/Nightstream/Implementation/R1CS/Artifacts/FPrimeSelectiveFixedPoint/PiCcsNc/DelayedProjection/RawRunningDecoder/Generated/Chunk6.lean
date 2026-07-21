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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk6

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 5, logicalColumn := 162, sourceArmColumn := 33182, finalColumn := 471140 }
, { child := 5, logicalColumn := 163, sourceArmColumn := 33187, finalColumn := 471145 }
, { child := 5, logicalColumn := 164, sourceArmColumn := 33192, finalColumn := 471150 }
, { child := 5, logicalColumn := 165, sourceArmColumn := 33197, finalColumn := 471155 }
, { child := 5, logicalColumn := 166, sourceArmColumn := 33202, finalColumn := 471160 }
, { child := 5, logicalColumn := 167, sourceArmColumn := 33207, finalColumn := 471165 }
, { child := 5, logicalColumn := 168, sourceArmColumn := 33212, finalColumn := 471170 }
, { child := 5, logicalColumn := 169, sourceArmColumn := 33217, finalColumn := 471175 }
, { child := 5, logicalColumn := 170, sourceArmColumn := 33222, finalColumn := 471180 }
, { child := 5, logicalColumn := 171, sourceArmColumn := 33227, finalColumn := 471185 }
, { child := 5, logicalColumn := 172, sourceArmColumn := 33232, finalColumn := 471190 }
, { child := 5, logicalColumn := 173, sourceArmColumn := 33237, finalColumn := 471195 }
, { child := 5, logicalColumn := 174, sourceArmColumn := 33242, finalColumn := 471200 }
, { child := 5, logicalColumn := 175, sourceArmColumn := 33247, finalColumn := 471205 }
, { child := 5, logicalColumn := 176, sourceArmColumn := 33252, finalColumn := 471210 }
, { child := 5, logicalColumn := 177, sourceArmColumn := 33257, finalColumn := 471215 }
, { child := 5, logicalColumn := 178, sourceArmColumn := 33262, finalColumn := 471220 }
, { child := 5, logicalColumn := 179, sourceArmColumn := 33267, finalColumn := 471225 }
, { child := 5, logicalColumn := 180, sourceArmColumn := 33272, finalColumn := 471230 }
, { child := 5, logicalColumn := 181, sourceArmColumn := 33277, finalColumn := 471235 }
, { child := 5, logicalColumn := 182, sourceArmColumn := 33282, finalColumn := 471240 }
, { child := 5, logicalColumn := 183, sourceArmColumn := 33287, finalColumn := 471245 }
, { child := 5, logicalColumn := 184, sourceArmColumn := 33292, finalColumn := 471250 }
, { child := 5, logicalColumn := 185, sourceArmColumn := 33297, finalColumn := 471255 }
, { child := 5, logicalColumn := 186, sourceArmColumn := 33302, finalColumn := 471260 }
, { child := 5, logicalColumn := 187, sourceArmColumn := 33307, finalColumn := 471265 }
, { child := 5, logicalColumn := 188, sourceArmColumn := 33312, finalColumn := 471270 }
, { child := 5, logicalColumn := 189, sourceArmColumn := 33317, finalColumn := 471275 }
, { child := 5, logicalColumn := 190, sourceArmColumn := 33322, finalColumn := 471280 }
, { child := 5, logicalColumn := 191, sourceArmColumn := 33327, finalColumn := 471285 }
, { child := 5, logicalColumn := 192, sourceArmColumn := 33332, finalColumn := 471290 }
, { child := 5, logicalColumn := 193, sourceArmColumn := 33337, finalColumn := 471295 }
, { child := 5, logicalColumn := 194, sourceArmColumn := 33342, finalColumn := 471300 }
, { child := 5, logicalColumn := 195, sourceArmColumn := 33347, finalColumn := 471305 }
, { child := 5, logicalColumn := 196, sourceArmColumn := 33352, finalColumn := 471310 }
, { child := 5, logicalColumn := 197, sourceArmColumn := 33357, finalColumn := 471315 }
, { child := 5, logicalColumn := 198, sourceArmColumn := 33362, finalColumn := 471320 }
, { child := 5, logicalColumn := 199, sourceArmColumn := 33367, finalColumn := 471325 }
, { child := 5, logicalColumn := 200, sourceArmColumn := 33372, finalColumn := 471330 }
, { child := 5, logicalColumn := 201, sourceArmColumn := 33377, finalColumn := 471335 }
, { child := 5, logicalColumn := 202, sourceArmColumn := 33382, finalColumn := 471340 }
, { child := 5, logicalColumn := 203, sourceArmColumn := 33387, finalColumn := 471345 }
, { child := 5, logicalColumn := 204, sourceArmColumn := 33392, finalColumn := 471350 }
, { child := 5, logicalColumn := 205, sourceArmColumn := 33397, finalColumn := 471355 }
, { child := 5, logicalColumn := 206, sourceArmColumn := 33402, finalColumn := 471360 }
, { child := 5, logicalColumn := 207, sourceArmColumn := 33407, finalColumn := 471365 }
, { child := 5, logicalColumn := 208, sourceArmColumn := 33412, finalColumn := 471370 }
, { child := 5, logicalColumn := 209, sourceArmColumn := 33417, finalColumn := 471375 }
, { child := 5, logicalColumn := 210, sourceArmColumn := 33422, finalColumn := 471380 }
, { child := 5, logicalColumn := 211, sourceArmColumn := 33427, finalColumn := 471385 }
, { child := 5, logicalColumn := 212, sourceArmColumn := 33432, finalColumn := 471390 }
, { child := 5, logicalColumn := 213, sourceArmColumn := 33437, finalColumn := 471395 }
, { child := 5, logicalColumn := 214, sourceArmColumn := 33442, finalColumn := 471400 }
, { child := 5, logicalColumn := 215, sourceArmColumn := 33447, finalColumn := 471405 }
, { child := 5, logicalColumn := 216, sourceArmColumn := 33183, finalColumn := 471141 }
, { child := 5, logicalColumn := 217, sourceArmColumn := 33188, finalColumn := 471146 }
, { child := 5, logicalColumn := 218, sourceArmColumn := 33193, finalColumn := 471151 }
, { child := 5, logicalColumn := 219, sourceArmColumn := 33198, finalColumn := 471156 }
, { child := 5, logicalColumn := 220, sourceArmColumn := 33203, finalColumn := 471161 }
, { child := 5, logicalColumn := 221, sourceArmColumn := 33208, finalColumn := 471166 }
, { child := 5, logicalColumn := 222, sourceArmColumn := 33213, finalColumn := 471171 }
, { child := 5, logicalColumn := 223, sourceArmColumn := 33218, finalColumn := 471176 }
, { child := 5, logicalColumn := 224, sourceArmColumn := 33223, finalColumn := 471181 }
, { child := 5, logicalColumn := 225, sourceArmColumn := 33228, finalColumn := 471186 }
, { child := 5, logicalColumn := 226, sourceArmColumn := 33233, finalColumn := 471191 }
, { child := 5, logicalColumn := 227, sourceArmColumn := 33238, finalColumn := 471196 }
, { child := 5, logicalColumn := 228, sourceArmColumn := 33243, finalColumn := 471201 }
, { child := 5, logicalColumn := 229, sourceArmColumn := 33248, finalColumn := 471206 }
, { child := 5, logicalColumn := 230, sourceArmColumn := 33253, finalColumn := 471211 }
, { child := 5, logicalColumn := 231, sourceArmColumn := 33258, finalColumn := 471216 }
, { child := 5, logicalColumn := 232, sourceArmColumn := 33263, finalColumn := 471221 }
, { child := 5, logicalColumn := 233, sourceArmColumn := 33268, finalColumn := 471226 }
, { child := 5, logicalColumn := 234, sourceArmColumn := 33273, finalColumn := 471231 }
, { child := 5, logicalColumn := 235, sourceArmColumn := 33278, finalColumn := 471236 }
, { child := 5, logicalColumn := 236, sourceArmColumn := 33283, finalColumn := 471241 }
, { child := 5, logicalColumn := 237, sourceArmColumn := 33288, finalColumn := 471246 }
, { child := 5, logicalColumn := 238, sourceArmColumn := 33293, finalColumn := 471251 }
, { child := 5, logicalColumn := 239, sourceArmColumn := 33298, finalColumn := 471256 }
, { child := 5, logicalColumn := 240, sourceArmColumn := 33303, finalColumn := 471261 }
, { child := 5, logicalColumn := 241, sourceArmColumn := 33308, finalColumn := 471266 }
, { child := 5, logicalColumn := 242, sourceArmColumn := 33313, finalColumn := 471271 }
, { child := 5, logicalColumn := 243, sourceArmColumn := 33318, finalColumn := 471276 }
, { child := 5, logicalColumn := 244, sourceArmColumn := 33323, finalColumn := 471281 }
, { child := 5, logicalColumn := 245, sourceArmColumn := 33328, finalColumn := 471286 }
, { child := 5, logicalColumn := 246, sourceArmColumn := 33333, finalColumn := 471291 }
, { child := 5, logicalColumn := 247, sourceArmColumn := 33338, finalColumn := 471296 }
, { child := 5, logicalColumn := 248, sourceArmColumn := 33343, finalColumn := 471301 }
, { child := 5, logicalColumn := 249, sourceArmColumn := 33348, finalColumn := 471306 }
, { child := 5, logicalColumn := 250, sourceArmColumn := 33353, finalColumn := 471311 }
, { child := 5, logicalColumn := 251, sourceArmColumn := 33358, finalColumn := 471316 }
, { child := 5, logicalColumn := 252, sourceArmColumn := 33363, finalColumn := 471321 }
, { child := 5, logicalColumn := 253, sourceArmColumn := 33368, finalColumn := 471326 }
, { child := 5, logicalColumn := 254, sourceArmColumn := 33373, finalColumn := 471331 }
, { child := 5, logicalColumn := 255, sourceArmColumn := 33378, finalColumn := 471336 }
, { child := 5, logicalColumn := 256, sourceArmColumn := 33383, finalColumn := 471341 }
, { child := 5, logicalColumn := 257, sourceArmColumn := 33388, finalColumn := 471346 }
, { child := 5, logicalColumn := 258, sourceArmColumn := 33393, finalColumn := 471351 }
, { child := 5, logicalColumn := 259, sourceArmColumn := 33398, finalColumn := 471356 }
, { child := 5, logicalColumn := 260, sourceArmColumn := 33403, finalColumn := 471361 }
, { child := 5, logicalColumn := 261, sourceArmColumn := 33408, finalColumn := 471366 }
, { child := 5, logicalColumn := 262, sourceArmColumn := 33413, finalColumn := 471371 }
, { child := 5, logicalColumn := 263, sourceArmColumn := 33418, finalColumn := 471376 }
, { child := 5, logicalColumn := 264, sourceArmColumn := 33423, finalColumn := 471381 }
, { child := 5, logicalColumn := 265, sourceArmColumn := 33428, finalColumn := 471386 }
, { child := 5, logicalColumn := 266, sourceArmColumn := 33433, finalColumn := 471391 }
, { child := 5, logicalColumn := 267, sourceArmColumn := 33438, finalColumn := 471396 }
, { child := 5, logicalColumn := 268, sourceArmColumn := 33443, finalColumn := 471401 }
, { child := 5, logicalColumn := 269, sourceArmColumn := 33448, finalColumn := 471406 }
, { child := 6, logicalColumn := 0, sourceArmColumn := 35451, finalColumn := 537827 }
, { child := 6, logicalColumn := 1, sourceArmColumn := 35456, finalColumn := 537832 }
, { child := 6, logicalColumn := 2, sourceArmColumn := 35461, finalColumn := 537837 }
, { child := 6, logicalColumn := 3, sourceArmColumn := 35466, finalColumn := 537842 }
, { child := 6, logicalColumn := 4, sourceArmColumn := 35471, finalColumn := 537847 }
, { child := 6, logicalColumn := 5, sourceArmColumn := 35476, finalColumn := 537852 }
, { child := 6, logicalColumn := 6, sourceArmColumn := 35481, finalColumn := 537857 }
, { child := 6, logicalColumn := 7, sourceArmColumn := 35486, finalColumn := 537862 }
, { child := 6, logicalColumn := 8, sourceArmColumn := 35491, finalColumn := 537867 }
, { child := 6, logicalColumn := 9, sourceArmColumn := 35496, finalColumn := 537872 }
, { child := 6, logicalColumn := 10, sourceArmColumn := 35501, finalColumn := 537877 }
, { child := 6, logicalColumn := 11, sourceArmColumn := 35506, finalColumn := 537882 }
, { child := 6, logicalColumn := 12, sourceArmColumn := 35511, finalColumn := 537887 }
, { child := 6, logicalColumn := 13, sourceArmColumn := 35516, finalColumn := 537892 }
, { child := 6, logicalColumn := 14, sourceArmColumn := 35521, finalColumn := 537897 }
, { child := 6, logicalColumn := 15, sourceArmColumn := 35526, finalColumn := 537902 }
, { child := 6, logicalColumn := 16, sourceArmColumn := 35531, finalColumn := 537907 }
, { child := 6, logicalColumn := 17, sourceArmColumn := 35536, finalColumn := 537912 }
, { child := 6, logicalColumn := 18, sourceArmColumn := 35541, finalColumn := 537917 }
, { child := 6, logicalColumn := 19, sourceArmColumn := 35546, finalColumn := 537922 }
, { child := 6, logicalColumn := 20, sourceArmColumn := 35551, finalColumn := 537927 }
, { child := 6, logicalColumn := 21, sourceArmColumn := 35556, finalColumn := 537932 }
, { child := 6, logicalColumn := 22, sourceArmColumn := 35561, finalColumn := 537937 }
, { child := 6, logicalColumn := 23, sourceArmColumn := 35566, finalColumn := 537942 }
, { child := 6, logicalColumn := 24, sourceArmColumn := 35571, finalColumn := 537947 }
, { child := 6, logicalColumn := 25, sourceArmColumn := 35576, finalColumn := 537952 }
, { child := 6, logicalColumn := 26, sourceArmColumn := 35581, finalColumn := 537957 }
, { child := 6, logicalColumn := 27, sourceArmColumn := 35586, finalColumn := 537962 }
, { child := 6, logicalColumn := 28, sourceArmColumn := 35591, finalColumn := 537967 }
, { child := 6, logicalColumn := 29, sourceArmColumn := 35596, finalColumn := 537972 }
, { child := 6, logicalColumn := 30, sourceArmColumn := 35601, finalColumn := 537977 }
, { child := 6, logicalColumn := 31, sourceArmColumn := 35606, finalColumn := 537982 }
, { child := 6, logicalColumn := 32, sourceArmColumn := 35611, finalColumn := 537987 }
, { child := 6, logicalColumn := 33, sourceArmColumn := 35616, finalColumn := 537992 }
, { child := 6, logicalColumn := 34, sourceArmColumn := 35621, finalColumn := 537997 }
, { child := 6, logicalColumn := 35, sourceArmColumn := 35626, finalColumn := 538002 }
, { child := 6, logicalColumn := 36, sourceArmColumn := 35631, finalColumn := 538007 }
, { child := 6, logicalColumn := 37, sourceArmColumn := 35636, finalColumn := 538012 }
, { child := 6, logicalColumn := 38, sourceArmColumn := 35641, finalColumn := 538017 }
, { child := 6, logicalColumn := 39, sourceArmColumn := 35646, finalColumn := 538022 }
, { child := 6, logicalColumn := 40, sourceArmColumn := 35651, finalColumn := 538027 }
, { child := 6, logicalColumn := 41, sourceArmColumn := 35656, finalColumn := 538032 }
, { child := 6, logicalColumn := 42, sourceArmColumn := 35661, finalColumn := 538037 }
, { child := 6, logicalColumn := 43, sourceArmColumn := 35666, finalColumn := 538042 }
, { child := 6, logicalColumn := 44, sourceArmColumn := 35671, finalColumn := 538047 }
, { child := 6, logicalColumn := 45, sourceArmColumn := 35676, finalColumn := 538052 }
, { child := 6, logicalColumn := 46, sourceArmColumn := 35681, finalColumn := 538057 }
, { child := 6, logicalColumn := 47, sourceArmColumn := 35686, finalColumn := 538062 }
, { child := 6, logicalColumn := 48, sourceArmColumn := 35691, finalColumn := 538067 }
, { child := 6, logicalColumn := 49, sourceArmColumn := 35696, finalColumn := 538072 }
, { child := 6, logicalColumn := 50, sourceArmColumn := 35701, finalColumn := 538077 }
, { child := 6, logicalColumn := 51, sourceArmColumn := 35706, finalColumn := 538082 }
, { child := 6, logicalColumn := 52, sourceArmColumn := 35711, finalColumn := 538087 }
, { child := 6, logicalColumn := 53, sourceArmColumn := 35716, finalColumn := 538092 }
, { child := 6, logicalColumn := 54, sourceArmColumn := 35452, finalColumn := 537828 }
, { child := 6, logicalColumn := 55, sourceArmColumn := 35457, finalColumn := 537833 }
, { child := 6, logicalColumn := 56, sourceArmColumn := 35462, finalColumn := 537838 }
, { child := 6, logicalColumn := 57, sourceArmColumn := 35467, finalColumn := 537843 }
, { child := 6, logicalColumn := 58, sourceArmColumn := 35472, finalColumn := 537848 }
, { child := 6, logicalColumn := 59, sourceArmColumn := 35477, finalColumn := 537853 }
, { child := 6, logicalColumn := 60, sourceArmColumn := 35482, finalColumn := 537858 }
, { child := 6, logicalColumn := 61, sourceArmColumn := 35487, finalColumn := 537863 }
, { child := 6, logicalColumn := 62, sourceArmColumn := 35492, finalColumn := 537868 }
, { child := 6, logicalColumn := 63, sourceArmColumn := 35497, finalColumn := 537873 }
, { child := 6, logicalColumn := 64, sourceArmColumn := 35502, finalColumn := 537878 }
, { child := 6, logicalColumn := 65, sourceArmColumn := 35507, finalColumn := 537883 }
, { child := 6, logicalColumn := 66, sourceArmColumn := 35512, finalColumn := 537888 }
, { child := 6, logicalColumn := 67, sourceArmColumn := 35517, finalColumn := 537893 }
, { child := 6, logicalColumn := 68, sourceArmColumn := 35522, finalColumn := 537898 }
, { child := 6, logicalColumn := 69, sourceArmColumn := 35527, finalColumn := 537903 }
, { child := 6, logicalColumn := 70, sourceArmColumn := 35532, finalColumn := 537908 }
, { child := 6, logicalColumn := 71, sourceArmColumn := 35537, finalColumn := 537913 }
, { child := 6, logicalColumn := 72, sourceArmColumn := 35542, finalColumn := 537918 }
, { child := 6, logicalColumn := 73, sourceArmColumn := 35547, finalColumn := 537923 }
, { child := 6, logicalColumn := 74, sourceArmColumn := 35552, finalColumn := 537928 }
, { child := 6, logicalColumn := 75, sourceArmColumn := 35557, finalColumn := 537933 }
, { child := 6, logicalColumn := 76, sourceArmColumn := 35562, finalColumn := 537938 }
, { child := 6, logicalColumn := 77, sourceArmColumn := 35567, finalColumn := 537943 }
, { child := 6, logicalColumn := 78, sourceArmColumn := 35572, finalColumn := 537948 }
, { child := 6, logicalColumn := 79, sourceArmColumn := 35577, finalColumn := 537953 }
, { child := 6, logicalColumn := 80, sourceArmColumn := 35582, finalColumn := 537958 }
, { child := 6, logicalColumn := 81, sourceArmColumn := 35587, finalColumn := 537963 }
, { child := 6, logicalColumn := 82, sourceArmColumn := 35592, finalColumn := 537968 }
, { child := 6, logicalColumn := 83, sourceArmColumn := 35597, finalColumn := 537973 }
, { child := 6, logicalColumn := 84, sourceArmColumn := 35602, finalColumn := 537978 }
, { child := 6, logicalColumn := 85, sourceArmColumn := 35607, finalColumn := 537983 }
, { child := 6, logicalColumn := 86, sourceArmColumn := 35612, finalColumn := 537988 }
, { child := 6, logicalColumn := 87, sourceArmColumn := 35617, finalColumn := 537993 }
, { child := 6, logicalColumn := 88, sourceArmColumn := 35622, finalColumn := 537998 }
, { child := 6, logicalColumn := 89, sourceArmColumn := 35627, finalColumn := 538003 }
, { child := 6, logicalColumn := 90, sourceArmColumn := 35632, finalColumn := 538008 }
, { child := 6, logicalColumn := 91, sourceArmColumn := 35637, finalColumn := 538013 }
, { child := 6, logicalColumn := 92, sourceArmColumn := 35642, finalColumn := 538018 }
, { child := 6, logicalColumn := 93, sourceArmColumn := 35647, finalColumn := 538023 }
, { child := 6, logicalColumn := 94, sourceArmColumn := 35652, finalColumn := 538028 }
, { child := 6, logicalColumn := 95, sourceArmColumn := 35657, finalColumn := 538033 }
, { child := 6, logicalColumn := 96, sourceArmColumn := 35662, finalColumn := 538038 }
, { child := 6, logicalColumn := 97, sourceArmColumn := 35667, finalColumn := 538043 }
, { child := 6, logicalColumn := 98, sourceArmColumn := 35672, finalColumn := 538048 }
, { child := 6, logicalColumn := 99, sourceArmColumn := 35677, finalColumn := 538053 }
, { child := 6, logicalColumn := 100, sourceArmColumn := 35682, finalColumn := 538058 }
, { child := 6, logicalColumn := 101, sourceArmColumn := 35687, finalColumn := 538063 }
, { child := 6, logicalColumn := 102, sourceArmColumn := 35692, finalColumn := 538068 }
, { child := 6, logicalColumn := 103, sourceArmColumn := 35697, finalColumn := 538073 }
, { child := 6, logicalColumn := 104, sourceArmColumn := 35702, finalColumn := 538078 }
, { child := 6, logicalColumn := 105, sourceArmColumn := 35707, finalColumn := 538083 }
, { child := 6, logicalColumn := 106, sourceArmColumn := 35712, finalColumn := 538088 }
, { child := 6, logicalColumn := 107, sourceArmColumn := 35717, finalColumn := 538093 }
, { child := 6, logicalColumn := 108, sourceArmColumn := 35453, finalColumn := 537829 }
, { child := 6, logicalColumn := 109, sourceArmColumn := 35458, finalColumn := 537834 }
, { child := 6, logicalColumn := 110, sourceArmColumn := 35463, finalColumn := 537839 }
, { child := 6, logicalColumn := 111, sourceArmColumn := 35468, finalColumn := 537844 }
, { child := 6, logicalColumn := 112, sourceArmColumn := 35473, finalColumn := 537849 }
, { child := 6, logicalColumn := 113, sourceArmColumn := 35478, finalColumn := 537854 }
, { child := 6, logicalColumn := 114, sourceArmColumn := 35483, finalColumn := 537859 }
, { child := 6, logicalColumn := 115, sourceArmColumn := 35488, finalColumn := 537864 }
, { child := 6, logicalColumn := 116, sourceArmColumn := 35493, finalColumn := 537869 }
, { child := 6, logicalColumn := 117, sourceArmColumn := 35498, finalColumn := 537874 }
, { child := 6, logicalColumn := 118, sourceArmColumn := 35503, finalColumn := 537879 }
, { child := 6, logicalColumn := 119, sourceArmColumn := 35508, finalColumn := 537884 }
, { child := 6, logicalColumn := 120, sourceArmColumn := 35513, finalColumn := 537889 }
, { child := 6, logicalColumn := 121, sourceArmColumn := 35518, finalColumn := 537894 }
, { child := 6, logicalColumn := 122, sourceArmColumn := 35523, finalColumn := 537899 }
, { child := 6, logicalColumn := 123, sourceArmColumn := 35528, finalColumn := 537904 }
, { child := 6, logicalColumn := 124, sourceArmColumn := 35533, finalColumn := 537909 }
, { child := 6, logicalColumn := 125, sourceArmColumn := 35538, finalColumn := 537914 }
, { child := 6, logicalColumn := 126, sourceArmColumn := 35543, finalColumn := 537919 }
, { child := 6, logicalColumn := 127, sourceArmColumn := 35548, finalColumn := 537924 }
, { child := 6, logicalColumn := 128, sourceArmColumn := 35553, finalColumn := 537929 }
, { child := 6, logicalColumn := 129, sourceArmColumn := 35558, finalColumn := 537934 }
, { child := 6, logicalColumn := 130, sourceArmColumn := 35563, finalColumn := 537939 }
, { child := 6, logicalColumn := 131, sourceArmColumn := 35568, finalColumn := 537944 }
, { child := 6, logicalColumn := 132, sourceArmColumn := 35573, finalColumn := 537949 }
, { child := 6, logicalColumn := 133, sourceArmColumn := 35578, finalColumn := 537954 }
, { child := 6, logicalColumn := 134, sourceArmColumn := 35583, finalColumn := 537959 }
, { child := 6, logicalColumn := 135, sourceArmColumn := 35588, finalColumn := 537964 }
, { child := 6, logicalColumn := 136, sourceArmColumn := 35593, finalColumn := 537969 }
, { child := 6, logicalColumn := 137, sourceArmColumn := 35598, finalColumn := 537974 }
, { child := 6, logicalColumn := 138, sourceArmColumn := 35603, finalColumn := 537979 }
, { child := 6, logicalColumn := 139, sourceArmColumn := 35608, finalColumn := 537984 }
, { child := 6, logicalColumn := 140, sourceArmColumn := 35613, finalColumn := 537989 }
, { child := 6, logicalColumn := 141, sourceArmColumn := 35618, finalColumn := 537994 }
, { child := 6, logicalColumn := 142, sourceArmColumn := 35623, finalColumn := 537999 }
, { child := 6, logicalColumn := 143, sourceArmColumn := 35628, finalColumn := 538004 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk6
