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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk0

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 0, logicalColumn := 0, sourceArmColumn := 21819, finalColumn := 133997 }
, { child := 0, logicalColumn := 1, sourceArmColumn := 21824, finalColumn := 134002 }
, { child := 0, logicalColumn := 2, sourceArmColumn := 21829, finalColumn := 134007 }
, { child := 0, logicalColumn := 3, sourceArmColumn := 21834, finalColumn := 134012 }
, { child := 0, logicalColumn := 4, sourceArmColumn := 21839, finalColumn := 134017 }
, { child := 0, logicalColumn := 5, sourceArmColumn := 21844, finalColumn := 134022 }
, { child := 0, logicalColumn := 6, sourceArmColumn := 21849, finalColumn := 134027 }
, { child := 0, logicalColumn := 7, sourceArmColumn := 21854, finalColumn := 134032 }
, { child := 0, logicalColumn := 8, sourceArmColumn := 21859, finalColumn := 134037 }
, { child := 0, logicalColumn := 9, sourceArmColumn := 21864, finalColumn := 134042 }
, { child := 0, logicalColumn := 10, sourceArmColumn := 21869, finalColumn := 134047 }
, { child := 0, logicalColumn := 11, sourceArmColumn := 21874, finalColumn := 134052 }
, { child := 0, logicalColumn := 12, sourceArmColumn := 21879, finalColumn := 134057 }
, { child := 0, logicalColumn := 13, sourceArmColumn := 21884, finalColumn := 134062 }
, { child := 0, logicalColumn := 14, sourceArmColumn := 21889, finalColumn := 134067 }
, { child := 0, logicalColumn := 15, sourceArmColumn := 21894, finalColumn := 134072 }
, { child := 0, logicalColumn := 16, sourceArmColumn := 21899, finalColumn := 134077 }
, { child := 0, logicalColumn := 17, sourceArmColumn := 21904, finalColumn := 134082 }
, { child := 0, logicalColumn := 18, sourceArmColumn := 21909, finalColumn := 134087 }
, { child := 0, logicalColumn := 19, sourceArmColumn := 21914, finalColumn := 134092 }
, { child := 0, logicalColumn := 20, sourceArmColumn := 21919, finalColumn := 134097 }
, { child := 0, logicalColumn := 21, sourceArmColumn := 21924, finalColumn := 134102 }
, { child := 0, logicalColumn := 22, sourceArmColumn := 21929, finalColumn := 134107 }
, { child := 0, logicalColumn := 23, sourceArmColumn := 21934, finalColumn := 134112 }
, { child := 0, logicalColumn := 24, sourceArmColumn := 21939, finalColumn := 134117 }
, { child := 0, logicalColumn := 25, sourceArmColumn := 21944, finalColumn := 134122 }
, { child := 0, logicalColumn := 26, sourceArmColumn := 21949, finalColumn := 134127 }
, { child := 0, logicalColumn := 27, sourceArmColumn := 21954, finalColumn := 134132 }
, { child := 0, logicalColumn := 28, sourceArmColumn := 21959, finalColumn := 134137 }
, { child := 0, logicalColumn := 29, sourceArmColumn := 21964, finalColumn := 134142 }
, { child := 0, logicalColumn := 30, sourceArmColumn := 21969, finalColumn := 134147 }
, { child := 0, logicalColumn := 31, sourceArmColumn := 21974, finalColumn := 134152 }
, { child := 0, logicalColumn := 32, sourceArmColumn := 21979, finalColumn := 134157 }
, { child := 0, logicalColumn := 33, sourceArmColumn := 21984, finalColumn := 134162 }
, { child := 0, logicalColumn := 34, sourceArmColumn := 21989, finalColumn := 134167 }
, { child := 0, logicalColumn := 35, sourceArmColumn := 21994, finalColumn := 134172 }
, { child := 0, logicalColumn := 36, sourceArmColumn := 21999, finalColumn := 134177 }
, { child := 0, logicalColumn := 37, sourceArmColumn := 22004, finalColumn := 134182 }
, { child := 0, logicalColumn := 38, sourceArmColumn := 22009, finalColumn := 134187 }
, { child := 0, logicalColumn := 39, sourceArmColumn := 22014, finalColumn := 134192 }
, { child := 0, logicalColumn := 40, sourceArmColumn := 22019, finalColumn := 134197 }
, { child := 0, logicalColumn := 41, sourceArmColumn := 22024, finalColumn := 134202 }
, { child := 0, logicalColumn := 42, sourceArmColumn := 22029, finalColumn := 134207 }
, { child := 0, logicalColumn := 43, sourceArmColumn := 22034, finalColumn := 134212 }
, { child := 0, logicalColumn := 44, sourceArmColumn := 22039, finalColumn := 134217 }
, { child := 0, logicalColumn := 45, sourceArmColumn := 22044, finalColumn := 134222 }
, { child := 0, logicalColumn := 46, sourceArmColumn := 22049, finalColumn := 134227 }
, { child := 0, logicalColumn := 47, sourceArmColumn := 22054, finalColumn := 134232 }
, { child := 0, logicalColumn := 48, sourceArmColumn := 22059, finalColumn := 134237 }
, { child := 0, logicalColumn := 49, sourceArmColumn := 22064, finalColumn := 134242 }
, { child := 0, logicalColumn := 50, sourceArmColumn := 22069, finalColumn := 134247 }
, { child := 0, logicalColumn := 51, sourceArmColumn := 22074, finalColumn := 134252 }
, { child := 0, logicalColumn := 52, sourceArmColumn := 22079, finalColumn := 134257 }
, { child := 0, logicalColumn := 53, sourceArmColumn := 22084, finalColumn := 134262 }
, { child := 0, logicalColumn := 54, sourceArmColumn := 21820, finalColumn := 133998 }
, { child := 0, logicalColumn := 55, sourceArmColumn := 21825, finalColumn := 134003 }
, { child := 0, logicalColumn := 56, sourceArmColumn := 21830, finalColumn := 134008 }
, { child := 0, logicalColumn := 57, sourceArmColumn := 21835, finalColumn := 134013 }
, { child := 0, logicalColumn := 58, sourceArmColumn := 21840, finalColumn := 134018 }
, { child := 0, logicalColumn := 59, sourceArmColumn := 21845, finalColumn := 134023 }
, { child := 0, logicalColumn := 60, sourceArmColumn := 21850, finalColumn := 134028 }
, { child := 0, logicalColumn := 61, sourceArmColumn := 21855, finalColumn := 134033 }
, { child := 0, logicalColumn := 62, sourceArmColumn := 21860, finalColumn := 134038 }
, { child := 0, logicalColumn := 63, sourceArmColumn := 21865, finalColumn := 134043 }
, { child := 0, logicalColumn := 64, sourceArmColumn := 21870, finalColumn := 134048 }
, { child := 0, logicalColumn := 65, sourceArmColumn := 21875, finalColumn := 134053 }
, { child := 0, logicalColumn := 66, sourceArmColumn := 21880, finalColumn := 134058 }
, { child := 0, logicalColumn := 67, sourceArmColumn := 21885, finalColumn := 134063 }
, { child := 0, logicalColumn := 68, sourceArmColumn := 21890, finalColumn := 134068 }
, { child := 0, logicalColumn := 69, sourceArmColumn := 21895, finalColumn := 134073 }
, { child := 0, logicalColumn := 70, sourceArmColumn := 21900, finalColumn := 134078 }
, { child := 0, logicalColumn := 71, sourceArmColumn := 21905, finalColumn := 134083 }
, { child := 0, logicalColumn := 72, sourceArmColumn := 21910, finalColumn := 134088 }
, { child := 0, logicalColumn := 73, sourceArmColumn := 21915, finalColumn := 134093 }
, { child := 0, logicalColumn := 74, sourceArmColumn := 21920, finalColumn := 134098 }
, { child := 0, logicalColumn := 75, sourceArmColumn := 21925, finalColumn := 134103 }
, { child := 0, logicalColumn := 76, sourceArmColumn := 21930, finalColumn := 134108 }
, { child := 0, logicalColumn := 77, sourceArmColumn := 21935, finalColumn := 134113 }
, { child := 0, logicalColumn := 78, sourceArmColumn := 21940, finalColumn := 134118 }
, { child := 0, logicalColumn := 79, sourceArmColumn := 21945, finalColumn := 134123 }
, { child := 0, logicalColumn := 80, sourceArmColumn := 21950, finalColumn := 134128 }
, { child := 0, logicalColumn := 81, sourceArmColumn := 21955, finalColumn := 134133 }
, { child := 0, logicalColumn := 82, sourceArmColumn := 21960, finalColumn := 134138 }
, { child := 0, logicalColumn := 83, sourceArmColumn := 21965, finalColumn := 134143 }
, { child := 0, logicalColumn := 84, sourceArmColumn := 21970, finalColumn := 134148 }
, { child := 0, logicalColumn := 85, sourceArmColumn := 21975, finalColumn := 134153 }
, { child := 0, logicalColumn := 86, sourceArmColumn := 21980, finalColumn := 134158 }
, { child := 0, logicalColumn := 87, sourceArmColumn := 21985, finalColumn := 134163 }
, { child := 0, logicalColumn := 88, sourceArmColumn := 21990, finalColumn := 134168 }
, { child := 0, logicalColumn := 89, sourceArmColumn := 21995, finalColumn := 134173 }
, { child := 0, logicalColumn := 90, sourceArmColumn := 22000, finalColumn := 134178 }
, { child := 0, logicalColumn := 91, sourceArmColumn := 22005, finalColumn := 134183 }
, { child := 0, logicalColumn := 92, sourceArmColumn := 22010, finalColumn := 134188 }
, { child := 0, logicalColumn := 93, sourceArmColumn := 22015, finalColumn := 134193 }
, { child := 0, logicalColumn := 94, sourceArmColumn := 22020, finalColumn := 134198 }
, { child := 0, logicalColumn := 95, sourceArmColumn := 22025, finalColumn := 134203 }
, { child := 0, logicalColumn := 96, sourceArmColumn := 22030, finalColumn := 134208 }
, { child := 0, logicalColumn := 97, sourceArmColumn := 22035, finalColumn := 134213 }
, { child := 0, logicalColumn := 98, sourceArmColumn := 22040, finalColumn := 134218 }
, { child := 0, logicalColumn := 99, sourceArmColumn := 22045, finalColumn := 134223 }
, { child := 0, logicalColumn := 100, sourceArmColumn := 22050, finalColumn := 134228 }
, { child := 0, logicalColumn := 101, sourceArmColumn := 22055, finalColumn := 134233 }
, { child := 0, logicalColumn := 102, sourceArmColumn := 22060, finalColumn := 134238 }
, { child := 0, logicalColumn := 103, sourceArmColumn := 22065, finalColumn := 134243 }
, { child := 0, logicalColumn := 104, sourceArmColumn := 22070, finalColumn := 134248 }
, { child := 0, logicalColumn := 105, sourceArmColumn := 22075, finalColumn := 134253 }
, { child := 0, logicalColumn := 106, sourceArmColumn := 22080, finalColumn := 134258 }
, { child := 0, logicalColumn := 107, sourceArmColumn := 22085, finalColumn := 134263 }
, { child := 0, logicalColumn := 108, sourceArmColumn := 21821, finalColumn := 133999 }
, { child := 0, logicalColumn := 109, sourceArmColumn := 21826, finalColumn := 134004 }
, { child := 0, logicalColumn := 110, sourceArmColumn := 21831, finalColumn := 134009 }
, { child := 0, logicalColumn := 111, sourceArmColumn := 21836, finalColumn := 134014 }
, { child := 0, logicalColumn := 112, sourceArmColumn := 21841, finalColumn := 134019 }
, { child := 0, logicalColumn := 113, sourceArmColumn := 21846, finalColumn := 134024 }
, { child := 0, logicalColumn := 114, sourceArmColumn := 21851, finalColumn := 134029 }
, { child := 0, logicalColumn := 115, sourceArmColumn := 21856, finalColumn := 134034 }
, { child := 0, logicalColumn := 116, sourceArmColumn := 21861, finalColumn := 134039 }
, { child := 0, logicalColumn := 117, sourceArmColumn := 21866, finalColumn := 134044 }
, { child := 0, logicalColumn := 118, sourceArmColumn := 21871, finalColumn := 134049 }
, { child := 0, logicalColumn := 119, sourceArmColumn := 21876, finalColumn := 134054 }
, { child := 0, logicalColumn := 120, sourceArmColumn := 21881, finalColumn := 134059 }
, { child := 0, logicalColumn := 121, sourceArmColumn := 21886, finalColumn := 134064 }
, { child := 0, logicalColumn := 122, sourceArmColumn := 21891, finalColumn := 134069 }
, { child := 0, logicalColumn := 123, sourceArmColumn := 21896, finalColumn := 134074 }
, { child := 0, logicalColumn := 124, sourceArmColumn := 21901, finalColumn := 134079 }
, { child := 0, logicalColumn := 125, sourceArmColumn := 21906, finalColumn := 134084 }
, { child := 0, logicalColumn := 126, sourceArmColumn := 21911, finalColumn := 134089 }
, { child := 0, logicalColumn := 127, sourceArmColumn := 21916, finalColumn := 134094 }
, { child := 0, logicalColumn := 128, sourceArmColumn := 21921, finalColumn := 134099 }
, { child := 0, logicalColumn := 129, sourceArmColumn := 21926, finalColumn := 134104 }
, { child := 0, logicalColumn := 130, sourceArmColumn := 21931, finalColumn := 134109 }
, { child := 0, logicalColumn := 131, sourceArmColumn := 21936, finalColumn := 134114 }
, { child := 0, logicalColumn := 132, sourceArmColumn := 21941, finalColumn := 134119 }
, { child := 0, logicalColumn := 133, sourceArmColumn := 21946, finalColumn := 134124 }
, { child := 0, logicalColumn := 134, sourceArmColumn := 21951, finalColumn := 134129 }
, { child := 0, logicalColumn := 135, sourceArmColumn := 21956, finalColumn := 134134 }
, { child := 0, logicalColumn := 136, sourceArmColumn := 21961, finalColumn := 134139 }
, { child := 0, logicalColumn := 137, sourceArmColumn := 21966, finalColumn := 134144 }
, { child := 0, logicalColumn := 138, sourceArmColumn := 21971, finalColumn := 134149 }
, { child := 0, logicalColumn := 139, sourceArmColumn := 21976, finalColumn := 134154 }
, { child := 0, logicalColumn := 140, sourceArmColumn := 21981, finalColumn := 134159 }
, { child := 0, logicalColumn := 141, sourceArmColumn := 21986, finalColumn := 134164 }
, { child := 0, logicalColumn := 142, sourceArmColumn := 21991, finalColumn := 134169 }
, { child := 0, logicalColumn := 143, sourceArmColumn := 21996, finalColumn := 134174 }
, { child := 0, logicalColumn := 144, sourceArmColumn := 22001, finalColumn := 134179 }
, { child := 0, logicalColumn := 145, sourceArmColumn := 22006, finalColumn := 134184 }
, { child := 0, logicalColumn := 146, sourceArmColumn := 22011, finalColumn := 134189 }
, { child := 0, logicalColumn := 147, sourceArmColumn := 22016, finalColumn := 134194 }
, { child := 0, logicalColumn := 148, sourceArmColumn := 22021, finalColumn := 134199 }
, { child := 0, logicalColumn := 149, sourceArmColumn := 22026, finalColumn := 134204 }
, { child := 0, logicalColumn := 150, sourceArmColumn := 22031, finalColumn := 134209 }
, { child := 0, logicalColumn := 151, sourceArmColumn := 22036, finalColumn := 134214 }
, { child := 0, logicalColumn := 152, sourceArmColumn := 22041, finalColumn := 134219 }
, { child := 0, logicalColumn := 153, sourceArmColumn := 22046, finalColumn := 134224 }
, { child := 0, logicalColumn := 154, sourceArmColumn := 22051, finalColumn := 134229 }
, { child := 0, logicalColumn := 155, sourceArmColumn := 22056, finalColumn := 134234 }
, { child := 0, logicalColumn := 156, sourceArmColumn := 22061, finalColumn := 134239 }
, { child := 0, logicalColumn := 157, sourceArmColumn := 22066, finalColumn := 134244 }
, { child := 0, logicalColumn := 158, sourceArmColumn := 22071, finalColumn := 134249 }
, { child := 0, logicalColumn := 159, sourceArmColumn := 22076, finalColumn := 134254 }
, { child := 0, logicalColumn := 160, sourceArmColumn := 22081, finalColumn := 134259 }
, { child := 0, logicalColumn := 161, sourceArmColumn := 22086, finalColumn := 134264 }
, { child := 0, logicalColumn := 162, sourceArmColumn := 21822, finalColumn := 134000 }
, { child := 0, logicalColumn := 163, sourceArmColumn := 21827, finalColumn := 134005 }
, { child := 0, logicalColumn := 164, sourceArmColumn := 21832, finalColumn := 134010 }
, { child := 0, logicalColumn := 165, sourceArmColumn := 21837, finalColumn := 134015 }
, { child := 0, logicalColumn := 166, sourceArmColumn := 21842, finalColumn := 134020 }
, { child := 0, logicalColumn := 167, sourceArmColumn := 21847, finalColumn := 134025 }
, { child := 0, logicalColumn := 168, sourceArmColumn := 21852, finalColumn := 134030 }
, { child := 0, logicalColumn := 169, sourceArmColumn := 21857, finalColumn := 134035 }
, { child := 0, logicalColumn := 170, sourceArmColumn := 21862, finalColumn := 134040 }
, { child := 0, logicalColumn := 171, sourceArmColumn := 21867, finalColumn := 134045 }
, { child := 0, logicalColumn := 172, sourceArmColumn := 21872, finalColumn := 134050 }
, { child := 0, logicalColumn := 173, sourceArmColumn := 21877, finalColumn := 134055 }
, { child := 0, logicalColumn := 174, sourceArmColumn := 21882, finalColumn := 134060 }
, { child := 0, logicalColumn := 175, sourceArmColumn := 21887, finalColumn := 134065 }
, { child := 0, logicalColumn := 176, sourceArmColumn := 21892, finalColumn := 134070 }
, { child := 0, logicalColumn := 177, sourceArmColumn := 21897, finalColumn := 134075 }
, { child := 0, logicalColumn := 178, sourceArmColumn := 21902, finalColumn := 134080 }
, { child := 0, logicalColumn := 179, sourceArmColumn := 21907, finalColumn := 134085 }
, { child := 0, logicalColumn := 180, sourceArmColumn := 21912, finalColumn := 134090 }
, { child := 0, logicalColumn := 181, sourceArmColumn := 21917, finalColumn := 134095 }
, { child := 0, logicalColumn := 182, sourceArmColumn := 21922, finalColumn := 134100 }
, { child := 0, logicalColumn := 183, sourceArmColumn := 21927, finalColumn := 134105 }
, { child := 0, logicalColumn := 184, sourceArmColumn := 21932, finalColumn := 134110 }
, { child := 0, logicalColumn := 185, sourceArmColumn := 21937, finalColumn := 134115 }
, { child := 0, logicalColumn := 186, sourceArmColumn := 21942, finalColumn := 134120 }
, { child := 0, logicalColumn := 187, sourceArmColumn := 21947, finalColumn := 134125 }
, { child := 0, logicalColumn := 188, sourceArmColumn := 21952, finalColumn := 134130 }
, { child := 0, logicalColumn := 189, sourceArmColumn := 21957, finalColumn := 134135 }
, { child := 0, logicalColumn := 190, sourceArmColumn := 21962, finalColumn := 134140 }
, { child := 0, logicalColumn := 191, sourceArmColumn := 21967, finalColumn := 134145 }
, { child := 0, logicalColumn := 192, sourceArmColumn := 21972, finalColumn := 134150 }
, { child := 0, logicalColumn := 193, sourceArmColumn := 21977, finalColumn := 134155 }
, { child := 0, logicalColumn := 194, sourceArmColumn := 21982, finalColumn := 134160 }
, { child := 0, logicalColumn := 195, sourceArmColumn := 21987, finalColumn := 134165 }
, { child := 0, logicalColumn := 196, sourceArmColumn := 21992, finalColumn := 134170 }
, { child := 0, logicalColumn := 197, sourceArmColumn := 21997, finalColumn := 134175 }
, { child := 0, logicalColumn := 198, sourceArmColumn := 22002, finalColumn := 134180 }
, { child := 0, logicalColumn := 199, sourceArmColumn := 22007, finalColumn := 134185 }
, { child := 0, logicalColumn := 200, sourceArmColumn := 22012, finalColumn := 134190 }
, { child := 0, logicalColumn := 201, sourceArmColumn := 22017, finalColumn := 134195 }
, { child := 0, logicalColumn := 202, sourceArmColumn := 22022, finalColumn := 134200 }
, { child := 0, logicalColumn := 203, sourceArmColumn := 22027, finalColumn := 134205 }
, { child := 0, logicalColumn := 204, sourceArmColumn := 22032, finalColumn := 134210 }
, { child := 0, logicalColumn := 205, sourceArmColumn := 22037, finalColumn := 134215 }
, { child := 0, logicalColumn := 206, sourceArmColumn := 22042, finalColumn := 134220 }
, { child := 0, logicalColumn := 207, sourceArmColumn := 22047, finalColumn := 134225 }
, { child := 0, logicalColumn := 208, sourceArmColumn := 22052, finalColumn := 134230 }
, { child := 0, logicalColumn := 209, sourceArmColumn := 22057, finalColumn := 134235 }
, { child := 0, logicalColumn := 210, sourceArmColumn := 22062, finalColumn := 134240 }
, { child := 0, logicalColumn := 211, sourceArmColumn := 22067, finalColumn := 134245 }
, { child := 0, logicalColumn := 212, sourceArmColumn := 22072, finalColumn := 134250 }
, { child := 0, logicalColumn := 213, sourceArmColumn := 22077, finalColumn := 134255 }
, { child := 0, logicalColumn := 214, sourceArmColumn := 22082, finalColumn := 134260 }
, { child := 0, logicalColumn := 215, sourceArmColumn := 22087, finalColumn := 134265 }
, { child := 0, logicalColumn := 216, sourceArmColumn := 21823, finalColumn := 134001 }
, { child := 0, logicalColumn := 217, sourceArmColumn := 21828, finalColumn := 134006 }
, { child := 0, logicalColumn := 218, sourceArmColumn := 21833, finalColumn := 134011 }
, { child := 0, logicalColumn := 219, sourceArmColumn := 21838, finalColumn := 134016 }
, { child := 0, logicalColumn := 220, sourceArmColumn := 21843, finalColumn := 134021 }
, { child := 0, logicalColumn := 221, sourceArmColumn := 21848, finalColumn := 134026 }
, { child := 0, logicalColumn := 222, sourceArmColumn := 21853, finalColumn := 134031 }
, { child := 0, logicalColumn := 223, sourceArmColumn := 21858, finalColumn := 134036 }
, { child := 0, logicalColumn := 224, sourceArmColumn := 21863, finalColumn := 134041 }
, { child := 0, logicalColumn := 225, sourceArmColumn := 21868, finalColumn := 134046 }
, { child := 0, logicalColumn := 226, sourceArmColumn := 21873, finalColumn := 134051 }
, { child := 0, logicalColumn := 227, sourceArmColumn := 21878, finalColumn := 134056 }
, { child := 0, logicalColumn := 228, sourceArmColumn := 21883, finalColumn := 134061 }
, { child := 0, logicalColumn := 229, sourceArmColumn := 21888, finalColumn := 134066 }
, { child := 0, logicalColumn := 230, sourceArmColumn := 21893, finalColumn := 134071 }
, { child := 0, logicalColumn := 231, sourceArmColumn := 21898, finalColumn := 134076 }
, { child := 0, logicalColumn := 232, sourceArmColumn := 21903, finalColumn := 134081 }
, { child := 0, logicalColumn := 233, sourceArmColumn := 21908, finalColumn := 134086 }
, { child := 0, logicalColumn := 234, sourceArmColumn := 21913, finalColumn := 134091 }
, { child := 0, logicalColumn := 235, sourceArmColumn := 21918, finalColumn := 134096 }
, { child := 0, logicalColumn := 236, sourceArmColumn := 21923, finalColumn := 134101 }
, { child := 0, logicalColumn := 237, sourceArmColumn := 21928, finalColumn := 134106 }
, { child := 0, logicalColumn := 238, sourceArmColumn := 21933, finalColumn := 134111 }
, { child := 0, logicalColumn := 239, sourceArmColumn := 21938, finalColumn := 134116 }
, { child := 0, logicalColumn := 240, sourceArmColumn := 21943, finalColumn := 134121 }
, { child := 0, logicalColumn := 241, sourceArmColumn := 21948, finalColumn := 134126 }
, { child := 0, logicalColumn := 242, sourceArmColumn := 21953, finalColumn := 134131 }
, { child := 0, logicalColumn := 243, sourceArmColumn := 21958, finalColumn := 134136 }
, { child := 0, logicalColumn := 244, sourceArmColumn := 21963, finalColumn := 134141 }
, { child := 0, logicalColumn := 245, sourceArmColumn := 21968, finalColumn := 134146 }
, { child := 0, logicalColumn := 246, sourceArmColumn := 21973, finalColumn := 134151 }
, { child := 0, logicalColumn := 247, sourceArmColumn := 21978, finalColumn := 134156 }
, { child := 0, logicalColumn := 248, sourceArmColumn := 21983, finalColumn := 134161 }
, { child := 0, logicalColumn := 249, sourceArmColumn := 21988, finalColumn := 134166 }
, { child := 0, logicalColumn := 250, sourceArmColumn := 21993, finalColumn := 134171 }
, { child := 0, logicalColumn := 251, sourceArmColumn := 21998, finalColumn := 134176 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk0
