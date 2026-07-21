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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk9

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 8, logicalColumn := 108, sourceArmColumn := 39997, finalColumn := 671209 }
, { child := 8, logicalColumn := 109, sourceArmColumn := 40002, finalColumn := 671214 }
, { child := 8, logicalColumn := 110, sourceArmColumn := 40007, finalColumn := 671219 }
, { child := 8, logicalColumn := 111, sourceArmColumn := 40012, finalColumn := 671224 }
, { child := 8, logicalColumn := 112, sourceArmColumn := 40017, finalColumn := 671229 }
, { child := 8, logicalColumn := 113, sourceArmColumn := 40022, finalColumn := 671234 }
, { child := 8, logicalColumn := 114, sourceArmColumn := 40027, finalColumn := 671239 }
, { child := 8, logicalColumn := 115, sourceArmColumn := 40032, finalColumn := 671244 }
, { child := 8, logicalColumn := 116, sourceArmColumn := 40037, finalColumn := 671249 }
, { child := 8, logicalColumn := 117, sourceArmColumn := 40042, finalColumn := 671254 }
, { child := 8, logicalColumn := 118, sourceArmColumn := 40047, finalColumn := 671259 }
, { child := 8, logicalColumn := 119, sourceArmColumn := 40052, finalColumn := 671264 }
, { child := 8, logicalColumn := 120, sourceArmColumn := 40057, finalColumn := 671269 }
, { child := 8, logicalColumn := 121, sourceArmColumn := 40062, finalColumn := 671274 }
, { child := 8, logicalColumn := 122, sourceArmColumn := 40067, finalColumn := 671279 }
, { child := 8, logicalColumn := 123, sourceArmColumn := 40072, finalColumn := 671284 }
, { child := 8, logicalColumn := 124, sourceArmColumn := 40077, finalColumn := 671289 }
, { child := 8, logicalColumn := 125, sourceArmColumn := 40082, finalColumn := 671294 }
, { child := 8, logicalColumn := 126, sourceArmColumn := 40087, finalColumn := 671299 }
, { child := 8, logicalColumn := 127, sourceArmColumn := 40092, finalColumn := 671304 }
, { child := 8, logicalColumn := 128, sourceArmColumn := 40097, finalColumn := 671309 }
, { child := 8, logicalColumn := 129, sourceArmColumn := 40102, finalColumn := 671314 }
, { child := 8, logicalColumn := 130, sourceArmColumn := 40107, finalColumn := 671319 }
, { child := 8, logicalColumn := 131, sourceArmColumn := 40112, finalColumn := 671324 }
, { child := 8, logicalColumn := 132, sourceArmColumn := 40117, finalColumn := 671329 }
, { child := 8, logicalColumn := 133, sourceArmColumn := 40122, finalColumn := 671334 }
, { child := 8, logicalColumn := 134, sourceArmColumn := 40127, finalColumn := 671339 }
, { child := 8, logicalColumn := 135, sourceArmColumn := 40132, finalColumn := 671344 }
, { child := 8, logicalColumn := 136, sourceArmColumn := 40137, finalColumn := 671349 }
, { child := 8, logicalColumn := 137, sourceArmColumn := 40142, finalColumn := 671354 }
, { child := 8, logicalColumn := 138, sourceArmColumn := 40147, finalColumn := 671359 }
, { child := 8, logicalColumn := 139, sourceArmColumn := 40152, finalColumn := 671364 }
, { child := 8, logicalColumn := 140, sourceArmColumn := 40157, finalColumn := 671369 }
, { child := 8, logicalColumn := 141, sourceArmColumn := 40162, finalColumn := 671374 }
, { child := 8, logicalColumn := 142, sourceArmColumn := 40167, finalColumn := 671379 }
, { child := 8, logicalColumn := 143, sourceArmColumn := 40172, finalColumn := 671384 }
, { child := 8, logicalColumn := 144, sourceArmColumn := 40177, finalColumn := 671389 }
, { child := 8, logicalColumn := 145, sourceArmColumn := 40182, finalColumn := 671394 }
, { child := 8, logicalColumn := 146, sourceArmColumn := 40187, finalColumn := 671399 }
, { child := 8, logicalColumn := 147, sourceArmColumn := 40192, finalColumn := 671404 }
, { child := 8, logicalColumn := 148, sourceArmColumn := 40197, finalColumn := 671409 }
, { child := 8, logicalColumn := 149, sourceArmColumn := 40202, finalColumn := 671414 }
, { child := 8, logicalColumn := 150, sourceArmColumn := 40207, finalColumn := 671419 }
, { child := 8, logicalColumn := 151, sourceArmColumn := 40212, finalColumn := 671424 }
, { child := 8, logicalColumn := 152, sourceArmColumn := 40217, finalColumn := 671429 }
, { child := 8, logicalColumn := 153, sourceArmColumn := 40222, finalColumn := 671434 }
, { child := 8, logicalColumn := 154, sourceArmColumn := 40227, finalColumn := 671439 }
, { child := 8, logicalColumn := 155, sourceArmColumn := 40232, finalColumn := 671444 }
, { child := 8, logicalColumn := 156, sourceArmColumn := 40237, finalColumn := 671449 }
, { child := 8, logicalColumn := 157, sourceArmColumn := 40242, finalColumn := 671454 }
, { child := 8, logicalColumn := 158, sourceArmColumn := 40247, finalColumn := 671459 }
, { child := 8, logicalColumn := 159, sourceArmColumn := 40252, finalColumn := 671464 }
, { child := 8, logicalColumn := 160, sourceArmColumn := 40257, finalColumn := 671469 }
, { child := 8, logicalColumn := 161, sourceArmColumn := 40262, finalColumn := 671474 }
, { child := 8, logicalColumn := 162, sourceArmColumn := 39998, finalColumn := 671210 }
, { child := 8, logicalColumn := 163, sourceArmColumn := 40003, finalColumn := 671215 }
, { child := 8, logicalColumn := 164, sourceArmColumn := 40008, finalColumn := 671220 }
, { child := 8, logicalColumn := 165, sourceArmColumn := 40013, finalColumn := 671225 }
, { child := 8, logicalColumn := 166, sourceArmColumn := 40018, finalColumn := 671230 }
, { child := 8, logicalColumn := 167, sourceArmColumn := 40023, finalColumn := 671235 }
, { child := 8, logicalColumn := 168, sourceArmColumn := 40028, finalColumn := 671240 }
, { child := 8, logicalColumn := 169, sourceArmColumn := 40033, finalColumn := 671245 }
, { child := 8, logicalColumn := 170, sourceArmColumn := 40038, finalColumn := 671250 }
, { child := 8, logicalColumn := 171, sourceArmColumn := 40043, finalColumn := 671255 }
, { child := 8, logicalColumn := 172, sourceArmColumn := 40048, finalColumn := 671260 }
, { child := 8, logicalColumn := 173, sourceArmColumn := 40053, finalColumn := 671265 }
, { child := 8, logicalColumn := 174, sourceArmColumn := 40058, finalColumn := 671270 }
, { child := 8, logicalColumn := 175, sourceArmColumn := 40063, finalColumn := 671275 }
, { child := 8, logicalColumn := 176, sourceArmColumn := 40068, finalColumn := 671280 }
, { child := 8, logicalColumn := 177, sourceArmColumn := 40073, finalColumn := 671285 }
, { child := 8, logicalColumn := 178, sourceArmColumn := 40078, finalColumn := 671290 }
, { child := 8, logicalColumn := 179, sourceArmColumn := 40083, finalColumn := 671295 }
, { child := 8, logicalColumn := 180, sourceArmColumn := 40088, finalColumn := 671300 }
, { child := 8, logicalColumn := 181, sourceArmColumn := 40093, finalColumn := 671305 }
, { child := 8, logicalColumn := 182, sourceArmColumn := 40098, finalColumn := 671310 }
, { child := 8, logicalColumn := 183, sourceArmColumn := 40103, finalColumn := 671315 }
, { child := 8, logicalColumn := 184, sourceArmColumn := 40108, finalColumn := 671320 }
, { child := 8, logicalColumn := 185, sourceArmColumn := 40113, finalColumn := 671325 }
, { child := 8, logicalColumn := 186, sourceArmColumn := 40118, finalColumn := 671330 }
, { child := 8, logicalColumn := 187, sourceArmColumn := 40123, finalColumn := 671335 }
, { child := 8, logicalColumn := 188, sourceArmColumn := 40128, finalColumn := 671340 }
, { child := 8, logicalColumn := 189, sourceArmColumn := 40133, finalColumn := 671345 }
, { child := 8, logicalColumn := 190, sourceArmColumn := 40138, finalColumn := 671350 }
, { child := 8, logicalColumn := 191, sourceArmColumn := 40143, finalColumn := 671355 }
, { child := 8, logicalColumn := 192, sourceArmColumn := 40148, finalColumn := 671360 }
, { child := 8, logicalColumn := 193, sourceArmColumn := 40153, finalColumn := 671365 }
, { child := 8, logicalColumn := 194, sourceArmColumn := 40158, finalColumn := 671370 }
, { child := 8, logicalColumn := 195, sourceArmColumn := 40163, finalColumn := 671375 }
, { child := 8, logicalColumn := 196, sourceArmColumn := 40168, finalColumn := 671380 }
, { child := 8, logicalColumn := 197, sourceArmColumn := 40173, finalColumn := 671385 }
, { child := 8, logicalColumn := 198, sourceArmColumn := 40178, finalColumn := 671390 }
, { child := 8, logicalColumn := 199, sourceArmColumn := 40183, finalColumn := 671395 }
, { child := 8, logicalColumn := 200, sourceArmColumn := 40188, finalColumn := 671400 }
, { child := 8, logicalColumn := 201, sourceArmColumn := 40193, finalColumn := 671405 }
, { child := 8, logicalColumn := 202, sourceArmColumn := 40198, finalColumn := 671410 }
, { child := 8, logicalColumn := 203, sourceArmColumn := 40203, finalColumn := 671415 }
, { child := 8, logicalColumn := 204, sourceArmColumn := 40208, finalColumn := 671420 }
, { child := 8, logicalColumn := 205, sourceArmColumn := 40213, finalColumn := 671425 }
, { child := 8, logicalColumn := 206, sourceArmColumn := 40218, finalColumn := 671430 }
, { child := 8, logicalColumn := 207, sourceArmColumn := 40223, finalColumn := 671435 }
, { child := 8, logicalColumn := 208, sourceArmColumn := 40228, finalColumn := 671440 }
, { child := 8, logicalColumn := 209, sourceArmColumn := 40233, finalColumn := 671445 }
, { child := 8, logicalColumn := 210, sourceArmColumn := 40238, finalColumn := 671450 }
, { child := 8, logicalColumn := 211, sourceArmColumn := 40243, finalColumn := 671455 }
, { child := 8, logicalColumn := 212, sourceArmColumn := 40248, finalColumn := 671460 }
, { child := 8, logicalColumn := 213, sourceArmColumn := 40253, finalColumn := 671465 }
, { child := 8, logicalColumn := 214, sourceArmColumn := 40258, finalColumn := 671470 }
, { child := 8, logicalColumn := 215, sourceArmColumn := 40263, finalColumn := 671475 }
, { child := 8, logicalColumn := 216, sourceArmColumn := 39999, finalColumn := 671211 }
, { child := 8, logicalColumn := 217, sourceArmColumn := 40004, finalColumn := 671216 }
, { child := 8, logicalColumn := 218, sourceArmColumn := 40009, finalColumn := 671221 }
, { child := 8, logicalColumn := 219, sourceArmColumn := 40014, finalColumn := 671226 }
, { child := 8, logicalColumn := 220, sourceArmColumn := 40019, finalColumn := 671231 }
, { child := 8, logicalColumn := 221, sourceArmColumn := 40024, finalColumn := 671236 }
, { child := 8, logicalColumn := 222, sourceArmColumn := 40029, finalColumn := 671241 }
, { child := 8, logicalColumn := 223, sourceArmColumn := 40034, finalColumn := 671246 }
, { child := 8, logicalColumn := 224, sourceArmColumn := 40039, finalColumn := 671251 }
, { child := 8, logicalColumn := 225, sourceArmColumn := 40044, finalColumn := 671256 }
, { child := 8, logicalColumn := 226, sourceArmColumn := 40049, finalColumn := 671261 }
, { child := 8, logicalColumn := 227, sourceArmColumn := 40054, finalColumn := 671266 }
, { child := 8, logicalColumn := 228, sourceArmColumn := 40059, finalColumn := 671271 }
, { child := 8, logicalColumn := 229, sourceArmColumn := 40064, finalColumn := 671276 }
, { child := 8, logicalColumn := 230, sourceArmColumn := 40069, finalColumn := 671281 }
, { child := 8, logicalColumn := 231, sourceArmColumn := 40074, finalColumn := 671286 }
, { child := 8, logicalColumn := 232, sourceArmColumn := 40079, finalColumn := 671291 }
, { child := 8, logicalColumn := 233, sourceArmColumn := 40084, finalColumn := 671296 }
, { child := 8, logicalColumn := 234, sourceArmColumn := 40089, finalColumn := 671301 }
, { child := 8, logicalColumn := 235, sourceArmColumn := 40094, finalColumn := 671306 }
, { child := 8, logicalColumn := 236, sourceArmColumn := 40099, finalColumn := 671311 }
, { child := 8, logicalColumn := 237, sourceArmColumn := 40104, finalColumn := 671316 }
, { child := 8, logicalColumn := 238, sourceArmColumn := 40109, finalColumn := 671321 }
, { child := 8, logicalColumn := 239, sourceArmColumn := 40114, finalColumn := 671326 }
, { child := 8, logicalColumn := 240, sourceArmColumn := 40119, finalColumn := 671331 }
, { child := 8, logicalColumn := 241, sourceArmColumn := 40124, finalColumn := 671336 }
, { child := 8, logicalColumn := 242, sourceArmColumn := 40129, finalColumn := 671341 }
, { child := 8, logicalColumn := 243, sourceArmColumn := 40134, finalColumn := 671346 }
, { child := 8, logicalColumn := 244, sourceArmColumn := 40139, finalColumn := 671351 }
, { child := 8, logicalColumn := 245, sourceArmColumn := 40144, finalColumn := 671356 }
, { child := 8, logicalColumn := 246, sourceArmColumn := 40149, finalColumn := 671361 }
, { child := 8, logicalColumn := 247, sourceArmColumn := 40154, finalColumn := 671366 }
, { child := 8, logicalColumn := 248, sourceArmColumn := 40159, finalColumn := 671371 }
, { child := 8, logicalColumn := 249, sourceArmColumn := 40164, finalColumn := 671376 }
, { child := 8, logicalColumn := 250, sourceArmColumn := 40169, finalColumn := 671381 }
, { child := 8, logicalColumn := 251, sourceArmColumn := 40174, finalColumn := 671386 }
, { child := 8, logicalColumn := 252, sourceArmColumn := 40179, finalColumn := 671391 }
, { child := 8, logicalColumn := 253, sourceArmColumn := 40184, finalColumn := 671396 }
, { child := 8, logicalColumn := 254, sourceArmColumn := 40189, finalColumn := 671401 }
, { child := 8, logicalColumn := 255, sourceArmColumn := 40194, finalColumn := 671406 }
, { child := 8, logicalColumn := 256, sourceArmColumn := 40199, finalColumn := 671411 }
, { child := 8, logicalColumn := 257, sourceArmColumn := 40204, finalColumn := 671416 }
, { child := 8, logicalColumn := 258, sourceArmColumn := 40209, finalColumn := 671421 }
, { child := 8, logicalColumn := 259, sourceArmColumn := 40214, finalColumn := 671426 }
, { child := 8, logicalColumn := 260, sourceArmColumn := 40219, finalColumn := 671431 }
, { child := 8, logicalColumn := 261, sourceArmColumn := 40224, finalColumn := 671436 }
, { child := 8, logicalColumn := 262, sourceArmColumn := 40229, finalColumn := 671441 }
, { child := 8, logicalColumn := 263, sourceArmColumn := 40234, finalColumn := 671446 }
, { child := 8, logicalColumn := 264, sourceArmColumn := 40239, finalColumn := 671451 }
, { child := 8, logicalColumn := 265, sourceArmColumn := 40244, finalColumn := 671456 }
, { child := 8, logicalColumn := 266, sourceArmColumn := 40249, finalColumn := 671461 }
, { child := 8, logicalColumn := 267, sourceArmColumn := 40254, finalColumn := 671466 }
, { child := 8, logicalColumn := 268, sourceArmColumn := 40259, finalColumn := 671471 }
, { child := 8, logicalColumn := 269, sourceArmColumn := 40264, finalColumn := 671476 }
, { child := 9, logicalColumn := 0, sourceArmColumn := 42267, finalColumn := 737897 }
, { child := 9, logicalColumn := 1, sourceArmColumn := 42272, finalColumn := 737902 }
, { child := 9, logicalColumn := 2, sourceArmColumn := 42277, finalColumn := 737907 }
, { child := 9, logicalColumn := 3, sourceArmColumn := 42282, finalColumn := 737912 }
, { child := 9, logicalColumn := 4, sourceArmColumn := 42287, finalColumn := 737917 }
, { child := 9, logicalColumn := 5, sourceArmColumn := 42292, finalColumn := 737922 }
, { child := 9, logicalColumn := 6, sourceArmColumn := 42297, finalColumn := 737927 }
, { child := 9, logicalColumn := 7, sourceArmColumn := 42302, finalColumn := 737932 }
, { child := 9, logicalColumn := 8, sourceArmColumn := 42307, finalColumn := 737937 }
, { child := 9, logicalColumn := 9, sourceArmColumn := 42312, finalColumn := 737942 }
, { child := 9, logicalColumn := 10, sourceArmColumn := 42317, finalColumn := 737947 }
, { child := 9, logicalColumn := 11, sourceArmColumn := 42322, finalColumn := 737952 }
, { child := 9, logicalColumn := 12, sourceArmColumn := 42327, finalColumn := 737957 }
, { child := 9, logicalColumn := 13, sourceArmColumn := 42332, finalColumn := 737962 }
, { child := 9, logicalColumn := 14, sourceArmColumn := 42337, finalColumn := 737967 }
, { child := 9, logicalColumn := 15, sourceArmColumn := 42342, finalColumn := 737972 }
, { child := 9, logicalColumn := 16, sourceArmColumn := 42347, finalColumn := 737977 }
, { child := 9, logicalColumn := 17, sourceArmColumn := 42352, finalColumn := 737982 }
, { child := 9, logicalColumn := 18, sourceArmColumn := 42357, finalColumn := 737987 }
, { child := 9, logicalColumn := 19, sourceArmColumn := 42362, finalColumn := 737992 }
, { child := 9, logicalColumn := 20, sourceArmColumn := 42367, finalColumn := 737997 }
, { child := 9, logicalColumn := 21, sourceArmColumn := 42372, finalColumn := 738002 }
, { child := 9, logicalColumn := 22, sourceArmColumn := 42377, finalColumn := 738007 }
, { child := 9, logicalColumn := 23, sourceArmColumn := 42382, finalColumn := 738012 }
, { child := 9, logicalColumn := 24, sourceArmColumn := 42387, finalColumn := 738017 }
, { child := 9, logicalColumn := 25, sourceArmColumn := 42392, finalColumn := 738022 }
, { child := 9, logicalColumn := 26, sourceArmColumn := 42397, finalColumn := 738027 }
, { child := 9, logicalColumn := 27, sourceArmColumn := 42402, finalColumn := 738032 }
, { child := 9, logicalColumn := 28, sourceArmColumn := 42407, finalColumn := 738037 }
, { child := 9, logicalColumn := 29, sourceArmColumn := 42412, finalColumn := 738042 }
, { child := 9, logicalColumn := 30, sourceArmColumn := 42417, finalColumn := 738047 }
, { child := 9, logicalColumn := 31, sourceArmColumn := 42422, finalColumn := 738052 }
, { child := 9, logicalColumn := 32, sourceArmColumn := 42427, finalColumn := 738057 }
, { child := 9, logicalColumn := 33, sourceArmColumn := 42432, finalColumn := 738062 }
, { child := 9, logicalColumn := 34, sourceArmColumn := 42437, finalColumn := 738067 }
, { child := 9, logicalColumn := 35, sourceArmColumn := 42442, finalColumn := 738072 }
, { child := 9, logicalColumn := 36, sourceArmColumn := 42447, finalColumn := 738077 }
, { child := 9, logicalColumn := 37, sourceArmColumn := 42452, finalColumn := 738082 }
, { child := 9, logicalColumn := 38, sourceArmColumn := 42457, finalColumn := 738087 }
, { child := 9, logicalColumn := 39, sourceArmColumn := 42462, finalColumn := 738092 }
, { child := 9, logicalColumn := 40, sourceArmColumn := 42467, finalColumn := 738097 }
, { child := 9, logicalColumn := 41, sourceArmColumn := 42472, finalColumn := 738102 }
, { child := 9, logicalColumn := 42, sourceArmColumn := 42477, finalColumn := 738107 }
, { child := 9, logicalColumn := 43, sourceArmColumn := 42482, finalColumn := 738112 }
, { child := 9, logicalColumn := 44, sourceArmColumn := 42487, finalColumn := 738117 }
, { child := 9, logicalColumn := 45, sourceArmColumn := 42492, finalColumn := 738122 }
, { child := 9, logicalColumn := 46, sourceArmColumn := 42497, finalColumn := 738127 }
, { child := 9, logicalColumn := 47, sourceArmColumn := 42502, finalColumn := 738132 }
, { child := 9, logicalColumn := 48, sourceArmColumn := 42507, finalColumn := 738137 }
, { child := 9, logicalColumn := 49, sourceArmColumn := 42512, finalColumn := 738142 }
, { child := 9, logicalColumn := 50, sourceArmColumn := 42517, finalColumn := 738147 }
, { child := 9, logicalColumn := 51, sourceArmColumn := 42522, finalColumn := 738152 }
, { child := 9, logicalColumn := 52, sourceArmColumn := 42527, finalColumn := 738157 }
, { child := 9, logicalColumn := 53, sourceArmColumn := 42532, finalColumn := 738162 }
, { child := 9, logicalColumn := 54, sourceArmColumn := 42268, finalColumn := 737898 }
, { child := 9, logicalColumn := 55, sourceArmColumn := 42273, finalColumn := 737903 }
, { child := 9, logicalColumn := 56, sourceArmColumn := 42278, finalColumn := 737908 }
, { child := 9, logicalColumn := 57, sourceArmColumn := 42283, finalColumn := 737913 }
, { child := 9, logicalColumn := 58, sourceArmColumn := 42288, finalColumn := 737918 }
, { child := 9, logicalColumn := 59, sourceArmColumn := 42293, finalColumn := 737923 }
, { child := 9, logicalColumn := 60, sourceArmColumn := 42298, finalColumn := 737928 }
, { child := 9, logicalColumn := 61, sourceArmColumn := 42303, finalColumn := 737933 }
, { child := 9, logicalColumn := 62, sourceArmColumn := 42308, finalColumn := 737938 }
, { child := 9, logicalColumn := 63, sourceArmColumn := 42313, finalColumn := 737943 }
, { child := 9, logicalColumn := 64, sourceArmColumn := 42318, finalColumn := 737948 }
, { child := 9, logicalColumn := 65, sourceArmColumn := 42323, finalColumn := 737953 }
, { child := 9, logicalColumn := 66, sourceArmColumn := 42328, finalColumn := 737958 }
, { child := 9, logicalColumn := 67, sourceArmColumn := 42333, finalColumn := 737963 }
, { child := 9, logicalColumn := 68, sourceArmColumn := 42338, finalColumn := 737968 }
, { child := 9, logicalColumn := 69, sourceArmColumn := 42343, finalColumn := 737973 }
, { child := 9, logicalColumn := 70, sourceArmColumn := 42348, finalColumn := 737978 }
, { child := 9, logicalColumn := 71, sourceArmColumn := 42353, finalColumn := 737983 }
, { child := 9, logicalColumn := 72, sourceArmColumn := 42358, finalColumn := 737988 }
, { child := 9, logicalColumn := 73, sourceArmColumn := 42363, finalColumn := 737993 }
, { child := 9, logicalColumn := 74, sourceArmColumn := 42368, finalColumn := 737998 }
, { child := 9, logicalColumn := 75, sourceArmColumn := 42373, finalColumn := 738003 }
, { child := 9, logicalColumn := 76, sourceArmColumn := 42378, finalColumn := 738008 }
, { child := 9, logicalColumn := 77, sourceArmColumn := 42383, finalColumn := 738013 }
, { child := 9, logicalColumn := 78, sourceArmColumn := 42388, finalColumn := 738018 }
, { child := 9, logicalColumn := 79, sourceArmColumn := 42393, finalColumn := 738023 }
, { child := 9, logicalColumn := 80, sourceArmColumn := 42398, finalColumn := 738028 }
, { child := 9, logicalColumn := 81, sourceArmColumn := 42403, finalColumn := 738033 }
, { child := 9, logicalColumn := 82, sourceArmColumn := 42408, finalColumn := 738038 }
, { child := 9, logicalColumn := 83, sourceArmColumn := 42413, finalColumn := 738043 }
, { child := 9, logicalColumn := 84, sourceArmColumn := 42418, finalColumn := 738048 }
, { child := 9, logicalColumn := 85, sourceArmColumn := 42423, finalColumn := 738053 }
, { child := 9, logicalColumn := 86, sourceArmColumn := 42428, finalColumn := 738058 }
, { child := 9, logicalColumn := 87, sourceArmColumn := 42433, finalColumn := 738063 }
, { child := 9, logicalColumn := 88, sourceArmColumn := 42438, finalColumn := 738068 }
, { child := 9, logicalColumn := 89, sourceArmColumn := 42443, finalColumn := 738073 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk9
