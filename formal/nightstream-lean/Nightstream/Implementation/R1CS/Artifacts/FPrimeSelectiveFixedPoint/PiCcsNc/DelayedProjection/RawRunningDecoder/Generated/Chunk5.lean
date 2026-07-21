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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk5

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 4, logicalColumn := 180, sourceArmColumn := 31000, finalColumn := 404540 }
, { child := 4, logicalColumn := 181, sourceArmColumn := 31005, finalColumn := 404545 }
, { child := 4, logicalColumn := 182, sourceArmColumn := 31010, finalColumn := 404550 }
, { child := 4, logicalColumn := 183, sourceArmColumn := 31015, finalColumn := 404555 }
, { child := 4, logicalColumn := 184, sourceArmColumn := 31020, finalColumn := 404560 }
, { child := 4, logicalColumn := 185, sourceArmColumn := 31025, finalColumn := 404565 }
, { child := 4, logicalColumn := 186, sourceArmColumn := 31030, finalColumn := 404570 }
, { child := 4, logicalColumn := 187, sourceArmColumn := 31035, finalColumn := 404575 }
, { child := 4, logicalColumn := 188, sourceArmColumn := 31040, finalColumn := 404580 }
, { child := 4, logicalColumn := 189, sourceArmColumn := 31045, finalColumn := 404585 }
, { child := 4, logicalColumn := 190, sourceArmColumn := 31050, finalColumn := 404590 }
, { child := 4, logicalColumn := 191, sourceArmColumn := 31055, finalColumn := 404595 }
, { child := 4, logicalColumn := 192, sourceArmColumn := 31060, finalColumn := 404600 }
, { child := 4, logicalColumn := 193, sourceArmColumn := 31065, finalColumn := 404605 }
, { child := 4, logicalColumn := 194, sourceArmColumn := 31070, finalColumn := 404610 }
, { child := 4, logicalColumn := 195, sourceArmColumn := 31075, finalColumn := 404615 }
, { child := 4, logicalColumn := 196, sourceArmColumn := 31080, finalColumn := 404620 }
, { child := 4, logicalColumn := 197, sourceArmColumn := 31085, finalColumn := 404625 }
, { child := 4, logicalColumn := 198, sourceArmColumn := 31090, finalColumn := 404630 }
, { child := 4, logicalColumn := 199, sourceArmColumn := 31095, finalColumn := 404635 }
, { child := 4, logicalColumn := 200, sourceArmColumn := 31100, finalColumn := 404640 }
, { child := 4, logicalColumn := 201, sourceArmColumn := 31105, finalColumn := 404645 }
, { child := 4, logicalColumn := 202, sourceArmColumn := 31110, finalColumn := 404650 }
, { child := 4, logicalColumn := 203, sourceArmColumn := 31115, finalColumn := 404655 }
, { child := 4, logicalColumn := 204, sourceArmColumn := 31120, finalColumn := 404660 }
, { child := 4, logicalColumn := 205, sourceArmColumn := 31125, finalColumn := 404665 }
, { child := 4, logicalColumn := 206, sourceArmColumn := 31130, finalColumn := 404670 }
, { child := 4, logicalColumn := 207, sourceArmColumn := 31135, finalColumn := 404675 }
, { child := 4, logicalColumn := 208, sourceArmColumn := 31140, finalColumn := 404680 }
, { child := 4, logicalColumn := 209, sourceArmColumn := 31145, finalColumn := 404685 }
, { child := 4, logicalColumn := 210, sourceArmColumn := 31150, finalColumn := 404690 }
, { child := 4, logicalColumn := 211, sourceArmColumn := 31155, finalColumn := 404695 }
, { child := 4, logicalColumn := 212, sourceArmColumn := 31160, finalColumn := 404700 }
, { child := 4, logicalColumn := 213, sourceArmColumn := 31165, finalColumn := 404705 }
, { child := 4, logicalColumn := 214, sourceArmColumn := 31170, finalColumn := 404710 }
, { child := 4, logicalColumn := 215, sourceArmColumn := 31175, finalColumn := 404715 }
, { child := 4, logicalColumn := 216, sourceArmColumn := 30911, finalColumn := 404451 }
, { child := 4, logicalColumn := 217, sourceArmColumn := 30916, finalColumn := 404456 }
, { child := 4, logicalColumn := 218, sourceArmColumn := 30921, finalColumn := 404461 }
, { child := 4, logicalColumn := 219, sourceArmColumn := 30926, finalColumn := 404466 }
, { child := 4, logicalColumn := 220, sourceArmColumn := 30931, finalColumn := 404471 }
, { child := 4, logicalColumn := 221, sourceArmColumn := 30936, finalColumn := 404476 }
, { child := 4, logicalColumn := 222, sourceArmColumn := 30941, finalColumn := 404481 }
, { child := 4, logicalColumn := 223, sourceArmColumn := 30946, finalColumn := 404486 }
, { child := 4, logicalColumn := 224, sourceArmColumn := 30951, finalColumn := 404491 }
, { child := 4, logicalColumn := 225, sourceArmColumn := 30956, finalColumn := 404496 }
, { child := 4, logicalColumn := 226, sourceArmColumn := 30961, finalColumn := 404501 }
, { child := 4, logicalColumn := 227, sourceArmColumn := 30966, finalColumn := 404506 }
, { child := 4, logicalColumn := 228, sourceArmColumn := 30971, finalColumn := 404511 }
, { child := 4, logicalColumn := 229, sourceArmColumn := 30976, finalColumn := 404516 }
, { child := 4, logicalColumn := 230, sourceArmColumn := 30981, finalColumn := 404521 }
, { child := 4, logicalColumn := 231, sourceArmColumn := 30986, finalColumn := 404526 }
, { child := 4, logicalColumn := 232, sourceArmColumn := 30991, finalColumn := 404531 }
, { child := 4, logicalColumn := 233, sourceArmColumn := 30996, finalColumn := 404536 }
, { child := 4, logicalColumn := 234, sourceArmColumn := 31001, finalColumn := 404541 }
, { child := 4, logicalColumn := 235, sourceArmColumn := 31006, finalColumn := 404546 }
, { child := 4, logicalColumn := 236, sourceArmColumn := 31011, finalColumn := 404551 }
, { child := 4, logicalColumn := 237, sourceArmColumn := 31016, finalColumn := 404556 }
, { child := 4, logicalColumn := 238, sourceArmColumn := 31021, finalColumn := 404561 }
, { child := 4, logicalColumn := 239, sourceArmColumn := 31026, finalColumn := 404566 }
, { child := 4, logicalColumn := 240, sourceArmColumn := 31031, finalColumn := 404571 }
, { child := 4, logicalColumn := 241, sourceArmColumn := 31036, finalColumn := 404576 }
, { child := 4, logicalColumn := 242, sourceArmColumn := 31041, finalColumn := 404581 }
, { child := 4, logicalColumn := 243, sourceArmColumn := 31046, finalColumn := 404586 }
, { child := 4, logicalColumn := 244, sourceArmColumn := 31051, finalColumn := 404591 }
, { child := 4, logicalColumn := 245, sourceArmColumn := 31056, finalColumn := 404596 }
, { child := 4, logicalColumn := 246, sourceArmColumn := 31061, finalColumn := 404601 }
, { child := 4, logicalColumn := 247, sourceArmColumn := 31066, finalColumn := 404606 }
, { child := 4, logicalColumn := 248, sourceArmColumn := 31071, finalColumn := 404611 }
, { child := 4, logicalColumn := 249, sourceArmColumn := 31076, finalColumn := 404616 }
, { child := 4, logicalColumn := 250, sourceArmColumn := 31081, finalColumn := 404621 }
, { child := 4, logicalColumn := 251, sourceArmColumn := 31086, finalColumn := 404626 }
, { child := 4, logicalColumn := 252, sourceArmColumn := 31091, finalColumn := 404631 }
, { child := 4, logicalColumn := 253, sourceArmColumn := 31096, finalColumn := 404636 }
, { child := 4, logicalColumn := 254, sourceArmColumn := 31101, finalColumn := 404641 }
, { child := 4, logicalColumn := 255, sourceArmColumn := 31106, finalColumn := 404646 }
, { child := 4, logicalColumn := 256, sourceArmColumn := 31111, finalColumn := 404651 }
, { child := 4, logicalColumn := 257, sourceArmColumn := 31116, finalColumn := 404656 }
, { child := 4, logicalColumn := 258, sourceArmColumn := 31121, finalColumn := 404661 }
, { child := 4, logicalColumn := 259, sourceArmColumn := 31126, finalColumn := 404666 }
, { child := 4, logicalColumn := 260, sourceArmColumn := 31131, finalColumn := 404671 }
, { child := 4, logicalColumn := 261, sourceArmColumn := 31136, finalColumn := 404676 }
, { child := 4, logicalColumn := 262, sourceArmColumn := 31141, finalColumn := 404681 }
, { child := 4, logicalColumn := 263, sourceArmColumn := 31146, finalColumn := 404686 }
, { child := 4, logicalColumn := 264, sourceArmColumn := 31151, finalColumn := 404691 }
, { child := 4, logicalColumn := 265, sourceArmColumn := 31156, finalColumn := 404696 }
, { child := 4, logicalColumn := 266, sourceArmColumn := 31161, finalColumn := 404701 }
, { child := 4, logicalColumn := 267, sourceArmColumn := 31166, finalColumn := 404706 }
, { child := 4, logicalColumn := 268, sourceArmColumn := 31171, finalColumn := 404711 }
, { child := 4, logicalColumn := 269, sourceArmColumn := 31176, finalColumn := 404716 }
, { child := 5, logicalColumn := 0, sourceArmColumn := 33179, finalColumn := 471137 }
, { child := 5, logicalColumn := 1, sourceArmColumn := 33184, finalColumn := 471142 }
, { child := 5, logicalColumn := 2, sourceArmColumn := 33189, finalColumn := 471147 }
, { child := 5, logicalColumn := 3, sourceArmColumn := 33194, finalColumn := 471152 }
, { child := 5, logicalColumn := 4, sourceArmColumn := 33199, finalColumn := 471157 }
, { child := 5, logicalColumn := 5, sourceArmColumn := 33204, finalColumn := 471162 }
, { child := 5, logicalColumn := 6, sourceArmColumn := 33209, finalColumn := 471167 }
, { child := 5, logicalColumn := 7, sourceArmColumn := 33214, finalColumn := 471172 }
, { child := 5, logicalColumn := 8, sourceArmColumn := 33219, finalColumn := 471177 }
, { child := 5, logicalColumn := 9, sourceArmColumn := 33224, finalColumn := 471182 }
, { child := 5, logicalColumn := 10, sourceArmColumn := 33229, finalColumn := 471187 }
, { child := 5, logicalColumn := 11, sourceArmColumn := 33234, finalColumn := 471192 }
, { child := 5, logicalColumn := 12, sourceArmColumn := 33239, finalColumn := 471197 }
, { child := 5, logicalColumn := 13, sourceArmColumn := 33244, finalColumn := 471202 }
, { child := 5, logicalColumn := 14, sourceArmColumn := 33249, finalColumn := 471207 }
, { child := 5, logicalColumn := 15, sourceArmColumn := 33254, finalColumn := 471212 }
, { child := 5, logicalColumn := 16, sourceArmColumn := 33259, finalColumn := 471217 }
, { child := 5, logicalColumn := 17, sourceArmColumn := 33264, finalColumn := 471222 }
, { child := 5, logicalColumn := 18, sourceArmColumn := 33269, finalColumn := 471227 }
, { child := 5, logicalColumn := 19, sourceArmColumn := 33274, finalColumn := 471232 }
, { child := 5, logicalColumn := 20, sourceArmColumn := 33279, finalColumn := 471237 }
, { child := 5, logicalColumn := 21, sourceArmColumn := 33284, finalColumn := 471242 }
, { child := 5, logicalColumn := 22, sourceArmColumn := 33289, finalColumn := 471247 }
, { child := 5, logicalColumn := 23, sourceArmColumn := 33294, finalColumn := 471252 }
, { child := 5, logicalColumn := 24, sourceArmColumn := 33299, finalColumn := 471257 }
, { child := 5, logicalColumn := 25, sourceArmColumn := 33304, finalColumn := 471262 }
, { child := 5, logicalColumn := 26, sourceArmColumn := 33309, finalColumn := 471267 }
, { child := 5, logicalColumn := 27, sourceArmColumn := 33314, finalColumn := 471272 }
, { child := 5, logicalColumn := 28, sourceArmColumn := 33319, finalColumn := 471277 }
, { child := 5, logicalColumn := 29, sourceArmColumn := 33324, finalColumn := 471282 }
, { child := 5, logicalColumn := 30, sourceArmColumn := 33329, finalColumn := 471287 }
, { child := 5, logicalColumn := 31, sourceArmColumn := 33334, finalColumn := 471292 }
, { child := 5, logicalColumn := 32, sourceArmColumn := 33339, finalColumn := 471297 }
, { child := 5, logicalColumn := 33, sourceArmColumn := 33344, finalColumn := 471302 }
, { child := 5, logicalColumn := 34, sourceArmColumn := 33349, finalColumn := 471307 }
, { child := 5, logicalColumn := 35, sourceArmColumn := 33354, finalColumn := 471312 }
, { child := 5, logicalColumn := 36, sourceArmColumn := 33359, finalColumn := 471317 }
, { child := 5, logicalColumn := 37, sourceArmColumn := 33364, finalColumn := 471322 }
, { child := 5, logicalColumn := 38, sourceArmColumn := 33369, finalColumn := 471327 }
, { child := 5, logicalColumn := 39, sourceArmColumn := 33374, finalColumn := 471332 }
, { child := 5, logicalColumn := 40, sourceArmColumn := 33379, finalColumn := 471337 }
, { child := 5, logicalColumn := 41, sourceArmColumn := 33384, finalColumn := 471342 }
, { child := 5, logicalColumn := 42, sourceArmColumn := 33389, finalColumn := 471347 }
, { child := 5, logicalColumn := 43, sourceArmColumn := 33394, finalColumn := 471352 }
, { child := 5, logicalColumn := 44, sourceArmColumn := 33399, finalColumn := 471357 }
, { child := 5, logicalColumn := 45, sourceArmColumn := 33404, finalColumn := 471362 }
, { child := 5, logicalColumn := 46, sourceArmColumn := 33409, finalColumn := 471367 }
, { child := 5, logicalColumn := 47, sourceArmColumn := 33414, finalColumn := 471372 }
, { child := 5, logicalColumn := 48, sourceArmColumn := 33419, finalColumn := 471377 }
, { child := 5, logicalColumn := 49, sourceArmColumn := 33424, finalColumn := 471382 }
, { child := 5, logicalColumn := 50, sourceArmColumn := 33429, finalColumn := 471387 }
, { child := 5, logicalColumn := 51, sourceArmColumn := 33434, finalColumn := 471392 }
, { child := 5, logicalColumn := 52, sourceArmColumn := 33439, finalColumn := 471397 }
, { child := 5, logicalColumn := 53, sourceArmColumn := 33444, finalColumn := 471402 }
, { child := 5, logicalColumn := 54, sourceArmColumn := 33180, finalColumn := 471138 }
, { child := 5, logicalColumn := 55, sourceArmColumn := 33185, finalColumn := 471143 }
, { child := 5, logicalColumn := 56, sourceArmColumn := 33190, finalColumn := 471148 }
, { child := 5, logicalColumn := 57, sourceArmColumn := 33195, finalColumn := 471153 }
, { child := 5, logicalColumn := 58, sourceArmColumn := 33200, finalColumn := 471158 }
, { child := 5, logicalColumn := 59, sourceArmColumn := 33205, finalColumn := 471163 }
, { child := 5, logicalColumn := 60, sourceArmColumn := 33210, finalColumn := 471168 }
, { child := 5, logicalColumn := 61, sourceArmColumn := 33215, finalColumn := 471173 }
, { child := 5, logicalColumn := 62, sourceArmColumn := 33220, finalColumn := 471178 }
, { child := 5, logicalColumn := 63, sourceArmColumn := 33225, finalColumn := 471183 }
, { child := 5, logicalColumn := 64, sourceArmColumn := 33230, finalColumn := 471188 }
, { child := 5, logicalColumn := 65, sourceArmColumn := 33235, finalColumn := 471193 }
, { child := 5, logicalColumn := 66, sourceArmColumn := 33240, finalColumn := 471198 }
, { child := 5, logicalColumn := 67, sourceArmColumn := 33245, finalColumn := 471203 }
, { child := 5, logicalColumn := 68, sourceArmColumn := 33250, finalColumn := 471208 }
, { child := 5, logicalColumn := 69, sourceArmColumn := 33255, finalColumn := 471213 }
, { child := 5, logicalColumn := 70, sourceArmColumn := 33260, finalColumn := 471218 }
, { child := 5, logicalColumn := 71, sourceArmColumn := 33265, finalColumn := 471223 }
, { child := 5, logicalColumn := 72, sourceArmColumn := 33270, finalColumn := 471228 }
, { child := 5, logicalColumn := 73, sourceArmColumn := 33275, finalColumn := 471233 }
, { child := 5, logicalColumn := 74, sourceArmColumn := 33280, finalColumn := 471238 }
, { child := 5, logicalColumn := 75, sourceArmColumn := 33285, finalColumn := 471243 }
, { child := 5, logicalColumn := 76, sourceArmColumn := 33290, finalColumn := 471248 }
, { child := 5, logicalColumn := 77, sourceArmColumn := 33295, finalColumn := 471253 }
, { child := 5, logicalColumn := 78, sourceArmColumn := 33300, finalColumn := 471258 }
, { child := 5, logicalColumn := 79, sourceArmColumn := 33305, finalColumn := 471263 }
, { child := 5, logicalColumn := 80, sourceArmColumn := 33310, finalColumn := 471268 }
, { child := 5, logicalColumn := 81, sourceArmColumn := 33315, finalColumn := 471273 }
, { child := 5, logicalColumn := 82, sourceArmColumn := 33320, finalColumn := 471278 }
, { child := 5, logicalColumn := 83, sourceArmColumn := 33325, finalColumn := 471283 }
, { child := 5, logicalColumn := 84, sourceArmColumn := 33330, finalColumn := 471288 }
, { child := 5, logicalColumn := 85, sourceArmColumn := 33335, finalColumn := 471293 }
, { child := 5, logicalColumn := 86, sourceArmColumn := 33340, finalColumn := 471298 }
, { child := 5, logicalColumn := 87, sourceArmColumn := 33345, finalColumn := 471303 }
, { child := 5, logicalColumn := 88, sourceArmColumn := 33350, finalColumn := 471308 }
, { child := 5, logicalColumn := 89, sourceArmColumn := 33355, finalColumn := 471313 }
, { child := 5, logicalColumn := 90, sourceArmColumn := 33360, finalColumn := 471318 }
, { child := 5, logicalColumn := 91, sourceArmColumn := 33365, finalColumn := 471323 }
, { child := 5, logicalColumn := 92, sourceArmColumn := 33370, finalColumn := 471328 }
, { child := 5, logicalColumn := 93, sourceArmColumn := 33375, finalColumn := 471333 }
, { child := 5, logicalColumn := 94, sourceArmColumn := 33380, finalColumn := 471338 }
, { child := 5, logicalColumn := 95, sourceArmColumn := 33385, finalColumn := 471343 }
, { child := 5, logicalColumn := 96, sourceArmColumn := 33390, finalColumn := 471348 }
, { child := 5, logicalColumn := 97, sourceArmColumn := 33395, finalColumn := 471353 }
, { child := 5, logicalColumn := 98, sourceArmColumn := 33400, finalColumn := 471358 }
, { child := 5, logicalColumn := 99, sourceArmColumn := 33405, finalColumn := 471363 }
, { child := 5, logicalColumn := 100, sourceArmColumn := 33410, finalColumn := 471368 }
, { child := 5, logicalColumn := 101, sourceArmColumn := 33415, finalColumn := 471373 }
, { child := 5, logicalColumn := 102, sourceArmColumn := 33420, finalColumn := 471378 }
, { child := 5, logicalColumn := 103, sourceArmColumn := 33425, finalColumn := 471383 }
, { child := 5, logicalColumn := 104, sourceArmColumn := 33430, finalColumn := 471388 }
, { child := 5, logicalColumn := 105, sourceArmColumn := 33435, finalColumn := 471393 }
, { child := 5, logicalColumn := 106, sourceArmColumn := 33440, finalColumn := 471398 }
, { child := 5, logicalColumn := 107, sourceArmColumn := 33445, finalColumn := 471403 }
, { child := 5, logicalColumn := 108, sourceArmColumn := 33181, finalColumn := 471139 }
, { child := 5, logicalColumn := 109, sourceArmColumn := 33186, finalColumn := 471144 }
, { child := 5, logicalColumn := 110, sourceArmColumn := 33191, finalColumn := 471149 }
, { child := 5, logicalColumn := 111, sourceArmColumn := 33196, finalColumn := 471154 }
, { child := 5, logicalColumn := 112, sourceArmColumn := 33201, finalColumn := 471159 }
, { child := 5, logicalColumn := 113, sourceArmColumn := 33206, finalColumn := 471164 }
, { child := 5, logicalColumn := 114, sourceArmColumn := 33211, finalColumn := 471169 }
, { child := 5, logicalColumn := 115, sourceArmColumn := 33216, finalColumn := 471174 }
, { child := 5, logicalColumn := 116, sourceArmColumn := 33221, finalColumn := 471179 }
, { child := 5, logicalColumn := 117, sourceArmColumn := 33226, finalColumn := 471184 }
, { child := 5, logicalColumn := 118, sourceArmColumn := 33231, finalColumn := 471189 }
, { child := 5, logicalColumn := 119, sourceArmColumn := 33236, finalColumn := 471194 }
, { child := 5, logicalColumn := 120, sourceArmColumn := 33241, finalColumn := 471199 }
, { child := 5, logicalColumn := 121, sourceArmColumn := 33246, finalColumn := 471204 }
, { child := 5, logicalColumn := 122, sourceArmColumn := 33251, finalColumn := 471209 }
, { child := 5, logicalColumn := 123, sourceArmColumn := 33256, finalColumn := 471214 }
, { child := 5, logicalColumn := 124, sourceArmColumn := 33261, finalColumn := 471219 }
, { child := 5, logicalColumn := 125, sourceArmColumn := 33266, finalColumn := 471224 }
, { child := 5, logicalColumn := 126, sourceArmColumn := 33271, finalColumn := 471229 }
, { child := 5, logicalColumn := 127, sourceArmColumn := 33276, finalColumn := 471234 }
, { child := 5, logicalColumn := 128, sourceArmColumn := 33281, finalColumn := 471239 }
, { child := 5, logicalColumn := 129, sourceArmColumn := 33286, finalColumn := 471244 }
, { child := 5, logicalColumn := 130, sourceArmColumn := 33291, finalColumn := 471249 }
, { child := 5, logicalColumn := 131, sourceArmColumn := 33296, finalColumn := 471254 }
, { child := 5, logicalColumn := 132, sourceArmColumn := 33301, finalColumn := 471259 }
, { child := 5, logicalColumn := 133, sourceArmColumn := 33306, finalColumn := 471264 }
, { child := 5, logicalColumn := 134, sourceArmColumn := 33311, finalColumn := 471269 }
, { child := 5, logicalColumn := 135, sourceArmColumn := 33316, finalColumn := 471274 }
, { child := 5, logicalColumn := 136, sourceArmColumn := 33321, finalColumn := 471279 }
, { child := 5, logicalColumn := 137, sourceArmColumn := 33326, finalColumn := 471284 }
, { child := 5, logicalColumn := 138, sourceArmColumn := 33331, finalColumn := 471289 }
, { child := 5, logicalColumn := 139, sourceArmColumn := 33336, finalColumn := 471294 }
, { child := 5, logicalColumn := 140, sourceArmColumn := 33341, finalColumn := 471299 }
, { child := 5, logicalColumn := 141, sourceArmColumn := 33346, finalColumn := 471304 }
, { child := 5, logicalColumn := 142, sourceArmColumn := 33351, finalColumn := 471309 }
, { child := 5, logicalColumn := 143, sourceArmColumn := 33356, finalColumn := 471314 }
, { child := 5, logicalColumn := 144, sourceArmColumn := 33361, finalColumn := 471319 }
, { child := 5, logicalColumn := 145, sourceArmColumn := 33366, finalColumn := 471324 }
, { child := 5, logicalColumn := 146, sourceArmColumn := 33371, finalColumn := 471329 }
, { child := 5, logicalColumn := 147, sourceArmColumn := 33376, finalColumn := 471334 }
, { child := 5, logicalColumn := 148, sourceArmColumn := 33381, finalColumn := 471339 }
, { child := 5, logicalColumn := 149, sourceArmColumn := 33386, finalColumn := 471344 }
, { child := 5, logicalColumn := 150, sourceArmColumn := 33391, finalColumn := 471349 }
, { child := 5, logicalColumn := 151, sourceArmColumn := 33396, finalColumn := 471354 }
, { child := 5, logicalColumn := 152, sourceArmColumn := 33401, finalColumn := 471359 }
, { child := 5, logicalColumn := 153, sourceArmColumn := 33406, finalColumn := 471364 }
, { child := 5, logicalColumn := 154, sourceArmColumn := 33411, finalColumn := 471369 }
, { child := 5, logicalColumn := 155, sourceArmColumn := 33416, finalColumn := 471374 }
, { child := 5, logicalColumn := 156, sourceArmColumn := 33421, finalColumn := 471379 }
, { child := 5, logicalColumn := 157, sourceArmColumn := 33426, finalColumn := 471384 }
, { child := 5, logicalColumn := 158, sourceArmColumn := 33431, finalColumn := 471389 }
, { child := 5, logicalColumn := 159, sourceArmColumn := 33436, finalColumn := 471394 }
, { child := 5, logicalColumn := 160, sourceArmColumn := 33441, finalColumn := 471399 }
, { child := 5, logicalColumn := 161, sourceArmColumn := 33446, finalColumn := 471404 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk5
