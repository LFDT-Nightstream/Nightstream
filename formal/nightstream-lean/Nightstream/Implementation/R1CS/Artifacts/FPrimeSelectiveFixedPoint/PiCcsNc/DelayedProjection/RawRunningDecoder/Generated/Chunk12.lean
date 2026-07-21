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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk12

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 11, logicalColumn := 54, sourceArmColumn := 46812, finalColumn := 871278 }
, { child := 11, logicalColumn := 55, sourceArmColumn := 46817, finalColumn := 871283 }
, { child := 11, logicalColumn := 56, sourceArmColumn := 46822, finalColumn := 871288 }
, { child := 11, logicalColumn := 57, sourceArmColumn := 46827, finalColumn := 871293 }
, { child := 11, logicalColumn := 58, sourceArmColumn := 46832, finalColumn := 871298 }
, { child := 11, logicalColumn := 59, sourceArmColumn := 46837, finalColumn := 871303 }
, { child := 11, logicalColumn := 60, sourceArmColumn := 46842, finalColumn := 871308 }
, { child := 11, logicalColumn := 61, sourceArmColumn := 46847, finalColumn := 871313 }
, { child := 11, logicalColumn := 62, sourceArmColumn := 46852, finalColumn := 871318 }
, { child := 11, logicalColumn := 63, sourceArmColumn := 46857, finalColumn := 871323 }
, { child := 11, logicalColumn := 64, sourceArmColumn := 46862, finalColumn := 871328 }
, { child := 11, logicalColumn := 65, sourceArmColumn := 46867, finalColumn := 871333 }
, { child := 11, logicalColumn := 66, sourceArmColumn := 46872, finalColumn := 871338 }
, { child := 11, logicalColumn := 67, sourceArmColumn := 46877, finalColumn := 871343 }
, { child := 11, logicalColumn := 68, sourceArmColumn := 46882, finalColumn := 871348 }
, { child := 11, logicalColumn := 69, sourceArmColumn := 46887, finalColumn := 871353 }
, { child := 11, logicalColumn := 70, sourceArmColumn := 46892, finalColumn := 871358 }
, { child := 11, logicalColumn := 71, sourceArmColumn := 46897, finalColumn := 871363 }
, { child := 11, logicalColumn := 72, sourceArmColumn := 46902, finalColumn := 871368 }
, { child := 11, logicalColumn := 73, sourceArmColumn := 46907, finalColumn := 871373 }
, { child := 11, logicalColumn := 74, sourceArmColumn := 46912, finalColumn := 871378 }
, { child := 11, logicalColumn := 75, sourceArmColumn := 46917, finalColumn := 871383 }
, { child := 11, logicalColumn := 76, sourceArmColumn := 46922, finalColumn := 871388 }
, { child := 11, logicalColumn := 77, sourceArmColumn := 46927, finalColumn := 871393 }
, { child := 11, logicalColumn := 78, sourceArmColumn := 46932, finalColumn := 871398 }
, { child := 11, logicalColumn := 79, sourceArmColumn := 46937, finalColumn := 871403 }
, { child := 11, logicalColumn := 80, sourceArmColumn := 46942, finalColumn := 871408 }
, { child := 11, logicalColumn := 81, sourceArmColumn := 46947, finalColumn := 871413 }
, { child := 11, logicalColumn := 82, sourceArmColumn := 46952, finalColumn := 871418 }
, { child := 11, logicalColumn := 83, sourceArmColumn := 46957, finalColumn := 871423 }
, { child := 11, logicalColumn := 84, sourceArmColumn := 46962, finalColumn := 871428 }
, { child := 11, logicalColumn := 85, sourceArmColumn := 46967, finalColumn := 871433 }
, { child := 11, logicalColumn := 86, sourceArmColumn := 46972, finalColumn := 871438 }
, { child := 11, logicalColumn := 87, sourceArmColumn := 46977, finalColumn := 871443 }
, { child := 11, logicalColumn := 88, sourceArmColumn := 46982, finalColumn := 871448 }
, { child := 11, logicalColumn := 89, sourceArmColumn := 46987, finalColumn := 871453 }
, { child := 11, logicalColumn := 90, sourceArmColumn := 46992, finalColumn := 871458 }
, { child := 11, logicalColumn := 91, sourceArmColumn := 46997, finalColumn := 871463 }
, { child := 11, logicalColumn := 92, sourceArmColumn := 47002, finalColumn := 871468 }
, { child := 11, logicalColumn := 93, sourceArmColumn := 47007, finalColumn := 871473 }
, { child := 11, logicalColumn := 94, sourceArmColumn := 47012, finalColumn := 871478 }
, { child := 11, logicalColumn := 95, sourceArmColumn := 47017, finalColumn := 871483 }
, { child := 11, logicalColumn := 96, sourceArmColumn := 47022, finalColumn := 871488 }
, { child := 11, logicalColumn := 97, sourceArmColumn := 47027, finalColumn := 871493 }
, { child := 11, logicalColumn := 98, sourceArmColumn := 47032, finalColumn := 871498 }
, { child := 11, logicalColumn := 99, sourceArmColumn := 47037, finalColumn := 871503 }
, { child := 11, logicalColumn := 100, sourceArmColumn := 47042, finalColumn := 871508 }
, { child := 11, logicalColumn := 101, sourceArmColumn := 47047, finalColumn := 871513 }
, { child := 11, logicalColumn := 102, sourceArmColumn := 47052, finalColumn := 871518 }
, { child := 11, logicalColumn := 103, sourceArmColumn := 47057, finalColumn := 871523 }
, { child := 11, logicalColumn := 104, sourceArmColumn := 47062, finalColumn := 871528 }
, { child := 11, logicalColumn := 105, sourceArmColumn := 47067, finalColumn := 871533 }
, { child := 11, logicalColumn := 106, sourceArmColumn := 47072, finalColumn := 871538 }
, { child := 11, logicalColumn := 107, sourceArmColumn := 47077, finalColumn := 871543 }
, { child := 11, logicalColumn := 108, sourceArmColumn := 46813, finalColumn := 871279 }
, { child := 11, logicalColumn := 109, sourceArmColumn := 46818, finalColumn := 871284 }
, { child := 11, logicalColumn := 110, sourceArmColumn := 46823, finalColumn := 871289 }
, { child := 11, logicalColumn := 111, sourceArmColumn := 46828, finalColumn := 871294 }
, { child := 11, logicalColumn := 112, sourceArmColumn := 46833, finalColumn := 871299 }
, { child := 11, logicalColumn := 113, sourceArmColumn := 46838, finalColumn := 871304 }
, { child := 11, logicalColumn := 114, sourceArmColumn := 46843, finalColumn := 871309 }
, { child := 11, logicalColumn := 115, sourceArmColumn := 46848, finalColumn := 871314 }
, { child := 11, logicalColumn := 116, sourceArmColumn := 46853, finalColumn := 871319 }
, { child := 11, logicalColumn := 117, sourceArmColumn := 46858, finalColumn := 871324 }
, { child := 11, logicalColumn := 118, sourceArmColumn := 46863, finalColumn := 871329 }
, { child := 11, logicalColumn := 119, sourceArmColumn := 46868, finalColumn := 871334 }
, { child := 11, logicalColumn := 120, sourceArmColumn := 46873, finalColumn := 871339 }
, { child := 11, logicalColumn := 121, sourceArmColumn := 46878, finalColumn := 871344 }
, { child := 11, logicalColumn := 122, sourceArmColumn := 46883, finalColumn := 871349 }
, { child := 11, logicalColumn := 123, sourceArmColumn := 46888, finalColumn := 871354 }
, { child := 11, logicalColumn := 124, sourceArmColumn := 46893, finalColumn := 871359 }
, { child := 11, logicalColumn := 125, sourceArmColumn := 46898, finalColumn := 871364 }
, { child := 11, logicalColumn := 126, sourceArmColumn := 46903, finalColumn := 871369 }
, { child := 11, logicalColumn := 127, sourceArmColumn := 46908, finalColumn := 871374 }
, { child := 11, logicalColumn := 128, sourceArmColumn := 46913, finalColumn := 871379 }
, { child := 11, logicalColumn := 129, sourceArmColumn := 46918, finalColumn := 871384 }
, { child := 11, logicalColumn := 130, sourceArmColumn := 46923, finalColumn := 871389 }
, { child := 11, logicalColumn := 131, sourceArmColumn := 46928, finalColumn := 871394 }
, { child := 11, logicalColumn := 132, sourceArmColumn := 46933, finalColumn := 871399 }
, { child := 11, logicalColumn := 133, sourceArmColumn := 46938, finalColumn := 871404 }
, { child := 11, logicalColumn := 134, sourceArmColumn := 46943, finalColumn := 871409 }
, { child := 11, logicalColumn := 135, sourceArmColumn := 46948, finalColumn := 871414 }
, { child := 11, logicalColumn := 136, sourceArmColumn := 46953, finalColumn := 871419 }
, { child := 11, logicalColumn := 137, sourceArmColumn := 46958, finalColumn := 871424 }
, { child := 11, logicalColumn := 138, sourceArmColumn := 46963, finalColumn := 871429 }
, { child := 11, logicalColumn := 139, sourceArmColumn := 46968, finalColumn := 871434 }
, { child := 11, logicalColumn := 140, sourceArmColumn := 46973, finalColumn := 871439 }
, { child := 11, logicalColumn := 141, sourceArmColumn := 46978, finalColumn := 871444 }
, { child := 11, logicalColumn := 142, sourceArmColumn := 46983, finalColumn := 871449 }
, { child := 11, logicalColumn := 143, sourceArmColumn := 46988, finalColumn := 871454 }
, { child := 11, logicalColumn := 144, sourceArmColumn := 46993, finalColumn := 871459 }
, { child := 11, logicalColumn := 145, sourceArmColumn := 46998, finalColumn := 871464 }
, { child := 11, logicalColumn := 146, sourceArmColumn := 47003, finalColumn := 871469 }
, { child := 11, logicalColumn := 147, sourceArmColumn := 47008, finalColumn := 871474 }
, { child := 11, logicalColumn := 148, sourceArmColumn := 47013, finalColumn := 871479 }
, { child := 11, logicalColumn := 149, sourceArmColumn := 47018, finalColumn := 871484 }
, { child := 11, logicalColumn := 150, sourceArmColumn := 47023, finalColumn := 871489 }
, { child := 11, logicalColumn := 151, sourceArmColumn := 47028, finalColumn := 871494 }
, { child := 11, logicalColumn := 152, sourceArmColumn := 47033, finalColumn := 871499 }
, { child := 11, logicalColumn := 153, sourceArmColumn := 47038, finalColumn := 871504 }
, { child := 11, logicalColumn := 154, sourceArmColumn := 47043, finalColumn := 871509 }
, { child := 11, logicalColumn := 155, sourceArmColumn := 47048, finalColumn := 871514 }
, { child := 11, logicalColumn := 156, sourceArmColumn := 47053, finalColumn := 871519 }
, { child := 11, logicalColumn := 157, sourceArmColumn := 47058, finalColumn := 871524 }
, { child := 11, logicalColumn := 158, sourceArmColumn := 47063, finalColumn := 871529 }
, { child := 11, logicalColumn := 159, sourceArmColumn := 47068, finalColumn := 871534 }
, { child := 11, logicalColumn := 160, sourceArmColumn := 47073, finalColumn := 871539 }
, { child := 11, logicalColumn := 161, sourceArmColumn := 47078, finalColumn := 871544 }
, { child := 11, logicalColumn := 162, sourceArmColumn := 46814, finalColumn := 871280 }
, { child := 11, logicalColumn := 163, sourceArmColumn := 46819, finalColumn := 871285 }
, { child := 11, logicalColumn := 164, sourceArmColumn := 46824, finalColumn := 871290 }
, { child := 11, logicalColumn := 165, sourceArmColumn := 46829, finalColumn := 871295 }
, { child := 11, logicalColumn := 166, sourceArmColumn := 46834, finalColumn := 871300 }
, { child := 11, logicalColumn := 167, sourceArmColumn := 46839, finalColumn := 871305 }
, { child := 11, logicalColumn := 168, sourceArmColumn := 46844, finalColumn := 871310 }
, { child := 11, logicalColumn := 169, sourceArmColumn := 46849, finalColumn := 871315 }
, { child := 11, logicalColumn := 170, sourceArmColumn := 46854, finalColumn := 871320 }
, { child := 11, logicalColumn := 171, sourceArmColumn := 46859, finalColumn := 871325 }
, { child := 11, logicalColumn := 172, sourceArmColumn := 46864, finalColumn := 871330 }
, { child := 11, logicalColumn := 173, sourceArmColumn := 46869, finalColumn := 871335 }
, { child := 11, logicalColumn := 174, sourceArmColumn := 46874, finalColumn := 871340 }
, { child := 11, logicalColumn := 175, sourceArmColumn := 46879, finalColumn := 871345 }
, { child := 11, logicalColumn := 176, sourceArmColumn := 46884, finalColumn := 871350 }
, { child := 11, logicalColumn := 177, sourceArmColumn := 46889, finalColumn := 871355 }
, { child := 11, logicalColumn := 178, sourceArmColumn := 46894, finalColumn := 871360 }
, { child := 11, logicalColumn := 179, sourceArmColumn := 46899, finalColumn := 871365 }
, { child := 11, logicalColumn := 180, sourceArmColumn := 46904, finalColumn := 871370 }
, { child := 11, logicalColumn := 181, sourceArmColumn := 46909, finalColumn := 871375 }
, { child := 11, logicalColumn := 182, sourceArmColumn := 46914, finalColumn := 871380 }
, { child := 11, logicalColumn := 183, sourceArmColumn := 46919, finalColumn := 871385 }
, { child := 11, logicalColumn := 184, sourceArmColumn := 46924, finalColumn := 871390 }
, { child := 11, logicalColumn := 185, sourceArmColumn := 46929, finalColumn := 871395 }
, { child := 11, logicalColumn := 186, sourceArmColumn := 46934, finalColumn := 871400 }
, { child := 11, logicalColumn := 187, sourceArmColumn := 46939, finalColumn := 871405 }
, { child := 11, logicalColumn := 188, sourceArmColumn := 46944, finalColumn := 871410 }
, { child := 11, logicalColumn := 189, sourceArmColumn := 46949, finalColumn := 871415 }
, { child := 11, logicalColumn := 190, sourceArmColumn := 46954, finalColumn := 871420 }
, { child := 11, logicalColumn := 191, sourceArmColumn := 46959, finalColumn := 871425 }
, { child := 11, logicalColumn := 192, sourceArmColumn := 46964, finalColumn := 871430 }
, { child := 11, logicalColumn := 193, sourceArmColumn := 46969, finalColumn := 871435 }
, { child := 11, logicalColumn := 194, sourceArmColumn := 46974, finalColumn := 871440 }
, { child := 11, logicalColumn := 195, sourceArmColumn := 46979, finalColumn := 871445 }
, { child := 11, logicalColumn := 196, sourceArmColumn := 46984, finalColumn := 871450 }
, { child := 11, logicalColumn := 197, sourceArmColumn := 46989, finalColumn := 871455 }
, { child := 11, logicalColumn := 198, sourceArmColumn := 46994, finalColumn := 871460 }
, { child := 11, logicalColumn := 199, sourceArmColumn := 46999, finalColumn := 871465 }
, { child := 11, logicalColumn := 200, sourceArmColumn := 47004, finalColumn := 871470 }
, { child := 11, logicalColumn := 201, sourceArmColumn := 47009, finalColumn := 871475 }
, { child := 11, logicalColumn := 202, sourceArmColumn := 47014, finalColumn := 871480 }
, { child := 11, logicalColumn := 203, sourceArmColumn := 47019, finalColumn := 871485 }
, { child := 11, logicalColumn := 204, sourceArmColumn := 47024, finalColumn := 871490 }
, { child := 11, logicalColumn := 205, sourceArmColumn := 47029, finalColumn := 871495 }
, { child := 11, logicalColumn := 206, sourceArmColumn := 47034, finalColumn := 871500 }
, { child := 11, logicalColumn := 207, sourceArmColumn := 47039, finalColumn := 871505 }
, { child := 11, logicalColumn := 208, sourceArmColumn := 47044, finalColumn := 871510 }
, { child := 11, logicalColumn := 209, sourceArmColumn := 47049, finalColumn := 871515 }
, { child := 11, logicalColumn := 210, sourceArmColumn := 47054, finalColumn := 871520 }
, { child := 11, logicalColumn := 211, sourceArmColumn := 47059, finalColumn := 871525 }
, { child := 11, logicalColumn := 212, sourceArmColumn := 47064, finalColumn := 871530 }
, { child := 11, logicalColumn := 213, sourceArmColumn := 47069, finalColumn := 871535 }
, { child := 11, logicalColumn := 214, sourceArmColumn := 47074, finalColumn := 871540 }
, { child := 11, logicalColumn := 215, sourceArmColumn := 47079, finalColumn := 871545 }
, { child := 11, logicalColumn := 216, sourceArmColumn := 46815, finalColumn := 871281 }
, { child := 11, logicalColumn := 217, sourceArmColumn := 46820, finalColumn := 871286 }
, { child := 11, logicalColumn := 218, sourceArmColumn := 46825, finalColumn := 871291 }
, { child := 11, logicalColumn := 219, sourceArmColumn := 46830, finalColumn := 871296 }
, { child := 11, logicalColumn := 220, sourceArmColumn := 46835, finalColumn := 871301 }
, { child := 11, logicalColumn := 221, sourceArmColumn := 46840, finalColumn := 871306 }
, { child := 11, logicalColumn := 222, sourceArmColumn := 46845, finalColumn := 871311 }
, { child := 11, logicalColumn := 223, sourceArmColumn := 46850, finalColumn := 871316 }
, { child := 11, logicalColumn := 224, sourceArmColumn := 46855, finalColumn := 871321 }
, { child := 11, logicalColumn := 225, sourceArmColumn := 46860, finalColumn := 871326 }
, { child := 11, logicalColumn := 226, sourceArmColumn := 46865, finalColumn := 871331 }
, { child := 11, logicalColumn := 227, sourceArmColumn := 46870, finalColumn := 871336 }
, { child := 11, logicalColumn := 228, sourceArmColumn := 46875, finalColumn := 871341 }
, { child := 11, logicalColumn := 229, sourceArmColumn := 46880, finalColumn := 871346 }
, { child := 11, logicalColumn := 230, sourceArmColumn := 46885, finalColumn := 871351 }
, { child := 11, logicalColumn := 231, sourceArmColumn := 46890, finalColumn := 871356 }
, { child := 11, logicalColumn := 232, sourceArmColumn := 46895, finalColumn := 871361 }
, { child := 11, logicalColumn := 233, sourceArmColumn := 46900, finalColumn := 871366 }
, { child := 11, logicalColumn := 234, sourceArmColumn := 46905, finalColumn := 871371 }
, { child := 11, logicalColumn := 235, sourceArmColumn := 46910, finalColumn := 871376 }
, { child := 11, logicalColumn := 236, sourceArmColumn := 46915, finalColumn := 871381 }
, { child := 11, logicalColumn := 237, sourceArmColumn := 46920, finalColumn := 871386 }
, { child := 11, logicalColumn := 238, sourceArmColumn := 46925, finalColumn := 871391 }
, { child := 11, logicalColumn := 239, sourceArmColumn := 46930, finalColumn := 871396 }
, { child := 11, logicalColumn := 240, sourceArmColumn := 46935, finalColumn := 871401 }
, { child := 11, logicalColumn := 241, sourceArmColumn := 46940, finalColumn := 871406 }
, { child := 11, logicalColumn := 242, sourceArmColumn := 46945, finalColumn := 871411 }
, { child := 11, logicalColumn := 243, sourceArmColumn := 46950, finalColumn := 871416 }
, { child := 11, logicalColumn := 244, sourceArmColumn := 46955, finalColumn := 871421 }
, { child := 11, logicalColumn := 245, sourceArmColumn := 46960, finalColumn := 871426 }
, { child := 11, logicalColumn := 246, sourceArmColumn := 46965, finalColumn := 871431 }
, { child := 11, logicalColumn := 247, sourceArmColumn := 46970, finalColumn := 871436 }
, { child := 11, logicalColumn := 248, sourceArmColumn := 46975, finalColumn := 871441 }
, { child := 11, logicalColumn := 249, sourceArmColumn := 46980, finalColumn := 871446 }
, { child := 11, logicalColumn := 250, sourceArmColumn := 46985, finalColumn := 871451 }
, { child := 11, logicalColumn := 251, sourceArmColumn := 46990, finalColumn := 871456 }
, { child := 11, logicalColumn := 252, sourceArmColumn := 46995, finalColumn := 871461 }
, { child := 11, logicalColumn := 253, sourceArmColumn := 47000, finalColumn := 871466 }
, { child := 11, logicalColumn := 254, sourceArmColumn := 47005, finalColumn := 871471 }
, { child := 11, logicalColumn := 255, sourceArmColumn := 47010, finalColumn := 871476 }
, { child := 11, logicalColumn := 256, sourceArmColumn := 47015, finalColumn := 871481 }
, { child := 11, logicalColumn := 257, sourceArmColumn := 47020, finalColumn := 871486 }
, { child := 11, logicalColumn := 258, sourceArmColumn := 47025, finalColumn := 871491 }
, { child := 11, logicalColumn := 259, sourceArmColumn := 47030, finalColumn := 871496 }
, { child := 11, logicalColumn := 260, sourceArmColumn := 47035, finalColumn := 871501 }
, { child := 11, logicalColumn := 261, sourceArmColumn := 47040, finalColumn := 871506 }
, { child := 11, logicalColumn := 262, sourceArmColumn := 47045, finalColumn := 871511 }
, { child := 11, logicalColumn := 263, sourceArmColumn := 47050, finalColumn := 871516 }
, { child := 11, logicalColumn := 264, sourceArmColumn := 47055, finalColumn := 871521 }
, { child := 11, logicalColumn := 265, sourceArmColumn := 47060, finalColumn := 871526 }
, { child := 11, logicalColumn := 266, sourceArmColumn := 47065, finalColumn := 871531 }
, { child := 11, logicalColumn := 267, sourceArmColumn := 47070, finalColumn := 871536 }
, { child := 11, logicalColumn := 268, sourceArmColumn := 47075, finalColumn := 871541 }
, { child := 11, logicalColumn := 269, sourceArmColumn := 47080, finalColumn := 871546 }
, { child := 12, logicalColumn := 0, sourceArmColumn := 49083, finalColumn := 937967 }
, { child := 12, logicalColumn := 1, sourceArmColumn := 49088, finalColumn := 937972 }
, { child := 12, logicalColumn := 2, sourceArmColumn := 49093, finalColumn := 937977 }
, { child := 12, logicalColumn := 3, sourceArmColumn := 49098, finalColumn := 937982 }
, { child := 12, logicalColumn := 4, sourceArmColumn := 49103, finalColumn := 937987 }
, { child := 12, logicalColumn := 5, sourceArmColumn := 49108, finalColumn := 937992 }
, { child := 12, logicalColumn := 6, sourceArmColumn := 49113, finalColumn := 937997 }
, { child := 12, logicalColumn := 7, sourceArmColumn := 49118, finalColumn := 938002 }
, { child := 12, logicalColumn := 8, sourceArmColumn := 49123, finalColumn := 938007 }
, { child := 12, logicalColumn := 9, sourceArmColumn := 49128, finalColumn := 938012 }
, { child := 12, logicalColumn := 10, sourceArmColumn := 49133, finalColumn := 938017 }
, { child := 12, logicalColumn := 11, sourceArmColumn := 49138, finalColumn := 938022 }
, { child := 12, logicalColumn := 12, sourceArmColumn := 49143, finalColumn := 938027 }
, { child := 12, logicalColumn := 13, sourceArmColumn := 49148, finalColumn := 938032 }
, { child := 12, logicalColumn := 14, sourceArmColumn := 49153, finalColumn := 938037 }
, { child := 12, logicalColumn := 15, sourceArmColumn := 49158, finalColumn := 938042 }
, { child := 12, logicalColumn := 16, sourceArmColumn := 49163, finalColumn := 938047 }
, { child := 12, logicalColumn := 17, sourceArmColumn := 49168, finalColumn := 938052 }
, { child := 12, logicalColumn := 18, sourceArmColumn := 49173, finalColumn := 938057 }
, { child := 12, logicalColumn := 19, sourceArmColumn := 49178, finalColumn := 938062 }
, { child := 12, logicalColumn := 20, sourceArmColumn := 49183, finalColumn := 938067 }
, { child := 12, logicalColumn := 21, sourceArmColumn := 49188, finalColumn := 938072 }
, { child := 12, logicalColumn := 22, sourceArmColumn := 49193, finalColumn := 938077 }
, { child := 12, logicalColumn := 23, sourceArmColumn := 49198, finalColumn := 938082 }
, { child := 12, logicalColumn := 24, sourceArmColumn := 49203, finalColumn := 938087 }
, { child := 12, logicalColumn := 25, sourceArmColumn := 49208, finalColumn := 938092 }
, { child := 12, logicalColumn := 26, sourceArmColumn := 49213, finalColumn := 938097 }
, { child := 12, logicalColumn := 27, sourceArmColumn := 49218, finalColumn := 938102 }
, { child := 12, logicalColumn := 28, sourceArmColumn := 49223, finalColumn := 938107 }
, { child := 12, logicalColumn := 29, sourceArmColumn := 49228, finalColumn := 938112 }
, { child := 12, logicalColumn := 30, sourceArmColumn := 49233, finalColumn := 938117 }
, { child := 12, logicalColumn := 31, sourceArmColumn := 49238, finalColumn := 938122 }
, { child := 12, logicalColumn := 32, sourceArmColumn := 49243, finalColumn := 938127 }
, { child := 12, logicalColumn := 33, sourceArmColumn := 49248, finalColumn := 938132 }
, { child := 12, logicalColumn := 34, sourceArmColumn := 49253, finalColumn := 938137 }
, { child := 12, logicalColumn := 35, sourceArmColumn := 49258, finalColumn := 938142 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk12
