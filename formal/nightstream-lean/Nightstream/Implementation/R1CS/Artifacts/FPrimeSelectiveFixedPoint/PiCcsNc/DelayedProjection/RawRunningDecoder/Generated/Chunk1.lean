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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk1

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 0, logicalColumn := 252, sourceArmColumn := 22003, finalColumn := 134181 }
, { child := 0, logicalColumn := 253, sourceArmColumn := 22008, finalColumn := 134186 }
, { child := 0, logicalColumn := 254, sourceArmColumn := 22013, finalColumn := 134191 }
, { child := 0, logicalColumn := 255, sourceArmColumn := 22018, finalColumn := 134196 }
, { child := 0, logicalColumn := 256, sourceArmColumn := 22023, finalColumn := 134201 }
, { child := 0, logicalColumn := 257, sourceArmColumn := 22028, finalColumn := 134206 }
, { child := 0, logicalColumn := 258, sourceArmColumn := 22033, finalColumn := 134211 }
, { child := 0, logicalColumn := 259, sourceArmColumn := 22038, finalColumn := 134216 }
, { child := 0, logicalColumn := 260, sourceArmColumn := 22043, finalColumn := 134221 }
, { child := 0, logicalColumn := 261, sourceArmColumn := 22048, finalColumn := 134226 }
, { child := 0, logicalColumn := 262, sourceArmColumn := 22053, finalColumn := 134231 }
, { child := 0, logicalColumn := 263, sourceArmColumn := 22058, finalColumn := 134236 }
, { child := 0, logicalColumn := 264, sourceArmColumn := 22063, finalColumn := 134241 }
, { child := 0, logicalColumn := 265, sourceArmColumn := 22068, finalColumn := 134246 }
, { child := 0, logicalColumn := 266, sourceArmColumn := 22073, finalColumn := 134251 }
, { child := 0, logicalColumn := 267, sourceArmColumn := 22078, finalColumn := 134256 }
, { child := 0, logicalColumn := 268, sourceArmColumn := 22083, finalColumn := 134261 }
, { child := 0, logicalColumn := 269, sourceArmColumn := 22088, finalColumn := 134266 }
, { child := 1, logicalColumn := 0, sourceArmColumn := 24091, finalColumn := 204377 }
, { child := 1, logicalColumn := 1, sourceArmColumn := 24096, finalColumn := 204382 }
, { child := 1, logicalColumn := 2, sourceArmColumn := 24101, finalColumn := 204387 }
, { child := 1, logicalColumn := 3, sourceArmColumn := 24106, finalColumn := 204392 }
, { child := 1, logicalColumn := 4, sourceArmColumn := 24111, finalColumn := 204397 }
, { child := 1, logicalColumn := 5, sourceArmColumn := 24116, finalColumn := 204402 }
, { child := 1, logicalColumn := 6, sourceArmColumn := 24121, finalColumn := 204407 }
, { child := 1, logicalColumn := 7, sourceArmColumn := 24126, finalColumn := 204412 }
, { child := 1, logicalColumn := 8, sourceArmColumn := 24131, finalColumn := 204417 }
, { child := 1, logicalColumn := 9, sourceArmColumn := 24136, finalColumn := 204422 }
, { child := 1, logicalColumn := 10, sourceArmColumn := 24141, finalColumn := 204427 }
, { child := 1, logicalColumn := 11, sourceArmColumn := 24146, finalColumn := 204432 }
, { child := 1, logicalColumn := 12, sourceArmColumn := 24151, finalColumn := 204437 }
, { child := 1, logicalColumn := 13, sourceArmColumn := 24156, finalColumn := 204442 }
, { child := 1, logicalColumn := 14, sourceArmColumn := 24161, finalColumn := 204447 }
, { child := 1, logicalColumn := 15, sourceArmColumn := 24166, finalColumn := 204452 }
, { child := 1, logicalColumn := 16, sourceArmColumn := 24171, finalColumn := 204457 }
, { child := 1, logicalColumn := 17, sourceArmColumn := 24176, finalColumn := 204462 }
, { child := 1, logicalColumn := 18, sourceArmColumn := 24181, finalColumn := 204467 }
, { child := 1, logicalColumn := 19, sourceArmColumn := 24186, finalColumn := 204472 }
, { child := 1, logicalColumn := 20, sourceArmColumn := 24191, finalColumn := 204477 }
, { child := 1, logicalColumn := 21, sourceArmColumn := 24196, finalColumn := 204482 }
, { child := 1, logicalColumn := 22, sourceArmColumn := 24201, finalColumn := 204487 }
, { child := 1, logicalColumn := 23, sourceArmColumn := 24206, finalColumn := 204492 }
, { child := 1, logicalColumn := 24, sourceArmColumn := 24211, finalColumn := 204497 }
, { child := 1, logicalColumn := 25, sourceArmColumn := 24216, finalColumn := 204502 }
, { child := 1, logicalColumn := 26, sourceArmColumn := 24221, finalColumn := 204507 }
, { child := 1, logicalColumn := 27, sourceArmColumn := 24226, finalColumn := 204512 }
, { child := 1, logicalColumn := 28, sourceArmColumn := 24231, finalColumn := 204517 }
, { child := 1, logicalColumn := 29, sourceArmColumn := 24236, finalColumn := 204522 }
, { child := 1, logicalColumn := 30, sourceArmColumn := 24241, finalColumn := 204527 }
, { child := 1, logicalColumn := 31, sourceArmColumn := 24246, finalColumn := 204532 }
, { child := 1, logicalColumn := 32, sourceArmColumn := 24251, finalColumn := 204537 }
, { child := 1, logicalColumn := 33, sourceArmColumn := 24256, finalColumn := 204542 }
, { child := 1, logicalColumn := 34, sourceArmColumn := 24261, finalColumn := 204547 }
, { child := 1, logicalColumn := 35, sourceArmColumn := 24266, finalColumn := 204552 }
, { child := 1, logicalColumn := 36, sourceArmColumn := 24271, finalColumn := 204557 }
, { child := 1, logicalColumn := 37, sourceArmColumn := 24276, finalColumn := 204562 }
, { child := 1, logicalColumn := 38, sourceArmColumn := 24281, finalColumn := 204567 }
, { child := 1, logicalColumn := 39, sourceArmColumn := 24286, finalColumn := 204572 }
, { child := 1, logicalColumn := 40, sourceArmColumn := 24291, finalColumn := 204577 }
, { child := 1, logicalColumn := 41, sourceArmColumn := 24296, finalColumn := 204582 }
, { child := 1, logicalColumn := 42, sourceArmColumn := 24301, finalColumn := 204587 }
, { child := 1, logicalColumn := 43, sourceArmColumn := 24306, finalColumn := 204592 }
, { child := 1, logicalColumn := 44, sourceArmColumn := 24311, finalColumn := 204597 }
, { child := 1, logicalColumn := 45, sourceArmColumn := 24316, finalColumn := 204602 }
, { child := 1, logicalColumn := 46, sourceArmColumn := 24321, finalColumn := 204607 }
, { child := 1, logicalColumn := 47, sourceArmColumn := 24326, finalColumn := 204612 }
, { child := 1, logicalColumn := 48, sourceArmColumn := 24331, finalColumn := 204617 }
, { child := 1, logicalColumn := 49, sourceArmColumn := 24336, finalColumn := 204622 }
, { child := 1, logicalColumn := 50, sourceArmColumn := 24341, finalColumn := 204627 }
, { child := 1, logicalColumn := 51, sourceArmColumn := 24346, finalColumn := 204632 }
, { child := 1, logicalColumn := 52, sourceArmColumn := 24351, finalColumn := 204637 }
, { child := 1, logicalColumn := 53, sourceArmColumn := 24356, finalColumn := 204642 }
, { child := 1, logicalColumn := 54, sourceArmColumn := 24092, finalColumn := 204378 }
, { child := 1, logicalColumn := 55, sourceArmColumn := 24097, finalColumn := 204383 }
, { child := 1, logicalColumn := 56, sourceArmColumn := 24102, finalColumn := 204388 }
, { child := 1, logicalColumn := 57, sourceArmColumn := 24107, finalColumn := 204393 }
, { child := 1, logicalColumn := 58, sourceArmColumn := 24112, finalColumn := 204398 }
, { child := 1, logicalColumn := 59, sourceArmColumn := 24117, finalColumn := 204403 }
, { child := 1, logicalColumn := 60, sourceArmColumn := 24122, finalColumn := 204408 }
, { child := 1, logicalColumn := 61, sourceArmColumn := 24127, finalColumn := 204413 }
, { child := 1, logicalColumn := 62, sourceArmColumn := 24132, finalColumn := 204418 }
, { child := 1, logicalColumn := 63, sourceArmColumn := 24137, finalColumn := 204423 }
, { child := 1, logicalColumn := 64, sourceArmColumn := 24142, finalColumn := 204428 }
, { child := 1, logicalColumn := 65, sourceArmColumn := 24147, finalColumn := 204433 }
, { child := 1, logicalColumn := 66, sourceArmColumn := 24152, finalColumn := 204438 }
, { child := 1, logicalColumn := 67, sourceArmColumn := 24157, finalColumn := 204443 }
, { child := 1, logicalColumn := 68, sourceArmColumn := 24162, finalColumn := 204448 }
, { child := 1, logicalColumn := 69, sourceArmColumn := 24167, finalColumn := 204453 }
, { child := 1, logicalColumn := 70, sourceArmColumn := 24172, finalColumn := 204458 }
, { child := 1, logicalColumn := 71, sourceArmColumn := 24177, finalColumn := 204463 }
, { child := 1, logicalColumn := 72, sourceArmColumn := 24182, finalColumn := 204468 }
, { child := 1, logicalColumn := 73, sourceArmColumn := 24187, finalColumn := 204473 }
, { child := 1, logicalColumn := 74, sourceArmColumn := 24192, finalColumn := 204478 }
, { child := 1, logicalColumn := 75, sourceArmColumn := 24197, finalColumn := 204483 }
, { child := 1, logicalColumn := 76, sourceArmColumn := 24202, finalColumn := 204488 }
, { child := 1, logicalColumn := 77, sourceArmColumn := 24207, finalColumn := 204493 }
, { child := 1, logicalColumn := 78, sourceArmColumn := 24212, finalColumn := 204498 }
, { child := 1, logicalColumn := 79, sourceArmColumn := 24217, finalColumn := 204503 }
, { child := 1, logicalColumn := 80, sourceArmColumn := 24222, finalColumn := 204508 }
, { child := 1, logicalColumn := 81, sourceArmColumn := 24227, finalColumn := 204513 }
, { child := 1, logicalColumn := 82, sourceArmColumn := 24232, finalColumn := 204518 }
, { child := 1, logicalColumn := 83, sourceArmColumn := 24237, finalColumn := 204523 }
, { child := 1, logicalColumn := 84, sourceArmColumn := 24242, finalColumn := 204528 }
, { child := 1, logicalColumn := 85, sourceArmColumn := 24247, finalColumn := 204533 }
, { child := 1, logicalColumn := 86, sourceArmColumn := 24252, finalColumn := 204538 }
, { child := 1, logicalColumn := 87, sourceArmColumn := 24257, finalColumn := 204543 }
, { child := 1, logicalColumn := 88, sourceArmColumn := 24262, finalColumn := 204548 }
, { child := 1, logicalColumn := 89, sourceArmColumn := 24267, finalColumn := 204553 }
, { child := 1, logicalColumn := 90, sourceArmColumn := 24272, finalColumn := 204558 }
, { child := 1, logicalColumn := 91, sourceArmColumn := 24277, finalColumn := 204563 }
, { child := 1, logicalColumn := 92, sourceArmColumn := 24282, finalColumn := 204568 }
, { child := 1, logicalColumn := 93, sourceArmColumn := 24287, finalColumn := 204573 }
, { child := 1, logicalColumn := 94, sourceArmColumn := 24292, finalColumn := 204578 }
, { child := 1, logicalColumn := 95, sourceArmColumn := 24297, finalColumn := 204583 }
, { child := 1, logicalColumn := 96, sourceArmColumn := 24302, finalColumn := 204588 }
, { child := 1, logicalColumn := 97, sourceArmColumn := 24307, finalColumn := 204593 }
, { child := 1, logicalColumn := 98, sourceArmColumn := 24312, finalColumn := 204598 }
, { child := 1, logicalColumn := 99, sourceArmColumn := 24317, finalColumn := 204603 }
, { child := 1, logicalColumn := 100, sourceArmColumn := 24322, finalColumn := 204608 }
, { child := 1, logicalColumn := 101, sourceArmColumn := 24327, finalColumn := 204613 }
, { child := 1, logicalColumn := 102, sourceArmColumn := 24332, finalColumn := 204618 }
, { child := 1, logicalColumn := 103, sourceArmColumn := 24337, finalColumn := 204623 }
, { child := 1, logicalColumn := 104, sourceArmColumn := 24342, finalColumn := 204628 }
, { child := 1, logicalColumn := 105, sourceArmColumn := 24347, finalColumn := 204633 }
, { child := 1, logicalColumn := 106, sourceArmColumn := 24352, finalColumn := 204638 }
, { child := 1, logicalColumn := 107, sourceArmColumn := 24357, finalColumn := 204643 }
, { child := 1, logicalColumn := 108, sourceArmColumn := 24093, finalColumn := 204379 }
, { child := 1, logicalColumn := 109, sourceArmColumn := 24098, finalColumn := 204384 }
, { child := 1, logicalColumn := 110, sourceArmColumn := 24103, finalColumn := 204389 }
, { child := 1, logicalColumn := 111, sourceArmColumn := 24108, finalColumn := 204394 }
, { child := 1, logicalColumn := 112, sourceArmColumn := 24113, finalColumn := 204399 }
, { child := 1, logicalColumn := 113, sourceArmColumn := 24118, finalColumn := 204404 }
, { child := 1, logicalColumn := 114, sourceArmColumn := 24123, finalColumn := 204409 }
, { child := 1, logicalColumn := 115, sourceArmColumn := 24128, finalColumn := 204414 }
, { child := 1, logicalColumn := 116, sourceArmColumn := 24133, finalColumn := 204419 }
, { child := 1, logicalColumn := 117, sourceArmColumn := 24138, finalColumn := 204424 }
, { child := 1, logicalColumn := 118, sourceArmColumn := 24143, finalColumn := 204429 }
, { child := 1, logicalColumn := 119, sourceArmColumn := 24148, finalColumn := 204434 }
, { child := 1, logicalColumn := 120, sourceArmColumn := 24153, finalColumn := 204439 }
, { child := 1, logicalColumn := 121, sourceArmColumn := 24158, finalColumn := 204444 }
, { child := 1, logicalColumn := 122, sourceArmColumn := 24163, finalColumn := 204449 }
, { child := 1, logicalColumn := 123, sourceArmColumn := 24168, finalColumn := 204454 }
, { child := 1, logicalColumn := 124, sourceArmColumn := 24173, finalColumn := 204459 }
, { child := 1, logicalColumn := 125, sourceArmColumn := 24178, finalColumn := 204464 }
, { child := 1, logicalColumn := 126, sourceArmColumn := 24183, finalColumn := 204469 }
, { child := 1, logicalColumn := 127, sourceArmColumn := 24188, finalColumn := 204474 }
, { child := 1, logicalColumn := 128, sourceArmColumn := 24193, finalColumn := 204479 }
, { child := 1, logicalColumn := 129, sourceArmColumn := 24198, finalColumn := 204484 }
, { child := 1, logicalColumn := 130, sourceArmColumn := 24203, finalColumn := 204489 }
, { child := 1, logicalColumn := 131, sourceArmColumn := 24208, finalColumn := 204494 }
, { child := 1, logicalColumn := 132, sourceArmColumn := 24213, finalColumn := 204499 }
, { child := 1, logicalColumn := 133, sourceArmColumn := 24218, finalColumn := 204504 }
, { child := 1, logicalColumn := 134, sourceArmColumn := 24223, finalColumn := 204509 }
, { child := 1, logicalColumn := 135, sourceArmColumn := 24228, finalColumn := 204514 }
, { child := 1, logicalColumn := 136, sourceArmColumn := 24233, finalColumn := 204519 }
, { child := 1, logicalColumn := 137, sourceArmColumn := 24238, finalColumn := 204524 }
, { child := 1, logicalColumn := 138, sourceArmColumn := 24243, finalColumn := 204529 }
, { child := 1, logicalColumn := 139, sourceArmColumn := 24248, finalColumn := 204534 }
, { child := 1, logicalColumn := 140, sourceArmColumn := 24253, finalColumn := 204539 }
, { child := 1, logicalColumn := 141, sourceArmColumn := 24258, finalColumn := 204544 }
, { child := 1, logicalColumn := 142, sourceArmColumn := 24263, finalColumn := 204549 }
, { child := 1, logicalColumn := 143, sourceArmColumn := 24268, finalColumn := 204554 }
, { child := 1, logicalColumn := 144, sourceArmColumn := 24273, finalColumn := 204559 }
, { child := 1, logicalColumn := 145, sourceArmColumn := 24278, finalColumn := 204564 }
, { child := 1, logicalColumn := 146, sourceArmColumn := 24283, finalColumn := 204569 }
, { child := 1, logicalColumn := 147, sourceArmColumn := 24288, finalColumn := 204574 }
, { child := 1, logicalColumn := 148, sourceArmColumn := 24293, finalColumn := 204579 }
, { child := 1, logicalColumn := 149, sourceArmColumn := 24298, finalColumn := 204584 }
, { child := 1, logicalColumn := 150, sourceArmColumn := 24303, finalColumn := 204589 }
, { child := 1, logicalColumn := 151, sourceArmColumn := 24308, finalColumn := 204594 }
, { child := 1, logicalColumn := 152, sourceArmColumn := 24313, finalColumn := 204599 }
, { child := 1, logicalColumn := 153, sourceArmColumn := 24318, finalColumn := 204604 }
, { child := 1, logicalColumn := 154, sourceArmColumn := 24323, finalColumn := 204609 }
, { child := 1, logicalColumn := 155, sourceArmColumn := 24328, finalColumn := 204614 }
, { child := 1, logicalColumn := 156, sourceArmColumn := 24333, finalColumn := 204619 }
, { child := 1, logicalColumn := 157, sourceArmColumn := 24338, finalColumn := 204624 }
, { child := 1, logicalColumn := 158, sourceArmColumn := 24343, finalColumn := 204629 }
, { child := 1, logicalColumn := 159, sourceArmColumn := 24348, finalColumn := 204634 }
, { child := 1, logicalColumn := 160, sourceArmColumn := 24353, finalColumn := 204639 }
, { child := 1, logicalColumn := 161, sourceArmColumn := 24358, finalColumn := 204644 }
, { child := 1, logicalColumn := 162, sourceArmColumn := 24094, finalColumn := 204380 }
, { child := 1, logicalColumn := 163, sourceArmColumn := 24099, finalColumn := 204385 }
, { child := 1, logicalColumn := 164, sourceArmColumn := 24104, finalColumn := 204390 }
, { child := 1, logicalColumn := 165, sourceArmColumn := 24109, finalColumn := 204395 }
, { child := 1, logicalColumn := 166, sourceArmColumn := 24114, finalColumn := 204400 }
, { child := 1, logicalColumn := 167, sourceArmColumn := 24119, finalColumn := 204405 }
, { child := 1, logicalColumn := 168, sourceArmColumn := 24124, finalColumn := 204410 }
, { child := 1, logicalColumn := 169, sourceArmColumn := 24129, finalColumn := 204415 }
, { child := 1, logicalColumn := 170, sourceArmColumn := 24134, finalColumn := 204420 }
, { child := 1, logicalColumn := 171, sourceArmColumn := 24139, finalColumn := 204425 }
, { child := 1, logicalColumn := 172, sourceArmColumn := 24144, finalColumn := 204430 }
, { child := 1, logicalColumn := 173, sourceArmColumn := 24149, finalColumn := 204435 }
, { child := 1, logicalColumn := 174, sourceArmColumn := 24154, finalColumn := 204440 }
, { child := 1, logicalColumn := 175, sourceArmColumn := 24159, finalColumn := 204445 }
, { child := 1, logicalColumn := 176, sourceArmColumn := 24164, finalColumn := 204450 }
, { child := 1, logicalColumn := 177, sourceArmColumn := 24169, finalColumn := 204455 }
, { child := 1, logicalColumn := 178, sourceArmColumn := 24174, finalColumn := 204460 }
, { child := 1, logicalColumn := 179, sourceArmColumn := 24179, finalColumn := 204465 }
, { child := 1, logicalColumn := 180, sourceArmColumn := 24184, finalColumn := 204470 }
, { child := 1, logicalColumn := 181, sourceArmColumn := 24189, finalColumn := 204475 }
, { child := 1, logicalColumn := 182, sourceArmColumn := 24194, finalColumn := 204480 }
, { child := 1, logicalColumn := 183, sourceArmColumn := 24199, finalColumn := 204485 }
, { child := 1, logicalColumn := 184, sourceArmColumn := 24204, finalColumn := 204490 }
, { child := 1, logicalColumn := 185, sourceArmColumn := 24209, finalColumn := 204495 }
, { child := 1, logicalColumn := 186, sourceArmColumn := 24214, finalColumn := 204500 }
, { child := 1, logicalColumn := 187, sourceArmColumn := 24219, finalColumn := 204505 }
, { child := 1, logicalColumn := 188, sourceArmColumn := 24224, finalColumn := 204510 }
, { child := 1, logicalColumn := 189, sourceArmColumn := 24229, finalColumn := 204515 }
, { child := 1, logicalColumn := 190, sourceArmColumn := 24234, finalColumn := 204520 }
, { child := 1, logicalColumn := 191, sourceArmColumn := 24239, finalColumn := 204525 }
, { child := 1, logicalColumn := 192, sourceArmColumn := 24244, finalColumn := 204530 }
, { child := 1, logicalColumn := 193, sourceArmColumn := 24249, finalColumn := 204535 }
, { child := 1, logicalColumn := 194, sourceArmColumn := 24254, finalColumn := 204540 }
, { child := 1, logicalColumn := 195, sourceArmColumn := 24259, finalColumn := 204545 }
, { child := 1, logicalColumn := 196, sourceArmColumn := 24264, finalColumn := 204550 }
, { child := 1, logicalColumn := 197, sourceArmColumn := 24269, finalColumn := 204555 }
, { child := 1, logicalColumn := 198, sourceArmColumn := 24274, finalColumn := 204560 }
, { child := 1, logicalColumn := 199, sourceArmColumn := 24279, finalColumn := 204565 }
, { child := 1, logicalColumn := 200, sourceArmColumn := 24284, finalColumn := 204570 }
, { child := 1, logicalColumn := 201, sourceArmColumn := 24289, finalColumn := 204575 }
, { child := 1, logicalColumn := 202, sourceArmColumn := 24294, finalColumn := 204580 }
, { child := 1, logicalColumn := 203, sourceArmColumn := 24299, finalColumn := 204585 }
, { child := 1, logicalColumn := 204, sourceArmColumn := 24304, finalColumn := 204590 }
, { child := 1, logicalColumn := 205, sourceArmColumn := 24309, finalColumn := 204595 }
, { child := 1, logicalColumn := 206, sourceArmColumn := 24314, finalColumn := 204600 }
, { child := 1, logicalColumn := 207, sourceArmColumn := 24319, finalColumn := 204605 }
, { child := 1, logicalColumn := 208, sourceArmColumn := 24324, finalColumn := 204610 }
, { child := 1, logicalColumn := 209, sourceArmColumn := 24329, finalColumn := 204615 }
, { child := 1, logicalColumn := 210, sourceArmColumn := 24334, finalColumn := 204620 }
, { child := 1, logicalColumn := 211, sourceArmColumn := 24339, finalColumn := 204625 }
, { child := 1, logicalColumn := 212, sourceArmColumn := 24344, finalColumn := 204630 }
, { child := 1, logicalColumn := 213, sourceArmColumn := 24349, finalColumn := 204635 }
, { child := 1, logicalColumn := 214, sourceArmColumn := 24354, finalColumn := 204640 }
, { child := 1, logicalColumn := 215, sourceArmColumn := 24359, finalColumn := 204645 }
, { child := 1, logicalColumn := 216, sourceArmColumn := 24095, finalColumn := 204381 }
, { child := 1, logicalColumn := 217, sourceArmColumn := 24100, finalColumn := 204386 }
, { child := 1, logicalColumn := 218, sourceArmColumn := 24105, finalColumn := 204391 }
, { child := 1, logicalColumn := 219, sourceArmColumn := 24110, finalColumn := 204396 }
, { child := 1, logicalColumn := 220, sourceArmColumn := 24115, finalColumn := 204401 }
, { child := 1, logicalColumn := 221, sourceArmColumn := 24120, finalColumn := 204406 }
, { child := 1, logicalColumn := 222, sourceArmColumn := 24125, finalColumn := 204411 }
, { child := 1, logicalColumn := 223, sourceArmColumn := 24130, finalColumn := 204416 }
, { child := 1, logicalColumn := 224, sourceArmColumn := 24135, finalColumn := 204421 }
, { child := 1, logicalColumn := 225, sourceArmColumn := 24140, finalColumn := 204426 }
, { child := 1, logicalColumn := 226, sourceArmColumn := 24145, finalColumn := 204431 }
, { child := 1, logicalColumn := 227, sourceArmColumn := 24150, finalColumn := 204436 }
, { child := 1, logicalColumn := 228, sourceArmColumn := 24155, finalColumn := 204441 }
, { child := 1, logicalColumn := 229, sourceArmColumn := 24160, finalColumn := 204446 }
, { child := 1, logicalColumn := 230, sourceArmColumn := 24165, finalColumn := 204451 }
, { child := 1, logicalColumn := 231, sourceArmColumn := 24170, finalColumn := 204456 }
, { child := 1, logicalColumn := 232, sourceArmColumn := 24175, finalColumn := 204461 }
, { child := 1, logicalColumn := 233, sourceArmColumn := 24180, finalColumn := 204466 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk1
