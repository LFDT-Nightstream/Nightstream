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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk4

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 3, logicalColumn := 198, sourceArmColumn := 28818, finalColumn := 337940 }
, { child := 3, logicalColumn := 199, sourceArmColumn := 28823, finalColumn := 337945 }
, { child := 3, logicalColumn := 200, sourceArmColumn := 28828, finalColumn := 337950 }
, { child := 3, logicalColumn := 201, sourceArmColumn := 28833, finalColumn := 337955 }
, { child := 3, logicalColumn := 202, sourceArmColumn := 28838, finalColumn := 337960 }
, { child := 3, logicalColumn := 203, sourceArmColumn := 28843, finalColumn := 337965 }
, { child := 3, logicalColumn := 204, sourceArmColumn := 28848, finalColumn := 337970 }
, { child := 3, logicalColumn := 205, sourceArmColumn := 28853, finalColumn := 337975 }
, { child := 3, logicalColumn := 206, sourceArmColumn := 28858, finalColumn := 337980 }
, { child := 3, logicalColumn := 207, sourceArmColumn := 28863, finalColumn := 337985 }
, { child := 3, logicalColumn := 208, sourceArmColumn := 28868, finalColumn := 337990 }
, { child := 3, logicalColumn := 209, sourceArmColumn := 28873, finalColumn := 337995 }
, { child := 3, logicalColumn := 210, sourceArmColumn := 28878, finalColumn := 338000 }
, { child := 3, logicalColumn := 211, sourceArmColumn := 28883, finalColumn := 338005 }
, { child := 3, logicalColumn := 212, sourceArmColumn := 28888, finalColumn := 338010 }
, { child := 3, logicalColumn := 213, sourceArmColumn := 28893, finalColumn := 338015 }
, { child := 3, logicalColumn := 214, sourceArmColumn := 28898, finalColumn := 338020 }
, { child := 3, logicalColumn := 215, sourceArmColumn := 28903, finalColumn := 338025 }
, { child := 3, logicalColumn := 216, sourceArmColumn := 28639, finalColumn := 337761 }
, { child := 3, logicalColumn := 217, sourceArmColumn := 28644, finalColumn := 337766 }
, { child := 3, logicalColumn := 218, sourceArmColumn := 28649, finalColumn := 337771 }
, { child := 3, logicalColumn := 219, sourceArmColumn := 28654, finalColumn := 337776 }
, { child := 3, logicalColumn := 220, sourceArmColumn := 28659, finalColumn := 337781 }
, { child := 3, logicalColumn := 221, sourceArmColumn := 28664, finalColumn := 337786 }
, { child := 3, logicalColumn := 222, sourceArmColumn := 28669, finalColumn := 337791 }
, { child := 3, logicalColumn := 223, sourceArmColumn := 28674, finalColumn := 337796 }
, { child := 3, logicalColumn := 224, sourceArmColumn := 28679, finalColumn := 337801 }
, { child := 3, logicalColumn := 225, sourceArmColumn := 28684, finalColumn := 337806 }
, { child := 3, logicalColumn := 226, sourceArmColumn := 28689, finalColumn := 337811 }
, { child := 3, logicalColumn := 227, sourceArmColumn := 28694, finalColumn := 337816 }
, { child := 3, logicalColumn := 228, sourceArmColumn := 28699, finalColumn := 337821 }
, { child := 3, logicalColumn := 229, sourceArmColumn := 28704, finalColumn := 337826 }
, { child := 3, logicalColumn := 230, sourceArmColumn := 28709, finalColumn := 337831 }
, { child := 3, logicalColumn := 231, sourceArmColumn := 28714, finalColumn := 337836 }
, { child := 3, logicalColumn := 232, sourceArmColumn := 28719, finalColumn := 337841 }
, { child := 3, logicalColumn := 233, sourceArmColumn := 28724, finalColumn := 337846 }
, { child := 3, logicalColumn := 234, sourceArmColumn := 28729, finalColumn := 337851 }
, { child := 3, logicalColumn := 235, sourceArmColumn := 28734, finalColumn := 337856 }
, { child := 3, logicalColumn := 236, sourceArmColumn := 28739, finalColumn := 337861 }
, { child := 3, logicalColumn := 237, sourceArmColumn := 28744, finalColumn := 337866 }
, { child := 3, logicalColumn := 238, sourceArmColumn := 28749, finalColumn := 337871 }
, { child := 3, logicalColumn := 239, sourceArmColumn := 28754, finalColumn := 337876 }
, { child := 3, logicalColumn := 240, sourceArmColumn := 28759, finalColumn := 337881 }
, { child := 3, logicalColumn := 241, sourceArmColumn := 28764, finalColumn := 337886 }
, { child := 3, logicalColumn := 242, sourceArmColumn := 28769, finalColumn := 337891 }
, { child := 3, logicalColumn := 243, sourceArmColumn := 28774, finalColumn := 337896 }
, { child := 3, logicalColumn := 244, sourceArmColumn := 28779, finalColumn := 337901 }
, { child := 3, logicalColumn := 245, sourceArmColumn := 28784, finalColumn := 337906 }
, { child := 3, logicalColumn := 246, sourceArmColumn := 28789, finalColumn := 337911 }
, { child := 3, logicalColumn := 247, sourceArmColumn := 28794, finalColumn := 337916 }
, { child := 3, logicalColumn := 248, sourceArmColumn := 28799, finalColumn := 337921 }
, { child := 3, logicalColumn := 249, sourceArmColumn := 28804, finalColumn := 337926 }
, { child := 3, logicalColumn := 250, sourceArmColumn := 28809, finalColumn := 337931 }
, { child := 3, logicalColumn := 251, sourceArmColumn := 28814, finalColumn := 337936 }
, { child := 3, logicalColumn := 252, sourceArmColumn := 28819, finalColumn := 337941 }
, { child := 3, logicalColumn := 253, sourceArmColumn := 28824, finalColumn := 337946 }
, { child := 3, logicalColumn := 254, sourceArmColumn := 28829, finalColumn := 337951 }
, { child := 3, logicalColumn := 255, sourceArmColumn := 28834, finalColumn := 337956 }
, { child := 3, logicalColumn := 256, sourceArmColumn := 28839, finalColumn := 337961 }
, { child := 3, logicalColumn := 257, sourceArmColumn := 28844, finalColumn := 337966 }
, { child := 3, logicalColumn := 258, sourceArmColumn := 28849, finalColumn := 337971 }
, { child := 3, logicalColumn := 259, sourceArmColumn := 28854, finalColumn := 337976 }
, { child := 3, logicalColumn := 260, sourceArmColumn := 28859, finalColumn := 337981 }
, { child := 3, logicalColumn := 261, sourceArmColumn := 28864, finalColumn := 337986 }
, { child := 3, logicalColumn := 262, sourceArmColumn := 28869, finalColumn := 337991 }
, { child := 3, logicalColumn := 263, sourceArmColumn := 28874, finalColumn := 337996 }
, { child := 3, logicalColumn := 264, sourceArmColumn := 28879, finalColumn := 338001 }
, { child := 3, logicalColumn := 265, sourceArmColumn := 28884, finalColumn := 338006 }
, { child := 3, logicalColumn := 266, sourceArmColumn := 28889, finalColumn := 338011 }
, { child := 3, logicalColumn := 267, sourceArmColumn := 28894, finalColumn := 338016 }
, { child := 3, logicalColumn := 268, sourceArmColumn := 28899, finalColumn := 338021 }
, { child := 3, logicalColumn := 269, sourceArmColumn := 28904, finalColumn := 338026 }
, { child := 4, logicalColumn := 0, sourceArmColumn := 30907, finalColumn := 404447 }
, { child := 4, logicalColumn := 1, sourceArmColumn := 30912, finalColumn := 404452 }
, { child := 4, logicalColumn := 2, sourceArmColumn := 30917, finalColumn := 404457 }
, { child := 4, logicalColumn := 3, sourceArmColumn := 30922, finalColumn := 404462 }
, { child := 4, logicalColumn := 4, sourceArmColumn := 30927, finalColumn := 404467 }
, { child := 4, logicalColumn := 5, sourceArmColumn := 30932, finalColumn := 404472 }
, { child := 4, logicalColumn := 6, sourceArmColumn := 30937, finalColumn := 404477 }
, { child := 4, logicalColumn := 7, sourceArmColumn := 30942, finalColumn := 404482 }
, { child := 4, logicalColumn := 8, sourceArmColumn := 30947, finalColumn := 404487 }
, { child := 4, logicalColumn := 9, sourceArmColumn := 30952, finalColumn := 404492 }
, { child := 4, logicalColumn := 10, sourceArmColumn := 30957, finalColumn := 404497 }
, { child := 4, logicalColumn := 11, sourceArmColumn := 30962, finalColumn := 404502 }
, { child := 4, logicalColumn := 12, sourceArmColumn := 30967, finalColumn := 404507 }
, { child := 4, logicalColumn := 13, sourceArmColumn := 30972, finalColumn := 404512 }
, { child := 4, logicalColumn := 14, sourceArmColumn := 30977, finalColumn := 404517 }
, { child := 4, logicalColumn := 15, sourceArmColumn := 30982, finalColumn := 404522 }
, { child := 4, logicalColumn := 16, sourceArmColumn := 30987, finalColumn := 404527 }
, { child := 4, logicalColumn := 17, sourceArmColumn := 30992, finalColumn := 404532 }
, { child := 4, logicalColumn := 18, sourceArmColumn := 30997, finalColumn := 404537 }
, { child := 4, logicalColumn := 19, sourceArmColumn := 31002, finalColumn := 404542 }
, { child := 4, logicalColumn := 20, sourceArmColumn := 31007, finalColumn := 404547 }
, { child := 4, logicalColumn := 21, sourceArmColumn := 31012, finalColumn := 404552 }
, { child := 4, logicalColumn := 22, sourceArmColumn := 31017, finalColumn := 404557 }
, { child := 4, logicalColumn := 23, sourceArmColumn := 31022, finalColumn := 404562 }
, { child := 4, logicalColumn := 24, sourceArmColumn := 31027, finalColumn := 404567 }
, { child := 4, logicalColumn := 25, sourceArmColumn := 31032, finalColumn := 404572 }
, { child := 4, logicalColumn := 26, sourceArmColumn := 31037, finalColumn := 404577 }
, { child := 4, logicalColumn := 27, sourceArmColumn := 31042, finalColumn := 404582 }
, { child := 4, logicalColumn := 28, sourceArmColumn := 31047, finalColumn := 404587 }
, { child := 4, logicalColumn := 29, sourceArmColumn := 31052, finalColumn := 404592 }
, { child := 4, logicalColumn := 30, sourceArmColumn := 31057, finalColumn := 404597 }
, { child := 4, logicalColumn := 31, sourceArmColumn := 31062, finalColumn := 404602 }
, { child := 4, logicalColumn := 32, sourceArmColumn := 31067, finalColumn := 404607 }
, { child := 4, logicalColumn := 33, sourceArmColumn := 31072, finalColumn := 404612 }
, { child := 4, logicalColumn := 34, sourceArmColumn := 31077, finalColumn := 404617 }
, { child := 4, logicalColumn := 35, sourceArmColumn := 31082, finalColumn := 404622 }
, { child := 4, logicalColumn := 36, sourceArmColumn := 31087, finalColumn := 404627 }
, { child := 4, logicalColumn := 37, sourceArmColumn := 31092, finalColumn := 404632 }
, { child := 4, logicalColumn := 38, sourceArmColumn := 31097, finalColumn := 404637 }
, { child := 4, logicalColumn := 39, sourceArmColumn := 31102, finalColumn := 404642 }
, { child := 4, logicalColumn := 40, sourceArmColumn := 31107, finalColumn := 404647 }
, { child := 4, logicalColumn := 41, sourceArmColumn := 31112, finalColumn := 404652 }
, { child := 4, logicalColumn := 42, sourceArmColumn := 31117, finalColumn := 404657 }
, { child := 4, logicalColumn := 43, sourceArmColumn := 31122, finalColumn := 404662 }
, { child := 4, logicalColumn := 44, sourceArmColumn := 31127, finalColumn := 404667 }
, { child := 4, logicalColumn := 45, sourceArmColumn := 31132, finalColumn := 404672 }
, { child := 4, logicalColumn := 46, sourceArmColumn := 31137, finalColumn := 404677 }
, { child := 4, logicalColumn := 47, sourceArmColumn := 31142, finalColumn := 404682 }
, { child := 4, logicalColumn := 48, sourceArmColumn := 31147, finalColumn := 404687 }
, { child := 4, logicalColumn := 49, sourceArmColumn := 31152, finalColumn := 404692 }
, { child := 4, logicalColumn := 50, sourceArmColumn := 31157, finalColumn := 404697 }
, { child := 4, logicalColumn := 51, sourceArmColumn := 31162, finalColumn := 404702 }
, { child := 4, logicalColumn := 52, sourceArmColumn := 31167, finalColumn := 404707 }
, { child := 4, logicalColumn := 53, sourceArmColumn := 31172, finalColumn := 404712 }
, { child := 4, logicalColumn := 54, sourceArmColumn := 30908, finalColumn := 404448 }
, { child := 4, logicalColumn := 55, sourceArmColumn := 30913, finalColumn := 404453 }
, { child := 4, logicalColumn := 56, sourceArmColumn := 30918, finalColumn := 404458 }
, { child := 4, logicalColumn := 57, sourceArmColumn := 30923, finalColumn := 404463 }
, { child := 4, logicalColumn := 58, sourceArmColumn := 30928, finalColumn := 404468 }
, { child := 4, logicalColumn := 59, sourceArmColumn := 30933, finalColumn := 404473 }
, { child := 4, logicalColumn := 60, sourceArmColumn := 30938, finalColumn := 404478 }
, { child := 4, logicalColumn := 61, sourceArmColumn := 30943, finalColumn := 404483 }
, { child := 4, logicalColumn := 62, sourceArmColumn := 30948, finalColumn := 404488 }
, { child := 4, logicalColumn := 63, sourceArmColumn := 30953, finalColumn := 404493 }
, { child := 4, logicalColumn := 64, sourceArmColumn := 30958, finalColumn := 404498 }
, { child := 4, logicalColumn := 65, sourceArmColumn := 30963, finalColumn := 404503 }
, { child := 4, logicalColumn := 66, sourceArmColumn := 30968, finalColumn := 404508 }
, { child := 4, logicalColumn := 67, sourceArmColumn := 30973, finalColumn := 404513 }
, { child := 4, logicalColumn := 68, sourceArmColumn := 30978, finalColumn := 404518 }
, { child := 4, logicalColumn := 69, sourceArmColumn := 30983, finalColumn := 404523 }
, { child := 4, logicalColumn := 70, sourceArmColumn := 30988, finalColumn := 404528 }
, { child := 4, logicalColumn := 71, sourceArmColumn := 30993, finalColumn := 404533 }
, { child := 4, logicalColumn := 72, sourceArmColumn := 30998, finalColumn := 404538 }
, { child := 4, logicalColumn := 73, sourceArmColumn := 31003, finalColumn := 404543 }
, { child := 4, logicalColumn := 74, sourceArmColumn := 31008, finalColumn := 404548 }
, { child := 4, logicalColumn := 75, sourceArmColumn := 31013, finalColumn := 404553 }
, { child := 4, logicalColumn := 76, sourceArmColumn := 31018, finalColumn := 404558 }
, { child := 4, logicalColumn := 77, sourceArmColumn := 31023, finalColumn := 404563 }
, { child := 4, logicalColumn := 78, sourceArmColumn := 31028, finalColumn := 404568 }
, { child := 4, logicalColumn := 79, sourceArmColumn := 31033, finalColumn := 404573 }
, { child := 4, logicalColumn := 80, sourceArmColumn := 31038, finalColumn := 404578 }
, { child := 4, logicalColumn := 81, sourceArmColumn := 31043, finalColumn := 404583 }
, { child := 4, logicalColumn := 82, sourceArmColumn := 31048, finalColumn := 404588 }
, { child := 4, logicalColumn := 83, sourceArmColumn := 31053, finalColumn := 404593 }
, { child := 4, logicalColumn := 84, sourceArmColumn := 31058, finalColumn := 404598 }
, { child := 4, logicalColumn := 85, sourceArmColumn := 31063, finalColumn := 404603 }
, { child := 4, logicalColumn := 86, sourceArmColumn := 31068, finalColumn := 404608 }
, { child := 4, logicalColumn := 87, sourceArmColumn := 31073, finalColumn := 404613 }
, { child := 4, logicalColumn := 88, sourceArmColumn := 31078, finalColumn := 404618 }
, { child := 4, logicalColumn := 89, sourceArmColumn := 31083, finalColumn := 404623 }
, { child := 4, logicalColumn := 90, sourceArmColumn := 31088, finalColumn := 404628 }
, { child := 4, logicalColumn := 91, sourceArmColumn := 31093, finalColumn := 404633 }
, { child := 4, logicalColumn := 92, sourceArmColumn := 31098, finalColumn := 404638 }
, { child := 4, logicalColumn := 93, sourceArmColumn := 31103, finalColumn := 404643 }
, { child := 4, logicalColumn := 94, sourceArmColumn := 31108, finalColumn := 404648 }
, { child := 4, logicalColumn := 95, sourceArmColumn := 31113, finalColumn := 404653 }
, { child := 4, logicalColumn := 96, sourceArmColumn := 31118, finalColumn := 404658 }
, { child := 4, logicalColumn := 97, sourceArmColumn := 31123, finalColumn := 404663 }
, { child := 4, logicalColumn := 98, sourceArmColumn := 31128, finalColumn := 404668 }
, { child := 4, logicalColumn := 99, sourceArmColumn := 31133, finalColumn := 404673 }
, { child := 4, logicalColumn := 100, sourceArmColumn := 31138, finalColumn := 404678 }
, { child := 4, logicalColumn := 101, sourceArmColumn := 31143, finalColumn := 404683 }
, { child := 4, logicalColumn := 102, sourceArmColumn := 31148, finalColumn := 404688 }
, { child := 4, logicalColumn := 103, sourceArmColumn := 31153, finalColumn := 404693 }
, { child := 4, logicalColumn := 104, sourceArmColumn := 31158, finalColumn := 404698 }
, { child := 4, logicalColumn := 105, sourceArmColumn := 31163, finalColumn := 404703 }
, { child := 4, logicalColumn := 106, sourceArmColumn := 31168, finalColumn := 404708 }
, { child := 4, logicalColumn := 107, sourceArmColumn := 31173, finalColumn := 404713 }
, { child := 4, logicalColumn := 108, sourceArmColumn := 30909, finalColumn := 404449 }
, { child := 4, logicalColumn := 109, sourceArmColumn := 30914, finalColumn := 404454 }
, { child := 4, logicalColumn := 110, sourceArmColumn := 30919, finalColumn := 404459 }
, { child := 4, logicalColumn := 111, sourceArmColumn := 30924, finalColumn := 404464 }
, { child := 4, logicalColumn := 112, sourceArmColumn := 30929, finalColumn := 404469 }
, { child := 4, logicalColumn := 113, sourceArmColumn := 30934, finalColumn := 404474 }
, { child := 4, logicalColumn := 114, sourceArmColumn := 30939, finalColumn := 404479 }
, { child := 4, logicalColumn := 115, sourceArmColumn := 30944, finalColumn := 404484 }
, { child := 4, logicalColumn := 116, sourceArmColumn := 30949, finalColumn := 404489 }
, { child := 4, logicalColumn := 117, sourceArmColumn := 30954, finalColumn := 404494 }
, { child := 4, logicalColumn := 118, sourceArmColumn := 30959, finalColumn := 404499 }
, { child := 4, logicalColumn := 119, sourceArmColumn := 30964, finalColumn := 404504 }
, { child := 4, logicalColumn := 120, sourceArmColumn := 30969, finalColumn := 404509 }
, { child := 4, logicalColumn := 121, sourceArmColumn := 30974, finalColumn := 404514 }
, { child := 4, logicalColumn := 122, sourceArmColumn := 30979, finalColumn := 404519 }
, { child := 4, logicalColumn := 123, sourceArmColumn := 30984, finalColumn := 404524 }
, { child := 4, logicalColumn := 124, sourceArmColumn := 30989, finalColumn := 404529 }
, { child := 4, logicalColumn := 125, sourceArmColumn := 30994, finalColumn := 404534 }
, { child := 4, logicalColumn := 126, sourceArmColumn := 30999, finalColumn := 404539 }
, { child := 4, logicalColumn := 127, sourceArmColumn := 31004, finalColumn := 404544 }
, { child := 4, logicalColumn := 128, sourceArmColumn := 31009, finalColumn := 404549 }
, { child := 4, logicalColumn := 129, sourceArmColumn := 31014, finalColumn := 404554 }
, { child := 4, logicalColumn := 130, sourceArmColumn := 31019, finalColumn := 404559 }
, { child := 4, logicalColumn := 131, sourceArmColumn := 31024, finalColumn := 404564 }
, { child := 4, logicalColumn := 132, sourceArmColumn := 31029, finalColumn := 404569 }
, { child := 4, logicalColumn := 133, sourceArmColumn := 31034, finalColumn := 404574 }
, { child := 4, logicalColumn := 134, sourceArmColumn := 31039, finalColumn := 404579 }
, { child := 4, logicalColumn := 135, sourceArmColumn := 31044, finalColumn := 404584 }
, { child := 4, logicalColumn := 136, sourceArmColumn := 31049, finalColumn := 404589 }
, { child := 4, logicalColumn := 137, sourceArmColumn := 31054, finalColumn := 404594 }
, { child := 4, logicalColumn := 138, sourceArmColumn := 31059, finalColumn := 404599 }
, { child := 4, logicalColumn := 139, sourceArmColumn := 31064, finalColumn := 404604 }
, { child := 4, logicalColumn := 140, sourceArmColumn := 31069, finalColumn := 404609 }
, { child := 4, logicalColumn := 141, sourceArmColumn := 31074, finalColumn := 404614 }
, { child := 4, logicalColumn := 142, sourceArmColumn := 31079, finalColumn := 404619 }
, { child := 4, logicalColumn := 143, sourceArmColumn := 31084, finalColumn := 404624 }
, { child := 4, logicalColumn := 144, sourceArmColumn := 31089, finalColumn := 404629 }
, { child := 4, logicalColumn := 145, sourceArmColumn := 31094, finalColumn := 404634 }
, { child := 4, logicalColumn := 146, sourceArmColumn := 31099, finalColumn := 404639 }
, { child := 4, logicalColumn := 147, sourceArmColumn := 31104, finalColumn := 404644 }
, { child := 4, logicalColumn := 148, sourceArmColumn := 31109, finalColumn := 404649 }
, { child := 4, logicalColumn := 149, sourceArmColumn := 31114, finalColumn := 404654 }
, { child := 4, logicalColumn := 150, sourceArmColumn := 31119, finalColumn := 404659 }
, { child := 4, logicalColumn := 151, sourceArmColumn := 31124, finalColumn := 404664 }
, { child := 4, logicalColumn := 152, sourceArmColumn := 31129, finalColumn := 404669 }
, { child := 4, logicalColumn := 153, sourceArmColumn := 31134, finalColumn := 404674 }
, { child := 4, logicalColumn := 154, sourceArmColumn := 31139, finalColumn := 404679 }
, { child := 4, logicalColumn := 155, sourceArmColumn := 31144, finalColumn := 404684 }
, { child := 4, logicalColumn := 156, sourceArmColumn := 31149, finalColumn := 404689 }
, { child := 4, logicalColumn := 157, sourceArmColumn := 31154, finalColumn := 404694 }
, { child := 4, logicalColumn := 158, sourceArmColumn := 31159, finalColumn := 404699 }
, { child := 4, logicalColumn := 159, sourceArmColumn := 31164, finalColumn := 404704 }
, { child := 4, logicalColumn := 160, sourceArmColumn := 31169, finalColumn := 404709 }
, { child := 4, logicalColumn := 161, sourceArmColumn := 31174, finalColumn := 404714 }
, { child := 4, logicalColumn := 162, sourceArmColumn := 30910, finalColumn := 404450 }
, { child := 4, logicalColumn := 163, sourceArmColumn := 30915, finalColumn := 404455 }
, { child := 4, logicalColumn := 164, sourceArmColumn := 30920, finalColumn := 404460 }
, { child := 4, logicalColumn := 165, sourceArmColumn := 30925, finalColumn := 404465 }
, { child := 4, logicalColumn := 166, sourceArmColumn := 30930, finalColumn := 404470 }
, { child := 4, logicalColumn := 167, sourceArmColumn := 30935, finalColumn := 404475 }
, { child := 4, logicalColumn := 168, sourceArmColumn := 30940, finalColumn := 404480 }
, { child := 4, logicalColumn := 169, sourceArmColumn := 30945, finalColumn := 404485 }
, { child := 4, logicalColumn := 170, sourceArmColumn := 30950, finalColumn := 404490 }
, { child := 4, logicalColumn := 171, sourceArmColumn := 30955, finalColumn := 404495 }
, { child := 4, logicalColumn := 172, sourceArmColumn := 30960, finalColumn := 404500 }
, { child := 4, logicalColumn := 173, sourceArmColumn := 30965, finalColumn := 404505 }
, { child := 4, logicalColumn := 174, sourceArmColumn := 30970, finalColumn := 404510 }
, { child := 4, logicalColumn := 175, sourceArmColumn := 30975, finalColumn := 404515 }
, { child := 4, logicalColumn := 176, sourceArmColumn := 30980, finalColumn := 404520 }
, { child := 4, logicalColumn := 177, sourceArmColumn := 30985, finalColumn := 404525 }
, { child := 4, logicalColumn := 178, sourceArmColumn := 30990, finalColumn := 404530 }
, { child := 4, logicalColumn := 179, sourceArmColumn := 30995, finalColumn := 404535 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk4
