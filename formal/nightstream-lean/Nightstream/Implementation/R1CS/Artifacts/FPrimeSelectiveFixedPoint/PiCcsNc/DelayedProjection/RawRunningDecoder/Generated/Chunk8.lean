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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk8

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 7, logicalColumn := 126, sourceArmColumn := 37815, finalColumn := 604609 }
, { child := 7, logicalColumn := 127, sourceArmColumn := 37820, finalColumn := 604614 }
, { child := 7, logicalColumn := 128, sourceArmColumn := 37825, finalColumn := 604619 }
, { child := 7, logicalColumn := 129, sourceArmColumn := 37830, finalColumn := 604624 }
, { child := 7, logicalColumn := 130, sourceArmColumn := 37835, finalColumn := 604629 }
, { child := 7, logicalColumn := 131, sourceArmColumn := 37840, finalColumn := 604634 }
, { child := 7, logicalColumn := 132, sourceArmColumn := 37845, finalColumn := 604639 }
, { child := 7, logicalColumn := 133, sourceArmColumn := 37850, finalColumn := 604644 }
, { child := 7, logicalColumn := 134, sourceArmColumn := 37855, finalColumn := 604649 }
, { child := 7, logicalColumn := 135, sourceArmColumn := 37860, finalColumn := 604654 }
, { child := 7, logicalColumn := 136, sourceArmColumn := 37865, finalColumn := 604659 }
, { child := 7, logicalColumn := 137, sourceArmColumn := 37870, finalColumn := 604664 }
, { child := 7, logicalColumn := 138, sourceArmColumn := 37875, finalColumn := 604669 }
, { child := 7, logicalColumn := 139, sourceArmColumn := 37880, finalColumn := 604674 }
, { child := 7, logicalColumn := 140, sourceArmColumn := 37885, finalColumn := 604679 }
, { child := 7, logicalColumn := 141, sourceArmColumn := 37890, finalColumn := 604684 }
, { child := 7, logicalColumn := 142, sourceArmColumn := 37895, finalColumn := 604689 }
, { child := 7, logicalColumn := 143, sourceArmColumn := 37900, finalColumn := 604694 }
, { child := 7, logicalColumn := 144, sourceArmColumn := 37905, finalColumn := 604699 }
, { child := 7, logicalColumn := 145, sourceArmColumn := 37910, finalColumn := 604704 }
, { child := 7, logicalColumn := 146, sourceArmColumn := 37915, finalColumn := 604709 }
, { child := 7, logicalColumn := 147, sourceArmColumn := 37920, finalColumn := 604714 }
, { child := 7, logicalColumn := 148, sourceArmColumn := 37925, finalColumn := 604719 }
, { child := 7, logicalColumn := 149, sourceArmColumn := 37930, finalColumn := 604724 }
, { child := 7, logicalColumn := 150, sourceArmColumn := 37935, finalColumn := 604729 }
, { child := 7, logicalColumn := 151, sourceArmColumn := 37940, finalColumn := 604734 }
, { child := 7, logicalColumn := 152, sourceArmColumn := 37945, finalColumn := 604739 }
, { child := 7, logicalColumn := 153, sourceArmColumn := 37950, finalColumn := 604744 }
, { child := 7, logicalColumn := 154, sourceArmColumn := 37955, finalColumn := 604749 }
, { child := 7, logicalColumn := 155, sourceArmColumn := 37960, finalColumn := 604754 }
, { child := 7, logicalColumn := 156, sourceArmColumn := 37965, finalColumn := 604759 }
, { child := 7, logicalColumn := 157, sourceArmColumn := 37970, finalColumn := 604764 }
, { child := 7, logicalColumn := 158, sourceArmColumn := 37975, finalColumn := 604769 }
, { child := 7, logicalColumn := 159, sourceArmColumn := 37980, finalColumn := 604774 }
, { child := 7, logicalColumn := 160, sourceArmColumn := 37985, finalColumn := 604779 }
, { child := 7, logicalColumn := 161, sourceArmColumn := 37990, finalColumn := 604784 }
, { child := 7, logicalColumn := 162, sourceArmColumn := 37726, finalColumn := 604520 }
, { child := 7, logicalColumn := 163, sourceArmColumn := 37731, finalColumn := 604525 }
, { child := 7, logicalColumn := 164, sourceArmColumn := 37736, finalColumn := 604530 }
, { child := 7, logicalColumn := 165, sourceArmColumn := 37741, finalColumn := 604535 }
, { child := 7, logicalColumn := 166, sourceArmColumn := 37746, finalColumn := 604540 }
, { child := 7, logicalColumn := 167, sourceArmColumn := 37751, finalColumn := 604545 }
, { child := 7, logicalColumn := 168, sourceArmColumn := 37756, finalColumn := 604550 }
, { child := 7, logicalColumn := 169, sourceArmColumn := 37761, finalColumn := 604555 }
, { child := 7, logicalColumn := 170, sourceArmColumn := 37766, finalColumn := 604560 }
, { child := 7, logicalColumn := 171, sourceArmColumn := 37771, finalColumn := 604565 }
, { child := 7, logicalColumn := 172, sourceArmColumn := 37776, finalColumn := 604570 }
, { child := 7, logicalColumn := 173, sourceArmColumn := 37781, finalColumn := 604575 }
, { child := 7, logicalColumn := 174, sourceArmColumn := 37786, finalColumn := 604580 }
, { child := 7, logicalColumn := 175, sourceArmColumn := 37791, finalColumn := 604585 }
, { child := 7, logicalColumn := 176, sourceArmColumn := 37796, finalColumn := 604590 }
, { child := 7, logicalColumn := 177, sourceArmColumn := 37801, finalColumn := 604595 }
, { child := 7, logicalColumn := 178, sourceArmColumn := 37806, finalColumn := 604600 }
, { child := 7, logicalColumn := 179, sourceArmColumn := 37811, finalColumn := 604605 }
, { child := 7, logicalColumn := 180, sourceArmColumn := 37816, finalColumn := 604610 }
, { child := 7, logicalColumn := 181, sourceArmColumn := 37821, finalColumn := 604615 }
, { child := 7, logicalColumn := 182, sourceArmColumn := 37826, finalColumn := 604620 }
, { child := 7, logicalColumn := 183, sourceArmColumn := 37831, finalColumn := 604625 }
, { child := 7, logicalColumn := 184, sourceArmColumn := 37836, finalColumn := 604630 }
, { child := 7, logicalColumn := 185, sourceArmColumn := 37841, finalColumn := 604635 }
, { child := 7, logicalColumn := 186, sourceArmColumn := 37846, finalColumn := 604640 }
, { child := 7, logicalColumn := 187, sourceArmColumn := 37851, finalColumn := 604645 }
, { child := 7, logicalColumn := 188, sourceArmColumn := 37856, finalColumn := 604650 }
, { child := 7, logicalColumn := 189, sourceArmColumn := 37861, finalColumn := 604655 }
, { child := 7, logicalColumn := 190, sourceArmColumn := 37866, finalColumn := 604660 }
, { child := 7, logicalColumn := 191, sourceArmColumn := 37871, finalColumn := 604665 }
, { child := 7, logicalColumn := 192, sourceArmColumn := 37876, finalColumn := 604670 }
, { child := 7, logicalColumn := 193, sourceArmColumn := 37881, finalColumn := 604675 }
, { child := 7, logicalColumn := 194, sourceArmColumn := 37886, finalColumn := 604680 }
, { child := 7, logicalColumn := 195, sourceArmColumn := 37891, finalColumn := 604685 }
, { child := 7, logicalColumn := 196, sourceArmColumn := 37896, finalColumn := 604690 }
, { child := 7, logicalColumn := 197, sourceArmColumn := 37901, finalColumn := 604695 }
, { child := 7, logicalColumn := 198, sourceArmColumn := 37906, finalColumn := 604700 }
, { child := 7, logicalColumn := 199, sourceArmColumn := 37911, finalColumn := 604705 }
, { child := 7, logicalColumn := 200, sourceArmColumn := 37916, finalColumn := 604710 }
, { child := 7, logicalColumn := 201, sourceArmColumn := 37921, finalColumn := 604715 }
, { child := 7, logicalColumn := 202, sourceArmColumn := 37926, finalColumn := 604720 }
, { child := 7, logicalColumn := 203, sourceArmColumn := 37931, finalColumn := 604725 }
, { child := 7, logicalColumn := 204, sourceArmColumn := 37936, finalColumn := 604730 }
, { child := 7, logicalColumn := 205, sourceArmColumn := 37941, finalColumn := 604735 }
, { child := 7, logicalColumn := 206, sourceArmColumn := 37946, finalColumn := 604740 }
, { child := 7, logicalColumn := 207, sourceArmColumn := 37951, finalColumn := 604745 }
, { child := 7, logicalColumn := 208, sourceArmColumn := 37956, finalColumn := 604750 }
, { child := 7, logicalColumn := 209, sourceArmColumn := 37961, finalColumn := 604755 }
, { child := 7, logicalColumn := 210, sourceArmColumn := 37966, finalColumn := 604760 }
, { child := 7, logicalColumn := 211, sourceArmColumn := 37971, finalColumn := 604765 }
, { child := 7, logicalColumn := 212, sourceArmColumn := 37976, finalColumn := 604770 }
, { child := 7, logicalColumn := 213, sourceArmColumn := 37981, finalColumn := 604775 }
, { child := 7, logicalColumn := 214, sourceArmColumn := 37986, finalColumn := 604780 }
, { child := 7, logicalColumn := 215, sourceArmColumn := 37991, finalColumn := 604785 }
, { child := 7, logicalColumn := 216, sourceArmColumn := 37727, finalColumn := 604521 }
, { child := 7, logicalColumn := 217, sourceArmColumn := 37732, finalColumn := 604526 }
, { child := 7, logicalColumn := 218, sourceArmColumn := 37737, finalColumn := 604531 }
, { child := 7, logicalColumn := 219, sourceArmColumn := 37742, finalColumn := 604536 }
, { child := 7, logicalColumn := 220, sourceArmColumn := 37747, finalColumn := 604541 }
, { child := 7, logicalColumn := 221, sourceArmColumn := 37752, finalColumn := 604546 }
, { child := 7, logicalColumn := 222, sourceArmColumn := 37757, finalColumn := 604551 }
, { child := 7, logicalColumn := 223, sourceArmColumn := 37762, finalColumn := 604556 }
, { child := 7, logicalColumn := 224, sourceArmColumn := 37767, finalColumn := 604561 }
, { child := 7, logicalColumn := 225, sourceArmColumn := 37772, finalColumn := 604566 }
, { child := 7, logicalColumn := 226, sourceArmColumn := 37777, finalColumn := 604571 }
, { child := 7, logicalColumn := 227, sourceArmColumn := 37782, finalColumn := 604576 }
, { child := 7, logicalColumn := 228, sourceArmColumn := 37787, finalColumn := 604581 }
, { child := 7, logicalColumn := 229, sourceArmColumn := 37792, finalColumn := 604586 }
, { child := 7, logicalColumn := 230, sourceArmColumn := 37797, finalColumn := 604591 }
, { child := 7, logicalColumn := 231, sourceArmColumn := 37802, finalColumn := 604596 }
, { child := 7, logicalColumn := 232, sourceArmColumn := 37807, finalColumn := 604601 }
, { child := 7, logicalColumn := 233, sourceArmColumn := 37812, finalColumn := 604606 }
, { child := 7, logicalColumn := 234, sourceArmColumn := 37817, finalColumn := 604611 }
, { child := 7, logicalColumn := 235, sourceArmColumn := 37822, finalColumn := 604616 }
, { child := 7, logicalColumn := 236, sourceArmColumn := 37827, finalColumn := 604621 }
, { child := 7, logicalColumn := 237, sourceArmColumn := 37832, finalColumn := 604626 }
, { child := 7, logicalColumn := 238, sourceArmColumn := 37837, finalColumn := 604631 }
, { child := 7, logicalColumn := 239, sourceArmColumn := 37842, finalColumn := 604636 }
, { child := 7, logicalColumn := 240, sourceArmColumn := 37847, finalColumn := 604641 }
, { child := 7, logicalColumn := 241, sourceArmColumn := 37852, finalColumn := 604646 }
, { child := 7, logicalColumn := 242, sourceArmColumn := 37857, finalColumn := 604651 }
, { child := 7, logicalColumn := 243, sourceArmColumn := 37862, finalColumn := 604656 }
, { child := 7, logicalColumn := 244, sourceArmColumn := 37867, finalColumn := 604661 }
, { child := 7, logicalColumn := 245, sourceArmColumn := 37872, finalColumn := 604666 }
, { child := 7, logicalColumn := 246, sourceArmColumn := 37877, finalColumn := 604671 }
, { child := 7, logicalColumn := 247, sourceArmColumn := 37882, finalColumn := 604676 }
, { child := 7, logicalColumn := 248, sourceArmColumn := 37887, finalColumn := 604681 }
, { child := 7, logicalColumn := 249, sourceArmColumn := 37892, finalColumn := 604686 }
, { child := 7, logicalColumn := 250, sourceArmColumn := 37897, finalColumn := 604691 }
, { child := 7, logicalColumn := 251, sourceArmColumn := 37902, finalColumn := 604696 }
, { child := 7, logicalColumn := 252, sourceArmColumn := 37907, finalColumn := 604701 }
, { child := 7, logicalColumn := 253, sourceArmColumn := 37912, finalColumn := 604706 }
, { child := 7, logicalColumn := 254, sourceArmColumn := 37917, finalColumn := 604711 }
, { child := 7, logicalColumn := 255, sourceArmColumn := 37922, finalColumn := 604716 }
, { child := 7, logicalColumn := 256, sourceArmColumn := 37927, finalColumn := 604721 }
, { child := 7, logicalColumn := 257, sourceArmColumn := 37932, finalColumn := 604726 }
, { child := 7, logicalColumn := 258, sourceArmColumn := 37937, finalColumn := 604731 }
, { child := 7, logicalColumn := 259, sourceArmColumn := 37942, finalColumn := 604736 }
, { child := 7, logicalColumn := 260, sourceArmColumn := 37947, finalColumn := 604741 }
, { child := 7, logicalColumn := 261, sourceArmColumn := 37952, finalColumn := 604746 }
, { child := 7, logicalColumn := 262, sourceArmColumn := 37957, finalColumn := 604751 }
, { child := 7, logicalColumn := 263, sourceArmColumn := 37962, finalColumn := 604756 }
, { child := 7, logicalColumn := 264, sourceArmColumn := 37967, finalColumn := 604761 }
, { child := 7, logicalColumn := 265, sourceArmColumn := 37972, finalColumn := 604766 }
, { child := 7, logicalColumn := 266, sourceArmColumn := 37977, finalColumn := 604771 }
, { child := 7, logicalColumn := 267, sourceArmColumn := 37982, finalColumn := 604776 }
, { child := 7, logicalColumn := 268, sourceArmColumn := 37987, finalColumn := 604781 }
, { child := 7, logicalColumn := 269, sourceArmColumn := 37992, finalColumn := 604786 }
, { child := 8, logicalColumn := 0, sourceArmColumn := 39995, finalColumn := 671207 }
, { child := 8, logicalColumn := 1, sourceArmColumn := 40000, finalColumn := 671212 }
, { child := 8, logicalColumn := 2, sourceArmColumn := 40005, finalColumn := 671217 }
, { child := 8, logicalColumn := 3, sourceArmColumn := 40010, finalColumn := 671222 }
, { child := 8, logicalColumn := 4, sourceArmColumn := 40015, finalColumn := 671227 }
, { child := 8, logicalColumn := 5, sourceArmColumn := 40020, finalColumn := 671232 }
, { child := 8, logicalColumn := 6, sourceArmColumn := 40025, finalColumn := 671237 }
, { child := 8, logicalColumn := 7, sourceArmColumn := 40030, finalColumn := 671242 }
, { child := 8, logicalColumn := 8, sourceArmColumn := 40035, finalColumn := 671247 }
, { child := 8, logicalColumn := 9, sourceArmColumn := 40040, finalColumn := 671252 }
, { child := 8, logicalColumn := 10, sourceArmColumn := 40045, finalColumn := 671257 }
, { child := 8, logicalColumn := 11, sourceArmColumn := 40050, finalColumn := 671262 }
, { child := 8, logicalColumn := 12, sourceArmColumn := 40055, finalColumn := 671267 }
, { child := 8, logicalColumn := 13, sourceArmColumn := 40060, finalColumn := 671272 }
, { child := 8, logicalColumn := 14, sourceArmColumn := 40065, finalColumn := 671277 }
, { child := 8, logicalColumn := 15, sourceArmColumn := 40070, finalColumn := 671282 }
, { child := 8, logicalColumn := 16, sourceArmColumn := 40075, finalColumn := 671287 }
, { child := 8, logicalColumn := 17, sourceArmColumn := 40080, finalColumn := 671292 }
, { child := 8, logicalColumn := 18, sourceArmColumn := 40085, finalColumn := 671297 }
, { child := 8, logicalColumn := 19, sourceArmColumn := 40090, finalColumn := 671302 }
, { child := 8, logicalColumn := 20, sourceArmColumn := 40095, finalColumn := 671307 }
, { child := 8, logicalColumn := 21, sourceArmColumn := 40100, finalColumn := 671312 }
, { child := 8, logicalColumn := 22, sourceArmColumn := 40105, finalColumn := 671317 }
, { child := 8, logicalColumn := 23, sourceArmColumn := 40110, finalColumn := 671322 }
, { child := 8, logicalColumn := 24, sourceArmColumn := 40115, finalColumn := 671327 }
, { child := 8, logicalColumn := 25, sourceArmColumn := 40120, finalColumn := 671332 }
, { child := 8, logicalColumn := 26, sourceArmColumn := 40125, finalColumn := 671337 }
, { child := 8, logicalColumn := 27, sourceArmColumn := 40130, finalColumn := 671342 }
, { child := 8, logicalColumn := 28, sourceArmColumn := 40135, finalColumn := 671347 }
, { child := 8, logicalColumn := 29, sourceArmColumn := 40140, finalColumn := 671352 }
, { child := 8, logicalColumn := 30, sourceArmColumn := 40145, finalColumn := 671357 }
, { child := 8, logicalColumn := 31, sourceArmColumn := 40150, finalColumn := 671362 }
, { child := 8, logicalColumn := 32, sourceArmColumn := 40155, finalColumn := 671367 }
, { child := 8, logicalColumn := 33, sourceArmColumn := 40160, finalColumn := 671372 }
, { child := 8, logicalColumn := 34, sourceArmColumn := 40165, finalColumn := 671377 }
, { child := 8, logicalColumn := 35, sourceArmColumn := 40170, finalColumn := 671382 }
, { child := 8, logicalColumn := 36, sourceArmColumn := 40175, finalColumn := 671387 }
, { child := 8, logicalColumn := 37, sourceArmColumn := 40180, finalColumn := 671392 }
, { child := 8, logicalColumn := 38, sourceArmColumn := 40185, finalColumn := 671397 }
, { child := 8, logicalColumn := 39, sourceArmColumn := 40190, finalColumn := 671402 }
, { child := 8, logicalColumn := 40, sourceArmColumn := 40195, finalColumn := 671407 }
, { child := 8, logicalColumn := 41, sourceArmColumn := 40200, finalColumn := 671412 }
, { child := 8, logicalColumn := 42, sourceArmColumn := 40205, finalColumn := 671417 }
, { child := 8, logicalColumn := 43, sourceArmColumn := 40210, finalColumn := 671422 }
, { child := 8, logicalColumn := 44, sourceArmColumn := 40215, finalColumn := 671427 }
, { child := 8, logicalColumn := 45, sourceArmColumn := 40220, finalColumn := 671432 }
, { child := 8, logicalColumn := 46, sourceArmColumn := 40225, finalColumn := 671437 }
, { child := 8, logicalColumn := 47, sourceArmColumn := 40230, finalColumn := 671442 }
, { child := 8, logicalColumn := 48, sourceArmColumn := 40235, finalColumn := 671447 }
, { child := 8, logicalColumn := 49, sourceArmColumn := 40240, finalColumn := 671452 }
, { child := 8, logicalColumn := 50, sourceArmColumn := 40245, finalColumn := 671457 }
, { child := 8, logicalColumn := 51, sourceArmColumn := 40250, finalColumn := 671462 }
, { child := 8, logicalColumn := 52, sourceArmColumn := 40255, finalColumn := 671467 }
, { child := 8, logicalColumn := 53, sourceArmColumn := 40260, finalColumn := 671472 }
, { child := 8, logicalColumn := 54, sourceArmColumn := 39996, finalColumn := 671208 }
, { child := 8, logicalColumn := 55, sourceArmColumn := 40001, finalColumn := 671213 }
, { child := 8, logicalColumn := 56, sourceArmColumn := 40006, finalColumn := 671218 }
, { child := 8, logicalColumn := 57, sourceArmColumn := 40011, finalColumn := 671223 }
, { child := 8, logicalColumn := 58, sourceArmColumn := 40016, finalColumn := 671228 }
, { child := 8, logicalColumn := 59, sourceArmColumn := 40021, finalColumn := 671233 }
, { child := 8, logicalColumn := 60, sourceArmColumn := 40026, finalColumn := 671238 }
, { child := 8, logicalColumn := 61, sourceArmColumn := 40031, finalColumn := 671243 }
, { child := 8, logicalColumn := 62, sourceArmColumn := 40036, finalColumn := 671248 }
, { child := 8, logicalColumn := 63, sourceArmColumn := 40041, finalColumn := 671253 }
, { child := 8, logicalColumn := 64, sourceArmColumn := 40046, finalColumn := 671258 }
, { child := 8, logicalColumn := 65, sourceArmColumn := 40051, finalColumn := 671263 }
, { child := 8, logicalColumn := 66, sourceArmColumn := 40056, finalColumn := 671268 }
, { child := 8, logicalColumn := 67, sourceArmColumn := 40061, finalColumn := 671273 }
, { child := 8, logicalColumn := 68, sourceArmColumn := 40066, finalColumn := 671278 }
, { child := 8, logicalColumn := 69, sourceArmColumn := 40071, finalColumn := 671283 }
, { child := 8, logicalColumn := 70, sourceArmColumn := 40076, finalColumn := 671288 }
, { child := 8, logicalColumn := 71, sourceArmColumn := 40081, finalColumn := 671293 }
, { child := 8, logicalColumn := 72, sourceArmColumn := 40086, finalColumn := 671298 }
, { child := 8, logicalColumn := 73, sourceArmColumn := 40091, finalColumn := 671303 }
, { child := 8, logicalColumn := 74, sourceArmColumn := 40096, finalColumn := 671308 }
, { child := 8, logicalColumn := 75, sourceArmColumn := 40101, finalColumn := 671313 }
, { child := 8, logicalColumn := 76, sourceArmColumn := 40106, finalColumn := 671318 }
, { child := 8, logicalColumn := 77, sourceArmColumn := 40111, finalColumn := 671323 }
, { child := 8, logicalColumn := 78, sourceArmColumn := 40116, finalColumn := 671328 }
, { child := 8, logicalColumn := 79, sourceArmColumn := 40121, finalColumn := 671333 }
, { child := 8, logicalColumn := 80, sourceArmColumn := 40126, finalColumn := 671338 }
, { child := 8, logicalColumn := 81, sourceArmColumn := 40131, finalColumn := 671343 }
, { child := 8, logicalColumn := 82, sourceArmColumn := 40136, finalColumn := 671348 }
, { child := 8, logicalColumn := 83, sourceArmColumn := 40141, finalColumn := 671353 }
, { child := 8, logicalColumn := 84, sourceArmColumn := 40146, finalColumn := 671358 }
, { child := 8, logicalColumn := 85, sourceArmColumn := 40151, finalColumn := 671363 }
, { child := 8, logicalColumn := 86, sourceArmColumn := 40156, finalColumn := 671368 }
, { child := 8, logicalColumn := 87, sourceArmColumn := 40161, finalColumn := 671373 }
, { child := 8, logicalColumn := 88, sourceArmColumn := 40166, finalColumn := 671378 }
, { child := 8, logicalColumn := 89, sourceArmColumn := 40171, finalColumn := 671383 }
, { child := 8, logicalColumn := 90, sourceArmColumn := 40176, finalColumn := 671388 }
, { child := 8, logicalColumn := 91, sourceArmColumn := 40181, finalColumn := 671393 }
, { child := 8, logicalColumn := 92, sourceArmColumn := 40186, finalColumn := 671398 }
, { child := 8, logicalColumn := 93, sourceArmColumn := 40191, finalColumn := 671403 }
, { child := 8, logicalColumn := 94, sourceArmColumn := 40196, finalColumn := 671408 }
, { child := 8, logicalColumn := 95, sourceArmColumn := 40201, finalColumn := 671413 }
, { child := 8, logicalColumn := 96, sourceArmColumn := 40206, finalColumn := 671418 }
, { child := 8, logicalColumn := 97, sourceArmColumn := 40211, finalColumn := 671423 }
, { child := 8, logicalColumn := 98, sourceArmColumn := 40216, finalColumn := 671428 }
, { child := 8, logicalColumn := 99, sourceArmColumn := 40221, finalColumn := 671433 }
, { child := 8, logicalColumn := 100, sourceArmColumn := 40226, finalColumn := 671438 }
, { child := 8, logicalColumn := 101, sourceArmColumn := 40231, finalColumn := 671443 }
, { child := 8, logicalColumn := 102, sourceArmColumn := 40236, finalColumn := 671448 }
, { child := 8, logicalColumn := 103, sourceArmColumn := 40241, finalColumn := 671453 }
, { child := 8, logicalColumn := 104, sourceArmColumn := 40246, finalColumn := 671458 }
, { child := 8, logicalColumn := 105, sourceArmColumn := 40251, finalColumn := 671463 }
, { child := 8, logicalColumn := 106, sourceArmColumn := 40256, finalColumn := 671468 }
, { child := 8, logicalColumn := 107, sourceArmColumn := 40261, finalColumn := 671473 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk8
