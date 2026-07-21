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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk11

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 10, logicalColumn := 72, sourceArmColumn := 44630, finalColumn := 804678 }
, { child := 10, logicalColumn := 73, sourceArmColumn := 44635, finalColumn := 804683 }
, { child := 10, logicalColumn := 74, sourceArmColumn := 44640, finalColumn := 804688 }
, { child := 10, logicalColumn := 75, sourceArmColumn := 44645, finalColumn := 804693 }
, { child := 10, logicalColumn := 76, sourceArmColumn := 44650, finalColumn := 804698 }
, { child := 10, logicalColumn := 77, sourceArmColumn := 44655, finalColumn := 804703 }
, { child := 10, logicalColumn := 78, sourceArmColumn := 44660, finalColumn := 804708 }
, { child := 10, logicalColumn := 79, sourceArmColumn := 44665, finalColumn := 804713 }
, { child := 10, logicalColumn := 80, sourceArmColumn := 44670, finalColumn := 804718 }
, { child := 10, logicalColumn := 81, sourceArmColumn := 44675, finalColumn := 804723 }
, { child := 10, logicalColumn := 82, sourceArmColumn := 44680, finalColumn := 804728 }
, { child := 10, logicalColumn := 83, sourceArmColumn := 44685, finalColumn := 804733 }
, { child := 10, logicalColumn := 84, sourceArmColumn := 44690, finalColumn := 804738 }
, { child := 10, logicalColumn := 85, sourceArmColumn := 44695, finalColumn := 804743 }
, { child := 10, logicalColumn := 86, sourceArmColumn := 44700, finalColumn := 804748 }
, { child := 10, logicalColumn := 87, sourceArmColumn := 44705, finalColumn := 804753 }
, { child := 10, logicalColumn := 88, sourceArmColumn := 44710, finalColumn := 804758 }
, { child := 10, logicalColumn := 89, sourceArmColumn := 44715, finalColumn := 804763 }
, { child := 10, logicalColumn := 90, sourceArmColumn := 44720, finalColumn := 804768 }
, { child := 10, logicalColumn := 91, sourceArmColumn := 44725, finalColumn := 804773 }
, { child := 10, logicalColumn := 92, sourceArmColumn := 44730, finalColumn := 804778 }
, { child := 10, logicalColumn := 93, sourceArmColumn := 44735, finalColumn := 804783 }
, { child := 10, logicalColumn := 94, sourceArmColumn := 44740, finalColumn := 804788 }
, { child := 10, logicalColumn := 95, sourceArmColumn := 44745, finalColumn := 804793 }
, { child := 10, logicalColumn := 96, sourceArmColumn := 44750, finalColumn := 804798 }
, { child := 10, logicalColumn := 97, sourceArmColumn := 44755, finalColumn := 804803 }
, { child := 10, logicalColumn := 98, sourceArmColumn := 44760, finalColumn := 804808 }
, { child := 10, logicalColumn := 99, sourceArmColumn := 44765, finalColumn := 804813 }
, { child := 10, logicalColumn := 100, sourceArmColumn := 44770, finalColumn := 804818 }
, { child := 10, logicalColumn := 101, sourceArmColumn := 44775, finalColumn := 804823 }
, { child := 10, logicalColumn := 102, sourceArmColumn := 44780, finalColumn := 804828 }
, { child := 10, logicalColumn := 103, sourceArmColumn := 44785, finalColumn := 804833 }
, { child := 10, logicalColumn := 104, sourceArmColumn := 44790, finalColumn := 804838 }
, { child := 10, logicalColumn := 105, sourceArmColumn := 44795, finalColumn := 804843 }
, { child := 10, logicalColumn := 106, sourceArmColumn := 44800, finalColumn := 804848 }
, { child := 10, logicalColumn := 107, sourceArmColumn := 44805, finalColumn := 804853 }
, { child := 10, logicalColumn := 108, sourceArmColumn := 44541, finalColumn := 804589 }
, { child := 10, logicalColumn := 109, sourceArmColumn := 44546, finalColumn := 804594 }
, { child := 10, logicalColumn := 110, sourceArmColumn := 44551, finalColumn := 804599 }
, { child := 10, logicalColumn := 111, sourceArmColumn := 44556, finalColumn := 804604 }
, { child := 10, logicalColumn := 112, sourceArmColumn := 44561, finalColumn := 804609 }
, { child := 10, logicalColumn := 113, sourceArmColumn := 44566, finalColumn := 804614 }
, { child := 10, logicalColumn := 114, sourceArmColumn := 44571, finalColumn := 804619 }
, { child := 10, logicalColumn := 115, sourceArmColumn := 44576, finalColumn := 804624 }
, { child := 10, logicalColumn := 116, sourceArmColumn := 44581, finalColumn := 804629 }
, { child := 10, logicalColumn := 117, sourceArmColumn := 44586, finalColumn := 804634 }
, { child := 10, logicalColumn := 118, sourceArmColumn := 44591, finalColumn := 804639 }
, { child := 10, logicalColumn := 119, sourceArmColumn := 44596, finalColumn := 804644 }
, { child := 10, logicalColumn := 120, sourceArmColumn := 44601, finalColumn := 804649 }
, { child := 10, logicalColumn := 121, sourceArmColumn := 44606, finalColumn := 804654 }
, { child := 10, logicalColumn := 122, sourceArmColumn := 44611, finalColumn := 804659 }
, { child := 10, logicalColumn := 123, sourceArmColumn := 44616, finalColumn := 804664 }
, { child := 10, logicalColumn := 124, sourceArmColumn := 44621, finalColumn := 804669 }
, { child := 10, logicalColumn := 125, sourceArmColumn := 44626, finalColumn := 804674 }
, { child := 10, logicalColumn := 126, sourceArmColumn := 44631, finalColumn := 804679 }
, { child := 10, logicalColumn := 127, sourceArmColumn := 44636, finalColumn := 804684 }
, { child := 10, logicalColumn := 128, sourceArmColumn := 44641, finalColumn := 804689 }
, { child := 10, logicalColumn := 129, sourceArmColumn := 44646, finalColumn := 804694 }
, { child := 10, logicalColumn := 130, sourceArmColumn := 44651, finalColumn := 804699 }
, { child := 10, logicalColumn := 131, sourceArmColumn := 44656, finalColumn := 804704 }
, { child := 10, logicalColumn := 132, sourceArmColumn := 44661, finalColumn := 804709 }
, { child := 10, logicalColumn := 133, sourceArmColumn := 44666, finalColumn := 804714 }
, { child := 10, logicalColumn := 134, sourceArmColumn := 44671, finalColumn := 804719 }
, { child := 10, logicalColumn := 135, sourceArmColumn := 44676, finalColumn := 804724 }
, { child := 10, logicalColumn := 136, sourceArmColumn := 44681, finalColumn := 804729 }
, { child := 10, logicalColumn := 137, sourceArmColumn := 44686, finalColumn := 804734 }
, { child := 10, logicalColumn := 138, sourceArmColumn := 44691, finalColumn := 804739 }
, { child := 10, logicalColumn := 139, sourceArmColumn := 44696, finalColumn := 804744 }
, { child := 10, logicalColumn := 140, sourceArmColumn := 44701, finalColumn := 804749 }
, { child := 10, logicalColumn := 141, sourceArmColumn := 44706, finalColumn := 804754 }
, { child := 10, logicalColumn := 142, sourceArmColumn := 44711, finalColumn := 804759 }
, { child := 10, logicalColumn := 143, sourceArmColumn := 44716, finalColumn := 804764 }
, { child := 10, logicalColumn := 144, sourceArmColumn := 44721, finalColumn := 804769 }
, { child := 10, logicalColumn := 145, sourceArmColumn := 44726, finalColumn := 804774 }
, { child := 10, logicalColumn := 146, sourceArmColumn := 44731, finalColumn := 804779 }
, { child := 10, logicalColumn := 147, sourceArmColumn := 44736, finalColumn := 804784 }
, { child := 10, logicalColumn := 148, sourceArmColumn := 44741, finalColumn := 804789 }
, { child := 10, logicalColumn := 149, sourceArmColumn := 44746, finalColumn := 804794 }
, { child := 10, logicalColumn := 150, sourceArmColumn := 44751, finalColumn := 804799 }
, { child := 10, logicalColumn := 151, sourceArmColumn := 44756, finalColumn := 804804 }
, { child := 10, logicalColumn := 152, sourceArmColumn := 44761, finalColumn := 804809 }
, { child := 10, logicalColumn := 153, sourceArmColumn := 44766, finalColumn := 804814 }
, { child := 10, logicalColumn := 154, sourceArmColumn := 44771, finalColumn := 804819 }
, { child := 10, logicalColumn := 155, sourceArmColumn := 44776, finalColumn := 804824 }
, { child := 10, logicalColumn := 156, sourceArmColumn := 44781, finalColumn := 804829 }
, { child := 10, logicalColumn := 157, sourceArmColumn := 44786, finalColumn := 804834 }
, { child := 10, logicalColumn := 158, sourceArmColumn := 44791, finalColumn := 804839 }
, { child := 10, logicalColumn := 159, sourceArmColumn := 44796, finalColumn := 804844 }
, { child := 10, logicalColumn := 160, sourceArmColumn := 44801, finalColumn := 804849 }
, { child := 10, logicalColumn := 161, sourceArmColumn := 44806, finalColumn := 804854 }
, { child := 10, logicalColumn := 162, sourceArmColumn := 44542, finalColumn := 804590 }
, { child := 10, logicalColumn := 163, sourceArmColumn := 44547, finalColumn := 804595 }
, { child := 10, logicalColumn := 164, sourceArmColumn := 44552, finalColumn := 804600 }
, { child := 10, logicalColumn := 165, sourceArmColumn := 44557, finalColumn := 804605 }
, { child := 10, logicalColumn := 166, sourceArmColumn := 44562, finalColumn := 804610 }
, { child := 10, logicalColumn := 167, sourceArmColumn := 44567, finalColumn := 804615 }
, { child := 10, logicalColumn := 168, sourceArmColumn := 44572, finalColumn := 804620 }
, { child := 10, logicalColumn := 169, sourceArmColumn := 44577, finalColumn := 804625 }
, { child := 10, logicalColumn := 170, sourceArmColumn := 44582, finalColumn := 804630 }
, { child := 10, logicalColumn := 171, sourceArmColumn := 44587, finalColumn := 804635 }
, { child := 10, logicalColumn := 172, sourceArmColumn := 44592, finalColumn := 804640 }
, { child := 10, logicalColumn := 173, sourceArmColumn := 44597, finalColumn := 804645 }
, { child := 10, logicalColumn := 174, sourceArmColumn := 44602, finalColumn := 804650 }
, { child := 10, logicalColumn := 175, sourceArmColumn := 44607, finalColumn := 804655 }
, { child := 10, logicalColumn := 176, sourceArmColumn := 44612, finalColumn := 804660 }
, { child := 10, logicalColumn := 177, sourceArmColumn := 44617, finalColumn := 804665 }
, { child := 10, logicalColumn := 178, sourceArmColumn := 44622, finalColumn := 804670 }
, { child := 10, logicalColumn := 179, sourceArmColumn := 44627, finalColumn := 804675 }
, { child := 10, logicalColumn := 180, sourceArmColumn := 44632, finalColumn := 804680 }
, { child := 10, logicalColumn := 181, sourceArmColumn := 44637, finalColumn := 804685 }
, { child := 10, logicalColumn := 182, sourceArmColumn := 44642, finalColumn := 804690 }
, { child := 10, logicalColumn := 183, sourceArmColumn := 44647, finalColumn := 804695 }
, { child := 10, logicalColumn := 184, sourceArmColumn := 44652, finalColumn := 804700 }
, { child := 10, logicalColumn := 185, sourceArmColumn := 44657, finalColumn := 804705 }
, { child := 10, logicalColumn := 186, sourceArmColumn := 44662, finalColumn := 804710 }
, { child := 10, logicalColumn := 187, sourceArmColumn := 44667, finalColumn := 804715 }
, { child := 10, logicalColumn := 188, sourceArmColumn := 44672, finalColumn := 804720 }
, { child := 10, logicalColumn := 189, sourceArmColumn := 44677, finalColumn := 804725 }
, { child := 10, logicalColumn := 190, sourceArmColumn := 44682, finalColumn := 804730 }
, { child := 10, logicalColumn := 191, sourceArmColumn := 44687, finalColumn := 804735 }
, { child := 10, logicalColumn := 192, sourceArmColumn := 44692, finalColumn := 804740 }
, { child := 10, logicalColumn := 193, sourceArmColumn := 44697, finalColumn := 804745 }
, { child := 10, logicalColumn := 194, sourceArmColumn := 44702, finalColumn := 804750 }
, { child := 10, logicalColumn := 195, sourceArmColumn := 44707, finalColumn := 804755 }
, { child := 10, logicalColumn := 196, sourceArmColumn := 44712, finalColumn := 804760 }
, { child := 10, logicalColumn := 197, sourceArmColumn := 44717, finalColumn := 804765 }
, { child := 10, logicalColumn := 198, sourceArmColumn := 44722, finalColumn := 804770 }
, { child := 10, logicalColumn := 199, sourceArmColumn := 44727, finalColumn := 804775 }
, { child := 10, logicalColumn := 200, sourceArmColumn := 44732, finalColumn := 804780 }
, { child := 10, logicalColumn := 201, sourceArmColumn := 44737, finalColumn := 804785 }
, { child := 10, logicalColumn := 202, sourceArmColumn := 44742, finalColumn := 804790 }
, { child := 10, logicalColumn := 203, sourceArmColumn := 44747, finalColumn := 804795 }
, { child := 10, logicalColumn := 204, sourceArmColumn := 44752, finalColumn := 804800 }
, { child := 10, logicalColumn := 205, sourceArmColumn := 44757, finalColumn := 804805 }
, { child := 10, logicalColumn := 206, sourceArmColumn := 44762, finalColumn := 804810 }
, { child := 10, logicalColumn := 207, sourceArmColumn := 44767, finalColumn := 804815 }
, { child := 10, logicalColumn := 208, sourceArmColumn := 44772, finalColumn := 804820 }
, { child := 10, logicalColumn := 209, sourceArmColumn := 44777, finalColumn := 804825 }
, { child := 10, logicalColumn := 210, sourceArmColumn := 44782, finalColumn := 804830 }
, { child := 10, logicalColumn := 211, sourceArmColumn := 44787, finalColumn := 804835 }
, { child := 10, logicalColumn := 212, sourceArmColumn := 44792, finalColumn := 804840 }
, { child := 10, logicalColumn := 213, sourceArmColumn := 44797, finalColumn := 804845 }
, { child := 10, logicalColumn := 214, sourceArmColumn := 44802, finalColumn := 804850 }
, { child := 10, logicalColumn := 215, sourceArmColumn := 44807, finalColumn := 804855 }
, { child := 10, logicalColumn := 216, sourceArmColumn := 44543, finalColumn := 804591 }
, { child := 10, logicalColumn := 217, sourceArmColumn := 44548, finalColumn := 804596 }
, { child := 10, logicalColumn := 218, sourceArmColumn := 44553, finalColumn := 804601 }
, { child := 10, logicalColumn := 219, sourceArmColumn := 44558, finalColumn := 804606 }
, { child := 10, logicalColumn := 220, sourceArmColumn := 44563, finalColumn := 804611 }
, { child := 10, logicalColumn := 221, sourceArmColumn := 44568, finalColumn := 804616 }
, { child := 10, logicalColumn := 222, sourceArmColumn := 44573, finalColumn := 804621 }
, { child := 10, logicalColumn := 223, sourceArmColumn := 44578, finalColumn := 804626 }
, { child := 10, logicalColumn := 224, sourceArmColumn := 44583, finalColumn := 804631 }
, { child := 10, logicalColumn := 225, sourceArmColumn := 44588, finalColumn := 804636 }
, { child := 10, logicalColumn := 226, sourceArmColumn := 44593, finalColumn := 804641 }
, { child := 10, logicalColumn := 227, sourceArmColumn := 44598, finalColumn := 804646 }
, { child := 10, logicalColumn := 228, sourceArmColumn := 44603, finalColumn := 804651 }
, { child := 10, logicalColumn := 229, sourceArmColumn := 44608, finalColumn := 804656 }
, { child := 10, logicalColumn := 230, sourceArmColumn := 44613, finalColumn := 804661 }
, { child := 10, logicalColumn := 231, sourceArmColumn := 44618, finalColumn := 804666 }
, { child := 10, logicalColumn := 232, sourceArmColumn := 44623, finalColumn := 804671 }
, { child := 10, logicalColumn := 233, sourceArmColumn := 44628, finalColumn := 804676 }
, { child := 10, logicalColumn := 234, sourceArmColumn := 44633, finalColumn := 804681 }
, { child := 10, logicalColumn := 235, sourceArmColumn := 44638, finalColumn := 804686 }
, { child := 10, logicalColumn := 236, sourceArmColumn := 44643, finalColumn := 804691 }
, { child := 10, logicalColumn := 237, sourceArmColumn := 44648, finalColumn := 804696 }
, { child := 10, logicalColumn := 238, sourceArmColumn := 44653, finalColumn := 804701 }
, { child := 10, logicalColumn := 239, sourceArmColumn := 44658, finalColumn := 804706 }
, { child := 10, logicalColumn := 240, sourceArmColumn := 44663, finalColumn := 804711 }
, { child := 10, logicalColumn := 241, sourceArmColumn := 44668, finalColumn := 804716 }
, { child := 10, logicalColumn := 242, sourceArmColumn := 44673, finalColumn := 804721 }
, { child := 10, logicalColumn := 243, sourceArmColumn := 44678, finalColumn := 804726 }
, { child := 10, logicalColumn := 244, sourceArmColumn := 44683, finalColumn := 804731 }
, { child := 10, logicalColumn := 245, sourceArmColumn := 44688, finalColumn := 804736 }
, { child := 10, logicalColumn := 246, sourceArmColumn := 44693, finalColumn := 804741 }
, { child := 10, logicalColumn := 247, sourceArmColumn := 44698, finalColumn := 804746 }
, { child := 10, logicalColumn := 248, sourceArmColumn := 44703, finalColumn := 804751 }
, { child := 10, logicalColumn := 249, sourceArmColumn := 44708, finalColumn := 804756 }
, { child := 10, logicalColumn := 250, sourceArmColumn := 44713, finalColumn := 804761 }
, { child := 10, logicalColumn := 251, sourceArmColumn := 44718, finalColumn := 804766 }
, { child := 10, logicalColumn := 252, sourceArmColumn := 44723, finalColumn := 804771 }
, { child := 10, logicalColumn := 253, sourceArmColumn := 44728, finalColumn := 804776 }
, { child := 10, logicalColumn := 254, sourceArmColumn := 44733, finalColumn := 804781 }
, { child := 10, logicalColumn := 255, sourceArmColumn := 44738, finalColumn := 804786 }
, { child := 10, logicalColumn := 256, sourceArmColumn := 44743, finalColumn := 804791 }
, { child := 10, logicalColumn := 257, sourceArmColumn := 44748, finalColumn := 804796 }
, { child := 10, logicalColumn := 258, sourceArmColumn := 44753, finalColumn := 804801 }
, { child := 10, logicalColumn := 259, sourceArmColumn := 44758, finalColumn := 804806 }
, { child := 10, logicalColumn := 260, sourceArmColumn := 44763, finalColumn := 804811 }
, { child := 10, logicalColumn := 261, sourceArmColumn := 44768, finalColumn := 804816 }
, { child := 10, logicalColumn := 262, sourceArmColumn := 44773, finalColumn := 804821 }
, { child := 10, logicalColumn := 263, sourceArmColumn := 44778, finalColumn := 804826 }
, { child := 10, logicalColumn := 264, sourceArmColumn := 44783, finalColumn := 804831 }
, { child := 10, logicalColumn := 265, sourceArmColumn := 44788, finalColumn := 804836 }
, { child := 10, logicalColumn := 266, sourceArmColumn := 44793, finalColumn := 804841 }
, { child := 10, logicalColumn := 267, sourceArmColumn := 44798, finalColumn := 804846 }
, { child := 10, logicalColumn := 268, sourceArmColumn := 44803, finalColumn := 804851 }
, { child := 10, logicalColumn := 269, sourceArmColumn := 44808, finalColumn := 804856 }
, { child := 11, logicalColumn := 0, sourceArmColumn := 46811, finalColumn := 871277 }
, { child := 11, logicalColumn := 1, sourceArmColumn := 46816, finalColumn := 871282 }
, { child := 11, logicalColumn := 2, sourceArmColumn := 46821, finalColumn := 871287 }
, { child := 11, logicalColumn := 3, sourceArmColumn := 46826, finalColumn := 871292 }
, { child := 11, logicalColumn := 4, sourceArmColumn := 46831, finalColumn := 871297 }
, { child := 11, logicalColumn := 5, sourceArmColumn := 46836, finalColumn := 871302 }
, { child := 11, logicalColumn := 6, sourceArmColumn := 46841, finalColumn := 871307 }
, { child := 11, logicalColumn := 7, sourceArmColumn := 46846, finalColumn := 871312 }
, { child := 11, logicalColumn := 8, sourceArmColumn := 46851, finalColumn := 871317 }
, { child := 11, logicalColumn := 9, sourceArmColumn := 46856, finalColumn := 871322 }
, { child := 11, logicalColumn := 10, sourceArmColumn := 46861, finalColumn := 871327 }
, { child := 11, logicalColumn := 11, sourceArmColumn := 46866, finalColumn := 871332 }
, { child := 11, logicalColumn := 12, sourceArmColumn := 46871, finalColumn := 871337 }
, { child := 11, logicalColumn := 13, sourceArmColumn := 46876, finalColumn := 871342 }
, { child := 11, logicalColumn := 14, sourceArmColumn := 46881, finalColumn := 871347 }
, { child := 11, logicalColumn := 15, sourceArmColumn := 46886, finalColumn := 871352 }
, { child := 11, logicalColumn := 16, sourceArmColumn := 46891, finalColumn := 871357 }
, { child := 11, logicalColumn := 17, sourceArmColumn := 46896, finalColumn := 871362 }
, { child := 11, logicalColumn := 18, sourceArmColumn := 46901, finalColumn := 871367 }
, { child := 11, logicalColumn := 19, sourceArmColumn := 46906, finalColumn := 871372 }
, { child := 11, logicalColumn := 20, sourceArmColumn := 46911, finalColumn := 871377 }
, { child := 11, logicalColumn := 21, sourceArmColumn := 46916, finalColumn := 871382 }
, { child := 11, logicalColumn := 22, sourceArmColumn := 46921, finalColumn := 871387 }
, { child := 11, logicalColumn := 23, sourceArmColumn := 46926, finalColumn := 871392 }
, { child := 11, logicalColumn := 24, sourceArmColumn := 46931, finalColumn := 871397 }
, { child := 11, logicalColumn := 25, sourceArmColumn := 46936, finalColumn := 871402 }
, { child := 11, logicalColumn := 26, sourceArmColumn := 46941, finalColumn := 871407 }
, { child := 11, logicalColumn := 27, sourceArmColumn := 46946, finalColumn := 871412 }
, { child := 11, logicalColumn := 28, sourceArmColumn := 46951, finalColumn := 871417 }
, { child := 11, logicalColumn := 29, sourceArmColumn := 46956, finalColumn := 871422 }
, { child := 11, logicalColumn := 30, sourceArmColumn := 46961, finalColumn := 871427 }
, { child := 11, logicalColumn := 31, sourceArmColumn := 46966, finalColumn := 871432 }
, { child := 11, logicalColumn := 32, sourceArmColumn := 46971, finalColumn := 871437 }
, { child := 11, logicalColumn := 33, sourceArmColumn := 46976, finalColumn := 871442 }
, { child := 11, logicalColumn := 34, sourceArmColumn := 46981, finalColumn := 871447 }
, { child := 11, logicalColumn := 35, sourceArmColumn := 46986, finalColumn := 871452 }
, { child := 11, logicalColumn := 36, sourceArmColumn := 46991, finalColumn := 871457 }
, { child := 11, logicalColumn := 37, sourceArmColumn := 46996, finalColumn := 871462 }
, { child := 11, logicalColumn := 38, sourceArmColumn := 47001, finalColumn := 871467 }
, { child := 11, logicalColumn := 39, sourceArmColumn := 47006, finalColumn := 871472 }
, { child := 11, logicalColumn := 40, sourceArmColumn := 47011, finalColumn := 871477 }
, { child := 11, logicalColumn := 41, sourceArmColumn := 47016, finalColumn := 871482 }
, { child := 11, logicalColumn := 42, sourceArmColumn := 47021, finalColumn := 871487 }
, { child := 11, logicalColumn := 43, sourceArmColumn := 47026, finalColumn := 871492 }
, { child := 11, logicalColumn := 44, sourceArmColumn := 47031, finalColumn := 871497 }
, { child := 11, logicalColumn := 45, sourceArmColumn := 47036, finalColumn := 871502 }
, { child := 11, logicalColumn := 46, sourceArmColumn := 47041, finalColumn := 871507 }
, { child := 11, logicalColumn := 47, sourceArmColumn := 47046, finalColumn := 871512 }
, { child := 11, logicalColumn := 48, sourceArmColumn := 47051, finalColumn := 871517 }
, { child := 11, logicalColumn := 49, sourceArmColumn := 47056, finalColumn := 871522 }
, { child := 11, logicalColumn := 50, sourceArmColumn := 47061, finalColumn := 871527 }
, { child := 11, logicalColumn := 51, sourceArmColumn := 47066, finalColumn := 871532 }
, { child := 11, logicalColumn := 52, sourceArmColumn := 47071, finalColumn := 871537 }
, { child := 11, logicalColumn := 53, sourceArmColumn := 47076, finalColumn := 871542 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk11
