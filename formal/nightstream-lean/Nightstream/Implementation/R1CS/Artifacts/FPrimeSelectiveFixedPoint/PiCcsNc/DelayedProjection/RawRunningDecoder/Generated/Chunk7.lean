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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk7

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 6, logicalColumn := 144, sourceArmColumn := 35633, finalColumn := 538009 }
, { child := 6, logicalColumn := 145, sourceArmColumn := 35638, finalColumn := 538014 }
, { child := 6, logicalColumn := 146, sourceArmColumn := 35643, finalColumn := 538019 }
, { child := 6, logicalColumn := 147, sourceArmColumn := 35648, finalColumn := 538024 }
, { child := 6, logicalColumn := 148, sourceArmColumn := 35653, finalColumn := 538029 }
, { child := 6, logicalColumn := 149, sourceArmColumn := 35658, finalColumn := 538034 }
, { child := 6, logicalColumn := 150, sourceArmColumn := 35663, finalColumn := 538039 }
, { child := 6, logicalColumn := 151, sourceArmColumn := 35668, finalColumn := 538044 }
, { child := 6, logicalColumn := 152, sourceArmColumn := 35673, finalColumn := 538049 }
, { child := 6, logicalColumn := 153, sourceArmColumn := 35678, finalColumn := 538054 }
, { child := 6, logicalColumn := 154, sourceArmColumn := 35683, finalColumn := 538059 }
, { child := 6, logicalColumn := 155, sourceArmColumn := 35688, finalColumn := 538064 }
, { child := 6, logicalColumn := 156, sourceArmColumn := 35693, finalColumn := 538069 }
, { child := 6, logicalColumn := 157, sourceArmColumn := 35698, finalColumn := 538074 }
, { child := 6, logicalColumn := 158, sourceArmColumn := 35703, finalColumn := 538079 }
, { child := 6, logicalColumn := 159, sourceArmColumn := 35708, finalColumn := 538084 }
, { child := 6, logicalColumn := 160, sourceArmColumn := 35713, finalColumn := 538089 }
, { child := 6, logicalColumn := 161, sourceArmColumn := 35718, finalColumn := 538094 }
, { child := 6, logicalColumn := 162, sourceArmColumn := 35454, finalColumn := 537830 }
, { child := 6, logicalColumn := 163, sourceArmColumn := 35459, finalColumn := 537835 }
, { child := 6, logicalColumn := 164, sourceArmColumn := 35464, finalColumn := 537840 }
, { child := 6, logicalColumn := 165, sourceArmColumn := 35469, finalColumn := 537845 }
, { child := 6, logicalColumn := 166, sourceArmColumn := 35474, finalColumn := 537850 }
, { child := 6, logicalColumn := 167, sourceArmColumn := 35479, finalColumn := 537855 }
, { child := 6, logicalColumn := 168, sourceArmColumn := 35484, finalColumn := 537860 }
, { child := 6, logicalColumn := 169, sourceArmColumn := 35489, finalColumn := 537865 }
, { child := 6, logicalColumn := 170, sourceArmColumn := 35494, finalColumn := 537870 }
, { child := 6, logicalColumn := 171, sourceArmColumn := 35499, finalColumn := 537875 }
, { child := 6, logicalColumn := 172, sourceArmColumn := 35504, finalColumn := 537880 }
, { child := 6, logicalColumn := 173, sourceArmColumn := 35509, finalColumn := 537885 }
, { child := 6, logicalColumn := 174, sourceArmColumn := 35514, finalColumn := 537890 }
, { child := 6, logicalColumn := 175, sourceArmColumn := 35519, finalColumn := 537895 }
, { child := 6, logicalColumn := 176, sourceArmColumn := 35524, finalColumn := 537900 }
, { child := 6, logicalColumn := 177, sourceArmColumn := 35529, finalColumn := 537905 }
, { child := 6, logicalColumn := 178, sourceArmColumn := 35534, finalColumn := 537910 }
, { child := 6, logicalColumn := 179, sourceArmColumn := 35539, finalColumn := 537915 }
, { child := 6, logicalColumn := 180, sourceArmColumn := 35544, finalColumn := 537920 }
, { child := 6, logicalColumn := 181, sourceArmColumn := 35549, finalColumn := 537925 }
, { child := 6, logicalColumn := 182, sourceArmColumn := 35554, finalColumn := 537930 }
, { child := 6, logicalColumn := 183, sourceArmColumn := 35559, finalColumn := 537935 }
, { child := 6, logicalColumn := 184, sourceArmColumn := 35564, finalColumn := 537940 }
, { child := 6, logicalColumn := 185, sourceArmColumn := 35569, finalColumn := 537945 }
, { child := 6, logicalColumn := 186, sourceArmColumn := 35574, finalColumn := 537950 }
, { child := 6, logicalColumn := 187, sourceArmColumn := 35579, finalColumn := 537955 }
, { child := 6, logicalColumn := 188, sourceArmColumn := 35584, finalColumn := 537960 }
, { child := 6, logicalColumn := 189, sourceArmColumn := 35589, finalColumn := 537965 }
, { child := 6, logicalColumn := 190, sourceArmColumn := 35594, finalColumn := 537970 }
, { child := 6, logicalColumn := 191, sourceArmColumn := 35599, finalColumn := 537975 }
, { child := 6, logicalColumn := 192, sourceArmColumn := 35604, finalColumn := 537980 }
, { child := 6, logicalColumn := 193, sourceArmColumn := 35609, finalColumn := 537985 }
, { child := 6, logicalColumn := 194, sourceArmColumn := 35614, finalColumn := 537990 }
, { child := 6, logicalColumn := 195, sourceArmColumn := 35619, finalColumn := 537995 }
, { child := 6, logicalColumn := 196, sourceArmColumn := 35624, finalColumn := 538000 }
, { child := 6, logicalColumn := 197, sourceArmColumn := 35629, finalColumn := 538005 }
, { child := 6, logicalColumn := 198, sourceArmColumn := 35634, finalColumn := 538010 }
, { child := 6, logicalColumn := 199, sourceArmColumn := 35639, finalColumn := 538015 }
, { child := 6, logicalColumn := 200, sourceArmColumn := 35644, finalColumn := 538020 }
, { child := 6, logicalColumn := 201, sourceArmColumn := 35649, finalColumn := 538025 }
, { child := 6, logicalColumn := 202, sourceArmColumn := 35654, finalColumn := 538030 }
, { child := 6, logicalColumn := 203, sourceArmColumn := 35659, finalColumn := 538035 }
, { child := 6, logicalColumn := 204, sourceArmColumn := 35664, finalColumn := 538040 }
, { child := 6, logicalColumn := 205, sourceArmColumn := 35669, finalColumn := 538045 }
, { child := 6, logicalColumn := 206, sourceArmColumn := 35674, finalColumn := 538050 }
, { child := 6, logicalColumn := 207, sourceArmColumn := 35679, finalColumn := 538055 }
, { child := 6, logicalColumn := 208, sourceArmColumn := 35684, finalColumn := 538060 }
, { child := 6, logicalColumn := 209, sourceArmColumn := 35689, finalColumn := 538065 }
, { child := 6, logicalColumn := 210, sourceArmColumn := 35694, finalColumn := 538070 }
, { child := 6, logicalColumn := 211, sourceArmColumn := 35699, finalColumn := 538075 }
, { child := 6, logicalColumn := 212, sourceArmColumn := 35704, finalColumn := 538080 }
, { child := 6, logicalColumn := 213, sourceArmColumn := 35709, finalColumn := 538085 }
, { child := 6, logicalColumn := 214, sourceArmColumn := 35714, finalColumn := 538090 }
, { child := 6, logicalColumn := 215, sourceArmColumn := 35719, finalColumn := 538095 }
, { child := 6, logicalColumn := 216, sourceArmColumn := 35455, finalColumn := 537831 }
, { child := 6, logicalColumn := 217, sourceArmColumn := 35460, finalColumn := 537836 }
, { child := 6, logicalColumn := 218, sourceArmColumn := 35465, finalColumn := 537841 }
, { child := 6, logicalColumn := 219, sourceArmColumn := 35470, finalColumn := 537846 }
, { child := 6, logicalColumn := 220, sourceArmColumn := 35475, finalColumn := 537851 }
, { child := 6, logicalColumn := 221, sourceArmColumn := 35480, finalColumn := 537856 }
, { child := 6, logicalColumn := 222, sourceArmColumn := 35485, finalColumn := 537861 }
, { child := 6, logicalColumn := 223, sourceArmColumn := 35490, finalColumn := 537866 }
, { child := 6, logicalColumn := 224, sourceArmColumn := 35495, finalColumn := 537871 }
, { child := 6, logicalColumn := 225, sourceArmColumn := 35500, finalColumn := 537876 }
, { child := 6, logicalColumn := 226, sourceArmColumn := 35505, finalColumn := 537881 }
, { child := 6, logicalColumn := 227, sourceArmColumn := 35510, finalColumn := 537886 }
, { child := 6, logicalColumn := 228, sourceArmColumn := 35515, finalColumn := 537891 }
, { child := 6, logicalColumn := 229, sourceArmColumn := 35520, finalColumn := 537896 }
, { child := 6, logicalColumn := 230, sourceArmColumn := 35525, finalColumn := 537901 }
, { child := 6, logicalColumn := 231, sourceArmColumn := 35530, finalColumn := 537906 }
, { child := 6, logicalColumn := 232, sourceArmColumn := 35535, finalColumn := 537911 }
, { child := 6, logicalColumn := 233, sourceArmColumn := 35540, finalColumn := 537916 }
, { child := 6, logicalColumn := 234, sourceArmColumn := 35545, finalColumn := 537921 }
, { child := 6, logicalColumn := 235, sourceArmColumn := 35550, finalColumn := 537926 }
, { child := 6, logicalColumn := 236, sourceArmColumn := 35555, finalColumn := 537931 }
, { child := 6, logicalColumn := 237, sourceArmColumn := 35560, finalColumn := 537936 }
, { child := 6, logicalColumn := 238, sourceArmColumn := 35565, finalColumn := 537941 }
, { child := 6, logicalColumn := 239, sourceArmColumn := 35570, finalColumn := 537946 }
, { child := 6, logicalColumn := 240, sourceArmColumn := 35575, finalColumn := 537951 }
, { child := 6, logicalColumn := 241, sourceArmColumn := 35580, finalColumn := 537956 }
, { child := 6, logicalColumn := 242, sourceArmColumn := 35585, finalColumn := 537961 }
, { child := 6, logicalColumn := 243, sourceArmColumn := 35590, finalColumn := 537966 }
, { child := 6, logicalColumn := 244, sourceArmColumn := 35595, finalColumn := 537971 }
, { child := 6, logicalColumn := 245, sourceArmColumn := 35600, finalColumn := 537976 }
, { child := 6, logicalColumn := 246, sourceArmColumn := 35605, finalColumn := 537981 }
, { child := 6, logicalColumn := 247, sourceArmColumn := 35610, finalColumn := 537986 }
, { child := 6, logicalColumn := 248, sourceArmColumn := 35615, finalColumn := 537991 }
, { child := 6, logicalColumn := 249, sourceArmColumn := 35620, finalColumn := 537996 }
, { child := 6, logicalColumn := 250, sourceArmColumn := 35625, finalColumn := 538001 }
, { child := 6, logicalColumn := 251, sourceArmColumn := 35630, finalColumn := 538006 }
, { child := 6, logicalColumn := 252, sourceArmColumn := 35635, finalColumn := 538011 }
, { child := 6, logicalColumn := 253, sourceArmColumn := 35640, finalColumn := 538016 }
, { child := 6, logicalColumn := 254, sourceArmColumn := 35645, finalColumn := 538021 }
, { child := 6, logicalColumn := 255, sourceArmColumn := 35650, finalColumn := 538026 }
, { child := 6, logicalColumn := 256, sourceArmColumn := 35655, finalColumn := 538031 }
, { child := 6, logicalColumn := 257, sourceArmColumn := 35660, finalColumn := 538036 }
, { child := 6, logicalColumn := 258, sourceArmColumn := 35665, finalColumn := 538041 }
, { child := 6, logicalColumn := 259, sourceArmColumn := 35670, finalColumn := 538046 }
, { child := 6, logicalColumn := 260, sourceArmColumn := 35675, finalColumn := 538051 }
, { child := 6, logicalColumn := 261, sourceArmColumn := 35680, finalColumn := 538056 }
, { child := 6, logicalColumn := 262, sourceArmColumn := 35685, finalColumn := 538061 }
, { child := 6, logicalColumn := 263, sourceArmColumn := 35690, finalColumn := 538066 }
, { child := 6, logicalColumn := 264, sourceArmColumn := 35695, finalColumn := 538071 }
, { child := 6, logicalColumn := 265, sourceArmColumn := 35700, finalColumn := 538076 }
, { child := 6, logicalColumn := 266, sourceArmColumn := 35705, finalColumn := 538081 }
, { child := 6, logicalColumn := 267, sourceArmColumn := 35710, finalColumn := 538086 }
, { child := 6, logicalColumn := 268, sourceArmColumn := 35715, finalColumn := 538091 }
, { child := 6, logicalColumn := 269, sourceArmColumn := 35720, finalColumn := 538096 }
, { child := 7, logicalColumn := 0, sourceArmColumn := 37723, finalColumn := 604517 }
, { child := 7, logicalColumn := 1, sourceArmColumn := 37728, finalColumn := 604522 }
, { child := 7, logicalColumn := 2, sourceArmColumn := 37733, finalColumn := 604527 }
, { child := 7, logicalColumn := 3, sourceArmColumn := 37738, finalColumn := 604532 }
, { child := 7, logicalColumn := 4, sourceArmColumn := 37743, finalColumn := 604537 }
, { child := 7, logicalColumn := 5, sourceArmColumn := 37748, finalColumn := 604542 }
, { child := 7, logicalColumn := 6, sourceArmColumn := 37753, finalColumn := 604547 }
, { child := 7, logicalColumn := 7, sourceArmColumn := 37758, finalColumn := 604552 }
, { child := 7, logicalColumn := 8, sourceArmColumn := 37763, finalColumn := 604557 }
, { child := 7, logicalColumn := 9, sourceArmColumn := 37768, finalColumn := 604562 }
, { child := 7, logicalColumn := 10, sourceArmColumn := 37773, finalColumn := 604567 }
, { child := 7, logicalColumn := 11, sourceArmColumn := 37778, finalColumn := 604572 }
, { child := 7, logicalColumn := 12, sourceArmColumn := 37783, finalColumn := 604577 }
, { child := 7, logicalColumn := 13, sourceArmColumn := 37788, finalColumn := 604582 }
, { child := 7, logicalColumn := 14, sourceArmColumn := 37793, finalColumn := 604587 }
, { child := 7, logicalColumn := 15, sourceArmColumn := 37798, finalColumn := 604592 }
, { child := 7, logicalColumn := 16, sourceArmColumn := 37803, finalColumn := 604597 }
, { child := 7, logicalColumn := 17, sourceArmColumn := 37808, finalColumn := 604602 }
, { child := 7, logicalColumn := 18, sourceArmColumn := 37813, finalColumn := 604607 }
, { child := 7, logicalColumn := 19, sourceArmColumn := 37818, finalColumn := 604612 }
, { child := 7, logicalColumn := 20, sourceArmColumn := 37823, finalColumn := 604617 }
, { child := 7, logicalColumn := 21, sourceArmColumn := 37828, finalColumn := 604622 }
, { child := 7, logicalColumn := 22, sourceArmColumn := 37833, finalColumn := 604627 }
, { child := 7, logicalColumn := 23, sourceArmColumn := 37838, finalColumn := 604632 }
, { child := 7, logicalColumn := 24, sourceArmColumn := 37843, finalColumn := 604637 }
, { child := 7, logicalColumn := 25, sourceArmColumn := 37848, finalColumn := 604642 }
, { child := 7, logicalColumn := 26, sourceArmColumn := 37853, finalColumn := 604647 }
, { child := 7, logicalColumn := 27, sourceArmColumn := 37858, finalColumn := 604652 }
, { child := 7, logicalColumn := 28, sourceArmColumn := 37863, finalColumn := 604657 }
, { child := 7, logicalColumn := 29, sourceArmColumn := 37868, finalColumn := 604662 }
, { child := 7, logicalColumn := 30, sourceArmColumn := 37873, finalColumn := 604667 }
, { child := 7, logicalColumn := 31, sourceArmColumn := 37878, finalColumn := 604672 }
, { child := 7, logicalColumn := 32, sourceArmColumn := 37883, finalColumn := 604677 }
, { child := 7, logicalColumn := 33, sourceArmColumn := 37888, finalColumn := 604682 }
, { child := 7, logicalColumn := 34, sourceArmColumn := 37893, finalColumn := 604687 }
, { child := 7, logicalColumn := 35, sourceArmColumn := 37898, finalColumn := 604692 }
, { child := 7, logicalColumn := 36, sourceArmColumn := 37903, finalColumn := 604697 }
, { child := 7, logicalColumn := 37, sourceArmColumn := 37908, finalColumn := 604702 }
, { child := 7, logicalColumn := 38, sourceArmColumn := 37913, finalColumn := 604707 }
, { child := 7, logicalColumn := 39, sourceArmColumn := 37918, finalColumn := 604712 }
, { child := 7, logicalColumn := 40, sourceArmColumn := 37923, finalColumn := 604717 }
, { child := 7, logicalColumn := 41, sourceArmColumn := 37928, finalColumn := 604722 }
, { child := 7, logicalColumn := 42, sourceArmColumn := 37933, finalColumn := 604727 }
, { child := 7, logicalColumn := 43, sourceArmColumn := 37938, finalColumn := 604732 }
, { child := 7, logicalColumn := 44, sourceArmColumn := 37943, finalColumn := 604737 }
, { child := 7, logicalColumn := 45, sourceArmColumn := 37948, finalColumn := 604742 }
, { child := 7, logicalColumn := 46, sourceArmColumn := 37953, finalColumn := 604747 }
, { child := 7, logicalColumn := 47, sourceArmColumn := 37958, finalColumn := 604752 }
, { child := 7, logicalColumn := 48, sourceArmColumn := 37963, finalColumn := 604757 }
, { child := 7, logicalColumn := 49, sourceArmColumn := 37968, finalColumn := 604762 }
, { child := 7, logicalColumn := 50, sourceArmColumn := 37973, finalColumn := 604767 }
, { child := 7, logicalColumn := 51, sourceArmColumn := 37978, finalColumn := 604772 }
, { child := 7, logicalColumn := 52, sourceArmColumn := 37983, finalColumn := 604777 }
, { child := 7, logicalColumn := 53, sourceArmColumn := 37988, finalColumn := 604782 }
, { child := 7, logicalColumn := 54, sourceArmColumn := 37724, finalColumn := 604518 }
, { child := 7, logicalColumn := 55, sourceArmColumn := 37729, finalColumn := 604523 }
, { child := 7, logicalColumn := 56, sourceArmColumn := 37734, finalColumn := 604528 }
, { child := 7, logicalColumn := 57, sourceArmColumn := 37739, finalColumn := 604533 }
, { child := 7, logicalColumn := 58, sourceArmColumn := 37744, finalColumn := 604538 }
, { child := 7, logicalColumn := 59, sourceArmColumn := 37749, finalColumn := 604543 }
, { child := 7, logicalColumn := 60, sourceArmColumn := 37754, finalColumn := 604548 }
, { child := 7, logicalColumn := 61, sourceArmColumn := 37759, finalColumn := 604553 }
, { child := 7, logicalColumn := 62, sourceArmColumn := 37764, finalColumn := 604558 }
, { child := 7, logicalColumn := 63, sourceArmColumn := 37769, finalColumn := 604563 }
, { child := 7, logicalColumn := 64, sourceArmColumn := 37774, finalColumn := 604568 }
, { child := 7, logicalColumn := 65, sourceArmColumn := 37779, finalColumn := 604573 }
, { child := 7, logicalColumn := 66, sourceArmColumn := 37784, finalColumn := 604578 }
, { child := 7, logicalColumn := 67, sourceArmColumn := 37789, finalColumn := 604583 }
, { child := 7, logicalColumn := 68, sourceArmColumn := 37794, finalColumn := 604588 }
, { child := 7, logicalColumn := 69, sourceArmColumn := 37799, finalColumn := 604593 }
, { child := 7, logicalColumn := 70, sourceArmColumn := 37804, finalColumn := 604598 }
, { child := 7, logicalColumn := 71, sourceArmColumn := 37809, finalColumn := 604603 }
, { child := 7, logicalColumn := 72, sourceArmColumn := 37814, finalColumn := 604608 }
, { child := 7, logicalColumn := 73, sourceArmColumn := 37819, finalColumn := 604613 }
, { child := 7, logicalColumn := 74, sourceArmColumn := 37824, finalColumn := 604618 }
, { child := 7, logicalColumn := 75, sourceArmColumn := 37829, finalColumn := 604623 }
, { child := 7, logicalColumn := 76, sourceArmColumn := 37834, finalColumn := 604628 }
, { child := 7, logicalColumn := 77, sourceArmColumn := 37839, finalColumn := 604633 }
, { child := 7, logicalColumn := 78, sourceArmColumn := 37844, finalColumn := 604638 }
, { child := 7, logicalColumn := 79, sourceArmColumn := 37849, finalColumn := 604643 }
, { child := 7, logicalColumn := 80, sourceArmColumn := 37854, finalColumn := 604648 }
, { child := 7, logicalColumn := 81, sourceArmColumn := 37859, finalColumn := 604653 }
, { child := 7, logicalColumn := 82, sourceArmColumn := 37864, finalColumn := 604658 }
, { child := 7, logicalColumn := 83, sourceArmColumn := 37869, finalColumn := 604663 }
, { child := 7, logicalColumn := 84, sourceArmColumn := 37874, finalColumn := 604668 }
, { child := 7, logicalColumn := 85, sourceArmColumn := 37879, finalColumn := 604673 }
, { child := 7, logicalColumn := 86, sourceArmColumn := 37884, finalColumn := 604678 }
, { child := 7, logicalColumn := 87, sourceArmColumn := 37889, finalColumn := 604683 }
, { child := 7, logicalColumn := 88, sourceArmColumn := 37894, finalColumn := 604688 }
, { child := 7, logicalColumn := 89, sourceArmColumn := 37899, finalColumn := 604693 }
, { child := 7, logicalColumn := 90, sourceArmColumn := 37904, finalColumn := 604698 }
, { child := 7, logicalColumn := 91, sourceArmColumn := 37909, finalColumn := 604703 }
, { child := 7, logicalColumn := 92, sourceArmColumn := 37914, finalColumn := 604708 }
, { child := 7, logicalColumn := 93, sourceArmColumn := 37919, finalColumn := 604713 }
, { child := 7, logicalColumn := 94, sourceArmColumn := 37924, finalColumn := 604718 }
, { child := 7, logicalColumn := 95, sourceArmColumn := 37929, finalColumn := 604723 }
, { child := 7, logicalColumn := 96, sourceArmColumn := 37934, finalColumn := 604728 }
, { child := 7, logicalColumn := 97, sourceArmColumn := 37939, finalColumn := 604733 }
, { child := 7, logicalColumn := 98, sourceArmColumn := 37944, finalColumn := 604738 }
, { child := 7, logicalColumn := 99, sourceArmColumn := 37949, finalColumn := 604743 }
, { child := 7, logicalColumn := 100, sourceArmColumn := 37954, finalColumn := 604748 }
, { child := 7, logicalColumn := 101, sourceArmColumn := 37959, finalColumn := 604753 }
, { child := 7, logicalColumn := 102, sourceArmColumn := 37964, finalColumn := 604758 }
, { child := 7, logicalColumn := 103, sourceArmColumn := 37969, finalColumn := 604763 }
, { child := 7, logicalColumn := 104, sourceArmColumn := 37974, finalColumn := 604768 }
, { child := 7, logicalColumn := 105, sourceArmColumn := 37979, finalColumn := 604773 }
, { child := 7, logicalColumn := 106, sourceArmColumn := 37984, finalColumn := 604778 }
, { child := 7, logicalColumn := 107, sourceArmColumn := 37989, finalColumn := 604783 }
, { child := 7, logicalColumn := 108, sourceArmColumn := 37725, finalColumn := 604519 }
, { child := 7, logicalColumn := 109, sourceArmColumn := 37730, finalColumn := 604524 }
, { child := 7, logicalColumn := 110, sourceArmColumn := 37735, finalColumn := 604529 }
, { child := 7, logicalColumn := 111, sourceArmColumn := 37740, finalColumn := 604534 }
, { child := 7, logicalColumn := 112, sourceArmColumn := 37745, finalColumn := 604539 }
, { child := 7, logicalColumn := 113, sourceArmColumn := 37750, finalColumn := 604544 }
, { child := 7, logicalColumn := 114, sourceArmColumn := 37755, finalColumn := 604549 }
, { child := 7, logicalColumn := 115, sourceArmColumn := 37760, finalColumn := 604554 }
, { child := 7, logicalColumn := 116, sourceArmColumn := 37765, finalColumn := 604559 }
, { child := 7, logicalColumn := 117, sourceArmColumn := 37770, finalColumn := 604564 }
, { child := 7, logicalColumn := 118, sourceArmColumn := 37775, finalColumn := 604569 }
, { child := 7, logicalColumn := 119, sourceArmColumn := 37780, finalColumn := 604574 }
, { child := 7, logicalColumn := 120, sourceArmColumn := 37785, finalColumn := 604579 }
, { child := 7, logicalColumn := 121, sourceArmColumn := 37790, finalColumn := 604584 }
, { child := 7, logicalColumn := 122, sourceArmColumn := 37795, finalColumn := 604589 }
, { child := 7, logicalColumn := 123, sourceArmColumn := 37800, finalColumn := 604594 }
, { child := 7, logicalColumn := 124, sourceArmColumn := 37805, finalColumn := 604599 }
, { child := 7, logicalColumn := 125, sourceArmColumn := 37810, finalColumn := 604604 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk7
