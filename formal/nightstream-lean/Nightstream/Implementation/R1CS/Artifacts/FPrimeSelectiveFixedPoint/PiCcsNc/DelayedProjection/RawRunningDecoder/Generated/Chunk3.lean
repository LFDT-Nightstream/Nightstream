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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk3

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 2, logicalColumn := 216, sourceArmColumn := 26367, finalColumn := 271071 }
, { child := 2, logicalColumn := 217, sourceArmColumn := 26372, finalColumn := 271076 }
, { child := 2, logicalColumn := 218, sourceArmColumn := 26377, finalColumn := 271081 }
, { child := 2, logicalColumn := 219, sourceArmColumn := 26382, finalColumn := 271086 }
, { child := 2, logicalColumn := 220, sourceArmColumn := 26387, finalColumn := 271091 }
, { child := 2, logicalColumn := 221, sourceArmColumn := 26392, finalColumn := 271096 }
, { child := 2, logicalColumn := 222, sourceArmColumn := 26397, finalColumn := 271101 }
, { child := 2, logicalColumn := 223, sourceArmColumn := 26402, finalColumn := 271106 }
, { child := 2, logicalColumn := 224, sourceArmColumn := 26407, finalColumn := 271111 }
, { child := 2, logicalColumn := 225, sourceArmColumn := 26412, finalColumn := 271116 }
, { child := 2, logicalColumn := 226, sourceArmColumn := 26417, finalColumn := 271121 }
, { child := 2, logicalColumn := 227, sourceArmColumn := 26422, finalColumn := 271126 }
, { child := 2, logicalColumn := 228, sourceArmColumn := 26427, finalColumn := 271131 }
, { child := 2, logicalColumn := 229, sourceArmColumn := 26432, finalColumn := 271136 }
, { child := 2, logicalColumn := 230, sourceArmColumn := 26437, finalColumn := 271141 }
, { child := 2, logicalColumn := 231, sourceArmColumn := 26442, finalColumn := 271146 }
, { child := 2, logicalColumn := 232, sourceArmColumn := 26447, finalColumn := 271151 }
, { child := 2, logicalColumn := 233, sourceArmColumn := 26452, finalColumn := 271156 }
, { child := 2, logicalColumn := 234, sourceArmColumn := 26457, finalColumn := 271161 }
, { child := 2, logicalColumn := 235, sourceArmColumn := 26462, finalColumn := 271166 }
, { child := 2, logicalColumn := 236, sourceArmColumn := 26467, finalColumn := 271171 }
, { child := 2, logicalColumn := 237, sourceArmColumn := 26472, finalColumn := 271176 }
, { child := 2, logicalColumn := 238, sourceArmColumn := 26477, finalColumn := 271181 }
, { child := 2, logicalColumn := 239, sourceArmColumn := 26482, finalColumn := 271186 }
, { child := 2, logicalColumn := 240, sourceArmColumn := 26487, finalColumn := 271191 }
, { child := 2, logicalColumn := 241, sourceArmColumn := 26492, finalColumn := 271196 }
, { child := 2, logicalColumn := 242, sourceArmColumn := 26497, finalColumn := 271201 }
, { child := 2, logicalColumn := 243, sourceArmColumn := 26502, finalColumn := 271206 }
, { child := 2, logicalColumn := 244, sourceArmColumn := 26507, finalColumn := 271211 }
, { child := 2, logicalColumn := 245, sourceArmColumn := 26512, finalColumn := 271216 }
, { child := 2, logicalColumn := 246, sourceArmColumn := 26517, finalColumn := 271221 }
, { child := 2, logicalColumn := 247, sourceArmColumn := 26522, finalColumn := 271226 }
, { child := 2, logicalColumn := 248, sourceArmColumn := 26527, finalColumn := 271231 }
, { child := 2, logicalColumn := 249, sourceArmColumn := 26532, finalColumn := 271236 }
, { child := 2, logicalColumn := 250, sourceArmColumn := 26537, finalColumn := 271241 }
, { child := 2, logicalColumn := 251, sourceArmColumn := 26542, finalColumn := 271246 }
, { child := 2, logicalColumn := 252, sourceArmColumn := 26547, finalColumn := 271251 }
, { child := 2, logicalColumn := 253, sourceArmColumn := 26552, finalColumn := 271256 }
, { child := 2, logicalColumn := 254, sourceArmColumn := 26557, finalColumn := 271261 }
, { child := 2, logicalColumn := 255, sourceArmColumn := 26562, finalColumn := 271266 }
, { child := 2, logicalColumn := 256, sourceArmColumn := 26567, finalColumn := 271271 }
, { child := 2, logicalColumn := 257, sourceArmColumn := 26572, finalColumn := 271276 }
, { child := 2, logicalColumn := 258, sourceArmColumn := 26577, finalColumn := 271281 }
, { child := 2, logicalColumn := 259, sourceArmColumn := 26582, finalColumn := 271286 }
, { child := 2, logicalColumn := 260, sourceArmColumn := 26587, finalColumn := 271291 }
, { child := 2, logicalColumn := 261, sourceArmColumn := 26592, finalColumn := 271296 }
, { child := 2, logicalColumn := 262, sourceArmColumn := 26597, finalColumn := 271301 }
, { child := 2, logicalColumn := 263, sourceArmColumn := 26602, finalColumn := 271306 }
, { child := 2, logicalColumn := 264, sourceArmColumn := 26607, finalColumn := 271311 }
, { child := 2, logicalColumn := 265, sourceArmColumn := 26612, finalColumn := 271316 }
, { child := 2, logicalColumn := 266, sourceArmColumn := 26617, finalColumn := 271321 }
, { child := 2, logicalColumn := 267, sourceArmColumn := 26622, finalColumn := 271326 }
, { child := 2, logicalColumn := 268, sourceArmColumn := 26627, finalColumn := 271331 }
, { child := 2, logicalColumn := 269, sourceArmColumn := 26632, finalColumn := 271336 }
, { child := 3, logicalColumn := 0, sourceArmColumn := 28635, finalColumn := 337757 }
, { child := 3, logicalColumn := 1, sourceArmColumn := 28640, finalColumn := 337762 }
, { child := 3, logicalColumn := 2, sourceArmColumn := 28645, finalColumn := 337767 }
, { child := 3, logicalColumn := 3, sourceArmColumn := 28650, finalColumn := 337772 }
, { child := 3, logicalColumn := 4, sourceArmColumn := 28655, finalColumn := 337777 }
, { child := 3, logicalColumn := 5, sourceArmColumn := 28660, finalColumn := 337782 }
, { child := 3, logicalColumn := 6, sourceArmColumn := 28665, finalColumn := 337787 }
, { child := 3, logicalColumn := 7, sourceArmColumn := 28670, finalColumn := 337792 }
, { child := 3, logicalColumn := 8, sourceArmColumn := 28675, finalColumn := 337797 }
, { child := 3, logicalColumn := 9, sourceArmColumn := 28680, finalColumn := 337802 }
, { child := 3, logicalColumn := 10, sourceArmColumn := 28685, finalColumn := 337807 }
, { child := 3, logicalColumn := 11, sourceArmColumn := 28690, finalColumn := 337812 }
, { child := 3, logicalColumn := 12, sourceArmColumn := 28695, finalColumn := 337817 }
, { child := 3, logicalColumn := 13, sourceArmColumn := 28700, finalColumn := 337822 }
, { child := 3, logicalColumn := 14, sourceArmColumn := 28705, finalColumn := 337827 }
, { child := 3, logicalColumn := 15, sourceArmColumn := 28710, finalColumn := 337832 }
, { child := 3, logicalColumn := 16, sourceArmColumn := 28715, finalColumn := 337837 }
, { child := 3, logicalColumn := 17, sourceArmColumn := 28720, finalColumn := 337842 }
, { child := 3, logicalColumn := 18, sourceArmColumn := 28725, finalColumn := 337847 }
, { child := 3, logicalColumn := 19, sourceArmColumn := 28730, finalColumn := 337852 }
, { child := 3, logicalColumn := 20, sourceArmColumn := 28735, finalColumn := 337857 }
, { child := 3, logicalColumn := 21, sourceArmColumn := 28740, finalColumn := 337862 }
, { child := 3, logicalColumn := 22, sourceArmColumn := 28745, finalColumn := 337867 }
, { child := 3, logicalColumn := 23, sourceArmColumn := 28750, finalColumn := 337872 }
, { child := 3, logicalColumn := 24, sourceArmColumn := 28755, finalColumn := 337877 }
, { child := 3, logicalColumn := 25, sourceArmColumn := 28760, finalColumn := 337882 }
, { child := 3, logicalColumn := 26, sourceArmColumn := 28765, finalColumn := 337887 }
, { child := 3, logicalColumn := 27, sourceArmColumn := 28770, finalColumn := 337892 }
, { child := 3, logicalColumn := 28, sourceArmColumn := 28775, finalColumn := 337897 }
, { child := 3, logicalColumn := 29, sourceArmColumn := 28780, finalColumn := 337902 }
, { child := 3, logicalColumn := 30, sourceArmColumn := 28785, finalColumn := 337907 }
, { child := 3, logicalColumn := 31, sourceArmColumn := 28790, finalColumn := 337912 }
, { child := 3, logicalColumn := 32, sourceArmColumn := 28795, finalColumn := 337917 }
, { child := 3, logicalColumn := 33, sourceArmColumn := 28800, finalColumn := 337922 }
, { child := 3, logicalColumn := 34, sourceArmColumn := 28805, finalColumn := 337927 }
, { child := 3, logicalColumn := 35, sourceArmColumn := 28810, finalColumn := 337932 }
, { child := 3, logicalColumn := 36, sourceArmColumn := 28815, finalColumn := 337937 }
, { child := 3, logicalColumn := 37, sourceArmColumn := 28820, finalColumn := 337942 }
, { child := 3, logicalColumn := 38, sourceArmColumn := 28825, finalColumn := 337947 }
, { child := 3, logicalColumn := 39, sourceArmColumn := 28830, finalColumn := 337952 }
, { child := 3, logicalColumn := 40, sourceArmColumn := 28835, finalColumn := 337957 }
, { child := 3, logicalColumn := 41, sourceArmColumn := 28840, finalColumn := 337962 }
, { child := 3, logicalColumn := 42, sourceArmColumn := 28845, finalColumn := 337967 }
, { child := 3, logicalColumn := 43, sourceArmColumn := 28850, finalColumn := 337972 }
, { child := 3, logicalColumn := 44, sourceArmColumn := 28855, finalColumn := 337977 }
, { child := 3, logicalColumn := 45, sourceArmColumn := 28860, finalColumn := 337982 }
, { child := 3, logicalColumn := 46, sourceArmColumn := 28865, finalColumn := 337987 }
, { child := 3, logicalColumn := 47, sourceArmColumn := 28870, finalColumn := 337992 }
, { child := 3, logicalColumn := 48, sourceArmColumn := 28875, finalColumn := 337997 }
, { child := 3, logicalColumn := 49, sourceArmColumn := 28880, finalColumn := 338002 }
, { child := 3, logicalColumn := 50, sourceArmColumn := 28885, finalColumn := 338007 }
, { child := 3, logicalColumn := 51, sourceArmColumn := 28890, finalColumn := 338012 }
, { child := 3, logicalColumn := 52, sourceArmColumn := 28895, finalColumn := 338017 }
, { child := 3, logicalColumn := 53, sourceArmColumn := 28900, finalColumn := 338022 }
, { child := 3, logicalColumn := 54, sourceArmColumn := 28636, finalColumn := 337758 }
, { child := 3, logicalColumn := 55, sourceArmColumn := 28641, finalColumn := 337763 }
, { child := 3, logicalColumn := 56, sourceArmColumn := 28646, finalColumn := 337768 }
, { child := 3, logicalColumn := 57, sourceArmColumn := 28651, finalColumn := 337773 }
, { child := 3, logicalColumn := 58, sourceArmColumn := 28656, finalColumn := 337778 }
, { child := 3, logicalColumn := 59, sourceArmColumn := 28661, finalColumn := 337783 }
, { child := 3, logicalColumn := 60, sourceArmColumn := 28666, finalColumn := 337788 }
, { child := 3, logicalColumn := 61, sourceArmColumn := 28671, finalColumn := 337793 }
, { child := 3, logicalColumn := 62, sourceArmColumn := 28676, finalColumn := 337798 }
, { child := 3, logicalColumn := 63, sourceArmColumn := 28681, finalColumn := 337803 }
, { child := 3, logicalColumn := 64, sourceArmColumn := 28686, finalColumn := 337808 }
, { child := 3, logicalColumn := 65, sourceArmColumn := 28691, finalColumn := 337813 }
, { child := 3, logicalColumn := 66, sourceArmColumn := 28696, finalColumn := 337818 }
, { child := 3, logicalColumn := 67, sourceArmColumn := 28701, finalColumn := 337823 }
, { child := 3, logicalColumn := 68, sourceArmColumn := 28706, finalColumn := 337828 }
, { child := 3, logicalColumn := 69, sourceArmColumn := 28711, finalColumn := 337833 }
, { child := 3, logicalColumn := 70, sourceArmColumn := 28716, finalColumn := 337838 }
, { child := 3, logicalColumn := 71, sourceArmColumn := 28721, finalColumn := 337843 }
, { child := 3, logicalColumn := 72, sourceArmColumn := 28726, finalColumn := 337848 }
, { child := 3, logicalColumn := 73, sourceArmColumn := 28731, finalColumn := 337853 }
, { child := 3, logicalColumn := 74, sourceArmColumn := 28736, finalColumn := 337858 }
, { child := 3, logicalColumn := 75, sourceArmColumn := 28741, finalColumn := 337863 }
, { child := 3, logicalColumn := 76, sourceArmColumn := 28746, finalColumn := 337868 }
, { child := 3, logicalColumn := 77, sourceArmColumn := 28751, finalColumn := 337873 }
, { child := 3, logicalColumn := 78, sourceArmColumn := 28756, finalColumn := 337878 }
, { child := 3, logicalColumn := 79, sourceArmColumn := 28761, finalColumn := 337883 }
, { child := 3, logicalColumn := 80, sourceArmColumn := 28766, finalColumn := 337888 }
, { child := 3, logicalColumn := 81, sourceArmColumn := 28771, finalColumn := 337893 }
, { child := 3, logicalColumn := 82, sourceArmColumn := 28776, finalColumn := 337898 }
, { child := 3, logicalColumn := 83, sourceArmColumn := 28781, finalColumn := 337903 }
, { child := 3, logicalColumn := 84, sourceArmColumn := 28786, finalColumn := 337908 }
, { child := 3, logicalColumn := 85, sourceArmColumn := 28791, finalColumn := 337913 }
, { child := 3, logicalColumn := 86, sourceArmColumn := 28796, finalColumn := 337918 }
, { child := 3, logicalColumn := 87, sourceArmColumn := 28801, finalColumn := 337923 }
, { child := 3, logicalColumn := 88, sourceArmColumn := 28806, finalColumn := 337928 }
, { child := 3, logicalColumn := 89, sourceArmColumn := 28811, finalColumn := 337933 }
, { child := 3, logicalColumn := 90, sourceArmColumn := 28816, finalColumn := 337938 }
, { child := 3, logicalColumn := 91, sourceArmColumn := 28821, finalColumn := 337943 }
, { child := 3, logicalColumn := 92, sourceArmColumn := 28826, finalColumn := 337948 }
, { child := 3, logicalColumn := 93, sourceArmColumn := 28831, finalColumn := 337953 }
, { child := 3, logicalColumn := 94, sourceArmColumn := 28836, finalColumn := 337958 }
, { child := 3, logicalColumn := 95, sourceArmColumn := 28841, finalColumn := 337963 }
, { child := 3, logicalColumn := 96, sourceArmColumn := 28846, finalColumn := 337968 }
, { child := 3, logicalColumn := 97, sourceArmColumn := 28851, finalColumn := 337973 }
, { child := 3, logicalColumn := 98, sourceArmColumn := 28856, finalColumn := 337978 }
, { child := 3, logicalColumn := 99, sourceArmColumn := 28861, finalColumn := 337983 }
, { child := 3, logicalColumn := 100, sourceArmColumn := 28866, finalColumn := 337988 }
, { child := 3, logicalColumn := 101, sourceArmColumn := 28871, finalColumn := 337993 }
, { child := 3, logicalColumn := 102, sourceArmColumn := 28876, finalColumn := 337998 }
, { child := 3, logicalColumn := 103, sourceArmColumn := 28881, finalColumn := 338003 }
, { child := 3, logicalColumn := 104, sourceArmColumn := 28886, finalColumn := 338008 }
, { child := 3, logicalColumn := 105, sourceArmColumn := 28891, finalColumn := 338013 }
, { child := 3, logicalColumn := 106, sourceArmColumn := 28896, finalColumn := 338018 }
, { child := 3, logicalColumn := 107, sourceArmColumn := 28901, finalColumn := 338023 }
, { child := 3, logicalColumn := 108, sourceArmColumn := 28637, finalColumn := 337759 }
, { child := 3, logicalColumn := 109, sourceArmColumn := 28642, finalColumn := 337764 }
, { child := 3, logicalColumn := 110, sourceArmColumn := 28647, finalColumn := 337769 }
, { child := 3, logicalColumn := 111, sourceArmColumn := 28652, finalColumn := 337774 }
, { child := 3, logicalColumn := 112, sourceArmColumn := 28657, finalColumn := 337779 }
, { child := 3, logicalColumn := 113, sourceArmColumn := 28662, finalColumn := 337784 }
, { child := 3, logicalColumn := 114, sourceArmColumn := 28667, finalColumn := 337789 }
, { child := 3, logicalColumn := 115, sourceArmColumn := 28672, finalColumn := 337794 }
, { child := 3, logicalColumn := 116, sourceArmColumn := 28677, finalColumn := 337799 }
, { child := 3, logicalColumn := 117, sourceArmColumn := 28682, finalColumn := 337804 }
, { child := 3, logicalColumn := 118, sourceArmColumn := 28687, finalColumn := 337809 }
, { child := 3, logicalColumn := 119, sourceArmColumn := 28692, finalColumn := 337814 }
, { child := 3, logicalColumn := 120, sourceArmColumn := 28697, finalColumn := 337819 }
, { child := 3, logicalColumn := 121, sourceArmColumn := 28702, finalColumn := 337824 }
, { child := 3, logicalColumn := 122, sourceArmColumn := 28707, finalColumn := 337829 }
, { child := 3, logicalColumn := 123, sourceArmColumn := 28712, finalColumn := 337834 }
, { child := 3, logicalColumn := 124, sourceArmColumn := 28717, finalColumn := 337839 }
, { child := 3, logicalColumn := 125, sourceArmColumn := 28722, finalColumn := 337844 }
, { child := 3, logicalColumn := 126, sourceArmColumn := 28727, finalColumn := 337849 }
, { child := 3, logicalColumn := 127, sourceArmColumn := 28732, finalColumn := 337854 }
, { child := 3, logicalColumn := 128, sourceArmColumn := 28737, finalColumn := 337859 }
, { child := 3, logicalColumn := 129, sourceArmColumn := 28742, finalColumn := 337864 }
, { child := 3, logicalColumn := 130, sourceArmColumn := 28747, finalColumn := 337869 }
, { child := 3, logicalColumn := 131, sourceArmColumn := 28752, finalColumn := 337874 }
, { child := 3, logicalColumn := 132, sourceArmColumn := 28757, finalColumn := 337879 }
, { child := 3, logicalColumn := 133, sourceArmColumn := 28762, finalColumn := 337884 }
, { child := 3, logicalColumn := 134, sourceArmColumn := 28767, finalColumn := 337889 }
, { child := 3, logicalColumn := 135, sourceArmColumn := 28772, finalColumn := 337894 }
, { child := 3, logicalColumn := 136, sourceArmColumn := 28777, finalColumn := 337899 }
, { child := 3, logicalColumn := 137, sourceArmColumn := 28782, finalColumn := 337904 }
, { child := 3, logicalColumn := 138, sourceArmColumn := 28787, finalColumn := 337909 }
, { child := 3, logicalColumn := 139, sourceArmColumn := 28792, finalColumn := 337914 }
, { child := 3, logicalColumn := 140, sourceArmColumn := 28797, finalColumn := 337919 }
, { child := 3, logicalColumn := 141, sourceArmColumn := 28802, finalColumn := 337924 }
, { child := 3, logicalColumn := 142, sourceArmColumn := 28807, finalColumn := 337929 }
, { child := 3, logicalColumn := 143, sourceArmColumn := 28812, finalColumn := 337934 }
, { child := 3, logicalColumn := 144, sourceArmColumn := 28817, finalColumn := 337939 }
, { child := 3, logicalColumn := 145, sourceArmColumn := 28822, finalColumn := 337944 }
, { child := 3, logicalColumn := 146, sourceArmColumn := 28827, finalColumn := 337949 }
, { child := 3, logicalColumn := 147, sourceArmColumn := 28832, finalColumn := 337954 }
, { child := 3, logicalColumn := 148, sourceArmColumn := 28837, finalColumn := 337959 }
, { child := 3, logicalColumn := 149, sourceArmColumn := 28842, finalColumn := 337964 }
, { child := 3, logicalColumn := 150, sourceArmColumn := 28847, finalColumn := 337969 }
, { child := 3, logicalColumn := 151, sourceArmColumn := 28852, finalColumn := 337974 }
, { child := 3, logicalColumn := 152, sourceArmColumn := 28857, finalColumn := 337979 }
, { child := 3, logicalColumn := 153, sourceArmColumn := 28862, finalColumn := 337984 }
, { child := 3, logicalColumn := 154, sourceArmColumn := 28867, finalColumn := 337989 }
, { child := 3, logicalColumn := 155, sourceArmColumn := 28872, finalColumn := 337994 }
, { child := 3, logicalColumn := 156, sourceArmColumn := 28877, finalColumn := 337999 }
, { child := 3, logicalColumn := 157, sourceArmColumn := 28882, finalColumn := 338004 }
, { child := 3, logicalColumn := 158, sourceArmColumn := 28887, finalColumn := 338009 }
, { child := 3, logicalColumn := 159, sourceArmColumn := 28892, finalColumn := 338014 }
, { child := 3, logicalColumn := 160, sourceArmColumn := 28897, finalColumn := 338019 }
, { child := 3, logicalColumn := 161, sourceArmColumn := 28902, finalColumn := 338024 }
, { child := 3, logicalColumn := 162, sourceArmColumn := 28638, finalColumn := 337760 }
, { child := 3, logicalColumn := 163, sourceArmColumn := 28643, finalColumn := 337765 }
, { child := 3, logicalColumn := 164, sourceArmColumn := 28648, finalColumn := 337770 }
, { child := 3, logicalColumn := 165, sourceArmColumn := 28653, finalColumn := 337775 }
, { child := 3, logicalColumn := 166, sourceArmColumn := 28658, finalColumn := 337780 }
, { child := 3, logicalColumn := 167, sourceArmColumn := 28663, finalColumn := 337785 }
, { child := 3, logicalColumn := 168, sourceArmColumn := 28668, finalColumn := 337790 }
, { child := 3, logicalColumn := 169, sourceArmColumn := 28673, finalColumn := 337795 }
, { child := 3, logicalColumn := 170, sourceArmColumn := 28678, finalColumn := 337800 }
, { child := 3, logicalColumn := 171, sourceArmColumn := 28683, finalColumn := 337805 }
, { child := 3, logicalColumn := 172, sourceArmColumn := 28688, finalColumn := 337810 }
, { child := 3, logicalColumn := 173, sourceArmColumn := 28693, finalColumn := 337815 }
, { child := 3, logicalColumn := 174, sourceArmColumn := 28698, finalColumn := 337820 }
, { child := 3, logicalColumn := 175, sourceArmColumn := 28703, finalColumn := 337825 }
, { child := 3, logicalColumn := 176, sourceArmColumn := 28708, finalColumn := 337830 }
, { child := 3, logicalColumn := 177, sourceArmColumn := 28713, finalColumn := 337835 }
, { child := 3, logicalColumn := 178, sourceArmColumn := 28718, finalColumn := 337840 }
, { child := 3, logicalColumn := 179, sourceArmColumn := 28723, finalColumn := 337845 }
, { child := 3, logicalColumn := 180, sourceArmColumn := 28728, finalColumn := 337850 }
, { child := 3, logicalColumn := 181, sourceArmColumn := 28733, finalColumn := 337855 }
, { child := 3, logicalColumn := 182, sourceArmColumn := 28738, finalColumn := 337860 }
, { child := 3, logicalColumn := 183, sourceArmColumn := 28743, finalColumn := 337865 }
, { child := 3, logicalColumn := 184, sourceArmColumn := 28748, finalColumn := 337870 }
, { child := 3, logicalColumn := 185, sourceArmColumn := 28753, finalColumn := 337875 }
, { child := 3, logicalColumn := 186, sourceArmColumn := 28758, finalColumn := 337880 }
, { child := 3, logicalColumn := 187, sourceArmColumn := 28763, finalColumn := 337885 }
, { child := 3, logicalColumn := 188, sourceArmColumn := 28768, finalColumn := 337890 }
, { child := 3, logicalColumn := 189, sourceArmColumn := 28773, finalColumn := 337895 }
, { child := 3, logicalColumn := 190, sourceArmColumn := 28778, finalColumn := 337900 }
, { child := 3, logicalColumn := 191, sourceArmColumn := 28783, finalColumn := 337905 }
, { child := 3, logicalColumn := 192, sourceArmColumn := 28788, finalColumn := 337910 }
, { child := 3, logicalColumn := 193, sourceArmColumn := 28793, finalColumn := 337915 }
, { child := 3, logicalColumn := 194, sourceArmColumn := 28798, finalColumn := 337920 }
, { child := 3, logicalColumn := 195, sourceArmColumn := 28803, finalColumn := 337925 }
, { child := 3, logicalColumn := 196, sourceArmColumn := 28808, finalColumn := 337930 }
, { child := 3, logicalColumn := 197, sourceArmColumn := 28813, finalColumn := 337935 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk3
