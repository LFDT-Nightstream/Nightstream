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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk14

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 13, logicalColumn := 18, sourceArmColumn := 51445, finalColumn := 1004747 }
, { child := 13, logicalColumn := 19, sourceArmColumn := 51450, finalColumn := 1004752 }
, { child := 13, logicalColumn := 20, sourceArmColumn := 51455, finalColumn := 1004757 }
, { child := 13, logicalColumn := 21, sourceArmColumn := 51460, finalColumn := 1004762 }
, { child := 13, logicalColumn := 22, sourceArmColumn := 51465, finalColumn := 1004767 }
, { child := 13, logicalColumn := 23, sourceArmColumn := 51470, finalColumn := 1004772 }
, { child := 13, logicalColumn := 24, sourceArmColumn := 51475, finalColumn := 1004777 }
, { child := 13, logicalColumn := 25, sourceArmColumn := 51480, finalColumn := 1004782 }
, { child := 13, logicalColumn := 26, sourceArmColumn := 51485, finalColumn := 1004787 }
, { child := 13, logicalColumn := 27, sourceArmColumn := 51490, finalColumn := 1004792 }
, { child := 13, logicalColumn := 28, sourceArmColumn := 51495, finalColumn := 1004797 }
, { child := 13, logicalColumn := 29, sourceArmColumn := 51500, finalColumn := 1004802 }
, { child := 13, logicalColumn := 30, sourceArmColumn := 51505, finalColumn := 1004807 }
, { child := 13, logicalColumn := 31, sourceArmColumn := 51510, finalColumn := 1004812 }
, { child := 13, logicalColumn := 32, sourceArmColumn := 51515, finalColumn := 1004817 }
, { child := 13, logicalColumn := 33, sourceArmColumn := 51520, finalColumn := 1004822 }
, { child := 13, logicalColumn := 34, sourceArmColumn := 51525, finalColumn := 1004827 }
, { child := 13, logicalColumn := 35, sourceArmColumn := 51530, finalColumn := 1004832 }
, { child := 13, logicalColumn := 36, sourceArmColumn := 51535, finalColumn := 1004837 }
, { child := 13, logicalColumn := 37, sourceArmColumn := 51540, finalColumn := 1004842 }
, { child := 13, logicalColumn := 38, sourceArmColumn := 51545, finalColumn := 1004847 }
, { child := 13, logicalColumn := 39, sourceArmColumn := 51550, finalColumn := 1004852 }
, { child := 13, logicalColumn := 40, sourceArmColumn := 51555, finalColumn := 1004857 }
, { child := 13, logicalColumn := 41, sourceArmColumn := 51560, finalColumn := 1004862 }
, { child := 13, logicalColumn := 42, sourceArmColumn := 51565, finalColumn := 1004867 }
, { child := 13, logicalColumn := 43, sourceArmColumn := 51570, finalColumn := 1004872 }
, { child := 13, logicalColumn := 44, sourceArmColumn := 51575, finalColumn := 1004877 }
, { child := 13, logicalColumn := 45, sourceArmColumn := 51580, finalColumn := 1004882 }
, { child := 13, logicalColumn := 46, sourceArmColumn := 51585, finalColumn := 1004887 }
, { child := 13, logicalColumn := 47, sourceArmColumn := 51590, finalColumn := 1004892 }
, { child := 13, logicalColumn := 48, sourceArmColumn := 51595, finalColumn := 1004897 }
, { child := 13, logicalColumn := 49, sourceArmColumn := 51600, finalColumn := 1004902 }
, { child := 13, logicalColumn := 50, sourceArmColumn := 51605, finalColumn := 1004907 }
, { child := 13, logicalColumn := 51, sourceArmColumn := 51610, finalColumn := 1004912 }
, { child := 13, logicalColumn := 52, sourceArmColumn := 51615, finalColumn := 1004917 }
, { child := 13, logicalColumn := 53, sourceArmColumn := 51620, finalColumn := 1004922 }
, { child := 13, logicalColumn := 54, sourceArmColumn := 51356, finalColumn := 1004658 }
, { child := 13, logicalColumn := 55, sourceArmColumn := 51361, finalColumn := 1004663 }
, { child := 13, logicalColumn := 56, sourceArmColumn := 51366, finalColumn := 1004668 }
, { child := 13, logicalColumn := 57, sourceArmColumn := 51371, finalColumn := 1004673 }
, { child := 13, logicalColumn := 58, sourceArmColumn := 51376, finalColumn := 1004678 }
, { child := 13, logicalColumn := 59, sourceArmColumn := 51381, finalColumn := 1004683 }
, { child := 13, logicalColumn := 60, sourceArmColumn := 51386, finalColumn := 1004688 }
, { child := 13, logicalColumn := 61, sourceArmColumn := 51391, finalColumn := 1004693 }
, { child := 13, logicalColumn := 62, sourceArmColumn := 51396, finalColumn := 1004698 }
, { child := 13, logicalColumn := 63, sourceArmColumn := 51401, finalColumn := 1004703 }
, { child := 13, logicalColumn := 64, sourceArmColumn := 51406, finalColumn := 1004708 }
, { child := 13, logicalColumn := 65, sourceArmColumn := 51411, finalColumn := 1004713 }
, { child := 13, logicalColumn := 66, sourceArmColumn := 51416, finalColumn := 1004718 }
, { child := 13, logicalColumn := 67, sourceArmColumn := 51421, finalColumn := 1004723 }
, { child := 13, logicalColumn := 68, sourceArmColumn := 51426, finalColumn := 1004728 }
, { child := 13, logicalColumn := 69, sourceArmColumn := 51431, finalColumn := 1004733 }
, { child := 13, logicalColumn := 70, sourceArmColumn := 51436, finalColumn := 1004738 }
, { child := 13, logicalColumn := 71, sourceArmColumn := 51441, finalColumn := 1004743 }
, { child := 13, logicalColumn := 72, sourceArmColumn := 51446, finalColumn := 1004748 }
, { child := 13, logicalColumn := 73, sourceArmColumn := 51451, finalColumn := 1004753 }
, { child := 13, logicalColumn := 74, sourceArmColumn := 51456, finalColumn := 1004758 }
, { child := 13, logicalColumn := 75, sourceArmColumn := 51461, finalColumn := 1004763 }
, { child := 13, logicalColumn := 76, sourceArmColumn := 51466, finalColumn := 1004768 }
, { child := 13, logicalColumn := 77, sourceArmColumn := 51471, finalColumn := 1004773 }
, { child := 13, logicalColumn := 78, sourceArmColumn := 51476, finalColumn := 1004778 }
, { child := 13, logicalColumn := 79, sourceArmColumn := 51481, finalColumn := 1004783 }
, { child := 13, logicalColumn := 80, sourceArmColumn := 51486, finalColumn := 1004788 }
, { child := 13, logicalColumn := 81, sourceArmColumn := 51491, finalColumn := 1004793 }
, { child := 13, logicalColumn := 82, sourceArmColumn := 51496, finalColumn := 1004798 }
, { child := 13, logicalColumn := 83, sourceArmColumn := 51501, finalColumn := 1004803 }
, { child := 13, logicalColumn := 84, sourceArmColumn := 51506, finalColumn := 1004808 }
, { child := 13, logicalColumn := 85, sourceArmColumn := 51511, finalColumn := 1004813 }
, { child := 13, logicalColumn := 86, sourceArmColumn := 51516, finalColumn := 1004818 }
, { child := 13, logicalColumn := 87, sourceArmColumn := 51521, finalColumn := 1004823 }
, { child := 13, logicalColumn := 88, sourceArmColumn := 51526, finalColumn := 1004828 }
, { child := 13, logicalColumn := 89, sourceArmColumn := 51531, finalColumn := 1004833 }
, { child := 13, logicalColumn := 90, sourceArmColumn := 51536, finalColumn := 1004838 }
, { child := 13, logicalColumn := 91, sourceArmColumn := 51541, finalColumn := 1004843 }
, { child := 13, logicalColumn := 92, sourceArmColumn := 51546, finalColumn := 1004848 }
, { child := 13, logicalColumn := 93, sourceArmColumn := 51551, finalColumn := 1004853 }
, { child := 13, logicalColumn := 94, sourceArmColumn := 51556, finalColumn := 1004858 }
, { child := 13, logicalColumn := 95, sourceArmColumn := 51561, finalColumn := 1004863 }
, { child := 13, logicalColumn := 96, sourceArmColumn := 51566, finalColumn := 1004868 }
, { child := 13, logicalColumn := 97, sourceArmColumn := 51571, finalColumn := 1004873 }
, { child := 13, logicalColumn := 98, sourceArmColumn := 51576, finalColumn := 1004878 }
, { child := 13, logicalColumn := 99, sourceArmColumn := 51581, finalColumn := 1004883 }
, { child := 13, logicalColumn := 100, sourceArmColumn := 51586, finalColumn := 1004888 }
, { child := 13, logicalColumn := 101, sourceArmColumn := 51591, finalColumn := 1004893 }
, { child := 13, logicalColumn := 102, sourceArmColumn := 51596, finalColumn := 1004898 }
, { child := 13, logicalColumn := 103, sourceArmColumn := 51601, finalColumn := 1004903 }
, { child := 13, logicalColumn := 104, sourceArmColumn := 51606, finalColumn := 1004908 }
, { child := 13, logicalColumn := 105, sourceArmColumn := 51611, finalColumn := 1004913 }
, { child := 13, logicalColumn := 106, sourceArmColumn := 51616, finalColumn := 1004918 }
, { child := 13, logicalColumn := 107, sourceArmColumn := 51621, finalColumn := 1004923 }
, { child := 13, logicalColumn := 108, sourceArmColumn := 51357, finalColumn := 1004659 }
, { child := 13, logicalColumn := 109, sourceArmColumn := 51362, finalColumn := 1004664 }
, { child := 13, logicalColumn := 110, sourceArmColumn := 51367, finalColumn := 1004669 }
, { child := 13, logicalColumn := 111, sourceArmColumn := 51372, finalColumn := 1004674 }
, { child := 13, logicalColumn := 112, sourceArmColumn := 51377, finalColumn := 1004679 }
, { child := 13, logicalColumn := 113, sourceArmColumn := 51382, finalColumn := 1004684 }
, { child := 13, logicalColumn := 114, sourceArmColumn := 51387, finalColumn := 1004689 }
, { child := 13, logicalColumn := 115, sourceArmColumn := 51392, finalColumn := 1004694 }
, { child := 13, logicalColumn := 116, sourceArmColumn := 51397, finalColumn := 1004699 }
, { child := 13, logicalColumn := 117, sourceArmColumn := 51402, finalColumn := 1004704 }
, { child := 13, logicalColumn := 118, sourceArmColumn := 51407, finalColumn := 1004709 }
, { child := 13, logicalColumn := 119, sourceArmColumn := 51412, finalColumn := 1004714 }
, { child := 13, logicalColumn := 120, sourceArmColumn := 51417, finalColumn := 1004719 }
, { child := 13, logicalColumn := 121, sourceArmColumn := 51422, finalColumn := 1004724 }
, { child := 13, logicalColumn := 122, sourceArmColumn := 51427, finalColumn := 1004729 }
, { child := 13, logicalColumn := 123, sourceArmColumn := 51432, finalColumn := 1004734 }
, { child := 13, logicalColumn := 124, sourceArmColumn := 51437, finalColumn := 1004739 }
, { child := 13, logicalColumn := 125, sourceArmColumn := 51442, finalColumn := 1004744 }
, { child := 13, logicalColumn := 126, sourceArmColumn := 51447, finalColumn := 1004749 }
, { child := 13, logicalColumn := 127, sourceArmColumn := 51452, finalColumn := 1004754 }
, { child := 13, logicalColumn := 128, sourceArmColumn := 51457, finalColumn := 1004759 }
, { child := 13, logicalColumn := 129, sourceArmColumn := 51462, finalColumn := 1004764 }
, { child := 13, logicalColumn := 130, sourceArmColumn := 51467, finalColumn := 1004769 }
, { child := 13, logicalColumn := 131, sourceArmColumn := 51472, finalColumn := 1004774 }
, { child := 13, logicalColumn := 132, sourceArmColumn := 51477, finalColumn := 1004779 }
, { child := 13, logicalColumn := 133, sourceArmColumn := 51482, finalColumn := 1004784 }
, { child := 13, logicalColumn := 134, sourceArmColumn := 51487, finalColumn := 1004789 }
, { child := 13, logicalColumn := 135, sourceArmColumn := 51492, finalColumn := 1004794 }
, { child := 13, logicalColumn := 136, sourceArmColumn := 51497, finalColumn := 1004799 }
, { child := 13, logicalColumn := 137, sourceArmColumn := 51502, finalColumn := 1004804 }
, { child := 13, logicalColumn := 138, sourceArmColumn := 51507, finalColumn := 1004809 }
, { child := 13, logicalColumn := 139, sourceArmColumn := 51512, finalColumn := 1004814 }
, { child := 13, logicalColumn := 140, sourceArmColumn := 51517, finalColumn := 1004819 }
, { child := 13, logicalColumn := 141, sourceArmColumn := 51522, finalColumn := 1004824 }
, { child := 13, logicalColumn := 142, sourceArmColumn := 51527, finalColumn := 1004829 }
, { child := 13, logicalColumn := 143, sourceArmColumn := 51532, finalColumn := 1004834 }
, { child := 13, logicalColumn := 144, sourceArmColumn := 51537, finalColumn := 1004839 }
, { child := 13, logicalColumn := 145, sourceArmColumn := 51542, finalColumn := 1004844 }
, { child := 13, logicalColumn := 146, sourceArmColumn := 51547, finalColumn := 1004849 }
, { child := 13, logicalColumn := 147, sourceArmColumn := 51552, finalColumn := 1004854 }
, { child := 13, logicalColumn := 148, sourceArmColumn := 51557, finalColumn := 1004859 }
, { child := 13, logicalColumn := 149, sourceArmColumn := 51562, finalColumn := 1004864 }
, { child := 13, logicalColumn := 150, sourceArmColumn := 51567, finalColumn := 1004869 }
, { child := 13, logicalColumn := 151, sourceArmColumn := 51572, finalColumn := 1004874 }
, { child := 13, logicalColumn := 152, sourceArmColumn := 51577, finalColumn := 1004879 }
, { child := 13, logicalColumn := 153, sourceArmColumn := 51582, finalColumn := 1004884 }
, { child := 13, logicalColumn := 154, sourceArmColumn := 51587, finalColumn := 1004889 }
, { child := 13, logicalColumn := 155, sourceArmColumn := 51592, finalColumn := 1004894 }
, { child := 13, logicalColumn := 156, sourceArmColumn := 51597, finalColumn := 1004899 }
, { child := 13, logicalColumn := 157, sourceArmColumn := 51602, finalColumn := 1004904 }
, { child := 13, logicalColumn := 158, sourceArmColumn := 51607, finalColumn := 1004909 }
, { child := 13, logicalColumn := 159, sourceArmColumn := 51612, finalColumn := 1004914 }
, { child := 13, logicalColumn := 160, sourceArmColumn := 51617, finalColumn := 1004919 }
, { child := 13, logicalColumn := 161, sourceArmColumn := 51622, finalColumn := 1004924 }
, { child := 13, logicalColumn := 162, sourceArmColumn := 51358, finalColumn := 1004660 }
, { child := 13, logicalColumn := 163, sourceArmColumn := 51363, finalColumn := 1004665 }
, { child := 13, logicalColumn := 164, sourceArmColumn := 51368, finalColumn := 1004670 }
, { child := 13, logicalColumn := 165, sourceArmColumn := 51373, finalColumn := 1004675 }
, { child := 13, logicalColumn := 166, sourceArmColumn := 51378, finalColumn := 1004680 }
, { child := 13, logicalColumn := 167, sourceArmColumn := 51383, finalColumn := 1004685 }
, { child := 13, logicalColumn := 168, sourceArmColumn := 51388, finalColumn := 1004690 }
, { child := 13, logicalColumn := 169, sourceArmColumn := 51393, finalColumn := 1004695 }
, { child := 13, logicalColumn := 170, sourceArmColumn := 51398, finalColumn := 1004700 }
, { child := 13, logicalColumn := 171, sourceArmColumn := 51403, finalColumn := 1004705 }
, { child := 13, logicalColumn := 172, sourceArmColumn := 51408, finalColumn := 1004710 }
, { child := 13, logicalColumn := 173, sourceArmColumn := 51413, finalColumn := 1004715 }
, { child := 13, logicalColumn := 174, sourceArmColumn := 51418, finalColumn := 1004720 }
, { child := 13, logicalColumn := 175, sourceArmColumn := 51423, finalColumn := 1004725 }
, { child := 13, logicalColumn := 176, sourceArmColumn := 51428, finalColumn := 1004730 }
, { child := 13, logicalColumn := 177, sourceArmColumn := 51433, finalColumn := 1004735 }
, { child := 13, logicalColumn := 178, sourceArmColumn := 51438, finalColumn := 1004740 }
, { child := 13, logicalColumn := 179, sourceArmColumn := 51443, finalColumn := 1004745 }
, { child := 13, logicalColumn := 180, sourceArmColumn := 51448, finalColumn := 1004750 }
, { child := 13, logicalColumn := 181, sourceArmColumn := 51453, finalColumn := 1004755 }
, { child := 13, logicalColumn := 182, sourceArmColumn := 51458, finalColumn := 1004760 }
, { child := 13, logicalColumn := 183, sourceArmColumn := 51463, finalColumn := 1004765 }
, { child := 13, logicalColumn := 184, sourceArmColumn := 51468, finalColumn := 1004770 }
, { child := 13, logicalColumn := 185, sourceArmColumn := 51473, finalColumn := 1004775 }
, { child := 13, logicalColumn := 186, sourceArmColumn := 51478, finalColumn := 1004780 }
, { child := 13, logicalColumn := 187, sourceArmColumn := 51483, finalColumn := 1004785 }
, { child := 13, logicalColumn := 188, sourceArmColumn := 51488, finalColumn := 1004790 }
, { child := 13, logicalColumn := 189, sourceArmColumn := 51493, finalColumn := 1004795 }
, { child := 13, logicalColumn := 190, sourceArmColumn := 51498, finalColumn := 1004800 }
, { child := 13, logicalColumn := 191, sourceArmColumn := 51503, finalColumn := 1004805 }
, { child := 13, logicalColumn := 192, sourceArmColumn := 51508, finalColumn := 1004810 }
, { child := 13, logicalColumn := 193, sourceArmColumn := 51513, finalColumn := 1004815 }
, { child := 13, logicalColumn := 194, sourceArmColumn := 51518, finalColumn := 1004820 }
, { child := 13, logicalColumn := 195, sourceArmColumn := 51523, finalColumn := 1004825 }
, { child := 13, logicalColumn := 196, sourceArmColumn := 51528, finalColumn := 1004830 }
, { child := 13, logicalColumn := 197, sourceArmColumn := 51533, finalColumn := 1004835 }
, { child := 13, logicalColumn := 198, sourceArmColumn := 51538, finalColumn := 1004840 }
, { child := 13, logicalColumn := 199, sourceArmColumn := 51543, finalColumn := 1004845 }
, { child := 13, logicalColumn := 200, sourceArmColumn := 51548, finalColumn := 1004850 }
, { child := 13, logicalColumn := 201, sourceArmColumn := 51553, finalColumn := 1004855 }
, { child := 13, logicalColumn := 202, sourceArmColumn := 51558, finalColumn := 1004860 }
, { child := 13, logicalColumn := 203, sourceArmColumn := 51563, finalColumn := 1004865 }
, { child := 13, logicalColumn := 204, sourceArmColumn := 51568, finalColumn := 1004870 }
, { child := 13, logicalColumn := 205, sourceArmColumn := 51573, finalColumn := 1004875 }
, { child := 13, logicalColumn := 206, sourceArmColumn := 51578, finalColumn := 1004880 }
, { child := 13, logicalColumn := 207, sourceArmColumn := 51583, finalColumn := 1004885 }
, { child := 13, logicalColumn := 208, sourceArmColumn := 51588, finalColumn := 1004890 }
, { child := 13, logicalColumn := 209, sourceArmColumn := 51593, finalColumn := 1004895 }
, { child := 13, logicalColumn := 210, sourceArmColumn := 51598, finalColumn := 1004900 }
, { child := 13, logicalColumn := 211, sourceArmColumn := 51603, finalColumn := 1004905 }
, { child := 13, logicalColumn := 212, sourceArmColumn := 51608, finalColumn := 1004910 }
, { child := 13, logicalColumn := 213, sourceArmColumn := 51613, finalColumn := 1004915 }
, { child := 13, logicalColumn := 214, sourceArmColumn := 51618, finalColumn := 1004920 }
, { child := 13, logicalColumn := 215, sourceArmColumn := 51623, finalColumn := 1004925 }
, { child := 13, logicalColumn := 216, sourceArmColumn := 51359, finalColumn := 1004661 }
, { child := 13, logicalColumn := 217, sourceArmColumn := 51364, finalColumn := 1004666 }
, { child := 13, logicalColumn := 218, sourceArmColumn := 51369, finalColumn := 1004671 }
, { child := 13, logicalColumn := 219, sourceArmColumn := 51374, finalColumn := 1004676 }
, { child := 13, logicalColumn := 220, sourceArmColumn := 51379, finalColumn := 1004681 }
, { child := 13, logicalColumn := 221, sourceArmColumn := 51384, finalColumn := 1004686 }
, { child := 13, logicalColumn := 222, sourceArmColumn := 51389, finalColumn := 1004691 }
, { child := 13, logicalColumn := 223, sourceArmColumn := 51394, finalColumn := 1004696 }
, { child := 13, logicalColumn := 224, sourceArmColumn := 51399, finalColumn := 1004701 }
, { child := 13, logicalColumn := 225, sourceArmColumn := 51404, finalColumn := 1004706 }
, { child := 13, logicalColumn := 226, sourceArmColumn := 51409, finalColumn := 1004711 }
, { child := 13, logicalColumn := 227, sourceArmColumn := 51414, finalColumn := 1004716 }
, { child := 13, logicalColumn := 228, sourceArmColumn := 51419, finalColumn := 1004721 }
, { child := 13, logicalColumn := 229, sourceArmColumn := 51424, finalColumn := 1004726 }
, { child := 13, logicalColumn := 230, sourceArmColumn := 51429, finalColumn := 1004731 }
, { child := 13, logicalColumn := 231, sourceArmColumn := 51434, finalColumn := 1004736 }
, { child := 13, logicalColumn := 232, sourceArmColumn := 51439, finalColumn := 1004741 }
, { child := 13, logicalColumn := 233, sourceArmColumn := 51444, finalColumn := 1004746 }
, { child := 13, logicalColumn := 234, sourceArmColumn := 51449, finalColumn := 1004751 }
, { child := 13, logicalColumn := 235, sourceArmColumn := 51454, finalColumn := 1004756 }
, { child := 13, logicalColumn := 236, sourceArmColumn := 51459, finalColumn := 1004761 }
, { child := 13, logicalColumn := 237, sourceArmColumn := 51464, finalColumn := 1004766 }
, { child := 13, logicalColumn := 238, sourceArmColumn := 51469, finalColumn := 1004771 }
, { child := 13, logicalColumn := 239, sourceArmColumn := 51474, finalColumn := 1004776 }
, { child := 13, logicalColumn := 240, sourceArmColumn := 51479, finalColumn := 1004781 }
, { child := 13, logicalColumn := 241, sourceArmColumn := 51484, finalColumn := 1004786 }
, { child := 13, logicalColumn := 242, sourceArmColumn := 51489, finalColumn := 1004791 }
, { child := 13, logicalColumn := 243, sourceArmColumn := 51494, finalColumn := 1004796 }
, { child := 13, logicalColumn := 244, sourceArmColumn := 51499, finalColumn := 1004801 }
, { child := 13, logicalColumn := 245, sourceArmColumn := 51504, finalColumn := 1004806 }
, { child := 13, logicalColumn := 246, sourceArmColumn := 51509, finalColumn := 1004811 }
, { child := 13, logicalColumn := 247, sourceArmColumn := 51514, finalColumn := 1004816 }
, { child := 13, logicalColumn := 248, sourceArmColumn := 51519, finalColumn := 1004821 }
, { child := 13, logicalColumn := 249, sourceArmColumn := 51524, finalColumn := 1004826 }
, { child := 13, logicalColumn := 250, sourceArmColumn := 51529, finalColumn := 1004831 }
, { child := 13, logicalColumn := 251, sourceArmColumn := 51534, finalColumn := 1004836 }
, { child := 13, logicalColumn := 252, sourceArmColumn := 51539, finalColumn := 1004841 }
, { child := 13, logicalColumn := 253, sourceArmColumn := 51544, finalColumn := 1004846 }
, { child := 13, logicalColumn := 254, sourceArmColumn := 51549, finalColumn := 1004851 }
, { child := 13, logicalColumn := 255, sourceArmColumn := 51554, finalColumn := 1004856 }
, { child := 13, logicalColumn := 256, sourceArmColumn := 51559, finalColumn := 1004861 }
, { child := 13, logicalColumn := 257, sourceArmColumn := 51564, finalColumn := 1004866 }
, { child := 13, logicalColumn := 258, sourceArmColumn := 51569, finalColumn := 1004871 }
, { child := 13, logicalColumn := 259, sourceArmColumn := 51574, finalColumn := 1004876 }
, { child := 13, logicalColumn := 260, sourceArmColumn := 51579, finalColumn := 1004881 }
, { child := 13, logicalColumn := 261, sourceArmColumn := 51584, finalColumn := 1004886 }
, { child := 13, logicalColumn := 262, sourceArmColumn := 51589, finalColumn := 1004891 }
, { child := 13, logicalColumn := 263, sourceArmColumn := 51594, finalColumn := 1004896 }
, { child := 13, logicalColumn := 264, sourceArmColumn := 51599, finalColumn := 1004901 }
, { child := 13, logicalColumn := 265, sourceArmColumn := 51604, finalColumn := 1004906 }
, { child := 13, logicalColumn := 266, sourceArmColumn := 51609, finalColumn := 1004911 }
, { child := 13, logicalColumn := 267, sourceArmColumn := 51614, finalColumn := 1004916 }
, { child := 13, logicalColumn := 268, sourceArmColumn := 51619, finalColumn := 1004921 }
, { child := 13, logicalColumn := 269, sourceArmColumn := 51624, finalColumn := 1004926 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk14
