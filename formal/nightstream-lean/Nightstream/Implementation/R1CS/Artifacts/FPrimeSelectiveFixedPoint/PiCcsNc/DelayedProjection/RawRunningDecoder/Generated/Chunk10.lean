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

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk10

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def childCount : Nat := 14
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def allocationRecords : List AllocationRecord := [
  { child := 9, logicalColumn := 90, sourceArmColumn := 42448, finalColumn := 738078 }
, { child := 9, logicalColumn := 91, sourceArmColumn := 42453, finalColumn := 738083 }
, { child := 9, logicalColumn := 92, sourceArmColumn := 42458, finalColumn := 738088 }
, { child := 9, logicalColumn := 93, sourceArmColumn := 42463, finalColumn := 738093 }
, { child := 9, logicalColumn := 94, sourceArmColumn := 42468, finalColumn := 738098 }
, { child := 9, logicalColumn := 95, sourceArmColumn := 42473, finalColumn := 738103 }
, { child := 9, logicalColumn := 96, sourceArmColumn := 42478, finalColumn := 738108 }
, { child := 9, logicalColumn := 97, sourceArmColumn := 42483, finalColumn := 738113 }
, { child := 9, logicalColumn := 98, sourceArmColumn := 42488, finalColumn := 738118 }
, { child := 9, logicalColumn := 99, sourceArmColumn := 42493, finalColumn := 738123 }
, { child := 9, logicalColumn := 100, sourceArmColumn := 42498, finalColumn := 738128 }
, { child := 9, logicalColumn := 101, sourceArmColumn := 42503, finalColumn := 738133 }
, { child := 9, logicalColumn := 102, sourceArmColumn := 42508, finalColumn := 738138 }
, { child := 9, logicalColumn := 103, sourceArmColumn := 42513, finalColumn := 738143 }
, { child := 9, logicalColumn := 104, sourceArmColumn := 42518, finalColumn := 738148 }
, { child := 9, logicalColumn := 105, sourceArmColumn := 42523, finalColumn := 738153 }
, { child := 9, logicalColumn := 106, sourceArmColumn := 42528, finalColumn := 738158 }
, { child := 9, logicalColumn := 107, sourceArmColumn := 42533, finalColumn := 738163 }
, { child := 9, logicalColumn := 108, sourceArmColumn := 42269, finalColumn := 737899 }
, { child := 9, logicalColumn := 109, sourceArmColumn := 42274, finalColumn := 737904 }
, { child := 9, logicalColumn := 110, sourceArmColumn := 42279, finalColumn := 737909 }
, { child := 9, logicalColumn := 111, sourceArmColumn := 42284, finalColumn := 737914 }
, { child := 9, logicalColumn := 112, sourceArmColumn := 42289, finalColumn := 737919 }
, { child := 9, logicalColumn := 113, sourceArmColumn := 42294, finalColumn := 737924 }
, { child := 9, logicalColumn := 114, sourceArmColumn := 42299, finalColumn := 737929 }
, { child := 9, logicalColumn := 115, sourceArmColumn := 42304, finalColumn := 737934 }
, { child := 9, logicalColumn := 116, sourceArmColumn := 42309, finalColumn := 737939 }
, { child := 9, logicalColumn := 117, sourceArmColumn := 42314, finalColumn := 737944 }
, { child := 9, logicalColumn := 118, sourceArmColumn := 42319, finalColumn := 737949 }
, { child := 9, logicalColumn := 119, sourceArmColumn := 42324, finalColumn := 737954 }
, { child := 9, logicalColumn := 120, sourceArmColumn := 42329, finalColumn := 737959 }
, { child := 9, logicalColumn := 121, sourceArmColumn := 42334, finalColumn := 737964 }
, { child := 9, logicalColumn := 122, sourceArmColumn := 42339, finalColumn := 737969 }
, { child := 9, logicalColumn := 123, sourceArmColumn := 42344, finalColumn := 737974 }
, { child := 9, logicalColumn := 124, sourceArmColumn := 42349, finalColumn := 737979 }
, { child := 9, logicalColumn := 125, sourceArmColumn := 42354, finalColumn := 737984 }
, { child := 9, logicalColumn := 126, sourceArmColumn := 42359, finalColumn := 737989 }
, { child := 9, logicalColumn := 127, sourceArmColumn := 42364, finalColumn := 737994 }
, { child := 9, logicalColumn := 128, sourceArmColumn := 42369, finalColumn := 737999 }
, { child := 9, logicalColumn := 129, sourceArmColumn := 42374, finalColumn := 738004 }
, { child := 9, logicalColumn := 130, sourceArmColumn := 42379, finalColumn := 738009 }
, { child := 9, logicalColumn := 131, sourceArmColumn := 42384, finalColumn := 738014 }
, { child := 9, logicalColumn := 132, sourceArmColumn := 42389, finalColumn := 738019 }
, { child := 9, logicalColumn := 133, sourceArmColumn := 42394, finalColumn := 738024 }
, { child := 9, logicalColumn := 134, sourceArmColumn := 42399, finalColumn := 738029 }
, { child := 9, logicalColumn := 135, sourceArmColumn := 42404, finalColumn := 738034 }
, { child := 9, logicalColumn := 136, sourceArmColumn := 42409, finalColumn := 738039 }
, { child := 9, logicalColumn := 137, sourceArmColumn := 42414, finalColumn := 738044 }
, { child := 9, logicalColumn := 138, sourceArmColumn := 42419, finalColumn := 738049 }
, { child := 9, logicalColumn := 139, sourceArmColumn := 42424, finalColumn := 738054 }
, { child := 9, logicalColumn := 140, sourceArmColumn := 42429, finalColumn := 738059 }
, { child := 9, logicalColumn := 141, sourceArmColumn := 42434, finalColumn := 738064 }
, { child := 9, logicalColumn := 142, sourceArmColumn := 42439, finalColumn := 738069 }
, { child := 9, logicalColumn := 143, sourceArmColumn := 42444, finalColumn := 738074 }
, { child := 9, logicalColumn := 144, sourceArmColumn := 42449, finalColumn := 738079 }
, { child := 9, logicalColumn := 145, sourceArmColumn := 42454, finalColumn := 738084 }
, { child := 9, logicalColumn := 146, sourceArmColumn := 42459, finalColumn := 738089 }
, { child := 9, logicalColumn := 147, sourceArmColumn := 42464, finalColumn := 738094 }
, { child := 9, logicalColumn := 148, sourceArmColumn := 42469, finalColumn := 738099 }
, { child := 9, logicalColumn := 149, sourceArmColumn := 42474, finalColumn := 738104 }
, { child := 9, logicalColumn := 150, sourceArmColumn := 42479, finalColumn := 738109 }
, { child := 9, logicalColumn := 151, sourceArmColumn := 42484, finalColumn := 738114 }
, { child := 9, logicalColumn := 152, sourceArmColumn := 42489, finalColumn := 738119 }
, { child := 9, logicalColumn := 153, sourceArmColumn := 42494, finalColumn := 738124 }
, { child := 9, logicalColumn := 154, sourceArmColumn := 42499, finalColumn := 738129 }
, { child := 9, logicalColumn := 155, sourceArmColumn := 42504, finalColumn := 738134 }
, { child := 9, logicalColumn := 156, sourceArmColumn := 42509, finalColumn := 738139 }
, { child := 9, logicalColumn := 157, sourceArmColumn := 42514, finalColumn := 738144 }
, { child := 9, logicalColumn := 158, sourceArmColumn := 42519, finalColumn := 738149 }
, { child := 9, logicalColumn := 159, sourceArmColumn := 42524, finalColumn := 738154 }
, { child := 9, logicalColumn := 160, sourceArmColumn := 42529, finalColumn := 738159 }
, { child := 9, logicalColumn := 161, sourceArmColumn := 42534, finalColumn := 738164 }
, { child := 9, logicalColumn := 162, sourceArmColumn := 42270, finalColumn := 737900 }
, { child := 9, logicalColumn := 163, sourceArmColumn := 42275, finalColumn := 737905 }
, { child := 9, logicalColumn := 164, sourceArmColumn := 42280, finalColumn := 737910 }
, { child := 9, logicalColumn := 165, sourceArmColumn := 42285, finalColumn := 737915 }
, { child := 9, logicalColumn := 166, sourceArmColumn := 42290, finalColumn := 737920 }
, { child := 9, logicalColumn := 167, sourceArmColumn := 42295, finalColumn := 737925 }
, { child := 9, logicalColumn := 168, sourceArmColumn := 42300, finalColumn := 737930 }
, { child := 9, logicalColumn := 169, sourceArmColumn := 42305, finalColumn := 737935 }
, { child := 9, logicalColumn := 170, sourceArmColumn := 42310, finalColumn := 737940 }
, { child := 9, logicalColumn := 171, sourceArmColumn := 42315, finalColumn := 737945 }
, { child := 9, logicalColumn := 172, sourceArmColumn := 42320, finalColumn := 737950 }
, { child := 9, logicalColumn := 173, sourceArmColumn := 42325, finalColumn := 737955 }
, { child := 9, logicalColumn := 174, sourceArmColumn := 42330, finalColumn := 737960 }
, { child := 9, logicalColumn := 175, sourceArmColumn := 42335, finalColumn := 737965 }
, { child := 9, logicalColumn := 176, sourceArmColumn := 42340, finalColumn := 737970 }
, { child := 9, logicalColumn := 177, sourceArmColumn := 42345, finalColumn := 737975 }
, { child := 9, logicalColumn := 178, sourceArmColumn := 42350, finalColumn := 737980 }
, { child := 9, logicalColumn := 179, sourceArmColumn := 42355, finalColumn := 737985 }
, { child := 9, logicalColumn := 180, sourceArmColumn := 42360, finalColumn := 737990 }
, { child := 9, logicalColumn := 181, sourceArmColumn := 42365, finalColumn := 737995 }
, { child := 9, logicalColumn := 182, sourceArmColumn := 42370, finalColumn := 738000 }
, { child := 9, logicalColumn := 183, sourceArmColumn := 42375, finalColumn := 738005 }
, { child := 9, logicalColumn := 184, sourceArmColumn := 42380, finalColumn := 738010 }
, { child := 9, logicalColumn := 185, sourceArmColumn := 42385, finalColumn := 738015 }
, { child := 9, logicalColumn := 186, sourceArmColumn := 42390, finalColumn := 738020 }
, { child := 9, logicalColumn := 187, sourceArmColumn := 42395, finalColumn := 738025 }
, { child := 9, logicalColumn := 188, sourceArmColumn := 42400, finalColumn := 738030 }
, { child := 9, logicalColumn := 189, sourceArmColumn := 42405, finalColumn := 738035 }
, { child := 9, logicalColumn := 190, sourceArmColumn := 42410, finalColumn := 738040 }
, { child := 9, logicalColumn := 191, sourceArmColumn := 42415, finalColumn := 738045 }
, { child := 9, logicalColumn := 192, sourceArmColumn := 42420, finalColumn := 738050 }
, { child := 9, logicalColumn := 193, sourceArmColumn := 42425, finalColumn := 738055 }
, { child := 9, logicalColumn := 194, sourceArmColumn := 42430, finalColumn := 738060 }
, { child := 9, logicalColumn := 195, sourceArmColumn := 42435, finalColumn := 738065 }
, { child := 9, logicalColumn := 196, sourceArmColumn := 42440, finalColumn := 738070 }
, { child := 9, logicalColumn := 197, sourceArmColumn := 42445, finalColumn := 738075 }
, { child := 9, logicalColumn := 198, sourceArmColumn := 42450, finalColumn := 738080 }
, { child := 9, logicalColumn := 199, sourceArmColumn := 42455, finalColumn := 738085 }
, { child := 9, logicalColumn := 200, sourceArmColumn := 42460, finalColumn := 738090 }
, { child := 9, logicalColumn := 201, sourceArmColumn := 42465, finalColumn := 738095 }
, { child := 9, logicalColumn := 202, sourceArmColumn := 42470, finalColumn := 738100 }
, { child := 9, logicalColumn := 203, sourceArmColumn := 42475, finalColumn := 738105 }
, { child := 9, logicalColumn := 204, sourceArmColumn := 42480, finalColumn := 738110 }
, { child := 9, logicalColumn := 205, sourceArmColumn := 42485, finalColumn := 738115 }
, { child := 9, logicalColumn := 206, sourceArmColumn := 42490, finalColumn := 738120 }
, { child := 9, logicalColumn := 207, sourceArmColumn := 42495, finalColumn := 738125 }
, { child := 9, logicalColumn := 208, sourceArmColumn := 42500, finalColumn := 738130 }
, { child := 9, logicalColumn := 209, sourceArmColumn := 42505, finalColumn := 738135 }
, { child := 9, logicalColumn := 210, sourceArmColumn := 42510, finalColumn := 738140 }
, { child := 9, logicalColumn := 211, sourceArmColumn := 42515, finalColumn := 738145 }
, { child := 9, logicalColumn := 212, sourceArmColumn := 42520, finalColumn := 738150 }
, { child := 9, logicalColumn := 213, sourceArmColumn := 42525, finalColumn := 738155 }
, { child := 9, logicalColumn := 214, sourceArmColumn := 42530, finalColumn := 738160 }
, { child := 9, logicalColumn := 215, sourceArmColumn := 42535, finalColumn := 738165 }
, { child := 9, logicalColumn := 216, sourceArmColumn := 42271, finalColumn := 737901 }
, { child := 9, logicalColumn := 217, sourceArmColumn := 42276, finalColumn := 737906 }
, { child := 9, logicalColumn := 218, sourceArmColumn := 42281, finalColumn := 737911 }
, { child := 9, logicalColumn := 219, sourceArmColumn := 42286, finalColumn := 737916 }
, { child := 9, logicalColumn := 220, sourceArmColumn := 42291, finalColumn := 737921 }
, { child := 9, logicalColumn := 221, sourceArmColumn := 42296, finalColumn := 737926 }
, { child := 9, logicalColumn := 222, sourceArmColumn := 42301, finalColumn := 737931 }
, { child := 9, logicalColumn := 223, sourceArmColumn := 42306, finalColumn := 737936 }
, { child := 9, logicalColumn := 224, sourceArmColumn := 42311, finalColumn := 737941 }
, { child := 9, logicalColumn := 225, sourceArmColumn := 42316, finalColumn := 737946 }
, { child := 9, logicalColumn := 226, sourceArmColumn := 42321, finalColumn := 737951 }
, { child := 9, logicalColumn := 227, sourceArmColumn := 42326, finalColumn := 737956 }
, { child := 9, logicalColumn := 228, sourceArmColumn := 42331, finalColumn := 737961 }
, { child := 9, logicalColumn := 229, sourceArmColumn := 42336, finalColumn := 737966 }
, { child := 9, logicalColumn := 230, sourceArmColumn := 42341, finalColumn := 737971 }
, { child := 9, logicalColumn := 231, sourceArmColumn := 42346, finalColumn := 737976 }
, { child := 9, logicalColumn := 232, sourceArmColumn := 42351, finalColumn := 737981 }
, { child := 9, logicalColumn := 233, sourceArmColumn := 42356, finalColumn := 737986 }
, { child := 9, logicalColumn := 234, sourceArmColumn := 42361, finalColumn := 737991 }
, { child := 9, logicalColumn := 235, sourceArmColumn := 42366, finalColumn := 737996 }
, { child := 9, logicalColumn := 236, sourceArmColumn := 42371, finalColumn := 738001 }
, { child := 9, logicalColumn := 237, sourceArmColumn := 42376, finalColumn := 738006 }
, { child := 9, logicalColumn := 238, sourceArmColumn := 42381, finalColumn := 738011 }
, { child := 9, logicalColumn := 239, sourceArmColumn := 42386, finalColumn := 738016 }
, { child := 9, logicalColumn := 240, sourceArmColumn := 42391, finalColumn := 738021 }
, { child := 9, logicalColumn := 241, sourceArmColumn := 42396, finalColumn := 738026 }
, { child := 9, logicalColumn := 242, sourceArmColumn := 42401, finalColumn := 738031 }
, { child := 9, logicalColumn := 243, sourceArmColumn := 42406, finalColumn := 738036 }
, { child := 9, logicalColumn := 244, sourceArmColumn := 42411, finalColumn := 738041 }
, { child := 9, logicalColumn := 245, sourceArmColumn := 42416, finalColumn := 738046 }
, { child := 9, logicalColumn := 246, sourceArmColumn := 42421, finalColumn := 738051 }
, { child := 9, logicalColumn := 247, sourceArmColumn := 42426, finalColumn := 738056 }
, { child := 9, logicalColumn := 248, sourceArmColumn := 42431, finalColumn := 738061 }
, { child := 9, logicalColumn := 249, sourceArmColumn := 42436, finalColumn := 738066 }
, { child := 9, logicalColumn := 250, sourceArmColumn := 42441, finalColumn := 738071 }
, { child := 9, logicalColumn := 251, sourceArmColumn := 42446, finalColumn := 738076 }
, { child := 9, logicalColumn := 252, sourceArmColumn := 42451, finalColumn := 738081 }
, { child := 9, logicalColumn := 253, sourceArmColumn := 42456, finalColumn := 738086 }
, { child := 9, logicalColumn := 254, sourceArmColumn := 42461, finalColumn := 738091 }
, { child := 9, logicalColumn := 255, sourceArmColumn := 42466, finalColumn := 738096 }
, { child := 9, logicalColumn := 256, sourceArmColumn := 42471, finalColumn := 738101 }
, { child := 9, logicalColumn := 257, sourceArmColumn := 42476, finalColumn := 738106 }
, { child := 9, logicalColumn := 258, sourceArmColumn := 42481, finalColumn := 738111 }
, { child := 9, logicalColumn := 259, sourceArmColumn := 42486, finalColumn := 738116 }
, { child := 9, logicalColumn := 260, sourceArmColumn := 42491, finalColumn := 738121 }
, { child := 9, logicalColumn := 261, sourceArmColumn := 42496, finalColumn := 738126 }
, { child := 9, logicalColumn := 262, sourceArmColumn := 42501, finalColumn := 738131 }
, { child := 9, logicalColumn := 263, sourceArmColumn := 42506, finalColumn := 738136 }
, { child := 9, logicalColumn := 264, sourceArmColumn := 42511, finalColumn := 738141 }
, { child := 9, logicalColumn := 265, sourceArmColumn := 42516, finalColumn := 738146 }
, { child := 9, logicalColumn := 266, sourceArmColumn := 42521, finalColumn := 738151 }
, { child := 9, logicalColumn := 267, sourceArmColumn := 42526, finalColumn := 738156 }
, { child := 9, logicalColumn := 268, sourceArmColumn := 42531, finalColumn := 738161 }
, { child := 9, logicalColumn := 269, sourceArmColumn := 42536, finalColumn := 738166 }
, { child := 10, logicalColumn := 0, sourceArmColumn := 44539, finalColumn := 804587 }
, { child := 10, logicalColumn := 1, sourceArmColumn := 44544, finalColumn := 804592 }
, { child := 10, logicalColumn := 2, sourceArmColumn := 44549, finalColumn := 804597 }
, { child := 10, logicalColumn := 3, sourceArmColumn := 44554, finalColumn := 804602 }
, { child := 10, logicalColumn := 4, sourceArmColumn := 44559, finalColumn := 804607 }
, { child := 10, logicalColumn := 5, sourceArmColumn := 44564, finalColumn := 804612 }
, { child := 10, logicalColumn := 6, sourceArmColumn := 44569, finalColumn := 804617 }
, { child := 10, logicalColumn := 7, sourceArmColumn := 44574, finalColumn := 804622 }
, { child := 10, logicalColumn := 8, sourceArmColumn := 44579, finalColumn := 804627 }
, { child := 10, logicalColumn := 9, sourceArmColumn := 44584, finalColumn := 804632 }
, { child := 10, logicalColumn := 10, sourceArmColumn := 44589, finalColumn := 804637 }
, { child := 10, logicalColumn := 11, sourceArmColumn := 44594, finalColumn := 804642 }
, { child := 10, logicalColumn := 12, sourceArmColumn := 44599, finalColumn := 804647 }
, { child := 10, logicalColumn := 13, sourceArmColumn := 44604, finalColumn := 804652 }
, { child := 10, logicalColumn := 14, sourceArmColumn := 44609, finalColumn := 804657 }
, { child := 10, logicalColumn := 15, sourceArmColumn := 44614, finalColumn := 804662 }
, { child := 10, logicalColumn := 16, sourceArmColumn := 44619, finalColumn := 804667 }
, { child := 10, logicalColumn := 17, sourceArmColumn := 44624, finalColumn := 804672 }
, { child := 10, logicalColumn := 18, sourceArmColumn := 44629, finalColumn := 804677 }
, { child := 10, logicalColumn := 19, sourceArmColumn := 44634, finalColumn := 804682 }
, { child := 10, logicalColumn := 20, sourceArmColumn := 44639, finalColumn := 804687 }
, { child := 10, logicalColumn := 21, sourceArmColumn := 44644, finalColumn := 804692 }
, { child := 10, logicalColumn := 22, sourceArmColumn := 44649, finalColumn := 804697 }
, { child := 10, logicalColumn := 23, sourceArmColumn := 44654, finalColumn := 804702 }
, { child := 10, logicalColumn := 24, sourceArmColumn := 44659, finalColumn := 804707 }
, { child := 10, logicalColumn := 25, sourceArmColumn := 44664, finalColumn := 804712 }
, { child := 10, logicalColumn := 26, sourceArmColumn := 44669, finalColumn := 804717 }
, { child := 10, logicalColumn := 27, sourceArmColumn := 44674, finalColumn := 804722 }
, { child := 10, logicalColumn := 28, sourceArmColumn := 44679, finalColumn := 804727 }
, { child := 10, logicalColumn := 29, sourceArmColumn := 44684, finalColumn := 804732 }
, { child := 10, logicalColumn := 30, sourceArmColumn := 44689, finalColumn := 804737 }
, { child := 10, logicalColumn := 31, sourceArmColumn := 44694, finalColumn := 804742 }
, { child := 10, logicalColumn := 32, sourceArmColumn := 44699, finalColumn := 804747 }
, { child := 10, logicalColumn := 33, sourceArmColumn := 44704, finalColumn := 804752 }
, { child := 10, logicalColumn := 34, sourceArmColumn := 44709, finalColumn := 804757 }
, { child := 10, logicalColumn := 35, sourceArmColumn := 44714, finalColumn := 804762 }
, { child := 10, logicalColumn := 36, sourceArmColumn := 44719, finalColumn := 804767 }
, { child := 10, logicalColumn := 37, sourceArmColumn := 44724, finalColumn := 804772 }
, { child := 10, logicalColumn := 38, sourceArmColumn := 44729, finalColumn := 804777 }
, { child := 10, logicalColumn := 39, sourceArmColumn := 44734, finalColumn := 804782 }
, { child := 10, logicalColumn := 40, sourceArmColumn := 44739, finalColumn := 804787 }
, { child := 10, logicalColumn := 41, sourceArmColumn := 44744, finalColumn := 804792 }
, { child := 10, logicalColumn := 42, sourceArmColumn := 44749, finalColumn := 804797 }
, { child := 10, logicalColumn := 43, sourceArmColumn := 44754, finalColumn := 804802 }
, { child := 10, logicalColumn := 44, sourceArmColumn := 44759, finalColumn := 804807 }
, { child := 10, logicalColumn := 45, sourceArmColumn := 44764, finalColumn := 804812 }
, { child := 10, logicalColumn := 46, sourceArmColumn := 44769, finalColumn := 804817 }
, { child := 10, logicalColumn := 47, sourceArmColumn := 44774, finalColumn := 804822 }
, { child := 10, logicalColumn := 48, sourceArmColumn := 44779, finalColumn := 804827 }
, { child := 10, logicalColumn := 49, sourceArmColumn := 44784, finalColumn := 804832 }
, { child := 10, logicalColumn := 50, sourceArmColumn := 44789, finalColumn := 804837 }
, { child := 10, logicalColumn := 51, sourceArmColumn := 44794, finalColumn := 804842 }
, { child := 10, logicalColumn := 52, sourceArmColumn := 44799, finalColumn := 804847 }
, { child := 10, logicalColumn := 53, sourceArmColumn := 44804, finalColumn := 804852 }
, { child := 10, logicalColumn := 54, sourceArmColumn := 44540, finalColumn := 804588 }
, { child := 10, logicalColumn := 55, sourceArmColumn := 44545, finalColumn := 804593 }
, { child := 10, logicalColumn := 56, sourceArmColumn := 44550, finalColumn := 804598 }
, { child := 10, logicalColumn := 57, sourceArmColumn := 44555, finalColumn := 804603 }
, { child := 10, logicalColumn := 58, sourceArmColumn := 44560, finalColumn := 804608 }
, { child := 10, logicalColumn := 59, sourceArmColumn := 44565, finalColumn := 804613 }
, { child := 10, logicalColumn := 60, sourceArmColumn := 44570, finalColumn := 804618 }
, { child := 10, logicalColumn := 61, sourceArmColumn := 44575, finalColumn := 804623 }
, { child := 10, logicalColumn := 62, sourceArmColumn := 44580, finalColumn := 804628 }
, { child := 10, logicalColumn := 63, sourceArmColumn := 44585, finalColumn := 804633 }
, { child := 10, logicalColumn := 64, sourceArmColumn := 44590, finalColumn := 804638 }
, { child := 10, logicalColumn := 65, sourceArmColumn := 44595, finalColumn := 804643 }
, { child := 10, logicalColumn := 66, sourceArmColumn := 44600, finalColumn := 804648 }
, { child := 10, logicalColumn := 67, sourceArmColumn := 44605, finalColumn := 804653 }
, { child := 10, logicalColumn := 68, sourceArmColumn := 44610, finalColumn := 804658 }
, { child := 10, logicalColumn := 69, sourceArmColumn := 44615, finalColumn := 804663 }
, { child := 10, logicalColumn := 70, sourceArmColumn := 44620, finalColumn := 804668 }
, { child := 10, logicalColumn := 71, sourceArmColumn := 44625, finalColumn := 804673 }
]

def records : List SourceColumnRecord :=
  allocationRecords.map AllocationRecord.sourceRecord

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk10
