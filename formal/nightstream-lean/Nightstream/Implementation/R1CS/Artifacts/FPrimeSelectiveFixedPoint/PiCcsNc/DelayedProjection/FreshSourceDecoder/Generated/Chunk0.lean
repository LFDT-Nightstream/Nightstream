import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Schema

/-!
Generated file: exact fresh public-X source decoder chunk; do not hand-edit.

The records describe only `prior_link.fresh_public_inputs[0]`, coordinates
0 through 269. This is the public-X source prefix consumed by the recursive
step, not the full private witness `Z` and not commitment authority.

The current Rust wire surface does not identify the exact binding row owned by
each coordinate. Consequently this artifact records normalized column and
selective-decoder provenance only; the row-level prior-link bridge remains
open.

Owns: one exact 256-record proof-free decoder shard.

Does not own: source values, full-witness coordinates, per-coordinate binding
rows, commitment binding, or permission to remove constraints.

Emits constraints: none; generated certificate data only.

| Stage path | Mathematical obligation | Authority class | Artifact owner |
|---|---|---|---|
| `pi_ccs.nc.fresh_x.generated.chunk0` | exact ordered source column and fail-closed selective disposition | generated/checked | `fresh_source.rs` |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Generated.Chunk0

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder

def schemaVersion : Nat := 1
def sourceArm : Nat := 2
def sourceCount : Nat := 1
def logicalColumnCount : Nat := 270
def finalColumnCount : Nat := 11725506
def records : List SourceColumnRecord := [
  { logicalColumn := 0, sourceArmColumn := 21547, resolution := .direct 133423 41 false }
, { logicalColumn := 1, sourceArmColumn := 21548, resolution := .equalityAlias 7803 46531 1 false }
, { logicalColumn := 2, sourceArmColumn := 21549, resolution := .equalityAlias 7804 46532 1 false }
, { logicalColumn := 3, sourceArmColumn := 21550, resolution := .equalityAlias 7805 46533 1 false }
, { logicalColumn := 4, sourceArmColumn := 21551, resolution := .equalityAlias 7806 46534 1 false }
, { logicalColumn := 5, sourceArmColumn := 21552, resolution := .equalityAlias 7807 46535 1 false }
, { logicalColumn := 6, sourceArmColumn := 21553, resolution := .equalityAlias 7808 46536 1 false }
, { logicalColumn := 7, sourceArmColumn := 21554, resolution := .equalityAlias 7809 46537 1 false }
, { logicalColumn := 8, sourceArmColumn := 21555, resolution := .equalityAlias 7810 46538 1 false }
, { logicalColumn := 9, sourceArmColumn := 21556, resolution := .equalityAlias 7811 46539 1 false }
, { logicalColumn := 10, sourceArmColumn := 21557, resolution := .equalityAlias 7812 46540 1 false }
, { logicalColumn := 11, sourceArmColumn := 21558, resolution := .equalityAlias 7813 46541 1 false }
, { logicalColumn := 12, sourceArmColumn := 21559, resolution := .equalityAlias 7814 46542 1 false }
, { logicalColumn := 13, sourceArmColumn := 21560, resolution := .equalityAlias 7815 46543 1 false }
, { logicalColumn := 14, sourceArmColumn := 21561, resolution := .equalityAlias 7816 46544 1 false }
, { logicalColumn := 15, sourceArmColumn := 21562, resolution := .equalityAlias 7817 46545 1 false }
, { logicalColumn := 16, sourceArmColumn := 21563, resolution := .equalityAlias 7818 46546 1 false }
, { logicalColumn := 17, sourceArmColumn := 21564, resolution := .equalityAlias 7819 46547 1 false }
, { logicalColumn := 18, sourceArmColumn := 21565, resolution := .equalityAlias 7820 46548 1 false }
, { logicalColumn := 19, sourceArmColumn := 21566, resolution := .equalityAlias 7821 46549 1 false }
, { logicalColumn := 20, sourceArmColumn := 21567, resolution := .equalityAlias 7822 46550 1 false }
, { logicalColumn := 21, sourceArmColumn := 21568, resolution := .equalityAlias 7823 46551 1 false }
, { logicalColumn := 22, sourceArmColumn := 21569, resolution := .equalityAlias 7824 46552 1 false }
, { logicalColumn := 23, sourceArmColumn := 21570, resolution := .equalityAlias 7825 46553 1 false }
, { logicalColumn := 24, sourceArmColumn := 21571, resolution := .equalityAlias 7826 46554 1 false }
, { logicalColumn := 25, sourceArmColumn := 21572, resolution := .equalityAlias 7827 46555 1 false }
, { logicalColumn := 26, sourceArmColumn := 21573, resolution := .equalityAlias 7828 46556 1 false }
, { logicalColumn := 27, sourceArmColumn := 21574, resolution := .equalityAlias 7829 46557 1 false }
, { logicalColumn := 28, sourceArmColumn := 21575, resolution := .equalityAlias 7830 46558 1 false }
, { logicalColumn := 29, sourceArmColumn := 21576, resolution := .equalityAlias 7831 46559 1 false }
, { logicalColumn := 30, sourceArmColumn := 21577, resolution := .equalityAlias 7832 46560 1 false }
, { logicalColumn := 31, sourceArmColumn := 21578, resolution := .equalityAlias 7833 46561 1 false }
, { logicalColumn := 32, sourceArmColumn := 21579, resolution := .equalityAlias 7834 46562 1 false }
, { logicalColumn := 33, sourceArmColumn := 21580, resolution := .equalityAlias 7835 46563 1 false }
, { logicalColumn := 34, sourceArmColumn := 21581, resolution := .equalityAlias 7836 46564 1 false }
, { logicalColumn := 35, sourceArmColumn := 21582, resolution := .equalityAlias 7837 46565 1 false }
, { logicalColumn := 36, sourceArmColumn := 21583, resolution := .equalityAlias 7838 46566 1 false }
, { logicalColumn := 37, sourceArmColumn := 21584, resolution := .equalityAlias 7839 46567 1 false }
, { logicalColumn := 38, sourceArmColumn := 21585, resolution := .equalityAlias 7840 46568 1 false }
, { logicalColumn := 39, sourceArmColumn := 21586, resolution := .equalityAlias 7841 46569 1 false }
, { logicalColumn := 40, sourceArmColumn := 21587, resolution := .equalityAlias 7842 46570 1 false }
, { logicalColumn := 41, sourceArmColumn := 21588, resolution := .equalityAlias 7843 46571 1 false }
, { logicalColumn := 42, sourceArmColumn := 21589, resolution := .equalityAlias 7844 46572 1 false }
, { logicalColumn := 43, sourceArmColumn := 21590, resolution := .equalityAlias 7845 46573 1 false }
, { logicalColumn := 44, sourceArmColumn := 21591, resolution := .equalityAlias 7846 46574 1 false }
, { logicalColumn := 45, sourceArmColumn := 21592, resolution := .equalityAlias 7847 46575 1 false }
, { logicalColumn := 46, sourceArmColumn := 21593, resolution := .equalityAlias 7848 46576 1 false }
, { logicalColumn := 47, sourceArmColumn := 21594, resolution := .equalityAlias 7849 46577 1 false }
, { logicalColumn := 48, sourceArmColumn := 21595, resolution := .equalityAlias 7850 46578 1 false }
, { logicalColumn := 49, sourceArmColumn := 21596, resolution := .equalityAlias 7851 46579 1 false }
, { logicalColumn := 50, sourceArmColumn := 21597, resolution := .equalityAlias 7852 46580 1 false }
, { logicalColumn := 51, sourceArmColumn := 21598, resolution := .equalityAlias 7853 46581 1 false }
, { logicalColumn := 52, sourceArmColumn := 21599, resolution := .equalityAlias 7854 46582 1 false }
, { logicalColumn := 53, sourceArmColumn := 21600, resolution := .equalityAlias 7855 46583 1 false }
, { logicalColumn := 54, sourceArmColumn := 21601, resolution := .equalityAlias 7856 46584 1 false }
, { logicalColumn := 55, sourceArmColumn := 21602, resolution := .equalityAlias 7857 46585 1 false }
, { logicalColumn := 56, sourceArmColumn := 21603, resolution := .equalityAlias 7858 46586 1 false }
, { logicalColumn := 57, sourceArmColumn := 21604, resolution := .equalityAlias 7859 46587 1 false }
, { logicalColumn := 58, sourceArmColumn := 21605, resolution := .equalityAlias 7860 46588 1 false }
, { logicalColumn := 59, sourceArmColumn := 21606, resolution := .equalityAlias 7861 46589 1 false }
, { logicalColumn := 60, sourceArmColumn := 21607, resolution := .equalityAlias 7862 46590 1 false }
, { logicalColumn := 61, sourceArmColumn := 21608, resolution := .equalityAlias 7863 46591 1 false }
, { logicalColumn := 62, sourceArmColumn := 21609, resolution := .equalityAlias 7864 46592 1 false }
, { logicalColumn := 63, sourceArmColumn := 21610, resolution := .equalityAlias 7865 46593 1 false }
, { logicalColumn := 64, sourceArmColumn := 21611, resolution := .equalityAlias 7866 46594 1 false }
, { logicalColumn := 65, sourceArmColumn := 21612, resolution := .equalityAlias 7867 46595 1 false }
, { logicalColumn := 66, sourceArmColumn := 21613, resolution := .equalityAlias 7868 46596 1 false }
, { logicalColumn := 67, sourceArmColumn := 21614, resolution := .equalityAlias 7869 46597 1 false }
, { logicalColumn := 68, sourceArmColumn := 21615, resolution := .equalityAlias 7870 46598 1 false }
, { logicalColumn := 69, sourceArmColumn := 21616, resolution := .equalityAlias 7871 46599 1 false }
, { logicalColumn := 70, sourceArmColumn := 21617, resolution := .equalityAlias 7872 46600 1 false }
, { logicalColumn := 71, sourceArmColumn := 21618, resolution := .equalityAlias 7873 46601 1 false }
, { logicalColumn := 72, sourceArmColumn := 21619, resolution := .equalityAlias 7874 46602 1 false }
, { logicalColumn := 73, sourceArmColumn := 21620, resolution := .equalityAlias 7875 46603 1 false }
, { logicalColumn := 74, sourceArmColumn := 21621, resolution := .equalityAlias 7876 46604 1 false }
, { logicalColumn := 75, sourceArmColumn := 21622, resolution := .equalityAlias 7877 46605 1 false }
, { logicalColumn := 76, sourceArmColumn := 21623, resolution := .equalityAlias 7878 46606 1 false }
, { logicalColumn := 77, sourceArmColumn := 21624, resolution := .equalityAlias 7879 46607 1 false }
, { logicalColumn := 78, sourceArmColumn := 21625, resolution := .equalityAlias 7880 46608 1 false }
, { logicalColumn := 79, sourceArmColumn := 21626, resolution := .equalityAlias 7881 46609 1 false }
, { logicalColumn := 80, sourceArmColumn := 21627, resolution := .equalityAlias 7882 46610 1 false }
, { logicalColumn := 81, sourceArmColumn := 21628, resolution := .equalityAlias 7883 46611 1 false }
, { logicalColumn := 82, sourceArmColumn := 21629, resolution := .equalityAlias 7884 46612 1 false }
, { logicalColumn := 83, sourceArmColumn := 21630, resolution := .equalityAlias 7885 46613 1 false }
, { logicalColumn := 84, sourceArmColumn := 21631, resolution := .equalityAlias 7886 46614 1 false }
, { logicalColumn := 85, sourceArmColumn := 21632, resolution := .equalityAlias 7887 46615 1 false }
, { logicalColumn := 86, sourceArmColumn := 21633, resolution := .equalityAlias 7888 46616 1 false }
, { logicalColumn := 87, sourceArmColumn := 21634, resolution := .equalityAlias 7889 46617 1 false }
, { logicalColumn := 88, sourceArmColumn := 21635, resolution := .equalityAlias 7890 46618 1 false }
, { logicalColumn := 89, sourceArmColumn := 21636, resolution := .equalityAlias 7891 46619 1 false }
, { logicalColumn := 90, sourceArmColumn := 21637, resolution := .equalityAlias 7892 46620 1 false }
, { logicalColumn := 91, sourceArmColumn := 21638, resolution := .equalityAlias 7893 46621 1 false }
, { logicalColumn := 92, sourceArmColumn := 21639, resolution := .equalityAlias 7894 46622 1 false }
, { logicalColumn := 93, sourceArmColumn := 21640, resolution := .equalityAlias 7895 46623 1 false }
, { logicalColumn := 94, sourceArmColumn := 21641, resolution := .equalityAlias 7896 46624 1 false }
, { logicalColumn := 95, sourceArmColumn := 21642, resolution := .equalityAlias 7897 46625 1 false }
, { logicalColumn := 96, sourceArmColumn := 21643, resolution := .equalityAlias 7898 46626 1 false }
, { logicalColumn := 97, sourceArmColumn := 21644, resolution := .equalityAlias 7899 46627 1 false }
, { logicalColumn := 98, sourceArmColumn := 21645, resolution := .equalityAlias 7900 46628 1 false }
, { logicalColumn := 99, sourceArmColumn := 21646, resolution := .equalityAlias 7901 46629 1 false }
, { logicalColumn := 100, sourceArmColumn := 21647, resolution := .equalityAlias 7902 46630 1 false }
, { logicalColumn := 101, sourceArmColumn := 21648, resolution := .equalityAlias 7903 46631 1 false }
, { logicalColumn := 102, sourceArmColumn := 21649, resolution := .equalityAlias 7904 46632 1 false }
, { logicalColumn := 103, sourceArmColumn := 21650, resolution := .equalityAlias 7905 46633 1 false }
, { logicalColumn := 104, sourceArmColumn := 21651, resolution := .equalityAlias 7906 46634 1 false }
, { logicalColumn := 105, sourceArmColumn := 21652, resolution := .equalityAlias 7907 46635 1 false }
, { logicalColumn := 106, sourceArmColumn := 21653, resolution := .equalityAlias 7908 46636 1 false }
, { logicalColumn := 107, sourceArmColumn := 21654, resolution := .equalityAlias 7909 46637 1 false }
, { logicalColumn := 108, sourceArmColumn := 21655, resolution := .equalityAlias 7910 46638 1 false }
, { logicalColumn := 109, sourceArmColumn := 21656, resolution := .equalityAlias 7911 46639 1 false }
, { logicalColumn := 110, sourceArmColumn := 21657, resolution := .equalityAlias 7912 46640 1 false }
, { logicalColumn := 111, sourceArmColumn := 21658, resolution := .equalityAlias 7913 46641 1 false }
, { logicalColumn := 112, sourceArmColumn := 21659, resolution := .equalityAlias 7914 46642 1 false }
, { logicalColumn := 113, sourceArmColumn := 21660, resolution := .equalityAlias 7915 46643 1 false }
, { logicalColumn := 114, sourceArmColumn := 21661, resolution := .equalityAlias 7916 46644 1 false }
, { logicalColumn := 115, sourceArmColumn := 21662, resolution := .equalityAlias 7917 46645 1 false }
, { logicalColumn := 116, sourceArmColumn := 21663, resolution := .equalityAlias 7918 46646 1 false }
, { logicalColumn := 117, sourceArmColumn := 21664, resolution := .equalityAlias 7919 46647 1 false }
, { logicalColumn := 118, sourceArmColumn := 21665, resolution := .equalityAlias 7920 46648 1 false }
, { logicalColumn := 119, sourceArmColumn := 21666, resolution := .equalityAlias 7921 46649 1 false }
, { logicalColumn := 120, sourceArmColumn := 21667, resolution := .equalityAlias 7922 46650 1 false }
, { logicalColumn := 121, sourceArmColumn := 21668, resolution := .equalityAlias 7923 46651 1 false }
, { logicalColumn := 122, sourceArmColumn := 21669, resolution := .equalityAlias 7924 46652 1 false }
, { logicalColumn := 123, sourceArmColumn := 21670, resolution := .equalityAlias 7925 46653 1 false }
, { logicalColumn := 124, sourceArmColumn := 21671, resolution := .equalityAlias 7926 46654 1 false }
, { logicalColumn := 125, sourceArmColumn := 21672, resolution := .equalityAlias 7927 46655 1 false }
, { logicalColumn := 126, sourceArmColumn := 21673, resolution := .equalityAlias 7928 46656 1 false }
, { logicalColumn := 127, sourceArmColumn := 21674, resolution := .equalityAlias 7929 46657 1 false }
, { logicalColumn := 128, sourceArmColumn := 21675, resolution := .equalityAlias 7930 46658 1 false }
, { logicalColumn := 129, sourceArmColumn := 21676, resolution := .equalityAlias 7931 46659 1 false }
, { logicalColumn := 130, sourceArmColumn := 21677, resolution := .equalityAlias 7932 46660 1 false }
, { logicalColumn := 131, sourceArmColumn := 21678, resolution := .equalityAlias 7933 46661 1 false }
, { logicalColumn := 132, sourceArmColumn := 21679, resolution := .equalityAlias 7934 46662 1 false }
, { logicalColumn := 133, sourceArmColumn := 21680, resolution := .equalityAlias 7935 46663 1 false }
, { logicalColumn := 134, sourceArmColumn := 21681, resolution := .equalityAlias 7936 46664 1 false }
, { logicalColumn := 135, sourceArmColumn := 21682, resolution := .equalityAlias 7937 46665 1 false }
, { logicalColumn := 136, sourceArmColumn := 21683, resolution := .equalityAlias 7938 46666 1 false }
, { logicalColumn := 137, sourceArmColumn := 21684, resolution := .equalityAlias 7939 46667 1 false }
, { logicalColumn := 138, sourceArmColumn := 21685, resolution := .equalityAlias 7940 46668 1 false }
, { logicalColumn := 139, sourceArmColumn := 21686, resolution := .equalityAlias 7941 46669 1 false }
, { logicalColumn := 140, sourceArmColumn := 21687, resolution := .equalityAlias 7942 46670 1 false }
, { logicalColumn := 141, sourceArmColumn := 21688, resolution := .equalityAlias 7943 46671 1 false }
, { logicalColumn := 142, sourceArmColumn := 21689, resolution := .equalityAlias 7944 46672 1 false }
, { logicalColumn := 143, sourceArmColumn := 21690, resolution := .equalityAlias 7945 46673 1 false }
, { logicalColumn := 144, sourceArmColumn := 21691, resolution := .equalityAlias 7946 46674 1 false }
, { logicalColumn := 145, sourceArmColumn := 21692, resolution := .equalityAlias 7947 46675 1 false }
, { logicalColumn := 146, sourceArmColumn := 21693, resolution := .equalityAlias 7948 46676 1 false }
, { logicalColumn := 147, sourceArmColumn := 21694, resolution := .equalityAlias 7949 46677 1 false }
, { logicalColumn := 148, sourceArmColumn := 21695, resolution := .equalityAlias 7950 46678 1 false }
, { logicalColumn := 149, sourceArmColumn := 21696, resolution := .equalityAlias 7951 46679 1 false }
, { logicalColumn := 150, sourceArmColumn := 21697, resolution := .equalityAlias 7952 46680 1 false }
, { logicalColumn := 151, sourceArmColumn := 21698, resolution := .equalityAlias 7953 46681 1 false }
, { logicalColumn := 152, sourceArmColumn := 21699, resolution := .equalityAlias 7954 46682 1 false }
, { logicalColumn := 153, sourceArmColumn := 21700, resolution := .equalityAlias 7955 46683 1 false }
, { logicalColumn := 154, sourceArmColumn := 21701, resolution := .equalityAlias 7956 46684 1 false }
, { logicalColumn := 155, sourceArmColumn := 21702, resolution := .equalityAlias 7957 46685 1 false }
, { logicalColumn := 156, sourceArmColumn := 21703, resolution := .equalityAlias 7958 46686 1 false }
, { logicalColumn := 157, sourceArmColumn := 21704, resolution := .equalityAlias 7959 46687 1 false }
, { logicalColumn := 158, sourceArmColumn := 21705, resolution := .equalityAlias 7960 46688 1 false }
, { logicalColumn := 159, sourceArmColumn := 21706, resolution := .equalityAlias 7961 46689 1 false }
, { logicalColumn := 160, sourceArmColumn := 21707, resolution := .equalityAlias 7962 46690 1 false }
, { logicalColumn := 161, sourceArmColumn := 21708, resolution := .equalityAlias 7963 46691 1 false }
, { logicalColumn := 162, sourceArmColumn := 21709, resolution := .equalityAlias 7964 46692 1 false }
, { logicalColumn := 163, sourceArmColumn := 21710, resolution := .equalityAlias 7965 46693 1 false }
, { logicalColumn := 164, sourceArmColumn := 21711, resolution := .equalityAlias 7966 46694 1 false }
, { logicalColumn := 165, sourceArmColumn := 21712, resolution := .equalityAlias 7967 46695 1 false }
, { logicalColumn := 166, sourceArmColumn := 21713, resolution := .equalityAlias 7968 46696 1 false }
, { logicalColumn := 167, sourceArmColumn := 21714, resolution := .equalityAlias 7969 46697 1 false }
, { logicalColumn := 168, sourceArmColumn := 21715, resolution := .equalityAlias 7970 46698 1 false }
, { logicalColumn := 169, sourceArmColumn := 21716, resolution := .equalityAlias 7971 46699 1 false }
, { logicalColumn := 170, sourceArmColumn := 21717, resolution := .equalityAlias 7972 46700 1 false }
, { logicalColumn := 171, sourceArmColumn := 21718, resolution := .equalityAlias 7973 46701 1 false }
, { logicalColumn := 172, sourceArmColumn := 21719, resolution := .equalityAlias 7974 46702 1 false }
, { logicalColumn := 173, sourceArmColumn := 21720, resolution := .equalityAlias 7975 46703 1 false }
, { logicalColumn := 174, sourceArmColumn := 21721, resolution := .equalityAlias 7976 46704 1 false }
, { logicalColumn := 175, sourceArmColumn := 21722, resolution := .equalityAlias 7977 46705 1 false }
, { logicalColumn := 176, sourceArmColumn := 21723, resolution := .equalityAlias 7978 46706 1 false }
, { logicalColumn := 177, sourceArmColumn := 21724, resolution := .equalityAlias 7979 46707 1 false }
, { logicalColumn := 178, sourceArmColumn := 21725, resolution := .equalityAlias 7980 46708 1 false }
, { logicalColumn := 179, sourceArmColumn := 21726, resolution := .equalityAlias 7981 46709 1 false }
, { logicalColumn := 180, sourceArmColumn := 21727, resolution := .equalityAlias 7982 46710 1 false }
, { logicalColumn := 181, sourceArmColumn := 21728, resolution := .equalityAlias 7983 46711 1 false }
, { logicalColumn := 182, sourceArmColumn := 21729, resolution := .equalityAlias 7984 46712 1 false }
, { logicalColumn := 183, sourceArmColumn := 21730, resolution := .equalityAlias 7985 46713 1 false }
, { logicalColumn := 184, sourceArmColumn := 21731, resolution := .equalityAlias 7986 46714 1 false }
, { logicalColumn := 185, sourceArmColumn := 21732, resolution := .equalityAlias 7987 46715 1 false }
, { logicalColumn := 186, sourceArmColumn := 21733, resolution := .equalityAlias 7988 46716 1 false }
, { logicalColumn := 187, sourceArmColumn := 21734, resolution := .equalityAlias 7989 46717 1 false }
, { logicalColumn := 188, sourceArmColumn := 21735, resolution := .equalityAlias 7990 46718 1 false }
, { logicalColumn := 189, sourceArmColumn := 21736, resolution := .equalityAlias 7991 46719 1 false }
, { logicalColumn := 190, sourceArmColumn := 21737, resolution := .equalityAlias 7992 46720 1 false }
, { logicalColumn := 191, sourceArmColumn := 21738, resolution := .equalityAlias 7993 46721 1 false }
, { logicalColumn := 192, sourceArmColumn := 21739, resolution := .equalityAlias 7994 46722 1 false }
, { logicalColumn := 193, sourceArmColumn := 21740, resolution := .equalityAlias 7995 46723 1 false }
, { logicalColumn := 194, sourceArmColumn := 21741, resolution := .equalityAlias 7996 46724 1 false }
, { logicalColumn := 195, sourceArmColumn := 21742, resolution := .equalityAlias 7997 46725 1 false }
, { logicalColumn := 196, sourceArmColumn := 21743, resolution := .equalityAlias 7998 46726 1 false }
, { logicalColumn := 197, sourceArmColumn := 21744, resolution := .equalityAlias 7999 46727 1 false }
, { logicalColumn := 198, sourceArmColumn := 21745, resolution := .equalityAlias 8000 46728 1 false }
, { logicalColumn := 199, sourceArmColumn := 21746, resolution := .equalityAlias 8001 46729 1 false }
, { logicalColumn := 200, sourceArmColumn := 21747, resolution := .equalityAlias 8002 46730 1 false }
, { logicalColumn := 201, sourceArmColumn := 21748, resolution := .equalityAlias 8003 46731 1 false }
, { logicalColumn := 202, sourceArmColumn := 21749, resolution := .equalityAlias 8004 46732 1 false }
, { logicalColumn := 203, sourceArmColumn := 21750, resolution := .equalityAlias 8005 46733 1 false }
, { logicalColumn := 204, sourceArmColumn := 21751, resolution := .equalityAlias 8006 46734 1 false }
, { logicalColumn := 205, sourceArmColumn := 21752, resolution := .equalityAlias 8007 46735 1 false }
, { logicalColumn := 206, sourceArmColumn := 21753, resolution := .equalityAlias 8008 46736 1 false }
, { logicalColumn := 207, sourceArmColumn := 21754, resolution := .equalityAlias 8009 46737 1 false }
, { logicalColumn := 208, sourceArmColumn := 21755, resolution := .equalityAlias 8010 46738 1 false }
, { logicalColumn := 209, sourceArmColumn := 21756, resolution := .equalityAlias 8011 46739 1 false }
, { logicalColumn := 210, sourceArmColumn := 21757, resolution := .equalityAlias 8012 46740 1 false }
, { logicalColumn := 211, sourceArmColumn := 21758, resolution := .equalityAlias 8013 46741 1 false }
, { logicalColumn := 212, sourceArmColumn := 21759, resolution := .equalityAlias 8014 46742 1 false }
, { logicalColumn := 213, sourceArmColumn := 21760, resolution := .equalityAlias 8015 46743 1 false }
, { logicalColumn := 214, sourceArmColumn := 21761, resolution := .equalityAlias 8016 46744 1 false }
, { logicalColumn := 215, sourceArmColumn := 21762, resolution := .equalityAlias 8017 46745 1 false }
, { logicalColumn := 216, sourceArmColumn := 21763, resolution := .equalityAlias 8018 46746 1 false }
, { logicalColumn := 217, sourceArmColumn := 21764, resolution := .equalityAlias 8019 46747 1 false }
, { logicalColumn := 218, sourceArmColumn := 21765, resolution := .equalityAlias 8020 46748 1 false }
, { logicalColumn := 219, sourceArmColumn := 21766, resolution := .equalityAlias 8021 46749 1 false }
, { logicalColumn := 220, sourceArmColumn := 21767, resolution := .equalityAlias 8022 46750 1 false }
, { logicalColumn := 221, sourceArmColumn := 21768, resolution := .equalityAlias 8023 46751 1 false }
, { logicalColumn := 222, sourceArmColumn := 21769, resolution := .equalityAlias 8024 46752 1 false }
, { logicalColumn := 223, sourceArmColumn := 21770, resolution := .equalityAlias 8025 46753 1 false }
, { logicalColumn := 224, sourceArmColumn := 21771, resolution := .equalityAlias 8026 46754 1 false }
, { logicalColumn := 225, sourceArmColumn := 21772, resolution := .equalityAlias 8027 46755 1 false }
, { logicalColumn := 226, sourceArmColumn := 21773, resolution := .equalityAlias 8028 46756 1 false }
, { logicalColumn := 227, sourceArmColumn := 21774, resolution := .equalityAlias 8029 46757 1 false }
, { logicalColumn := 228, sourceArmColumn := 21775, resolution := .equalityAlias 8030 46758 1 false }
, { logicalColumn := 229, sourceArmColumn := 21776, resolution := .equalityAlias 8031 46759 1 false }
, { logicalColumn := 230, sourceArmColumn := 21777, resolution := .equalityAlias 8032 46760 1 false }
, { logicalColumn := 231, sourceArmColumn := 21778, resolution := .equalityAlias 8033 46761 1 false }
, { logicalColumn := 232, sourceArmColumn := 21779, resolution := .equalityAlias 8034 46762 1 false }
, { logicalColumn := 233, sourceArmColumn := 21780, resolution := .equalityAlias 8035 46763 1 false }
, { logicalColumn := 234, sourceArmColumn := 21781, resolution := .equalityAlias 8036 46764 1 false }
, { logicalColumn := 235, sourceArmColumn := 21782, resolution := .equalityAlias 8037 46765 1 false }
, { logicalColumn := 236, sourceArmColumn := 21783, resolution := .equalityAlias 8038 46766 1 false }
, { logicalColumn := 237, sourceArmColumn := 21784, resolution := .equalityAlias 8039 46767 1 false }
, { logicalColumn := 238, sourceArmColumn := 21785, resolution := .equalityAlias 8040 46768 1 false }
, { logicalColumn := 239, sourceArmColumn := 21786, resolution := .equalityAlias 8041 46769 1 false }
, { logicalColumn := 240, sourceArmColumn := 21787, resolution := .equalityAlias 8042 46770 1 false }
, { logicalColumn := 241, sourceArmColumn := 21788, resolution := .equalityAlias 8043 46771 1 false }
, { logicalColumn := 242, sourceArmColumn := 21789, resolution := .equalityAlias 8044 46772 1 false }
, { logicalColumn := 243, sourceArmColumn := 21790, resolution := .equalityAlias 8045 46773 1 false }
, { logicalColumn := 244, sourceArmColumn := 21791, resolution := .equalityAlias 8046 46774 1 false }
, { logicalColumn := 245, sourceArmColumn := 21792, resolution := .equalityAlias 8047 46775 1 false }
, { logicalColumn := 246, sourceArmColumn := 21793, resolution := .equalityAlias 8048 46776 1 false }
, { logicalColumn := 247, sourceArmColumn := 21794, resolution := .equalityAlias 8049 46777 1 false }
, { logicalColumn := 248, sourceArmColumn := 21795, resolution := .equalityAlias 8050 46778 1 false }
, { logicalColumn := 249, sourceArmColumn := 21796, resolution := .equalityAlias 8051 46779 1 false }
, { logicalColumn := 250, sourceArmColumn := 21797, resolution := .equalityAlias 8052 46780 1 false }
, { logicalColumn := 251, sourceArmColumn := 21798, resolution := .equalityAlias 8053 46781 1 false }
, { logicalColumn := 252, sourceArmColumn := 21799, resolution := .equalityAlias 8054 46782 1 false }
, { logicalColumn := 253, sourceArmColumn := 21800, resolution := .equalityAlias 8055 46783 1 false }
, { logicalColumn := 254, sourceArmColumn := 21801, resolution := .equalityAlias 8056 46784 1 false }
, { logicalColumn := 255, sourceArmColumn := 21802, resolution := .equalityAlias 8057 46785 1 false }
]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Generated.Chunk0
