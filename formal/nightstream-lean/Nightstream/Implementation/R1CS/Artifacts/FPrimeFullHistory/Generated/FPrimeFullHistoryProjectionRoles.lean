import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated semantic ownership for the full-history PiRLC projection rows.

Owns: exact production row and column evidence for projection identities and affine glue.
Does not own: protocol semantics, typed `combined_c` wire identity, witness authority,
or verifier acceptance.
Emits constraints: no.

Each 10-column tail is inferred by continuing the checked active-column stride and
requiring every inferred column to occur in the exact `y_zcol` zero-pin rows. It is
artifact-checked layout evidence; direct typed-wire refinement remains open.

Scope: recursive arity 1 and terminal arity 15 in the full-history audit relation.
This is not evidence for the fixed-F-prime recursive profile.

| Child path | Mathematical obligation | Rust evidence | Lean evidence |
|---|---|---|---|
| `nifs.pi_rlc.identities.y_zcol.{0,1}` | Evaluate each 54-coefficient active output limb at beta | `ProjectionIdentityAudit` | `YZColIdentityOwner` |
| `nifs.pi_rlc.padding.y_zcol` | Pin each inferred 10-column tail to zero | `ProjectionGlueAudit` plus stride check | `YZColOutputPadding` |
| derived padded output | Concatenate each active prefix with its checked tail | generator composition | `recursive/terminalYZColPaddedOutputColumns` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles

open Nightstream.Implementation.R1CS

inductive Role where
| commitmentLane (lane : Nat)
| activeXColumn (column : Nat)
| yRingLimb (row limb : Nat)
| yZColLimb (limb : Nat)
deriving DecidableEq, Repr

inductive GlueRole where
| inactiveXZero
| yRingPaddingZero (row : Nat)
| yZColPaddingZero
deriving DecidableEq, Repr

structure GlueOwner where
role : GlueRole
rowStart : Nat
rowEnd : Nat
deriving DecidableEq, Repr

structure YZColIdentityOwner where
limb : Nat
rowStart : Nat
rowEnd : Nat
activeCoefficientColumns : List Nat
outputEvaluationColumns : List Nat
deriving DecidableEq, Repr

structure YZColOutputPadding where
limb : Nat
sharedRowStart : Nat
sharedRowEnd : Nat
zeroColumns : List Nat
deriving DecidableEq, Repr

def nativeVerifierOrder : List Role :=
(List.range 18).map .commitmentLane ++
(List.range 5).map .activeXColumn ++
((List.range 3).flatMap fun row =>
(List.range 2).map fun limb => .yRingLimb row limb) ++
(List.range 2).map .yZColLimb

def recursiveProjectionArity : Nat := 1
def terminalProjectionArity : Nat := 15

def recursiveRoles : List Role := [.commitmentLane 0, .commitmentLane 1, .commitmentLane 2, .commitmentLane 3, .commitmentLane 4, .commitmentLane 5, .commitmentLane 6, .commitmentLane 7, .commitmentLane 8, .commitmentLane 9, .commitmentLane 10, .commitmentLane 11, .commitmentLane 12, .commitmentLane 13, .commitmentLane 14, .commitmentLane 15, .commitmentLane 16, .commitmentLane 17, .activeXColumn 0, .activeXColumn 1, .activeXColumn 2, .activeXColumn 3, .activeXColumn 4, .yRingLimb 0 0, .yRingLimb 0 1, .yRingLimb 1 0, .yRingLimb 1 1, .yRingLimb 2 0, .yRingLimb 2 1, .yZColLimb 0, .yZColLimb 1]
def terminalRoles : List Role := [.commitmentLane 0, .commitmentLane 1, .commitmentLane 2, .commitmentLane 3, .commitmentLane 4, .commitmentLane 5, .commitmentLane 6, .commitmentLane 7, .commitmentLane 8, .commitmentLane 9, .commitmentLane 10, .commitmentLane 11, .commitmentLane 12, .commitmentLane 13, .commitmentLane 14, .commitmentLane 15, .commitmentLane 16, .commitmentLane 17, .activeXColumn 0, .activeXColumn 1, .activeXColumn 2, .activeXColumn 3, .activeXColumn 4, .yRingLimb 0 0, .yRingLimb 0 1, .yRingLimb 1 0, .yRingLimb 1 1, .yRingLimb 2 0, .yRingLimb 2 1, .yZColLimb 0, .yZColLimb 1]

def recursiveYZColIdentities : List YZColIdentityOwner :=
  [ ⟨0, 847318, 847652, [360283, 360285, 360287, 360289, 360291, 360293, 360295, 360297, 360299, 360301, 360303, 360305, 360307, 360309, 360311, 360313, 360315, 360317, 360319, 360321, 360323, 360325, 360327, 360329, 360331, 360333, 360335, 360337, 360339, 360341, 360343, 360345, 360347, 360349, 360351, 360353, 360355, 360357, 360359, 360361, 360363, 360365, 360367, 360369, 360371, 360373, 360375, 360377, 360379, 360381, 360383, 360385, 360387, 360389], [863643, 863644]⟩
  , ⟨1, 847652, 847986, [360284, 360286, 360288, 360290, 360292, 360294, 360296, 360298, 360300, 360302, 360304, 360306, 360308, 360310, 360312, 360314, 360316, 360318, 360320, 360322, 360324, 360326, 360328, 360330, 360332, 360334, 360336, 360338, 360340, 360342, 360344, 360346, 360348, 360350, 360352, 360354, 360356, 360358, 360360, 360362, 360364, 360366, 360368, 360370, 360372, 360374, 360376, 360378, 360380, 360382, 360384, 360386, 360388, 360390], [863975, 863976]⟩
  ]

def terminalYZColIdentities : List YZColIdentityOwner :=
  [ ⟨0, 3388177, 3390093, [2676664, 2676666, 2676668, 2676670, 2676672, 2676674, 2676676, 2676678, 2676680, 2676682, 2676684, 2676686, 2676688, 2676690, 2676692, 2676694, 2676696, 2676698, 2676700, 2676702, 2676704, 2676706, 2676708, 2676710, 2676712, 2676714, 2676716, 2676718, 2676720, 2676722, 2676724, 2676726, 2676728, 2676730, 2676732, 2676734, 2676736, 2676738, 2676740, 2676742, 2676744, 2676746, 2676748, 2676750, 2676752, 2676754, 2676756, 2676758, 2676760, 2676762, 2676764, 2676766, 2676768, 2676770], [3228996, 3228997]⟩
  , ⟨1, 3390093, 3392009, [2676665, 2676667, 2676669, 2676671, 2676673, 2676675, 2676677, 2676679, 2676681, 2676683, 2676685, 2676687, 2676689, 2676691, 2676693, 2676695, 2676697, 2676699, 2676701, 2676703, 2676705, 2676707, 2676709, 2676711, 2676713, 2676715, 2676717, 2676719, 2676721, 2676723, 2676725, 2676727, 2676729, 2676731, 2676733, 2676735, 2676737, 2676739, 2676741, 2676743, 2676745, 2676747, 2676749, 2676751, 2676753, 2676755, 2676757, 2676759, 2676761, 2676763, 2676765, 2676767, 2676769, 2676771], [3230910, 3230911]⟩
  ]

def recursiveYZColOutputPadding : List YZColOutputPadding :=
  [ ⟨0, 847986, 848026, [360391, 360393, 360395, 360397, 360399, 360401, 360403, 360405, 360407, 360409]⟩
  , ⟨1, 847986, 848026, [360392, 360394, 360396, 360398, 360400, 360402, 360404, 360406, 360408, 360410]⟩
  ]

def terminalYZColOutputPadding : List YZColOutputPadding :=
  [ ⟨0, 3392009, 3392329, [2676772, 2676774, 2676776, 2676778, 2676780, 2676782, 2676784, 2676786, 2676788, 2676790]⟩
  , ⟨1, 3392009, 3392329, [2676773, 2676775, 2676777, 2676779, 2676781, 2676783, 2676785, 2676787, 2676789, 2676791]⟩
  ]

def paddedOutputColumns
(identity : YZColIdentityOwner) (padding : YZColOutputPadding) : List Nat :=
identity.activeCoefficientColumns ++ padding.zeroColumns

def recursiveYZColPaddedOutputColumns : List (List Nat) :=
List.zipWith paddedOutputColumns recursiveYZColIdentities recursiveYZColOutputPadding
def terminalYZColPaddedOutputColumns : List (List Nat) :=
List.zipWith paddedOutputColumns terminalYZColIdentities terminalYZColOutputPadding

def outputPaddingPins (owners : List YZColOutputPadding) : List AffinePins.Pin :=
owners.flatMap fun owner => owner.zeroColumns.map .zero

def recursiveGlueOwners : List GlueOwner :=
  [ ⟨.inactiveXZero, 845192, 845194⟩
  , ⟨.yRingPaddingZero 0, 845862, 845902⟩
  , ⟨.yRingPaddingZero 1, 846570, 846610⟩
  , ⟨.yRingPaddingZero 2, 847278, 847318⟩
  , ⟨.yZColPaddingZero, 847986, 848026⟩
  ]

def terminalGlueOwners : List GlueOwner :=
  [ ⟨.inactiveXZero, 3375705, 3375721⟩
  , ⟨.yRingPaddingZero 0, 3379553, 3379873⟩
  , ⟨.yRingPaddingZero 1, 3383705, 3384025⟩
  , ⟨.yRingPaddingZero 2, 3387857, 3388177⟩
  , ⟨.yZColPaddingZero, 3392009, 3392329⟩
  ]

def recursiveGlueRuns : List AffinePins.Run :=
  [ .zero 31524 328078 2
  , .zero 32899 1 20
  , .zero 359981 1 20
  , .zero 33027 1 20
  , .zero 360109 1 20
  , .zero 33155 1 20
  , .zero 360237 1 20
  , .zero 33289 1 20
  , .zero 360391 1 20
  ]

def terminalGlueRuns : List AffinePins.Run :=
  [ .zero 1158598 1790 15
  , .zero 2675983 0 1
  , .zero 1159973 1 20
  , .zero 1161763 1 20
  , .zero 1163553 1 20
  , .zero 1165343 1 20
  , .zero 1167133 1 20
  , .zero 1168923 1 20
  , .zero 1170713 1 20
  , .zero 1172503 1 20
  , .zero 1174293 1 20
  , .zero 1176083 1 20
  , .zero 1177873 1 20
  , .zero 1179663 1 20
  , .zero 1181453 1 20
  , .zero 1183243 1 20
  , .zero 1185033 1 20
  , .zero 2676362 1 20
  , .zero 1160101 1 20
  , .zero 1161891 1 20
  , .zero 1163681 1 20
  , .zero 1165471 1 20
  , .zero 1167261 1 20
  , .zero 1169051 1 20
  , .zero 1170841 1 20
  , .zero 1172631 1 20
  , .zero 1174421 1 20
  , .zero 1176211 1 20
  , .zero 1178001 1 20
  , .zero 1179791 1 20
  , .zero 1181581 1 20
  , .zero 1183371 1 20
  , .zero 1185161 1 20
  , .zero 2676490 1 20
  , .zero 1160229 1 20
  , .zero 1162019 1 20
  , .zero 1163809 1 20
  , .zero 1165599 1 20
  , .zero 1167389 1 20
  , .zero 1169179 1 20
  , .zero 1170969 1 20
  , .zero 1172759 1 20
  , .zero 1174549 1 20
  , .zero 1176339 1 20
  , .zero 1178129 1 20
  , .zero 1179919 1 20
  , .zero 1181709 1 20
  , .zero 1183499 1 20
  , .zero 1185289 1 20
  , .zero 2676618 1 20
  , .zero 1160363 1 20
  , .zero 1162153 1 20
  , .zero 1163943 1 20
  , .zero 1165733 1 20
  , .zero 1167523 1 20
  , .zero 1169313 1 20
  , .zero 1171103 1 20
  , .zero 1172893 1 20
  , .zero 1174683 1 20
  , .zero 1176473 1 20
  , .zero 1178263 1 20
  , .zero 1180053 1 20
  , .zero 1181843 1 20
  , .zero 1183633 1 20
  , .zero 1185423 1 20
  , .zero 2676772 1 20
  ]

def recursiveGluePins : List AffinePins.Pin :=
AffinePins.expandRuns recursiveGlueRuns
def terminalGluePins : List AffinePins.Pin :=
AffinePins.expandRuns terminalGlueRuns

def recursiveGlueRows : List Row := AffinePins.rows recursiveGluePins
def terminalGlueRows : List Row := AffinePins.rows terminalGluePins

theorem recursive_roles_native_order : recursiveRoles = nativeVerifierOrder := by native_decide
theorem terminal_roles_native_order : terminalRoles = nativeVerifierOrder := by native_decide
theorem role_census : recursiveRoles.length = 31 ∧ terminalRoles.length = 31 := by native_decide
theorem full_history_profile_arities :
recursiveProjectionArity = 1 ∧ terminalProjectionArity = 15 := by native_decide
theorem recursive_y_zcol_identity_census :
recursiveYZColIdentities.map (fun owner => owner.limb) = [0, 1] := by native_decide
theorem terminal_y_zcol_identity_census :
terminalYZColIdentities.map (fun owner => owner.limb) = [0, 1] := by native_decide
theorem y_zcol_padding_census :
recursiveYZColOutputPadding.map (fun owner => owner.limb) = [0, 1] ∧
terminalYZColOutputPadding.map (fun owner => owner.limb) = [0, 1] := by native_decide
theorem y_zcol_active_coefficient_width :
(recursiveYZColIdentities ++ terminalYZColIdentities).all
(fun owner => owner.activeCoefficientColumns.length == 54) = true := by native_decide
theorem y_zcol_output_evaluation_width :
(recursiveYZColIdentities ++ terminalYZColIdentities).all
(fun owner => owner.outputEvaluationColumns.length == 2) = true := by native_decide
theorem y_zcol_padding_width :
(recursiveYZColOutputPadding ++ terminalYZColOutputPadding).all
(fun owner => owner.zeroColumns.length == 10) = true := by native_decide
theorem y_zcol_padded_output_width :
(recursiveYZColPaddedOutputColumns ++ terminalYZColPaddedOutputColumns).all
(fun columns => columns.length == 64) = true := by native_decide
theorem y_zcol_padded_output_columns_disjoint :
recursiveYZColPaddedOutputColumns.flatten.eraseDups.length = 128 ∧
terminalYZColPaddedOutputColumns.flatten.eraseDups.length = 128 := by native_decide
theorem y_zcol_output_padding_is_glue :
(outputPaddingPins recursiveYZColOutputPadding).all recursiveGluePins.contains = true ∧
(outputPaddingPins terminalYZColOutputPadding).all terminalGluePins.contains = true := by
native_decide
theorem y_zcol_output_padding_rows_match_glue_owner :
recursiveYZColOutputPadding.all (fun padding =>
recursiveGlueOwners.contains
⟨.yZColPaddingZero, padding.sharedRowStart, padding.sharedRowEnd⟩) = true ∧
terminalYZColOutputPadding.all (fun padding =>
terminalGlueOwners.contains
⟨.yZColPaddingZero, padding.sharedRowStart, padding.sharedRowEnd⟩) = true := by
native_decide
theorem y_zcol_identity_ranges_nonempty :
(recursiveYZColIdentities ++ terminalYZColIdentities).all
(fun owner => decide (owner.rowStart < owner.rowEnd)) = true := by native_decide
theorem recursive_glue_owner_census :
recursiveGlueOwners.map (fun owner => owner.role) =
[.inactiveXZero, .yRingPaddingZero 0, .yRingPaddingZero 1,
.yRingPaddingZero 2, .yZColPaddingZero] := by native_decide
theorem terminal_glue_owner_census :
terminalGlueOwners.map (fun owner => owner.role) =
[.inactiveXZero, .yRingPaddingZero 0, .yRingPaddingZero 1,
.yRingPaddingZero 2, .yZColPaddingZero] := by native_decide
theorem recursive_glue_rows : recursiveGluePins.length = 162 := by
rw [recursiveGluePins, AffinePins.expandRuns_length]
native_decide
theorem terminal_glue_rows : terminalGluePins.length = 1296 := by
rw [terminalGluePins, AffinePins.expandRuns_length]
native_decide

def pinIsZero : AffinePins.Pin → Bool
| .zero _ => true
| _ => false

theorem recursive_glue_only_zero :
recursiveGluePins.all pinIsZero = true := by native_decide
theorem terminal_glue_only_zero :
terminalGluePins.all pinIsZero = true := by native_decide

theorem recursive_glue_sound
{assignment : Nat → Nat}
(canonical : ∀ column, assignment column < goldilocksP)
(one : assignment 0 = 1)
(satisfies : Satisfies recursiveGlueRows assignment) :
∀ pin ∈ recursiveGluePins, AffinePins.Pin.Holds assignment pin := by
exact AffinePins.rows_sound (by native_decide) canonical one satisfies

theorem recursive_glue_complete
{assignment : Nat → Nat}
(canonical : ∀ column, assignment column < goldilocksP)
(one : assignment 0 = 1)
(holds : ∀ pin ∈ recursiveGluePins, AffinePins.Pin.Holds assignment pin) :
Satisfies recursiveGlueRows assignment := by
exact AffinePins.rows_complete (by native_decide) canonical one holds

theorem terminal_glue_sound
{assignment : Nat → Nat}
(canonical : ∀ column, assignment column < goldilocksP)
(one : assignment 0 = 1)
(satisfies : Satisfies terminalGlueRows assignment) :
∀ pin ∈ terminalGluePins, AffinePins.Pin.Holds assignment pin := by
exact AffinePins.rows_sound (by native_decide) canonical one satisfies

theorem terminal_glue_complete
{assignment : Nat → Nat}
(canonical : ∀ column, assignment column < goldilocksP)
(one : assignment 0 = 1)
(holds : ∀ pin ∈ terminalGluePins, AffinePins.Pin.Holds assignment pin) :
Satisfies terminalGlueRows assignment := by
exact AffinePins.rows_complete (by native_decide) canonical one holds

end Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles
