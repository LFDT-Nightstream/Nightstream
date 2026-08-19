import Nightstream.Implementation.R1CS.Core.AffinePins

/-! Generated semantic ownership for every plain-profile PiRLC projection row. -/

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

def nativeVerifierOrder : List Role :=
(List.range 18).map .commitmentLane ++
(List.range 5).map .activeXColumn ++
((List.range 3).flatMap fun row =>
(List.range 2).map fun limb => .yRingLimb row limb) ++
(List.range 2).map .yZColLimb

def recursiveRoles : List Role := [.commitmentLane 0, .commitmentLane 1, .commitmentLane 2, .commitmentLane 3, .commitmentLane 4, .commitmentLane 5, .commitmentLane 6, .commitmentLane 7, .commitmentLane 8, .commitmentLane 9, .commitmentLane 10, .commitmentLane 11, .commitmentLane 12, .commitmentLane 13, .commitmentLane 14, .commitmentLane 15, .commitmentLane 16, .commitmentLane 17, .activeXColumn 0, .activeXColumn 1, .activeXColumn 2, .activeXColumn 3, .activeXColumn 4, .yRingLimb 0 0, .yRingLimb 0 1, .yRingLimb 1 0, .yRingLimb 1 1, .yRingLimb 2 0, .yRingLimb 2 1, .yZColLimb 0, .yZColLimb 1]
def terminalRoles : List Role := [.commitmentLane 0, .commitmentLane 1, .commitmentLane 2, .commitmentLane 3, .commitmentLane 4, .commitmentLane 5, .commitmentLane 6, .commitmentLane 7, .commitmentLane 8, .commitmentLane 9, .commitmentLane 10, .commitmentLane 11, .commitmentLane 12, .commitmentLane 13, .commitmentLane 14, .commitmentLane 15, .commitmentLane 16, .commitmentLane 17, .activeXColumn 0, .activeXColumn 1, .activeXColumn 2, .activeXColumn 3, .activeXColumn 4, .yRingLimb 0 0, .yRingLimb 0 1, .yRingLimb 1 0, .yRingLimb 1 1, .yRingLimb 2 0, .yRingLimb 2 1, .yZColLimb 0, .yZColLimb 1]

def recursiveGlueOwners : List GlueOwner :=
  [ ⟨.inactiveXZero, 868723, 868725⟩
  , ⟨.yRingPaddingZero 0, 869393, 869433⟩
  , ⟨.yRingPaddingZero 1, 870101, 870141⟩
  , ⟨.yRingPaddingZero 2, 870809, 870849⟩
  , ⟨.yZColPaddingZero, 871517, 871557⟩
  ]

def terminalGlueOwners : List GlueOwner :=
  [ ⟨.inactiveXZero, 3475009, 3475025⟩
  , ⟨.yRingPaddingZero 0, 3478857, 3479177⟩
  , ⟨.yRingPaddingZero 1, 3483009, 3483329⟩
  , ⟨.yRingPaddingZero 2, 3487161, 3487481⟩
  , ⟨.yZColPaddingZero, 3491313, 3491633⟩
  ]

def recursiveGlueRuns : List AffinePins.Run :=
  [ .zero 31524 341329 2
  , .zero 32899 1 20
  , .zero 373232 1 20
  , .zero 33027 1 20
  , .zero 373360 1 20
  , .zero 33155 1 20
  , .zero 373488 1 20
  , .zero 33289 1 20
  , .zero 373642 1 20
  ]

def terminalGlueRuns : List AffinePins.Run :=
  [ .zero 957433 1790 15
  , .zero 2610463 0 1
  , .zero 958808 1 20
  , .zero 960598 1 20
  , .zero 962388 1 20
  , .zero 964178 1 20
  , .zero 965968 1 20
  , .zero 967758 1 20
  , .zero 969548 1 20
  , .zero 971338 1 20
  , .zero 973128 1 20
  , .zero 974918 1 20
  , .zero 976708 1 20
  , .zero 978498 1 20
  , .zero 980288 1 20
  , .zero 982078 1 20
  , .zero 983868 1 20
  , .zero 2610842 1 20
  , .zero 958936 1 20
  , .zero 960726 1 20
  , .zero 962516 1 20
  , .zero 964306 1 20
  , .zero 966096 1 20
  , .zero 967886 1 20
  , .zero 969676 1 20
  , .zero 971466 1 20
  , .zero 973256 1 20
  , .zero 975046 1 20
  , .zero 976836 1 20
  , .zero 978626 1 20
  , .zero 980416 1 20
  , .zero 982206 1 20
  , .zero 983996 1 20
  , .zero 2610970 1 20
  , .zero 959064 1 20
  , .zero 960854 1 20
  , .zero 962644 1 20
  , .zero 964434 1 20
  , .zero 966224 1 20
  , .zero 968014 1 20
  , .zero 969804 1 20
  , .zero 971594 1 20
  , .zero 973384 1 20
  , .zero 975174 1 20
  , .zero 976964 1 20
  , .zero 978754 1 20
  , .zero 980544 1 20
  , .zero 982334 1 20
  , .zero 984124 1 20
  , .zero 2611098 1 20
  , .zero 959198 1 20
  , .zero 960988 1 20
  , .zero 962778 1 20
  , .zero 964568 1 20
  , .zero 966358 1 20
  , .zero 968148 1 20
  , .zero 969938 1 20
  , .zero 971728 1 20
  , .zero 973518 1 20
  , .zero 975308 1 20
  , .zero 977098 1 20
  , .zero 978888 1 20
  , .zero 980678 1 20
  , .zero 982468 1 20
  , .zero 984258 1 20
  , .zero 2611252 1 20
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
