/-
Generated file: production combined-NC artifact; do not hand-edit.

Owns: the bounded terminal fixture profile, row ranges, absolute source columns, and shared raw-witness allocation.

Does not own: decoding, row satisfaction, transcript authority, commitment
binding, semantic acceptance, costs, or permission to remove rows.

Emits constraints: no.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.generated` | The generated payload named by `Owns` above | computed artifact |
-/

import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Schema

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Metadata

def pendingProjectionProfileTag : Nat := 1
def pendingProjectionJoinId : Nat := 1
def logicalWidth : Nat := 108
def blockCount : Nat := 2
def childCount : Nat := 14
def activeLanes : Nat := 54
def paddedLanes : Nat := 64
def radixBase : Nat := 2
def rowFirst : Nat := 0
def rowStop : Nat := 509
def rowCount : Nat := 509
def rowChunkMaximum : Nat := 200
def tensorRowFirst : Nat := 0
def tensorRowStop : Nat := 185
def productRowFirst : Nat := 185
def productRowStop : Nat := 401
def terminalRowFirst : Nat := 401
def terminalRowStop : Nat := 509
def tensorAbsoluteFirstColumn : Nat := 1659
def productAbsoluteFirstColumn : Nat := 1844
def projectionColumnStop : Nat := 2060
def assignmentColumnCount : Nat := 2060
def assignmentChunkMaximum : Nat := 224
def selectorAbsoluteColumn : Option Nat := none
def selectorValue : Option Nat := none
def pendingOldBlockAbsoluteColumnList : List RawKColumns := [{ c0 := 1, c1 := 2 }, { c0 := 3, c1 := 4 }, { c0 := 5, c1 := 6 }, { c0 := 7, c1 := 8 }, { c0 := 9, c1 := 10 }, { c0 := 11, c1 := 12 }, { c0 := 13, c1 := 14 }, { c0 := 15, c1 := 16 }, { c0 := 17, c1 := 18 }, { c0 := 19, c1 := 20 }, { c0 := 21, c1 := 22 }, { c0 := 23, c1 := 24 }, { c0 := 25, c1 := 26 }, { c0 := 27, c1 := 28 }, { c0 := 29, c1 := 30 }, { c0 := 31, c1 := 32 }, { c0 := 33, c1 := 34 }, { c0 := 35, c1 := 36 }, { c0 := 37, c1 := 38 }]
def pendingParentAbsoluteColumnList : List RawKColumns := [{ c0 := 39, c1 := 40 }, { c0 := 41, c1 := 42 }, { c0 := 43, c1 := 44 }, { c0 := 45, c1 := 46 }, { c0 := 47, c1 := 48 }, { c0 := 49, c1 := 50 }, { c0 := 51, c1 := 52 }, { c0 := 53, c1 := 54 }, { c0 := 55, c1 := 56 }, { c0 := 57, c1 := 58 }, { c0 := 59, c1 := 60 }, { c0 := 61, c1 := 62 }, { c0 := 63, c1 := 64 }, { c0 := 65, c1 := 66 }, { c0 := 67, c1 := 68 }, { c0 := 69, c1 := 70 }, { c0 := 71, c1 := 72 }, { c0 := 73, c1 := 74 }, { c0 := 75, c1 := 76 }, { c0 := 77, c1 := 78 }, { c0 := 79, c1 := 80 }, { c0 := 81, c1 := 82 }, { c0 := 83, c1 := 84 }, { c0 := 85, c1 := 86 }, { c0 := 87, c1 := 88 }, { c0 := 89, c1 := 90 }, { c0 := 91, c1 := 92 }, { c0 := 93, c1 := 94 }, { c0 := 95, c1 := 96 }, { c0 := 97, c1 := 98 }, { c0 := 99, c1 := 100 }, { c0 := 101, c1 := 102 }, { c0 := 103, c1 := 104 }, { c0 := 105, c1 := 106 }, { c0 := 107, c1 := 108 }, { c0 := 109, c1 := 110 }, { c0 := 111, c1 := 112 }, { c0 := 113, c1 := 114 }, { c0 := 115, c1 := 116 }, { c0 := 117, c1 := 118 }, { c0 := 119, c1 := 120 }, { c0 := 121, c1 := 122 }, { c0 := 123, c1 := 124 }, { c0 := 125, c1 := 126 }, { c0 := 127, c1 := 128 }, { c0 := 129, c1 := 130 }, { c0 := 131, c1 := 132 }, { c0 := 133, c1 := 134 }, { c0 := 135, c1 := 136 }, { c0 := 137, c1 := 138 }, { c0 := 139, c1 := 140 }, { c0 := 141, c1 := 142 }, { c0 := 143, c1 := 144 }, { c0 := 145, c1 := 146 }]
def childWitnessAbsoluteFirstList : List Nat := [147, 255, 363, 471, 579, 687, 795, 903, 1011, 1119, 1227, 1335, 1443, 1551]
def ajtaiChildWitnessAbsoluteFirstList : List Nat := [147, 255, 363, 471, 579, 687, 795, 903, 1011, 1119, 1227, 1335, 1443, 1551]

def pendingOldBlockAbsoluteColumns (index : Fin 19) : RawKColumns :=
pendingOldBlockAbsoluteColumnList.getD index.val default
def pendingParentAbsoluteColumns (index : Fin 54) : RawKColumns :=
pendingParentAbsoluteColumnList.getD index.val default
def childWitnessAbsoluteFirst (index : Fin 14) : Nat :=
childWitnessAbsoluteFirstList.getD index.val 0
def childWitnessOffset (lane block : Nat) : Nat := lane * blockCount + block
def sharedFinalWitnessAllocation : Bool :=
childWitnessAbsoluteFirstList == ajtaiChildWitnessAbsoluteFirstList

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Metadata
