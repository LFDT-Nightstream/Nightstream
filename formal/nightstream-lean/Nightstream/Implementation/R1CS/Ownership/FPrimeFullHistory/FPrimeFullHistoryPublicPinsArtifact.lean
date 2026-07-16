import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPublicPinsInstructions0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPublicPinsInstructions1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPublicPinsInstructions2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPublicPinsInstructions3
import Nightstream.Implementation.R1CS.Core.AffinePins
import Nightstream.Implementation.R1CS.Core.TrivialRows

/-! Exact checked program for the full-history public-image pins. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPins

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram

set_option maxRecDepth 524288

def inputColumns : List Nat := [0, 10833, 10834, 10835, 10836, 10837, 10838, 10839, 10840, 10841, 10844, 10845, 10846, 10847, 10864, 10865, 10866, 10867, 868005, 868006, 868007, 868008, 868009, 868010, 868011, 868012, 868013, 868014, 868015, 868016, 868017, 868018, 868019, 868020, 868021, 868022, 868023, 868024, 868025, 868026, 868027, 868028, 868029, 868030, 868031, 868032, 868033, 868034, 868035, 868036, 868037, 868038, 868039, 868040, 868041, 868042, 868043, 868044, 868045, 868046, 868047, 868048, 868049, 868050, 868051, 868052, 868053, 868054, 868055, 868056, 868057, 868058, 868059, 868060, 868061, 868062, 868063, 868064, 868065, 868066, 868067, 868068, 1127468, 1127469, 1127470, 1127471, 1127476, 1127477, 1127478, 1127479, 1127480, 1127481, 1127482, 1127483, 1127484, 1127485, 1127486, 1127487, 1127488, 1127489, 1127490, 1127491, 1127492, 1127493, 1127494, 1127495, 1127496, 1127497, 1127498, 1127499, 1127500, 1127501, 1127502, 1127503, 1127504, 1127505, 1127506, 1127507, 1127508, 1127509, 1127510, 1127511, 1127512, 1127513, 1127514, 1127515, 1127516, 1127517, 1127518, 1127519, 1127520, 1127521, 1127522, 1127523, 1127524, 1127525, 1127526, 1127527, 1127528, 1127529, 1127530, 1127531, 1127532, 1127533, 1127534, 1127535, 1127536, 1127537, 1127538, 1127539, 1127540, 1127541, 1127542, 1127543, 1127544, 1127545, 1127675, 1127676, 1127677, 1127678, 1127679, 1127680, 1127681, 1127682, 1127683, 1127684, 1127685, 1127686, 1127687, 1127688, 1127689, 1127690, 1127691, 1127692, 1127693, 1127694, 1127695, 1127696, 1127697, 1127698, 1127699, 1127700, 1127701, 1127702, 1127703, 1127704, 1127705, 1127706, 1127707, 1127708, 1127709, 1127710, 1127711, 1127712, 1127713, 1127714, 1127715, 1127716, 1127717, 1127718, 1127719, 1127720, 1127721, 1127722, 1127723, 1127724, 1127725, 1127726, 1127727, 1127728, 1127729, 1127730, 1127731, 1127732, 1127733, 1127734, 1127735, 1127736, 1127737, 1127738, 3489705, 3489706, 3489707, 3489708]
def rowStart : Nat := 3887263
def rowEnd : Nat := 3891546
def rowCount : Nat := 4283
def definitionCount : Nat := 4232
def checkCount : Nat := 51

def instructions : List Instruction :=
    Generated.instructions0 ++
    Generated.instructions1 ++
    Generated.instructions2 ++
    Generated.instructions3

def rows : List Row := CheckedProgram.rows instructions
def pins : List AffinePins.Pin :=
  [.constant 10834 13435500393251129205,
   .constant 10835 17638995623206032663,
   .constant 10836 6177228518728942806,
   .constant 10837 4908850793563520300,
   .constant 1127480 2,
   .constant 1127481 2,
   .constant 10844 12016668175201939073,
   .constant 10845 18153209110320184117,
   .constant 10846 13406471054362354849,
   .constant 10847 7608310618811630534,
   .constant 10864 18026318123786225651,
   .constant 10865 17371433206299088813,
   .constant 10866 803933326767968932,
   .constant 10867 3103427918208527763,
   .constant 10833 1,
   .constant 1127476 7210764229333866465,
   .constant 1127477 2273995740022794128,
   .constant 1127478 13881913245559828750,
   .constant 1127479 16304800731285715778,
   .equal 1127476 1127468,
   .equal 1127477 1127469,
   .equal 1127478 1127470,
   .equal 1127479 1127471,
   .constant 3489705 13099493426629809331,
   .constant 3489706 10796066764906729069,
   .constant 3489707 5645914237077002863,
   .constant 3489708 13837570780862358496,
   .constant 10864 18026318123786225651,
   .constant 10865 17371433206299088813,
   .constant 10866 803933326767968932,
   .constant 10867 3103427918208527763,
   .constant 3495729 17223033872353671751,
   .constant 3495730 995480178504601417,
   .constant 3495731 1591204376617547159,
   .constant 3495732 1910797563118984589,
   .constant 10838 17168707872888128320,
   .constant 10839 11050799198242575901,
   .constant 10840 16730522141919911230,
   .constant 10841 5655123306428251295]

def trivialRows : List Row :=
  [⟨[], [(0, 1)], []⟩,
   ⟨[], [(0, 1)], []⟩,
   ⟨[], [(0, 1)], []⟩,
   ⟨[], [(0, 1)], []⟩,
   ⟨[], [(0, 1)], []⟩,
   ⟨[], [(0, 1)], []⟩,
   ⟨[], [(0, 1)], []⟩,
   ⟨[], [(0, 1)], []⟩,
   ⟨[], [(0, 1)], []⟩,
   ⟨[], [(0, 1)], []⟩,
   ⟨[], [(0, 1)], []⟩,
   ⟨[], [(0, 1)], []⟩]

theorem instructions_length : instructions.length = rowCount := by native_decide
theorem rows_length : rows.length = rowCount := by native_decide
theorem definitions_length :
(definitions instructions).length = definitionCount := by native_decide
theorem checks_length :
(checks instructions).length = checkCount := by native_decide
theorem definitions_canonical :
∀ definition ∈ definitions instructions, definition.Canonical := by native_decide
theorem definitions_wellFormed :
WellFormed inputColumns (definitions instructions) := by native_decide
theorem checks_reference :
ChecksReference (knownAfter inputColumns (definitions instructions))
instructions := by native_decide
theorem pins_canonical : AffinePins.PinsCanonical pins := by native_decide
theorem pin_rows_in_checks :
rowsIncluded (AffinePins.rows pins) (checks instructions) = true := by
native_decide
theorem trivial_rows_in_checks :
rowsIncluded trivialRows (checks instructions) = true := by native_decide
theorem checks_covered :
∀ row ∈ checks instructions,
row ∈ AffinePins.rows pins ∨ row ∈ trivialRows := by native_decide
theorem trivial_rows_valid : TrivialRows.Valid trivialRows := by native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPins
