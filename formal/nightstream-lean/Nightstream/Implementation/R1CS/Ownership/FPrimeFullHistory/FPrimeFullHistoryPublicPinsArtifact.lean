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

def inputColumns : List Nat := [0, 10833, 10834, 10835, 10836, 10837, 10838, 10839, 10840, 10841, 10844, 10845, 10846, 10847, 10864, 10865, 10866, 10867, 883048, 883049, 883050, 883051, 883052, 883053, 883054, 883055, 883056, 883057, 883058, 883059, 883060, 883061, 883062, 883063, 883064, 883065, 883066, 883067, 883068, 883069, 883070, 883071, 883072, 883073, 883074, 883075, 883076, 883077, 883078, 883079, 883080, 883081, 883082, 883083, 883084, 883085, 883086, 883087, 883088, 883089, 883090, 883091, 883092, 883093, 883094, 883095, 883096, 883097, 883098, 883099, 883100, 883101, 883102, 883103, 883104, 883105, 883106, 883107, 883108, 883109, 883110, 883111, 924511, 924512, 924513, 924514, 924519, 924520, 924521, 924522, 924523, 924524, 924525, 924526, 924527, 924528, 924529, 924530, 924531, 924532, 924533, 924534, 924535, 924536, 924537, 924538, 924539, 924540, 924541, 924542, 924543, 924544, 924545, 924546, 924547, 924548, 924549, 924550, 924551, 924552, 924553, 924554, 924555, 924556, 924557, 924558, 924559, 924560, 924561, 924562, 924563, 924564, 924565, 924566, 924567, 924568, 924569, 924570, 924571, 924572, 924573, 924574, 924575, 924576, 924577, 924578, 924579, 924580, 924581, 924582, 924583, 924584, 924585, 924586, 924587, 924588, 924718, 924719, 924720, 924721, 924722, 924723, 924724, 924725, 924726, 924727, 924728, 924729, 924730, 924731, 924732, 924733, 924734, 924735, 924736, 924737, 924738, 924739, 924740, 924741, 924742, 924743, 924744, 924745, 924746, 924747, 924748, 924749, 924750, 924751, 924752, 924753, 924754, 924755, 924756, 924757, 924758, 924759, 924760, 924761, 924762, 924763, 924764, 924765, 924766, 924767, 924768, 924769, 924770, 924771, 924772, 924773, 924774, 924775, 924776, 924777, 924778, 924779, 924780, 924781, 3207977, 3207978, 3207979, 3207980]
def rowStart : Nat := 3770743
def rowEnd : Nat := 3775026
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
  [.constant 10834 13105892220216807217,
   .constant 10835 9061102668333545749,
   .constant 10836 1787228973076538554,
   .constant 10837 10620686771465448400,
   .constant 924523 2,
   .constant 924524 2,
   .constant 10844 12016668175201939073,
   .constant 10845 18153209110320184117,
   .constant 10846 13406471054362354849,
   .constant 10847 7608310618811630534,
   .constant 10864 18026318123786225651,
   .constant 10865 17371433206299088813,
   .constant 10866 803933326767968932,
   .constant 10867 3103427918208527763,
   .constant 10833 1,
   .constant 924519 5704599314127614118,
   .constant 924520 6143833471849673002,
   .constant 924521 15618725063194727333,
   .constant 924522 6586396918271467411,
   .equal 924519 924511,
   .equal 924520 924512,
   .equal 924521 924513,
   .equal 924522 924514,
   .constant 3207977 1246913954205390813,
   .constant 3207978 8915578952494079849,
   .constant 3207979 9828500853140795608,
   .constant 3207980 5318854945093167232,
   .constant 10864 18026318123786225651,
   .constant 10865 17371433206299088813,
   .constant 10866 803933326767968932,
   .constant 10867 3103427918208527763,
   .constant 3212209 9379004758169494217,
   .constant 3212210 6070842424385929866,
   .constant 3212211 5341748968233787131,
   .constant 3212212 15711776321793274157,
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
