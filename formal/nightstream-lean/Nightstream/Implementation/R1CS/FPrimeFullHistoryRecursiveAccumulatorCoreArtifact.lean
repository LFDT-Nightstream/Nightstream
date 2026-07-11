import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Instructions0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Instructions1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Instructions2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Instructions3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Instructions4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Instructions5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Instructions6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Instructions7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Instructions8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Inputs0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Inputs1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Inputs2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Inputs3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Inputs4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Inputs5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Inputs6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Inputs7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Inputs8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment0Inputs9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Instructions0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Instructions1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Instructions2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Instructions3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Instructions4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Instructions5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Instructions6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Instructions7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Instructions8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Instructions9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Instructions10
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Instructions11
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Inputs0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Inputs1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Inputs2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Inputs3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Inputs4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Inputs5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Inputs6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Inputs7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Inputs8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Inputs9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Inputs10
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment1Inputs11
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Instructions0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Instructions1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Instructions2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Instructions3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Instructions4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Instructions5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Instructions6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Instructions7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Instructions8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Instructions9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Instructions10
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Instructions11
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSegment2Inputs0
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Artifact

/-! Exact checked program for the recursive accumulator core owner. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCore

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram

set_option maxRecDepth 1048576

def parentCeDigestColumns : List Nat := [921485, 921486, 921487, 921488]
def accumulatorDigestColumns : List Nat := [924511, 924512, 924513, 924514]
def rowStart : Nat := 887392
def rowEnd : Nat := 924687
def rowCount : Nat := 37295
def definitionCount : Nat := 15695
def checkCount : Nat := 21438

def segment0RowStart : Nat := 0
def segment0RowEnd : Nat := 10437
def segment0InputColumns : List Nat :=
    Generated.segment0Inputs0 ++
    Generated.segment0Inputs1 ++
    Generated.segment0Inputs2 ++
    Generated.segment0Inputs3 ++
    Generated.segment0Inputs4 ++
    Generated.segment0Inputs5 ++
    Generated.segment0Inputs6 ++
    Generated.segment0Inputs7 ++
    Generated.segment0Inputs8 ++
    Generated.segment0Inputs9
def segment0DefinitionCount : Nat := 1095
def segment0CheckCount : Nat := 9342
def segment0Instructions : List Instruction :=
    Generated.segment0Instructions0 ++
    Generated.segment0Instructions1 ++
    Generated.segment0Instructions2 ++
    Generated.segment0Instructions3 ++
    Generated.segment0Instructions4 ++
    Generated.segment0Instructions5 ++
    Generated.segment0Instructions6 ++
    Generated.segment0Instructions7 ++
    Generated.segment0Instructions8
def segment0Rows : List Row :=
  CheckedProgram.rows segment0Instructions

theorem segment0_instructions_length :
    segment0Instructions.length =
      segment0RowEnd - segment0RowStart := by native_decide
theorem segment0_rows_length :
    segment0Rows.length =
      segment0RowEnd - segment0RowStart := by
  simpa [segment0Rows, CheckedProgram.rows] using
    segment0_instructions_length
theorem segment0_definitions_canonical :
    ∀ definition ∈ definitions segment0Instructions,
      definition.Canonical := by native_decide
theorem segment0_definitions_wellFormed :
    WellFormed segment0InputColumns
      (definitions segment0Instructions) := by native_decide
theorem segment0_checks_reference :
    ChecksReference
      (knownAfter segment0InputColumns
        (definitions segment0Instructions))
      segment0Instructions := by native_decide

def segment1RowStart : Nat := 10545
def segment1RowEnd : Nat := 23939
def segment1InputColumns : List Nat :=
    Generated.segment1Inputs0 ++
    Generated.segment1Inputs1 ++
    Generated.segment1Inputs2 ++
    Generated.segment1Inputs3 ++
    Generated.segment1Inputs4 ++
    Generated.segment1Inputs5 ++
    Generated.segment1Inputs6 ++
    Generated.segment1Inputs7 ++
    Generated.segment1Inputs8 ++
    Generated.segment1Inputs9 ++
    Generated.segment1Inputs10 ++
    Generated.segment1Inputs11
def segment1DefinitionCount : Nat := 1298
def segment1CheckCount : Nat := 12096
def segment1Instructions : List Instruction :=
    Generated.segment1Instructions0 ++
    Generated.segment1Instructions1 ++
    Generated.segment1Instructions2 ++
    Generated.segment1Instructions3 ++
    Generated.segment1Instructions4 ++
    Generated.segment1Instructions5 ++
    Generated.segment1Instructions6 ++
    Generated.segment1Instructions7 ++
    Generated.segment1Instructions8 ++
    Generated.segment1Instructions9 ++
    Generated.segment1Instructions10 ++
    Generated.segment1Instructions11
def segment1Rows : List Row :=
  CheckedProgram.rows segment1Instructions

theorem segment1_instructions_length :
    segment1Instructions.length =
      segment1RowEnd - segment1RowStart := by native_decide
theorem segment1_rows_length :
    segment1Rows.length =
      segment1RowEnd - segment1RowStart := by
  simpa [segment1Rows, CheckedProgram.rows] using
    segment1_instructions_length
theorem segment1_definitions_canonical :
    ∀ definition ∈ definitions segment1Instructions,
      definition.Canonical := by native_decide
theorem segment1_definitions_wellFormed :
    WellFormed segment1InputColumns
      (definitions segment1Instructions) := by native_decide
theorem segment1_checks_reference :
    ChecksReference
      (knownAfter segment1InputColumns
        (definitions segment1Instructions))
      segment1Instructions := by native_decide

def segment2RowStart : Nat := 23993
def segment2RowEnd : Nat := 37295
def segment2InputColumns : List Nat :=
    Generated.segment2Inputs0
def segment2DefinitionCount : Nat := 13302
def segment2CheckCount : Nat := 0
def segment2Instructions : List Instruction :=
    Generated.segment2Instructions0 ++
    Generated.segment2Instructions1 ++
    Generated.segment2Instructions2 ++
    Generated.segment2Instructions3 ++
    Generated.segment2Instructions4 ++
    Generated.segment2Instructions5 ++
    Generated.segment2Instructions6 ++
    Generated.segment2Instructions7 ++
    Generated.segment2Instructions8 ++
    Generated.segment2Instructions9 ++
    Generated.segment2Instructions10 ++
    Generated.segment2Instructions11
def segment2Rows : List Row :=
  CheckedProgram.rows segment2Instructions

theorem segment2_instructions_length :
    segment2Instructions.length =
      segment2RowEnd - segment2RowStart := by native_decide
theorem segment2_rows_length :
    segment2Rows.length =
      segment2RowEnd - segment2RowStart := by
  simpa [segment2Rows, CheckedProgram.rows] using
    segment2_instructions_length
theorem segment2_definitions_canonical :
    ∀ definition ∈ definitions segment2Instructions,
      definition.Canonical := by native_decide
theorem segment2_definitions_wellFormed :
    WellFormed segment2InputColumns
      (definitions segment2Instructions) := by native_decide
theorem segment2_checks_reference :
    ChecksReference
      (knownAfter segment2InputColumns
        (definitions segment2Instructions))
      segment2Instructions := by native_decide


structure ShiftedTernaryMap where
  rowStart : Nat
  fieldColumn : Nat
  digitColumns : List Nat
  negativeColumns : List Nat
  borrowColumns : List Nat
deriving DecidableEq, Repr, Inhabited

def shiftedTernaryMaps : List ShiftedTernaryMap :=
  [{ rowStart := 21, fieldColumn := 887609, digitColumns := ((List.range 41).map (fun index => 887737 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 887778 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 887819 + 1 * index)) },
   { rowStart := 145, fieldColumn := 887610, digitColumns := ((List.range 41).map (fun index => 887859 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 887900 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 887941 + 1 * index)) },
   { rowStart := 269, fieldColumn := 887611, digitColumns := ((List.range 41).map (fun index => 887981 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 888022 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 888063 + 1 * index)) },
   { rowStart := 393, fieldColumn := 887612, digitColumns := ((List.range 41).map (fun index => 888103 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 888144 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 888185 + 1 * index)) },
   { rowStart := 517, fieldColumn := 887613, digitColumns := ((List.range 41).map (fun index => 888225 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 888266 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 888307 + 1 * index)) },
   { rowStart := 641, fieldColumn := 887614, digitColumns := ((List.range 41).map (fun index => 888347 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 888388 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 888429 + 1 * index)) },
   { rowStart := 765, fieldColumn := 887615, digitColumns := ((List.range 41).map (fun index => 888469 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 888510 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 888551 + 1 * index)) },
   { rowStart := 889, fieldColumn := 887616, digitColumns := ((List.range 41).map (fun index => 888591 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 888632 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 888673 + 1 * index)) },
   { rowStart := 1013, fieldColumn := 887617, digitColumns := ((List.range 41).map (fun index => 888713 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 888754 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 888795 + 1 * index)) },
   { rowStart := 1137, fieldColumn := 887618, digitColumns := ((List.range 41).map (fun index => 888835 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 888876 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 888917 + 1 * index)) },
   { rowStart := 1261, fieldColumn := 887619, digitColumns := ((List.range 41).map (fun index => 888957 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 888998 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 889039 + 1 * index)) },
   { rowStart := 1385, fieldColumn := 887620, digitColumns := ((List.range 41).map (fun index => 889079 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 889120 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 889161 + 1 * index)) },
   { rowStart := 1509, fieldColumn := 887621, digitColumns := ((List.range 41).map (fun index => 889201 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 889242 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 889283 + 1 * index)) },
   { rowStart := 1633, fieldColumn := 373514, digitColumns := ((List.range 41).map (fun index => 889323 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 889364 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 889405 + 1 * index)) },
   { rowStart := 1757, fieldColumn := 373515, digitColumns := ((List.range 41).map (fun index => 889445 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 889486 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 889527 + 1 * index)) },
   { rowStart := 1881, fieldColumn := 887622, digitColumns := ((List.range 41).map (fun index => 889567 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 889608 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 889649 + 1 * index)) },
   { rowStart := 2005, fieldColumn := 887623, digitColumns := ((List.range 41).map (fun index => 889689 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 889730 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 889771 + 1 * index)) },
   { rowStart := 2129, fieldColumn := 373232, digitColumns := ((List.range 41).map (fun index => 889811 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 889852 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 889893 + 1 * index)) },
   { rowStart := 2253, fieldColumn := 373233, digitColumns := ((List.range 41).map (fun index => 889933 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 889974 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 890015 + 1 * index)) },
   { rowStart := 2377, fieldColumn := 373234, digitColumns := ((List.range 41).map (fun index => 890055 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 890096 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 890137 + 1 * index)) },
   { rowStart := 2501, fieldColumn := 373235, digitColumns := ((List.range 41).map (fun index => 890177 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 890218 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 890259 + 1 * index)) },
   { rowStart := 2625, fieldColumn := 373236, digitColumns := ((List.range 41).map (fun index => 890299 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 890340 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 890381 + 1 * index)) },
   { rowStart := 2749, fieldColumn := 373237, digitColumns := ((List.range 41).map (fun index => 890421 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 890462 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 890503 + 1 * index)) },
   { rowStart := 2873, fieldColumn := 373238, digitColumns := ((List.range 41).map (fun index => 890543 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 890584 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 890625 + 1 * index)) },
   { rowStart := 2997, fieldColumn := 373239, digitColumns := ((List.range 41).map (fun index => 890665 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 890706 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 890747 + 1 * index)) },
   { rowStart := 3121, fieldColumn := 373240, digitColumns := ((List.range 41).map (fun index => 890787 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 890828 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 890869 + 1 * index)) },
   { rowStart := 3245, fieldColumn := 373241, digitColumns := ((List.range 41).map (fun index => 890909 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 890950 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 890991 + 1 * index)) },
   { rowStart := 3369, fieldColumn := 373242, digitColumns := ((List.range 41).map (fun index => 891031 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 891072 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 891113 + 1 * index)) },
   { rowStart := 3493, fieldColumn := 373243, digitColumns := ((List.range 41).map (fun index => 891153 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 891194 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 891235 + 1 * index)) },
   { rowStart := 3617, fieldColumn := 373244, digitColumns := ((List.range 41).map (fun index => 891275 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 891316 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 891357 + 1 * index)) },
   { rowStart := 3741, fieldColumn := 373245, digitColumns := ((List.range 41).map (fun index => 891397 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 891438 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 891479 + 1 * index)) },
   { rowStart := 3865, fieldColumn := 373246, digitColumns := ((List.range 41).map (fun index => 891519 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 891560 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 891601 + 1 * index)) },
   { rowStart := 3989, fieldColumn := 373247, digitColumns := ((List.range 41).map (fun index => 891641 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 891682 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 891723 + 1 * index)) },
   { rowStart := 4113, fieldColumn := 373248, digitColumns := ((List.range 41).map (fun index => 891763 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 891804 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 891845 + 1 * index)) },
   { rowStart := 4237, fieldColumn := 373249, digitColumns := ((List.range 41).map (fun index => 891885 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 891926 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 891967 + 1 * index)) },
   { rowStart := 4361, fieldColumn := 373250, digitColumns := ((List.range 41).map (fun index => 892007 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 892048 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 892089 + 1 * index)) },
   { rowStart := 4485, fieldColumn := 373251, digitColumns := ((List.range 41).map (fun index => 892129 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 892170 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 892211 + 1 * index)) },
   { rowStart := 4609, fieldColumn := 887624, digitColumns := ((List.range 41).map (fun index => 892251 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 892292 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 892333 + 1 * index)) },
   { rowStart := 4733, fieldColumn := 373360, digitColumns := ((List.range 41).map (fun index => 892373 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 892414 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 892455 + 1 * index)) },
   { rowStart := 4857, fieldColumn := 373361, digitColumns := ((List.range 41).map (fun index => 892495 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 892536 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 892577 + 1 * index)) },
   { rowStart := 4981, fieldColumn := 373362, digitColumns := ((List.range 41).map (fun index => 892617 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 892658 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 892699 + 1 * index)) },
   { rowStart := 5105, fieldColumn := 373363, digitColumns := ((List.range 41).map (fun index => 892739 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 892780 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 892821 + 1 * index)) },
   { rowStart := 5229, fieldColumn := 373364, digitColumns := ((List.range 41).map (fun index => 892861 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 892902 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 892943 + 1 * index)) },
   { rowStart := 5353, fieldColumn := 373365, digitColumns := ((List.range 41).map (fun index => 892983 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 893024 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 893065 + 1 * index)) },
   { rowStart := 5477, fieldColumn := 373366, digitColumns := ((List.range 41).map (fun index => 893105 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 893146 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 893187 + 1 * index)) },
   { rowStart := 5601, fieldColumn := 373367, digitColumns := ((List.range 41).map (fun index => 893227 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 893268 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 893309 + 1 * index)) },
   { rowStart := 5725, fieldColumn := 373368, digitColumns := ((List.range 41).map (fun index => 893349 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 893390 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 893431 + 1 * index)) },
   { rowStart := 5849, fieldColumn := 373369, digitColumns := ((List.range 41).map (fun index => 893471 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 893512 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 893553 + 1 * index)) },
   { rowStart := 5973, fieldColumn := 373370, digitColumns := ((List.range 41).map (fun index => 893593 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 893634 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 893675 + 1 * index)) },
   { rowStart := 6097, fieldColumn := 373371, digitColumns := ((List.range 41).map (fun index => 893715 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 893756 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 893797 + 1 * index)) },
   { rowStart := 6221, fieldColumn := 373372, digitColumns := ((List.range 41).map (fun index => 893837 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 893878 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 893919 + 1 * index)) },
   { rowStart := 6345, fieldColumn := 373373, digitColumns := ((List.range 41).map (fun index => 893959 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 894000 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 894041 + 1 * index)) },
   { rowStart := 6469, fieldColumn := 373374, digitColumns := ((List.range 41).map (fun index => 894081 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 894122 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 894163 + 1 * index)) },
   { rowStart := 6593, fieldColumn := 373375, digitColumns := ((List.range 41).map (fun index => 894203 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 894244 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 894285 + 1 * index)) },
   { rowStart := 6717, fieldColumn := 373376, digitColumns := ((List.range 41).map (fun index => 894325 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 894366 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 894407 + 1 * index)) },
   { rowStart := 6841, fieldColumn := 373377, digitColumns := ((List.range 41).map (fun index => 894447 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 894488 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 894529 + 1 * index)) },
   { rowStart := 6965, fieldColumn := 373378, digitColumns := ((List.range 41).map (fun index => 894569 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 894610 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 894651 + 1 * index)) },
   { rowStart := 7089, fieldColumn := 373379, digitColumns := ((List.range 41).map (fun index => 894691 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 894732 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 894773 + 1 * index)) },
   { rowStart := 7213, fieldColumn := 887625, digitColumns := ((List.range 41).map (fun index => 894813 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 894854 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 894895 + 1 * index)) },
   { rowStart := 7337, fieldColumn := 373488, digitColumns := ((List.range 41).map (fun index => 894935 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 894976 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 895017 + 1 * index)) },
   { rowStart := 7461, fieldColumn := 373489, digitColumns := ((List.range 41).map (fun index => 895057 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 895098 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 895139 + 1 * index)) },
   { rowStart := 7585, fieldColumn := 373490, digitColumns := ((List.range 41).map (fun index => 895179 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 895220 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 895261 + 1 * index)) },
   { rowStart := 7709, fieldColumn := 373491, digitColumns := ((List.range 41).map (fun index => 895301 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 895342 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 895383 + 1 * index)) },
   { rowStart := 7833, fieldColumn := 373492, digitColumns := ((List.range 41).map (fun index => 895423 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 895464 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 895505 + 1 * index)) },
   { rowStart := 7957, fieldColumn := 373493, digitColumns := ((List.range 41).map (fun index => 895545 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 895586 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 895627 + 1 * index)) },
   { rowStart := 8081, fieldColumn := 373494, digitColumns := ((List.range 41).map (fun index => 895667 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 895708 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 895749 + 1 * index)) },
   { rowStart := 8205, fieldColumn := 373495, digitColumns := ((List.range 41).map (fun index => 895789 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 895830 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 895871 + 1 * index)) },
   { rowStart := 8329, fieldColumn := 373496, digitColumns := ((List.range 41).map (fun index => 895911 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 895952 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 895993 + 1 * index)) },
   { rowStart := 8453, fieldColumn := 373497, digitColumns := ((List.range 41).map (fun index => 896033 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 896074 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 896115 + 1 * index)) },
   { rowStart := 8577, fieldColumn := 373498, digitColumns := ((List.range 41).map (fun index => 896155 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 896196 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 896237 + 1 * index)) },
   { rowStart := 8701, fieldColumn := 373499, digitColumns := ((List.range 41).map (fun index => 896277 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 896318 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 896359 + 1 * index)) },
   { rowStart := 8825, fieldColumn := 373500, digitColumns := ((List.range 41).map (fun index => 896399 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 896440 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 896481 + 1 * index)) },
   { rowStart := 8949, fieldColumn := 373501, digitColumns := ((List.range 41).map (fun index => 896521 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 896562 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 896603 + 1 * index)) },
   { rowStart := 9073, fieldColumn := 373502, digitColumns := ((List.range 41).map (fun index => 896643 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 896684 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 896725 + 1 * index)) },
   { rowStart := 9197, fieldColumn := 373503, digitColumns := ((List.range 41).map (fun index => 896765 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 896806 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 896847 + 1 * index)) },
   { rowStart := 9321, fieldColumn := 373504, digitColumns := ((List.range 41).map (fun index => 896887 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 896928 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 896969 + 1 * index)) },
   { rowStart := 9445, fieldColumn := 373505, digitColumns := ((List.range 41).map (fun index => 897009 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 897050 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 897091 + 1 * index)) },
   { rowStart := 9569, fieldColumn := 373506, digitColumns := ((List.range 41).map (fun index => 897131 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 897172 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 897213 + 1 * index)) },
   { rowStart := 9693, fieldColumn := 373507, digitColumns := ((List.range 41).map (fun index => 897253 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 897294 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 897335 + 1 * index)) },
   { rowStart := 9817, fieldColumn := 887626, digitColumns := ((List.range 41).map (fun index => 897375 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 897416 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 897457 + 1 * index)) },
   { rowStart := 9941, fieldColumn := 373662, digitColumns := ((List.range 41).map (fun index => 897497 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 897538 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 897579 + 1 * index)) },
   { rowStart := 10065, fieldColumn := 373663, digitColumns := ((List.range 41).map (fun index => 897619 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 897660 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 897701 + 1 * index)) },
   { rowStart := 10189, fieldColumn := 373664, digitColumns := ((List.range 41).map (fun index => 897741 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 897782 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 897823 + 1 * index)) },
   { rowStart := 10313, fieldColumn := 373665, digitColumns := ((List.range 41).map (fun index => 897863 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 897904 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 897945 + 1 * index)) },
   { rowStart := 10547, fieldColumn := 887629, digitColumns := ((List.range 41).map (fun index => 898041 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 898082 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 898123 + 1 * index)) },
   { rowStart := 10671, fieldColumn := 887630, digitColumns := ((List.range 41).map (fun index => 898163 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 898204 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 898245 + 1 * index)) },
   { rowStart := 10795, fieldColumn := 887631, digitColumns := ((List.range 41).map (fun index => 898285 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 898326 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 898367 + 1 * index)) },
   { rowStart := 10919, fieldColumn := 887632, digitColumns := ((List.range 41).map (fun index => 898407 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 898448 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 898489 + 1 * index)) },
   { rowStart := 11043, fieldColumn := 887633, digitColumns := ((List.range 41).map (fun index => 898529 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 898570 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 898611 + 1 * index)) },
   { rowStart := 11167, fieldColumn := 887634, digitColumns := ((List.range 41).map (fun index => 898651 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 898692 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 898733 + 1 * index)) },
   { rowStart := 11291, fieldColumn := 887635, digitColumns := ((List.range 41).map (fun index => 898773 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 898814 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 898855 + 1 * index)) },
   { rowStart := 11415, fieldColumn := 887636, digitColumns := ((List.range 41).map (fun index => 898895 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 898936 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 898977 + 1 * index)) },
   { rowStart := 11539, fieldColumn := 887637, digitColumns := ((List.range 41).map (fun index => 899017 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 899058 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 899099 + 1 * index)) },
   { rowStart := 11663, fieldColumn := 887638, digitColumns := ((List.range 41).map (fun index => 899139 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 899180 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 899221 + 1 * index)) },
   { rowStart := 11787, fieldColumn := 887639, digitColumns := ((List.range 41).map (fun index => 899261 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 899302 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 899343 + 1 * index)) },
   { rowStart := 11911, fieldColumn := 887640, digitColumns := ((List.range 41).map (fun index => 899383 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 899424 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 899465 + 1 * index)) },
   { rowStart := 12035, fieldColumn := 887641, digitColumns := ((List.range 41).map (fun index => 899505 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 899546 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 899587 + 1 * index)) },
   { rowStart := 12159, fieldColumn := 887642, digitColumns := ((List.range 41).map (fun index => 899627 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 899668 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 899709 + 1 * index)) },
   { rowStart := 12283, fieldColumn := 887643, digitColumns := ((List.range 41).map (fun index => 899749 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 899790 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 899831 + 1 * index)) },
   { rowStart := 12407, fieldColumn := 887644, digitColumns := ((List.range 41).map (fun index => 899871 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 899912 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 899953 + 1 * index)) },
   { rowStart := 12531, fieldColumn := 887645, digitColumns := ((List.range 41).map (fun index => 899993 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 900034 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 900075 + 1 * index)) },
   { rowStart := 12655, fieldColumn := 887646, digitColumns := ((List.range 41).map (fun index => 900115 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 900156 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 900197 + 1 * index)) },
   { rowStart := 12779, fieldColumn := 887647, digitColumns := ((List.range 41).map (fun index => 900237 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 900278 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 900319 + 1 * index)) },
   { rowStart := 12903, fieldColumn := 887648, digitColumns := ((List.range 41).map (fun index => 900359 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 900400 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 900441 + 1 * index)) },
   { rowStart := 13027, fieldColumn := 887649, digitColumns := ((List.range 41).map (fun index => 900481 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 900522 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 900563 + 1 * index)) },
   { rowStart := 13151, fieldColumn := 887650, digitColumns := ((List.range 41).map (fun index => 900603 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 900644 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 900685 + 1 * index)) },
   { rowStart := 13275, fieldColumn := 887651, digitColumns := ((List.range 41).map (fun index => 900725 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 900766 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 900807 + 1 * index)) },
   { rowStart := 13399, fieldColumn := 887652, digitColumns := ((List.range 41).map (fun index => 900847 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 900888 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 900929 + 1 * index)) },
   { rowStart := 13523, fieldColumn := 887653, digitColumns := ((List.range 41).map (fun index => 900969 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 901010 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 901051 + 1 * index)) },
   { rowStart := 13647, fieldColumn := 887654, digitColumns := ((List.range 41).map (fun index => 901091 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 901132 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 901173 + 1 * index)) },
   { rowStart := 13771, fieldColumn := 887655, digitColumns := ((List.range 41).map (fun index => 901213 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 901254 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 901295 + 1 * index)) },
   { rowStart := 13895, fieldColumn := 887656, digitColumns := ((List.range 41).map (fun index => 901335 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 901376 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 901417 + 1 * index)) },
   { rowStart := 14019, fieldColumn := 887657, digitColumns := ((List.range 41).map (fun index => 901457 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 901498 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 901539 + 1 * index)) },
   { rowStart := 14143, fieldColumn := 887658, digitColumns := ((List.range 41).map (fun index => 901579 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 901620 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 901661 + 1 * index)) },
   { rowStart := 14267, fieldColumn := 887659, digitColumns := ((List.range 41).map (fun index => 901701 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 901742 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 901783 + 1 * index)) },
   { rowStart := 14391, fieldColumn := 887660, digitColumns := ((List.range 41).map (fun index => 901823 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 901864 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 901905 + 1 * index)) },
   { rowStart := 14515, fieldColumn := 887661, digitColumns := ((List.range 41).map (fun index => 901945 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 901986 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 902027 + 1 * index)) },
   { rowStart := 14639, fieldColumn := 887662, digitColumns := ((List.range 41).map (fun index => 902067 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 902108 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 902149 + 1 * index)) },
   { rowStart := 14763, fieldColumn := 887663, digitColumns := ((List.range 41).map (fun index => 902189 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 902230 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 902271 + 1 * index)) },
   { rowStart := 14887, fieldColumn := 887664, digitColumns := ((List.range 41).map (fun index => 902311 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 902352 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 902393 + 1 * index)) },
   { rowStart := 15011, fieldColumn := 887665, digitColumns := ((List.range 41).map (fun index => 902433 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 902474 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 902515 + 1 * index)) },
   { rowStart := 15135, fieldColumn := 887666, digitColumns := ((List.range 41).map (fun index => 902555 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 902596 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 902637 + 1 * index)) },
   { rowStart := 15259, fieldColumn := 887667, digitColumns := ((List.range 41).map (fun index => 902677 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 902718 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 902759 + 1 * index)) },
   { rowStart := 15383, fieldColumn := 887668, digitColumns := ((List.range 41).map (fun index => 902799 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 902840 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 902881 + 1 * index)) },
   { rowStart := 15507, fieldColumn := 887669, digitColumns := ((List.range 41).map (fun index => 902921 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 902962 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 903003 + 1 * index)) },
   { rowStart := 15631, fieldColumn := 887670, digitColumns := ((List.range 41).map (fun index => 903043 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 903084 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 903125 + 1 * index)) },
   { rowStart := 15755, fieldColumn := 887671, digitColumns := ((List.range 41).map (fun index => 903165 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 903206 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 903247 + 1 * index)) },
   { rowStart := 15879, fieldColumn := 887672, digitColumns := ((List.range 41).map (fun index => 903287 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 903328 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 903369 + 1 * index)) },
   { rowStart := 16003, fieldColumn := 887673, digitColumns := ((List.range 41).map (fun index => 903409 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 903450 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 903491 + 1 * index)) },
   { rowStart := 16127, fieldColumn := 887674, digitColumns := ((List.range 41).map (fun index => 903531 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 903572 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 903613 + 1 * index)) },
   { rowStart := 16251, fieldColumn := 887675, digitColumns := ((List.range 41).map (fun index => 903653 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 903694 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 903735 + 1 * index)) },
   { rowStart := 16375, fieldColumn := 887676, digitColumns := ((List.range 41).map (fun index => 903775 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 903816 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 903857 + 1 * index)) },
   { rowStart := 16499, fieldColumn := 887677, digitColumns := ((List.range 41).map (fun index => 903897 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 903938 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 903979 + 1 * index)) },
   { rowStart := 16623, fieldColumn := 887678, digitColumns := ((List.range 41).map (fun index => 904019 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 904060 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 904101 + 1 * index)) },
   { rowStart := 16747, fieldColumn := 887679, digitColumns := ((List.range 41).map (fun index => 904141 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 904182 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 904223 + 1 * index)) },
   { rowStart := 16871, fieldColumn := 887680, digitColumns := ((List.range 41).map (fun index => 904263 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 904304 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 904345 + 1 * index)) },
   { rowStart := 16995, fieldColumn := 887681, digitColumns := ((List.range 41).map (fun index => 904385 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 904426 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 904467 + 1 * index)) },
   { rowStart := 17119, fieldColumn := 887682, digitColumns := ((List.range 41).map (fun index => 904507 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 904548 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 904589 + 1 * index)) },
   { rowStart := 17243, fieldColumn := 887683, digitColumns := ((List.range 41).map (fun index => 904629 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 904670 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 904711 + 1 * index)) },
   { rowStart := 17367, fieldColumn := 887684, digitColumns := ((List.range 41).map (fun index => 904751 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 904792 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 904833 + 1 * index)) },
   { rowStart := 17491, fieldColumn := 887685, digitColumns := ((List.range 41).map (fun index => 904873 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 904914 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 904955 + 1 * index)) },
   { rowStart := 17615, fieldColumn := 887686, digitColumns := ((List.range 41).map (fun index => 904995 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 905036 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 905077 + 1 * index)) },
   { rowStart := 17739, fieldColumn := 887687, digitColumns := ((List.range 41).map (fun index => 905117 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 905158 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 905199 + 1 * index)) },
   { rowStart := 17863, fieldColumn := 887688, digitColumns := ((List.range 41).map (fun index => 905239 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 905280 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 905321 + 1 * index)) },
   { rowStart := 17987, fieldColumn := 887689, digitColumns := ((List.range 41).map (fun index => 905361 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 905402 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 905443 + 1 * index)) },
   { rowStart := 18111, fieldColumn := 887690, digitColumns := ((List.range 41).map (fun index => 905483 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 905524 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 905565 + 1 * index)) },
   { rowStart := 18235, fieldColumn := 887691, digitColumns := ((List.range 41).map (fun index => 905605 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 905646 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 905687 + 1 * index)) },
   { rowStart := 18359, fieldColumn := 887692, digitColumns := ((List.range 41).map (fun index => 905727 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 905768 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 905809 + 1 * index)) },
   { rowStart := 18483, fieldColumn := 887693, digitColumns := ((List.range 41).map (fun index => 905849 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 905890 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 905931 + 1 * index)) },
   { rowStart := 18607, fieldColumn := 887694, digitColumns := ((List.range 41).map (fun index => 905971 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 906012 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 906053 + 1 * index)) },
   { rowStart := 18731, fieldColumn := 887695, digitColumns := ((List.range 41).map (fun index => 906093 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 906134 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 906175 + 1 * index)) },
   { rowStart := 18855, fieldColumn := 887696, digitColumns := ((List.range 41).map (fun index => 906215 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 906256 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 906297 + 1 * index)) },
   { rowStart := 18979, fieldColumn := 887697, digitColumns := ((List.range 41).map (fun index => 906337 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 906378 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 906419 + 1 * index)) },
   { rowStart := 19103, fieldColumn := 887698, digitColumns := ((List.range 41).map (fun index => 906459 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 906500 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 906541 + 1 * index)) },
   { rowStart := 19227, fieldColumn := 887699, digitColumns := ((List.range 41).map (fun index => 906581 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 906622 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 906663 + 1 * index)) },
   { rowStart := 19351, fieldColumn := 887700, digitColumns := ((List.range 41).map (fun index => 906703 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 906744 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 906785 + 1 * index)) },
   { rowStart := 19475, fieldColumn := 887701, digitColumns := ((List.range 41).map (fun index => 906825 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 906866 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 906907 + 1 * index)) },
   { rowStart := 19599, fieldColumn := 887702, digitColumns := ((List.range 41).map (fun index => 906947 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 906988 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 907029 + 1 * index)) },
   { rowStart := 19723, fieldColumn := 887703, digitColumns := ((List.range 41).map (fun index => 907069 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 907110 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 907151 + 1 * index)) },
   { rowStart := 19847, fieldColumn := 887704, digitColumns := ((List.range 41).map (fun index => 907191 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 907232 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 907273 + 1 * index)) },
   { rowStart := 19971, fieldColumn := 887705, digitColumns := ((List.range 41).map (fun index => 907313 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 907354 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 907395 + 1 * index)) },
   { rowStart := 20095, fieldColumn := 887706, digitColumns := ((List.range 41).map (fun index => 907435 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 907476 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 907517 + 1 * index)) },
   { rowStart := 20219, fieldColumn := 887707, digitColumns := ((List.range 41).map (fun index => 907557 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 907598 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 907639 + 1 * index)) },
   { rowStart := 20343, fieldColumn := 887708, digitColumns := ((List.range 41).map (fun index => 907679 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 907720 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 907761 + 1 * index)) },
   { rowStart := 20467, fieldColumn := 887709, digitColumns := ((List.range 41).map (fun index => 907801 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 907842 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 907883 + 1 * index)) },
   { rowStart := 20591, fieldColumn := 887710, digitColumns := ((List.range 41).map (fun index => 907923 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 907964 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 908005 + 1 * index)) },
   { rowStart := 20715, fieldColumn := 887711, digitColumns := ((List.range 41).map (fun index => 908045 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 908086 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 908127 + 1 * index)) },
   { rowStart := 20839, fieldColumn := 887712, digitColumns := ((List.range 41).map (fun index => 908167 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 908208 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 908249 + 1 * index)) },
   { rowStart := 20963, fieldColumn := 887713, digitColumns := ((List.range 41).map (fun index => 908289 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 908330 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 908371 + 1 * index)) },
   { rowStart := 21087, fieldColumn := 887714, digitColumns := ((List.range 41).map (fun index => 908411 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 908452 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 908493 + 1 * index)) },
   { rowStart := 21211, fieldColumn := 887715, digitColumns := ((List.range 41).map (fun index => 908533 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 908574 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 908615 + 1 * index)) },
   { rowStart := 21335, fieldColumn := 887716, digitColumns := ((List.range 41).map (fun index => 908655 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 908696 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 908737 + 1 * index)) },
   { rowStart := 21459, fieldColumn := 887717, digitColumns := ((List.range 41).map (fun index => 908777 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 908818 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 908859 + 1 * index)) },
   { rowStart := 21583, fieldColumn := 887718, digitColumns := ((List.range 41).map (fun index => 908899 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 908940 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 908981 + 1 * index)) },
   { rowStart := 21707, fieldColumn := 887719, digitColumns := ((List.range 41).map (fun index => 909021 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 909062 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 909103 + 1 * index)) },
   { rowStart := 21831, fieldColumn := 887720, digitColumns := ((List.range 41).map (fun index => 909143 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 909184 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 909225 + 1 * index)) },
   { rowStart := 21955, fieldColumn := 887721, digitColumns := ((List.range 41).map (fun index => 909265 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 909306 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 909347 + 1 * index)) },
   { rowStart := 22079, fieldColumn := 887722, digitColumns := ((List.range 41).map (fun index => 909387 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 909428 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 909469 + 1 * index)) },
   { rowStart := 22203, fieldColumn := 887723, digitColumns := ((List.range 41).map (fun index => 909509 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 909550 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 909591 + 1 * index)) },
   { rowStart := 22327, fieldColumn := 887724, digitColumns := ((List.range 41).map (fun index => 909631 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 909672 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 909713 + 1 * index)) },
   { rowStart := 22451, fieldColumn := 887725, digitColumns := ((List.range 41).map (fun index => 909753 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 909794 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 909835 + 1 * index)) },
   { rowStart := 22575, fieldColumn := 887726, digitColumns := ((List.range 41).map (fun index => 909875 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 909916 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 909957 + 1 * index)) },
   { rowStart := 22699, fieldColumn := 887727, digitColumns := ((List.range 41).map (fun index => 909997 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 910038 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 910079 + 1 * index)) },
   { rowStart := 22823, fieldColumn := 887728, digitColumns := ((List.range 41).map (fun index => 910119 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 910160 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 910201 + 1 * index)) },
   { rowStart := 22947, fieldColumn := 887729, digitColumns := ((List.range 41).map (fun index => 910241 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 910282 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 910323 + 1 * index)) },
   { rowStart := 23071, fieldColumn := 887730, digitColumns := ((List.range 41).map (fun index => 910363 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 910404 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 910445 + 1 * index)) },
   { rowStart := 23195, fieldColumn := 887731, digitColumns := ((List.range 41).map (fun index => 910485 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 910526 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 910567 + 1 * index)) },
   { rowStart := 23319, fieldColumn := 887732, digitColumns := ((List.range 41).map (fun index => 910607 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 910648 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 910689 + 1 * index)) },
   { rowStart := 23443, fieldColumn := 887733, digitColumns := ((List.range 41).map (fun index => 910729 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 910770 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 910811 + 1 * index)) },
   { rowStart := 23567, fieldColumn := 887734, digitColumns := ((List.range 41).map (fun index => 910851 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 910892 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 910933 + 1 * index)) },
   { rowStart := 23691, fieldColumn := 887735, digitColumns := ((List.range 41).map (fun index => 910973 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 911014 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 911055 + 1 * index)) },
   { rowStart := 23815, fieldColumn := 887736, digitColumns := ((List.range 41).map (fun index => 911095 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 911136 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 911177 + 1 * index)) }]

def externalDigitWordStarts : List Nat := ((List.range 972).map (fun index => 402491 + 122 * index)) ++
    ((List.range 54).map (fun index => 651371 + 122 * index)) ++
    ((List.range 54).map (fun index => 666011 + 122 * index)) ++
    ((List.range 54).map (fun index => 680651 + 122 * index)) ++
    ((List.range 54).map (fun index => 695291 + 122 * index)) ++
    ((List.range 54).map (fun index => 709931 + 122 * index)) ++
    ((List.range 54).map (fun index => 724693 + 122 * index)) ++
    ((List.range 54).map (fun index => 739455 + 122 * index)) ++
    ((List.range 54).map (fun index => 754217 + 122 * index)) ++
    ((List.range 54).map (fun index => 768979 + 122 * index)) ++
    ((List.range 54).map (fun index => 783741 + 122 * index)) ++
    ((List.range 54).map (fun index => 798503 + 122 * index))

def seededPhi81SourceColumns : List (List Nat) :=
  [((List.range 9).map (fun index => 887609 + 1 * index)) ++
    ((List.range 972).map (fun index => 371881 + 1 * index)) ++
    [887618, 887619, 887620] ++
    ((List.range 270).map (fun index => 372854 + 1 * index)) ++
    [887621, 373514, 373515, 887622, 887623] ++
    ((List.range 128).map (fun index => 373124 + 1 * index)) ++
    [887624] ++
    ((List.range 128).map (fun index => 373252 + 1 * index)) ++
    [887625] ++
    ((List.range 128).map (fun index => 373380 + 1 * index)) ++
    [887626] ++
    ((List.range 4).map (fun index => 373662 + 1 * index)),
   ((List.range 108).map (fun index => 887629 + 1 * index))]

/-- Raw `ce_claim_digest/v2` fields consumed by the first SIS map. -/
def parentCeClaimSourceColumns : List Nat :=
  seededPhi81SourceColumns.getD 0 []

/-- First SIS commitment fields consumed by the compression map. -/
def parentCeDigestCompressionSourceColumns : List Nat :=
  seededPhi81SourceColumns.getD 1 []

structure SeededPhi81Placement where
  blockIndex : Nat
  rowStart : Nat
  rowEnd : Nat

def seededPhi81Placements : List SeededPhi81Placement :=
  [{ blockIndex := 6, rowStart := 10437, rowEnd := 10545 }, { blockIndex := 7, rowStart := 23939, rowEnd := 23993 }]

structure Segment where
  rowStart : Nat
  rowEnd : Nat
  inputColumns : List Nat
  instructions : List Instruction

def segments : List Segment :=
  [{ rowStart := segment0RowStart, rowEnd := segment0RowEnd, inputColumns := segment0InputColumns, instructions := segment0Instructions },
   { rowStart := segment1RowStart, rowEnd := segment1RowEnd, inputColumns := segment1InputColumns, instructions := segment1Instructions },
   { rowStart := segment2RowStart, rowEnd := segment2RowEnd, inputColumns := segment2InputColumns, instructions := segment2Instructions }]

def seededBlocks : List SeededPhi81.Block :=
  [FPrimeFullHistorySeededPhi81.block6, FPrimeFullHistorySeededPhi81.block7]

theorem seededBlocks_length :
    seededBlocks.length = seededPhi81Placements.length := by native_decide

def rowPieces : List (List Row) :=
  [segment0Rows,
   FPrimeFullHistorySeededPhi81.block6.rows,
   segment1Rows,
   FPrimeFullHistorySeededPhi81.block7.rows,
   segment2Rows]

def rows : List Row := rowPieces.flatten

theorem rows_length : rows.length = rowCount := by
  simp only [rows, rowPieces, List.flatten_cons, List.flatten_nil,
    List.length_append, List.length_nil, segment0_rows_length, segment1_rows_length, segment2_rows_length,
    SeededPhi81.Block.rows_length]
  native_decide
end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCore
