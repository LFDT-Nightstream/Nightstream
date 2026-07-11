import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Instructions0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Instructions1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Instructions2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Instructions3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Instructions4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Instructions5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Instructions6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Instructions7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Instructions8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Inputs0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Inputs1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Inputs2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Inputs3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Inputs4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Inputs5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Inputs6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Inputs7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Inputs8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment0Inputs9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Instructions0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Instructions1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Instructions2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Instructions3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Instructions4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Instructions5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Instructions6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Instructions7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Instructions8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Instructions9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Instructions10
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Instructions11
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Inputs0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Inputs1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Inputs2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Inputs3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Inputs4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Inputs5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Inputs6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Inputs7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Inputs8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Inputs9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Inputs10
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment1Inputs11
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Instructions0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Instructions1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Instructions2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Instructions3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Instructions4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Instructions5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Instructions6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Instructions7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Instructions8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Instructions9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Instructions10
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Instructions11
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSegment2Inputs0
import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Artifact

/-! Exact checked program for the terminal post-fold accumulator owner. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram

set_option maxRecDepth 1048576

def parentCeDigestColumns : List Nat := [3204951, 3204952, 3204953, 3204954]
def accumulatorDigestColumns : List Nat := [3207977, 3207978, 3207979, 3207980]
def rowStart : Nat := 3517890
def rowEnd : Nat := 3555185
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
  [{ rowStart := 21, fieldColumn := 3171075, digitColumns := ((List.range 41).map (fun index => 3171203 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3171244 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3171285 + 1 * index)) },
   { rowStart := 145, fieldColumn := 3171076, digitColumns := ((List.range 41).map (fun index => 3171325 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3171366 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3171407 + 1 * index)) },
   { rowStart := 269, fieldColumn := 3171077, digitColumns := ((List.range 41).map (fun index => 3171447 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3171488 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3171529 + 1 * index)) },
   { rowStart := 393, fieldColumn := 3171078, digitColumns := ((List.range 41).map (fun index => 3171569 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3171610 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3171651 + 1 * index)) },
   { rowStart := 517, fieldColumn := 3171079, digitColumns := ((List.range 41).map (fun index => 3171691 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3171732 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3171773 + 1 * index)) },
   { rowStart := 641, fieldColumn := 3171080, digitColumns := ((List.range 41).map (fun index => 3171813 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3171854 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3171895 + 1 * index)) },
   { rowStart := 765, fieldColumn := 3171081, digitColumns := ((List.range 41).map (fun index => 3171935 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3171976 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3172017 + 1 * index)) },
   { rowStart := 889, fieldColumn := 3171082, digitColumns := ((List.range 41).map (fun index => 3172057 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3172098 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3172139 + 1 * index)) },
   { rowStart := 1013, fieldColumn := 3171083, digitColumns := ((List.range 41).map (fun index => 3172179 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3172220 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3172261 + 1 * index)) },
   { rowStart := 1137, fieldColumn := 3171084, digitColumns := ((List.range 41).map (fun index => 3172301 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3172342 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3172383 + 1 * index)) },
   { rowStart := 1261, fieldColumn := 3171085, digitColumns := ((List.range 41).map (fun index => 3172423 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3172464 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3172505 + 1 * index)) },
   { rowStart := 1385, fieldColumn := 3171086, digitColumns := ((List.range 41).map (fun index => 3172545 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3172586 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3172627 + 1 * index)) },
   { rowStart := 1509, fieldColumn := 3171087, digitColumns := ((List.range 41).map (fun index => 3172667 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3172708 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3172749 + 1 * index)) },
   { rowStart := 1633, fieldColumn := 2611124, digitColumns := ((List.range 41).map (fun index => 3172789 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3172830 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3172871 + 1 * index)) },
   { rowStart := 1757, fieldColumn := 2611125, digitColumns := ((List.range 41).map (fun index => 3172911 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3172952 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3172993 + 1 * index)) },
   { rowStart := 1881, fieldColumn := 3171088, digitColumns := ((List.range 41).map (fun index => 3173033 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3173074 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3173115 + 1 * index)) },
   { rowStart := 2005, fieldColumn := 3171089, digitColumns := ((List.range 41).map (fun index => 3173155 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3173196 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3173237 + 1 * index)) },
   { rowStart := 2129, fieldColumn := 2610842, digitColumns := ((List.range 41).map (fun index => 3173277 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3173318 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3173359 + 1 * index)) },
   { rowStart := 2253, fieldColumn := 2610843, digitColumns := ((List.range 41).map (fun index => 3173399 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3173440 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3173481 + 1 * index)) },
   { rowStart := 2377, fieldColumn := 2610844, digitColumns := ((List.range 41).map (fun index => 3173521 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3173562 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3173603 + 1 * index)) },
   { rowStart := 2501, fieldColumn := 2610845, digitColumns := ((List.range 41).map (fun index => 3173643 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3173684 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3173725 + 1 * index)) },
   { rowStart := 2625, fieldColumn := 2610846, digitColumns := ((List.range 41).map (fun index => 3173765 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3173806 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3173847 + 1 * index)) },
   { rowStart := 2749, fieldColumn := 2610847, digitColumns := ((List.range 41).map (fun index => 3173887 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3173928 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3173969 + 1 * index)) },
   { rowStart := 2873, fieldColumn := 2610848, digitColumns := ((List.range 41).map (fun index => 3174009 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3174050 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3174091 + 1 * index)) },
   { rowStart := 2997, fieldColumn := 2610849, digitColumns := ((List.range 41).map (fun index => 3174131 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3174172 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3174213 + 1 * index)) },
   { rowStart := 3121, fieldColumn := 2610850, digitColumns := ((List.range 41).map (fun index => 3174253 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3174294 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3174335 + 1 * index)) },
   { rowStart := 3245, fieldColumn := 2610851, digitColumns := ((List.range 41).map (fun index => 3174375 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3174416 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3174457 + 1 * index)) },
   { rowStart := 3369, fieldColumn := 2610852, digitColumns := ((List.range 41).map (fun index => 3174497 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3174538 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3174579 + 1 * index)) },
   { rowStart := 3493, fieldColumn := 2610853, digitColumns := ((List.range 41).map (fun index => 3174619 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3174660 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3174701 + 1 * index)) },
   { rowStart := 3617, fieldColumn := 2610854, digitColumns := ((List.range 41).map (fun index => 3174741 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3174782 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3174823 + 1 * index)) },
   { rowStart := 3741, fieldColumn := 2610855, digitColumns := ((List.range 41).map (fun index => 3174863 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3174904 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3174945 + 1 * index)) },
   { rowStart := 3865, fieldColumn := 2610856, digitColumns := ((List.range 41).map (fun index => 3174985 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3175026 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3175067 + 1 * index)) },
   { rowStart := 3989, fieldColumn := 2610857, digitColumns := ((List.range 41).map (fun index => 3175107 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3175148 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3175189 + 1 * index)) },
   { rowStart := 4113, fieldColumn := 2610858, digitColumns := ((List.range 41).map (fun index => 3175229 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3175270 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3175311 + 1 * index)) },
   { rowStart := 4237, fieldColumn := 2610859, digitColumns := ((List.range 41).map (fun index => 3175351 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3175392 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3175433 + 1 * index)) },
   { rowStart := 4361, fieldColumn := 2610860, digitColumns := ((List.range 41).map (fun index => 3175473 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3175514 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3175555 + 1 * index)) },
   { rowStart := 4485, fieldColumn := 2610861, digitColumns := ((List.range 41).map (fun index => 3175595 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3175636 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3175677 + 1 * index)) },
   { rowStart := 4609, fieldColumn := 3171090, digitColumns := ((List.range 41).map (fun index => 3175717 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3175758 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3175799 + 1 * index)) },
   { rowStart := 4733, fieldColumn := 2610970, digitColumns := ((List.range 41).map (fun index => 3175839 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3175880 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3175921 + 1 * index)) },
   { rowStart := 4857, fieldColumn := 2610971, digitColumns := ((List.range 41).map (fun index => 3175961 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3176002 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3176043 + 1 * index)) },
   { rowStart := 4981, fieldColumn := 2610972, digitColumns := ((List.range 41).map (fun index => 3176083 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3176124 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3176165 + 1 * index)) },
   { rowStart := 5105, fieldColumn := 2610973, digitColumns := ((List.range 41).map (fun index => 3176205 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3176246 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3176287 + 1 * index)) },
   { rowStart := 5229, fieldColumn := 2610974, digitColumns := ((List.range 41).map (fun index => 3176327 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3176368 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3176409 + 1 * index)) },
   { rowStart := 5353, fieldColumn := 2610975, digitColumns := ((List.range 41).map (fun index => 3176449 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3176490 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3176531 + 1 * index)) },
   { rowStart := 5477, fieldColumn := 2610976, digitColumns := ((List.range 41).map (fun index => 3176571 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3176612 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3176653 + 1 * index)) },
   { rowStart := 5601, fieldColumn := 2610977, digitColumns := ((List.range 41).map (fun index => 3176693 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3176734 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3176775 + 1 * index)) },
   { rowStart := 5725, fieldColumn := 2610978, digitColumns := ((List.range 41).map (fun index => 3176815 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3176856 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3176897 + 1 * index)) },
   { rowStart := 5849, fieldColumn := 2610979, digitColumns := ((List.range 41).map (fun index => 3176937 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3176978 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3177019 + 1 * index)) },
   { rowStart := 5973, fieldColumn := 2610980, digitColumns := ((List.range 41).map (fun index => 3177059 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3177100 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3177141 + 1 * index)) },
   { rowStart := 6097, fieldColumn := 2610981, digitColumns := ((List.range 41).map (fun index => 3177181 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3177222 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3177263 + 1 * index)) },
   { rowStart := 6221, fieldColumn := 2610982, digitColumns := ((List.range 41).map (fun index => 3177303 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3177344 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3177385 + 1 * index)) },
   { rowStart := 6345, fieldColumn := 2610983, digitColumns := ((List.range 41).map (fun index => 3177425 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3177466 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3177507 + 1 * index)) },
   { rowStart := 6469, fieldColumn := 2610984, digitColumns := ((List.range 41).map (fun index => 3177547 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3177588 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3177629 + 1 * index)) },
   { rowStart := 6593, fieldColumn := 2610985, digitColumns := ((List.range 41).map (fun index => 3177669 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3177710 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3177751 + 1 * index)) },
   { rowStart := 6717, fieldColumn := 2610986, digitColumns := ((List.range 41).map (fun index => 3177791 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3177832 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3177873 + 1 * index)) },
   { rowStart := 6841, fieldColumn := 2610987, digitColumns := ((List.range 41).map (fun index => 3177913 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3177954 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3177995 + 1 * index)) },
   { rowStart := 6965, fieldColumn := 2610988, digitColumns := ((List.range 41).map (fun index => 3178035 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3178076 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3178117 + 1 * index)) },
   { rowStart := 7089, fieldColumn := 2610989, digitColumns := ((List.range 41).map (fun index => 3178157 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3178198 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3178239 + 1 * index)) },
   { rowStart := 7213, fieldColumn := 3171091, digitColumns := ((List.range 41).map (fun index => 3178279 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3178320 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3178361 + 1 * index)) },
   { rowStart := 7337, fieldColumn := 2611098, digitColumns := ((List.range 41).map (fun index => 3178401 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3178442 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3178483 + 1 * index)) },
   { rowStart := 7461, fieldColumn := 2611099, digitColumns := ((List.range 41).map (fun index => 3178523 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3178564 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3178605 + 1 * index)) },
   { rowStart := 7585, fieldColumn := 2611100, digitColumns := ((List.range 41).map (fun index => 3178645 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3178686 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3178727 + 1 * index)) },
   { rowStart := 7709, fieldColumn := 2611101, digitColumns := ((List.range 41).map (fun index => 3178767 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3178808 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3178849 + 1 * index)) },
   { rowStart := 7833, fieldColumn := 2611102, digitColumns := ((List.range 41).map (fun index => 3178889 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3178930 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3178971 + 1 * index)) },
   { rowStart := 7957, fieldColumn := 2611103, digitColumns := ((List.range 41).map (fun index => 3179011 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3179052 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3179093 + 1 * index)) },
   { rowStart := 8081, fieldColumn := 2611104, digitColumns := ((List.range 41).map (fun index => 3179133 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3179174 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3179215 + 1 * index)) },
   { rowStart := 8205, fieldColumn := 2611105, digitColumns := ((List.range 41).map (fun index => 3179255 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3179296 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3179337 + 1 * index)) },
   { rowStart := 8329, fieldColumn := 2611106, digitColumns := ((List.range 41).map (fun index => 3179377 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3179418 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3179459 + 1 * index)) },
   { rowStart := 8453, fieldColumn := 2611107, digitColumns := ((List.range 41).map (fun index => 3179499 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3179540 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3179581 + 1 * index)) },
   { rowStart := 8577, fieldColumn := 2611108, digitColumns := ((List.range 41).map (fun index => 3179621 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3179662 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3179703 + 1 * index)) },
   { rowStart := 8701, fieldColumn := 2611109, digitColumns := ((List.range 41).map (fun index => 3179743 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3179784 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3179825 + 1 * index)) },
   { rowStart := 8825, fieldColumn := 2611110, digitColumns := ((List.range 41).map (fun index => 3179865 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3179906 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3179947 + 1 * index)) },
   { rowStart := 8949, fieldColumn := 2611111, digitColumns := ((List.range 41).map (fun index => 3179987 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3180028 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3180069 + 1 * index)) },
   { rowStart := 9073, fieldColumn := 2611112, digitColumns := ((List.range 41).map (fun index => 3180109 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3180150 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3180191 + 1 * index)) },
   { rowStart := 9197, fieldColumn := 2611113, digitColumns := ((List.range 41).map (fun index => 3180231 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3180272 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3180313 + 1 * index)) },
   { rowStart := 9321, fieldColumn := 2611114, digitColumns := ((List.range 41).map (fun index => 3180353 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3180394 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3180435 + 1 * index)) },
   { rowStart := 9445, fieldColumn := 2611115, digitColumns := ((List.range 41).map (fun index => 3180475 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3180516 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3180557 + 1 * index)) },
   { rowStart := 9569, fieldColumn := 2611116, digitColumns := ((List.range 41).map (fun index => 3180597 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3180638 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3180679 + 1 * index)) },
   { rowStart := 9693, fieldColumn := 2611117, digitColumns := ((List.range 41).map (fun index => 3180719 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3180760 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3180801 + 1 * index)) },
   { rowStart := 9817, fieldColumn := 3171092, digitColumns := ((List.range 41).map (fun index => 3180841 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3180882 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3180923 + 1 * index)) },
   { rowStart := 9941, fieldColumn := 2611272, digitColumns := ((List.range 41).map (fun index => 3180963 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3181004 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3181045 + 1 * index)) },
   { rowStart := 10065, fieldColumn := 2611273, digitColumns := ((List.range 41).map (fun index => 3181085 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3181126 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3181167 + 1 * index)) },
   { rowStart := 10189, fieldColumn := 2611274, digitColumns := ((List.range 41).map (fun index => 3181207 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3181248 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3181289 + 1 * index)) },
   { rowStart := 10313, fieldColumn := 2611275, digitColumns := ((List.range 41).map (fun index => 3181329 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3181370 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3181411 + 1 * index)) },
   { rowStart := 10547, fieldColumn := 3171095, digitColumns := ((List.range 41).map (fun index => 3181507 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3181548 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3181589 + 1 * index)) },
   { rowStart := 10671, fieldColumn := 3171096, digitColumns := ((List.range 41).map (fun index => 3181629 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3181670 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3181711 + 1 * index)) },
   { rowStart := 10795, fieldColumn := 3171097, digitColumns := ((List.range 41).map (fun index => 3181751 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3181792 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3181833 + 1 * index)) },
   { rowStart := 10919, fieldColumn := 3171098, digitColumns := ((List.range 41).map (fun index => 3181873 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3181914 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3181955 + 1 * index)) },
   { rowStart := 11043, fieldColumn := 3171099, digitColumns := ((List.range 41).map (fun index => 3181995 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3182036 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3182077 + 1 * index)) },
   { rowStart := 11167, fieldColumn := 3171100, digitColumns := ((List.range 41).map (fun index => 3182117 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3182158 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3182199 + 1 * index)) },
   { rowStart := 11291, fieldColumn := 3171101, digitColumns := ((List.range 41).map (fun index => 3182239 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3182280 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3182321 + 1 * index)) },
   { rowStart := 11415, fieldColumn := 3171102, digitColumns := ((List.range 41).map (fun index => 3182361 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3182402 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3182443 + 1 * index)) },
   { rowStart := 11539, fieldColumn := 3171103, digitColumns := ((List.range 41).map (fun index => 3182483 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3182524 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3182565 + 1 * index)) },
   { rowStart := 11663, fieldColumn := 3171104, digitColumns := ((List.range 41).map (fun index => 3182605 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3182646 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3182687 + 1 * index)) },
   { rowStart := 11787, fieldColumn := 3171105, digitColumns := ((List.range 41).map (fun index => 3182727 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3182768 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3182809 + 1 * index)) },
   { rowStart := 11911, fieldColumn := 3171106, digitColumns := ((List.range 41).map (fun index => 3182849 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3182890 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3182931 + 1 * index)) },
   { rowStart := 12035, fieldColumn := 3171107, digitColumns := ((List.range 41).map (fun index => 3182971 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3183012 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3183053 + 1 * index)) },
   { rowStart := 12159, fieldColumn := 3171108, digitColumns := ((List.range 41).map (fun index => 3183093 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3183134 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3183175 + 1 * index)) },
   { rowStart := 12283, fieldColumn := 3171109, digitColumns := ((List.range 41).map (fun index => 3183215 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3183256 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3183297 + 1 * index)) },
   { rowStart := 12407, fieldColumn := 3171110, digitColumns := ((List.range 41).map (fun index => 3183337 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3183378 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3183419 + 1 * index)) },
   { rowStart := 12531, fieldColumn := 3171111, digitColumns := ((List.range 41).map (fun index => 3183459 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3183500 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3183541 + 1 * index)) },
   { rowStart := 12655, fieldColumn := 3171112, digitColumns := ((List.range 41).map (fun index => 3183581 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3183622 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3183663 + 1 * index)) },
   { rowStart := 12779, fieldColumn := 3171113, digitColumns := ((List.range 41).map (fun index => 3183703 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3183744 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3183785 + 1 * index)) },
   { rowStart := 12903, fieldColumn := 3171114, digitColumns := ((List.range 41).map (fun index => 3183825 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3183866 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3183907 + 1 * index)) },
   { rowStart := 13027, fieldColumn := 3171115, digitColumns := ((List.range 41).map (fun index => 3183947 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3183988 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3184029 + 1 * index)) },
   { rowStart := 13151, fieldColumn := 3171116, digitColumns := ((List.range 41).map (fun index => 3184069 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3184110 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3184151 + 1 * index)) },
   { rowStart := 13275, fieldColumn := 3171117, digitColumns := ((List.range 41).map (fun index => 3184191 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3184232 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3184273 + 1 * index)) },
   { rowStart := 13399, fieldColumn := 3171118, digitColumns := ((List.range 41).map (fun index => 3184313 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3184354 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3184395 + 1 * index)) },
   { rowStart := 13523, fieldColumn := 3171119, digitColumns := ((List.range 41).map (fun index => 3184435 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3184476 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3184517 + 1 * index)) },
   { rowStart := 13647, fieldColumn := 3171120, digitColumns := ((List.range 41).map (fun index => 3184557 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3184598 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3184639 + 1 * index)) },
   { rowStart := 13771, fieldColumn := 3171121, digitColumns := ((List.range 41).map (fun index => 3184679 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3184720 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3184761 + 1 * index)) },
   { rowStart := 13895, fieldColumn := 3171122, digitColumns := ((List.range 41).map (fun index => 3184801 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3184842 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3184883 + 1 * index)) },
   { rowStart := 14019, fieldColumn := 3171123, digitColumns := ((List.range 41).map (fun index => 3184923 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3184964 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3185005 + 1 * index)) },
   { rowStart := 14143, fieldColumn := 3171124, digitColumns := ((List.range 41).map (fun index => 3185045 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3185086 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3185127 + 1 * index)) },
   { rowStart := 14267, fieldColumn := 3171125, digitColumns := ((List.range 41).map (fun index => 3185167 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3185208 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3185249 + 1 * index)) },
   { rowStart := 14391, fieldColumn := 3171126, digitColumns := ((List.range 41).map (fun index => 3185289 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3185330 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3185371 + 1 * index)) },
   { rowStart := 14515, fieldColumn := 3171127, digitColumns := ((List.range 41).map (fun index => 3185411 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3185452 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3185493 + 1 * index)) },
   { rowStart := 14639, fieldColumn := 3171128, digitColumns := ((List.range 41).map (fun index => 3185533 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3185574 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3185615 + 1 * index)) },
   { rowStart := 14763, fieldColumn := 3171129, digitColumns := ((List.range 41).map (fun index => 3185655 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3185696 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3185737 + 1 * index)) },
   { rowStart := 14887, fieldColumn := 3171130, digitColumns := ((List.range 41).map (fun index => 3185777 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3185818 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3185859 + 1 * index)) },
   { rowStart := 15011, fieldColumn := 3171131, digitColumns := ((List.range 41).map (fun index => 3185899 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3185940 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3185981 + 1 * index)) },
   { rowStart := 15135, fieldColumn := 3171132, digitColumns := ((List.range 41).map (fun index => 3186021 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3186062 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3186103 + 1 * index)) },
   { rowStart := 15259, fieldColumn := 3171133, digitColumns := ((List.range 41).map (fun index => 3186143 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3186184 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3186225 + 1 * index)) },
   { rowStart := 15383, fieldColumn := 3171134, digitColumns := ((List.range 41).map (fun index => 3186265 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3186306 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3186347 + 1 * index)) },
   { rowStart := 15507, fieldColumn := 3171135, digitColumns := ((List.range 41).map (fun index => 3186387 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3186428 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3186469 + 1 * index)) },
   { rowStart := 15631, fieldColumn := 3171136, digitColumns := ((List.range 41).map (fun index => 3186509 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3186550 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3186591 + 1 * index)) },
   { rowStart := 15755, fieldColumn := 3171137, digitColumns := ((List.range 41).map (fun index => 3186631 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3186672 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3186713 + 1 * index)) },
   { rowStart := 15879, fieldColumn := 3171138, digitColumns := ((List.range 41).map (fun index => 3186753 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3186794 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3186835 + 1 * index)) },
   { rowStart := 16003, fieldColumn := 3171139, digitColumns := ((List.range 41).map (fun index => 3186875 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3186916 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3186957 + 1 * index)) },
   { rowStart := 16127, fieldColumn := 3171140, digitColumns := ((List.range 41).map (fun index => 3186997 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3187038 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3187079 + 1 * index)) },
   { rowStart := 16251, fieldColumn := 3171141, digitColumns := ((List.range 41).map (fun index => 3187119 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3187160 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3187201 + 1 * index)) },
   { rowStart := 16375, fieldColumn := 3171142, digitColumns := ((List.range 41).map (fun index => 3187241 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3187282 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3187323 + 1 * index)) },
   { rowStart := 16499, fieldColumn := 3171143, digitColumns := ((List.range 41).map (fun index => 3187363 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3187404 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3187445 + 1 * index)) },
   { rowStart := 16623, fieldColumn := 3171144, digitColumns := ((List.range 41).map (fun index => 3187485 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3187526 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3187567 + 1 * index)) },
   { rowStart := 16747, fieldColumn := 3171145, digitColumns := ((List.range 41).map (fun index => 3187607 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3187648 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3187689 + 1 * index)) },
   { rowStart := 16871, fieldColumn := 3171146, digitColumns := ((List.range 41).map (fun index => 3187729 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3187770 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3187811 + 1 * index)) },
   { rowStart := 16995, fieldColumn := 3171147, digitColumns := ((List.range 41).map (fun index => 3187851 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3187892 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3187933 + 1 * index)) },
   { rowStart := 17119, fieldColumn := 3171148, digitColumns := ((List.range 41).map (fun index => 3187973 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3188014 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3188055 + 1 * index)) },
   { rowStart := 17243, fieldColumn := 3171149, digitColumns := ((List.range 41).map (fun index => 3188095 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3188136 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3188177 + 1 * index)) },
   { rowStart := 17367, fieldColumn := 3171150, digitColumns := ((List.range 41).map (fun index => 3188217 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3188258 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3188299 + 1 * index)) },
   { rowStart := 17491, fieldColumn := 3171151, digitColumns := ((List.range 41).map (fun index => 3188339 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3188380 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3188421 + 1 * index)) },
   { rowStart := 17615, fieldColumn := 3171152, digitColumns := ((List.range 41).map (fun index => 3188461 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3188502 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3188543 + 1 * index)) },
   { rowStart := 17739, fieldColumn := 3171153, digitColumns := ((List.range 41).map (fun index => 3188583 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3188624 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3188665 + 1 * index)) },
   { rowStart := 17863, fieldColumn := 3171154, digitColumns := ((List.range 41).map (fun index => 3188705 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3188746 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3188787 + 1 * index)) },
   { rowStart := 17987, fieldColumn := 3171155, digitColumns := ((List.range 41).map (fun index => 3188827 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3188868 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3188909 + 1 * index)) },
   { rowStart := 18111, fieldColumn := 3171156, digitColumns := ((List.range 41).map (fun index => 3188949 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3188990 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3189031 + 1 * index)) },
   { rowStart := 18235, fieldColumn := 3171157, digitColumns := ((List.range 41).map (fun index => 3189071 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3189112 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3189153 + 1 * index)) },
   { rowStart := 18359, fieldColumn := 3171158, digitColumns := ((List.range 41).map (fun index => 3189193 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3189234 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3189275 + 1 * index)) },
   { rowStart := 18483, fieldColumn := 3171159, digitColumns := ((List.range 41).map (fun index => 3189315 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3189356 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3189397 + 1 * index)) },
   { rowStart := 18607, fieldColumn := 3171160, digitColumns := ((List.range 41).map (fun index => 3189437 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3189478 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3189519 + 1 * index)) },
   { rowStart := 18731, fieldColumn := 3171161, digitColumns := ((List.range 41).map (fun index => 3189559 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3189600 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3189641 + 1 * index)) },
   { rowStart := 18855, fieldColumn := 3171162, digitColumns := ((List.range 41).map (fun index => 3189681 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3189722 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3189763 + 1 * index)) },
   { rowStart := 18979, fieldColumn := 3171163, digitColumns := ((List.range 41).map (fun index => 3189803 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3189844 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3189885 + 1 * index)) },
   { rowStart := 19103, fieldColumn := 3171164, digitColumns := ((List.range 41).map (fun index => 3189925 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3189966 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3190007 + 1 * index)) },
   { rowStart := 19227, fieldColumn := 3171165, digitColumns := ((List.range 41).map (fun index => 3190047 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3190088 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3190129 + 1 * index)) },
   { rowStart := 19351, fieldColumn := 3171166, digitColumns := ((List.range 41).map (fun index => 3190169 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3190210 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3190251 + 1 * index)) },
   { rowStart := 19475, fieldColumn := 3171167, digitColumns := ((List.range 41).map (fun index => 3190291 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3190332 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3190373 + 1 * index)) },
   { rowStart := 19599, fieldColumn := 3171168, digitColumns := ((List.range 41).map (fun index => 3190413 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3190454 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3190495 + 1 * index)) },
   { rowStart := 19723, fieldColumn := 3171169, digitColumns := ((List.range 41).map (fun index => 3190535 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3190576 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3190617 + 1 * index)) },
   { rowStart := 19847, fieldColumn := 3171170, digitColumns := ((List.range 41).map (fun index => 3190657 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3190698 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3190739 + 1 * index)) },
   { rowStart := 19971, fieldColumn := 3171171, digitColumns := ((List.range 41).map (fun index => 3190779 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3190820 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3190861 + 1 * index)) },
   { rowStart := 20095, fieldColumn := 3171172, digitColumns := ((List.range 41).map (fun index => 3190901 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3190942 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3190983 + 1 * index)) },
   { rowStart := 20219, fieldColumn := 3171173, digitColumns := ((List.range 41).map (fun index => 3191023 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3191064 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3191105 + 1 * index)) },
   { rowStart := 20343, fieldColumn := 3171174, digitColumns := ((List.range 41).map (fun index => 3191145 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3191186 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3191227 + 1 * index)) },
   { rowStart := 20467, fieldColumn := 3171175, digitColumns := ((List.range 41).map (fun index => 3191267 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3191308 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3191349 + 1 * index)) },
   { rowStart := 20591, fieldColumn := 3171176, digitColumns := ((List.range 41).map (fun index => 3191389 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3191430 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3191471 + 1 * index)) },
   { rowStart := 20715, fieldColumn := 3171177, digitColumns := ((List.range 41).map (fun index => 3191511 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3191552 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3191593 + 1 * index)) },
   { rowStart := 20839, fieldColumn := 3171178, digitColumns := ((List.range 41).map (fun index => 3191633 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3191674 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3191715 + 1 * index)) },
   { rowStart := 20963, fieldColumn := 3171179, digitColumns := ((List.range 41).map (fun index => 3191755 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3191796 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3191837 + 1 * index)) },
   { rowStart := 21087, fieldColumn := 3171180, digitColumns := ((List.range 41).map (fun index => 3191877 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3191918 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3191959 + 1 * index)) },
   { rowStart := 21211, fieldColumn := 3171181, digitColumns := ((List.range 41).map (fun index => 3191999 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3192040 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3192081 + 1 * index)) },
   { rowStart := 21335, fieldColumn := 3171182, digitColumns := ((List.range 41).map (fun index => 3192121 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3192162 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3192203 + 1 * index)) },
   { rowStart := 21459, fieldColumn := 3171183, digitColumns := ((List.range 41).map (fun index => 3192243 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3192284 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3192325 + 1 * index)) },
   { rowStart := 21583, fieldColumn := 3171184, digitColumns := ((List.range 41).map (fun index => 3192365 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3192406 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3192447 + 1 * index)) },
   { rowStart := 21707, fieldColumn := 3171185, digitColumns := ((List.range 41).map (fun index => 3192487 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3192528 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3192569 + 1 * index)) },
   { rowStart := 21831, fieldColumn := 3171186, digitColumns := ((List.range 41).map (fun index => 3192609 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3192650 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3192691 + 1 * index)) },
   { rowStart := 21955, fieldColumn := 3171187, digitColumns := ((List.range 41).map (fun index => 3192731 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3192772 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3192813 + 1 * index)) },
   { rowStart := 22079, fieldColumn := 3171188, digitColumns := ((List.range 41).map (fun index => 3192853 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3192894 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3192935 + 1 * index)) },
   { rowStart := 22203, fieldColumn := 3171189, digitColumns := ((List.range 41).map (fun index => 3192975 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3193016 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3193057 + 1 * index)) },
   { rowStart := 22327, fieldColumn := 3171190, digitColumns := ((List.range 41).map (fun index => 3193097 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3193138 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3193179 + 1 * index)) },
   { rowStart := 22451, fieldColumn := 3171191, digitColumns := ((List.range 41).map (fun index => 3193219 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3193260 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3193301 + 1 * index)) },
   { rowStart := 22575, fieldColumn := 3171192, digitColumns := ((List.range 41).map (fun index => 3193341 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3193382 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3193423 + 1 * index)) },
   { rowStart := 22699, fieldColumn := 3171193, digitColumns := ((List.range 41).map (fun index => 3193463 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3193504 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3193545 + 1 * index)) },
   { rowStart := 22823, fieldColumn := 3171194, digitColumns := ((List.range 41).map (fun index => 3193585 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3193626 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3193667 + 1 * index)) },
   { rowStart := 22947, fieldColumn := 3171195, digitColumns := ((List.range 41).map (fun index => 3193707 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3193748 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3193789 + 1 * index)) },
   { rowStart := 23071, fieldColumn := 3171196, digitColumns := ((List.range 41).map (fun index => 3193829 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3193870 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3193911 + 1 * index)) },
   { rowStart := 23195, fieldColumn := 3171197, digitColumns := ((List.range 41).map (fun index => 3193951 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3193992 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3194033 + 1 * index)) },
   { rowStart := 23319, fieldColumn := 3171198, digitColumns := ((List.range 41).map (fun index => 3194073 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3194114 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3194155 + 1 * index)) },
   { rowStart := 23443, fieldColumn := 3171199, digitColumns := ((List.range 41).map (fun index => 3194195 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3194236 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3194277 + 1 * index)) },
   { rowStart := 23567, fieldColumn := 3171200, digitColumns := ((List.range 41).map (fun index => 3194317 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3194358 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3194399 + 1 * index)) },
   { rowStart := 23691, fieldColumn := 3171201, digitColumns := ((List.range 41).map (fun index => 3194439 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3194480 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3194521 + 1 * index)) },
   { rowStart := 23815, fieldColumn := 3171202, digitColumns := ((List.range 41).map (fun index => 3194561 + 1 * index)), negativeColumns := ((List.range 41).map (fun index => 3194602 + 1 * index)), borrowColumns := ((List.range 40).map (fun index => 3194643 + 1 * index)) }]

def externalDigitWordStarts : List Nat := ((List.range 972).map (fun index => 2640101 + 122 * index)) ++
    ((List.range 54).map (fun index => 2888981 + 122 * index)) ++
    ((List.range 54).map (fun index => 2903621 + 122 * index)) ++
    ((List.range 54).map (fun index => 2918261 + 122 * index)) ++
    ((List.range 54).map (fun index => 2932901 + 122 * index)) ++
    ((List.range 54).map (fun index => 2947541 + 122 * index)) ++
    ((List.range 54).map (fun index => 2962303 + 122 * index)) ++
    ((List.range 54).map (fun index => 2977065 + 122 * index)) ++
    ((List.range 54).map (fun index => 2991827 + 122 * index)) ++
    ((List.range 54).map (fun index => 3006589 + 122 * index)) ++
    ((List.range 54).map (fun index => 3021351 + 122 * index)) ++
    ((List.range 54).map (fun index => 3036113 + 122 * index))

def seededPhi81SourceColumns : List (List Nat) :=
  [((List.range 9).map (fun index => 3171075 + 1 * index)) ++
    ((List.range 972).map (fun index => 2609491 + 1 * index)) ++
    [3171084, 3171085, 3171086] ++
    ((List.range 270).map (fun index => 2610464 + 1 * index)) ++
    [3171087, 2611124, 2611125, 3171088, 3171089] ++
    ((List.range 128).map (fun index => 2610734 + 1 * index)) ++
    [3171090] ++
    ((List.range 128).map (fun index => 2610862 + 1 * index)) ++
    [3171091] ++
    ((List.range 128).map (fun index => 2610990 + 1 * index)) ++
    [3171092] ++
    ((List.range 4).map (fun index => 2611272 + 1 * index)),
   ((List.range 108).map (fun index => 3171095 + 1 * index))]

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
  [{ blockIndex := 16, rowStart := 10437, rowEnd := 10545 }, { blockIndex := 17, rowStart := 23939, rowEnd := 23993 }]

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
  [FPrimeFullHistorySeededPhi81.block16, FPrimeFullHistorySeededPhi81.block17]

theorem seededBlocks_length :
    seededBlocks.length = seededPhi81Placements.length := by native_decide

def rowPieces : List (List Row) :=
  [segment0Rows,
   FPrimeFullHistorySeededPhi81.block16.rows,
   segment1Rows,
   FPrimeFullHistorySeededPhi81.block17.rows,
   segment2Rows]

def rows : List Row := rowPieces.flatten

theorem rows_length : rows.length = rowCount := by
  simp only [rows, rowPieces, List.flatten_cons, List.flatten_nil,
    List.length_append, List.length_nil, segment0_rows_length, segment1_rows_length, segment2_rows_length,
    SeededPhi81.Block.rows_length]
  native_decide
end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator
