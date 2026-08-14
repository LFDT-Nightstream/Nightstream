import Nightstream.Protocol.Nebula

set_option autoImplicit false

namespace tests.NebulaCompletion

open Nightstream.Protocol.Nebula.Completion

def returnedResult : ExecutionResult Unit Nat :=
  { realApplicationRowCount := 1
    finalApplicationState := ()
    outcome := .returned none
    finalMemoryRoot := 9 }

def oneSegmentRows : List RowKind := canonicalRows returnedResult 1

def oneSegmentCompletion :
    ValidCompletedTrace returnedResult 1 oneSegmentRows where
  segmentCountPositive := by decide
  segmentCountBound := by decide
  realRowCountPositive := by decide
  realRowCountBound := by decide
  fitsDeclaredSegments := by decide
  smallestSegmentCount := by decide
  rowsCanonical := rfl

theorem one_segment_rows_have_exact_capacity :
    oneSegmentRows.length = segmentCapacity 1 :=
  valid_trace_has_exact_capacity oneSegmentCompletion

theorem returned_exit_code_is_zero : returnedResult.outcome.exitCode = 0 :=
  rfl

def trappedResult : ExecutionResult Unit Nat :=
  { realApplicationRowCount := 7
    finalApplicationState := ()
    outcome := .trapped .memoryOutOfBounds
    finalMemoryRoot := 10 }

theorem memory_trap_exit_code_is_seven :
    trappedResult.outcome.exitCode = 7 :=
  rfl

theorem post_completion_rows_are_padding :
    (canonicalRows returnedResult 1).drop 1 =
      List.replicate (segmentCapacity 1 - 1) .padding := by
  rfl

/- Without minimality, the same completed result accepts one segment or two
segments. The second trace contains one redundant full segment of padding. -/
namespace MissingMinimality

def validAtOne :
    ValidWithoutMinimality returnedResult 1
      (canonicalRows returnedResult 1) where
  segmentCountPositive := by decide
  segmentCountBound := by decide
  realRowCountPositive := by decide
  realRowCountBound := by decide
  fitsDeclaredSegments := by decide
  rowsCanonical := rfl

def validAtTwo :
    ValidWithoutMinimality returnedResult 2
      (canonicalRows returnedResult 2) where
  segmentCountPositive := by decide
  segmentCountBound := by decide
  realRowCountPositive := by decide
  realRowCountBound := by decide
  fitsDeclaredSegments := by decide
  rowsCanonical := rfl

theorem minimal_rule_rejects_two_segments :
    ¬ ∃ rows, ValidCompletedTrace returnedResult 2 rows := by
  rintro ⟨_rows, valid⟩
  have smallest := valid.smallestSegmentCount
  change 2 = 1 at smallest
  omega

end MissingMinimality

theorem profile_maximum_rows :
    segmentCapacity 64 = 208896 ∧ 208896 < 2 ^ 18 :=
  ⟨maximum_application_rows, maximum_application_rows_fit_18_bits⟩

end tests.NebulaCompletion
