import Nightstream.Protocol.NebulaV2.ApplicationTrace

set_option autoImplicit false

namespace tests.NebulaV2ApplicationTrace

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ApplicationTrace
open Nightstream.Protocol.NebulaV2.Completion
open Nightstream.Protocol.NebulaV2.Ports

def semantics : Semantics Unit Nat where
  active := fun _ before _ after => after = before + 1
  returned := fun _ before _ _ after => after = before
  trapped := fun _ before _ _ after => after = before

def real : RealExecution semantics () 0 0 (.returned none) where
  activeRows := []
  beforeTerminal := 0
  activeTrace := .nil 0
  terminalRow := NormalizedRow.inactive
  terminal := .returned none rfl

def result : ExecutionResult Nat Nat :=
  { realApplicationRowCount := 1
    finalApplicationState := 0
    outcome := .returned none
    finalMemoryRoot := 9 }

def completed : CompletedExecution semantics () 0 result 1 where
  real := real
  realRowCountExact := rfl
  segmentCountPositive := by decide
  segmentCountBound := by decide
  realRowCountBound := by decide
  fitsDeclaredSegments := by decide
  smallestSegmentCount := by decide

theorem completed_trace_is_canonical :
    ValidCompletedTrace result 1
      (completed.rows.map ApplicationRow.kind) :=
  completed.validCompletedTrace

theorem no_memory_ports_means_no_semantic_accesses :
    completed.accesses = [] :=
  rfl

theorem all_post_terminal_rows_are_padding :
    completed.rows.drop 1 =
      List.replicate (segmentCapacity 1 - 1) .padding :=
  completed.rowsAfterRealArePadding

/-- The port bridge rejects a claimed memory access for this no-memory
execution. -/
theorem cannot_hide_an_unported_access (access : Access) :
    ¬ completed.CoversMemory [[access]] := by
  simp [CompletedExecution.CoversMemory,
    CompletedExecution.segmentAccesses,
    CompletedExecution.fixedSegmentRows, RealExecution.rows,
    NormalizedRow.inactive, NormalizedRow.accesses, applicationRowsPerSegment,
    applicationRowsPerClaim, Lifecycle.claimsPerSegment, completed, real]

theorem canonical_partition_has_one_empty_segment :
    completed.segmentAccesses = [[]] := by
  rfl

/- A caller-selected permissive semantics can authenticate any result. This
is a direct countermodel for replacing the verifier-owned application
relation with an opaque `Unit` relation. It is not used by the V2 fixture or
the ideal theorem. -/
namespace PermissiveSemantics

def semantics : Semantics Unit Unit where
  active := fun _ _ _ _ => True
  returned := fun _ _ _ _ _ => True
  trapped := fun _ _ _ _ _ => True

def arbitraryOutput : OutputValue :=
  { low := 99
    high := 88
    lowInRange := by decide
    highInRange := by decide }

def arbitraryExecution :
    RealExecution semantics () () () (.returned (some arbitraryOutput)) where
  activeRows := []
  beforeTerminal := ()
  activeTrace := .nil ()
  terminalRow := NormalizedRow.inactive
  terminal := .returned (some arbitraryOutput) True.intro

theorem opaque_unit_semantics_accepts_arbitrary_result :
    Nonempty
      (RealExecution semantics () () ()
        (.returned (some arbitraryOutput))) :=
  ⟨arbitraryExecution⟩

end PermissiveSemantics

end tests.NebulaV2ApplicationTrace
