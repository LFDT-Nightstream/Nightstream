import Nightstream.Protocol.NebulaV2.ApplicationBatch

/-!
Contract: reverse extraction of a completed execution from exact application
batch rows.

Assurance tier: model-level.

Owns the reverse direction missing from `ApplicationBatch`: an exact run over
an explicit active-prefix, one typed terminal row, and canonical padding
reconstructs `CompletedExecution`. The theorem does not infer terminal payloads
from row-kind tags alone.

Does not own generated row decoding, WASM implementation refinement, memory
soundness, recursive verification, or a terminal proof system.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Protocol.NebulaV2.ApplicationBatchCompletion

open Nightstream.Protocol.NebulaV2.ApplicationBatch
open Nightstream.Protocol.NebulaV2.ApplicationTrace
open Nightstream.Protocol.NebulaV2.Completion
open Nightstream.Protocol.NebulaV2.Ports
open Nightstream.Protocol.NebulaV2.WasmState

/-- Real-row counting is additive over exact row concatenation. -/
theorem realRowCount_append (left right : List ApplicationRow) :
    ApplicationBatch.realRowCount (left ++ right) =
      ApplicationBatch.realRowCount left +
        ApplicationBatch.realRowCount right := by
  induction left with
  | nil => simp [ApplicationBatch.realRowCount]
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, ApplicationBatch.realRowCount]
      rw [inductionHypothesis, Nat.add_assoc]

@[simp] theorem realRowCount_active_rows (rows : List NormalizedRow) :
    ApplicationBatch.realRowCount (rows.map ApplicationRow.active) =
      rows.length := by
  induction rows with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [ApplicationBatch.realRowCount, ApplicationBatch.rowCount,
        inductionHypothesis, Nat.add_comm]

@[simp] theorem realRowCount_terminal
    (row : NormalizedRow) (outcome : Outcome) :
    ApplicationBatch.realRowCount [terminalApplicationRow row outcome] = 1 := by
  cases outcome <;> rfl

@[simp] theorem realRowCount_padding (count : Nat) :
    ApplicationBatch.realRowCount (List.replicate count .padding) = 0 := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [List.replicate_succ, ApplicationBatch.realRowCount,
        ApplicationBatch.rowCount, inductionHypothesis]

namespace Runs

/-- A run over only canonical padding cannot change the state or the real-row
count. This is the reverse form of `ApplicationBatch.Runs.padding`. -/
theorem padding_inverse
    {Program : Type} {machine : Machine Program} {program : Program}
    {before after : AppStateVector} {paddingCount count : Nat}
    (run : ApplicationBatch.Runs machine program before
      (List.replicate paddingCount .padding) after count) :
    after = before /\ count = 0 := by
  induction paddingCount generalizing before after count with
  | zero =>
      simp only [List.replicate_zero] at run
      cases run
      exact ⟨rfl, rfl⟩
  | succ paddingCount inductionHypothesis =>
      rw [List.replicate_succ] at run
      cases run with
      | cons head tail =>
          cases head with
          | padding terminal =>
              rcases inductionHypothesis tail with ⟨afterExact, countExact⟩
              subst after
              rw [countExact]
              exact ⟨rfl, rfl⟩

/-- Once a run starts in a terminal state, every remaining row is canonical
padding, the state is unchanged, and the real-row count is zero. -/
theorem after_terminal_is_padding
    {Program : Type} {machine : Machine Program} {program : Program}
    {before after : AppStateVector} {rows : List ApplicationRow} {count : Nat}
    (run : ApplicationBatch.Runs machine program before rows after count)
    (terminal : before.TerminalReady) :
    rows = List.replicate rows.length .padding /\
      after = before /\ count = 0 := by
  induction run with
  | nil => simp
  | @cons before middle after row rows headCount tailCount head tail
      inductionHypothesis =>
      rcases head.from_terminal_is_padding terminal with
        ⟨rowExact, middleExact, headCountExact⟩
      subst row
      subst middle
      subst headCount
      rcases inductionHypothesis terminal with
        ⟨rowsExact, afterExact, tailCountExact⟩
      subst after
      rw [tailCountExact]
      refine ⟨?_, rfl, rfl⟩
      calc
        .padding :: rows =
            .padding :: List.replicate rows.length .padding :=
          congrArg (List.cons (.padding : ApplicationRow)) rowsExact
        _ = List.replicate (.padding :: rows).length .padding := by
          simp only [List.length_cons, List.replicate_succ]

/-- An exact run from a nonterminal state to a typed terminal state has one
active prefix, one authority-bearing terminal row, and only padding after it.
The terminal payload is derived from the final state. -/
theorem terminal_shape_inverse
    {Program : Type} {machine : Machine Program} {program : Program}
    {initial final : AppStateVector} {rows : List ApplicationRow} {count : Nat}
    {outcome : Outcome}
    (run : ApplicationBatch.Runs machine program initial rows final count)
    (initialNotTerminal : ¬ initial.TerminalReady)
    (finalTerminal : final.Terminal outcome) :
    exists (activeRows : List NormalizedRow)
      (terminalRow : NormalizedRow) (paddingCount : Nat),
      rows = activeRows.map ApplicationRow.active ++
        [terminalApplicationRow terminalRow outcome] ++
        List.replicate paddingCount .padding /\
      count = activeRows.length + 1 := by
  induction run with
  | nil => exact False.elim (initialNotTerminal finalTerminal.ready)
  | @cons before middle after row rows headCount tailCount head tail
      inductionHypothesis =>
      cases head with
      | @active middle normalizedRow accepted =>
          rcases inductionHypothesis accepted.2.2.2.1 finalTerminal with
            ⟨activeRows, terminalRow, paddingCount, rowsExact, countExact⟩
          refine ⟨normalizedRow :: activeRows, terminalRow, paddingCount,
            ?_, ?_⟩
          · rw [rowsExact]
            rfl
          · simp only [List.length_cons]
            omega
      | @returned middle row output accepted =>
          have actualTerminal := accepted.2.2.2.2
          rcases after_terminal_is_padding tail actualTerminal.ready with
            ⟨rowsExact, finalExact, tailCountExact⟩
          have expectedAtMiddle : middle.Terminal outcome := by
            rw [← finalExact]
            exact finalTerminal
          have outcomeExact : Outcome.returned output = outcome :=
            actualTerminal.outcome_unique expectedAtMiddle
          refine ⟨[], row, rows.length, ?_, ?_⟩
          · rw [← outcomeExact]
            change ApplicationRow.returned row output :: rows =
              ApplicationRow.returned row output ::
                List.replicate rows.length .padding
            exact congrArg
              (List.cons (ApplicationRow.returned row output)) rowsExact
          · simp only [List.length_nil]
            omega
      | @trapped middle row trap accepted =>
          have actualTerminal := accepted.2.2.2.2
          rcases after_terminal_is_padding tail actualTerminal.ready with
            ⟨rowsExact, finalExact, tailCountExact⟩
          have expectedAtMiddle : middle.Terminal outcome := by
            rw [← finalExact]
            exact finalTerminal
          have outcomeExact : Outcome.trapped trap = outcome :=
            actualTerminal.outcome_unique expectedAtMiddle
          refine ⟨[], row, rows.length, ?_, ?_⟩
          · rw [← outcomeExact]
            change ApplicationRow.trapped row trap :: rows =
              ApplicationRow.trapped row trap ::
                List.replicate rows.length .padding
            exact congrArg
              (List.cons (ApplicationRow.trapped row trap)) rowsExact
          · simp only [List.length_nil]
            omega
      | padding terminal => exact False.elim (initialNotTerminal terminal)

/-- Exact execution of an explicit completed-row shape reconstructs the
semantic active prefix and typed terminal transition. The result is stronger
than a row-kind classification because the terminal output or trap is part of
the indexed row itself. -/
theorem exact_shape_inverse
    {Program : Type} {machine : Machine Program} {program : Program}
    {initial final : AppStateVector} {activeRows : List NormalizedRow}
    {terminalRow : NormalizedRow} {outcome : Outcome}
    {paddingCount count : Nat}
    (run : ApplicationBatch.Runs machine program initial
      (activeRows.map ApplicationRow.active ++
        [terminalApplicationRow terminalRow outcome] ++
        List.replicate paddingCount .padding)
      final count) :
    exists beforeTerminal,
      ActivePrefix machine.semantics program initial activeRows beforeTerminal /\
        ApplicationTrace.Terminal machine.semantics program beforeTerminal
          final terminalRow outcome /\
        count = activeRows.length + 1 := by
  induction activeRows generalizing initial count with
  | nil =>
      cases run with
      | cons head tail =>
          have paddingExact := padding_inverse tail
          rcases paddingExact with ⟨finalExact, tailCountExact⟩
          subst final
          cases outcome with
          | returned output =>
              cases head with
              | returned accepted =>
                  refine
                    ⟨_, ActivePrefix.nil _,
                      ApplicationTrace.Terminal.returned output accepted, ?_⟩
                  simp [tailCountExact]
          | trapped trap =>
              cases head with
              | trapped accepted =>
                  refine
                    ⟨_, ActivePrefix.nil _,
                      ApplicationTrace.Terminal.trapped trap accepted, ?_⟩
                  simp [tailCountExact]
  | cons row rest inductionHypothesis =>
      cases run with
      | cons head tail =>
          cases head with
          | active accepted =>
              rcases inductionHypothesis tail with
                ⟨beforeTerminal, activeTrace, terminal, countExact⟩
              refine ⟨beforeTerminal, ActivePrefix.cons accepted activeTrace,
                terminal, ?_⟩
              simp only [List.length_cons]
              omega

end Runs

/-- Reverse completion theorem for the exact authority-bearing row list.
Unlike a theorem over `ApplicationRow.kind`, this premise fixes the terminal
payload and each normalized row used by the operational relation. -/
theorem completedExecution_of_exact_rows
    {Program Digest : Type} {machine : Machine Program}
    {program : Program} {initial : AppStateVector}
    {result : ExecutionResult AppStateVector Digest}
    {segmentCount : Nat} {activeRows : List NormalizedRow}
    {terminalRow : NormalizedRow}
    (run : ApplicationBatch.Runs machine program initial
      (activeRows.map ApplicationRow.active ++
        [terminalApplicationRow terminalRow result.outcome] ++
        List.replicate
          (segmentCapacity segmentCount - result.realApplicationRowCount)
          .padding)
      result.finalApplicationState result.realApplicationRowCount)
    (realRowCountExact :
      result.realApplicationRowCount = activeRows.length + 1)
    (segmentCountPositive : 0 < segmentCount)
    (segmentCountBound : segmentCount <= Lifecycle.maximumSegments)
    (realRowCountBound :
      result.realApplicationRowCount < realApplicationRowLimit)
    (fitsDeclaredSegments :
      result.realApplicationRowCount <= segmentCapacity segmentCount)
    (smallestSegmentCount :
      segmentCount = minimumSegmentCount result.realApplicationRowCount) :
    Nonempty
      (CompletedExecution machine.semantics program initial result
        segmentCount) := by
  rcases Runs.exact_shape_inverse run with
    ⟨beforeTerminal, activeTrace, terminal, countExact⟩
  exact
    ⟨{ real :=
         { activeRows := activeRows
           beforeTerminal := beforeTerminal
           activeTrace := activeTrace
           terminalRow := terminalRow
           terminal := terminal }
       realRowCountExact := realRowCountExact
       segmentCountPositive := segmentCountPositive
       segmentCountBound := segmentCountBound
       realRowCountBound := realRowCountBound
       fitsDeclaredSegments := fitsDeclaredSegments
       smallestSegmentCount := smallestSegmentCount }⟩

/-- A completed execution together with the exact lifecycle row list from
which it was reconstructed. -/
structure ExactCompletedRun
    {Program Digest : Type} (machine : Machine Program) (program : Program)
    (initial : AppStateVector) (result : ExecutionResult AppStateVector Digest)
    (segmentCount : Nat) (rows : List ApplicationRow) where
  execution : CompletedExecution machine.semantics program initial result
    segmentCount
  rowsExact : execution.rows = rows

/-- Reverse completion with the exact source row list retained. -/
theorem exactCompletedRun_of_terminal_run
    {Program Digest : Type} {machine : Machine Program}
    {program : Program} {initial : AppStateVector}
    {result : ExecutionResult AppStateVector Digest}
    {segmentCount : Nat} {rows : List ApplicationRow}
    (run : ApplicationBatch.Runs machine program initial rows
      result.finalApplicationState result.realApplicationRowCount)
    (finalTerminal : result.finalApplicationState.Terminal result.outcome)
    (rowsLengthExact : rows.length = segmentCapacity segmentCount)
    (segmentCountPositive : 0 < segmentCount)
    (segmentCountBound : segmentCount <= Lifecycle.maximumSegments)
    (realRowCountPositive : 0 < result.realApplicationRowCount)
    (realRowCountBound :
      result.realApplicationRowCount < realApplicationRowLimit)
    (fitsDeclaredSegments :
      result.realApplicationRowCount <= segmentCapacity segmentCount)
    (smallestSegmentCount :
      segmentCount = minimumSegmentCount result.realApplicationRowCount) :
    Nonempty (ExactCompletedRun machine program initial result segmentCount
      rows) := by
  have initialNotTerminal : ¬ initial.TerminalReady := by
    intro initialTerminal
    have padding := Runs.after_terminal_is_padding run initialTerminal
    omega
  rcases Runs.terminal_shape_inverse run initialNotTerminal finalTerminal with
    ⟨activeRows, terminalRow, paddingCount, rowsExact, countExact⟩
  have shapeLength := congrArg List.length rowsExact
  simp only [List.length_append, List.length_map, List.length_singleton,
    List.length_replicate] at shapeLength
  have paddingExact : paddingCount =
      segmentCapacity segmentCount - result.realApplicationRowCount := by
    omega
  have canonicalRowsExact := rowsExact
  rw [paddingExact] at canonicalRowsExact
  rw [rowsExact, paddingExact] at run
  rcases Runs.exact_shape_inverse run with
    ⟨beforeTerminal, activeTrace, terminal, _countExact⟩
  let execution : CompletedExecution machine.semantics program initial result
      segmentCount :=
    { real :=
        { activeRows := activeRows
          beforeTerminal := beforeTerminal
          activeTrace := activeTrace
          terminalRow := terminalRow
          terminal := terminal }
      realRowCountExact := countExact
      segmentCountPositive := segmentCountPositive
      segmentCountBound := segmentCountBound
      realRowCountBound := realRowCountBound
      fitsDeclaredSegments := fitsDeclaredSegments
      smallestSegmentCount := smallestSegmentCount }
  exact ⟨
    { execution := execution
      rowsExact := by
        simpa [execution, CompletedExecution.rows] using
          canonicalRowsExact.symm }⟩

/-- The explicit completion-row shape is derivable from an operational run
that ends in the public typed terminal state and fills the declared segment
capacity. This removes completion syntax as an independent soundness
assumption. -/
theorem completedExecution_of_terminal_run
    {Program Digest : Type} {machine : Machine Program}
    {program : Program} {initial : AppStateVector}
    {result : ExecutionResult AppStateVector Digest}
    {segmentCount : Nat} {rows : List ApplicationRow}
    (run : ApplicationBatch.Runs machine program initial rows
      result.finalApplicationState result.realApplicationRowCount)
    (finalTerminal : result.finalApplicationState.Terminal result.outcome)
    (rowsLengthExact : rows.length = segmentCapacity segmentCount)
    (segmentCountPositive : 0 < segmentCount)
    (segmentCountBound : segmentCount <= Lifecycle.maximumSegments)
    (realRowCountPositive : 0 < result.realApplicationRowCount)
    (realRowCountBound :
      result.realApplicationRowCount < realApplicationRowLimit)
    (fitsDeclaredSegments :
      result.realApplicationRowCount <= segmentCapacity segmentCount)
    (smallestSegmentCount :
      segmentCount = minimumSegmentCount result.realApplicationRowCount) :
    Nonempty
      (CompletedExecution machine.semantics program initial result
        segmentCount) := by
  rcases exactCompletedRun_of_terminal_run run finalTerminal rowsLengthExact
      segmentCountPositive segmentCountBound realRowCountPositive
      realRowCountBound fitsDeclaredSegments smallestSegmentCount with
    ⟨completed⟩
  exact ⟨completed.execution⟩

/-- The lifecycle-row access view of a completed execution is its semantic
real-row access view. Canonical padding contributes no accesses. -/
theorem ExactCompletedRun.accessesExact
    {Program Digest : Type} {machine : Machine Program}
    {program : Program} {initial : AppStateVector}
    {result : ExecutionResult AppStateVector Digest}
    {segmentCount : Nat} {rows : List ApplicationRow}
    (completed : ExactCompletedRun machine program initial result segmentCount
      rows) :
    ApplicationBatch.accesses rows = completed.execution.accesses := by
  calc
    ApplicationBatch.accesses rows =
        ApplicationBatch.accesses completed.execution.rows :=
      congrArg ApplicationBatch.accesses completed.rowsExact.symm
    _ = completed.execution.accesses := by
      rw [CompletedExecution.rows, ApplicationBatch.accesses_append,
        ApplicationBatch.accesses_append,
        ApplicationBatch.active_rows_accesses,
        ApplicationBatch.padding_rows_accesses]
      have terminalAccesses :
          ApplicationBatch.accesses
              [terminalApplicationRow completed.execution.real.terminalRow
                result.outcome] =
            completed.execution.real.terminalRow.accesses := by
        simp only [ApplicationBatch.accesses, List.flatMap_cons,
          List.flatMap_nil, List.append_nil]
        exact ApplicationBatch.terminal_row_accesses _ _
      rw [terminalAccesses]
      simp [CompletedExecution.accesses, RealExecution.accesses,
        RealExecution.rows]

/-- Forward and reverse semantic equivalence for one already-typed completed
execution. This theorem is useful when generated rows have first been decoded
to the exact `CompletedExecution.rows` value. -/
theorem exact_rows_iff_completedExecution
    {Program Digest : Type} {machine : Machine Program}
    {program : Program} {initial : AppStateVector}
    {result : ExecutionResult AppStateVector Digest}
    {segmentCount : Nat}
    {execution : CompletedExecution machine.semantics program initial result
      segmentCount} :
    ApplicationBatch.Runs machine program initial execution.rows
      result.finalApplicationState result.realApplicationRowCount := by
  exact ApplicationBatch.Runs.ofCompletedExecution execution

end Nightstream.Protocol.NebulaV2.ApplicationBatchCompletion
