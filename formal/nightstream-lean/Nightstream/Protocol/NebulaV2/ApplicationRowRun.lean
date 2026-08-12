import Nightstream.Protocol.NebulaV2.ApplicationTrace

/-!
Contract: local row-by-row application acceptance for Nebula V2.

Assurance tier: model-level.

Owns an independent lifecycle relation over explicit application rows. A run
starts in a live state, checks zero or more active semantic transitions, checks
exactly one returned or trapped transition, and then permits only canonical
state-preserving padding. Exact public counts and bounds reconstruct
`CompletedExecution`; that conclusion is not a field of the checked input.

Does not own a generated R1CS compiler, WASM interpreter refinement, physical
port decoding, recursive verification, Rust, or deployed proof acceptance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.ApplicationRowRun

open Nightstream.Protocol.NebulaV2.ApplicationTrace
open Nightstream.Protocol.NebulaV2.Completion
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.Ports

/-- Authority-bearing lifecycle state. The terminal outcome is retained, so a
padding row cannot change either the final application state or its outcome. -/
inductive Phase (ApplicationState : Type) where
  | running (state : ApplicationState)
  | terminal (state : ApplicationState) (outcome : Outcome)

/-- One local row check against the verifier-owned application semantics. -/
inductive Transition
    {Program ApplicationState : Type}
    (semantics : Semantics Program ApplicationState) (program : Program) :
    Phase ApplicationState -> ApplicationRow -> Phase ApplicationState -> Prop
  | active
      {before after : ApplicationState} {row : NormalizedRow}
      (accepted : semantics.active program before row after) :
      Transition semantics program (.running before) (.active row)
        (.running after)
  | returned
      {before after : ApplicationState} {row : NormalizedRow}
      {output : Option OutputValue}
      (accepted : semantics.returned program before row output after) :
      Transition semantics program (.running before) (.returned row output)
        (.terminal after (.returned output))
  | trapped
      {before after : ApplicationState} {row : NormalizedRow}
      {trap : Trap}
      (accepted : semantics.trapped program before row trap after) :
      Transition semantics program (.running before) (.trapped row trap)
        (.terminal after (.trapped trap))
  | padding {state : ApplicationState} {outcome : Outcome} :
      Transition semantics program (.terminal state outcome) .padding
        (.terminal state outcome)

/-- Ordered closure of the local row relation. -/
inductive Runs
    {Program ApplicationState : Type}
    (semantics : Semantics Program ApplicationState) (program : Program) :
    Phase ApplicationState -> List ApplicationRow ->
      Phase ApplicationState -> Prop
  | nil (state : Phase ApplicationState) :
      Runs semantics program state [] state
  | cons
      {before middle after : Phase ApplicationState}
      {row : ApplicationRow} {rows : List ApplicationRow}
      (head : Transition semantics program before row middle)
      (tail : Runs semantics program middle rows after) :
      Runs semantics program before (row :: rows) after

namespace Runs

theorem append
    {Program ApplicationState : Type}
    {semantics : Semantics Program ApplicationState} {program : Program}
    {before middle after : Phase ApplicationState}
    {left right : List ApplicationRow}
    (first : Runs semantics program before left middle)
    (second : Runs semantics program middle right after) :
    Runs semantics program before (left ++ right) after := by
  induction first with
  | nil => simpa using second
  | cons head _ inductionHypothesis =>
      exact Runs.cons head (inductionHypothesis second)

/-- A run that starts terminal contains only padding and preserves the exact
state and typed outcome. -/
theorem terminal_inverse
    {Program ApplicationState : Type}
    {semantics : Semantics Program ApplicationState} {program : Program}
    {before after : ApplicationState} {beforeOutcome afterOutcome : Outcome}
    {rows : List ApplicationRow}
    (run : Runs semantics program (.terminal before beforeOutcome) rows
      (.terminal after afterOutcome)) :
    rows = List.replicate rows.length .padding /\
      after = before /\ afterOutcome = beforeOutcome := by
  induction rows generalizing before beforeOutcome after afterOutcome with
  | nil =>
      cases run
      exact ⟨rfl, rfl, rfl⟩
  | cons row rows inductionHypothesis =>
      cases run with
      | cons head tail =>
          cases head with
          | padding =>
              rcases inductionHypothesis tail with
                ⟨rowsExact, stateExact, outcomeExact⟩
              refine ⟨?_, stateExact, outcomeExact⟩
              rw [List.length_cons, List.replicate_succ]
              exact congrArg (List.cons (.padding : ApplicationRow)) rowsExact

/-- Inverse of the local lifecycle. It derives one semantic active prefix, one
typed terminal transition, and only padding after it. -/
theorem complete_inverse
    {Program ApplicationState : Type}
    {semantics : Semantics Program ApplicationState} {program : Program}
    {initial final : ApplicationState} {outcome : Outcome}
    {rows : List ApplicationRow}
    (run : Runs semantics program (.running initial) rows
      (.terminal final outcome)) :
    exists (activeRows : List NormalizedRow)
        (beforeTerminal : ApplicationState) (terminalRow : NormalizedRow)
        (paddingCount : Nat),
      rows = activeRows.map ApplicationRow.active ++
        [terminalApplicationRow terminalRow outcome] ++
        List.replicate paddingCount .padding /\
      ActivePrefix semantics program initial activeRows beforeTerminal /\
      Terminal semantics program beforeTerminal final terminalRow outcome := by
  induction rows generalizing initial with
  | nil => cases run
  | cons row rows inductionHypothesis =>
      cases run with
      | @cons _ middle _ _ _ head tailRun =>
          cases head with
          | @active _ _ activeRow accepted =>
              rcases inductionHypothesis tailRun with
                ⟨activeRows, beforeTerminal, terminalRow, paddingCount,
                  rowsExact, activeTrace, terminal⟩
              exact
                ⟨activeRow :: activeRows, beforeTerminal, terminalRow,
                  paddingCount,
                  by
                    simp only [List.map_cons, List.cons_append]
                    rw [rowsExact],
                  ActivePrefix.cons accepted activeTrace, terminal⟩
          | @returned _ _ terminalRow output accepted =>
              rcases terminal_inverse tailRun with
                ⟨paddingExact, stateExact, outcomeExact⟩
              subst final
              subst outcome
              exact
                ⟨[], _, terminalRow, rows.length,
                  by
                    change ApplicationRow.returned terminalRow output :: rows =
                      ApplicationRow.returned terminalRow output ::
                        List.replicate rows.length .padding
                    exact congrArg
                      (List.cons (ApplicationRow.returned terminalRow output))
                      paddingExact,
                  ActivePrefix.nil _,
                  Terminal.returned output accepted⟩
          | @trapped _ _ terminalRow trap accepted =>
              rcases terminal_inverse tailRun with
                ⟨paddingExact, stateExact, outcomeExact⟩
              subst final
              subst outcome
              exact
                ⟨[], _, terminalRow, rows.length,
                  by
                    change ApplicationRow.trapped terminalRow trap :: rows =
                      ApplicationRow.trapped terminalRow trap ::
                        List.replicate rows.length .padding
                    exact congrArg
                      (List.cons (ApplicationRow.trapped terminalRow trap))
                      paddingExact,
                  ActivePrefix.nil _,
                  Terminal.trapped trap accepted⟩

theorem ofActivePrefix
    {Program ApplicationState : Type}
    {semantics : Semantics Program ApplicationState} {program : Program}
    {before after : ApplicationState} {rows : List NormalizedRow}
    (trace : ActivePrefix semantics program before rows after) :
    Runs semantics program (.running before)
      (rows.map ApplicationRow.active) (.running after) := by
  induction trace with
  | nil => exact Runs.nil _
  | cons step _ inductionHypothesis =>
      exact Runs.cons (.active step) inductionHypothesis

theorem ofTerminal
    {Program ApplicationState : Type}
    {semantics : Semantics Program ApplicationState} {program : Program}
    {before after : ApplicationState} {row : NormalizedRow}
    {outcome : Outcome}
    (terminal : Terminal semantics program before after row outcome) :
    Runs semantics program (.running before)
      [terminalApplicationRow row outcome] (.terminal after outcome) := by
  cases terminal with
  | returned output accepted =>
      exact Runs.cons (.returned accepted) (Runs.nil _)
  | trapped trap accepted =>
      exact Runs.cons (.trapped accepted) (Runs.nil _)

theorem padding
    {Program ApplicationState : Type}
    {semantics : Semantics Program ApplicationState} {program : Program}
    (state : ApplicationState) (outcome : Outcome) (count : Nat) :
    Runs semantics program (.terminal state outcome)
      (List.replicate count .padding) (.terminal state outcome) := by
  induction count with
  | zero => exact Runs.nil _
  | succ count inductionHypothesis =>
      rw [List.replicate_succ]
      exact Runs.cons .padding inductionHypothesis

end Runs

/-- One row contributes to the public real-row count exactly when it is not
padding. -/
def rowCount : ApplicationRow -> Nat
  | .active _ => 1
  | .returned _ _ => 1
  | .trapped _ _ => 1
  | .padding => 0

def realRowCount : List ApplicationRow -> Nat
  | [] => 0
  | head :: tail => rowCount head + realRowCount tail

@[simp] theorem realRowCount_append (left right : List ApplicationRow) :
    realRowCount (left ++ right) = realRowCount left + realRowCount right := by
  induction left with
  | nil => simp [realRowCount]
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, realRowCount]
      rw [inductionHypothesis, Nat.add_assoc]

@[simp] theorem realRowCount_active (rows : List NormalizedRow) :
    realRowCount (rows.map ApplicationRow.active) = rows.length := by
  induction rows with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [realRowCount, rowCount, inductionHypothesis, Nat.add_comm]

@[simp] theorem realRowCount_terminal
    (row : NormalizedRow) (outcome : Outcome) :
    realRowCount [terminalApplicationRow row outcome] = 1 := by
  cases outcome <;> rfl

@[simp] theorem realRowCount_padding (count : Nat) :
    realRowCount (List.replicate count .padding) = 0 := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [List.replicate_succ, realRowCount, rowCount,
        inductionHypothesis]

/-- Canonical normalized view used only to extract the memory ports from raw
application rows. -/
def normalizedRow : ApplicationRow -> NormalizedRow
  | .active row => row
  | .returned row _ => row
  | .trapped row _ => row
  | .padding => NormalizedRow.inactive

@[simp] theorem normalized_terminalApplicationRow
    (row : NormalizedRow) (outcome : Outcome) :
    normalizedRow (terminalApplicationRow row outcome) = row := by
  cases outcome <;> rfl

/-- Raw authority-bearing access lists, split at the fixed application-row
segment boundary. Only the declared real-row prefix contributes. -/
def segmentAccessesOfRows
    (segmentCount realCount : Nat) (rows : List ApplicationRow) :
    List (List Access) :=
  (CompletedExecution.fixedSegmentRows segmentCount
      ((rows.take realCount).map normalizedRow)).map
    (fun segmentRows => segmentRows.flatMap NormalizedRow.accesses)

@[simp] theorem segmentAccessesOfRows_length
    (segmentCount realCount : Nat) (rows : List ApplicationRow) :
    (segmentAccessesOfRows segmentCount realCount rows).length = segmentCount := by
  simp [segmentAccessesOfRows]

/-- Raw local checks sufficient to reconstruct a complete bounded execution.
The semantic conclusion itself is intentionally absent. -/
structure CheckedCompletedRows
    {Program ApplicationState Digest : Type}
    (semantics : Semantics Program ApplicationState)
    (program : Program) (initial : ApplicationState)
    (result : ExecutionResult ApplicationState Digest)
    (segmentCount : Nat) where
  rows : List ApplicationRow
  run : Runs semantics program (.running initial) rows
    (.terminal result.finalApplicationState result.outcome)
  realRowCountExact : realRowCount rows = result.realApplicationRowCount
  rowsLengthExact : rows.length = segmentCapacity segmentCount
  segmentCountPositive : 0 < segmentCount
  segmentCountBound : segmentCount <= maximumSegments
  realRowCountBound : result.realApplicationRowCount < realApplicationRowLimit
  smallestSegmentCount :
    segmentCount = minimumSegmentCount result.realApplicationRowCount

namespace CheckedCompletedRows

/-- The explicit local run and public checks reconstruct a completed semantic
execution and retain the exact source row list. -/
theorem completedExecution
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (checked : CheckedCompletedRows semantics program initial result segmentCount) :
    exists execution :
        CompletedExecution semantics program initial result segmentCount,
      execution.rows = checked.rows := by
  rcases checked.run.complete_inverse with
    ⟨activeRows, beforeTerminal, terminalRow, paddingCount,
      rowsExact, activeTrace, terminal⟩
  have realCountExact :
      result.realApplicationRowCount = activeRows.length + 1 := by
    rw [← checked.realRowCountExact, rowsExact]
    rw [realRowCount_append, realRowCount_append, realRowCount_active,
      realRowCount_terminal, realRowCount_padding]
  have fitsDeclaredSegments :
      result.realApplicationRowCount <= segmentCapacity segmentCount := by
    rw [← checked.realRowCountExact, ← checked.rowsLengthExact]
    rw [rowsExact]
    rw [realRowCount_append, realRowCount_append, realRowCount_active,
      realRowCount_terminal, realRowCount_padding]
    simp only [List.length_append, List.length_map, List.length_singleton,
      List.length_replicate]
    omega
  have paddingExact :
      paddingCount =
        segmentCapacity segmentCount - result.realApplicationRowCount := by
    have lengthExact := checked.rowsLengthExact
    rw [rowsExact] at lengthExact
    simp only [List.length_append, List.length_map, List.length_singleton,
      List.length_replicate] at lengthExact
    omega
  let execution : CompletedExecution semantics program initial result
      segmentCount :=
    { real :=
        { activeRows := activeRows
          beforeTerminal := beforeTerminal
          activeTrace := activeTrace
          terminalRow := terminalRow
          terminal := terminal }
      realRowCountExact := realCountExact
      segmentCountPositive := checked.segmentCountPositive
      segmentCountBound := checked.segmentCountBound
      realRowCountBound := checked.realRowCountBound
      fitsDeclaredSegments := fitsDeclaredSegments
      smallestSegmentCount := checked.smallestSegmentCount }
  refine ⟨execution, ?_⟩
  rw [rowsExact, paddingExact]
  rfl

/-- The real prefix of a completed row list normalizes to the exact semantic
row list used by `CompletedExecution.segmentAccesses`. -/
theorem normalized_real_prefix
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount) :
    ((execution.rows.take result.realApplicationRowCount).map normalizedRow) =
      execution.real.rows := by
  have takeRows :
      execution.rows.take result.realApplicationRowCount =
        execution.real.activeRows.map ApplicationRow.active ++
          [terminalApplicationRow execution.real.terminalRow result.outcome] := by
    unfold CompletedExecution.rows
    rw [execution.realRowCountExact]
    let rowPrefix := execution.real.activeRows.map ApplicationRow.active ++
      [terminalApplicationRow execution.real.terminalRow result.outcome]
    have prefixLength :
        rowPrefix.length = execution.real.activeRows.length + 1 := by
      simp [rowPrefix]
    change List.take (execution.real.activeRows.length + 1)
      (rowPrefix ++ List.replicate
        (segmentCapacity segmentCount - (execution.real.activeRows.length + 1))
        .padding) = _
    rw [← prefixLength]
    rw [List.take_append_of_le_length (Nat.le_refl rowPrefix.length)]
    simp [rowPrefix]
  rw [takeRows]
  unfold RealExecution.rows
  rw [List.map_append]
  simp only [List.map_map, List.map_singleton]
  have activeExact :
      List.map (normalizedRow ∘ ApplicationRow.active)
          execution.real.activeRows =
        execution.real.activeRows := by
    induction execution.real.activeRows with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp [normalizedRow, inductionHypothesis]
  rw [activeExact]
  rw [normalized_terminalApplicationRow]

/-- Raw row segmentation agrees with the independently defined completed
execution segmentation. -/
theorem segmentAccessesOfRows_execution
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount) :
    segmentAccessesOfRows segmentCount result.realApplicationRowCount
        execution.rows =
      execution.segmentAccesses := by
  unfold segmentAccessesOfRows CompletedExecution.segmentAccesses
  rw [normalized_real_prefix execution]

/-- Honest completed execution data constructs the raw local-row checker. This
is the completeness direction only. -/
def ofCompletedExecution
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount) :
    CheckedCompletedRows semantics program initial result segmentCount := by
  let activeRun := Runs.ofActivePrefix execution.real.activeTrace
  let terminalRun := Runs.ofTerminal execution.real.terminal
  let paddingRun := Runs.padding (semantics := semantics) (program := program)
    result.finalApplicationState result.outcome
      (segmentCapacity segmentCount - result.realApplicationRowCount)
  refine
    { rows := execution.rows
      run := ?_
      realRowCountExact := ?_
      rowsLengthExact := ?_
      segmentCountPositive := execution.segmentCountPositive
      segmentCountBound := execution.segmentCountBound
      realRowCountBound := execution.realRowCountBound
      smallestSegmentCount := execution.smallestSegmentCount }
  · simpa [CompletedExecution.rows] using activeRun.append
      (terminalRun.append paddingRun)
  · unfold CompletedExecution.rows
    rw [execution.realRowCountExact]
    rw [realRowCount_append, realRowCount_append, realRowCount_active,
      realRowCount_terminal, realRowCount_padding]
  · have exactLength := Completion.valid_trace_has_exact_capacity
        execution.validCompletedTrace
    simpa using exactLength

end CheckedCompletedRows

end Nightstream.Protocol.NebulaV2.ApplicationRowRun
