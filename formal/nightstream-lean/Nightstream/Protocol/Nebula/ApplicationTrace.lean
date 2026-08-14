import Nightstream.Protocol.Nebula.Completion
import Nightstream.Protocol.Nebula.Ports

/-!
Contract: completed application execution with fixed memory ports for V2.

Assurance tier: model-level.

Owns active normalized-row execution, one authenticated return or trap,
canonical state-preserving padding, exact public result fields, and equality
between all semantic memory effects and the ordered fixed-port access list.

Does not own the concrete WASM instruction semantics, generated row table,
Rust interpreter refinement, or result byte codec.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.ApplicationTrace

open Nightstream.Protocol.Nebula.Completion
open Nightstream.Protocol.Nebula.Lifecycle
open Nightstream.Protocol.Nebula.Ports

/-- Verifier-owned operational relation for the selected application. -/
structure Semantics (Program ApplicationState : Type) where
  active :
    Program → ApplicationState → NormalizedRow → ApplicationState → Prop
  returned :
    Program → ApplicationState → NormalizedRow → Option OutputValue →
      ApplicationState → Prop
  trapped :
    Program → ApplicationState → NormalizedRow → Trap →
      ApplicationState → Prop

/-- Ordered execution of all nonterminal normalized rows. Every memory effect
is owned by the row's 21 fixed ports. -/
inductive ActivePrefix
    {Program ApplicationState : Type}
    (semantics : Semantics Program ApplicationState)
    (program : Program) :
    ApplicationState → List NormalizedRow → ApplicationState → Prop
  | nil (state : ApplicationState) :
      ActivePrefix semantics program state [] state
  | cons
      {before middle final : ApplicationState}
      {row : NormalizedRow} {rest : List NormalizedRow}
      (step : semantics.active program before row middle)
      (tail : ActivePrefix semantics program middle rest final) :
      ActivePrefix semantics program before (row :: rest) final

inductive Terminal
    {Program ApplicationState : Type}
    (semantics : Semantics Program ApplicationState)
    (program : Program) (before final : ApplicationState)
    (row : NormalizedRow) : Outcome → Prop
  | returned
      (output : Option OutputValue)
      (step : semantics.returned program before row output final) :
      Terminal semantics program before final row (.returned output)
  | trapped
      (trap : Trap)
      (step : semantics.trapped program before row trap final) :
      Terminal semantics program before final row (.trapped trap)

/-- One complete real execution has zero or more active rows and exactly one
terminal row. -/
structure RealExecution
    {Program ApplicationState : Type}
    (semantics : Semantics Program ApplicationState)
    (program : Program) (initial final : ApplicationState)
    (outcome : Outcome) where
  activeRows : List NormalizedRow
  beforeTerminal : ApplicationState
  activeTrace : ActivePrefix semantics program initial activeRows beforeTerminal
  terminalRow : NormalizedRow
  terminal : Terminal semantics program beforeTerminal final terminalRow outcome

namespace RealExecution

/-- Every real normalized row, including the authority-bearing terminal row. -/
def rows
    {Program ApplicationState : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial final : ApplicationState}
    {outcome : Outcome}
    (execution : RealExecution semantics program initial final outcome) :
    List NormalizedRow :=
  execution.activeRows ++ [execution.terminalRow]

def accesses
    {Program ApplicationState : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial final : ApplicationState}
    {outcome : Outcome}
    (execution : RealExecution semantics program initial final outcome) :
    List Access :=
  execution.rows.flatMap NormalizedRow.accesses

@[simp]
theorem rows_length
    {Program ApplicationState : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial final : ApplicationState}
    {outcome : Outcome}
    (execution : RealExecution semantics program initial final outcome) :
    execution.rows.length = execution.activeRows.length + 1 := by
  simp [rows]

end RealExecution

inductive ApplicationRow where
  | active (row : NormalizedRow)
  | returned (row : NormalizedRow) (output : Option OutputValue)
  | trapped (row : NormalizedRow) (trap : Trap)
  | padding

@[simp]
def ApplicationRow.kind : ApplicationRow → RowKind
  | .active _ => .active
  | .returned _ _ => .returned
  | .trapped _ _ => .trapped
  | .padding => .padding

def terminalApplicationRow
    (row : NormalizedRow) : Outcome → ApplicationRow
  | .returned output => .returned row output
  | .trapped trap => .trapped row trap

@[simp]
theorem terminalApplicationRow_kind
    (row : NormalizedRow) (outcome : Outcome) :
    (terminalApplicationRow row outcome).kind = outcome.terminalRow := by
  cases outcome <;> rfl

/-- Full bounded trace. Padding rows contain no state or port fields, so they
cannot mutate the completed state and cannot contain a memory access. -/
structure CompletedExecution
    {Program ApplicationState Digest : Type}
    (semantics : Semantics Program ApplicationState)
    (program : Program) (initial : ApplicationState)
    (result : ExecutionResult ApplicationState Digest)
    (segmentCount : Nat) where
  real : RealExecution semantics program initial
    result.finalApplicationState result.outcome
  realRowCountExact :
    result.realApplicationRowCount = real.activeRows.length + 1
  segmentCountPositive : 0 < segmentCount
  segmentCountBound : segmentCount ≤ maximumSegments
  realRowCountBound : result.realApplicationRowCount < realApplicationRowLimit
  fitsDeclaredSegments :
    result.realApplicationRowCount ≤ segmentCapacity segmentCount
  smallestSegmentCount :
    segmentCount = minimumSegmentCount result.realApplicationRowCount

namespace CompletedExecution

def rows
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount) :
    List ApplicationRow :=
  execution.real.activeRows.map .active ++
    [terminalApplicationRow execution.real.terminalRow result.outcome] ++
    List.replicate
      (segmentCapacity segmentCount - result.realApplicationRowCount)
      .padding

def accesses
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount) :
    List Access :=
  execution.real.accesses

theorem rowKindsCanonical
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount) :
    execution.rows.map ApplicationRow.kind =
      canonicalRows result segmentCount := by
  have activeKinds :
      (execution.real.activeRows.map ApplicationRow.active).map
          ApplicationRow.kind =
        List.replicate execution.real.activeRows.length .active := by
    induction execution.real.activeRows with
    | nil => rfl
    | cons row rest inductionHypothesis =>
        simp [inductionHypothesis, List.replicate_succ]
  unfold rows canonicalRows
  rw [execution.realRowCountExact]
  rw [List.map_append, List.map_append, activeKinds]
  rw [List.map_singleton, terminalApplicationRow_kind]
  simp

theorem validCompletedTrace
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount) :
    ValidCompletedTrace result segmentCount
      (execution.rows.map ApplicationRow.kind) where
  segmentCountPositive := execution.segmentCountPositive
  segmentCountBound := execution.segmentCountBound
  realRowCountPositive := by rw [execution.realRowCountExact]; omega
  realRowCountBound := execution.realRowCountBound
  fitsDeclaredSegments := execution.fitsDeclaredSegments
  smallestSegmentCount := execution.smallestSegmentCount
  rowsCanonical := execution.rowKindsCanonical

theorem rowsAfterRealArePadding
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount) :
    execution.rows.drop result.realApplicationRowCount =
      List.replicate
        (segmentCapacity segmentCount - result.realApplicationRowCount)
        .padding := by
  unfold rows
  rw [execution.realRowCountExact]
  let rowPrefix := execution.real.activeRows.map ApplicationRow.active ++
    [terminalApplicationRow execution.real.terminalRow result.outcome]
  have prefixLength :
      rowPrefix.length = execution.real.activeRows.length + 1 := by
    simp [rowPrefix]
  change List.drop (execution.real.activeRows.length + 1)
      (rowPrefix ++ List.replicate
        (segmentCapacity segmentCount -
          (execution.real.activeRows.length + 1)) .padding) = _
  rw [← prefixLength]
  simp

/-- Split active rows into exactly `segmentCount` consecutive pieces. Empty
pieces are retained. This makes the segment boundary part of the statement. -/
def fixedSegmentRows : Nat → List NormalizedRow → List (List NormalizedRow)
  | 0, _ => []
  | count + 1, rows =>
      rows.take applicationRowsPerSegment ::
        fixedSegmentRows count (rows.drop applicationRowsPerSegment)

@[simp]
theorem fixedSegmentRows_length
    (count : Nat) (rows : List NormalizedRow) :
    (fixedSegmentRows count rows).length = count := by
  induction count generalizing rows with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [fixedSegmentRows, inductionHypothesis]

theorem fixedSegmentRows_chunk_length_le
    {count : Nat} {rows chunk : List NormalizedRow}
    (member : chunk ∈ fixedSegmentRows count rows) :
    chunk.length ≤ applicationRowsPerSegment := by
  induction count generalizing rows with
  | zero => simp [fixedSegmentRows] at member
  | succ count inductionHypothesis =>
      simp only [fixedSegmentRows, List.mem_cons] at member
      rcases member with equal | member
      · subst chunk
        simp
      · exact inductionHypothesis member

theorem fixedSegmentRows_flatten
    {count : Nat} {rows : List NormalizedRow}
    (fits : rows.length ≤ count * applicationRowsPerSegment) :
    (fixedSegmentRows count rows).flatten = rows := by
  induction count generalizing rows with
  | zero =>
      have rowsEmpty : rows = [] := by
        apply List.eq_nil_of_length_eq_zero
        omega
      subst rows
      rfl
  | succ count inductionHypothesis =>
      have tailFits :
          (rows.drop applicationRowsPerSegment).length ≤
            count * applicationRowsPerSegment := by
        simp only [List.length_drop]
        rw [Nat.succ_mul] at fits
        omega
      simp only [fixedSegmentRows, List.flatten_cons]
      rw [inductionHypothesis tailFits]
      exact List.take_append_drop applicationRowsPerSegment rows

theorem rowList_access_count_le
    (rows : List NormalizedRow) :
    (rows.flatMap NormalizedRow.accesses).length ≤
      rows.length * slotsPerApplicationRow := by
  induction rows with
  | nil => simp
  | cons row rest inductionHypothesis =>
      simp only [List.flatMap_cons, List.length_append, List.length_cons]
      have rowBound := row.accessCount_le_capacity
      calc
        row.accesses.length +
            (List.flatMap NormalizedRow.accesses rest).length ≤
          slotsPerApplicationRow +
            rest.length * slotsPerApplicationRow :=
          Nat.add_le_add rowBound inductionHypothesis
        _ = rest.length * slotsPerApplicationRow +
            slotsPerApplicationRow := Nat.add_comm _ _
        _ = (rest.length + 1) * slotsPerApplicationRow := by
          rw [Nat.add_mul, Nat.one_mul]

theorem segment_access_capacity_eq :
    applicationRowsPerSegment * slotsPerApplicationRow = 63 * 1088 := by
  decide

/-- Canonical access list for each declared segment. The split comes from the
active application-row order and the fixed V2 segment capacity. -/
def segmentAccesses
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount) :
    List (List Access) :=
  (fixedSegmentRows segmentCount execution.real.rows).map
    (fun rows => rows.flatMap NormalizedRow.accesses)

theorem segmentAccesses_length
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount) :
    execution.segmentAccesses.length = segmentCount := by
  simp [segmentAccesses]

theorem segmentAccesses_each_length_le
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount)
    {accesses : List Access}
    (member : accesses ∈ execution.segmentAccesses) :
    accesses.length ≤
      applicationRowsPerSegment * slotsPerApplicationRow := by
  unfold segmentAccesses at member
  rcases List.mem_map.mp member with ⟨rows, rowsMember, rfl⟩
  exact (rowList_access_count_le rows).trans
    (Nat.mul_le_mul_right slotsPerApplicationRow
      (fixedSegmentRows_chunk_length_le rowsMember))

theorem realRows_fit_declared_segments
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount) :
    execution.real.rows.length ≤
      segmentCount * applicationRowsPerSegment := by
  have fits := execution.fitsDeclaredSegments
  rw [execution.realRowCountExact] at fits
  rw [execution.real.rows_length]
  unfold segmentCapacity at fits
  exact fits

private theorem flatten_map_row_accesses
    (chunks : List (List NormalizedRow)) :
    (chunks.map (fun rows =>
      rows.flatMap NormalizedRow.accesses)).flatten =
      chunks.flatten.flatMap NormalizedRow.accesses := by
  induction chunks with
  | nil => rfl
  | cons rows rest inductionHypothesis =>
      simp [inductionHypothesis]

theorem segmentAccesses_flatten
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount) :
    execution.segmentAccesses.flatten = execution.accesses := by
  unfold segmentAccesses accesses RealExecution.accesses
  rw [flatten_map_row_accesses]
  rw [fixedSegmentRows_flatten execution.realRows_fit_declared_segments]

/-- Exact bridge to the memory relation. This equality covers every active
application port, fixes each segment boundary, and forbids an alternate
semantic memory path. -/
def CoversMemory
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution semantics program initial result segmentCount)
    (claimedSegmentAccesses : List (List Access)) : Prop :=
  claimedSegmentAccesses = execution.segmentAccesses

theorem CoversMemory.segment_count
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    {claimedSegmentAccesses : List (List Access)}
    {execution :
      CompletedExecution semantics program initial result segmentCount}
    (covered : execution.CoversMemory claimedSegmentAccesses) :
    claimedSegmentAccesses.length = segmentCount := by
  rw [covered]
  exact execution.segmentAccesses_length

theorem CoversMemory.segment_access_count_le
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    {claimedSegmentAccesses : List (List Access)}
    {execution :
      CompletedExecution semantics program initial result segmentCount}
    (covered : execution.CoversMemory claimedSegmentAccesses)
    {accesses : List Access}
    (member : accesses ∈ claimedSegmentAccesses) :
    accesses.length ≤
      applicationRowsPerSegment * slotsPerApplicationRow := by
  rw [covered] at member
  exact execution.segmentAccesses_each_length_le member

theorem CoversMemory.flatten
    {Program ApplicationState Digest : Type}
    {semantics : Semantics Program ApplicationState}
    {program : Program} {initial : ApplicationState}
    {result : ExecutionResult ApplicationState Digest}
    {segmentCount : Nat}
    {claimedSegmentAccesses : List (List Access)}
    {execution :
      CompletedExecution semantics program initial result segmentCount}
    (covered : execution.CoversMemory claimedSegmentAccesses) :
    claimedSegmentAccesses.flatten = execution.accesses := by
  rw [covered]
  exact execution.segmentAccesses_flatten

end CompletedExecution

end Nightstream.Protocol.Nebula.ApplicationTrace
