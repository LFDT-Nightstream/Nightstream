import Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
import Nightstream.Protocol.NebulaV2.WasmState

/-!
Contract: exact application execution owned by one production fresh claim.

This file gives the F-prime producer an independent application transition.
Each row is active, terminal, or canonical padding. Active and terminal rows
use the verifier-key-owned deterministic WASM machine. Padding preserves an
already terminal state and has no memory ports. A batch contains exactly
`3 * E` rows for its candidate profile.

The relation does not use an accepted proof, a state digest, or a caller-owned
state-transition proposition. It does not own a generated R1CS compiler,
physical columns, the memory relation, Rust, or terminal proof verification.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.ApplicationBatch

open Nightstream.Protocol.NebulaV2.ApplicationTrace
open Nightstream.Protocol.NebulaV2.Ports
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmState

/-! ## Canonical row view -/

/-- Every lifecycle row has one fixed normalized memory-port view. Synthetic
padding uses the canonical inactive normalized row. -/
def normalizedRow : ApplicationRow -> NormalizedRow
  | .active row => row
  | .returned row _ => row
  | .trapped row _ => row
  | .padding => NormalizedRow.inactive

def rowAccesses (row : ApplicationRow) : List Access :=
  (normalizedRow row).accesses

def normalizedRows (rows : List ApplicationRow) : List NormalizedRow :=
  rows.map normalizedRow

def accesses (rows : List ApplicationRow) : List Access :=
  rows.flatMap rowAccesses

@[simp] theorem accesses_append (left right : List ApplicationRow) :
    accesses (left ++ right) = accesses left ++ accesses right := by
  simp [accesses]

@[simp] theorem padding_has_no_accesses :
    rowAccesses .padding = [] := by
  exact NormalizedRow.inactive_has_no_accesses

@[simp] theorem terminal_row_accesses
    (row : NormalizedRow) (outcome : Completion.Outcome) :
    rowAccesses (ApplicationTrace.terminalApplicationRow row outcome) =
      row.accesses := by
  cases outcome <;> rfl

@[simp] theorem active_rows_accesses (rows : List NormalizedRow) :
    accesses (rows.map ApplicationRow.active) =
      rows.flatMap NormalizedRow.accesses := by
  induction rows with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, accesses, List.flatMap_cons]
      change head.accesses ++
          List.flatMap rowAccesses (List.map ApplicationRow.active tail) =
        head.accesses ++ List.flatMap NormalizedRow.accesses tail
      have tailExact :
          List.flatMap rowAccesses (List.map ApplicationRow.active tail) =
            List.flatMap NormalizedRow.accesses tail := by
        simpa only [accesses] using inductionHypothesis
      rw [tailExact]

@[simp] theorem padding_rows_accesses (count : Nat) :
    accesses (List.replicate count .padding) = [] := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [List.replicate_succ, accesses]

/-- Normalization preserves the exact ordered access list. -/
theorem normalizedRows_flatMap_accesses (rows : List ApplicationRow) :
    (normalizedRows rows).flatMap NormalizedRow.accesses = accesses rows := by
  induction rows with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [normalizedRows, accesses, List.map_cons, List.flatMap_cons]
      have tailExact :
          List.flatMap NormalizedRow.accesses (List.map normalizedRow tail) =
            List.flatMap rowAccesses tail := by
        simpa only [normalizedRows, accesses] using inductionHypothesis
      rw [tailExact]
      rfl

/-- A real row contributes one to the public real-row counter. Padding
contributes zero. -/
def rowCount : ApplicationRow -> Nat
  | .active _ => 1
  | .returned _ _ => 1
  | .trapped _ _ => 1
  | .padding => 0

def realRowCount : List ApplicationRow -> Nat
  | [] => 0
  | head :: tail => rowCount head + realRowCount tail

/-! ## Exact deterministic row execution -/

/-- One lifecycle row under the verifier-key-owned application machine. -/
inductive Transition
    {Program : Type}
    (machine : Machine Program) (program : Program) :
    AppStateVector -> ApplicationRow -> AppStateVector -> Nat -> Prop
  | active
      {before after : AppStateVector} {row : NormalizedRow}
      (accepted : machine.semantics.active program before row after) :
      Transition machine program before (.active row) after 1
  | returned
      {before after : AppStateVector} {row : NormalizedRow}
      {output : Option Completion.OutputValue}
      (accepted : machine.semantics.returned program before row output after) :
      Transition machine program before (.returned row output) after 1
  | trapped
      {before after : AppStateVector} {row : NormalizedRow}
      {trap : Completion.Trap}
      (accepted : machine.semantics.trapped program before row trap after) :
      Transition machine program before (.trapped row trap) after 1
  | padding
      {state : AppStateVector}
      (terminal : state.TerminalReady) :
      Transition machine program state .padding state 0

namespace Transition

theorem count_eq_rowCount
    {Program : Type} {machine : Machine Program} {program : Program}
    {before after : AppStateVector} {row : ApplicationRow} {count : Nat}
    (transition : Transition machine program before row after count) :
    count = rowCount row := by
  cases transition <;> rfl

theorem count_le_one
    {Program : Type} {machine : Machine Program} {program : Program}
    {before after : AppStateVector} {row : ApplicationRow} {count : Nat}
    (transition : Transition machine program before row after count) :
    count <= 1 := by
  cases transition <;> decide

/-- Exact row execution preserves the complete canonical application-state
invariant. Active and terminal rows derive the outgoing validity from the
verifier-key-owned WASM semantics. Padding preserves the same valid state. -/
theorem after_valid
    {Program : Type} {machine : Machine Program} {program : Program}
    {before after : AppStateVector} {row : ApplicationRow} {count : Nat}
    (transition : Transition machine program before row after count)
    (beforeValid : before.Valid) :
    after.Valid := by
  cases transition with
  | active accepted => exact accepted.2.1
  | returned accepted => exact accepted.2.2.2.2.valid
  | trapped accepted => exact accepted.2.2.2.2.valid
  | padding => exact beforeValid

/-- Once the machine is terminal, the only accepted next lifecycle row is
canonical state-preserving padding. -/
theorem from_terminal_is_padding
    {Program : Type} {machine : Machine Program} {program : Program}
    {before after : AppStateVector} {row : ApplicationRow} {count : Nat}
    (transition : Transition machine program before row after count)
    (terminal : before.TerminalReady) :
    row = .padding /\ after = before /\ count = 0 := by
  cases transition with
  | active accepted => exact False.elim (accepted.2.2.1 terminal)
  | returned accepted => exact False.elim (accepted.2.1 terminal)
  | trapped accepted => exact False.elim (accepted.2.1 terminal)
  | padding => exact ⟨rfl, rfl, rfl⟩

end Transition

/-- Ordered execution of a list of lifecycle rows. The natural-number index
is the exact number of nonpadding rows. -/
inductive Runs
    {Program : Type}
    (machine : Machine Program) (program : Program) :
    AppStateVector -> List ApplicationRow -> AppStateVector -> Nat -> Prop
  | nil (state : AppStateVector) :
      Runs machine program state [] state 0
  | cons
      {before middle after : AppStateVector}
      {row : ApplicationRow} {rows : List ApplicationRow}
      {headCount tailCount : Nat}
      (head : Transition machine program before row middle headCount)
      (tail : Runs machine program middle rows after tailCount) :
      Runs machine program before (row :: rows) after
        (headCount + tailCount)

namespace Runs

theorem append
    {Program : Type} {machine : Machine Program} {program : Program}
    {before middle after : AppStateVector}
    {left right : List ApplicationRow} {leftCount rightCount : Nat}
    (first : Runs machine program before left middle leftCount)
    (second : Runs machine program middle right after rightCount) :
    Runs machine program before (left ++ right) after
      (leftCount + rightCount) := by
  induction first with
  | nil => simpa using second
  | cons head _ inductionHypothesis =>
      simpa [Nat.add_assoc] using
        Runs.cons head (inductionHypothesis second)

/-- The indexed count is derived from the row constructors. -/
theorem count_eq_realRowCount
    {Program : Type} {machine : Machine Program} {program : Program}
    {before after : AppStateVector} {rows : List ApplicationRow} {count : Nat}
    (run : Runs machine program before rows after count) :
    count = realRowCount rows := by
  induction run with
  | nil => rfl
  | cons head _ inductionHypothesis =>
      rw [realRowCount, <- inductionHypothesis, head.count_eq_rowCount]

theorem count_le_length
    {Program : Type} {machine : Machine Program} {program : Program}
    {before after : AppStateVector} {rows : List ApplicationRow} {count : Nat}
    (run : Runs machine program before rows after count) :
    count <= rows.length := by
  induction run with
  | nil => simp
  | cons head _ inductionHypothesis =>
      simp only [List.length_cons]
      have headBound := head.count_le_one
      omega

/-- An ordered exact run preserves the complete canonical state invariant. -/
theorem after_valid
    {Program : Type} {machine : Machine Program} {program : Program}
    {before after : AppStateVector} {rows : List ApplicationRow} {count : Nat}
    (run : Runs machine program before rows after count)
    (beforeValid : before.Valid) :
    after.Valid := by
  induction run with
  | nil => exact beforeValid
  | cons head _ inductionHypothesis =>
      exact inductionHypothesis (head.after_valid beforeValid)

/-- Split one exact run at an arbitrary row index. Both returned runs retain
the original intermediate state and their exact real-row counts. -/
theorem splitAt
    {Program : Type} {machine : Machine Program} {program : Program}
    {before after : AppStateVector} {rows : List ApplicationRow} {count : Nat}
    (run : Runs machine program before rows after count)
    (prefixLength : Nat) :
    exists middle prefixCount suffixCount,
      Runs machine program before (rows.take prefixLength) middle prefixCount /\
        Runs machine program middle (rows.drop prefixLength) after suffixCount /\
        count = prefixCount + suffixCount := by
  induction prefixLength generalizing before rows after count with
  | zero =>
      exact ⟨before, 0, count, Runs.nil before, by simpa using run, by simp⟩
  | succ prefixLength inductionHypothesis =>
      cases run with
      | nil =>
          exact ⟨before, 0, 0, Runs.nil before, Runs.nil before, rfl⟩
      | @cons _ middle _ row tail headCount tailCount head tailRun =>
          rcases inductionHypothesis tailRun with
            ⟨splitState, prefixCount, suffixCount,
              prefixRun, suffixRun, countExact⟩
          refine ⟨splitState, headCount + prefixCount, suffixCount,
            ?_, suffixRun, ?_⟩
          · simpa using Runs.cons head prefixRun
          · omega

/-- Exact execution of an active semantic prefix. -/
theorem ofActivePrefix
    {Program : Type} {machine : Machine Program} {program : Program}
    {before after : AppStateVector} {rows : List NormalizedRow}
    (trace : ActivePrefix machine.semantics program before rows after) :
    Runs machine program before (rows.map ApplicationRow.active) after
      rows.length := by
  induction trace with
  | nil => exact Runs.nil _
  | cons step _ inductionHypothesis =>
      simpa [Nat.add_comm] using
        Runs.cons (Transition.active step) inductionHypothesis

/-- Exact execution of the one authority-bearing terminal row. -/
theorem ofTerminal
    {Program : Type} {machine : Machine Program} {program : Program}
    {before after : AppStateVector} {row : NormalizedRow}
    {outcome : Completion.Outcome}
    (terminal : ApplicationTrace.Terminal machine.semantics program before
      after row outcome) :
    Runs machine program before [terminalApplicationRow row outcome] after 1 := by
  cases terminal with
  | returned output accepted =>
      simpa [terminalApplicationRow] using
        Runs.cons (Transition.returned accepted) (Runs.nil after)
  | trapped trap accepted =>
      simpa [terminalApplicationRow] using
        Runs.cons (Transition.trapped accepted) (Runs.nil after)

/-- Any number of padding rows preserves an already terminal state. -/
theorem padding
    {Program : Type} {machine : Machine Program} {program : Program}
    {state : AppStateVector} (terminal : state.TerminalReady) (count : Nat) :
    Runs machine program state (List.replicate count .padding) state 0 := by
  induction count with
  | zero => exact Runs.nil _
  | succ count inductionHypothesis =>
      rw [List.replicate_succ]
      exact Runs.cons (Transition.padding terminal) inductionHypothesis

/-- The completed-execution definition constructs the exact row relation;
the real-row count is not a separate assumption. -/
theorem ofCompletedExecution
    {Program Digest : Type} {machine : Machine Program}
    {program : Program} {initial : AppStateVector}
    {result : Completion.ExecutionResult AppStateVector Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution machine.semantics program initial result
      segmentCount) :
    Runs machine program initial execution.rows
      result.finalApplicationState result.realApplicationRowCount := by
  have activeRun := ofActivePrefix execution.real.activeTrace
  have terminalRun := ofTerminal execution.real.terminal
  have realRun := activeRun.append terminalRun
  have terminalReady : result.finalApplicationState.TerminalReady :=
    (machine.terminal_derives_state execution.real.terminal).ready
  have paddingRun := Runs.padding (machine := machine) (program := program)
    terminalReady
    (Completion.segmentCapacity segmentCount -
      result.realApplicationRowCount)
  have completeRun := realRun.append paddingRun
  simpa [CompletedExecution.rows, execution.realRowCountExact,
    Nat.add_assoc] using completeRun

/-- The completed row list fills exactly the declared segment capacity. -/
theorem completed_rows_length
    {Program Digest : Type} {machine : Machine Program}
    {program : Program} {initial : AppStateVector}
    {result : Completion.ExecutionResult AppStateVector Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution machine.semantics program initial result
      segmentCount) :
    execution.rows.length = Completion.segmentCapacity segmentCount := by
  calc
    execution.rows.length =
        (execution.rows.map ApplicationRow.kind).length := by simp
    _ = (Completion.canonicalRows result segmentCount).length := by
      rw [execution.rowKindsCanonical]
    _ = Completion.segmentCapacity segmentCount :=
      Completion.canonicalRows_length result segmentCount
        execution.validCompletedTrace.realRowCountPositive
        execution.fitsDeclaredSegments

end Runs

/-! ## Candidate-specific fresh-claim batch -/

def rowsPerFreshClaim (candidate : Id) : Nat :=
  checkedStepsPerFreshClaim candidate * applicationRowsPerStep

theorem rowsPerFreshClaim_positive (candidate : Id) :
    0 < rowsPerFreshClaim candidate := by
  cases candidate <;> decide

theorem rowsPerFreshClaim_table :
    rowsPerFreshClaim .e1 = 3 /\
      rowsPerFreshClaim .e4 = 12 /\
      rowsPerFreshClaim .e8 = 24 /\
      rowsPerFreshClaim .e16 = 48 := by
  decide

/-- Candidate batching partitions one whole segment without remainder. -/
theorem claims_rows_partition_segment (candidate : Id) :
    claimsPerSegment candidate * rowsPerFreshClaim candidate =
      Completion.applicationRowsPerSegment := by
  cases candidate <;> decide

/-- One candidate claim executes exactly its fixed number of sequential
application rows. -/
structure Batch
    {Program : Type}
    (candidate : Id) (machine : Machine Program) (program : Program)
    (before after : AppStateVector) where
  rows : List ApplicationRow
  rowsExact : rows.length = rowsPerFreshClaim candidate
  run : Runs machine program before rows after (realRowCount rows)

namespace Batch

theorem realRowCount_le_rowsPerFreshClaim
    {Program : Type} {candidate : Id} {machine : Machine Program}
    {program : Program} {before after : AppStateVector}
    (batch : Batch candidate machine program before after) :
    realRowCount batch.rows <= rowsPerFreshClaim candidate := by
  rw [<- batch.rowsExact]
  exact batch.run.count_le_length

/-- One exact fresh-claim application batch preserves canonical state. -/
theorem after_valid
    {Program : Type} {candidate : Id} {machine : Machine Program}
    {program : Program} {before after : AppStateVector}
    (batch : Batch candidate machine program before after)
    (beforeValid : before.Valid) :
    after.Valid :=
  batch.run.after_valid beforeValid

end Batch

/-! ## Exact batch chain and completeness -/

/-- State-contiguous application batches. The list index is the exact number
of fresh claims represented by the chain. -/
inductive Chain
    {Program : Type}
    (candidate : Id) (machine : Machine Program) (program : Program) :
    AppStateVector -> List ApplicationRow -> AppStateVector -> Nat -> Prop
  | nil (state : AppStateVector) :
      Chain candidate machine program state [] state 0
  | cons
      {before middle after : AppStateVector}
      (head : Batch candidate machine program before middle)
      {tailRows : List ApplicationRow} {tailCount : Nat}
      (tail : Chain candidate machine program middle tailRows after tailCount) :
      Chain candidate machine program before (head.rows ++ tailRows) after
        (tailCount + 1)

namespace Chain

theorem rows_length
    {Program : Type} {candidate : Id} {machine : Machine Program}
    {program : Program} {before after : AppStateVector}
    {rows : List ApplicationRow} {batchCount : Nat}
    (chain : Chain candidate machine program before rows after batchCount) :
    rows.length = batchCount * rowsPerFreshClaim candidate := by
  induction chain with
  | nil => simp
  | cons head _ inductionHypothesis =>
      rw [List.length_append, head.rowsExact, inductionHypothesis]
      simp [Nat.add_mul, Nat.add_comm]

/-- Forgetting batch boundaries gives one exact application run. -/
theorem toRuns
    {Program : Type} {candidate : Id} {machine : Machine Program}
    {program : Program} {before after : AppStateVector}
    {rows : List ApplicationRow} {batchCount : Nat}
    (chain : Chain candidate machine program before rows after batchCount) :
    exists count, Runs machine program before rows after count := by
  induction chain with
  | nil => exact ⟨0, Runs.nil _⟩
  | cons head _ inductionHypothesis =>
      rcases inductionHypothesis with ⟨tailCount, tailRun⟩
      exact ⟨realRowCount head.rows + tailCount,
        head.run.append tailRun⟩

/-- A state-contiguous chain of exact batches preserves canonical state. -/
theorem after_valid
    {Program : Type} {candidate : Id} {machine : Machine Program}
    {program : Program} {before after : AppStateVector}
    {rows : List ApplicationRow} {batchCount : Nat}
    (chain : Chain candidate machine program before rows after batchCount)
    (beforeValid : before.Valid) :
    after.Valid := by
  induction chain with
  | nil => exact beforeValid
  | cons head _ inductionHypothesis =>
      exact inductionHypothesis (head.after_valid beforeValid)

/-- Any exact run whose length is a whole number of candidate batches has a
unique-in-order state-contiguous batch partition. -/
theorem ofRuns
    {Program : Type} {candidate : Id} {machine : Machine Program}
    {program : Program} {before after : AppStateVector}
    {rows : List ApplicationRow} {count batchCount : Nat}
    (run : Runs machine program before rows after count)
    (lengthExact :
      rows.length = batchCount * rowsPerFreshClaim candidate) :
    Chain candidate machine program before rows after batchCount := by
  induction batchCount generalizing before rows after count with
  | zero =>
      have rowsEmpty : rows = [] := by
        apply List.eq_nil_of_length_eq_zero
        simpa using lengthExact
      subst rows
      cases run
      exact Chain.nil _
  | succ batchCount inductionHypothesis =>
      let width := rowsPerFreshClaim candidate
      rcases run.splitAt width with
        ⟨middle, prefixCount, suffixCount,
          prefixRun, suffixRun, countExact⟩
      have widthPositive : 0 < width := rowsPerFreshClaim_positive candidate
      have prefixLength : (rows.take width).length = width := by
        simp only [List.length_take]
        apply Nat.min_eq_left
        rw [lengthExact]
        simp only [Nat.succ_mul]
        omega
      have suffixLength :
          (rows.drop width).length =
            batchCount * rowsPerFreshClaim candidate := by
        simp only [List.length_drop, width]
        rw [lengthExact, Nat.succ_mul]
        omega
      have prefixRunExact :
          Runs machine program before (rows.take width) middle
            (realRowCount (rows.take width)) := by
        rw [<- prefixRun.count_eq_realRowCount]
        exact prefixRun
      let head : Batch candidate machine program before middle :=
        { rows := rows.take width
          rowsExact := by simpa [width] using prefixLength
          run := prefixRunExact }
      have tail := inductionHypothesis suffixRun suffixLength
      have combined := Chain.cons head tail
      simpa [head, List.take_append_drop width rows] using combined

/-- Every completed execution partitions into the exact number of application
batches selected by its candidate profile. -/
theorem ofCompletedExecution
    {Program Digest : Type} {candidate : Id} {machine : Machine Program}
    {program : Program} {initial : AppStateVector}
    {result : Completion.ExecutionResult AppStateVector Digest}
    {segmentCount : Nat}
    (execution : CompletedExecution machine.semantics program initial result
      segmentCount) :
    Chain candidate machine program initial execution.rows
      result.finalApplicationState
      (segmentCount * claimsPerSegment candidate) := by
  apply ofRuns (Runs.ofCompletedExecution execution)
  rw [Runs.completed_rows_length execution]
  unfold Completion.segmentCapacity
  rw [<- claims_rows_partition_segment candidate]
  simp [Nat.mul_assoc]

end Chain

end Nightstream.Protocol.NebulaV2.ApplicationBatch
