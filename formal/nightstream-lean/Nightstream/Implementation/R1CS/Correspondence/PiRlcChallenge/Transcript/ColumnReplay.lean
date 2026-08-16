import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.CallRefinement
import Nightstream.Implementation.R1CS.Core.TranscriptCertificate

/-!
Column-level replay kernel for Poseidon2 transcript artifacts.

Owns: a small physical sponge state containing eight column identifiers, the
rate cursor, and positions in compact pin/call tables; executable consumption
of pinned constants, authoritative external columns, digest requests, and the
pending permutation at a Rust slice boundary; and the matching value-level
execution of the independent transcript machine.

Does not own: any protocol operation schedule, generated column or row,
initial-state authority, digest-input authority, row satisfaction, native Rust
conformance, sampler semantics, costs, or row removal.

Emits constraints: no.

Authority boundary: generated data may instantiate only the physical pin/call
tables. A handwritten operation list supplies protocol meaning. Later
soundness theorems use accepted pin equations and independently replayed
Poseidon2 calls to relate this column machine to `TranscriptMachine`.

| Layer | State/operation | Exact role |
|---|---|---|
| physical | `Cursor` | eight lane columns, rate cursor, next pin/call indices |
| physical | `absorbPinned` | consume one expected constant row, then overwrite with any required boundary call |
| physical | `absorbExternal` | overwrite one separately authoritative field column |
| physical | `digest` | pin squeeze word one, replay one call, expose four output columns |
| semantic | `SemanticRun` | execute the same operations over independent canonical field states |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplay

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.CallRefinement

abbrev CanonicalAssignment (assignment : Nat → Nat) :=
  ∀ column, assignment column < goldilocksP

/-- Physical transcript cursor. The lane entries are assignment-column
identifiers, not field values. -/
structure Cursor where
  lanes : Fin width → Nat
  absorbed : Fin (rate + 1)
  nextPin : Nat
  nextCall : Nat

/-- Replace one physical lane identifier. -/
def overwriteColumn
    (lanes : Fin width → Nat) (index column : Nat) : Fin width → Nat :=
  fun lane => if lane.val = index then column else lanes lane

/-- Exact input-column vector expected by one Poseidon2 call. -/
def laneColumns (cursor : Cursor) : List Nat :=
  List.ofFn cursor.lanes

/-- The eight physical output columns of one compact Poseidon2 call. -/
def callOutputColumns (call : Poseidon2Call.Call) : Fin width → Nat :=
  fun lane => call.columnMap (601 + lane.val)

/-- The four rate-lane columns exposed by one digest request. -/
def firstFourColumns (cursor : Cursor) : Fin 4 → Nat :=
  fun lane => cursor.lanes ⟨lane.val, by
    have laneLt := lane.isLt
    simp only [width]
    omega⟩

/-- Consume one exact `(column, value)` entry from the compact pin table. -/
def consumePin
    (trace : TranscriptCertificate.Trace)
    (cursor : Cursor) (expected : Nat) : Option (Nat × Cursor) :=
  if bounded : cursor.nextPin < trace.pins.length then
    let pin := trace.pins.get ⟨cursor.nextPin, bounded⟩
    if pin.2 = expected then
      some (pin.1, { cursor with nextPin := cursor.nextPin + 1 })
    else
      none
  else
    none

/-- Consume one exact Poseidon2 call whose eight inputs are the current lane
columns. The call output becomes the next cursor-zero state. -/
def permute
    (trace : TranscriptCertificate.Trace) (cursor : Cursor) : Option Cursor :=
  if bounded : cursor.nextCall < trace.calls.length then
    let call := trace.calls.get ⟨cursor.nextCall, bounded⟩
    if call.inputColumns = laneColumns cursor then
      some {
        lanes := callOutputColumns call
        absorbed := ⟨0, by decide⟩
        nextPin := cursor.nextPin
        nextCall := cursor.nextCall + 1
      }
    else
      none
  else
    none

/-- Absorb one already materialized column. A full cursor consumes the next
connected Poseidon2 call before overwriting lane zero. -/
def absorbColumn
    (trace : TranscriptCertificate.Trace)
    (cursor : Cursor) (column : Nat) : Option Cursor :=
  if room : cursor.absorbed.val < rate then
    some {
      cursor with
      lanes := overwriteColumn cursor.lanes cursor.absorbed.val column
      absorbed := ⟨cursor.absorbed.val + 1, by
        have cursorLt := cursor.absorbed.isLt
        simp only [rate] at room cursorLt ⊢
        omega⟩
    }
  else do
    let ready ← permute trace cursor
    pure {
      ready with
      lanes := overwriteColumn ready.lanes 0 column
      absorbed := ⟨1, by decide⟩
    }

/-- Consume a verifier-owned constant binding before its transcript absorb. -/
def absorbPinned
    (trace : TranscriptCertificate.Trace)
    (cursor : Cursor) (expected : Nat) : Option Cursor := do
  let (column, afterPin) ← consumePin trace cursor expected
  absorbColumn trace afterPin column

/-- Absorb one separately authoritative field column. -/
def absorbExternal
    (trace : TranscriptCertificate.Trace)
    (cursor : Cursor) (column : Nat) : Option Cursor :=
  absorbColumn trace cursor column

/-- Exact physical `digest32` transition: pin one, absorb it, consume the
forced permutation, and expose the first four output columns. -/
def digest
    (trace : TranscriptCertificate.Trace) (cursor : Cursor) :
    Option (Cursor × (Fin 4 → Nat)) := do
  let beforePermutation ← absorbPinned trace cursor 1
  let afterPermutation ← permute trace beforePermutation
  pure (afterPermutation, firstFourColumns afterPermutation)

/-- Handwritten protocol operations consumed by the physical replay kernel. -/
inductive Operation where
  | pinned (value : Nat)
  | external (column : Nat)
  | digest
deriving DecidableEq

/-- Physical execution state plus every four-lane digest emitted so far. -/
structure Run where
  cursor : Cursor
  digests : List (Fin 4 → Nat)

def step
    (trace : TranscriptCertificate.Trace) (run : Run) :
    Operation → Option Run
  | .pinned value => do
      let cursor ← absorbPinned trace run.cursor value
      pure { cursor := cursor, digests := run.digests }
  | .external column => do
      let cursor ← absorbExternal trace run.cursor column
      pure { cursor := cursor, digests := run.digests }
  | .digest => do
      let (cursor, columns) ← digest trace run.cursor
      pure { cursor := cursor, digests := run.digests ++ [columns] }

def execute
    (trace : TranscriptCertificate.Trace) : Run → List Operation → Option Run
  | run, [] => some run
  | run, operation :: rest => do
      let next ← step trace run operation
      execute trace next rest

/-- Splitting an operation list does not change its physical execution. -/
theorem execute_append_eq
    (trace : TranscriptCertificate.Trace) (start : Run)
    (left right : List Operation) :
    execute trace start (left ++ right) =
      (execute trace start left).bind fun middle =>
        execute trace middle right := by
  induction left generalizing start with
  | nil => rfl
  | cons operation rest inductionHypothesis =>
      simp only [List.cons_append, execute]
      cases step trace start operation with
      | none => simp
      | some next => simp [inductionHypothesis]

/-- Compose two physical replay executions without reevaluating either list. -/
theorem execute_append
    {trace : TranscriptCertificate.Trace}
    {start middle result : Run} {left right : List Operation}
    (leftExecution : execute trace start left = some middle)
    (rightExecution : execute trace middle right = some result) :
    execute trace start (left ++ right) = some result := by
  rw [execute_append_eq, leftExecution]
  exact rightExecution

/-- Normalize one completed Rust `absorb_slice` boundary. Scalar absorption
leaves a full rate cursor pending. Rust consumes that permutation before the
slice call returns. -/
def normalizeSlice
    (trace : TranscriptCertificate.Trace) (run : Run) : Option Run :=
  if _full : rate ≤ run.cursor.absorbed.val then do
    let cursor ← permute trace run.cursor
    pure { run with cursor := cursor }
  else
    some run

/-- Execute scalar absorb operations and then apply the exact Rust slice
boundary rule. -/
def executeSlice
    (trace : TranscriptCertificate.Trace) (run : Run)
    (operations : List Operation) : Option Run := do
  let next ← execute trace run operations
  normalizeSlice trace next

/-- Canonical value-level state represented by one physical column cursor. -/
def decodeCursor
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (cursor : Cursor) : State where
  lanes := fun lane => fieldAt assignment canonical (cursor.lanes lane)
  absorbed := cursor.absorbed

/-- Canonical value-level digest represented by four physical columns. -/
def decodeDigest
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (columns : Fin 4 → Nat) : Fin 4 → Field :=
  fun lane => fieldAt assignment canonical (columns lane)

/-- Independent value-level execution state at the same operation boundary. -/
structure SemanticRun where
  state : State
  digests : List (Fin 4 → Field)

private theorem semanticRunExt {left right : SemanticRun}
    (state : left.state = right.state)
    (digests : left.digests = right.digests) : left = right := by
  cases left
  cases right
  simp_all

def decodeRun
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : Run) : SemanticRun where
  state := decodeCursor assignment canonical run.cursor
  digests := run.digests.map (decodeDigest assignment canonical)

def semanticStep
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) : Operation → SemanticRun
  | .pinned value =>
      { run with state := absorbElem run.state (wordField value) }
  | .external column =>
      { run with
        state := absorbElem run.state (fieldAt assignment canonical column) }
  | .digest =>
      let result := TranscriptMachine.digest run.state
      { state := result.1, digests := run.digests ++ [result.2] }

def semanticExecute
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment) :
    SemanticRun → List Operation → SemanticRun
  | run, [] => run
  | run, operation :: rest =>
      semanticExecute assignment canonical
        (semanticStep assignment canonical run operation) rest

/-- Value-level form of the Rust slice boundary. -/
def semanticNormalizeSlice (run : SemanticRun) : SemanticRun :=
  if rate ≤ run.state.absorbed.val then
    { run with state := TranscriptMachine.permute run.state }
  else
    run

def semanticExecuteSlice
    (assignment : Nat → Nat) (canonical : CanonicalAssignment assignment)
    (run : SemanticRun) (operations : List Operation) : SemanticRun :=
  semanticNormalizeSlice
    (semanticExecute assignment canonical run operations)

/-! ## Value-level refinement -/

/-- A successfully consumed pin denotes the independently expected canonical
field value. -/
theorem consumePin_value
    {trace : TranscriptCertificate.Trace}
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (pinValuesCanonical : ConstantPins.ValuesCanonical trace.pins)
    (accepted : trace.Accepted assignment)
    {cursor next : Cursor} {expected column : Nat}
    (execution : consumePin trace cursor expected = some (column, next)) :
    fieldAt assignment canonical column = wordField expected := by
  unfold consumePin at execution
  split at execution
  case isFalse => simp at execution
  case isTrue bounded =>
    dsimp only at execution
    split at execution
    case isFalse => simp at execution
    case isTrue correct =>
      simp only [Option.some.injEq, Prod.mk.injEq] at execution
      rcases execution with ⟨columnEq, _nextEq⟩
      let pin := trace.pins.get ⟨cursor.nextPin, bounded⟩
      have pinMember : pin ∈ trace.pins :=
        List.get_mem trace.pins ⟨cursor.nextPin, bounded⟩
      have pinAccepted := accepted.1 pin pinMember
      have pinCanonical := pinValuesCanonical pin pinMember
      have expectedCanonical : expected < goldilocksP := by
        change (trace.pins.get ⟨cursor.nextPin, bounded⟩).2 <
          goldilocksP at pinCanonical
        rw [correct] at pinCanonical
        exact pinCanonical
      have expectedU64 : expected < u64Modulus := by
        have modulusLt : goldilocksP < u64Modulus := by decide
        omega
      have assignmentEq : assignment column = expected := by
        change assignment (trace.pins.get ⟨cursor.nextPin, bounded⟩).1 =
          (trace.pins.get ⟨cursor.nextPin, bounded⟩).2 at pinAccepted
        rw [columnEq, correct] at pinAccepted
        exact pinAccepted
      apply Fin.ext
      change assignment column = (expected % u64Modulus) % goldilocksP
      rw [Nat.mod_eq_of_lt expectedU64,
        Nat.mod_eq_of_lt expectedCanonical]
      exact assignmentEq

theorem consumePin_cursor
    {trace : TranscriptCertificate.Trace}
    {cursor next : Cursor} {expected column : Nat}
    (execution : consumePin trace cursor expected = some (column, next)) :
    next = { cursor with nextPin := cursor.nextPin + 1 } := by
  unfold consumePin at execution
  split at execution
  case isFalse => simp at execution
  case isTrue bounded =>
    dsimp only at execution
    split at execution
    case isFalse => simp at execution
    case isTrue correct =>
      simp only [Option.some.injEq, Prod.mk.injEq] at execution
      exact execution.2.symm

private theorem stateExt {left right : State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

/-- A connected compact call consumes exactly the represented physical state
and produces its independently replayed Poseidon2 successor. -/
theorem permute_sound
    {trace : TranscriptCertificate.Trace}
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : trace.Accepted assignment)
    {cursor next : Cursor}
    (execution : ColumnReplay.permute trace cursor = some next) :
    TranscriptMachine.permute (decodeCursor assignment canonical cursor) =
      decodeCursor assignment canonical next := by
  unfold ColumnReplay.permute at execution
  split at execution
  case isFalse => simp at execution
  case isTrue bounded =>
    dsimp only at execution
    split at execution
    case isFalse => simp at execution
    case isTrue connected =>
      simp only [Option.some.injEq] at execution
      let call := trace.calls.get ⟨cursor.nextCall, bounded⟩
      have callMember : call ∈ trace.calls :=
        List.get_mem trace.calls ⟨cursor.nextCall, bounded⟩
      have callAccepted := accepted.2 call callMember
      have inputStateEq :
          decodeCursor assignment canonical cursor =
            callInputState assignment canonical call cursor.absorbed := by
        apply stateExt
        · funext lane
          apply Fin.ext
          change assignment (cursor.lanes lane) =
            assignment (call.columnMap (lane.val + 1))
          apply congrArg assignment
          unfold Poseidon2Call.Call.columnMap
          have laneLt : lane.val < 8 := by
            exact lane.isLt
          have columnLt : lane.val + 1 < 9 := by omega
          have inputAt := congrArg
            (fun columns => columns.getD lane.val 0) connected
          simp [laneColumns] at inputAt
          simp [columnLt]
          exact inputAt.symm
        · rfl
      have outputStateEq :
          decodeCursor assignment canonical next =
            callOutputState assignment canonical call := by
        rw [← execution]
        rfl
      calc
        TranscriptMachine.permute
            (decodeCursor assignment canonical cursor) =
            TranscriptMachine.permute
              (callInputState assignment canonical call cursor.absorbed) :=
          congrArg TranscriptMachine.permute inputStateEq
        _ = callOutputState assignment canonical call :=
          callAccepted_permute canonical one call cursor.absorbed callAccepted
        _ = decodeCursor assignment canonical next := outputStateEq.symm

/-- Absorbing one materialized column refines the independent overwrite
sponge transition, including the full-cursor permutation case. -/
theorem absorbColumn_sound
    {trace : TranscriptCertificate.Trace}
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : trace.Accepted assignment)
    {cursor next : Cursor} {column : Nat}
    (execution : absorbColumn trace cursor column = some next) :
    absorbElem (decodeCursor assignment canonical cursor)
        (fieldAt assignment canonical column) =
      decodeCursor assignment canonical next := by
  unfold absorbColumn at execution
  split at execution
  case isTrue room =>
    simp only [Option.some.injEq] at execution
    rw [← execution]
    unfold absorbElem decodeCursor
    simp only [room, reduceDIte]
    apply stateExt
    · funext lane
      apply Fin.ext
      by_cases selected : lane.val = cursor.absorbed.val
      · simp [overwriteLane, overwriteColumn, selected, fieldAt]
      · simp [overwriteLane, overwriteColumn, selected, fieldAt]
    · rfl
  case isFalse full =>
    cases readyExecution : ColumnReplay.permute trace cursor with
    | none => simp [readyExecution] at execution
    | some ready =>
        simp [readyExecution] at execution
        have permutation := permute_sound canonical one accepted readyExecution
        unfold absorbElem
        have decodedFull :
            ¬ (decodeCursor assignment canonical cursor).absorbed.val < rate := by
          simpa [decodeCursor] using full
        simp only [decodedFull, reduceDIte]
        rw [permutation]
        rw [← execution]
        apply stateExt
        · funext lane
          apply Fin.ext
          by_cases selected : lane.val = 0
          · simp [decodeCursor, overwriteLane, overwriteColumn, selected,
              fieldAt]
          · simp [decodeCursor, overwriteLane, overwriteColumn, selected,
              fieldAt]
        · rfl

/-- Consuming one accepted constant pin and absorbing its column refines
absorption of the independently expected word value. Pin-table position is
physical bookkeeping only and cannot affect the semantic sponge state. -/
theorem absorbPinned_sound
    {trace : TranscriptCertificate.Trace}
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (pinValuesCanonical : ConstantPins.ValuesCanonical trace.pins)
    (one : assignment 0 = 1)
    (accepted : trace.Accepted assignment)
    {cursor next : Cursor} {expected : Nat}
    (execution : absorbPinned trace cursor expected = some next) :
    absorbElem (decodeCursor assignment canonical cursor)
        (wordField expected) =
      decodeCursor assignment canonical next := by
  unfold absorbPinned at execution
  cases pinExecution : consumePin trace cursor expected with
  | none => simp [pinExecution] at execution
  | some pinned =>
      rcases pinned with ⟨column, afterPin⟩
      simp only [pinExecution] at execution
      have value := consumePin_value canonical pinValuesCanonical accepted
        pinExecution
      have cursorAdvance := consumePin_cursor pinExecution
      have decoded :
          decodeCursor assignment canonical afterPin =
            decodeCursor assignment canonical cursor := by
        rw [cursorAdvance]
        rfl
      calc
        absorbElem (decodeCursor assignment canonical cursor)
            (wordField expected) =
            absorbElem (decodeCursor assignment canonical afterPin)
              (wordField expected) :=
          congrArg (fun state => absorbElem state (wordField expected))
            decoded.symm
        _ = absorbElem (decodeCursor assignment canonical afterPin)
              (fieldAt assignment canonical column) :=
          congrArg (absorbElem (decodeCursor assignment canonical afterPin))
            value.symm
        _ = decodeCursor assignment canonical next :=
          absorbColumn_sound canonical one accepted execution

/-- One accepted physical digest transition refines the independent
`digest32` state transition and exposes exactly its first four lanes. -/
theorem digest_sound
    {trace : TranscriptCertificate.Trace}
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (pinValuesCanonical : ConstantPins.ValuesCanonical trace.pins)
    (one : assignment 0 = 1)
    (accepted : trace.Accepted assignment)
    {cursor next : Cursor} {columns : Fin 4 → Nat}
    (execution : ColumnReplay.digest trace cursor = some (next, columns)) :
    TranscriptMachine.digest (decodeCursor assignment canonical cursor) =
      (decodeCursor assignment canonical next,
        decodeDigest assignment canonical columns) := by
  unfold ColumnReplay.digest at execution
  cases pinExecution : absorbPinned trace cursor 1 with
  | none => simp [pinExecution] at execution
  | some beforePermutation =>
      cases permutationExecution : ColumnReplay.permute trace beforePermutation with
      | none => simp [pinExecution, permutationExecution] at execution
      | some afterPermutation =>
          simp [pinExecution, permutationExecution] at execution
          rcases execution with ⟨nextEq, columnsEq⟩
          subst next
          subst columns
          have absorption := absorbPinned_sound canonical pinValuesCanonical one
            accepted pinExecution
          have permutation := permute_sound canonical one accepted
            permutationExecution
          unfold TranscriptMachine.digest
          rw [absorption, permutation]
          apply Prod.ext
          · rfl
          · funext lane
            rfl

/-- One physical operation refines the matching operation of the independent
semantic transcript machine. -/
theorem step_sound
    {trace : TranscriptCertificate.Trace}
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (pinValuesCanonical : ConstantPins.ValuesCanonical trace.pins)
    (one : assignment 0 = 1)
    (accepted : trace.Accepted assignment)
    {run next : Run} {operation : Operation}
    (execution : step trace run operation = some next) :
    semanticStep assignment canonical (decodeRun assignment canonical run)
        operation =
      decodeRun assignment canonical next := by
  cases operation with
  | pinned value =>
      unfold step at execution
      cases cursorExecution : absorbPinned trace run.cursor value with
      | none => simp [cursorExecution] at execution
      | some cursor =>
          simp [cursorExecution] at execution
          subst next
          have refinement := absorbPinned_sound canonical pinValuesCanonical
            one accepted cursorExecution
          apply semanticRunExt
          · exact refinement
          · rfl
  | external column =>
      unfold step at execution
      cases cursorExecution : absorbExternal trace run.cursor column with
      | none => simp [cursorExecution] at execution
      | some cursor =>
          simp [cursorExecution] at execution
          subst next
          have refinement := absorbColumn_sound canonical one accepted
            cursorExecution
          apply semanticRunExt
          · exact refinement
          · rfl
  | digest =>
      unfold step at execution
      cases digestExecution : ColumnReplay.digest trace run.cursor with
      | none => simp [digestExecution] at execution
      | some result =>
          rcases result with ⟨cursor, columns⟩
          simp [digestExecution] at execution
          subst next
          have refinement := digest_sound canonical pinValuesCanonical one
            accepted digestExecution
          have stateRefinement := congrArg Prod.fst refinement
          have digestRefinement := congrArg Prod.snd refinement
          apply semanticRunExt
          · exact stateRefinement
          · unfold semanticStep
            simp only [decodeRun, List.map_append, List.map_cons, List.map_nil]
            rw [digestRefinement]

/-- Any accepted physical operation list refines the independent semantic
execution of that same handwritten list. This theorem is profile-independent:
generated artifacts may instantiate the trace, but not the operation meaning. -/
theorem execute_sound
    {trace : TranscriptCertificate.Trace}
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (pinValuesCanonical : ConstantPins.ValuesCanonical trace.pins)
    (one : assignment 0 = 1)
    (accepted : trace.Accepted assignment)
    {start result : Run} {operations : List Operation}
    (execution : execute trace start operations = some result) :
    semanticExecute assignment canonical
        (decodeRun assignment canonical start) operations =
      decodeRun assignment canonical result := by
  induction operations generalizing start with
  | nil =>
      simp [execute] at execution
      subst result
      rfl
  | cons operation rest induction =>
      unfold execute at execution
      cases stepExecution : step trace start operation with
      | none => simp [stepExecution] at execution
      | some next =>
          simp only [stepExecution] at execution
          calc
            semanticExecute assignment canonical
                (decodeRun assignment canonical start) (operation :: rest) =
                semanticExecute assignment canonical
                  (semanticStep assignment canonical
                    (decodeRun assignment canonical start) operation) rest :=
              rfl
            _ = semanticExecute assignment canonical
                  (decodeRun assignment canonical next) rest :=
              congrArg
                (fun run => semanticExecute assignment canonical run rest)
                (step_sound canonical pinValuesCanonical one accepted
                  stepExecution)
            _ = decodeRun assignment canonical result :=
              induction execution

/-- The physical slice normalization consumes exactly the pending semantic
permutation, if one exists. -/
theorem normalizeSlice_sound
    {trace : TranscriptCertificate.Trace}
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : trace.Accepted assignment)
    {run next : Run}
    (execution : normalizeSlice trace run = some next) :
    semanticNormalizeSlice (decodeRun assignment canonical run) =
      decodeRun assignment canonical next := by
  unfold normalizeSlice at execution
  split at execution
  case isTrue full =>
    cases permutationExecution : ColumnReplay.permute trace run.cursor with
    | none => simp [permutationExecution] at execution
    | some cursor =>
        simp [permutationExecution] at execution
        subst next
        have refinement := permute_sound canonical one accepted
          permutationExecution
        unfold semanticNormalizeSlice
        have decodedFull :
            rate ≤ (decodeRun assignment canonical run).state.absorbed.val := by
          simpa [decodeRun, decodeCursor] using full
        simp only [decodedFull, ↓reduceIte]
        apply semanticRunExt
        · exact refinement
        · rfl
  case isFalse notFull =>
    simp only [Option.some.injEq] at execution
    subst next
    unfold semanticNormalizeSlice
    have decodedNotFull :
        ¬ rate ≤ (decodeRun assignment canonical run).state.absorbed.val := by
      simpa [decodeRun, decodeCursor] using notFull
    simp only [decodedNotFull, ↓reduceIte]

/-- A complete accepted physical slice refines the independent bulk semantic
execution, including Rust's eager final permutation for exact rate multiples. -/
theorem executeSlice_sound
    {trace : TranscriptCertificate.Trace}
    {assignment : Nat → Nat}
    (canonical : CanonicalAssignment assignment)
    (pinValuesCanonical : ConstantPins.ValuesCanonical trace.pins)
    (one : assignment 0 = 1)
    (accepted : trace.Accepted assignment)
    {start result : Run} {operations : List Operation}
    (execution : executeSlice trace start operations = some result) :
    semanticExecuteSlice assignment canonical
        (decodeRun assignment canonical start) operations =
      decodeRun assignment canonical result := by
  unfold executeSlice at execution
  cases replayExecution : execute trace start operations with
  | none => simp [replayExecution] at execution
  | some next =>
      simp only [replayExecution] at execution
      have replayRefinement := execute_sound canonical pinValuesCanonical one
        accepted replayExecution
      have normalizationRefinement := normalizeSlice_sound canonical one
        accepted execution
      unfold semanticExecuteSlice
      rw [replayRefinement]
      exact normalizationRefinement

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplay
