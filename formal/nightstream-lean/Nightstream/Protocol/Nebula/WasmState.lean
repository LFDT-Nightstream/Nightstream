import Nightstream.Protocol.Nebula.ApplicationTrace
import Nightstream.Protocol.Nebula.Fingerprint

/-!
Contract: exact public WASM state and completed-row semantics for V2.

Assurance tier: model-level.

Owns the complete ordered application-state data named by `SPEC.md`, integer
bounds, terminal returned/trapped rules, pending-work exhaustion, deterministic
row execution, and the rule that a halted execution can only drain event rows.

Does not own the selected WASM instruction interpreter, generated row table,
Rust-state decoder, or public-input byte codec. The verifier-key-bound machine
must supply the deterministic `step` function and later refinement layers must
prove that Rust and the generated relation implement that same function.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.WasmState

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ApplicationTrace
open Nightstream.Protocol.Nebula.Completion
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.Ports

def u8Limit : Nat := 2 ^ 8
def u32Limit : Nat := 2 ^ 32
def u64Limit : Nat := 2 ^ 64
def wasm32MaximumPages : Nat := 65536
def eventBufferSlots : Nat := 4
def eventPermutationRows : Nat := 19
def grammarSlots : Nat := 8

/-- Canonical option encoding. An absent value has a zero payload. -/
structure OptionalU32 where
  present : Bool
  value : Nat
deriving DecidableEq, Repr

def OptionalU32.Canonical (value : OptionalU32) : Prop :=
  value.value < u32Limit ∧
    (value.present = false → value.value = 0)

/-- Countdown latches are canonical: inactivity is exactly a zero count. -/
structure Countdown where
  active : Bool
  remaining : Nat
deriving DecidableEq, Repr

def Countdown.Canonical (countdown : Countdown) : Prop :=
  countdown.remaining < u32Limit ∧
    (countdown.active = false ↔ countdown.remaining = 0)

def Countdown.zero : Countdown :=
  { active := false, remaining := 0 }

theorem Countdown.zero_canonical : Countdown.zero.Canonical := by
  constructor
  · decide
  · simp [Countdown.zero]

structure OutputState where
  enabled : Bool
  low : Nat
  high : Nat
deriving DecidableEq, Repr

def OutputState.Canonical (output : OutputState) : Prop :=
  output.low < u32Limit ∧ output.high < u32Limit ∧
    (output.enabled = false → output.low = 0 ∧ output.high = 0)

def OutputState.Matches
    (state : OutputState) : Option OutputValue → Prop
  | none => state.enabled = false ∧ state.low = 0 ∧ state.high = 0
  | some output =>
      state.enabled = true ∧ state.low = output.low ∧
        state.high = output.high

/-- One canonical output state identifies at most one typed output value. -/
theorem OutputState.matches_unique
    {state : OutputState} {left right : Option OutputValue}
    (leftMatches : state.Matches left)
    (rightMatches : state.Matches right) :
    left = right := by
  cases left with
  | none =>
      cases right with
      | none => rfl
      | some output =>
          exact False.elim (Bool.noConfusion
            (leftMatches.1.symm.trans rightMatches.1))
  | some leftOutput =>
      cases right with
      | none =>
          exact False.elim (Bool.noConfusion
            (leftMatches.1.symm.trans rightMatches.1))
      | some rightOutput =>
          apply congrArg some
          cases leftOutput
          cases rightOutput
          simp_all [OutputState.Matches]

structure EventAbsorbState where
  buffer : Fin 8 → Nat
  bufferSlot : Nat
  permutationPending : Bool
  permutationRound : Nat
  permutationState : Fin 12 → Nat

structure GrammarState where
  turnExportFunction : Nat
  eventsRemaining : Nat
  eventIndex : Nat
  argumentsBase : Nat
  slotCursor : Nat
deriving DecidableEq

/-- Complete V2 public application-state vector. `trapCode` is added to the
current Rust state because V2 requires the exact trap reason to survive until
terminal verification. -/
structure AppStateVector where
  pc : Nat
  operandStackPointer : Nat
  stackFrameBase : Nat
  output : OutputState
  callStackDepth : Nat
  memoryPages : OptionalU32
  maximumMemoryPages : OptionalU32
  localsFrameBase : Nat
  halted : Bool
  trapped : Bool
  trapCode : Nat
  parameterInitialization : Countdown
  tailCallPending : Bool
  hostArguments : Countdown
  hostResultPending : Bool
  hostCalleeFunction : Nat
  hostEventChain : Fin 4 → Nat
  eventAbsorb : EventAbsorbState
  grammarMode : Bool
  grammar : GrammarState

/-- Integer and canonical-representation checks for every state field. -/
structure AppStateVector.Valid (state : AppStateVector) : Prop where
  pcBound : state.pc < u64Limit
  operandStackPointerBound : state.operandStackPointer < u64Limit
  stackFrameBaseBound : state.stackFrameBase < u64Limit
  outputCanonical : state.output.Canonical
  callStackDepthBound : state.callStackDepth < u64Limit
  memoryPagesCanonical : state.memoryPages.Canonical
  maximumMemoryPagesCanonical : state.maximumMemoryPages.Canonical
  memoryPresenceMatches :
    state.memoryPages.present = state.maximumMemoryPages.present
  memoryPagesBound : state.memoryPages.value ≤ wasm32MaximumPages
  maximumMemoryPagesBound :
    state.maximumMemoryPages.value ≤ wasm32MaximumPages
  memoryPagesWithinMaximum :
    state.memoryPages.value ≤ state.maximumMemoryPages.value
  localsFrameBaseBound : state.localsFrameBase < u64Limit
  trapCodeBound : state.trapCode < u32Limit
  parameterInitializationCanonical :
    state.parameterInitialization.Canonical
  hostArgumentsCanonical : state.hostArguments.Canonical
  hostCalleeFunctionBound : state.hostCalleeFunction < u32Limit
  hostEventChainCanonical :
    ∀ index, state.hostEventChain index < goldilocksModulus
  eventBufferCanonical :
    ∀ index, state.eventAbsorb.buffer index < goldilocksModulus
  eventBufferSlotBound :
    state.eventAbsorb.bufferSlot < eventBufferSlots
  permutationRoundBound :
    state.eventAbsorb.permutationRound < eventPermutationRows
  permutationStateCanonical :
    ∀ index, state.eventAbsorb.permutationState index < goldilocksModulus
  grammarTurnFunctionBound : state.grammar.turnExportFunction < u32Limit
  grammarEventsRemainingBound : state.grammar.eventsRemaining < u32Limit
  grammarEventIndexBound : state.grammar.eventIndex < u32Limit
  grammarArgumentsBaseBound : state.grammar.argumentsBase < u64Limit
  grammarSlotCursorBound : state.grammar.slotCursor < grammarSlots

/-- A completed state has no call, result, event, permutation, or grammar work
left. Stale permutation-state and base-address values remain authenticated in
the full state vector; they are not used as pending-work flags. -/
structure AppStateVector.NoPendingWork (state : AppStateVector) : Prop where
  callStackDepth : state.callStackDepth = 0
  parameterInitialization : state.parameterInitialization = Countdown.zero
  tailCall : state.tailCallPending = false
  hostArguments : state.hostArguments = Countdown.zero
  hostResult : state.hostResultPending = false
  eventBufferSlot : state.eventAbsorb.bufferSlot = 0
  permutationPending : state.eventAbsorb.permutationPending = false
  permutationRound : state.eventAbsorb.permutationRound = 0
  grammarMode : state.grammarMode = false
  grammarEvents : state.grammar.eventsRemaining = 0
  grammarSlot : state.grammar.slotCursor = 0

/-- Change only the three terminal-status fields. This operation is also the
exact state change allowed when a terminal result is attached to an otherwise
fully checked machine state. -/
def AppStateVector.setStatus
    (state : AppStateVector) (halted trapped : Bool) (trapCode : Nat) :
    AppStateVector :=
  { state with
    halted := halted
    trapped := trapped
    trapCode := trapCode }

theorem AppStateVector.Valid.setStatus
    {state : AppStateVector} (valid : state.Valid)
    (halted trapped : Bool) {trapCode : Nat}
    (trapCodeBound : trapCode < u32Limit) :
    (state.setStatus halted trapped trapCode).Valid where
  pcBound := valid.pcBound
  operandStackPointerBound := valid.operandStackPointerBound
  stackFrameBaseBound := valid.stackFrameBaseBound
  outputCanonical := valid.outputCanonical
  callStackDepthBound := valid.callStackDepthBound
  memoryPagesCanonical := valid.memoryPagesCanonical
  maximumMemoryPagesCanonical := valid.maximumMemoryPagesCanonical
  memoryPresenceMatches := valid.memoryPresenceMatches
  memoryPagesBound := valid.memoryPagesBound
  maximumMemoryPagesBound := valid.maximumMemoryPagesBound
  memoryPagesWithinMaximum := valid.memoryPagesWithinMaximum
  localsFrameBaseBound := valid.localsFrameBaseBound
  trapCodeBound := trapCodeBound
  parameterInitializationCanonical := valid.parameterInitializationCanonical
  hostArgumentsCanonical := valid.hostArgumentsCanonical
  hostCalleeFunctionBound := valid.hostCalleeFunctionBound
  hostEventChainCanonical := valid.hostEventChainCanonical
  eventBufferCanonical := valid.eventBufferCanonical
  eventBufferSlotBound := valid.eventBufferSlotBound
  permutationRoundBound := valid.permutationRoundBound
  permutationStateCanonical := valid.permutationStateCanonical
  grammarTurnFunctionBound := valid.grammarTurnFunctionBound
  grammarEventsRemainingBound := valid.grammarEventsRemainingBound
  grammarEventIndexBound := valid.grammarEventIndexBound
  grammarArgumentsBaseBound := valid.grammarArgumentsBaseBound
  grammarSlotCursorBound := valid.grammarSlotCursorBound

theorem AppStateVector.NoPendingWork.setStatus
    {state : AppStateVector} (noPending : state.NoPendingWork)
    (halted trapped : Bool) (trapCode : Nat) :
    (state.setStatus halted trapped trapCode).NoPendingWork where
  callStackDepth := noPending.callStackDepth
  parameterInitialization := noPending.parameterInitialization
  tailCall := noPending.tailCall
  hostArguments := noPending.hostArguments
  hostResult := noPending.hostResult
  eventBufferSlot := noPending.eventBufferSlot
  permutationPending := noPending.permutationPending
  permutationRound := noPending.permutationRound
  grammarMode := noPending.grammarMode
  grammarEvents := noPending.grammarEvents
  grammarSlot := noPending.grammarSlot

def AppStateVector.TerminalReady (state : AppStateVector) : Prop :=
  state.halted = true ∧ state.NoPendingWork

def AppStateVector.OutcomeMatches
    (state : AppStateVector) : Outcome → Prop
  | .returned output =>
      state.trapped = false ∧ state.trapCode = 0 ∧
        state.output.Matches output
  | .trapped trap =>
      state.trapped = true ∧ state.trapCode = trap.exitCode ∧
        state.output.Matches none

/-- Exact terminal public-state rule. No execution result is assumed here;
the typed outcome must agree with the complete state. -/
structure AppStateVector.Terminal
    (state : AppStateVector) (outcome : Outcome) : Prop where
  valid : state.Valid
  halted : state.halted = true
  noPendingWork : state.NoPendingWork
  outcomeMatches : state.OutcomeMatches outcome

theorem AppStateVector.Terminal.ready
    {state : AppStateVector} {outcome : Outcome}
    (terminal : state.Terminal outcome) : state.TerminalReady :=
  ⟨terminal.halted, terminal.noPendingWork⟩

theorem AppStateVector.Terminal.returned_exit_code_zero
    {state : AppStateVector} {output : Option OutputValue}
    (terminal : state.Terminal (.returned output)) :
    state.trapCode = 0 :=
  terminal.outcomeMatches.2.1

theorem AppStateVector.Terminal.trapped_exit_code_exact
    {state : AppStateVector} {trap : Trap}
    (terminal : state.Terminal (.trapped trap)) :
    state.trapCode = trap.exitCode :=
  terminal.outcomeMatches.2.1

/-- A complete canonical application state identifies one typed outcome. -/
theorem AppStateVector.Terminal.outcome_unique
    {state : AppStateVector} {left right : Outcome}
    (leftTerminal : state.Terminal left)
    (rightTerminal : state.Terminal right) :
    left = right := by
  cases left with
  | returned leftOutput =>
      cases right with
      | returned rightOutput =>
          exact congrArg Outcome.returned
            (OutputState.matches_unique leftTerminal.outcomeMatches.2.2
              rightTerminal.outcomeMatches.2.2)
      | trapped rightTrap =>
          have leftFlag := leftTerminal.outcomeMatches.1
          have rightFlag := rightTerminal.outcomeMatches.1
          simp_all
  | trapped leftTrap =>
      cases right with
      | returned rightOutput =>
          have leftFlag := leftTerminal.outcomeMatches.1
          have rightFlag := rightTerminal.outcomeMatches.1
          simp_all
      | trapped rightTrap =>
          apply congrArg Outcome.trapped
          apply Trap.exitCode_injective
          calc
            leftTrap.exitCode = state.trapCode :=
              leftTerminal.outcomeMatches.2.1.symm
            _ = rightTrap.exitCode :=
              rightTerminal.outcomeMatches.2.1

/-- Verifier-key-owned deterministic application machine. A relation or Rust
implementation must refine this exact function; it cannot replace it with an
arbitrary proposition such as `True`. -/
structure Machine (Program : Type) where
  step : Program → AppStateVector → NormalizedRow → Option AppStateVector

def rowAllowedAfterState
    (state : AppStateVector) (row : NormalizedRow) : Prop :=
  state.halted = true → row.kind.canDrainAfterHalt

/-- The generic completed-trace layer instantiated with the deterministic V2
WASM machine. Active rows cannot enter a completed state. The one terminal
row must enter a state whose exact outcome and pending-work fields are valid. -/
def Machine.semantics
    {Program : Type} (machine : Machine Program) :
    ApplicationTrace.Semantics Program AppStateVector where
  active := fun program before row after =>
    before.Valid ∧ after.Valid ∧
      ¬ before.TerminalReady ∧ ¬ after.TerminalReady ∧
      rowAllowedAfterState before row ∧
      machine.step program before row = some after
  returned := fun program before row output after =>
    before.Valid ∧ ¬ before.TerminalReady ∧
      rowAllowedAfterState before row ∧
      machine.step program before row = some after ∧
      after.Terminal (.returned output)
  trapped := fun program before row trap after =>
    before.Valid ∧ ¬ before.TerminalReady ∧
      rowAllowedAfterState before row ∧
      machine.step program before row = some after ∧
      after.Terminal (.trapped trap)

theorem Machine.active_does_not_complete
    {Program : Type} {machine : Machine Program}
    {program : Program} {before after : AppStateVector}
    {row : NormalizedRow}
    (accepted : machine.semantics.active program before row after) :
    ¬ after.TerminalReady :=
  accepted.2.2.2.1

theorem Machine.halted_row_is_event_drain
    {Program : Type} {machine : Machine Program}
    {program : Program} {before after : AppStateVector}
    {row : NormalizedRow}
    (accepted : machine.semantics.active program before row after)
    (halted : before.halted = true) :
    row.kind.canDrainAfterHalt :=
  accepted.2.2.2.2.1 halted

theorem Machine.terminal_row_completes
    {Program : Type} {machine : Machine Program}
    {program : Program} {before after : AppStateVector}
    {row : NormalizedRow} {output : Option OutputValue}
    (accepted : machine.semantics.returned program before row output after) :
    after.Terminal (.returned output) :=
  accepted.2.2.2.2

theorem Machine.terminal_derives_state
    {Program : Type} {machine : Machine Program}
    {program : Program} {before after : AppStateVector}
    {row : NormalizedRow} {outcome : Outcome}
    (terminal :
      ApplicationTrace.Terminal machine.semantics program before after row
        outcome) :
    after.Terminal outcome := by
  cases terminal with
  | returned output step => exact step.2.2.2.2
  | trapped trap step => exact step.2.2.2.2

/-- A completed trace under the fixed machine semantics derives the exact
terminal state and outcome from its authority-bearing terminal row. -/
theorem Machine.completedExecution_final_terminal
    {Program Digest : Type} {machine : Machine Program}
    {program : Program} {initial : AppStateVector}
    {result : ExecutionResult AppStateVector Digest}
    {segmentCount : Nat}
    (execution :
      ApplicationTrace.CompletedExecution machine.semantics program initial
        result segmentCount) :
    result.finalApplicationState.Terminal result.outcome :=
  machine.terminal_derives_state execution.real.terminal

end Nightstream.Protocol.Nebula.WasmState
