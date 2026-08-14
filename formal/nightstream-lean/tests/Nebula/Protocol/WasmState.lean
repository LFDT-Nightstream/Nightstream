import Nightstream.Protocol.Nebula.WasmState

set_option autoImplicit false

namespace tests.NebulaWasmState

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ApplicationTrace
open Nightstream.Protocol.Nebula.Completion
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.Ports
open Nightstream.Protocol.Nebula.WasmState

def absent : OptionalU32 := { present := false, value := 0 }

def noOutput : OutputState :=
  { enabled := false, low := 0, high := 0 }

def idleAbsorb : EventAbsorbState where
  buffer := fun _ => 0
  bufferSlot := 0
  permutationPending := false
  permutationRound := 0
  permutationState := fun _ => 0

def idleGrammar : GrammarState where
  turnExportFunction := 0
  eventsRemaining := 0
  eventIndex := 0
  argumentsBase := 0
  slotCursor := 0

def running : AppStateVector where
  pc := 0
  operandStackPointer := 0
  stackFrameBase := 0
  output := noOutput
  callStackDepth := 0
  memoryPages := absent
  maximumMemoryPages := absent
  localsFrameBase := 0
  halted := false
  trapped := false
  trapCode := 0
  parameterInitialization := Countdown.zero
  tailCallPending := false
  hostArguments := Countdown.zero
  hostResultPending := false
  hostCalleeFunction := 0
  hostEventChain := fun _ => 0
  eventAbsorb := idleAbsorb
  grammarMode := false
  grammar := idleGrammar

def validRunning : running.Valid where
  pcBound := by decide
  operandStackPointerBound := by decide
  stackFrameBaseBound := by decide
  outputCanonical := by
    norm_num [OutputState.Canonical, running, noOutput, u32Limit]
  callStackDepthBound := by decide
  memoryPagesCanonical := by
    norm_num [OptionalU32.Canonical, running, absent, u32Limit]
  maximumMemoryPagesCanonical := by
    norm_num [OptionalU32.Canonical, running, absent, u32Limit]
  memoryPresenceMatches := rfl
  memoryPagesBound := by decide
  maximumMemoryPagesBound := by decide
  memoryPagesWithinMaximum := by decide
  localsFrameBaseBound := by decide
  trapCodeBound := by decide
  parameterInitializationCanonical := Countdown.zero_canonical
  hostArgumentsCanonical := Countdown.zero_canonical
  hostCalleeFunctionBound := by decide
  hostEventChainCanonical := by
    intro index
    norm_num [running, goldilocksModulus]
  eventBufferCanonical := by
    intro index
    norm_num [running, idleAbsorb, goldilocksModulus]
  eventBufferSlotBound := by decide
  permutationRoundBound := by decide
  permutationStateCanonical := by
    intro index
    norm_num [running, idleAbsorb, goldilocksModulus]
  grammarTurnFunctionBound := by decide
  grammarEventsRemainingBound := by decide
  grammarEventIndexBound := by decide
  grammarArgumentsBaseBound := by decide
  grammarSlotCursorBound := by decide

def completed : AppStateVector := running.setStatus true false 0

def completedValid : completed.Valid :=
  validRunning.setStatus true false (by decide)

def completedNoPending : completed.NoPendingWork where
  callStackDepth := rfl
  parameterInitialization := rfl
  tailCall := rfl
  hostArguments := rfl
  hostResult := rfl
  eventBufferSlot := rfl
  permutationPending := rfl
  permutationRound := rfl
  grammarMode := rfl
  grammarEvents := rfl
  grammarSlot := rfl

def completedReturned : completed.Terminal (.returned none) where
  valid := completedValid
  halted := rfl
  noPendingWork := completedNoPending
  outcomeMatches := by
    simp [AppStateVector.OutcomeMatches, OutputState.Matches,
      AppStateVector.setStatus, completed, running, noOutput]

def dummyAccess : Access where
  space := .ram
  address := 0
  kind := .read
  read := { timestamp := 0, globalIndex := romCells, value := 0 }
  write := { timestamp := 1, globalIndex := romCells, value := 0 }

def terminalRow : NormalizedRow where
  kind := .program
  memoryPorts := fun port =>
    if port.val = 0 then some dummyAccess else none

def machine : Machine Unit where
  step := fun _ _ _ => some completed

theorem terminalStepAccepted :
    machine.semantics.returned () running terminalRow none completed := by
  refine ⟨validRunning, ?_, ?_, rfl, completedReturned⟩
  · simp [AppStateVector.TerminalReady, running]
  · intro halted
    simp [running] at halted

def execution :
    RealExecution machine.semantics () running completed (.returned none) where
  activeRows := []
  beforeTerminal := running
  activeTrace := .nil running
  terminalRow := terminalRow
  terminal := .returned none terminalStepAccepted

/-- The terminal normalized row is part of the exact port trace. The old
model omitted this access because it flattened only nonterminal rows. -/
theorem terminal_memory_port_is_covered :
    execution.accesses = [dummyAccess] := by
  decide

theorem completed_state_cannot_take_an_active_program_row :
    ¬ machine.semantics.active () completed terminalRow completed := by
  intro accepted
  exact accepted.2.2.1 completedReturned.ready

def trapped : AppStateVector :=
  completed.setStatus true true Trap.memoryOutOfBounds.exitCode

def trappedTerminal : trapped.Terminal (.trapped .memoryOutOfBounds) where
  valid := by
    exact completedValid.setStatus true true (by decide)
  halted := rfl
  noPendingWork := by
    exact completedNoPending.setStatus true true
      Trap.memoryOutOfBounds.exitCode
  outcomeMatches := by
    simp [AppStateVector.OutcomeMatches, OutputState.Matches,
      AppStateVector.setStatus, trapped, completed, running, noOutput]

theorem trapped_code_is_derived :
    trapped.trapCode = Trap.memoryOutOfBounds.exitCode :=
  trappedTerminal.trapped_exit_code_exact

def completedWithLiveCall : AppStateVector :=
  { completed with callStackDepth := 1 }

/-- A halted state with a live call frame is not complete. -/
theorem live_call_frame_is_not_terminal :
    ¬ completedWithLiveCall.NoPendingWork := by
  intro invalid
  have := invalid.callStackDepth
  change 1 = 0 at this
  omega

def completedInGrammarMode : AppStateVector :=
  { completed with grammarMode := true }

/-- Zero grammar counters do not authorize a still-active grammar mode. -/
theorem active_grammar_mode_is_not_terminal :
    ¬ completedInGrammarMode.NoPendingWork := by
  intro invalid
  have := invalid.grammarMode
  simp [completedInGrammarMode] at this

end tests.NebulaWasmState
