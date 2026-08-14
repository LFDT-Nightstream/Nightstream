import Nightstream.Protocol.Nebula.Digest
import Nightstream.Protocol.Nebula.Soundness
import Nightstream.Protocol.Nebula.WasmStateEncoding

/-!
Contract: exact external completed-WASM result and statement for V2.

Assurance tier: model-level.

Owns the explicit result fields named by `SPEC.md`, their derivation from the
typed semantic result, and a V2 statement whose application state is the
complete `AppStateVector` rather than an opaque type.

Does not own the byte parser, generated public-input rows, program interpreter,
memory-root hash security, recursive verifier, or terminal backend.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.WasmStatement

open Nightstream.Protocol.Nebula.ApplicationTrace
open Nightstream.Protocol.Nebula.Completion
open Nightstream.Protocol.Nebula.Soundness
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStateEncoding

inductive TerminationMode where
  | returned
  | trapped
deriving DecidableEq, Repr

def terminationMode : Outcome → TerminationMode
  | .returned _ => .returned
  | .trapped _ => .trapped

def outcomeOutputPresent : Outcome → Bool
  | .returned none => false
  | .returned (some _) => true
  | .trapped _ => false

def outcomeOutputLow : Outcome → Nat
  | .returned none => 0
  | .returned (some output) => output.low
  | .trapped _ => 0

def outcomeOutputHigh : Outcome → Nat
  | .returned none => 0
  | .returned (some output) => output.high
  | .trapped _ => 0

theorem terminal_output_fields_exact
    {state : AppStateVector} {outcome : Outcome}
    (terminal : state.Terminal outcome) :
    outcomeOutputPresent outcome = state.output.enabled ∧
      outcomeOutputLow outcome = state.output.low ∧
      outcomeOutputHigh outcome = state.output.high := by
  cases outcome with
  | returned output =>
      cases output with
      | none =>
          have outputMatch := terminal.outcomeMatches.2.2
          exact
            ⟨outputMatch.1.symm,
              outputMatch.2.1.symm,
              outputMatch.2.2.symm⟩
      | some output =>
          have outputMatch := terminal.outcomeMatches.2.2
          exact
            ⟨outputMatch.1.symm,
              outputMatch.2.1.symm,
              outputMatch.2.2.symm⟩
  | trapped trap =>
      have outputMatch := terminal.outcomeMatches.2.2
      exact
        ⟨outputMatch.1.symm,
          outputMatch.2.1.symm,
          outputMatch.2.2.symm⟩

theorem terminal_status_fields_exact
    {state : AppStateVector} {outcome : Outcome}
    (terminal : state.Terminal outcome) :
    outcome.exitCode = state.trapCode ∧
      state.halted = true ∧
      ((terminationMode outcome = .returned ∧
          state.trapped = false ∧ outcome.exitCode = 0) ∨
        (terminationMode outcome = .trapped ∧
          state.trapped = true ∧
          1 ≤ outcome.exitCode ∧ outcome.exitCode ≤ 7)) := by
  cases outcome with
  | returned output =>
      exact
        ⟨terminal.returned_exit_code_zero.symm,
          terminal.halted,
          Or.inl ⟨rfl, terminal.outcomeMatches.1, rfl⟩⟩
  | trapped trap =>
      exact
        ⟨terminal.trapped_exit_code_exact.symm,
          terminal.halted,
          Or.inr
            ⟨rfl, terminal.outcomeMatches.1,
              Trap.exitCode_in_range trap⟩⟩

/-- Exact public result before byte encoding. The final state is the complete
55-field typed image, not its digest. -/
@[ext] structure ResultImage (Digest : Type) where
  realApplicationRowCount : Nat
  finalApplicationState : Image
  terminationMode : TerminationMode
  exitCode : Nat
  outputPresent : Bool
  outputValueLow : Nat
  outputValueHigh : Nat
  finalMemoryRoot : Digest

def ResultImage.ofResult
    {Digest : Type} (result : ExecutionResult AppStateVector Digest) :
    ResultImage Digest where
  realApplicationRowCount := result.realApplicationRowCount
  finalApplicationState := WasmStateEncoding.encode result.finalApplicationState
  terminationMode := WasmStatement.terminationMode result.outcome
  exitCode := result.outcome.exitCode
  outputPresent := outcomeOutputPresent result.outcome
  outputValueLow := outcomeOutputLow result.outcome
  outputValueHigh := outcomeOutputHigh result.outcome
  finalMemoryRoot := result.finalMemoryRoot

/-- Parser/terminal relation for one result image. The equality checks every
redundant public field; `terminal` derives state/output/trap consistency. -/
structure ResultImage.Decodes
    {Digest : Type}
    (image : ResultImage Digest)
    (result : ExecutionResult AppStateVector Digest) : Prop where
  exactImage : image = ResultImage.ofResult result
  terminal : result.finalApplicationState.Terminal result.outcome
  realRowCountPositive : 0 < result.realApplicationRowCount
  realRowCountBound : result.realApplicationRowCount < realApplicationRowLimit

namespace ResultImage.Decodes

theorem final_state_canonical
    {Digest : Type} {image : ResultImage Digest}
    {result : ExecutionResult AppStateVector Digest}
    (decoded : image.Decodes result) :
    image.finalApplicationState.Canonical := by
  rw [decoded.exactImage]
  simpa [ResultImage.ofResult] using
    (canonical_encode_iff result.finalApplicationState).2
      decoded.terminal.valid

theorem output_fields_equal_state
    {Digest : Type} {image : ResultImage Digest}
    {result : ExecutionResult AppStateVector Digest}
    (decoded : image.Decodes result) :
    image.outputPresent =
        result.finalApplicationState.output.enabled ∧
      image.outputValueLow = result.finalApplicationState.output.low ∧
      image.outputValueHigh = result.finalApplicationState.output.high := by
  rw [decoded.exactImage]
  simpa [ResultImage.ofResult] using
    terminal_output_fields_exact decoded.terminal

theorem mode_exit_and_flags_exact
    {Digest : Type} {image : ResultImage Digest}
    {result : ExecutionResult AppStateVector Digest}
    (decoded : image.Decodes result) :
    image.exitCode = result.finalApplicationState.trapCode ∧
      result.finalApplicationState.halted = true ∧
      ((image.terminationMode = .returned ∧
          result.finalApplicationState.trapped = false ∧
          image.exitCode = 0) ∨
        (image.terminationMode = .trapped ∧
          result.finalApplicationState.trapped = true ∧
          1 ≤ image.exitCode ∧ image.exitCode ≤ 7)) := by
  rw [decoded.exactImage]
  simpa [ResultImage.ofResult] using
    terminal_status_fields_exact decoded.terminal

end ResultImage.Decodes

/-- Exact external statement specialized to the verifier-key-owned WASM
machine state. `base.expectedResult` is fully exposed by `resultImage`. -/
structure Statement (Program Digest : Type) where
  base : PublicStatement Program AppStateVector Digest
  initialApplicationStateValid : base.initialApplicationState.Valid
  resultImage : ResultImage Digest
  resultDecoded : resultImage.Decodes base.expectedResult

abbrev ProductionResultImage := ResultImage Digest.Value

abbrev ProductionStatement (Program : Type) := Statement Program Digest.Value

theorem Statement.identity_is_explicit
    {Program Digest : Type} (statement : Statement Program Digest) :
    StatementIdentity.encode statement.base.identity =
      StatementIdentity.encode statement.base.identity :=
  rfl

/-- A completed execution under the fixed machine produces the terminal part
of the exact result-decoding relation. -/
theorem completed_execution_derives_terminal
    {Program Digest : Type} {machine : Machine Program}
    {statement : Statement Program Digest}
    (execution :
      CompletedExecution machine.semantics statement.base.program
        statement.base.initialApplicationState statement.base.expectedResult
        statement.base.segmentCount) :
    statement.base.expectedResult.finalApplicationState.Terminal
      statement.base.expectedResult.outcome :=
  machine.completedExecution_final_terminal execution

end Nightstream.Protocol.Nebula.WasmStatement
