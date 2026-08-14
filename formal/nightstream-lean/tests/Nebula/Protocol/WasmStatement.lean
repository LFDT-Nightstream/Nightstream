import Nightstream.Protocol.Nebula.WasmStatement
import tests.Nebula.Protocol.WasmState

set_option autoImplicit false

namespace tests.NebulaWasmStatement

open Nightstream.Protocol.Nebula.ApplicationTrace
open Nightstream.Protocol.Nebula.Completion
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStatement

def result : ExecutionResult AppStateVector Nat where
  realApplicationRowCount := 1
  finalApplicationState := tests.NebulaWasmState.completed
  outcome := .returned none
  finalMemoryRoot := 9

def resultImage : ResultImage Nat := ResultImage.ofResult result

def resultDecodes : resultImage.Decodes result where
  exactImage := rfl
  terminal := tests.NebulaWasmState.completedReturned
  realRowCountPositive := by decide
  realRowCountBound := by decide

theorem explicit_output_matches_state :
    resultImage.outputPresent = result.finalApplicationState.output.enabled ∧
      resultImage.outputValueLow = result.finalApplicationState.output.low ∧
      resultImage.outputValueHigh = result.finalApplicationState.output.high :=
  resultDecodes.output_fields_equal_state

theorem returned_status_is_exact :
    resultImage.exitCode = result.finalApplicationState.trapCode ∧
      result.finalApplicationState.halted = true ∧
      ((resultImage.terminationMode = .returned ∧
          result.finalApplicationState.trapped = false ∧
          resultImage.exitCode = 0) ∨
        (resultImage.terminationMode = .trapped ∧
          result.finalApplicationState.trapped = true ∧
          1 ≤ resultImage.exitCode ∧ resultImage.exitCode ≤ 7)) :=
  resultDecodes.mode_exit_and_flags_exact

end tests.NebulaWasmStatement
