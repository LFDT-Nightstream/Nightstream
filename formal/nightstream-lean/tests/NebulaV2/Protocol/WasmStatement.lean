import Nightstream.Protocol.NebulaV2.WasmStatement
import tests.NebulaV2.Protocol.WasmState

set_option autoImplicit false

namespace tests.NebulaV2WasmStatement

open Nightstream.Protocol.NebulaV2.ApplicationTrace
open Nightstream.Protocol.NebulaV2.Completion
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.Protocol.NebulaV2.WasmStatement

def result : ExecutionResult AppStateVector Nat where
  realApplicationRowCount := 1
  finalApplicationState := tests.NebulaV2WasmState.completed
  outcome := .returned none
  finalMemoryRoot := 9

def resultImage : ResultImage Nat := ResultImage.ofResult result

def resultDecodes : resultImage.Decodes result where
  exactImage := rfl
  terminal := tests.NebulaV2WasmState.completedReturned
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

end tests.NebulaV2WasmStatement
