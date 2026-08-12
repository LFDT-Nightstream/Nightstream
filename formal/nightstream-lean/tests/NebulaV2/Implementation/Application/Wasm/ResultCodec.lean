import Nightstream.Implementation.NebulaV2.Application.Wasm.ResultCodec
import tests.NebulaV2.Protocol.WasmState

set_option autoImplicit false

namespace tests.NebulaV2WasmResultCodec

open Nightstream.Implementation.NebulaV2.WasmResultCodec
open Nightstream.Protocol.NebulaV2.Completion
open Nightstream.Protocol.NebulaV2.Digest
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.Protocol.NebulaV2.WasmStatement

def zeroDigest : Nightstream.Protocol.NebulaV2.Digest.Value where
  lanes := fun _ => ⟨0, by decide⟩

def result : ExecutionResult AppStateVector
    Nightstream.Protocol.NebulaV2.Digest.Value where
  realApplicationRowCount := 1
  finalApplicationState := tests.NebulaV2WasmState.completed
  outcome := .returned none
  finalMemoryRoot := zeroDigest

def image : ProductionResultImage := ResultImage.ofResult result

def decoded : image.Decodes result where
  exactImage := rfl
  terminal := tests.NebulaV2WasmState.completedReturned
  realRowCountPositive := by decide
  realRowCountBound := by decide

theorem exact_result_length : (encode image).length = 2665 :=
  encode_length image

theorem accepted_result_has_unique_encoding
    (other : ProductionResultImage)
    (otherResult : ExecutionResult AppStateVector
      Nightstream.Protocol.NebulaV2.Digest.Value)
    (otherDecoded : other.Decodes otherResult)
    (same : encode image = encode other) :
    image = other :=
  encode_injective_of_decodes decoded otherDecoded same

end tests.NebulaV2WasmResultCodec
