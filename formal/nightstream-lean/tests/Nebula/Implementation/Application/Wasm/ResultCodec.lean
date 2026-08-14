import Nightstream.Implementation.Nebula.Application.Wasm.ResultCodec
import tests.Nebula.Protocol.WasmState

set_option autoImplicit false

namespace tests.NebulaWasmResultCodec

open Nightstream.Implementation.Nebula.WasmResultCodec
open Nightstream.Protocol.Nebula.Completion
open Nightstream.Protocol.Nebula.Digest
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStatement

def zeroDigest : Nightstream.Protocol.Nebula.Digest.Value where
  lanes := fun _ => ⟨0, by decide⟩

def result : ExecutionResult AppStateVector
    Nightstream.Protocol.Nebula.Digest.Value where
  realApplicationRowCount := 1
  finalApplicationState := tests.NebulaWasmState.completed
  outcome := .returned none
  finalMemoryRoot := zeroDigest

def image : ProductionResultImage := ResultImage.ofResult result

def decoded : image.Decodes result where
  exactImage := rfl
  terminal := tests.NebulaWasmState.completedReturned
  realRowCountPositive := by decide
  realRowCountBound := by decide

theorem exact_result_length : (encode image).length = 2665 :=
  encode_length image

theorem accepted_result_has_unique_encoding
    (other : ProductionResultImage)
    (otherResult : ExecutionResult AppStateVector
      Nightstream.Protocol.Nebula.Digest.Value)
    (otherDecoded : other.Decodes otherResult)
    (same : encode image = encode other) :
    image = other :=
  encode_injective_of_decodes decoded otherDecoded same

end tests.NebulaWasmResultCodec
