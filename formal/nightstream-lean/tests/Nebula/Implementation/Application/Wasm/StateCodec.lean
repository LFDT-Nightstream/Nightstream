import Nightstream.Implementation.Nebula.Application.Wasm.StateCodec
import tests.Nebula.Protocol.WasmState

set_option autoImplicit false

namespace tests.NebulaWasmStateCodec

open Nightstream.Implementation.Nebula.WasmStateCodec
open Nightstream.Protocol.Nebula.WasmStateEncoding

def runningImage : Image := encode tests.NebulaWasmState.running

theorem runningImageCanonical : runningImage.Canonical := by
  simpa [runningImage, canonical_encode_iff] using
    tests.NebulaWasmState.validRunning

theorem exact_wire_shape :
    (Nightstream.Implementation.Nebula.WasmStateCodec.encode
      runningImage).length = 2293 ∧
      ∀ digit ∈
        Nightstream.Implementation.Nebula.WasmStateCodec.encode
          runningImage,
        digit < 2 :=
  ⟨encode_exact_length runningImage, encode_binary runningImage⟩

theorem canonical_wire_has_unique_state
    (other : Image) (otherCanonical : other.Canonical)
    (same :
      Nightstream.Implementation.Nebula.WasmStateCodec.encode
          runningImage =
        Nightstream.Implementation.Nebula.WasmStateCodec.encode other) :
    runningImage = other :=
  encode_injective_on_canonical runningImageCanonical otherCanonical same

end tests.NebulaWasmStateCodec
