import Nightstream.Implementation.NebulaV2.WasmStateCodec
import tests.NebulaV2WasmState

set_option autoImplicit false

namespace tests.NebulaV2WasmStateCodec

open Nightstream.Implementation.NebulaV2.WasmStateCodec
open Nightstream.Protocol.NebulaV2.WasmStateEncoding

def runningImage : Image := encode tests.NebulaV2WasmState.running

theorem runningImageCanonical : runningImage.Canonical := by
  simpa [runningImage, canonical_encode_iff] using
    tests.NebulaV2WasmState.validRunning

theorem exact_wire_shape :
    (Nightstream.Implementation.NebulaV2.WasmStateCodec.encode
      runningImage).length = 2293 ∧
      ∀ digit ∈
        Nightstream.Implementation.NebulaV2.WasmStateCodec.encode
          runningImage,
        digit < 2 :=
  ⟨encode_exact_length runningImage, encode_binary runningImage⟩

theorem canonical_wire_has_unique_state
    (other : Image) (otherCanonical : other.Canonical)
    (same :
      Nightstream.Implementation.NebulaV2.WasmStateCodec.encode
          runningImage =
        Nightstream.Implementation.NebulaV2.WasmStateCodec.encode other) :
    runningImage = other :=
  encode_injective_on_canonical runningImageCanonical otherCanonical same

end tests.NebulaV2WasmStateCodec
