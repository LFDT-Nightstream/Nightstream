import Nightstream.Protocol.Nebula.WasmStateEncoding

set_option autoImplicit false

namespace tests.NebulaWasmStateEncoding

open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStateEncoding

theorem schema_has_exact_shape :
    fieldCount = 55 ∧ serializedBitCount = 2293 :=
  ⟨fieldCount_eq, serializedBitCount_eq⟩

theorem no_two_states_share_one_typed_image
    (left right : AppStateVector)
    (equal : encode left = encode right) :
    left = right :=
  encode_injective equal

theorem image_round_trip (image : Image) :
    encode (decode image) = image :=
  encode_decode image

end tests.NebulaWasmStateEncoding
