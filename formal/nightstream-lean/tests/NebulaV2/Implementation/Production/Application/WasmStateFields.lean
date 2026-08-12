import Nightstream.Implementation.NebulaV2.Production.Application.WasmStateFields

/-! Regression surface for the lossless field-native production WASM state. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionWasmStateFields

open Nightstream.Implementation.NebulaV2.ProductionWasmStateFields

#check native_width_sum_exact
#check encode_length
#check fieldValue_eq_of_encodeTag_eq
#check encode_injective
#check encode_fields_canonical

end tests.NebulaV2ProductionWasmStateFields
