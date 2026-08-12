import Nightstream.Implementation.NebulaV2.ProductionWasmStatePublicRows

/-! Surface gate for exact WASM-state public-bit recomposition rows. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionWasmStatePublicRows

open Nightstream.Implementation.NebulaV2.ProductionWasmStatePublicRows

#check rows_length_exact
#check encodePieces_eq
#check pieceValues_eq
#check fieldColumn_eq_pieceValue
#check fields_exact

end tests.NebulaV2ProductionWasmStatePublicRows
