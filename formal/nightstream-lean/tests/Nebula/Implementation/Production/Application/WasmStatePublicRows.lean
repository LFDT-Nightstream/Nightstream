import Nightstream.Implementation.Nebula.Production.Application.WasmStatePublicRows

/-! Surface gate for exact WASM-state public-bit recomposition rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionWasmStatePublicRows

open Nightstream.Implementation.Nebula.ProductionWasmStatePublicRows

#check rows_length_exact
#check encodePieces_eq
#check pieceValues_eq
#check fieldColumn_eq_pieceValue
#check fields_exact

end tests.NebulaProductionWasmStatePublicRows
