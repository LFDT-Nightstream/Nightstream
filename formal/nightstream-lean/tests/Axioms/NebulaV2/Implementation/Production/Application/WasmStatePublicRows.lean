import Nightstream.Implementation.NebulaV2.Production.Application.WasmStatePublicRows
import tests.Axioms.Support

/-! Dependency gate for exact WASM-state public-bit recomposition rows. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionWasmStatePublicRows

open Nightstream.Implementation.NebulaV2.ProductionWasmStatePublicRows

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionWasmStatePublicRows.fields_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fields_exact

end tests.Axioms.NebulaV2ProductionWasmStatePublicRows
