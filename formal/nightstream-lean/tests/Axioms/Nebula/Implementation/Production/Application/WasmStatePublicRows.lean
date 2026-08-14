import Nightstream.Implementation.Nebula.Production.Application.WasmStatePublicRows
import tests.Axioms.Support

/-! Dependency gate for exact WASM-state public-bit recomposition rows. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionWasmStatePublicRows

open Nightstream.Implementation.Nebula.ProductionWasmStatePublicRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionWasmStatePublicRows.fields_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fields_exact

end tests.Axioms.NebulaProductionWasmStatePublicRows
