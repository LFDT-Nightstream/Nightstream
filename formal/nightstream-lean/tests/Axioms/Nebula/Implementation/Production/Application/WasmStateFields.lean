import Nightstream.Implementation.Nebula.Production.Application.WasmStateFields
import tests.Axioms.Support

/-! Dependency audit for the field-native production WASM state encoding. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionWasmStateFields.encode_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionWasmStateFields.encode_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionWasmStateFields.encode_fields_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionWasmStateFields.encode_fields_canonical
