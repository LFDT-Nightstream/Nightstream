import Nightstream.Implementation.Lowering.Nebula.ProductSemantics
import tests.Axioms.Support

/-! Fail-closed dependency guards for the Nebula product semantics. -/

/-- info: 'Nightstream.Implementation.Lowering.Nebula.ProductSemantics.extensionRows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.ProductSemantics.extensionRows_sound

/-- info: 'Nightstream.Implementation.Lowering.Nebula.ProductSemantics.extensionRows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.ProductSemantics.extensionRows_honest

/-- info: 'Nightstream.Implementation.Lowering.Nebula.ProductSemantics.operationProduct_sound_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.ProductSemantics.operationProduct_sound_of_rows

/-- info: 'Nightstream.Implementation.Lowering.Nebula.ProductSemantics.scanProduct_sound_of_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.ProductSemantics.scanProduct_sound_of_rows

/-- info: 'Nightstream.Implementation.Lowering.Nebula.ProductSemantics.wasm42x6_public_products_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.ProductSemantics.wasm42x6_public_products_sound
