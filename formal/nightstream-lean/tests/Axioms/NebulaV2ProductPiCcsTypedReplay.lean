import Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplay
import tests.Axioms.Support

/-! Dependency audit for key-independent typed PiCCS transcript replay. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplay.valueReplay_eq_derived_of_components' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplay.valueReplay_eq_derived_of_components

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplay.decodedAlpha_coordinates_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplay.decodedAlpha_coordinates_eq

/-- info: 'Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplay.decodedPoint_coordinates_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductPiCcsTypedReplay.decodedPoint_coordinates_eq
