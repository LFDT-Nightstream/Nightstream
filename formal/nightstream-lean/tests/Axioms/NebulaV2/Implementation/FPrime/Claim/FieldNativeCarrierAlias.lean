import Nightstream.Implementation.NebulaV2.FPrime.Claim.FieldNativeCarrierAlias
import tests.Axioms.Support

/-! Dependency audit for the field-native physical alias. -/

/-- info: 'Nightstream.Implementation.NebulaV2.FieldNativeCarrierAlias.runningValues_eq' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FieldNativeCarrierAlias.runningValues_eq

/-- info: 'Nightstream.Implementation.NebulaV2.FieldNativeCarrierAlias.bundleValues_eq' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FieldNativeCarrierAlias.bundleValues_eq
