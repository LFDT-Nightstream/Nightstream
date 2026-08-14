import Nightstream.Implementation.Nebula.FPrime.Claim.FieldNativeCarrierAlias
import tests.Axioms.Support

/-! Dependency audit for the field-native physical alias. -/

/-- info: 'Nightstream.Implementation.Nebula.FieldNativeCarrierAlias.runningValues_eq' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FieldNativeCarrierAlias.runningValues_eq

/-- info: 'Nightstream.Implementation.Nebula.FieldNativeCarrierAlias.bundleValues_eq' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FieldNativeCarrierAlias.bundleValues_eq
