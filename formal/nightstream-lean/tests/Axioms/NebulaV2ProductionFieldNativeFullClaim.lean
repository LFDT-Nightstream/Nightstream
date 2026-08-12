import Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim
import tests.Axioms.Support

/-! Dependency audit for the field-native production full claim. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim.authorityImage_coordinate_count' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim.authorityImage_coordinate_count

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim.authorityImage_injective_on_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim.authorityImage_injective_on_canonical

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim.nifsInput_eq_recovers_direct_authority_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim.nifsInput_eq_recovers_direct_authority_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim.Value.toProtocolClaim_injective' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim.Value.toProtocolClaim_injective
