import Nightstream.Implementation.NebulaV2.NIFS.Core.ExactConfiguration
import tests.Axioms.Support

/-! Axiom gate for exact V2 paper-NIFS selection. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductExactNifsConfiguration.configuration_decoder_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductExactNifsConfiguration.configuration_decoder_exact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductExactNifsConfiguration.configuration_key_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductExactNifsConfiguration.configuration_key_exact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductExactNifsConfiguration.configuration_samplerCheck_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductExactNifsConfiguration.configuration_samplerCheck_exact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductExactNifsConfiguration.accepted_input_has_exact_fields' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductExactNifsConfiguration.accepted_input_has_exact_fields
