import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalConcreteNifsPlain270Profile

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.source_arity_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.source_arity_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.public_carrier_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.public_carrier_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.fresh_public_input_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.fresh_public_input_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.running_assignment_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.running_assignment_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.freshCompletionCoordinates_eq_zeros' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.freshCompletionCoordinates_eq_zeros

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.running_tail_nonzero_witness' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.running_tail_nonzero_witness

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.sampler_shift_256_to_257' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.sampler_shift_256_to_257

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.Phase4Application.certification' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.Phase4Application.certification

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.Phase4Application.selected_preimage_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.Phase4Application.selected_preimage_length

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.Phase4Application.same_payload_next_preimage_is_separated' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.Phase4Application.same_payload_next_preimage_is_separated

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.Phase4Application.terminal_calls_independent' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.Phase4Application.terminal_calls_independent

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.Phase4Application.cost_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.Phase4Application.cost_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.operational_fe_domain_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.operational_fe_domain_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.operational_nc_domain_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.operational_nc_domain_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile.selected_setup_nifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPlain270Profile.selected_setup_nifs

end NightstreamTests.Axioms.CanonicalConcreteNifsPlain270Profile
