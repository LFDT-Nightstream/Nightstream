import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23SeparatorConformance
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the separator conformance bridge.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.Poseidon23SeparatorConformance

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23SeparatorConformance.selected_slot_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon23SeparatorConformance.selected_slot_zero

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23SeparatorConformance.prior_next_differ_at_slot_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon23SeparatorConformance.prior_next_differ_at_slot_zero

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23SeparatorConformance.prior_next_agree_off_slot_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon23SeparatorConformance.prior_next_agree_off_slot_zero

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23SeparatorConformance.SeparatingPlan.selected_slot_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon23SeparatorConformance.SeparatingPlan.selected_slot_zero

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23SeparatorConformance.SeparatingPlan.prior_next_differ' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon23SeparatorConformance.SeparatingPlan.prior_next_differ


/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23SeparatorConformance.selected_at_iteration_slot' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon23SeparatorConformance.selected_at_iteration_slot

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23SeparatorConformance.SeparatingPlan.prior_next_agree' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon23SeparatorConformance.SeparatingPlan.prior_next_agree

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23SeparatorConformance.SeparatingPlan.next_is_separated' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon23SeparatorConformance.SeparatingPlan.next_is_separated

end NightstreamTests.Axioms.Poseidon23SeparatorConformance
