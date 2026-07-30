import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingActionAudit
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.FPrimePhi81RingAction

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction.rows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81RingAction.rows_length

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction.rows_cost' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81RingAction.rows_cost

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction.columns_cost' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81RingAction.columns_cost

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction.product_exact_of_satisfied' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81RingAction.product_exact_of_satisfied

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction.rows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81RingAction.rows_sound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction.rows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81RingAction.rows_honest

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction.ownership_is_positional' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81RingAction.ownership_is_positional

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction.row_ids_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81RingAction.row_ids_nodup

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction.rows_supported' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81RingAction.rows_supported

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction.rawRows_eq_map_owners' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81RingAction.rawRows_eq_map_owners

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction.allOwners_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81RingAction.allOwners_nodup

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction.allOwners_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81RingAction.allOwners_length

end NightstreamTests.Axioms.FPrimePhi81RingAction
