import Nightstream.Protocol.NebulaV2.Terminal
import tests.Axioms.Support

open Nightstream.Protocol.NebulaV2.Terminal

/-- info: 'Nightstream.Protocol.NebulaV2.Terminal.Accepted.consumes_exact_verified_trailing_claim' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Accepted.consumes_exact_verified_trailing_claim

/-- info: 'Nightstream.Protocol.NebulaV2.Terminal.Accepted.common_witness' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Accepted.common_witness

/-- info: 'Nightstream.Protocol.NebulaV2.Terminal.Countermodels.no_common_selector_opening' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Countermodels.no_common_selector_opening

/-- info: 'Nightstream.Protocol.NebulaV2.Terminal.Countermodels.no_common_opening_and_terminal_witness' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Countermodels.no_common_opening_and_terminal_witness

/-- info: 'Nightstream.Protocol.NebulaV2.Terminal.Countermodels.checking_only_first_folded_child_is_insufficient' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Countermodels.checking_only_first_folded_child_is_insufficient
