import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredSeptenaryRustConformance
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the finite-corpus centered-septenary Rust bridge.
-/

namespace NightstreamTests.Axioms.CenteredSeptenaryRustConformance

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.CenteredSeptenaryRustConformance

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryRustConformance.generated_sources_are_exact_boundaries' does not depend on any axioms -/
#guard_msgs in
#audit_axioms generated_sources_are_exact_boundaries

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryRustConformance.generated_cases_match_lean_encoder' does not depend on any axioms -/
#guard_msgs in
#audit_axioms generated_cases_match_lean_encoder

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryRustConformance.generated_case_count' does not depend on any axioms -/
#guard_msgs in
#audit_axioms generated_case_count

/-- info: 'Nightstream.Implementation.R1CS.CenteredSeptenaryRustConformance.generated_digit_counts' does not depend on any axioms -/
#guard_msgs in
#audit_axioms generated_digit_counts

end NightstreamTests.Axioms.CenteredSeptenaryRustConformance

