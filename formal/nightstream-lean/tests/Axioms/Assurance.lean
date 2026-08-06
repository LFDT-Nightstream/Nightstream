import Nightstream.Assurance
import tests.Axioms.Support

/-!
Fail-closed gate for the current composed assurance facade. Every expectation
is checked when this module is built. The aggregate entrypoint imports all
ownership groups for the selected protocol.
-/

/-- info: 'Nightstream.Assurance.FPrimeTrace.accepted_trace_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeTrace.accepted_trace_sound

/-- info: 'Nightstream.Assurance.FPrimeTrace.accepted_trace_valid_execution' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeTrace.accepted_trace_valid_execution

/-- info: 'Nightstream.Assurance.FPrimeCircuit.split_check_eq_true_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeCircuit.split_check_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeCircuitTrace.candidate_sound_or_bad' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeCircuitTrace.candidate_sound_or_bad

/-- info: 'Nightstream.Assurance.FPrimeCircuitTrace.accepted_to_candidate' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeCircuitTrace.accepted_to_candidate
