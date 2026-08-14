import Nightstream.Implementation.Nebula.FPrime.State.HashCountermodels
import tests.Axioms.Support

/-! Dependency audit for the F-prime state-hash countermodel. -/

/-- info: 'Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.not_replayable_of_claim_variation' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.not_replayable_of_claim_variation

/-- info: 'Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.claimDependentHash_not_replayable' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.claimDependentHash_not_replayable

/-- info: 'Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.stateOnlyHash_replayable' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.stateOnlyHash_replayable

/-- info: 'Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.omitting_verifier_keys_aliases_distinct_frames' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.omitting_verifier_keys_aliases_distinct_frames

/-- info: 'Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.omitting_iteration_aliases_distinct_frames' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.omitting_iteration_aliases_distinct_frames

/-- info: 'Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.omitting_initial_state_aliases_distinct_frames' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.omitting_initial_state_aliases_distinct_frames

/-- info: 'Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.omitting_current_state_aliases_distinct_frames' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.omitting_current_state_aliases_distinct_frames

/-- info: 'Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.omitting_running_aliases_distinct_frames' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.omitting_running_aliases_distinct_frames

/-- info: 'Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.omitting_program_counter_aliases_distinct_frames' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.omitting_program_counter_aliases_distinct_frames

/-- info: 'Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.weak_checks_allow_running_substitution' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.weak_checks_allow_running_substitution

/-- info: 'Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.exact_running_alias_rejects_substitution' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeStateHashCountermodels.exact_running_alias_rejects_substitution
