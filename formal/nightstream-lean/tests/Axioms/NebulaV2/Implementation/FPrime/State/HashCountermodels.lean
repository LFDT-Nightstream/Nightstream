import Nightstream.Implementation.NebulaV2.FPrime.State.HashCountermodels
import tests.Axioms.Support

/-! Dependency audit for the F-prime state-hash countermodel. -/

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.not_replayable_of_claim_variation' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.not_replayable_of_claim_variation

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.claimDependentHash_not_replayable' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.claimDependentHash_not_replayable

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.stateOnlyHash_replayable' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.stateOnlyHash_replayable

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.omitting_verifier_keys_aliases_distinct_frames' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.omitting_verifier_keys_aliases_distinct_frames

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.omitting_iteration_aliases_distinct_frames' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.omitting_iteration_aliases_distinct_frames

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.omitting_initial_state_aliases_distinct_frames' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.omitting_initial_state_aliases_distinct_frames

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.omitting_current_state_aliases_distinct_frames' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.omitting_current_state_aliases_distinct_frames

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.omitting_running_aliases_distinct_frames' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.omitting_running_aliases_distinct_frames

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.omitting_program_counter_aliases_distinct_frames' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.omitting_program_counter_aliases_distinct_frames

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.weak_checks_allow_running_substitution' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.weak_checks_allow_running_substitution

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.exact_running_alias_rejects_substitution' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeStateHashCountermodels.exact_running_alias_rejects_substitution
