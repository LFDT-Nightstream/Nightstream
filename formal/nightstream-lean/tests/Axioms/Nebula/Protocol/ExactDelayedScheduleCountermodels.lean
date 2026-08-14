import Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels
import tests.Axioms.Support

/-! Dependency audit for exact delayed-schedule countermodels. -/

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.missing_terminal_passes_prefix_check' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.missing_terminal_passes_prefix_check

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.missing_terminal_breaks_equal_claim_counts' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.missing_terminal_breaks_equal_claim_counts

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.current_claim_rule_accepts_wrong_index' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.current_claim_rule_accepts_wrong_index

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.terminal_consumption_does_not_forbid_successor' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.terminal_consumption_does_not_forbid_successor

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.base_production_does_not_forbid_consumption' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.base_production_does_not_forbid_consumption

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.erased_exponent_aliases_distinct_relations' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.erased_exponent_aliases_distinct_relations

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.prior_consumption_does_not_bind_produced_successor' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.prior_consumption_does_not_bind_produced_successor

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.acceptance_and_claim_equality_do_not_forward_the_exact_proof' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.acceptance_and_claim_equality_do_not_forward_the_exact_proof

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.trailing_consumption_does_not_imply_closed_state' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.trailing_consumption_does_not_imply_closed_state

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.no_prior_claim_does_not_fix_base_index' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.no_prior_claim_does_not_fix_base_index

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.claim_count_does_not_fix_consumer_indexes' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.claim_count_does_not_fix_consumer_indexes

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.exact_index_does_not_fix_full_state' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.exact_index_does_not_fix_full_state

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.PaperTerminalGap.paper_terminal_accepts_open_delayed_memory' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.PaperTerminalGap.paper_terminal_accepts_open_delayed_memory

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.PaperTerminalGap.paper_terminal_acceptance_does_not_imply_a_trailing_fold' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.PaperTerminalGap.paper_terminal_acceptance_does_not_imply_a_trailing_fold

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.PaperTerminalGap.v2_terminal_implies_paper_terminal' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.PaperTerminalGap.v2_terminal_implies_paper_terminal

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.PaperTerminalGap.v2_terminal_rejects_open_delayed_memory' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.PaperTerminalGap.v2_terminal_rejects_open_delayed_memory

/-- info: 'Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.PaperTerminalGap.paper_terminal_is_strictly_weaker' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ExactDelayedScheduleCountermodels.PaperTerminalGap.paper_terminal_is_strictly_weaker
