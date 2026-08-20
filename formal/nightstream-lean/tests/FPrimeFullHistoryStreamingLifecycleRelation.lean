import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRelation

/-! Focused surface for the fixed 32-field streaming lifecycle relation. -/

set_option autoImplicit false

namespace tests.FPrimeFullHistoryStreamingLifecycleRelation

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation

#check frame_length
#check frame_canonical
#check payload_preimage_exact
#check ActiveArm.selector_eq_true_iff
#check ActiveArm.selected_eq_of_cursor
#check Invocation.before_public_exact
#check Invocation.prior_frame_exact
#check Invocation.next_frame_exact
#check Invocation.prior_frame_canonical
#check Invocation.next_frame_canonical
#check Invocation.selected_cursor_exact
#check Invocation.selectedPhase
#check Base.baseLocalHolds
#check Base.prior_counters_zero
#check Base.selected_phase_starts_empty
#check Recursive.checked_fold
#check Recursive.selected_phase_consumes_latest
#check Terminal.phase_complete
#check Terminal.terminal_relations_complete
#check Terminal.delayed_nebula_finalized
#check Terminal.program_counter_exact
#check Terminal.accumulator_exact
#check Terminal.state_pinned
#check Terminal.frame_exact
#check Terminal.frame_canonical

end tests.FPrimeFullHistoryStreamingLifecycleRelation
