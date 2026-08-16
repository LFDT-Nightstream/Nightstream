import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRelation
import tests.Axioms.Support

/-! Fail-closed axiom guard for the fixed 32-field lifecycle relation. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.payload_preimage_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms payload_preimage_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.ActiveArm.selector_eq_true_iff' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ActiveArm.selector_eq_true_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Base.baseLocalHolds' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Base.baseLocalHolds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Base.selected_phase_starts_empty' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Base.selected_phase_starts_empty

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Recursive.checked_fold' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Recursive.checked_fold

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Recursive.selected_phase_consumes_latest' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Recursive.selected_phase_consumes_latest

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Terminal.phase_complete' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Terminal.phase_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Terminal.terminal_relations_complete' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Terminal.terminal_relations_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Terminal.delayed_nebula_finalized' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Terminal.delayed_nebula_finalized

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation.Terminal.frame_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Terminal.frame_exact
