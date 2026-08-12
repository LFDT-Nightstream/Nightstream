import Nightstream.Protocol.FPrime.DelayedTrace
import tests.Axioms.Support

/-! Fail-closed dependency guard for delayed F-prime trace closure. -/

/-- info: 'Nightstream.Protocol.FPrime.Step.localHolds_producer_facts' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.localHolds_producer_facts

/-- info: 'Nightstream.Protocol.FPrime.Step.localHolds_consumes_active_latest' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.localHolds_consumes_active_latest

/-- info: 'Nightstream.Protocol.FPrime.DelayedTrace.Invocation.classified' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.DelayedTrace.Invocation.classified

/-- info: 'Nightstream.Protocol.FPrime.DelayedTrace.Invocation.isBase_of_prior_initial' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.DelayedTrace.Invocation.isBase_of_prior_initial

/-- info: 'Nightstream.Protocol.FPrime.DelayedTrace.Invocation.next_isRecursive' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.DelayedTrace.Invocation.next_isRecursive

/-- info: 'Nightstream.Protocol.FPrime.DelayedTrace.Candidate.closeAll' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.DelayedTrace.Candidate.closeAll

/-- info: 'Nightstream.Protocol.FPrime.DelayedTrace.Candidate.rest_isRecursive' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.DelayedTrace.Candidate.rest_isRecursive

/-- info: 'Nightstream.Protocol.FPrime.DelayedTrace.Candidate.exactBranchSchedule' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.DelayedTrace.Candidate.exactBranchSchedule

/-- info: 'Nightstream.Protocol.FPrime.DelayedTrace.Candidate.headOutgoingLinked' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.DelayedTrace.Candidate.headOutgoingLinked
