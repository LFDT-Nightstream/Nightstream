import Nightstream.Implementation.NebulaV2.FPrime.Claim.DelayedTrace
import tests.Axioms.Support

/-! Fail-closed dependency guard for the exact V2 delayed producer link. -/

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimDelayedTrace.Producer.outgoing_of_carries' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimDelayedTrace.Producer.outgoing_of_carries

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimDelayedTrace.Producer.carries_of_outgoing' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimDelayedTrace.Producer.carries_of_outgoing

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimDelayedTrace.Producer.freshLinked_of_outgoing' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimDelayedTrace.Producer.freshLinked_of_outgoing
