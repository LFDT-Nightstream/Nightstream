import Nightstream.Implementation.Nebula.FPrime.Claim.DelayedTrace
import tests.Axioms.Support

/-! Fail-closed dependency guard for the exact V2 delayed producer link. -/

/-- info: 'Nightstream.Implementation.Nebula.FullClaimDelayedTrace.Producer.outgoing_of_carries' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimDelayedTrace.Producer.outgoing_of_carries

/-- info: 'Nightstream.Implementation.Nebula.FullClaimDelayedTrace.Producer.carries_of_outgoing' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimDelayedTrace.Producer.carries_of_outgoing

/-- info: 'Nightstream.Implementation.Nebula.FullClaimDelayedTrace.Producer.freshLinked_of_outgoing' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimDelayedTrace.Producer.freshLinked_of_outgoing
