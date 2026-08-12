import Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime
import tests.Axioms.Support

/-!
Fail-closed dependency guard for the exact delayed full-claim authority chain.
-/

/-- info: 'Nightstream.Implementation.NebulaV2.StateAuthorityFullClaim.carries_of_same' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.StateAuthorityFullClaim.carries_of_same

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.ConsumingInvocation.ofRecursive' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.ConsumingInvocation.ofRecursive

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.ConsumingInvocation.ofTerminal' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.ConsumingInvocation.ofTerminal

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.Candidate.sound_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.Candidate.sound_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.BaseProducer.initialExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.BaseProducer.initialExact

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.BaseProducer.opensExactInitialCarry' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.BaseProducer.opensExactInitialCarry

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.RecursiveEdge.ofDelayedProducer' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.RecursiveEdge.ofDelayedProducer

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.RecursiveEdge.producerCarries' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.RecursiveEdge.producerCarries

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.RecursiveEdge.sameOrFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.RecursiveEdge.sameOrFailure

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.RecursiveEdge.boundary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.RecursiveEdge.boundary

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.TerminalEdge.ofDelayedProducer' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.TerminalEdge.ofDelayedProducer

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.TerminalEdge.producerCarries' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.TerminalEdge.producerCarries

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.TerminalEdge.sameOrFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.TerminalEdge.sameOrFailure

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.TerminalEdge.boundary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.TerminalEdge.boundary

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.ManifestCandidate.sound_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.ManifestCandidate.sound_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.RowBoundChain.authoritySoundOrCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime.RowBoundChain.authoritySoundOrCollision
