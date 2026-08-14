import Nightstream.Implementation.Nebula.FPrime.Claim.AuthorityLifetime
import tests.Axioms.Support

/-!
Fail-closed dependency guard for the exact delayed full-claim authority chain.
-/

/-- info: 'Nightstream.Implementation.Nebula.StateAuthorityFullClaim.carries_of_same' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.StateAuthorityFullClaim.carries_of_same

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.ConsumingInvocation.ofRecursive' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.ConsumingInvocation.ofRecursive

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.ConsumingInvocation.ofTerminal' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.ConsumingInvocation.ofTerminal

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.Candidate.sound_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.Candidate.sound_or_collision

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.BaseProducer.initialExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.BaseProducer.initialExact

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.BaseProducer.opensExactInitialCarry' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.BaseProducer.opensExactInitialCarry

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.RecursiveEdge.ofDelayedProducer' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.RecursiveEdge.ofDelayedProducer

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.RecursiveEdge.producerCarries' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.RecursiveEdge.producerCarries

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.RecursiveEdge.sameOrFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.RecursiveEdge.sameOrFailure

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.RecursiveEdge.boundary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.RecursiveEdge.boundary

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.TerminalEdge.ofDelayedProducer' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.TerminalEdge.ofDelayedProducer

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.TerminalEdge.producerCarries' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.TerminalEdge.producerCarries

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.TerminalEdge.sameOrFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.TerminalEdge.sameOrFailure

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.TerminalEdge.boundary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.TerminalEdge.boundary

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.ManifestCandidate.sound_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.ManifestCandidate.sound_or_collision

/-- info: 'Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.RowBoundChain.authoritySoundOrCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime.RowBoundChain.authoritySoundOrCollision
