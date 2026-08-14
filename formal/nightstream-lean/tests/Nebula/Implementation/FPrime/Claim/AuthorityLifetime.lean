import Nightstream.Implementation.Nebula.FPrime.Claim.AuthorityLifetime

set_option autoImplicit false

namespace tests.NebulaFullClaimAuthorityLifetime

open Nightstream.Implementation.Nebula.FullClaimAuthorityLifetime

#check ConsumingInvocation.ofRecursive
#check ConsumingInvocation.ofTerminal
#check Candidate.sound_or_collision
#check Exact.headCarriesProducer
#check Exact.lastHasNoOutgoing
#check BaseProducer.initialExact
#check BaseProducer.opensExactInitialCarry
#check RecursiveEdge.ofDelayedProducer
#check RecursiveEdge.matchesSelected
#check RecursiveEdge.producerCarries
#check RecursiveEdge.sameOrFailure
#check RecursiveEdge.boundary
#check TerminalEdge.ofDelayedProducer
#check TerminalEdge.matchesSelected
#check TerminalEdge.producerCarries
#check TerminalEdge.sameOrFailure
#check TerminalEdge.boundary
#check ManifestCandidate.sound_or_collision
#check RowBoundChain.exactInvocationCount
#check RowBoundChain.completeDelayedSchedule
#check RowBoundChain.authoritySoundOrCollision

end tests.NebulaFullClaimAuthorityLifetime
