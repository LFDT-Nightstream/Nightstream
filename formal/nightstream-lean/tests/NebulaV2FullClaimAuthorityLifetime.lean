import Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime

set_option autoImplicit false

namespace tests.NebulaV2FullClaimAuthorityLifetime

open Nightstream.Implementation.NebulaV2.FullClaimAuthorityLifetime

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

end tests.NebulaV2FullClaimAuthorityLifetime
