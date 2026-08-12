import Nightstream.Implementation.NebulaV2.FullClaimManifestTrace

set_option autoImplicit false

namespace tests.NebulaV2FullClaimManifestTrace

open Nightstream.Implementation.NebulaV2.FullClaimManifestTrace

#check RecursiveNode.outgoingLinked_of_next
#check RecursiveNode.edgeOfNext
#check TerminalNode.boundary
#check TerminalNode.producerCarries
#check TerminalNode.trailingLink
#check TerminalNode.edge
#check Candidate.toDelayed
#check Candidate.toManifest
#check Candidate.closed
#check Candidate.authoritySoundOrCollision
#check Candidate.exactProducerCount
#check ExactChain.exactConsumerCount
#check ExactChain.exactProducerCount
#check ExactChain.completeDelayedSchedule
#check ExactChain.closedTrace
#check ExactChain.tailProducersAreRecursive
#check ExactChain.exactBranchSchedule
#check ExactChain.authoritySoundOrCollision
#check ExactChain.baseAuthority

end tests.NebulaV2FullClaimManifestTrace
