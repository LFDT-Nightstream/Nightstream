import Nightstream.Protocol.FPrime.DelayedTrace

set_option autoImplicit false

namespace tests.FPrimeDelayedTrace

open Nightstream.Protocol.FPrime.DelayedTrace

#check Invocation.producerFacts
#check Invocation.classified
#check Invocation.isBase_of_prior_initial
#check Invocation.next_isRecursive
#check Invocation.outgoingLinked_of_next
#check Invocation.close
#check Candidate.closeAll
#check Candidate.headOutgoingLinked
#check Candidate.rest_isRecursive
#check Candidate.exactBranchSchedule

end tests.FPrimeDelayedTrace
