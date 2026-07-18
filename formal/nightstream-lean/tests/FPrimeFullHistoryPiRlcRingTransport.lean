import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.RingTransport

/-!
Focused surface checks for production-list to typed-`RingF` transport.

Assurance tier: model-level compile-time surface checks.
-/

namespace tests.FPrimeFullHistoryPiRlcRingTransport

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RingTransport

#check scalarSum
#check phi81Combine_eq_scalarSum
#check phi81Combine_coefficient
#check productSum
#check ringOfList_phi81Combine

end tests.FPrimeFullHistoryPiRlcRingTransport
