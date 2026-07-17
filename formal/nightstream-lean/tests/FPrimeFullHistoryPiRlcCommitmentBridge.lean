import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.CommitmentBridge

/-!
Focused surface checks for typed production `Pi_RLC` commitments.

Assurance tier: model-level compile-time surface checks.
-/

namespace tests.FPrimeFullHistoryPiRlcCommitmentBridge

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.CommitmentBridge

#check decodeCommitmentRings
#check decodeCommitmentRings_phi81Combine

end tests.FPrimeFullHistoryPiRlcCommitmentBridge
