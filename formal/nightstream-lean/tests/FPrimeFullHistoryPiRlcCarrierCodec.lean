import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.CarrierCodec

/-! Focused surface checks for the matrix-indexed public-carrier codec. -/

namespace tests.FPrimeFullHistoryPiRlcCarrierCodec

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.CarrierCodec

#check encodeCommitment
#check encodeX
#check encodeYRing
#check canonical
#check canonical_yRing_encode
#check encodeCommitment_decodeOpening
#check encodeX_decodeOpening
#check encodeYRing_decodeOpening
#check canonical_artifact

end tests.FPrimeFullHistoryPiRlcCarrierCodec
