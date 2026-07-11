import Nightstream.Implementation.R1CS.FPrimeChunkDigestSound

namespace NightstreamTests.FPrimeChunkDigest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeChunkDigest
open Nightstream.Implementation.R1CS.FPrimeChunkDigestSound

set_option maxRecDepth 262144

example : bindingRows.length = 4 := bindingRows_length
example : claimedColumns.length = computedColumns.length := by native_decide
example : bindingRowStart + bindingRows.length = fullRowCount := by native_decide

#check definitions_length
#check fPrimeChunkDigest_sound
#check fPrimeChunkDigest_claim_unique
#check fPrimeChunkDigest_complete

end NightstreamTests.FPrimeChunkDigest
