import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.PointBridge

/-! Focused surface checks for checked production `Pi_RLC` point decoding. -/

namespace tests.FPrimeFullHistoryPiRlcPointBridge

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PointBridge

#check TypedPoint
#check pointOfLength
#check pointOfLength_coordinates
#check decodeTypedPoint
#check Bound
#check bound_iff
#check decodeTypedPoint_isSome_iff
#check OutputPointBound
#check inputPointBound_of_outputPointBound
#check parentPointBound_of_outputPointBound

end tests.FPrimeFullHistoryPiRlcPointBridge
