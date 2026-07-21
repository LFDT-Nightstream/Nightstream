import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Nifs.PiDec.ActiveResultBridge

/-!
Focused interface regression for the model-level active `PiDEC` result seam.

The artifact instantiation is intentionally absent: it still owes the exact
parent-point and ordered child-payload decoder facts exposed by this module.
-/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointPiDecActiveResultBridge

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.ActiveResultBridge

#check decodedFoldResult
#check ParentPointBound
#check ChildPayloadsBound
#check FamilyPayloadBound
#check familyPayloadBound_of_decoderFacts
#check decodedFoldResult_eq_resultOf
#check claimsAccepted_decodedFoldResult_eq_resultOf
#check claimsAccepted_outgoingState_rewrite

end Nightstream.Tests.FPrimeSelectiveFixedPointPiDecActiveResultBridge
