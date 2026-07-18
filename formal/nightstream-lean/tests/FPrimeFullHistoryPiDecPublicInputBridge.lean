import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiDec.PublicInputBridge

/-! Focused surface checks for the typed production Π_DEC public carrier. -/

namespace tests.FPrimeFullHistoryPiDecPublicInputBridge

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.PublicInputBridge

#check packedSlot_exact
#check typedColumn
#check decode_injective_of_length
#check decodedPublicInput_apply
#check decodedParent_length
#check decodedChild_length
#check semanticRecompose_apply
#check decode_combine
#check strictAccepted_typedPublicInputEquation

end tests.FPrimeFullHistoryPiDecPublicInputBridge
