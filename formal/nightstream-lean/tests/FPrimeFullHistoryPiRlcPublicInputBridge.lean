import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.PublicInputBridge

/-!
Focused surface checks for the typed production Π_RLC public carrier.

Assurance tier: model-level compile-time surface checks.
-/

namespace tests.FPrimeFullHistoryPiRlcPublicInputBridge

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge

#check decodeXRings
#check publicBlock_decodeXRings
#check assembleXColumns_length
#check assembleXColumns_getD
#check packXRings
#check packXRings_length
#check decode_packXRings
#check decode_assembledX
#check decode_codec_x
#check decodeXRings_phi81Combine
#check typedPublicInputEquation_of_refinement
#check typedOutput_eq_parent_of_wiring
#check typedOutput_eq_piDecParent_of_wiring
#check typedPiRlcToPiDecParentEquation_of_refinement
#check typedPiRlcPiDecPublicInputComposition
#check typedPiRlcPiDecPublicInputComposition_relabel

end tests.FPrimeFullHistoryPiRlcPublicInputBridge
