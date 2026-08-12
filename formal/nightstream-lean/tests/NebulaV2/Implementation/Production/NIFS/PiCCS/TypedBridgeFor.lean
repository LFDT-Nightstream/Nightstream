import Nightstream.Implementation.NebulaV2.Production.NIFS.PiCCS.TypedBridgeFor

/-! Regression surface for the exponent-indexed production PiCCS bridge. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionProductPiCcsTypedBridgeFor

open Nightstream.Implementation.NebulaV2.ProductionProductPiCcsTypedBridgeFor

#check ExactProof
#check Wires
#check Placement
#check decodedVerifierInput_eq
#check decodedCertificate_eq
#check decodedMessage_eq
#check valueReplay_eq_executionCoins
#check rows_imply_piCcsChain
#check rows_imply_piCcsCheck_true
#check rows_imply_outgoingState

end tests.NebulaV2ProductionProductPiCcsTypedBridgeFor
