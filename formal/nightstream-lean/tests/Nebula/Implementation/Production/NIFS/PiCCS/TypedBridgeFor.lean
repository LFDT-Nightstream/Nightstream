import Nightstream.Implementation.Nebula.Production.NIFS.PiCCS.TypedBridgeFor

/-! Regression surface for the exponent-indexed production PiCCS bridge. -/

set_option autoImplicit false

namespace tests.NebulaProductionProductPiCcsTypedBridgeFor

open Nightstream.Implementation.Nebula.ProductionProductPiCcsTypedBridgeFor

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

end tests.NebulaProductionProductPiCcsTypedBridgeFor
