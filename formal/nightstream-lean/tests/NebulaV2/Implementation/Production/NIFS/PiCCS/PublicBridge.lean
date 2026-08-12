import Nightstream.Implementation.NebulaV2.Production.NIFS.PiCCS.PublicBridge

/-! Regression surface for the production-profile PiCCS public-field bridge. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionProductPiCcsPublicBridge

open Nightstream.Implementation.NebulaV2.ProductionProductPiCcsPublicBridge

#check PublicWires
#check Placement
#check installPublicWires_fields
#check rows_imply_successor_public_state
#check rows_imply_value_public_state
#check no_cross_candidate_dual_placement

end tests.NebulaV2ProductionProductPiCcsPublicBridge
