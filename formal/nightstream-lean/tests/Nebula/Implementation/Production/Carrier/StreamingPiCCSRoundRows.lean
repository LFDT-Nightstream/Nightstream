import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCSRoundRows

/-! Regression surface for the exact production PiCCS round rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiCcsRoundRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundRows

#check coefficients_length
#check rows_length
#check rows_imply_concrete_round
#check ControlPlacement
#check rows_imply_roundPhaseRelation

end tests.NebulaProductionStreamingPiCcsRoundRows
