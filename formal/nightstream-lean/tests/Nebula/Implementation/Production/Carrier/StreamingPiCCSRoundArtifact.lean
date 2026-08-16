import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCSRoundArtifact

/-! Regression surface for the generated production PiCCS round relation. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiCcsRoundArtifact

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundArtifact.Generated

#check rawArtifact_valid
#check decode_rawArtifact
#check layout_currentStart
#check layout_coefficientStart
#check layout_challengeStart
#check layout_nextStart
#check layout_auxiliaryStart
#check sourceRows_length
#check columns_eq
#check sourceRows_below
#check directProgram_length
#check profile.rowDomain
#check generated_selective_ccs_implies_roundPhaseRelation

end tests.NebulaProductionStreamingPiCcsRoundArtifact
