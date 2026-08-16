import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCArtifact

/-! Regression surface for the generated production PiRLC phase relation. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcArtifact

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated

#check rawArtifact_valid
#check decode_rawArtifact
#check layout_base
#check sourceRows_length
#check sourceRows_below
#check directProgram_length
#check profile.rowDomain
#check generated_selective_ccs_implies_concrete_phase

end tests.NebulaProductionStreamingPiRlcArtifact
