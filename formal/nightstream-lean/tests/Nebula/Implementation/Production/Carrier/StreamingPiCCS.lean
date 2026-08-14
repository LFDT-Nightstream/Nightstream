import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCS

/-! Regression surface for bounded-round PiCCS semantics. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiCcs

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcs

#check checkRoundsFrom_exact
#check cubePointOrZero_coordinates_of_length
#check derive_transcript_exact
#check check_eq_protocolVerifier_check
#check check_implies_tableTruth_or_badEvent

end tests.NebulaProductionStreamingPiCcs
