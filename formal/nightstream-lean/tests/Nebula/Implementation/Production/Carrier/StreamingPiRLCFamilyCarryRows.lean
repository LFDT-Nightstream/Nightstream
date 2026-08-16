import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyCarryRows

/-! Regression surface for the production PiRLC challenge and cursor rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcFamilyCarryRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows

#check Layout
#check decodeRows_length
#check challengeRows_length
#check rows_length
#check StateColumnsPlaced
#check decoded_before_exact
#check challenges_exact
#check cursor_exact
#check Exact
#check rows_sound

end tests.NebulaProductionStreamingPiRlcFamilyCarryRows
