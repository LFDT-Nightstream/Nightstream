import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyCompleteRows

/-! Regression surface for the complete production PiRLC family rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcFamilyCompleteRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows

#check layout
#check exact_layout
#check parityFor
#check replayRows_length
#check rows_length
#check inputsPlaced
#check replayValuesPlaced
#check rows_sound

end tests.NebulaProductionStreamingPiRlcFamilyCompleteRows
