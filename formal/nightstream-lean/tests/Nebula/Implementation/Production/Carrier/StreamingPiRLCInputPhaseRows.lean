import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputPhaseRows

/-! Regression surface for the production PiRLC family commitment rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcInputPhaseRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows

#check Layout
#check exact_chunk_geometry
#check sourceRows_length
#check coordinateBlock_inputValue_exact
#check compact_output_exact_of_rows
#check rows_length
#check Exact.output_at
#check rows_sound

end tests.NebulaProductionStreamingPiRlcInputPhaseRows
