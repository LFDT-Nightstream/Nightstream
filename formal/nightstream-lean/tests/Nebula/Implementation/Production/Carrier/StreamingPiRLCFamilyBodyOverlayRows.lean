import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyBodyOverlayRows

/-! Regression surface for the split production PiRLC family rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcFamilyBodyOverlayRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows

#check sourceBodyRows_length
#check replayRowsFor_length
#check bodyRowsForParity_length
#check overlayRows_length
#check sourceColumnsExact_of_bodyRows
#check bodyRows_sound_of_output_exact
#check rows_sound

example : (bodyRowsForParity .even).length = 310646 := by
  simpa using bodyRowsForParity_length .even

example : (bodyRowsForParity .odd).length = 311846 := by
  simpa using bodyRowsForParity_length .odd

end tests.NebulaProductionStreamingPiRlcFamilyBodyOverlayRows
