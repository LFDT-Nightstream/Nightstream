import Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding

/-! Regression surface for the exact production fresh low-norm encoding. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionFreshLowNormEncoding

open Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding

#check publicWidth_le_logicalWidth
#check encodeLogical_public
#check encodeLogical_private
#check encodeLogical_private_word_exact
#check decode_privateTritWord
#check encodeCarrier_norm
#check projectPublicInput_encodeCarrier

end tests.NebulaV2ProductionFreshLowNormEncoding
