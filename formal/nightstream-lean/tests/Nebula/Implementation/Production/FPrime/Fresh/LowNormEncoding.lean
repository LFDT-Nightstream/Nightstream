import Nightstream.Implementation.Nebula.Production.FPrime.Fresh.LowNormEncoding

/-! Regression surface for the exact production fresh low-norm encoding. -/

set_option autoImplicit false

namespace tests.NebulaProductionFreshLowNormEncoding

open Nightstream.Implementation.Nebula.ProductionFreshLowNormEncoding

#check publicWidth_le_logicalWidth
#check encodeLogical_public
#check encodeLogical_private
#check encodeLogical_private_word_exact
#check decode_privateTritWord
#check encodeCarrier_norm
#check projectPublicInput_encodeCarrier

end tests.NebulaProductionFreshLowNormEncoding
